from __future__ import annotations
import re
import os
from pathlib import Path
from typing import Any

import modal

"""Deploy the clinical QA agent and chat endpoint on Modal."""

APP_NAME = "clinical-qa-assistant"
EVALUATION_VOLUME_NAME = "clinical-qa-evaluation-results"
RETRIEVAL_VOLUME_NAME = "clinical-qa-ingestion-data"
MODEL_CACHE_VOLUME_NAME = (
    "clinical-qa-retrieval-model-cache"
)
SESSION_DICT_NAME = "clinical-qa-chat-sessions"

evaluation_volume = modal.Volume.from_name(
    EVALUATION_VOLUME_NAME,
    create_if_missing=True,
)

DEFAULT_MODEL_ID = "openai/gpt-5-mini"

storage_image = modal.Image.debian_slim(
    python_version="3.12"
)

app = modal.App(APP_NAME)


retrieval_volume = modal.Volume.from_name(
    RETRIEVAL_VOLUME_NAME,
    create_if_missing=False,
)

model_cache_volume = modal.Volume.from_name(
    MODEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
)

session_store = modal.Dict.from_name(
    SESSION_DICT_NAME,
    create_if_missing=True,
)


model_secrets = [
    modal.Secret.from_name("openai-secret"),
    modal.Secret.from_name("anthropic-secret"),
    modal.Secret.from_name("Openrouter-secret"),
]


image = (
    modal.Image.debian_slim(
        python_version="3.12"
    )
    .apt_install(
        "libgomp1",
    )
    .pip_install(
        "agno>=2.1,<3.0",
        "openai>=2.0,<3.0",
        "pydantic>=2.9,<3.0",
        "sentence-transformers>=5.0,<6.0",
        "faiss-cpu==1.14.3",
        "numpy>=1.26,<3.0",
        "fastapi[standard]>=0.115,<1.0",
    )
    .add_local_dir(
        str(Path(__file__).parent),
        remote_path="/root/assistant",
    )
)


@app.cls(
    image=image,
    secrets=model_secrets,
    volumes={
        "/data": retrieval_volume.with_mount_options(
            read_only=True
        ),
        "/root/.cache/huggingface": (
            model_cache_volume
        ),
    },
    cpu=2.0,
    memory=8192,
    timeout=600,
    max_containers=10,
    buffer_containers=1,
    scaledown_window=300,
)
class ClinicalQAAgentBackend:
    """Keep the retrieval model and indexes warm on Modal."""

    @modal.enter()
    def load_backend(self) -> None:
        """Load the retrieval controller once per Modal container."""
        from assistant.clinical_agent import (
            ClinicalAgentService,
        )
        from assistant.retrieval_controller import (
            RetrievalController,
        )

        model_id = os.getenv(
            "CLINICAL_QA_MODEL",
            DEFAULT_MODEL_ID,
        )

        controller = RetrievalController(
            database_path=Path(
                "/data/retrieval/current/retrieval.sqlite"
            ),
            index_path=Path(
                "/data/retrieval/current/chunks.faiss"
            ),
            mapping_path=Path(
                "/data/retrieval/current/vector_mapping.json"
            ),
            embedding_model=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
        )

        self.agent_service = ClinicalAgentService(
            controller=controller,
            model_id=model_id,
        )

    @modal.method()
    def answer(
        self,
        question: str,
        session_id: str,
        user_id: str = "anonymous",
        include_diagnostics: bool = False,
    ) -> dict[str, Any]:
        """Answer one question and persist its conversation turn.

        Args:
            question (str): User's health question.
            session_id (str): Conversation session identifier.
            user_id (str): User identifier for future access control.
            include_diagnostics (bool): Return retrieval diagnostics.

        Returns:
            dict[str, Any]: Structured validated answer.

        Raises:
            ValueError: If the question is empty.
        """
        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        history = session_store.get(
            session_id,
            [],
        )

        result = self.agent_service.answer(
            question=question.strip(),
            session_id=session_id,
            conversation_history=history,
            include_diagnostics=include_diagnostics,
        )

        history.append({
            "role": "user",
            "content": question.strip(),
        })

        history.append({
            "role": "assistant",
            "content": result.answer,
            "evidence_ids": [
                citation.evidence_id
                for citation in result.citations
            ],
            "abstained": result.abstained,
        })

        session_store[session_id] = history[-16:]

        response = result.model_dump()
        response["user_id"] = user_id

        return response


backend = ClinicalQAAgentBackend()

@app.function(
    image=storage_image,
    volumes={
        "/results": evaluation_volume,
    },
    timeout=120,
)
def save_batch_results(
    run_id: str,
    backend_responses_jsonl: str,
    agent_answers_markdown: str,
) -> dict[str, Any]:
    """Save one evaluation run to a persistent Modal Volume.

    Args:
        run_id (str): Unique identifier for the evaluation run.
        backend_responses_jsonl (str): Full backend responses in JSONL.
        agent_answers_markdown (str): Human-readable agent answers.

    Returns:
        dict[str, Any]: Volume name, saved paths, and file sizes.

    Raises:
        ValueError: If the run ID or file contents are invalid.
    """
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}",
        run_id,
    ):
        raise ValueError(
            "run_id may only contain letters, numbers, dots, "
            "underscores, and hyphens."
        )

    if not backend_responses_jsonl.strip():
        raise ValueError(
            "backend_responses_jsonl cannot be empty."
        )

    if not agent_answers_markdown.strip():
        raise ValueError(
            "agent_answers_markdown cannot be empty."
        )

    run_directory = (
        Path("/results") / "runs" / run_id
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    backend_path = (
        run_directory / "backend_responses.jsonl"
    )
    answers_path = (
        run_directory / "agent_answers.md"
    )

    backend_path.write_text(
        backend_responses_jsonl,
        encoding="utf-8",
    )

    answers_path.write_text(
        agent_answers_markdown,
        encoding="utf-8",
    )

    evaluation_volume.commit()

    return {
        "run_id": run_id,
        "volume_name": EVALUATION_VOLUME_NAME,
        "backend_responses_path": (
            f"runs/{run_id}/backend_responses.jsonl"
        ),
        "agent_answers_path": (
            f"runs/{run_id}/agent_answers.md"
        ),
        "backend_responses_bytes": (
            backend_path.stat().st_size
        ),
        "agent_answers_bytes": (
            answers_path.stat().st_size
        ),
    }


@app.function(
    image=image,
    timeout=650,
)
@modal.fastapi_endpoint(
    method="POST",
    docs=True,
)
def chat_endpoint(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Expose the clinical QA agent through an HTTP endpoint.

    Args:
        payload (dict[str, Any]): Request containing the question,
            session ID, user ID, and diagnostic preference.

    Returns:
        dict[str, Any]: Structured clinical QA response.
    """
    question = str(
        payload.get(
            "question",
            "",
        )
    )

    session_id = str(
        payload.get(
            "session_id",
            "default-session",
        )
    )

    user_id = str(
        payload.get(
            "user_id",
            "anonymous",
        )
    )

    include_diagnostics = bool(
        payload.get(
            "include_diagnostics",
            False,
        )
    )

    return backend.answer.remote(
        question=question,
        session_id=session_id,
        user_id=user_id,
        include_diagnostics=include_diagnostics,
    )