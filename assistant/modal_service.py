from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import modal


APP_NAME = "clinical-qa-assistant"
EVALUATION_VOLUME_NAME = (
    "clinical-qa-evaluation-results"
)
RETRIEVAL_VOLUME_NAME = (
    "clinical-qa-ingestion-data"
)
MODEL_CACHE_VOLUME_NAME = (
    "clinical-qa-retrieval-model-cache"
)

DEFAULT_MODEL_ID = "openai/gpt-5-mini"
MAX_CONCURRENT_INPUTS = 4
TARGET_CONCURRENT_INPUTS = 2
MEMORY_HISTORY_MESSAGES = 16

app = modal.App(APP_NAME)

evaluation_volume = modal.Volume.from_name(
    EVALUATION_VOLUME_NAME,
    create_if_missing=True,
)

retrieval_volume = modal.Volume.from_name(
    RETRIEVAL_VOLUME_NAME,
    create_if_missing=False,
)

model_cache_volume = modal.Volume.from_name(
    MODEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
)

model_secrets = [
    modal.Secret.from_name(
        "openai-secret"
    ),
    modal.Secret.from_name(
        "anthropic-secret"
    ),
    modal.Secret.from_name(
        "Openrouter-secret"
    ),
    modal.Secret.from_name(
        "postgres-secret"
    ),
]

storage_image = modal.Image.debian_slim(
    python_version="3.12"
)

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
        "psycopg[binary,pool]>=3.2,<4.0",
    )
    .add_local_dir(
        str(Path(__file__).parent),
        remote_path="/root/assistant",
    )
)


def _validate_identifier(
    value: str,
    field_name: str,
) -> str:
    """Validate a user or session identifier.

    Args:
        value (str): Identifier supplied by the caller.
        field_name (str): Field name for error reporting.

    Returns:
        str: Stripped identifier.

    Raises:
        ValueError: If the identifier is empty or too long.
    """
    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    if len(normalized) > 200:
        raise ValueError(
            f"{field_name} cannot exceed 200 characters."
        )

    return normalized


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
@modal.concurrent(
    max_inputs=MAX_CONCURRENT_INPUTS,
    target_inputs=TARGET_CONCURRENT_INPUTS,
)
class ClinicalQAAgentBackend:
    """Serve grounded answers with persistent PostgreSQL memory."""

    @modal.enter()
    def load_backend(self) -> None:
        """Load retrieval, model, memory and build metadata."""
        from assistant.build_info import (
            ASSISTANT_BUILD_ID,
        )
        from assistant.clinical_agent import (
            ClinicalAgentService,
        )
        from assistant.postgres_memory import (
            PostgresMemory,
        )
        from assistant.retrieval_controller import (
            RetrievalController,
        )

        database_url = os.getenv(
            "DATABASE_URL",
            "",
        )

        if not database_url:
            raise RuntimeError(
                "postgres-secret must contain DATABASE_URL."
            )

        model_id = os.getenv(
            "CLINICAL_QA_MODEL",
            DEFAULT_MODEL_ID,
        )

        retrieval_root = Path(
            "/data/retrieval/current"
        )

        controller = RetrievalController(
            database_path=(
                retrieval_root
                / "retrieval.sqlite"
            ),
            index_path=(
                retrieval_root
                / "chunks.faiss"
            ),
            mapping_path=(
                retrieval_root
                / "vector_mapping.json"
            ),
            embedding_model=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
        )

        self.memory = PostgresMemory(
            database_url=database_url,
            minimum_connections=1,
            maximum_connections=(
                MAX_CONCURRENT_INPUTS
            ),
        )

        manifest_path = (
            retrieval_root
            / "build_manifest.json"
        )

        if manifest_path.is_file():
            self.retrieval_manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        else:
            self.retrieval_manifest = {}

        self.assistant_build_id = (
            ASSISTANT_BUILD_ID
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
        user_id: str,
        include_diagnostics: bool = False,
    ) -> dict[str, Any]:
        """Answer one question using persistent session memory.

        Calls for different users or sessions may run concurrently.
        Calls for the same user and session are serialized through a
        PostgreSQL advisory lock.

        Args:
            question (str): User's health question.
            session_id (str): Conversation session identifier.
            user_id (str): User who owns the session.
            include_diagnostics (bool): Include retrieval diagnostics.

        Returns:
            dict[str, Any]: Structured validated answer.

        Raises:
            ValueError: If an input is empty or too long.
        """
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError(
                "Question cannot be empty."
            )

        if len(normalized_question) > 5000:
            raise ValueError(
                "Question cannot exceed 5000 characters."
            )

        normalized_user_id = _validate_identifier(
            value=user_id,
            field_name="user_id",
        )

        normalized_session_id = _validate_identifier(
            value=session_id,
            field_name="session_id",
        )

        request_id = uuid4()

        with self.memory.open_conversation(
            user_id=normalized_user_id,
            session_id=normalized_session_id,
        ) as conversation:
            history = conversation.get_history(
                limit=MEMORY_HISTORY_MESSAGES
            )

            result = self.agent_service.answer(
                question=normalized_question,
                session_id=normalized_session_id,
                conversation_history=history,
                include_diagnostics=(
                    include_diagnostics
                ),
            )

            response = result.model_dump()
            response["request_id"] = str(
                request_id
            )
            response["user_id"] = (
                normalized_user_id
            )
            response["assistant_build_id"] = (
                self.assistant_build_id
            )
            response["retrieval_build"] = {
                "created_at": (
                    self.retrieval_manifest.get(
                        "created_at"
                    )
                ),
                "embedding_model": (
                    self.retrieval_manifest.get(
                        "embedding_model"
                    )
                ),
                "maximum_chunk_tokens": (
                    self.retrieval_manifest.get(
                        "maximum_chunk_tokens"
                    )
                ),
                "counts": (
                    self.retrieval_manifest.get(
                        "counts",
                        {},
                    )
                ),
            }

            evidence_ids = [
                citation.evidence_id
                for citation in result.citations
            ]

            conversation.save_turn(
                request_id=request_id,
                user_message=(
                    normalized_question
                ),
                assistant_message=result.answer,
                evidence_ids=evidence_ids,
                abstained=result.abstained,
                assistant_response=response,
            )

        return response

    @modal.method()
    def get_session_history(
        self,
        user_id: str,
        session_id: str,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Return stored messages for one user session.

        Args:
            user_id (str): User who owns the session.
            session_id (str): Conversation session identifier.
            limit (int): Maximum messages to return.

        Returns:
            dict[str, Any]: Session identity and stored messages.
        """
        normalized_user_id = _validate_identifier(
            value=user_id,
            field_name="user_id",
        )
        normalized_session_id = _validate_identifier(
            value=session_id,
            field_name="session_id",
        )

        messages = self.memory.get_messages(
            user_id=normalized_user_id,
            session_id=normalized_session_id,
            limit=limit,
        )

        return {
            "user_id": normalized_user_id,
            "session_id": (
                normalized_session_id
            ),
            "message_count": len(messages),
            "messages": messages,
        }

    @modal.method()
    def clear_session(
        self,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Delete one user's conversation session.

        Args:
            user_id (str): User who owns the session.
            session_id (str): Conversation session identifier.

        Returns:
            dict[str, Any]: Whether the session was deleted.
        """
        normalized_user_id = _validate_identifier(
            value=user_id,
            field_name="user_id",
        )
        normalized_session_id = _validate_identifier(
            value=session_id,
            field_name="session_id",
        )

        deleted = self.memory.clear_session(
            user_id=normalized_user_id,
            session_id=normalized_session_id,
        )

        return {
            "user_id": normalized_user_id,
            "session_id": (
                normalized_session_id
            ),
            "deleted": deleted,
        }

    @modal.method()
    def health(self) -> dict[str, Any]:
        """Return backend and PostgreSQL health information.

        Returns:
            dict[str, Any]: Backend health status.
        """
        return {
            "status": "healthy",
            "postgres": (
                self.memory.health_check()
            ),
            "assistant_build_id": (
                self.assistant_build_id
            ),
            "concurrent_inputs": (
                MAX_CONCURRENT_INPUTS
            ),
        }


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
        run_id (str): Unique evaluation run identifier.
        backend_responses_jsonl (str): Complete backend responses.
        agent_answers_markdown (str): Human-readable agent answers.

    Returns:
        dict[str, Any]: Saved paths and file sizes.

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
        Path("/results")
        / "runs"
        / run_id
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    backend_path = (
        run_directory
        / "backend_responses.jsonl"
    )
    answers_path = (
        run_directory
        / "agent_answers.md"
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
        "volume_name": (
            EVALUATION_VOLUME_NAME
        ),
        "backend_responses_path": (
            f"runs/{run_id}/"
            "backend_responses.jsonl"
        ),
        "agent_answers_path": (
            f"runs/{run_id}/"
            "agent_answers.md"
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
@modal.concurrent(
    max_inputs=20,
    target_inputs=10,
)
@modal.fastapi_endpoint(
    method="POST",
    docs=True,
)
def chat_endpoint(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Expose the clinical QA backend through HTTP POST.

    Args:
        payload (dict[str, Any]): Request containing question,
            user ID, session ID and diagnostics preference.

    Returns:
        dict[str, Any]: Structured assistant response.

    Raises:
        ValueError: If required fields are missing.
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
            "",
        )
    )
    user_id = str(
        payload.get(
            "user_id",
            "",
        )
    )
    include_diagnostics = bool(
        payload.get(
            "include_diagnostics",
            False,
        )
    )

    if not user_id.strip():
        raise ValueError(
            "user_id is required."
        )

    if not session_id.strip():
        raise ValueError(
            "session_id is required."
        )

    return backend.answer.remote(
        question=question,
        session_id=session_id,
        user_id=user_id,
        include_diagnostics=(
            include_diagnostics
        ),
    )