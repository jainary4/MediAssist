from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import requests
import streamlit as st


REQUEST_TIMEOUT_SECONDS = 300


def get_setting(name: str) -> str:
    """Read a setting from environment variables or Streamlit secrets.

    Args:
        name (str): Name of the configuration value.

    Returns:
        str: Configured value, or an empty string when it is missing.
    """
    environment_value = os.getenv(
        name,
        "",
    ).strip()

    if environment_value:
        return environment_value

    try:
        return str(
            st.secrets.get(
                name,
                "",
            )
        ).strip()
    except Exception:
        return ""


def initialize_session_state() -> None:
    """Initialize the user, conversation, and visible message state."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = (
            f"web-user-{uuid4().hex}"
        )

    if "session_id" not in st.session_state:
        st.session_state.session_id = (
            f"web-session-{uuid4().hex}"
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []


def start_new_conversation() -> None:
    """Start a new conversation while preserving the browser user ID."""
    st.session_state.session_id = (
        f"web-session-{uuid4().hex}"
    )
    st.session_state.messages = []


def send_question(
    backend_url: str,
    question: str,
    user_id: str,
    session_id: str,
) -> dict[str, Any]:
    """Send one question to the deployed Modal HTTP endpoint.

    Args:
        backend_url (str): Public URL of the Modal chat endpoint.
        question (str): Question entered by the user.
        user_id (str): Browser-specific user identifier.
        session_id (str): Current conversation identifier.

    Returns:
        dict[str, Any]: Structured answer returned by Modal.

    Raises:
        RuntimeError: If the backend request fails or returns invalid JSON.
    """
    payload = {
        "question": question,
        "user_id": user_id,
        "session_id": session_id,
        "include_diagnostics": False,
    }

    try:
        response = requests.post(
            backend_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout as error:
        raise RuntimeError(
            "The assistant took too long to respond. Please try again."
        ) from error
    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not connect to the assistant: {error}"
        ) from error

    if not response.ok:
        try:
            error_body = response.json()
        except ValueError:
            error_body = response.text[:500]

        raise RuntimeError(
            f"The assistant returned HTTP {response.status_code}: "
            f"{error_body}"
        )

    try:
        result = response.json()
    except ValueError as error:
        raise RuntimeError(
            "The assistant returned an invalid JSON response."
        ) from error

    if not isinstance(result, dict):
        raise RuntimeError(
            "The assistant returned an unexpected response format."
        )

    return result


def format_pages(page_numbers: list[Any]) -> str:
    """Format citation page numbers for display.

    Args:
        page_numbers (list[Any]): Page numbers returned by the backend.

    Returns:
        str: Human-readable page description.
    """
    valid_pages = [
        str(page)
        for page in page_numbers
        if page is not None
    ]

    if not valid_pages:
        return "Page not available"

    if len(valid_pages) == 1:
        return f"Page {valid_pages[0]}"

    return f"Pages {', '.join(valid_pages)}"


def render_citations(
    citations: list[dict[str, Any]],
) -> None:
    """Display citations returned with an assistant answer.

    Args:
        citations (list[dict[str, Any]]): Citation records from Modal.
    """
    if not citations:
        st.caption("No citations were returned.")
        return

    with st.expander(
        f"Sources ({len(citations)})",
        expanded=True,
    ):
        for position, citation in enumerate(
            citations,
            start=1,
        ):
            document_title = str(
                citation.get(
                    "document_title",
                    "Unknown document",
                )
            )

            pages = format_pages(
                citation.get(
                    "page_numbers",
                    [],
                )
            )

            content_type = str(
                citation.get(
                    "content_type",
                    "text",
                )
            ).replace(
                "_",
                " ",
            ).title()

            section = str(
                citation.get(
                    "section",
                    "",
                )
            ).strip()

            excerpt = str(
                citation.get(
                    "excerpt",
                    "",
                )
            ).strip()

            st.markdown(
                f"**{position}. {document_title}**"
            )

            source_details = (
                f"{pages} · {content_type}"
            )

            if section:
                source_details += f" · {section}"

            st.caption(source_details)

            if excerpt:
                st.write(excerpt)

            if position < len(citations):
                st.divider()


def render_assistant_response(
    response: dict[str, Any],
) -> None:
    """Display an assistant response in a user-friendly format.

    Args:
        response (dict[str, Any]): Structured response returned by Modal.
    """
    answer = str(
        response.get(
            "answer",
            "The assistant did not return an answer.",
        )
    )

    abstained = bool(
        response.get(
            "abstained",
            False,
        )
    )

    if abstained:
        st.warning(answer)
    else:
        st.markdown(answer)

    confidence = response.get(
        "confidence"
    )

    confidence_label = str(
        response.get(
            "confidence_label",
            "unknown",
        )
    ).title()

    if isinstance(
        confidence,
        (int, float),
    ):
        st.caption(
            f"Confidence: {float(confidence):.0%} "
            f"({confidence_label})"
        )

        st.progress(
            max(
                0,
                min(
                    int(float(confidence) * 100),
                    100,
                ),
            )
        )

    citations = response.get(
        "citations",
        [],
    )

    if isinstance(citations, list):
        render_citations(citations)

    abstention_reason = response.get(
        "abstention_reason"
    )

    if abstained and abstention_reason:
        with st.expander(
            "Why the assistant did not answer"
        ):
            st.write(abstention_reason)

    limitations = response.get(
        "limitations",
        [],
    )

    if limitations:
        with st.expander("Limitations"):
            for limitation in limitations:
                st.markdown(f"- {limitation}")


def render_saved_messages() -> None:
    """Render the messages stored in the current Streamlit session."""
    for message in st.session_state.messages:
        role = message.get(
            "role",
            "assistant",
        )

        with st.chat_message(role):
            if role == "user":
                st.markdown(
                    str(
                        message.get(
                            "content",
                            "",
                        )
                    )
                )
            else:
                response = message.get(
                    "response",
                    {},
                )

                if isinstance(response, dict):
                    render_assistant_response(
                        response
                    )


def main() -> None:
    """Run the Streamlit clinical QA chat application."""
    st.set_page_config(
        page_title="Clinical Document Assistant",
        page_icon="🩺",
        layout="centered",
    )

    initialize_session_state()

    backend_url = get_setting(
        "MODAL_CHAT_URL"
    )

    st.title("Clinical Document Assistant")

    st.caption(
        "Ask questions about the provided clinical reference documents. "
        "Answers are grounded in the document collection and include "
        "citations and a heuristic confidence score."
    )

    with st.sidebar:
        st.subheader("Conversation")

        st.caption(
            f"Session: "
            f"{st.session_state.session_id[-12:]}"
        )

        if st.button(
            "New conversation",
        ):
            start_new_conversation()
            st.rerun()

        st.markdown("---")

        st.info(
            "This prototype answers only from the supplied document "
            "collection. It is not medical advice."
        )

    if not backend_url:
        st.error(
            "MODAL_CHAT_URL has not been configured. "
            "Add it to Streamlit secrets before using the app."
        )
        st.stop()

    render_saved_messages()

    question = st.chat_input(
        "Ask a question about the clinical documents"
    )

    if not question:
        return

    normalized_question = question.strip()

    if not normalized_question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": normalized_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(normalized_question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching the documents and checking the evidence..."
        ):
            try:
                response = send_question(
                    backend_url=backend_url,
                    question=normalized_question,
                    user_id=st.session_state.user_id,
                    session_id=(
                        st.session_state.session_id
                    ),
                )
            except RuntimeError as error:
                st.error(str(error))
                return

        render_assistant_response(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "response": response,
        }
    )


if __name__ == "__main__":
    main()