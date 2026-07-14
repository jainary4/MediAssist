from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
from uuid import UUID

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool


@dataclass
class PostgresConversation:
    """Represent one locked user conversation.

    Attributes:
        connection (Connection[Any]): Active PostgreSQL connection.
        user_id (str): User who owns the conversation.
        session_id (str): Conversation session identifier.
    """

    connection: Connection[Any]
    user_id: str
    session_id: str

    def get_history(
        self,
        limit: int = 16,
    ) -> list[dict[str, str]]:
        """Load recent messages for the agent prompt.

        Args:
            limit (int): Maximum number of recent messages.

        Returns:
            list[dict[str, str]]: Messages ordered oldest to newest.
        """
        safe_limit = max(
            1,
            min(limit, 100),
        )

        rows = self.connection.execute(
            """
            SELECT
                recent.role,
                recent.content
            FROM (
                SELECT
                    message_id,
                    role,
                    content
                FROM chat_messages
                WHERE user_id = %s
                  AND session_id = %s
                ORDER BY message_id DESC
                LIMIT %s
            ) AS recent
            ORDER BY recent.message_id ASC
            """,
            (
                self.user_id,
                self.session_id,
                safe_limit,
            ),
        ).fetchall()

        return [
            {
                "role": str(row["role"]),
                "content": str(row["content"]),
            }
            for row in rows
        ]

    def save_turn(
        self,
        request_id: UUID,
        user_message: str,
        assistant_message: str,
        evidence_ids: list[str],
        abstained: bool,
        assistant_response: dict[str, Any],
    ) -> None:
        """Save one user and assistant turn atomically.

        Args:
            request_id (UUID): ID shared by both messages in the turn.
            user_message (str): Original user question.
            assistant_message (str): Final validated assistant answer.
            evidence_ids (list[str]): Evidence IDs cited by the answer.
            abstained (bool): Whether the assistant abstained.
            assistant_response (dict[str, Any]): Complete backend response.
        """
        with self.connection.transaction():
            self.connection.execute(
                """
                INSERT INTO chat_messages (
                    request_id,
                    user_id,
                    session_id,
                    role,
                    content,
                    evidence_ids,
                    abstained,
                    response_json
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    'user',
                    %s,
                    %s,
                    NULL,
                    NULL
                )
                """,
                (
                    request_id,
                    self.user_id,
                    self.session_id,
                    user_message,
                    Jsonb([]),
                ),
            )

            self.connection.execute(
                """
                INSERT INTO chat_messages (
                    request_id,
                    user_id,
                    session_id,
                    role,
                    content,
                    evidence_ids,
                    abstained,
                    response_json
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    'assistant',
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    request_id,
                    self.user_id,
                    self.session_id,
                    assistant_message,
                    Jsonb(evidence_ids),
                    abstained,
                    Jsonb(assistant_response),
                ),
            )

            self.connection.execute(
                """
                UPDATE chat_sessions
                SET updated_at = NOW()
                WHERE user_id = %s
                  AND session_id = %s
                """,
                (
                    self.user_id,
                    self.session_id,
                ),
            )

    def delete(self) -> bool:
        """Delete this conversation and all of its messages.

        Returns:
            bool: True when a session was deleted.
        """
        result = self.connection.execute(
            """
            DELETE FROM chat_sessions
            WHERE user_id = %s
              AND session_id = %s
            """,
            (
                self.user_id,
                self.session_id,
            ),
        )

        return bool(result.rowcount)


class PostgresMemory:
    """Manage persistent chat memory using a PostgreSQL pool."""

    def __init__(
        self,
        database_url: str,
        minimum_connections: int = 1,
        maximum_connections: int = 4,
    ) -> None:
        """Create the connection pool and database schema.

        Args:
            database_url (str): PostgreSQL connection URL.
            minimum_connections (int): Warm connections per container.
            maximum_connections (int): Maximum connections per container.

        Raises:
            ValueError: If the database URL is empty.
        """
        if not database_url.strip():
            raise ValueError(
                "DATABASE_URL cannot be empty."
            )

        if maximum_connections < minimum_connections:
            raise ValueError(
                "maximum_connections must be greater than or "
                "equal to minimum_connections."
            )

        self.pool = ConnectionPool(
            conninfo=database_url,
            min_size=minimum_connections,
            max_size=maximum_connections,
            open=True,
            timeout=30.0,
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
            },
            name="clinical-qa-memory",
        )

        self.pool.wait(
            timeout=30.0,
        )

        self._create_schema()

    def _create_schema(self) -> None:
        """Create the session and message tables when missing."""
        with self.pool.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                        DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL
                        DEFAULT NOW(),
                    PRIMARY KEY (
                        user_id,
                        session_id
                    )
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id BIGSERIAL PRIMARY KEY,
                    request_id UUID NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL
                        CHECK (
                            role IN (
                                'user',
                                'assistant'
                            )
                        ),
                    content TEXT NOT NULL,
                    evidence_ids JSONB NOT NULL
                        DEFAULT '[]'::jsonb,
                    abstained BOOLEAN,
                    response_json JSONB,
                    created_at TIMESTAMPTZ NOT NULL
                        DEFAULT NOW(),
                    FOREIGN KEY (
                        user_id,
                        session_id
                    )
                    REFERENCES chat_sessions (
                        user_id,
                        session_id
                    )
                    ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_chat_messages_session
                ON chat_messages (
                    user_id,
                    session_id,
                    message_id DESC
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_chat_messages_request
                ON chat_messages (
                    request_id
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_chat_sessions_updated
                ON chat_sessions (
                    user_id,
                    updated_at DESC
                )
                """
            )

    @contextmanager
    def open_conversation(
        self,
        user_id: str,
        session_id: str,
    ) -> Iterator[PostgresConversation]:
        """Open and lock one user conversation.

        The advisory lock prevents two concurrent requests for the same
        session from reading the same stale history. Different users or
        sessions can still run concurrently.

        Args:
            user_id (str): User who owns the conversation.
            session_id (str): Conversation session identifier.

        Yields:
            PostgresConversation: Locked session helper.
        """
        lock_name = (
            f"clinical-qa:{user_id}:{session_id}"
        )

        with self.pool.connection() as connection:
            connection.execute(
                """
                SELECT pg_advisory_lock(
                    hashtextextended(%s, 0)
                )
                """,
                (lock_name,),
            )

            try:
                connection.execute(
                    """
                    INSERT INTO chat_sessions (
                        user_id,
                        session_id
                    )
                    VALUES (
                        %s,
                        %s
                    )
                    ON CONFLICT (
                        user_id,
                        session_id
                    )
                    DO UPDATE SET
                        updated_at = NOW()
                    """,
                    (
                        user_id,
                        session_id,
                    ),
                )

                yield PostgresConversation(
                    connection=connection,
                    user_id=user_id,
                    session_id=session_id,
                )
            finally:
                connection.execute(
                    """
                    SELECT pg_advisory_unlock(
                        hashtextextended(%s, 0)
                    )
                    """,
                    (lock_name,),
                )

    def get_messages(
        self,
        user_id: str,
        session_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read stored messages for a user session.

        Args:
            user_id (str): User who owns the conversation.
            session_id (str): Conversation session identifier.
            limit (int): Maximum messages to return.

        Returns:
            list[dict[str, Any]]: Stored messages in chronological order.
        """
        safe_limit = max(
            1,
            min(limit, 500),
        )

        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    message_id,
                    request_id,
                    role,
                    content,
                    evidence_ids,
                    abstained,
                    response_json,
                    created_at
                FROM chat_messages
                WHERE user_id = %s
                  AND session_id = %s
                ORDER BY message_id ASC
                LIMIT %s
                """,
                (
                    user_id,
                    session_id,
                    safe_limit,
                ),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def clear_session(
        self,
        user_id: str,
        session_id: str,
    ) -> bool:
        """Delete one session and all associated messages.

        Args:
            user_id (str): User who owns the conversation.
            session_id (str): Conversation session identifier.

        Returns:
            bool: True when the session existed and was deleted.
        """
        with self.open_conversation(
            user_id=user_id,
            session_id=session_id,
        ) as conversation:
            return conversation.delete()

    def health_check(self) -> bool:
        """Check whether PostgreSQL is reachable.

        Returns:
            bool: True when PostgreSQL responds successfully.
        """
        with self.pool.connection() as connection:
            row = connection.execute(
                "SELECT 1 AS healthy"
            ).fetchone()

        return bool(
            row
            and row["healthy"] == 1
        )