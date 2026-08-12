from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from app.schemas import FeedbackRequest


def save_feedback(feedback: FeedbackRequest, database_path: str) -> None:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                corrected_text TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO feedback (analysis_id, accepted, corrected_text, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                feedback.analysis_id,
                int(feedback.accepted),
                feedback.corrected_text,
                datetime.now(UTC).isoformat(),
            ),
        )

