"""Local SQLite storage for practice history — no audio, just scores.

Schema::

    sessions  — one row per practice session (timestamp, target text, lang, …)
    words     — per-word scores linked to a session
    phonemes  — per-phoneme feedback linked to a word

Every write is an atomic transaction.  No raw audio is ever stored.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from phoneta.core.metrics.scoring import PhonemeFeedback, WordScore

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL NOT NULL,
    target_text TEXT NOT NULL,
    lang        TEXT NOT NULL,
    alignment   TEXT NOT NULL DEFAULT '',
    prosody_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS words (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    word        TEXT NOT NULL,
    accuracy    REAL NOT NULL,
    color       TEXT NOT NULL,
    UNIQUE(session_id, idx)
);

CREATE TABLE IF NOT EXISTS phonemes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id     INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    ref         TEXT,
    user_phoneme TEXT,
    kind        TEXT NOT NULL,
    confidence  REAL NOT NULL,
    flagged     INTEGER NOT NULL,
    UNIQUE(word_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_words_session ON words(session_id);
CREATE INDEX IF NOT EXISTS idx_phonemes_word ON phonemes(word_id);
"""


@dataclass(frozen=True)
class SessionRow:
    """A row from the sessions table with its words expanded."""

    id: int
    created_at: float
    target_text: str
    lang: str
    alignment: str
    words: tuple[WordScore, ...]


class PracticeStore:
    """Persist pronunciation practice results to a local SQLite database."""

    def __init__(self, db_path: str | Path = "phoneta_history.db") -> None:
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def save_session(
        self,
        target_text: str,
        lang: str,
        words: tuple[WordScore, ...],
        alignment_method: str = "",
        prosody_json: str | None = None,
    ) -> int:
        """Persist one practice session and return its row id."""
        with self._tx() as db:
            db.execute(
                "INSERT INTO sessions (created_at, target_text, lang, alignment, prosody_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    time.time(),
                    target_text,
                    lang,
                    alignment_method,
                    prosody_json or "{}",
                ),
            )
            session_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

            for i, ws in enumerate(words):
                db.execute(
                    "INSERT INTO words (session_id, idx, word, accuracy, color) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, i, ws.word, ws.accuracy, ws.color),
                )
                word_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

                for j, fb in enumerate(ws.feedback):
                    db.execute(
                        "INSERT INTO phonemes "
                        "(word_id, idx, ref, user_phoneme, kind, confidence, flagged) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            word_id,
                            j,
                            fb.ref,
                            fb.user,
                            fb.kind,
                            fb.confidence,
                            int(fb.flagged),
                        ),
                    )
        return session_id  # type: ignore[no-any-return]

    def list_sessions(
        self, limit: int = 50, offset: int = 0
    ) -> list[SessionRow]:
        """Return recent practice sessions, newest first."""
        with self._tx() as db:
            rows = db.execute(
                "SELECT id, created_at, target_text, lang, alignment, prosody_json "
                "FROM sessions ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

            result: list[SessionRow] = []
            for row in rows:
                sid = row[0]
                word_rows = db.execute(
                    "SELECT id, idx, word, accuracy, color FROM words "
                    "WHERE session_id = ? ORDER BY idx",
                    (sid,),
                ).fetchall()
                words = tuple(
                    self._load_word(db, wrow) for wrow in word_rows
                )
                result.append(
                    SessionRow(
                        id=sid,
                        created_at=row[1],
                        target_text=row[2],
                        lang=row[3],
                        alignment=row[4] or "",
                        words=words,
                    )
                )
            return result

    def get_session(self, session_id: int) -> SessionRow | None:
        """Retrieve one session by id, or None."""
        with self._tx() as db:
            row = db.execute(
                "SELECT id, created_at, target_text, lang, alignment, prosody_json "
                "FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None

            word_rows = db.execute(
                "SELECT id, idx, word, accuracy, color FROM words "
                "WHERE session_id = ? ORDER BY idx",
                (session_id,),
            ).fetchall()
            return SessionRow(
                id=row[0],
                created_at=row[1],
                target_text=row[2],
                lang=row[3],
                alignment=row[4] or "",
                words=tuple(
                    self._load_word(db, wrow) for wrow in word_rows
                ),
            )

    def clear(self) -> int:
        """Delete all sessions; returns the number of rows removed."""
        with self._tx() as db:
            db.execute("DELETE FROM phonemes")
            db.execute("DELETE FROM words")
            db.execute("DELETE FROM sessions")
            return db.total_changes

    def session_count(self) -> int:
        with self._tx() as db:
            row = db.execute("SELECT COUNT(*) FROM sessions").fetchone()
            assert row is not None
            return int(row[0])

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._tx() as db:
            for stmt in SCHEMA.split(";"):
                stmt = stmt.strip()
                if stmt:
                    db.execute(stmt)

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _load_word(
        db: sqlite3.Connection, row: tuple
    ) -> WordScore:
        wid, idx, word, accuracy, color = row
        phon_rows = db.execute(
            "SELECT ref, user_phoneme, kind, confidence, flagged "
            "FROM phonemes WHERE word_id = ? ORDER BY idx",
            (wid,),
        ).fetchall()
        feedback = tuple(
            PhonemeFeedback(
                ref=pr[0],
                user=pr[1],
                kind=pr[2],
                confidence=pr[3],
                flagged=bool(pr[4]),
            )
            for pr in phon_rows
        )
        return WordScore(
            word=word,
            accuracy=accuracy,
            color=color,
            feedback=feedback,
        )