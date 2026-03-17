"""
Memory module — persists sessions, jobs, and query performance in SQLite.
Provides retrieval helpers used by the Planner and Reflector.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from .config import DB_PATH
from .models import Job, SearchSession


class Memory:
    """SQLite-backed memory for the agent's search history."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = str(db_path or DB_PATH)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    # ── Schema ──────────────────────────────────────────

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id           TEXT PRIMARY KEY,
                iteration    INTEGER NOT NULL,
                timestamp    TEXT NOT NULL,
                reflection   TEXT DEFAULT '',
                queries_json TEXT DEFAULT '[]',
                next_queries TEXT DEFAULT '[]',
                avg_score    REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id                  TEXT PRIMARY KEY,
                title               TEXT NOT NULL,
                company             TEXT DEFAULT '',
                platform            TEXT DEFAULT '',
                url                 TEXT DEFAULT '',
                description         TEXT DEFAULT '',
                location            TEXT DEFAULT '',
                is_remote           INTEGER DEFAULT 0,
                posted_date         TEXT DEFAULT '',
                salary_range        TEXT DEFAULT '',
                score               REAL DEFAULT 0.0,
                score_breakdown     TEXT DEFAULT '{}',
                raw_query           TEXT DEFAULT '',
                session_id          TEXT DEFAULT '',
                created_at          TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS queries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text  TEXT NOT NULL,
                platform    TEXT DEFAULT 'all',
                avg_score   REAL DEFAULT 0.0,
                jobs_found  INTEGER DEFAULT 0,
                session_id  TEXT DEFAULT ''
            );
            """
        )
        self.conn.commit()

    # ── Write ───────────────────────────────────────────

    def save_session(self, session: SearchSession):
        """Insert or replace a full session record."""
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO sessions
                (id, iteration, timestamp, reflection, queries_json, next_queries, avg_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                session.iteration,
                session.timestamp.isoformat(),
                session.reflection,
                json.dumps(session.queries_used),
                json.dumps(session.next_queries),
                session.avg_score,
            ),
        )
        # Also persist per-query stats
        for qr in session.results:
            cur.execute(
                """
                INSERT INTO queries (query_text, platform, avg_score, jobs_found, session_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (qr.query, qr.platform, qr.avg_score, qr.jobs_found, session.session_id),
            )
        self.conn.commit()

    def save_jobs(self, jobs: list[Job], session_id: str = ""):
        """Upsert jobs — skip duplicates by hash id."""
        cur = self.conn.cursor()
        for job in jobs:
            cur.execute(
                """
                INSERT OR IGNORE INTO jobs
                    (id, title, company, platform, url, description, location,
                     is_remote, posted_date, salary_range, score, score_breakdown,
                     raw_query, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.title,
                    job.company,
                    job.platform,
                    job.url,
                    job.description,
                    job.location or "",
                    int(job.is_remote),
                    job.posted_date or "",
                    job.salary_range or "",
                    job.score,
                    json.dumps(job.score_breakdown),
                    job.raw_query,
                    session_id,
                ),
            )
        self.conn.commit()

    # ── Read ────────────────────────────────────────────

    def get_seen_job_ids(self) -> set[str]:
        """Return all job IDs already persisted — used for deduplication."""
        cur = self.conn.cursor()
        rows = cur.execute("SELECT id FROM jobs").fetchall()
        return {row["id"] for row in rows}

    def get_top_jobs(self, n: int = 20) -> list[dict]:
        """Return top N jobs by score across all sessions."""
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT * FROM jobs ORDER BY score DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(row) for row in rows]

    def get_last_reflection(self) -> str:
        """Return the most recent reflection text."""
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT reflection FROM sessions ORDER BY iteration DESC LIMIT 1"
        ).fetchone()
        return row["reflection"] if row else ""

    def get_query_performance_summary(self) -> str:
        """
        Build a human-readable summary of past query performance.
        Used by Planner and Reflector LLM prompts.
        """
        cur = self.conn.cursor()
        rows = cur.execute(
            """
            SELECT query_text, AVG(avg_score) as mean_score, SUM(jobs_found) as total_found
            FROM queries
            GROUP BY query_text
            ORDER BY mean_score DESC
            """
        ).fetchall()

        if not rows:
            return "No previous queries — this is the first iteration."

        lines = ["Past Query Performance (best → worst):"]
        for row in rows:
            lines.append(
                f"  • \"{row['query_text']}\" → avg_score={row['mean_score']:.3f}, "
                f"jobs_found={row['total_found']}"
            )
        return "\n".join(lines)

    def get_iteration_count(self) -> int:
        """Return the number of completed iterations."""
        cur = self.conn.cursor()
        row = cur.execute("SELECT MAX(iteration) as mx FROM sessions").fetchone()
        return (row["mx"] + 1) if row and row["mx"] is not None else 0

    def get_best_score(self) -> float:
        """Return the highest job score across all sessions."""
        cur = self.conn.cursor()
        row = cur.execute("SELECT MAX(score) as best FROM jobs").fetchone()
        return row["best"] if row and row["best"] is not None else 0.0

    def get_score_trend(self, last_n: int = 5) -> list[float]:
        """Return avg scores for the last N sessions, oldest first."""
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT avg_score FROM sessions ORDER BY iteration DESC LIMIT ?",
            (last_n,),
        ).fetchall()
        return [row["avg_score"] for row in reversed(rows)]

    # ── Lifecycle ───────────────────────────────────────

    def close(self):
        self.conn.close()

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass
