"""
Reporter module — produces structured JSON reports after each search session.
"""

import json
import logging
from pathlib import Path

from .config import OUTPUT_DIR
from .models import Job, SearchSession

log = logging.getLogger(__name__)


class Reporter:
    """Generates and saves per-session JSON reports."""

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_report(self, session: SearchSession, top_jobs: list[Job]) -> str:
        """
        Build a structured report dict and write it to a JSON file.
        Returns the path to the written file.
        """
        report = self._build_report(session, top_jobs)
        filename = f"session_{session.iteration}_{session.session_id[:8]}.json"
        path = self.output_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        log.info(f"Report saved: {path}")
        return str(path)

    def _build_report(self, session: SearchSession, top_jobs: list[Job]) -> dict:
        """Assemble the report dictionary."""
        return {
            "session_id": session.session_id,
            "iteration": session.iteration,
            "timestamp": session.timestamp.isoformat(),
            "queries_used": session.queries_used,
            "summary": {
                "total_jobs_found": session.total_jobs_found,
                "avg_score": round(session.avg_score, 3),
                "top_score": round(max((j.score for j in top_jobs), default=0.0), 3),
            },
            "top_jobs": [
                self._job_summary(j) for j in top_jobs[:15]
            ],
            "per_query_stats": [
                {
                    "query": qr.query,
                    "platform": qr.platform,
                    "jobs_found": qr.jobs_found,
                    "avg_score": round(qr.avg_score, 3),
                }
                for qr in session.results
            ],
            "reflection": session.reflection,
            "next_queries_planned": session.next_queries,
        }

    def _job_summary(self, job: Job) -> dict:
        """Create a compact dict for a single job in the report."""
        return {
            "title": job.title,
            "company": job.company,
            "platform": job.platform,
            "url": job.url,
            "score": round(job.score, 3),
            "score_breakdown": job.score_breakdown,
            "is_remote": job.is_remote,
            "salary_range": job.salary_range,
            "location": job.location,
            "query_used": job.raw_query,
        }

    def print_summary(self, session: SearchSession, top_jobs: list[Job]):
        """Print a human-readable summary to the console."""
        print(f"\n{'─' * 60}")
        print(f"  SESSION REPORT — Iteration {session.iteration}")
        print(f"{'─' * 60}")
        print(f"  Total jobs found : {session.total_jobs_found}")
        print(f"  Average score    : {session.avg_score:.3f}")
        print(f"  Queries used     : {len(session.queries_used)}")
        print(f"\n  Top 5 Jobs:")
        for i, j in enumerate(top_jobs[:5], 1):
            remote_tag = " 🌐" if j.is_remote else ""
            print(f"    {i}. [{j.score:.2f}] {j.title} @ {j.company} ({j.platform}){remote_tag}")
            if j.salary_range:
                print(f"       💰 {j.salary_range}")
            print(f"       🔗 {j.url}")
        print(f"\n  Reflection: {session.reflection[:200]}...")
        print(f"{'─' * 60}\n")
