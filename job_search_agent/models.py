"""
Data models for the Job Search AI Agent.
All structured data flows through these Pydantic schemas.
"""

import hashlib
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class Job(BaseModel):
    """Represents a single job listing found by the Searcher."""

    title: str
    company: str
    platform: str  # "linkedin" | "indeed" | "upwork" | "appen" | etc.
    url: str
    description: str = ""
    location: Optional[str] = None
    is_remote: bool = False
    posted_date: Optional[str] = None
    salary_range: Optional[str] = None
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    raw_query: str = ""  # which search query found this job

    @computed_field
    @property
    def id(self) -> str:
        """Deterministic hash ID based on title + company + platform for dedup."""
        key = f"{self.title.lower().strip()}|{self.company.lower().strip()}|{self.platform.lower().strip()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


class QueryResult(BaseModel):
    """Result summary for a single query executed across one or more platforms."""

    query: str
    platform: str  # which platform(s) were searched
    jobs_found: int = 0
    avg_score: float = 0.0
    top_jobs: list[Job] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class SearchSession(BaseModel):
    """Full record of one agent iteration — queries, results, reflection."""

    session_id: str
    iteration: int
    queries_used: list[str] = Field(default_factory=list)
    results: list[QueryResult] = Field(default_factory=list)
    reflection: str = ""  # LLM reflection text
    next_queries: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def total_jobs_found(self) -> int:
        return sum(qr.jobs_found for qr in self.results)

    @property
    def avg_score(self) -> float:
        all_scores = [j.score for qr in self.results for j in qr.top_jobs]
        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    @property
    def all_jobs(self) -> list[Job]:
        return [j for qr in self.results for j in qr.top_jobs]
