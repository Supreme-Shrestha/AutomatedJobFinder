"""
Evaluator module — scores each Job against a weighted rubric.
Pure logic, no LLM calls. Designed for speed and testability.
"""

import re
from datetime import datetime, timedelta
from typing import Optional

from .config import PLATFORM_TRUST_SCORES, SCORING_WEIGHTS, TARGET_KEYWORDS
from .models import Job


class Evaluator:
    """Scores jobs using a multi-criteria weighted rubric."""

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        keywords: Optional[list[str]] = None,
    ):
        self.weights = dict(weights or SCORING_WEIGHTS)
        self.keywords = [kw.lower() for kw in (keywords or TARGET_KEYWORDS)]
        self._normalize_weights()

    def _normalize_weights(self):
        """Ensure weights sum to exactly 1.0 by normalizing."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def update_weights(self, adjustments: dict[str, float]):
        """
        Apply weight adjustments from the Reflector.
        adjustments: e.g. {"keyword_match": +0.05, "pay_clarity": -0.03}
        Clamps each weight to [0.01, 0.60] and re-normalizes.
        """
        for key, delta in adjustments.items():
            if key in self.weights:
                self.weights[key] = max(0.01, min(0.60, self.weights[key] + delta))
        self._normalize_weights()

    # ── Public API ──────────────────────────────────────

    def score_job(self, job: Job) -> Job:
        """
        Compute a 0–1 relevance score for a single job.
        Populates `job.score` and `job.score_breakdown`.
        Returns the mutated job.
        """
        text = f"{job.title} {job.description}".lower()
        breakdown: dict[str, float] = {}

        breakdown["keyword_match"] = self._keyword_score(text)
        breakdown["experience_level"] = self._experience_score(text)
        breakdown["remote_available"] = self._remote_score(job, text)
        breakdown["pay_clarity"] = self._pay_score(text)
        breakdown["recency"] = self._recency_score(job.posted_date)
        breakdown["platform_trust"] = self._platform_score(job.platform)

        # Weighted sum
        job.score = round(
            sum(self.weights[k] * breakdown[k] for k in self.weights), 4
        )
        job.score_breakdown = {k: round(v, 3) for k, v in breakdown.items()}
        return job

    def score_all(self, jobs: list[Job]) -> list[Job]:
        """Score and sort a list of jobs, highest score first."""
        scored = [self.score_job(j) for j in jobs]
        return sorted(scored, key=lambda j: j.score, reverse=True)

    # ── Scoring Criteria ────────────────────────────────

    def _keyword_score(self, text: str) -> float:
        """Fraction of target keywords present in the text (capped at 1.0)."""
        matches = sum(1 for kw in self.keywords if kw in text)
        return min(matches / 5.0, 1.0)

    def _experience_score(self, text: str) -> float:
        """
        High score if entry-level signals present, penalized if senior signals found.
        """
        entry_signals = [
            "entry level", "entry-level", "no experience", "beginner",
            "junior", "fresher", "0-1 year", "0 - 1 year", "intern",
            "trainee", "associate",
        ]
        senior_signals = [
            "senior", "lead", "principal", "5+ years", "7+ years",
            "10+ years", "expert", "manager", "director", "staff engineer",
        ]

        entry_hits = sum(1 for s in entry_signals if s in text)
        senior_hits = sum(1 for s in senior_signals if s in text)

        raw = entry_hits * 0.35 - senior_hits * 0.5
        return max(0.0, min(raw, 1.0))

    def _remote_score(self, job: Job, text: str) -> float:
        """
        1.0 if remote/worldwide/Nepal.
        0.0 if explicitly restricted to another region (e.g., US only).
        """
        # Bad signals — explicitly restricts away from Nepal/Worldwide
        geo_restrictions = [
            "us only", "usa only", "uk only", "europe only", "eu only",
            "must be located in the us", "must live in the us",
            "must reside in the united states", "must be in uk",
            "states only", "north america only"
        ]
        
        # Good signals — explicitly allows Nepal
        geo_good = [
            "worldwide", "global remote", "anywhere in the world",
            "nepal", "nepali", "hire anywhere"
        ]

        # 1. Immediate penalty if it restricts exactly
        if any(bad in text for bad in geo_restrictions):
            return 0.0

        # 2. Perfect score if it specifically includes our geo preference
        if any(good in text for good in geo_good):
            return 1.0

        # 3. Standard remote check
        if job.is_remote:
            return 0.8  # Good, but not as good as explicitly worldwide
            
        remote_signals = ["remote", "work from home", "wfh", "distributed team"]
        return 0.8 if any(s in text for s in remote_signals) else 0.0

    def _pay_score(self, text: str) -> float:
        """1.0 if salary or rate information is present."""
        pay_patterns = [
            r"\$[\d,]+",
            r"\/hr",
            r"per hour",
            r"hourly",
            r"monthly",
            r"annual",
            r"salary",
            r"compensation",
            r"₹[\d,]+",
            r"€[\d,]+",
        ]
        for pattern in pay_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 1.0
        return 0.0

    def _recency_score(self, posted_date: Optional[str]) -> float:
        """
        Score based on how recently the job was posted.
        1.0 = today, 0.7 = within 3 days, 0.5 = within 7 days,
        0.2 = within 30 days, 0.0 = older or unknown.
        """
        if not posted_date:
            return 0.3  # unknown → slight benefit of the doubt

        try:
            posted = datetime.fromisoformat(posted_date.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            # Try relative date parsing ("2 days ago", "1 week ago")
            return self._parse_relative_date_score(posted_date)

        now = datetime.now(posted.tzinfo) if posted.tzinfo else datetime.now()
        delta = now - posted

        if delta <= timedelta(days=1):
            return 1.0
        elif delta <= timedelta(days=3):
            return 0.7
        elif delta <= timedelta(days=7):
            return 0.5
        elif delta <= timedelta(days=30):
            return 0.2
        return 0.0

    def _parse_relative_date_score(self, text: str) -> float:
        """Handle strings like '2 days ago', 'Just posted', '1 week ago'."""
        text = text.lower().strip()
        if "just" in text or "today" in text or "now" in text:
            return 1.0

        match = re.search(r"(\d+)\s*(hour|day|week|month)", text)
        if not match:
            return 0.3  # can't parse → moderate default

        num = int(match.group(1))
        unit = match.group(2)

        if unit == "hour":
            return 1.0
        elif unit == "day":
            if num <= 1:
                return 1.0
            elif num <= 3:
                return 0.7
            elif num <= 7:
                return 0.5
            elif num <= 30:
                return 0.2
            return 0.0
        elif unit == "week":
            if num <= 1:
                return 0.5
            elif num <= 4:
                return 0.2
            return 0.0
        elif unit == "month":
            if num <= 1:
                return 0.2
            return 0.0

        return 0.3

    def _platform_score(self, platform: str) -> float:
        """Trust score based on platform reliability for AI gigs."""
        return PLATFORM_TRUST_SCORES.get(platform.lower(), 0.5)
