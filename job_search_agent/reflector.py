"""
Reflector module — uses the LLM to analyse search performance,
produce actionable insights, AND suggest scoring weight adjustments.
"""

import json
import logging
import re

from .memory import Memory
from .models import SearchSession

log = logging.getLogger(__name__)


class ReflectionResult:
    """Structured output from the Reflector."""

    def __init__(self, reflection: str, weight_adjustments: dict[str, float]):
        self.reflection = reflection
        self.weight_adjustments = weight_adjustments


class Reflector:
    """
    Analyses each search session, writes a natural-language reflection,
    and suggests scoring weight adjustments to improve future scoring.
    """

    def __init__(self, llm_client, memory: Memory):
        self.llm = llm_client
        self.memory = memory

    def reflect(self, session: SearchSession) -> ReflectionResult:
        """
        Produce a reflection + weight adjustments for the given session.
        Returns a ReflectionResult with both text and structured adjustments.
        """
        perf_summary = self.memory.get_query_performance_summary()
        top_jobs_data = self._top_jobs_snapshot(session)
        score_trend = self.memory.get_score_trend()
        best_score = self.memory.get_best_score()

        prompt = self._build_prompt(session, perf_summary, top_jobs_data, score_trend, best_score)

        try:
            response = self.llm.invoke(prompt)
            raw = response.content.strip()
            log.info(f"Reflector produced output ({len(raw)} chars)")
            return self._parse_response(raw)
        except Exception as e:
            log.warning(f"Reflector LLM call failed: {e}")
            return ReflectionResult(
                reflection=self._fallback_reflection(session),
                weight_adjustments={},
            )

    # ── Prompt ──────────────────────────────────────────

    def _build_prompt(
        self,
        session: SearchSession,
        perf_summary: str,
        top_jobs_data: str,
        score_trend: list[float],
        best_score: float,
    ) -> str:
        trend_str = " → ".join(f"{s:.3f}" for s in score_trend) if score_trend else "N/A"
        return f"""You are a job search strategist analysing an AI agent that is AUTONOMOUSLY searching for the perfect beginner AI job.

The agent will KEEP RUNNING until it finds a job scoring ≥ 0.85 or stagnates. Your job is to help it converge faster.

CURRENT ITERATION #{session.iteration} STATS:
- Queries used: {json.dumps(session.queries_used)}
- Total jobs found: {session.total_jobs_found}
- Average score this round: {session.avg_score:.3f}
- Best score ever found: {best_score:.3f}
- Score trend (last iterations): {trend_str}

TOP SCORING JOBS THIS ROUND:
{top_jobs_data}

HISTORICAL PERFORMANCE:
{perf_summary}

CURRENT SCORING WEIGHTS:
  keyword_match: 0.30, experience_level: 0.25, remote_available: 0.20,
  pay_clarity: 0.10, recency: 0.10, platform_trust: 0.05

Respond with EXACTLY this JSON format (no markdown fences, no extra text):
{{
  "reflection": "4-6 sentence analysis: what worked, what failed, specific strategies for next round",
  "weight_adjustments": {{
    "keyword_match": 0.0,
    "experience_level": 0.0,
    "remote_available": 0.0,
    "pay_clarity": 0.0,
    "recency": 0.0,
    "platform_trust": 0.0
  }}
}}

RULES for weight_adjustments:
- Values must be small deltas between -0.05 and +0.05
- Use 0.0 if no change needed for that criterion
- If high-scoring jobs share a pattern (e.g. all remote), INCREASE that weight
- If a criterion doesn't help distinguish good jobs, DECREASE it
- The reflection should suggest 2-3 NEW query ideas that are different from past queries"""

    def _parse_response(self, raw: str) -> ReflectionResult:
        """Parse the LLM response into structured ReflectionResult."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
            reflection = data.get("reflection", "")
            adjustments = data.get("weight_adjustments", {})

            # Validate adjustments are reasonable
            valid_keys = {"keyword_match", "experience_level", "remote_available",
                          "pay_clarity", "recency", "platform_trust"}
            safe_adjustments = {}
            for k, v in adjustments.items():
                if k in valid_keys and isinstance(v, (int, float)):
                    safe_adjustments[k] = max(-0.05, min(0.05, float(v)))

            return ReflectionResult(reflection=reflection, weight_adjustments=safe_adjustments)
        except (json.JSONDecodeError, AttributeError, TypeError):
            log.warning("Could not parse structured reflection, using raw text.")
            return ReflectionResult(reflection=raw, weight_adjustments={})

    def _top_jobs_snapshot(self, session: SearchSession, n: int = 5) -> str:
        """Create a compact text summary of the top N jobs from this session."""
        all_jobs = session.all_jobs
        top = sorted(all_jobs, key=lambda j: j.score, reverse=True)[:n]

        if not top:
            return "No jobs found in this iteration."

        lines = []
        for j in top:
            breakdown = ", ".join(f"{k}={v:.2f}" for k, v in j.score_breakdown.items())
            lines.append(
                f"  • [{j.score:.2f}] \"{j.title}\" at {j.company} ({j.platform}) "
                f"— remote={j.is_remote}, query=\"{j.raw_query}\""
                f"\n    Breakdown: {breakdown}"
            )
        return "\n".join(lines)

    def _fallback_reflection(self, session: SearchSession) -> str:
        """Simple heuristic reflection when the LLM is unavailable."""
        avg = session.avg_score
        n = session.total_jobs_found
        return (
            f"Iteration {session.iteration} found {n} jobs with avg score {avg:.3f}. "
            f"Consider broadening queries if few results, or narrowing if scores are low. "
            f"Try adding specific platform names or role titles to queries."
        )
