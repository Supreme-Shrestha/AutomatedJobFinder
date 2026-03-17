"""
Reflector module — uses the LLM to analyse search performance and
produce actionable insights for the Planner to improve queries.
"""

import json
import logging

from .memory import Memory
from .models import SearchSession

log = logging.getLogger(__name__)


class Reflector:
    """
    Analyses each search session and writes a natural-language reflection.
    The reflection is stored in Memory and consumed by the Planner.
    """

    def __init__(self, llm_client, memory: Memory):
        self.llm = llm_client
        self.memory = memory

    def reflect(self, session: SearchSession) -> str:
        """
        Produce a reflection for the given session.
        Compares current performance against historical data.
        Returns a concise strategy note (3–6 sentences).
        """
        perf_summary = self.memory.get_query_performance_summary()
        top_jobs_data = self._top_jobs_snapshot(session)

        prompt = self._build_prompt(session, perf_summary, top_jobs_data)

        try:
            response = self.llm.invoke(prompt)
            reflection = response.content.strip()
            log.info(f"Reflector produced reflection ({len(reflection)} chars)")
            return reflection
        except Exception as e:
            log.warning(f"Reflector LLM call failed: {e}")
            return self._fallback_reflection(session)

    # ── Prompt ──────────────────────────────────────────

    def _build_prompt(
        self, session: SearchSession, perf_summary: str, top_jobs_data: str
    ) -> str:
        return f"""You are a job search strategist analysing an AI agent's performance.

CURRENT ITERATION #{session.iteration} STATS:
- Queries used: {json.dumps(session.queries_used)}
- Total jobs found: {session.total_jobs_found}
- Average score: {session.avg_score:.3f}

TOP SCORING JOBS THIS ROUND:
{top_jobs_data}

HISTORICAL PERFORMANCE:
{perf_summary}

TASK — Write a concise reflection (4–6 sentences) addressing:
1. Which queries performed best and why?
2. What patterns do you see in high-scoring jobs (keywords, platforms, job types)?
3. What should change in the next iteration to find better results?
4. Suggest 2–3 specific new query ideas or platform strategies to try.

Be specific, data-driven, and actionable. No fluff."""

    def _top_jobs_snapshot(self, session: SearchSession, n: int = 5) -> str:
        """Create a compact text summary of the top N jobs from this session."""
        all_jobs = session.all_jobs
        top = sorted(all_jobs, key=lambda j: j.score, reverse=True)[:n]

        if not top:
            return "No jobs found in this iteration."

        lines = []
        for j in top:
            lines.append(
                f"  • [{j.score:.2f}] \"{j.title}\" at {j.company} ({j.platform}) "
                f"— remote={j.is_remote}, query=\"{j.raw_query}\""
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
