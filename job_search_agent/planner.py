"""
Planner module — uses the LLM to generate optimized search queries.
On iteration 0 it uses static seed queries; on later iterations it
incorporates reflections and past performance.
"""

import json
import logging
from typing import Optional

from .config import LLM_MODEL, LLM_TEMPERATURE, QUERIES_PER_ITERATION, SEED_QUERIES
from .memory import Memory

log = logging.getLogger(__name__)


class Planner:
    """
    Generates search queries for each iteration.
    Uses LLM reasoning on iteration N>0 to improve queries.
    """

    def __init__(self, llm_client, memory: Memory):
        """
        Args:
            llm_client: A LangChain ChatModel (e.g., ChatOpenAI).
            memory: The Memory instance for retrieving past performance.
        """
        self.llm = llm_client
        self.memory = memory

    # ── Public API ──────────────────────────────────────

    def get_initial_queries(self) -> list[str]:
        """
        Return seed queries for iteration 0.
        No LLM call needed — these are hand-crafted starting points.
        """
        return SEED_QUERIES.copy()

    def generate_queries(self, iteration: int, reflection: str = "") -> list[str]:
        """
        Use the LLM to generate improved search queries based on
        past performance and the latest reflection.

        Returns a list of query strings (typically 5–7).
        Falls back to seed queries if LLM call fails.
        """
        past_performance = self.memory.get_query_performance_summary()

        prompt = self._build_prompt(iteration, reflection, past_performance)

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # The LLM should return a JSON list of strings
            queries = self._parse_queries(content)
            log.info(f"Planner generated {len(queries)} queries for iteration {iteration}")
            return queries

        except Exception as e:
            log.warning(f"Planner LLM call failed: {e}. Falling back to seed queries.")
            return self.get_initial_queries()

    # ── Prompt Engineering ──────────────────────────────

    def _build_prompt(self, iteration: int, reflection: str, past_performance: str) -> str:
        return f"""You are a job search strategist helping a beginner AI student find entry-level AI jobs.

TARGET ROLES:
- Data annotation / data labeling
- Image classification / labeling
- Text classification / content moderation
- AI training data collection
- RLHF (Reinforcement Learning from Human Feedback) tasks
- Prompt engineering / AI evaluation
- Crowdsourced AI tasks (Appen, Scale AI, etc.)

CONSTRAINTS:
- Jobs must be suitable for beginners with no prior professional experience.
- Prefer REMOTE jobs.
- Prefer platforms: Indeed, LinkedIn, Upwork, RemoteOK, Appen.

CURRENT STATE:
- Iteration: {iteration}
- Past query performance:
{past_performance}

REFLECTION FROM LAST ITERATION:
{reflection if reflection else "N/A — this is an early iteration."}

TASK:
Generate exactly {QUERIES_PER_ITERATION} optimized search queries.
Each query should be a short string (3-8 words) suitable for pasting into a job search bar.
Prioritise queries that are DIFFERENT from past low-performing queries.
Include at least 2 queries targeting different platforms or job types than previous rounds.

RESPOND WITH ONLY a JSON array of strings. No explanation, no markdown formatting.
Example: ["query one", "query two", "query three"]"""

    def _parse_queries(self, raw: str) -> list[str]:
        """
        Parse the LLM response into a list of query strings.
        Handles markdown code fences and minor formatting issues.
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening and closing fences
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            queries = json.loads(cleaned)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:QUERIES_PER_ITERATION]
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract quoted strings
        import re
        found = re.findall(r'"([^"]+)"', raw)
        if found:
            return found[:QUERIES_PER_ITERATION]

        log.warning("Could not parse LLM response into queries.")
        return self.get_initial_queries()
