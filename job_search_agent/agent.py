"""
Agent Controller — the main orchestration loop.

Runs the Plan → Search → Evaluate → Reflect → Report cycle
for up to MAX_ITERATIONS, with early stopping when improvement stalls.

Usage:
    python -m job_search_agent.agent
    python -m job_search_agent.agent --iterations 3
    python -m job_search_agent.agent --dry-run
"""

import argparse
import logging
import uuid
from datetime import datetime

from . import config
from .evaluator import Evaluator
from .memory import Memory
from .models import QueryResult, SearchSession
from .planner import Planner
from .reflector import Reflector
from .reporter import Reporter
from .searcher import Searcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")


class AgentController:
    """
    Orchestrates the autonomous job-search loop.

    Each iteration:
      1. Planner generates queries (LLM on iter ≥ 1)
      2. Searcher fetches listings from multiple platforms
      3. Evaluator scores every listing
      4. Reflector analyses results and produces strategy notes
      5. Reporter writes a structured JSON report
      6. Memory persists everything for future iterations
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.memory = Memory()
        self.evaluator = Evaluator()
        self.searcher = Searcher()
        self.reporter = Reporter()

        # LLM client — only create if we have an API key
        self.llm = self._init_llm()
        self.planner = Planner(self.llm, self.memory)
        self.reflector = Reflector(self.llm, self.memory)

    def _init_llm(self):
        """Initialise the LangChain ChatModel."""
        if not config.OPENAI_API_KEY:
            log.warning(
                "OPENAI_API_KEY not set. Planner/Reflector will use fallback mode."
            )
            return _DummyLLM()

        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.LLM_MODEL,
                api_key=config.OPENAI_API_KEY,
                temperature=config.LLM_TEMPERATURE,
            )
        except ImportError:
            log.warning("langchain-openai not installed. Using fallback LLM.")
            return _DummyLLM()

    # ── Main Loop ───────────────────────────────────────

    def run(self, max_iterations: int | None = None):
        """Execute the agentic search loop."""
        max_iter = max_iterations or config.MAX_ITERATIONS
        prev_avg_score = 0.0
        reflection = ""

        print(
            f"\n{'═' * 60}\n"
            f"  🤖 Job-Search AI Agent v{__import__('job_search_agent').__version__}\n"
            f"  Max iterations : {max_iter}\n"
            f"  Platforms      : {', '.join(config.PLATFORMS)}\n"
            f"  LLM model      : {config.LLM_MODEL}\n"
            f"  Dry run        : {self.dry_run}\n"
            f"{'═' * 60}\n"
        )

        for iteration in range(max_iter):
            print(f"\n{'━' * 60}")
            print(f"  ▶ ITERATION {iteration + 1} / {max_iter}")
            print(f"{'━' * 60}")

            session = self._run_iteration(iteration, reflection)

            # Print console summary
            self.reporter.print_summary(session, session.all_jobs)

            # Check for improvement → early stopping
            avg = session.avg_score
            improvement = avg - prev_avg_score

            if iteration > 0 and improvement < config.SCORE_IMPROVEMENT_THRESHOLD:
                log.info(
                    f"Minimal improvement ({improvement:+.3f}). "
                    f"Stopping after {iteration + 1} iterations."
                )
                break

            prev_avg_score = avg
            reflection = session.reflection

        # Final summary
        self._print_final_summary()

    def _run_iteration(self, iteration: int, reflection: str) -> SearchSession:
        """Execute one full Plan → Search → Evaluate → Reflect → Report cycle."""

        # ── STEP 1: PLAN ────────────────────────────────
        log.info("Step 1/5: Planning queries…")
        if iteration == 0:
            queries = self.planner.get_initial_queries()
        else:
            queries = self.planner.generate_queries(iteration, reflection)

        log.info(f"  Queries: {queries}")

        # ── STEP 2: SEARCH ──────────────────────────────
        log.info("Step 2/5: Searching platforms…")
        query_results: list[QueryResult] = []
        all_jobs = []
        seen_ids = self.memory.get_seen_job_ids()

        for query in queries:
            if self.dry_run:
                log.info(f"  [DRY RUN] Would search: '{query}'")
                raw_jobs = []
            else:
                raw_jobs = self.searcher.search_all(query)

            # Deduplicate against memory
            new_jobs = [j for j in raw_jobs if j.id not in seen_ids]
            for j in new_jobs:
                seen_ids.add(j.id)

            # ── STEP 3: EVALUATE ────────────────────────
            scored_jobs = self.evaluator.score_all(new_jobs)

            qr = QueryResult(
                query=query,
                platform="all",
                jobs_found=len(scored_jobs),
                avg_score=(
                    sum(j.score for j in scored_jobs) / len(scored_jobs)
                    if scored_jobs
                    else 0.0
                ),
                top_jobs=scored_jobs[:config.MAX_JOBS_PER_QUERY],
                timestamp=datetime.now(),
            )
            query_results.append(qr)
            all_jobs.extend(scored_jobs)

        log.info(f"Step 3/5: Evaluated {len(all_jobs)} total jobs")

        # Build session object
        session = SearchSession(
            session_id=str(uuid.uuid4()),
            iteration=iteration,
            queries_used=queries,
            results=query_results,
            timestamp=datetime.now(),
        )

        # ── STEP 4: REFLECT ────────────────────────────
        log.info("Step 4/5: Reflecting on results…")
        if all_jobs:
            session.reflection = self.reflector.reflect(session)
            session.next_queries = self.planner.generate_queries(
                iteration + 1, session.reflection
            )
        else:
            session.reflection = "No jobs found this iteration. Broaden queries."
            session.next_queries = self.planner.get_initial_queries()

        # ── STEP 5: SAVE & REPORT ──────────────────────
        log.info("Step 5/5: Saving to memory & writing report…")
        self.memory.save_session(session)
        self.memory.save_jobs(all_jobs, session.session_id)
        report_path = self.reporter.write_report(session, all_jobs[:15])
        log.info(f"  Report: {report_path}")

        return session

    def _print_final_summary(self):
        """Print aggregate summary across all iterations."""
        top_jobs = self.memory.get_top_jobs(10)

        print(f"\n{'═' * 60}")
        print("  ✅ AGENT COMPLETE — Top 10 Jobs Overall")
        print(f"{'═' * 60}")

        if not top_jobs:
            print("  No jobs found. Try adjusting config or adding API keys.")
            return

        for i, j in enumerate(top_jobs, 1):
            remote = "🌐" if j.get("is_remote") else "🏢"
            print(
                f"  {i:>2}. [{j['score']:.2f}] {remote} {j['title']}"
                f" @ {j['company']} ({j['platform']})"
            )
            if j.get("url"):
                print(f"      🔗 {j['url']}")

        print(f"\n  📁 All reports saved in: {config.OUTPUT_DIR}")
        print(f"  🗄️  Memory DB: {config.DB_PATH}")
        print(f"{'═' * 60}\n")


class _DummyLLM:
    """
    Fallback LLM that returns empty responses.
    Used when no API key is configured.
    """

    def invoke(self, prompt: str):
        class _R:
            content = "[]"

        log.debug("DummyLLM invoked — returning empty response")
        return _R()


# ── CLI Entry Point ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Job-Search AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=config.MAX_ITERATIONS,
        help=f"Max iterations to run (default: {config.MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual web requests; test the pipeline logic only",
    )

    args = parser.parse_args()
    agent = AgentController(dry_run=args.dry_run)
    agent.run(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
