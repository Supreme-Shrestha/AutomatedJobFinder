"""
Agent Controller — the main orchestration loop.

Runs the Plan → Search → Evaluate → Reflect → Adapt cycle
UNTIL it finds a perfect job (score ≥ threshold) or stagnates.

Usage:
    python -m job_search_agent
    python -m job_search_agent --iterations 10
    python -m job_search_agent --dry-run
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
    Orchestrates the autonomous, self-improving job-search loop.

    The agent keeps iterating until ONE of these conditions is met:
      1. ✅ A job scoring ≥ PERFECT_SCORE_THRESHOLD is found
      2. 🛑 No improvement for MAX_STAGNANT_ITERATIONS consecutive rounds
      3. 🔒 MAX_ITERATIONS safety cap is reached

    Each iteration:
      1. Planner generates queries (LLM on iter ≥ 1)
      2. Searcher fetches listings from multiple platforms
      3. Evaluator scores every listing
      4. Reflector analyses results AND suggests weight adjustments
      5. Evaluator adapts its scoring weights based on reflection
      6. Reporter writes a structured JSON report
      7. Memory persists everything for future iterations
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.memory = Memory()
        self.evaluator = Evaluator()
        self.searcher = Searcher()
        self.reporter = Reporter()

        # LLM client — connects to local Ollama instance
        self.llm = self._init_llm()
        self.planner = Planner(self.llm, self.memory)
        self.reflector = Reflector(self.llm, self.memory)

    def _init_llm(self):
        """Initialise the LangChain ChatModel using Ollama."""
        try:
            from langchain_ollama import ChatOllama

            log.info(
                f"Connecting to Ollama at {config.OLLAMA_BASE_URL} "
                f"with model '{config.LLM_MODEL}'"
            )
            return ChatOllama(
                model=config.LLM_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.LLM_TEMPERATURE,
            )
        except ImportError:
            log.error(
                "langchain-ollama is not installed. "
                "Run: pip install langchain-ollama"
            )
            raise SystemExit(1)

    # ── Main Loop ───────────────────────────────────────

    def run(self, max_iterations: int | None = None):
        """
        Execute the self-improving search loop.
        Keeps running until a perfect job is found or the agent stagnates.
        """
        max_iter = max_iterations or config.MAX_ITERATIONS
        reflection_text = ""
        stagnant_count = 0
        prev_best_score = 0.0

        print(
            f"\n{'═' * 60}\n"
            f"  🤖 Job-Search AI Agent v{__import__('job_search_agent').__version__}\n"
            f"  Mode           : Search until perfect match\n"
            f"  Perfect score  : ≥ {config.PERFECT_SCORE_THRESHOLD}\n"
            f"  Safety cap     : {max_iter} iterations\n"
            f"  Platforms      : {', '.join(config.PLATFORMS)}\n"
            f"  LLM model      : {config.LLM_MODEL}\n"
            f"  Dry run        : {self.dry_run}\n"
            f"{'═' * 60}\n"
        )

        for iteration in range(max_iter):
            print(f"\n{'━' * 60}")
            print(f"  ▶ ITERATION {iteration + 1} (stagnant: {stagnant_count}/{config.MAX_STAGNANT_ITERATIONS})")
            print(f"{'━' * 60}")

            # ── Run one full cycle ──────────────────────
            session, best_job_score = self._run_iteration(iteration, reflection_text)

            # Print console summary
            self.reporter.print_summary(session, session.all_jobs)

            # ── CHECK: Perfect job found? ───────────────
            if best_job_score >= config.PERFECT_SCORE_THRESHOLD:
                print(f"\n  🎯 PERFECT MATCH FOUND! Score: {best_job_score:.3f}")
                print(f"  Stopping after {iteration + 1} iterations.\n")
                break

            # ── CHECK: Is the agent improving? ──────────
            improvement = best_job_score - prev_best_score
            if improvement < config.SCORE_IMPROVEMENT_THRESHOLD and iteration > 0:
                stagnant_count += 1
                log.info(
                    f"No significant improvement ({improvement:+.3f}). "
                    f"Stagnant count: {stagnant_count}/{config.MAX_STAGNANT_ITERATIONS}"
                )
            else:
                stagnant_count = 0  # reset on improvement
                if improvement > 0:
                    log.info(f"Score improved by {improvement:+.3f} ✓")

            # ── CHECK: Stagnation limit? ────────────────
            if stagnant_count >= config.MAX_STAGNANT_ITERATIONS:
                print(
                    f"\n  🛑 No improvement for {stagnant_count} iterations. "
                    f"Best score achieved: {best_job_score:.3f}\n"
                )
                break

            prev_best_score = max(prev_best_score, best_job_score)
            reflection_text = session.reflection

        # Final summary
        self._print_final_summary()

    def _run_iteration(self, iteration: int, reflection: str) -> tuple:
        """
        Execute one full Plan → Search → Evaluate → Reflect → Adapt cycle.
        Returns (session, best_job_score).
        """

        # ── STEP 1: PLAN ────────────────────────────────
        log.info("Step 1/6: Planning queries…")
        if iteration == 0:
            queries = self.planner.get_initial_queries()
        else:
            queries = self.planner.generate_queries(iteration, reflection)

        log.info(f"  Queries: {queries}")

        # ── STEP 2: SEARCH ──────────────────────────────
        log.info("Step 2/6: Searching platforms…")
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

        log.info(f"Step 3/6: Evaluated {len(all_jobs)} total jobs")

        # Track best score this iteration
        best_job_score = max((j.score for j in all_jobs), default=0.0)

        # Build session object
        session = SearchSession(
            session_id=str(uuid.uuid4()),
            iteration=iteration,
            queries_used=queries,
            results=query_results,
            timestamp=datetime.now(),
        )

        # ── STEP 4: REFLECT ────────────────────────────
        log.info("Step 4/6: Reflecting on results…")
        if all_jobs:
            result = self.reflector.reflect(session)
            session.reflection = result.reflection

            # ── STEP 5: ADAPT WEIGHTS ──────────────────
            if result.weight_adjustments:
                log.info(f"Step 5/6: Adapting scoring weights…")
                log.info(f"  Adjustments: {result.weight_adjustments}")
                self.evaluator.update_weights(result.weight_adjustments)
                log.info(f"  New weights: { {k: round(v, 3) for k, v in self.evaluator.weights.items()} }")

                # Re-score all jobs with updated weights
                all_jobs = self.evaluator.score_all(all_jobs)
                best_job_score = max((j.score for j in all_jobs), default=best_job_score)
                log.info(f"  Re-scored best: {best_job_score:.3f}")
            else:
                log.info("Step 5/6: No weight adjustments this round.")

            session.next_queries = self.planner.generate_queries(
                iteration + 1, session.reflection
            )
        else:
            session.reflection = "No jobs found this iteration. Broaden queries."
            session.next_queries = self.planner.get_initial_queries()
            log.info("Step 5/6: Skipped (no jobs to reflect on).")

        # ── STEP 6: SAVE & REPORT ──────────────────────
        log.info("Step 6/6: Saving to memory & writing report…")
        self.memory.save_session(session)
        self.memory.save_jobs(all_jobs, session.session_id)
        report_path = self.reporter.write_report(session, all_jobs[:15])
        log.info(f"  Report: {report_path}")

        return session, best_job_score

    def _print_final_summary(self):
        """Print aggregate summary across all iterations."""
        top_jobs = self.memory.get_top_jobs(10)
        best = self.memory.get_best_score()
        trend = self.memory.get_score_trend()

        print(f"\n{'═' * 60}")
        print("  ✅ AGENT COMPLETE — Top 10 Jobs Overall")
        print(f"{'═' * 60}")
        print(f"  Best score achieved : {best:.3f}")
        if trend:
            print(f"  Score trend         : {' → '.join(f'{s:.3f}' for s in trend)}")
        print(f"  Final weights       : { {k: round(v, 3) for k, v in self.evaluator.weights.items()} }")

        if not top_jobs:
            print("\n  No jobs found. Check that Ollama is running and platforms are accessible.")
            return

        print()
        for i, j in enumerate(top_jobs, 1):
            remote = "🌐" if j.get("is_remote") else "🏢"
            score = j["score"]
            marker = " ⭐" if score >= config.PERFECT_SCORE_THRESHOLD else ""
            print(
                f"  {i:>2}. [{score:.2f}]{marker} {remote} {j['title']}"
                f" @ {j['company']} ({j['platform']})"
            )
            if j.get("url"):
                print(f"      🔗 {j['url']}")

        print(f"\n  📁 All reports saved in: {config.OUTPUT_DIR}")
        print(f"  🗄️  Memory DB: {config.DB_PATH}")
        print(f"{'═' * 60}\n")


# ── CLI Entry Point ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Job-Search AI Agent — searches until it finds the perfect job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=config.MAX_ITERATIONS,
        help=f"Max iterations (safety cap, default: {config.MAX_ITERATIONS})",
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
