# 🤖 Job-Search AI Agent

An autonomous, self-improving AI agent that searches multiple job platforms for beginner-friendly AI roles
(data annotation, labeling, RLHF, content moderation, etc.), scores results for relevance,
and uses LLM-powered reflection to improve queries over time.

## Features

- **Multi-platform search** — Indeed, LinkedIn, RemoteOK, Appen
- **Weighted scoring rubric** — 6 criteria: keywords, experience level, remote, pay, recency, platform trust
- **LLM-powered planning** — GPT-4o generates optimised search queries
- **Self-improving reflection** — analyses what worked and adapts strategy
- **Persistent memory** — SQLite stores all sessions, jobs, and reflections
- **Structured JSON output** — per-session reports with scores and reasoning

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# 3. Run the agent
python -m job_search_agent

# Or with options:
python -m job_search_agent --iterations 3
python -m job_search_agent --dry-run    # no web requests
```

## Architecture

```
AgentController → Planner → Searcher → Evaluator → Reflector → Reporter
                     ↑                                  │
                     └──────── Memory (SQLite) ←────────┘
```

| Module | Role |
|---|---|
| `agent.py` | Orchestrates the Plan → Search → Evaluate → Reflect loop |
| `planner.py` | LLM-powered query generation |
| `searcher.py` | Multi-platform scraping (Indeed, LinkedIn, RemoteOK, Appen) |
| `evaluator.py` | Scores jobs using weighted rubric (0–1) |
| `reflector.py` | LLM analyses performance and generates strategy notes |
| `reporter.py` | JSON reports + console summaries |
| `memory.py` | SQLite persistence for sessions, jobs, query performance |
| `models.py` | Pydantic data schemas |
| `config.py` | All tuneable parameters |

## Output

Each iteration produces a JSON report in `job_search_agent/memory/outputs/`:

```json
{
  "session_id": "abc-123",
  "iteration": 2,
  "summary": { "total_jobs_found": 87, "avg_score": 0.61, "top_score": 0.89 },
  "top_jobs": [ { "title": "Data Annotator", "score": 0.89, ... } ],
  "reflection": "Queries with 'remote no experience' scored highest...",
  "next_queries_planned": ["AI data trainer remote beginner", ...]
}
```

## Configuration

Edit `config.py` to adjust:
- **Scoring weights** — change how job relevance is measured
- **Target keywords** — add domain-specific terms
- **Platforms** — enable/disable scrapers
- **MAX_ITERATIONS** — how many improvement cycles to run
- **SEED_QUERIES** — initial queries for the first iteration

## License

MIT
