"""
Configuration for the Job Search AI Agent.
All tuneable knobs, API keys, scoring weights, and platform definitions live here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MEMORY_DIR = PROJECT_ROOT / "memory"
DB_PATH = MEMORY_DIR / "sessions.db"
OUTPUT_DIR = MEMORY_DIR / "outputs"

# Ensure dirs exist
MEMORY_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# LLM Configuration (Ollama)
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
LLM_TEMPERATURE = 0.4  # lower = more focused reasoning

# ──────────────────────────────────────────────
# Scoring Weights (must sum to 1.0)
# ──────────────────────────────────────────────
SCORING_WEIGHTS: dict[str, float] = {
    "keyword_match": 0.30,
    "experience_level": 0.25,
    "remote_available": 0.20,
    "pay_clarity": 0.10,
    "recency": 0.10,
    "platform_trust": 0.05,
}

# ──────────────────────────────────────────────
# Target Keywords for beginner AI jobs
# ──────────────────────────────────────────────
TARGET_KEYWORDS: list[str] = [
    "data annotation",
    "data labeling",
    "image labeling",
    "text classification",
    "content moderation",
    "AI trainer",
    "RLHF",
    "prompt engineering",
    "data collection",
    "entry level",
    "no experience required",
    "remote",
    "freelance",
    "crowdsourcing",
    "nepal",
    "nepali",
    "worldwide",
    "global remote",
    "anywhere in the world"
]

# ──────────────────────────────────────────────
# Platform Configuration
# ──────────────────────────────────────────────
PLATFORMS: list[str] = [
    "indeed",
    "linkedin",
    "upwork",
    "appen",
]

PLATFORM_TRUST_SCORES: dict[str, float] = {
    "appen": 1.0,
    "scale_ai": 1.0,
    "upwork": 0.8,
    "indeed": 0.7,
    "linkedin": 0.7,
    "freelancer": 0.6,
}

# ──────────────────────────────────────────────
# Agent Behaviour
# ──────────────────────────────────────────────
MAX_JOBS_PER_QUERY = 20
MAX_ITERATIONS = 50                 # safety cap — agent usually stops earlier
PERFECT_SCORE_THRESHOLD = 0.85      # stop when best job scores ≥ this
MAX_STAGNANT_ITERATIONS = 5         # stop if no improvement for N consecutive rounds
SCORE_IMPROVEMENT_THRESHOLD = 0.02  # minimum avg score gain to count as "improving"
QUERIES_PER_ITERATION = 7
REQUEST_DELAY_RANGE = (1.0, 3.0)    # seconds between HTTP requests (rate-limit)

# ──────────────────────────────────────────────
# Seed Queries (used on iteration 0)
# ──────────────────────────────────────────────
SEED_QUERIES: list[str] = [
    "data annotation remote entry level nepal",
    "image labeling no experience worldwide",
    "AI content moderation beginner global remote",
    "RLHF data trainer remote anywhere",
    "text classification freelance nepal",
    "AI training data collection remote worldwide",
    "nepali speaking data labeling crowdsource",
]
