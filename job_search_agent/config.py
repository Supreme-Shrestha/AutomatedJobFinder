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
# LLM Configuration
# ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
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
MAX_ITERATIONS = 5
SCORE_IMPROVEMENT_THRESHOLD = 0.05  # stop if avg score improves < 5%
QUERIES_PER_ITERATION = 7
REQUEST_DELAY_RANGE = (1.0, 3.0)  # seconds between HTTP requests (rate-limit)

# ──────────────────────────────────────────────
# Seed Queries (used on iteration 0)
# ──────────────────────────────────────────────
SEED_QUERIES: list[str] = [
    "data annotation remote entry level",
    "image labeling no experience",
    "AI content moderation beginner",
    "RLHF data trainer remote",
    "text classification freelance",
    "AI training data collection remote",
    "data labeling crowdsource work from home",
]
