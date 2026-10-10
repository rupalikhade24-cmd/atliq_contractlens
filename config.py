"""Central configuration. All secrets come from .env (no keys in code)."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

# Vercel sets VERCEL=1. Its functions can only write to /tmp and can't run a local
# embedding model, so the deployed app keeps runtime files in /tmp and uses keyword search.
ON_VERCEL = bool(os.getenv("VERCEL"))

# --- Paths ----------------------------------------------------------------
# The capstone dataset sits next to this folder (..\Dataset) on the laptop, or inside it
# (Dataset\) in the deployed repo.
_default_dataset = BASE_DIR / "Dataset" if (BASE_DIR / "Dataset").is_dir() else BASE_DIR.parent / "Dataset"
_ds = Path(os.getenv("DATASET_DIR", _default_dataset))
DATASET_DIR = (_ds if _ds.is_absolute() else BASE_DIR / _ds).resolve()
DATA_DIR = BASE_DIR / "data"            # playbook, register seed, demo files
_RUNTIME = Path("/tmp/contractlens") if ON_VERCEL else BASE_DIR
REVIEWS_DIR = _RUNTIME / "reviews"      # one JSON per reviewed contract (brief + decisions)
CHROMA_DIR = BASE_DIR / os.getenv("CHROMA_DIR", "chroma_store")
LOG_DIR = _RUNTIME / "logs"

# The dataset README: treat "today" as 28 September 2026.
TODAY = date.fromisoformat(os.getenv("CL_TODAY", "2026-09-28"))

# --- Models ---------------------------------------------------------------
# Local embeddings: no embedding API cost, and retrieval never sends contract text anywhere.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Generation LLM on Groq, same provider and settings style as the RAG chatbot.
# Two tiers (PRD section 5): a fast model for extraction, the main one for clause scan,
# conflict reasoning and Q&A. Both default to the chatbot's model.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL_FAST = os.getenv("GROQ_MODEL_FAST", GROQ_MODEL)
TEMPERATURE = 0  # deterministic, factual output
REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "none")
# Groq's free tier rejects a request whose expected output breaches the per-minute
# ceiling, so every call sets an explicit cap.
MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "2000"))
# Long contracts are scanned in parts of this many characters (about 3,500 tokens each).
SCAN_CHARS = int(os.getenv("SCAN_CHARS", "14000"))

# --- Retrieval ------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "6"))
# "chroma": local embeddings + Chroma (laptop, same as the RAG chatbot).
# "keyword": BM25 over the same clause chunks, no model (Vercel).
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "keyword" if ON_VERCEL else "chroma")


def require_api_key() -> str:
    """Fail loudly and early rather than deep inside a review."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(free tier: https://console.groq.com)."
        )
    return GROQ_API_KEY
