"""
Central configuration for the SR-MA pipeline.
Set environment variables or replace placeholder values before running.
"""

import os
import pathlib

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent
SKILLS_DIR = PROJECT_ROOT / "skills"
META_SKILLS_DIR = PROJECT_ROOT / "meta_skills"
CONTEXT_STORE_DIR = PROJECT_ROOT / "context_store"
CONTEXT_STORE_DIR.mkdir(exist_ok=True)

# ── OpenRouter / LLM Config ───────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
# OPENROUTER_API_KEY = "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"
# Model assignments (OpenRouter model IDs)
MODELS = {
    # General executor / adjudicator
    "default":     "qwen/qwen3.5-plus-02-15",
    # Heterogeneous dual-review pair (must be from different model families)
    "reviewer_a":  "deepseek/deepseek-v3.2",
    "reviewer_b":  "google/gemini-3-flash-preview",
    "adjudicator": "qwen/qwen3.5-plus-02-15",
}

# LLM call parameters
LLM_TEMPERATURE   = 0.1
LLM_MAX_TOKENS    = 4096
LLM_MAX_RETRIES   = 2      # Max correction retries when Skill validation fails

# ── NCBI / PubMed Config ──────────────────────────────────────────────────────
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "2f5aca5502939d9a8cc59b3e5b8b3e9fc509")
NCBI_EMAIL   = os.getenv("NCBI_EMAIL",   "2022302191541@whu.edu.cn")
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# PubMed search defaults
PUBMED_MAX_RESULTS = 5000   # Max records per query
PUBMED_RETMAX      = 10000  # Max UIDs per fetch call

# ── Pipeline defaults ─────────────────────────────────────────────────────────
# Screening: Kappa thresholds
KAPPA_GOOD       = 0.80   # >=0.80: excellent, no special action
KAPPA_ACCEPTABLE = 0.60   # 0.60-0.80: acceptable, log warning
KAPPA_POOR       = 0.40   # 0.40-0.60: poor, flag for human review
                           # <0.40: block and require calibration

# RAG search augmentation
RAG_TOP_K = 5             # Top-K abstracts for RAG-augmented query expansion

# Citation chaining
CITATION_CO_CITE_THRESHOLD = 3  # Min co-citation frequency to include

# Extraction
FULL_TEXT_FALLBACK_PDF = True   # Fall back to PDF if PMC XML not available
