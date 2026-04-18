"""
data/dataset_config.py
Single source of truth for all data preparation constants.
Adjust CHARS_TO_USE and TARGET_SAMPLES here if you need a smaller test run.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).resolve().parent.parent   # project root
RAW_DIR        = _ROOT / "data/raw"
CHUNKS_DIR     = _ROOT / "data/chunks"
RAW_CORPUS     = RAW_DIR / "internet_archive_scifi_v3.txt"

TRAIN_JSONL    = CHUNKS_DIR / "scifi_train.jsonl"
VAL_JSONL      = CHUNKS_DIR / "scifi_val.jsonl"
STATS_JSON     = CHUNKS_DIR / "dataset_stats.json"

# ── Corpus slice (matches skill spec: first 10M chars) ────────────────────
CORPUS_START   = 580            # skip header (same as V1 notebook)
CORPUS_END     = 10_000_580     # first 10M usable characters

# ── Sliding window parameters ─────────────────────────────────────────────
# Skill spec: 512 token context, 128 token completion
# Using characters here; tokeniser will handle token alignment at train time.
# Qwen2.5 averages ~3.5 chars/token → 512 tok ≈ 1792 chars, 128 tok ≈ 448 chars
CONTEXT_CHARS  = 1_792          # input passage length in characters
COMPLETION_CHARS = 448          # continuation length in characters
WINDOW_STEP    = 200            # stride between samples (dense overlap)

# ── Dataset size ──────────────────────────────────────────────────────────
TARGET_SAMPLES = 50_000         # skill spec: 50k samples
VAL_FRACTION   = 0.05           # 5% validation split → ~2,500 val samples

# ── Instruction template (skill spec format) ──────────────────────────────
INSTRUCTION    = "Continue this science fiction story:"

# ── Quality filters ───────────────────────────────────────────────────────
MIN_ALPHA_RATIO   = 0.70        # drop chunks with < 70% alphabetic chars
MIN_WORDS         = 60          # drop passages shorter than 60 words
MAX_REPEAT_CHAR   = 0.15        # drop if any single char > 15% of passage
