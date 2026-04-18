"""
v2_finetuned/model_config.py
All constants for V2 QLoRA fine-tuning and inference.
Change MODEL_ID here to swap to the GPT-2 fallback if VRAM is tight.
"""

from pathlib import Path

# ── Model ─────────────────────────────────────────────────────────────────
MODEL_ID        = "Qwen/Qwen2.5-1.5B-Instruct"   # ~3GB in 4-bit — 8GB safe
FALLBACK_MODEL  = "gpt2-large"                    # ~2GB, use if OOM persists

# ── Paths ─────────────────────────────────────────────────────────────────
_ROOT           = Path(__file__).resolve().parent.parent   # project root
ADAPTER_DIR     = _ROOT / "v2_finetuned/adapters"
TRAIN_JSONL     = _ROOT / "data/chunks/scifi_train.jsonl"
VAL_JSONL       = _ROOT / "data/chunks/scifi_val.jsonl"
EVAL_RESULTS    = _ROOT / "v2_finetuned/eval_results_v2.json"

# ── Tokeniser ─────────────────────────────────────────────────────────────
MAX_LENGTH      = 640       # 512 context + 128 completion tokens (with buffer)
PADDING_SIDE    = "right"

# ── LoRA (skill spec values — do not change) ──────────────────────────────
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
LORA_TARGETS    = ["q_proj", "v_proj"]

# ── Training (skill spec values — VRAM-safe, do not change) ───────────────
BATCH_SIZE      = 1         # VRAM hard limit
GRAD_ACCUM      = 8         # effective batch = 8
EPOCHS          = 3
LR              = 2e-4
WARMUP_RATIO    = 0.03
LOGGING_STEPS   = 50
SAVE_STEPS      = 200
SAVE_TOTAL_LIMIT = 2
WEIGHT_DECAY    = 0.01

# ── Generation defaults ───────────────────────────────────────────────────
GEN_MAX_TOKENS  = 300
GEN_TEMPERATURE = 0.7
GEN_TOP_P       = 0.9
GEN_REP_PENALTY = 1.1      # discourages repetition common in fine-tuned LLMs

# ── Eval ──────────────────────────────────────────────────────────────────
EVAL_N_SAMPLES  = 100       # passages to score for perplexity/BLEU
BLEU_REF_KEY    = "output"  # field in val JSONL used as BLEU reference
