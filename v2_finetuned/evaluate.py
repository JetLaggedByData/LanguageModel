"""
v2_finetuned/evaluate.py
Benchmark evaluation for V2 fine-tuned model.
Produces eval_results_v2.json in the same schema as V1 for direct comparison.

Metrics:
  - Word-level perplexity (cross-entropy on val set)
  - BLEU-2 and BLEU-4 vs held-out val completions
  - Inference speed: tokens/sec
  - Genre consistency score: SciFi keyword density proxy

Run:
  python v2_finetuned/evaluate.py
  python v2_finetuned/evaluate.py --samples 20   # quick smoke test
"""

import gc
import sys
import json
import time
import math
import random
import argparse
import torch
import mlflow
import numpy as np
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from model_config import (
    MODEL_ID, ADAPTER_DIR, VAL_JSONL, EVAL_RESULTS,
    MAX_LENGTH, EVAL_N_SAMPLES,
    GEN_MAX_TOKENS, GEN_TEMPERATURE, GEN_TOP_P, GEN_REP_PENALTY,
)
from generate import load_model, build_prompt, generate_text, unload_model


# ── SCIFI genre keywords for consistency proxy ────────────────────────────
SCIFI_KEYWORDS = {
    "spaceship", "planet", "alien", "galaxy", "warp", "orbit", "nebula",
    "asteroid", "starship", "laser", "android", "robot", "cyborg", "clone",
    "hyperspace", "quantum", "radiation", "shuttle", "station", "crew",
    "mission", "captain", "colony", "terraform", "signal", "sensor",
    "reactor", "void", "cosmos", "stellar", "interstellar", "federation",
    "empire", "republic", "fleet", "hull", "airlock", "hangar", "docking",
}


# ── GPU helpers ───────────────────────────────────────────────────────────

def clear_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# ── Data loading ──────────────────────────────────────────────────────────

def load_val_samples(n: int = EVAL_N_SAMPLES) -> list[dict]:
    """Load n random samples from val JSONL."""
    if not VAL_JSONL.exists():
        sys.exit(f"Val JSONL not found: {VAL_JSONL}\nRun: python data/prepare_dataset.py")
    records = [json.loads(l) for l in VAL_JSONL.read_text().splitlines() if l.strip()]
    random.seed(42)
    return random.sample(records, min(n, len(records)))


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_perplexity(
    model: AutoModelForCausalLM,
    tokeniser: AutoTokenizer,
    samples: list[dict],
) -> float:
    """
    Word-level perplexity: average cross-entropy loss over val completions.
    Input = full prompt, loss measured only on the output/completion tokens.
    """
    total_loss, total_tokens = 0.0, 0
    smoother = SmoothingFunction()

    model.eval()
    with torch.no_grad():
        for sample in samples:
            prompt  = build_prompt(sample["input"])
            full    = prompt + sample["output"]

            enc = tokeniser(
                full, return_tensors="pt",
                truncation=True, max_length=MAX_LENGTH,
            ).to(model.device)

            prompt_enc = tokeniser(
                prompt, return_tensors="pt",
                truncation=True, max_length=MAX_LENGTH,
            )
            prompt_len = prompt_enc["input_ids"].shape[1]

            labels = enc["input_ids"].clone()
            labels[0, :prompt_len] = -100   # mask prompt — only score completion

            outputs = model(**enc, labels=labels)
            n_tokens = (labels[0] != -100).sum().item()
            if n_tokens > 0:
                total_loss   += outputs.loss.item() * n_tokens
                total_tokens += n_tokens

    avg_loss   = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))   # cap at e^20 to avoid inf
    return round(perplexity, 4)


def compute_bleu(
    samples: list[dict],
    model,
    tokeniser,
) -> tuple[float, float]:
    """
    BLEU-2 and BLEU-4 vs ground-truth val completions.
    Tokenised at word level (split on whitespace).
    """
    smoother  = SmoothingFunction().method1
    bleu2_scores, bleu4_scores = [], []

    for sample in samples:
        reference  = [sample["output"].split()]
        hypothesis = generate_text(
            sample["input"],
            max_new_tokens=GEN_MAX_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
        ).split()

        bleu2_scores.append(
            sentence_bleu(reference, hypothesis, weights=(0.5, 0.5),
                          smoothing_function=smoother)
        )
        bleu4_scores.append(
            sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smoother)
        )

    return round(float(np.mean(bleu2_scores)), 4), round(float(np.mean(bleu4_scores)), 4)


def compute_inference_speed(
    samples: list[dict],
    n: int = 10,
) -> float:
    """Tokens generated per second averaged over n samples."""
    tokeniser = load_model()[1]
    timings: list[float] = []

    for sample in random.sample(samples, min(n, len(samples))):
        start  = time.perf_counter()
        result = generate_text(sample["input"], max_new_tokens=GEN_MAX_TOKENS)
        elapsed = time.perf_counter() - start
        n_tokens = len(tokeniser.encode(result))
        if n_tokens > 0 and elapsed > 0:
            timings.append(n_tokens / elapsed)

    return round(float(np.mean(timings)), 2) if timings else 0.0


def compute_genre_score(samples: list[dict], n: int = 20) -> float:
    """
    SciFi genre consistency proxy: fraction of generated passages
    containing at least 3 SciFi keywords from SCIFI_KEYWORDS.
    Higher = model stays on-genre.
    """
    hits = 0
    for sample in random.sample(samples, min(n, len(samples))):
        text   = generate_text(sample["input"], max_new_tokens=GEN_MAX_TOKENS).lower()
        words  = set(text.split())
        n_hits = len(words & SCIFI_KEYWORDS)
        if n_hits >= 3:
            hits += 1
    return round(hits / n, 4)


# ── Main ──────────────────────────────────────────────────────────────────

def run_evaluation(n_samples: int = EVAL_N_SAMPLES) -> dict:
    """Run all V2 benchmark metrics, log to MLflow, save JSON."""
    print(f"Loading {n_samples} val samples...")
    samples = load_val_samples(n_samples)

    print("Loading model for evaluation...")
    model, tokeniser = load_model()

    print("Computing perplexity...")
    perplexity = compute_perplexity(model, tokeniser, samples)
    print(f"  Perplexity: {perplexity}")

    print("Computing BLEU-2 and BLEU-4...")
    bleu2, bleu4 = compute_bleu(samples, model, tokeniser)
    print(f"  BLEU-2: {bleu2}  |  BLEU-4: {bleu4}")

    print("Computing inference speed...")
    tokens_per_sec = compute_inference_speed(samples)
    print(f"  Speed: {tokens_per_sec} tokens/sec")

    print("Computing genre consistency score...")
    genre_score = compute_genre_score(samples)
    print(f"  Genre score: {genre_score}")

    results = {
        "model":            "V2_QLoRA_Qwen2.5-1.5B",
        "adapter_dir":      str(ADAPTER_DIR),
        "n_eval_samples":   n_samples,
        "word_perplexity":  perplexity,
        "bleu2":            bleu2,
        "bleu4":            bleu4,
        "inference_tokens_per_sec": tokens_per_sec,
        "genre_consistency_score":  genre_score,
    }

    EVAL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS.write_text(json.dumps(results, indent=2))

    with mlflow.start_run(run_name="v2_evaluation"):
        mlflow.log_metrics({
            "v2_word_perplexity":           perplexity,
            "v2_bleu2":                     bleu2,
            "v2_bleu4":                     bleu4,
            "v2_inference_tokens_per_sec":  tokens_per_sec,
            "v2_genre_consistency_score":   genre_score,
        })

    print("\n── V2 Benchmark Results ───────────────────────────────────")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\n  Saved → {EVAL_RESULTS}")

    unload_model()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SciFi Forge V2 — evaluation")
    parser.add_argument("--samples", type=int, default=EVAL_N_SAMPLES,
                        help=f"Number of val samples to evaluate (default: {EVAL_N_SAMPLES})")
    args = parser.parse_args()
    run_evaluation(n_samples=args.samples)
