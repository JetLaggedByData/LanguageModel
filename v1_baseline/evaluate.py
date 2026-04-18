"""
v1_baseline/evaluate.py
Benchmark evaluation for V1 LSTM baseline — PyTorch port.
Produces eval_results_v1.json for the V1 vs V2 vs V3 comparison table.

Metrics:
  - Character-level perplexity on held-out test slice
  - BLEU-2 score vs held-out SciFi passages
  - Inference speed (chars/sec)
  - Avg sentence length (coherence proxy)

Run:
  python v1_baseline/evaluate.py
  python v1_baseline/evaluate.py --samples 20  # quick smoke test
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from lstm_model import build_lstm, loss_fn
from train      import load_and_preprocess, DATA_PATH, CHECKPOINT_DIR, DEVICE, SEQUENCE_LENGTH
from generate   import load_inference_model, generate_text


# ── Config ────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT   = CHECKPOINT_DIR / "checkpt_best.pt"
TEST_SAMPLE_SIZE     = 1_000
GENERATION_SAMPLE    = 500
BLEU_REFERENCE_START = 149_290_000
BLEU_REFERENCE_LEN   = 500
RESULTS_PATH         = Path("eval_results_v1.json")


# ── Perplexity ────────────────────────────────────────────────────────────

def compute_perplexity(
    model,
    int_text:   np.ndarray,
    sample_size: int = TEST_SAMPLE_SIZE,
) -> float:
    """Character-level perplexity on held-out tail of corpus."""
    model.eval()
    model.reset_states()

    test_slice = torch.tensor(
        int_text[-(sample_size + 1):], dtype=torch.long
    ).to(DEVICE)

    inputs  = test_slice[:-1].unsqueeze(0)   # (1, T)
    targets = test_slice[1:].view(-1)        # (T,)

    with torch.no_grad():
        logits, _ = model(inputs)            # (1, T, vocab)
        logits_flat = logits.view(-1, logits.size(-1))
        loss = nn.functional.cross_entropy(logits_flat, targets)

    return round(math.exp(loss.item()), 4)


# ── Inference speed ───────────────────────────────────────────────────────

def compute_inference_speed(
    model,
    chartoindex:  dict,
    indextochar:  np.ndarray,
    seed:         str = "The spaceship landed on",
    num_chars:    int = GENERATION_SAMPLE,
) -> float:
    """Characters generated per second."""
    start   = time.perf_counter()
    generate_text(model, chartoindex, indextochar, seed, num=num_chars, temperature=0.5)
    elapsed = time.perf_counter() - start
    return round(num_chars / elapsed, 2)


# ── BLEU-2 ────────────────────────────────────────────────────────────────

def compute_bleu2(
    model,
    chartoindex:  dict,
    indextochar:  np.ndarray,
    int_text:     np.ndarray,
) -> float:
    """Char-level BLEU-2 vs held-out corpus passage."""
    reference_chars = indextochar[
        int_text[BLEU_REFERENCE_START: BLEU_REFERENCE_START + BLEU_REFERENCE_LEN]
    ]
    reference = [list(reference_chars)]
    seed      = "".join(reference_chars[:20])
    generated = generate_text(
        model, chartoindex, indextochar, seed,
        num=BLEU_REFERENCE_LEN, temperature=0.5,
    )
    hypothesis = list(generated[len(seed): len(seed) + BLEU_REFERENCE_LEN])
    smoother   = SmoothingFunction().method1
    return round(sentence_bleu(reference, hypothesis,
                               weights=(0.5, 0.5),
                               smoothing_function=smoother), 4)


# ── Sentence length ───────────────────────────────────────────────────────

def compute_avg_sentence_length(text: str) -> float:
    sentences = [s.strip() for s in
                 text.replace("?", ".").replace("!", ".").split(".")
                 if s.strip()]
    return round(float(np.mean([len(s) for s in sentences])), 2) if sentences else 0.0


# ── Main ──────────────────────────────────────────────────────────────────

def run_evaluation(checkpoint: Path = DEFAULT_CHECKPOINT) -> dict:
    print(f"Loading model from {checkpoint}...")
    model, chartoindex, indextochar = load_inference_model(checkpoint)

    int_text, _, _, _ = load_and_preprocess(DATA_PATH)

    print("Computing perplexity...")
    perplexity = compute_perplexity(model, int_text)

    print("Computing inference speed...")
    chars_per_sec = compute_inference_speed(model, chartoindex, indextochar)

    print("Computing BLEU-2...")
    bleu2 = compute_bleu2(model, chartoindex, indextochar, int_text)

    seed      = "The alien vessel emerged from hyperspace"
    generated = generate_text(model, chartoindex, indextochar, seed, num=500)
    avg_len   = compute_avg_sentence_length(generated[len(seed):])

    results = {
        "model":                    "V1_LSTM_PyTorch",
        "checkpoint":               str(checkpoint),
        "char_perplexity":          perplexity,
        "bleu2":                    bleu2,
        "inference_chars_per_sec":  chars_per_sec,
        "avg_sentence_length_chars": avg_len,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n── V1 Benchmark Results ──────────────────────────────────")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\nSaved → {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    run_evaluation(checkpoint=args.checkpoint)
