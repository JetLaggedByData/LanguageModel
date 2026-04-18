"""
data/prepare_dataset.py
Step 2: Data preparation pipeline for QLoRA fine-tuning (V2).

Reads:  data/raw/internet_archive_scifi_v3.txt
Writes: data/chunks/scifi_train.jsonl
        data/chunks/scifi_val.jsonl
        data/chunks/dataset_stats.json

Output format (one JSON object per line):
  {
    "instruction": "Continue this science fiction story:",
    "input":  "<512-token context passage>",
    "output": "<128-token completion passage>"
  }

Run:
  python data/prepare_dataset.py
  python data/prepare_dataset.py --dry-run   # process 1000 samples only
"""

import re
import json
import random
import argparse
from pathlib import Path
from typing import Iterator

from dataset_config import (
    RAW_CORPUS, CHUNKS_DIR, TRAIN_JSONL, VAL_JSONL, STATS_JSON,
    CORPUS_START, CORPUS_END,
    CONTEXT_CHARS, COMPLETION_CHARS, WINDOW_STEP,
    TARGET_SAMPLES, VAL_FRACTION,
    INSTRUCTION, MIN_ALPHA_RATIO, MIN_WORDS, MAX_REPEAT_CHAR,
)


# ── Text cleaning ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise corpus text:
    - Collapse runs of whitespace / tabs to single space
    - Normalise smart quotes and dashes to ASCII equivalents
    - Strip leading/trailing whitespace
    """
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Quality filters ───────────────────────────────────────────────────────

def is_quality_passage(text: str) -> bool:
    """
    Return True if passage meets minimum quality thresholds.
    Filters out header/footer junk, OCR artefacts, and repetitive noise.
    """
    if len(text.split()) < MIN_WORDS:
        return False

    total = len(text)
    if total == 0:
        return False

    alpha_ratio = sum(c.isalpha() for c in text) / total
    if alpha_ratio < MIN_ALPHA_RATIO:
        return False

    for char in set(text):
        if text.count(char) / total > MAX_REPEAT_CHAR:
            return False

    return True


# ── Sliding window chunker ────────────────────────────────────────────────

def sliding_window_samples(
    text: str,
    context_chars: int = CONTEXT_CHARS,
    completion_chars: int = COMPLETION_CHARS,
    step: int = WINDOW_STEP,
) -> Iterator[dict]:
    """
    Yield instruction-format dicts using a sliding window over `text`.
    Each sample: CONTEXT_CHARS of input + COMPLETION_CHARS of output.
    Window advances by `step` characters.
    """
    window = context_chars + completion_chars
    pos = 0
    while pos + window <= len(text):
        chunk = text[pos: pos + window]

        # Split at nearest sentence boundary within ±50 chars of context end
        split_point = context_chars
        search_start = max(0, context_chars - 50)
        search_end = min(window, context_chars + 50)
        sentence_end = chunk.rfind(". ", search_start, search_end)
        if sentence_end != -1:
            split_point = sentence_end + 2       # include the space after period

        input_text  = chunk[:split_point].strip()
        output_text = chunk[split_point:split_point + completion_chars].strip()

        if input_text and output_text and is_quality_passage(input_text):
            yield {
                "instruction": INSTRUCTION,
                "input":  input_text,
                "output": output_text,
            }

        pos += step


# ── Train / val split ─────────────────────────────────────────────────────

def split_samples(
    samples: list[dict],
    val_fraction: float = VAL_FRACTION,
) -> tuple[list[dict], list[dict]]:
    """Shuffle then split into train / val."""
    random.shuffle(samples)
    val_n = max(1, int(len(samples) * val_fraction))
    return samples[val_n:], samples[:val_n]


# ── Writers ───────────────────────────────────────────────────────────────

def write_jsonl(samples: list[dict], path: Path) -> None:
    """Write list of dicts to a .jsonl file, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(samples):,} samples → {path}")


def write_stats(train: list, val: list, path: Path) -> None:
    """Save dataset statistics for later reference in benchmarks."""
    all_samples = train + val
    avg_input_len  = sum(len(s["input"])  for s in all_samples) / len(all_samples)
    avg_output_len = sum(len(s["output"]) for s in all_samples) / len(all_samples)
    stats = {
        "total_samples":      len(all_samples),
        "train_samples":      len(train),
        "val_samples":        len(val),
        "avg_input_chars":    round(avg_input_len,  1),
        "avg_output_chars":   round(avg_output_len, 1),
        "context_chars":      CONTEXT_CHARS,
        "completion_chars":   COMPLETION_CHARS,
        "window_step":        WINDOW_STEP,
        "corpus_chars_used":  CORPUS_END - CORPUS_START,
        "instruction":        INSTRUCTION,
    }
    path.write_text(json.dumps(stats, indent=2))
    print(f"\n── Dataset Stats ────────────────────────")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n  Saved stats → {path}")


# ── Main pipeline ─────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    """
    Full pipeline:
      1. Load and clean corpus slice (first 10M chars)
      2. Generate sliding-window samples
      3. Filter to TARGET_SAMPLES with quality checks
      4. Split train/val and write JSONL files
    """
    print(f"Loading corpus: {RAW_CORPUS}")
    if not RAW_CORPUS.exists():
        raise FileNotFoundError(
            f"Corpus not found at {RAW_CORPUS}\n"
            "Place internet_archive_scifi_v3.txt in data/raw/"
        )

    raw = RAW_CORPUS.read_text(encoding="utf-8")
    corpus = clean_text(raw[CORPUS_START:CORPUS_END])
    print(f"Corpus slice: {len(corpus):,} characters after cleaning")

    limit = 1_000 if dry_run else TARGET_SAMPLES
    print(f"Generating samples (target: {limit:,}, dry_run={dry_run})...")

    samples: list[dict] = []
    for sample in sliding_window_samples(corpus):
        samples.append(sample)
        if len(samples) % 5_000 == 0:
            print(f"  {len(samples):,} samples collected...")
        if len(samples) >= limit:
            break

    print(f"Total quality samples collected: {len(samples):,}")

    random.seed(42)
    train_samples, val_samples = split_samples(samples)

    print("\nWriting JSONL files...")
    write_jsonl(train_samples, TRAIN_JSONL)
    write_jsonl(val_samples,   VAL_JSONL)
    write_stats(train_samples, val_samples, STATS_JSON)

    print("\n✅ Data prep complete.")
    print(f"   Train: {TRAIN_JSONL}")
    print(f"   Val:   {VAL_JSONL}")
    print(f"   Stats: {STATS_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SciFi Forge — data prep pipeline")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate only 1,000 samples for fast testing"
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)
