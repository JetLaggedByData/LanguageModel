"""
data/verify_dataset.py
Sanity-check the generated JSONL files before kicking off fine-tuning.

Checks:
  - Files exist and are non-empty
  - Every line is valid JSON with required keys
  - No duplicate samples (input hash collision check)
  - Length distribution looks sane
  - Prints 3 random samples for manual inspection

Run:
  python data/verify_dataset.py
"""

import json
import random
import hashlib
from pathlib import Path
from collections import Counter

from dataset_config import TRAIN_JSONL, VAL_JSONL, STATS_JSON


REQUIRED_KEYS = {"instruction", "input", "output"}
SAMPLE_PREVIEW_N = 3


# ── Loaders ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a .jsonl file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {e}") from e
    return records


# ── Checks ────────────────────────────────────────────────────────────────

def check_schema(records: list[dict], split: str) -> list[str]:
    """Return list of error messages for schema violations."""
    errors = []
    for i, rec in enumerate(records):
        missing = REQUIRED_KEYS - set(rec.keys())
        if missing:
            errors.append(f"[{split}] Record {i} missing keys: {missing}")
        for key in REQUIRED_KEYS:
            if key in rec and not isinstance(rec[key], str):
                errors.append(f"[{split}] Record {i} key '{key}' is not a string")
    return errors


def check_duplicates(records: list[dict], split: str) -> tuple[int, list[str]]:
    """Hash input fields and return duplicate count + example messages."""
    hashes = [hashlib.md5(r["input"].encode()).hexdigest() for r in records]
    counts = Counter(hashes)
    dupes = {h: c for h, c in counts.items() if c > 1}
    messages = []
    for h, c in list(dupes.items())[:5]:       # show up to 5 examples
        messages.append(f"[{split}] Hash {h[:8]}... appears {c} times")
    return len(dupes), messages


def length_stats(records: list[dict], split: str) -> dict:
    """Return input/output length percentiles."""
    input_lens  = [len(r["input"])  for r in records]
    output_lens = [len(r["output"]) for r in records]

    def percentiles(vals: list[int]) -> dict:
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        return {
            "min":  vals_sorted[0],
            "p10":  vals_sorted[n // 10],
            "p50":  vals_sorted[n // 2],
            "p90":  vals_sorted[int(n * 0.9)],
            "max":  vals_sorted[-1],
            "mean": round(sum(vals) / n, 1),
        }

    return {
        "split":  split,
        "n":      len(records),
        "input":  percentiles(input_lens),
        "output": percentiles(output_lens),
    }


def check_empty_fields(records: list[dict], split: str) -> list[str]:
    """Flag records where input or output is suspiciously short."""
    errors = []
    for i, rec in enumerate(records):
        if len(rec.get("input", "")) < 50:
            errors.append(f"[{split}] Record {i} input too short ({len(rec['input'])} chars)")
        if len(rec.get("output", "")) < 20:
            errors.append(f"[{split}] Record {i} output too short ({len(rec['output'])} chars)")
    return errors


def print_samples(records: list[dict], split: str, n: int = SAMPLE_PREVIEW_N) -> None:
    """Print n random samples for manual eyeballing."""
    print(f"\n── Random Samples from {split} ──────────────────────────────")
    for rec in random.sample(records, min(n, len(records))):
        print(f"\n  [instruction] {rec['instruction']}")
        print(f"  [input]  ...{rec['input'][-200:].strip()!r}")
        print(f"  [output] {rec['output'][:200].strip()!r}...")
        print("  " + "─" * 60)


# ── Main ──────────────────────────────────────────────────────────────────

def verify(train_path: Path = TRAIN_JSONL, val_path: Path = VAL_JSONL) -> bool:
    """
    Run all checks. Returns True if dataset passes, False if errors found.
    """
    all_errors: list[str] = []
    all_warnings: list[str] = []

    for path, split in [(train_path, "train"), (val_path, "val")]:
        print(f"\nChecking {split}: {path}")

        if not path.exists():
            all_errors.append(f"[{split}] File not found: {path}")
            continue

        records = load_jsonl(path)
        if not records:
            all_errors.append(f"[{split}] File is empty")
            continue

        print(f"  Records loaded: {len(records):,}")

        # Schema
        schema_errors = check_schema(records, split)
        all_errors.extend(schema_errors)

        # Duplicates
        dupe_count, dupe_msgs = check_duplicates(records, split)
        if dupe_count > 0:
            all_warnings.append(
                f"[{split}] {dupe_count} duplicate inputs found (may be acceptable for sliding window)"
            )
            all_warnings.extend(dupe_msgs[:3])

        # Empty fields
        empty_errors = check_empty_fields(records, split)
        if len(empty_errors) > 10:
            all_errors.append(f"[{split}] {len(empty_errors)} records with suspiciously short fields")
        else:
            all_errors.extend(empty_errors)

        # Length stats
        stats = length_stats(records, split)
        print(f"  Input  length — min:{stats['input']['min']} "
              f"p50:{stats['input']['p50']} p90:{stats['input']['p90']} "
              f"max:{stats['input']['max']} mean:{stats['input']['mean']}")
        print(f"  Output length — min:{stats['output']['min']} "
              f"p50:{stats['output']['p50']} p90:{stats['output']['p90']} "
              f"max:{stats['output']['max']} mean:{stats['output']['mean']}")

        # Instruction uniformity
        instructions = Counter(r["instruction"] for r in records)
        if len(instructions) > 1:
            all_warnings.append(f"[{split}] Multiple instruction variants: {dict(instructions)}")

        print_samples(records, split)

    # Load and print saved stats
    if STATS_JSON.exists():
        saved = json.loads(STATS_JSON.read_text())
        print(f"\n── Saved Dataset Stats ({STATS_JSON}) ───────────────────────")
        for k, v in saved.items():
            print(f"  {k}: {v}")

    # Report
    print("\n── Verification Summary ─────────────────────────────────────")
    if all_warnings:
        print(f"  ⚠️  Warnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"    {w}")

    if all_errors:
        print(f"  ❌ Errors ({len(all_errors)}):")
        for e in all_errors:
            print(f"    {e}")
        print("\nDataset has errors — fix before fine-tuning.")
        return False

    print("  ✅ All checks passed — dataset is ready for fine-tuning.")
    return True


if __name__ == "__main__":
    random.seed(0)
    passed = verify()
    raise SystemExit(0 if passed else 1)
