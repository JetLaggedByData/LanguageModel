"""
v3_agentic/evaluate/benchmark.py
Full V1 vs V2 vs V3 benchmark report.

Reads:
  v1_baseline/eval_results_v1.json        (from v1_baseline/evaluate.py)
  v2_finetuned/eval_results_v2.json       (from v2_finetuned/evaluate.py)
  data/stories/*/story.json               (from pipeline/runner.py)

Writes:
  mlflow_runs/benchmark_report.json       (full metrics for Streamlit + README)
  mlflow_runs/v2_loss_curve.json          (if MLflow DB is available)

Logs everything to a single MLflow run: "v1_vs_v2_vs_v3_benchmark"

Run:
  python v3_agentic/evaluate/benchmark.py
  python v3_agentic/evaluate/benchmark.py --rerun-v1  # re-evaluate V1 first
  python v3_agentic/evaluate/benchmark.py --rerun-v2  # re-evaluate V2 first
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v3_agentic.evaluate.consistency_scorer import (
    get_avg_consistency_score,
    get_avg_revision_cycles,
    get_score_distribution,
    get_per_story_summary,
)


# ── Paths ─────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
V1_RESULTS    = ROOT / "v1_baseline"  / "eval_results_v1.json"
V2_RESULTS    = ROOT / "v2_finetuned" / "eval_results_v2.json"
REPORT_PATH   = ROOT / "mlflow_runs"  / "benchmark_report.json"
LOSS_PATH     = ROOT / "mlflow_runs"  / "v2_loss_curve.json"


# ── Loaders ───────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _run_subprocess(cmd: list[str]) -> bool:
    """Run a subprocess and return True on success."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


# ── V1 metrics ────────────────────────────────────────────────────────────

def get_v1_metrics(rerun: bool = False) -> dict:
    """
    Load V1 metrics. Re-runs evaluate.py if rerun=True or results missing.
    """
    if rerun or not V1_RESULTS.exists():
        print("Running V1 evaluation...")
        ok = _run_subprocess([
            sys.executable, str(ROOT / "v1_baseline" / "evaluate.py")
        ])
        if not ok:
            print("  ⚠️  V1 evaluation failed — using zeros.")
            return {}

    data = _load_json(V1_RESULTS)
    return {
        "v1_char_perplexity":         data.get("char_perplexity", 0.0),
        "v1_bleu2":                   data.get("bleu2", 0.0),
        "v1_inference_chars_per_sec": data.get("inference_chars_per_sec", 0.0),
        "v1_avg_sentence_length":     data.get("avg_sentence_length_chars", 0.0),
    }


# ── V2 metrics ────────────────────────────────────────────────────────────

def get_v2_metrics(rerun: bool = False) -> dict:
    """
    Load V2 metrics. Re-runs evaluate.py if rerun=True or results missing.
    """
    if rerun or not V2_RESULTS.exists():
        print("Running V2 evaluation...")
        ok = _run_subprocess([
            sys.executable, str(ROOT / "v2_finetuned" / "evaluate.py"),
            "--samples", "50",
        ])
        if not ok:
            print("  ⚠️  V2 evaluation failed — using zeros.")
            return {}

    data = _load_json(V2_RESULTS)
    return {
        "v2_word_perplexity":           data.get("word_perplexity", 0.0),
        "v2_bleu2":                     data.get("bleu2", 0.0),
        "v2_bleu4":                     data.get("bleu4", 0.0),
        "v2_inference_tokens_per_sec":  data.get("inference_tokens_per_sec", 0.0),
        "v2_genre_consistency_score":   data.get("genre_consistency_score", 0.0),
    }


# ── V3 metrics ────────────────────────────────────────────────────────────

def get_v3_metrics() -> dict:
    """Compute V3 metrics from stored story JSONs — no model inference."""
    stories_dir = ROOT / "data" / "stories"
    dist = get_score_distribution(stories_dir)

    return {
        "v3_avg_consistency_score": get_avg_consistency_score(stories_dir),
        "v3_avg_revision_cycles":   get_avg_revision_cycles(stories_dir),
        "v3_score_mean":            dist.get("mean", 0.0),
        "v3_score_median":          dist.get("median", 0.0),
        "v3_score_stdev":           dist.get("stdev", 0.0),
        "v3_chapters_excellent":    dist.get("excellent", 0),
        "v3_chapters_good":         dist.get("good", 0),
        "v3_chapters_poor":         dist.get("poor", 0),
    }


# ── Improvement deltas ────────────────────────────────────────────────────

def compute_deltas(v1: dict, v2: dict) -> dict:
    """
    Compute V1→V2 improvement percentages for the benchmark table.
    These are the headline numbers for the README and LinkedIn post.
    """
    deltas = {}

    v1_ppl = v1.get("v1_char_perplexity", 0)
    v2_ppl = v2.get("v2_word_perplexity", 0)
    if v1_ppl > 0 and v2_ppl > 0:
        deltas["perplexity_pct_change"] = round((v2_ppl - v1_ppl) / v1_ppl * 100, 1)

    v1_bleu = v1.get("v1_bleu2", 0)
    v2_bleu = v2.get("v2_bleu2", 0)
    if v1_bleu > 0:
        deltas["bleu2_pct_change"] = round((v2_bleu - v1_bleu) / max(v1_bleu, 1e-9) * 100, 1)

    return deltas


# ── MLflow loss curve export ──────────────────────────────────────────────

def export_loss_curve() -> None:
    """
    Pull V2 training loss from MLflow SQLite DB and save as JSON
    for the Model Arena training loss chart.
    Non-fatal if MLflow DB is unavailable.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        runs   = client.search_runs(
            experiment_ids=["0"],
            filter_string="tags.mlflow.runName = 'v2_qlora_finetune'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return

        run_id  = runs[0].info.run_id
        history = client.get_metric_history(run_id, "train_loss")
        curve   = [{"step": m.step, "loss": m.value} for m in history]

        LOSS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOSS_PATH.write_text(json.dumps(curve, indent=2))
        print(f"  Loss curve exported → {LOSS_PATH} ({len(curve)} steps)")

    except Exception as exc:
        print(f"  ⚠️  Could not export loss curve: {exc}")


# ── Report writer ─────────────────────────────────────────────────────────

def write_report(v1: dict, v2: dict, v3: dict, deltas: dict) -> None:
    """Write full benchmark report JSON and a human-readable summary."""
    report = {
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "deltas": deltas,
        "per_story": get_per_story_summary(ROOT / "data" / "stories"),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n── Benchmark Report ─────────────────────────────────────────────")
    print("\n  V1 LSTM:")
    for k, v in v1.items():
        print(f"    {k}: {v}")
    print("\n  V2 QLoRA:")
    for k, v in v2.items():
        print(f"    {k}: {v}")
    print("\n  V3 Agentic:")
    for k, v in v3.items():
        print(f"    {k}: {v}")
    print("\n  Deltas (V1 → V2):")
    for k, v in deltas.items():
        arrow = "▼" if "perplexity" in k and v < 0 else "▲"
        print(f"    {k}: {arrow} {v:+.1f}%")
    print(f"\n  Saved → {REPORT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────

def run_full_benchmark(rerun_v1: bool = False, rerun_v2: bool = False) -> dict:
    """
    Full V1 vs V2 vs V3 benchmark. Logs to MLflow, writes JSON report.
    Returns the full report dict.
    """
    print("SciFi Forge — Full Benchmark\n")

    print("Loading V1 metrics...")
    v1 = get_v1_metrics(rerun=rerun_v1)

    print("Loading V2 metrics...")
    v2 = get_v2_metrics(rerun=rerun_v2)

    print("Computing V3 metrics from stored stories...")
    v3 = get_v3_metrics()

    deltas = compute_deltas(v1, v2)

    with mlflow.start_run(run_name="v1_vs_v2_vs_v3_benchmark"):
        all_metrics = {**v1, **v2, **v3, **deltas}
        mlflow.log_metrics({
            k: float(v) for k, v in all_metrics.items()
            if isinstance(v, (int, float))
        })
        mlflow.log_artifact(str(REPORT_PATH)) if REPORT_PATH.exists() else None
        print("\n  Metrics logged to MLflow.")

    print("\nExporting V2 loss curve from MLflow...")
    export_loss_curve()

    write_report(v1, v2, v3, deltas)
    return {"v1": v1, "v2": v2, "v3": v3, "deltas": deltas}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SciFi Forge — full benchmark")
    parser.add_argument("--rerun-v1", action="store_true",
                        help="Re-run V1 evaluate.py before loading results")
    parser.add_argument("--rerun-v2", action="store_true",
                        help="Re-run V2 evaluate.py before loading results")
    args = parser.parse_args()
    run_full_benchmark(rerun_v1=args.rerun_v1, rerun_v2=args.rerun_v2)
