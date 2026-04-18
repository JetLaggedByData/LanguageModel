"""
scripts/pregenerate_stories.py
Generate 5 seed stories locally using the full V3 GPU pipeline,
then export benchmark charts — making the repo deployment-ready.

Run this ONCE locally before pushing to HF Spaces:
  python scripts/pregenerate_stories.py

What it does:
  1. Runs the full V3 pipeline (Planner→Writer→Critic→Editor) for each seed
  2. Saves story JSON to data/stories/<story_id>/story.json
  3. Runs the benchmark report (reads V1 + V2 eval JSONs)
  4. Exports all 4 Plotly charts to mlflow_runs/charts/

After this script completes, commit the outputs:
  git add data/stories/ mlflow_runs/charts/ mlflow_runs/benchmark_report.json
  git commit -m "Add pre-generated stories and benchmark charts"
  git push

Flags:
  --stories-only   Skip benchmark + chart export
  --charts-only    Skip story generation, just rebuild charts
  --dry-run        Generate 1 chapter per story for quick testing
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from v3_agentic.pipeline.runner import run_pipeline


# ── Seed stories (skill spec seeds) ──────────────────────────────────────
SEEDS = [
    "A dying Earth colony ship discovers an alien signal emanating from an abandoned moon",
    "The last human city on Mars faces a catastrophic energy crisis as the sun dims",
    "An AI archaeologist uncovers evidence of a previous technological civilisation buried beneath Europa",
    "A time-locked space station receives a distress message that originated from fifty years in the future",
    "Two rival megacorporations race to terraform Venus — but only one can survive the process",
]

# Generation config
DEFAULT_CHAPTERS  = 4
DEFAULT_REVISIONS = 2
DRY_RUN_CHAPTERS  = 1


# ── Story generation ──────────────────────────────────────────────────────

def generate_stories(chapters: int, revisions: int) -> list[dict]:
    """Run pipeline for all seeds. Returns list of story output dicts."""
    results = []
    total   = len(SEEDS)

    for i, seed in enumerate(SEEDS, 1):
        print(f"\n{'='*65}")
        print(f"  Story {i}/{total}: {seed[:70]}...")
        print(f"{'='*65}")

        try:
            result = run_pipeline(
                seed_prompt=seed,
                total_chapters=chapters,
                max_revisions=revisions,
                mlflow_run_name=f"pregenerate_story_{i:02d}",
            )
            title = result.get("title", f"Story {i}")
            n_ch  = len(result.get("chapters", []))
            err   = result.get("error")
            print(f"\n  ✅ Generated: '{title}' | {n_ch} chapters | error={err}")
            results.append(result)
        except Exception as exc:
            print(f"\n  ❌ Failed: {exc}")

    return results


# ── Benchmark + chart export ──────────────────────────────────────────────

def run_benchmark() -> None:
    """Run benchmark.py then export_charts.py as subprocesses."""
    print("\n" + "="*65)
    print("  Running full benchmark report...")
    print("="*65)

    benchmark_script = ROOT / "v3_agentic" / "evaluate" / "benchmark.py"
    result = subprocess.run(
        [sys.executable, str(benchmark_script)],
        capture_output=False,
    )
    if result.returncode != 0:
        print("  ⚠️  Benchmark script failed — check eval JSONs exist.")
        return

    print("\n" + "="*65)
    print("  Exporting Plotly charts...")
    print("="*65)

    export_script = ROOT / "v3_agentic" / "evaluate" / "export_charts.py"
    subprocess.run([sys.executable, str(export_script)], capture_output=False)


# ── Summary ───────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    print("\n" + "="*65)
    print("  PRE-GENERATION SUMMARY")
    print("="*65)
    for i, r in enumerate(results, 1):
        title   = r.get("title", "Untitled")
        n_ch    = len(r.get("chapters", []))
        chapters = r.get("chapters", [])
        scores  = [c.get("critique_score") for c in chapters if c.get("critique_score")]
        avg     = f"{sum(scores)/len(scores):.2f}" if scores else "—"
        story_id = r.get("story_id", "")
        print(f"  {i}. {title[:50]:<50} | {n_ch} ch | avg score {avg} | {story_id}")

    print(f"\n  Stories saved to: {ROOT / 'data' / 'stories'}/")
    print(f"  Charts saved to:  {ROOT / 'mlflow_runs' / 'charts'}/")
    print("""
  Next steps:
    git add data/stories/ mlflow_runs/charts/ mlflow_runs/benchmark_report.json
    git commit -m "Add pre-generated stories and benchmark charts"
    git push
    git push space main   # deploy to HF Spaces
""")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SciFi Forge — pre-generate stories")
    parser.add_argument("--stories-only", action="store_true",
                        help="Skip benchmark and chart export")
    parser.add_argument("--charts-only",  action="store_true",
                        help="Skip story generation, rebuild charts only")
    parser.add_argument("--dry-run",      action="store_true",
                        help=f"Generate only {DRY_RUN_CHAPTERS} chapter per story (fast test)")
    args = parser.parse_args()

    chapters  = DRY_RUN_CHAPTERS if args.dry_run else DEFAULT_CHAPTERS
    revisions = 1 if args.dry_run else DEFAULT_REVISIONS

    if args.dry_run:
        print(f"DRY RUN — {chapters} chapter per story, {revisions} revision max\n")

    results: list[dict] = []

    if not args.charts_only:
        results = generate_stories(chapters, revisions)

    if not args.stories_only:
        run_benchmark()

    print_summary(results)


if __name__ == "__main__":
    main()
