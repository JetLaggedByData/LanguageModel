"""
v3_agentic/evaluate/consistency_scorer.py
Standalone utilities for computing V3 story-level metrics
from stored story JSON files.

Used by benchmark.py and the Model Arena Streamlit page.
No model inference — operates entirely on stored JSON output.
"""

import json
import statistics
from pathlib import Path


STORIES_DIR = Path(__file__).resolve().parent.parent.parent / "data/stories"


# ── Story loader ──────────────────────────────────────────────────────────

def load_all_story_jsons(stories_dir: Path = STORIES_DIR) -> list[dict]:
    """Load all story.json files. Returns empty list if none found."""
    stories = []
    for path in sorted(stories_dir.glob("*/story.json")):
        try:
            stories.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return stories


# ── Per-story metrics ─────────────────────────────────────────────────────

def story_avg_consistency(story: dict) -> float | None:
    """Average consistency_score across all chapters in one story."""
    scores = [
        ch.get("critique_score")
        for ch in story.get("chapters", [])
        if ch.get("critique_score") is not None
    ]
    return round(statistics.mean(scores), 4) if scores else None


def story_chapter_count(story: dict) -> int:
    return len(story.get("chapters", []))


def story_error_rate(story: dict) -> float:
    """Fraction of chapters with generation errors."""
    chapters = story.get("chapters", [])
    if not chapters:
        return 0.0
    failed = sum(
        1 for ch in chapters
        if str(ch.get("content", "")).startswith("[Generation failed")
    )
    return round(failed / len(chapters), 4)


# ── Corpus-level metrics ──────────────────────────────────────────────────

def get_avg_consistency_score(stories_dir: Path = STORIES_DIR) -> float:
    """
    Mean composite critique score across ALL chapters in ALL stories.
    Used as the V3 headline metric in the benchmark table.
    """
    all_scores = []
    for story in load_all_story_jsons(stories_dir):
        for ch in story.get("chapters", []):
            s = ch.get("critique_score")
            if s is not None:
                all_scores.append(float(s))
    return round(statistics.mean(all_scores), 4) if all_scores else 0.0


def get_avg_revision_cycles(stories_dir: Path = STORIES_DIR) -> float:
    """
    Estimate average Editor revision cycles per chapter.

    Proxy: chapters with critique_score < 0.6 were likely revised at least once.
    (Exact revision_count is not stored in story.json — only in pipeline state.)
    """
    total_chapters, revised = 0, 0
    for story in load_all_story_jsons(stories_dir):
        for ch in story.get("chapters", []):
            s = ch.get("critique_score")
            if s is not None:
                total_chapters += 1
                if float(s) < 0.6:    # proxy: matches REVISION_THRESHOLD in critic.py
                    revised += 1
    if total_chapters == 0:
        return 0.0
    return round(revised / total_chapters, 4)


def get_score_distribution(stories_dir: Path = STORIES_DIR) -> dict:
    """
    Return score distribution buckets for the Model Arena histogram.
    Buckets: excellent (≥0.8), good (0.6–0.8), poor (<0.6)
    """
    all_scores = []
    for story in load_all_story_jsons(stories_dir):
        for ch in story.get("chapters", []):
            s = ch.get("critique_score")
            if s is not None:
                all_scores.append(float(s))

    if not all_scores:
        return {"excellent": 0, "good": 0, "poor": 0, "all_scores": []}

    return {
        "excellent":  sum(1 for s in all_scores if s >= 0.8),
        "good":       sum(1 for s in all_scores if 0.6 <= s < 0.8),
        "poor":       sum(1 for s in all_scores if s < 0.6),
        "mean":       round(statistics.mean(all_scores), 4),
        "median":     round(statistics.median(all_scores), 4),
        "stdev":      round(statistics.stdev(all_scores), 4) if len(all_scores) > 1 else 0.0,
        "all_scores": all_scores,
    }


def get_per_story_summary(stories_dir: Path = STORIES_DIR) -> list[dict]:
    """
    Return one summary dict per story for the benchmark report table.
    """
    rows = []
    for story in load_all_story_jsons(stories_dir):
        rows.append({
            "story_id":    story.get("story_id", "unknown"),
            "title":       story.get("title", "Untitled"),
            "chapters":    story_chapter_count(story),
            "avg_score":   story_avg_consistency(story),
            "error_rate":  story_error_rate(story),
        })
    return rows
