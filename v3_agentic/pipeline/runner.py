"""
v3_agentic/pipeline/runner.py
Entry point for a full V3 story generation run.

Supports:
  - CLI: python v3_agentic/pipeline/runner.py --prompt "..." --chapters 6
  - Programmatic: result = run_pipeline(seed_prompt="...", total_chapters=6)
  - Streaming: stream_pipeline(...) yields (node_name, full_state) for live Streamlit UI

All runs are logged to MLflow with story_id as the run name.
Final story JSON is saved to data/stories/<story_id>/story.json.
"""

# ── CUDA library preload ───────────────────────────────────────────────────
import ctypes as _ctypes
import site as _site
from pathlib import Path as _Path

for _site_dir in _site.getsitepackages():
    _nvjit = _Path(_site_dir) / "nvidia/cu13/lib/libnvJitLink.so.13"
    if _nvjit.exists():
        try:
            _ctypes.CDLL(str(_nvjit), mode=_ctypes.RTLD_GLOBAL)
        except OSError:
            pass
        break

del _ctypes, _site, _Path

import json
import uuid
import argparse
import mlflow
from pathlib import Path
from typing import Iterator

from pipeline.state  import StoryState, initial_state
from pipeline.graph  import get_app


STORIES_DIR = Path(__file__).resolve().parent.parent.parent / "data/stories"


def story_output_path(story_id: str) -> Path:
    return STORIES_DIR / story_id / "story.json"


def _thread_id(seed_prompt: str) -> str:
    """Unique thread ID per run — deterministic IDs caused LangGraph to resume
    a prior checkpoint on repeated prompts instead of starting fresh."""
    return str(uuid.uuid4())


# ── Serialisation ─────────────────────────────────────────────────────────

def state_to_output(state: StoryState) -> dict:
    """
    Extract the user-facing story output from the final pipeline state.
    Strips internal control fields (revision_count, status, etc.).
    """
    chapters = state.get("chapters_written", [])
    scores   = [c.get("critique_score") for c in chapters if c.get("critique_score") is not None]
    any_writer_error = any(
        str(c.get("content", "")).startswith("[Generation failed")
        for c in chapters
    )
    if not chapters:
        completion_status = "failed"
    elif any_writer_error:
        completion_status = "failed"
    elif scores:
        completion_status = "complete"
    else:
        completion_status = "partial"

    return {
        "story_id":          state.get("story_id", ""),
        "title":             state.get("title", "Untitled"),
        "logline":           state.get("logline", ""),
        "seed_prompt":       state.get("seed_prompt", ""),
        "characters":        state.get("characters", []),
        "world_rules":       state.get("world_rules", []),
        "technology":        state.get("technology", []),
        "acts":              state.get("acts", []),
        "chapters":          chapters,
        "total_chapters":    state.get("total_chapters", 0),
        "error":             state.get("error"),
        "generated_by":      "V3_agentic",
        "completion_status": completion_status,
    }


def save_story(state: StoryState) -> Path:
    """Write final story JSON to data/stories/<story_id>/story.json."""
    story_id = state.get("story_id") or "unknown"
    output   = state_to_output(state)
    path     = story_output_path(story_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    return path


# ── MLflow helpers ────────────────────────────────────────────────────────

def log_run_metrics(state: StoryState) -> None:
    chapters = state.get("chapters_written", [])
    scores = [
        c.get("critique_score", 0.0)
        for c in chapters
        if c.get("critique_score") is not None
    ]
    mlflow.log_params({
        "story_id":       state.get("story_id", ""),
        "total_chapters": state.get("total_chapters", 0),
        "max_revisions":  state.get("max_revisions", 2),
        "seed_prompt":    state.get("seed_prompt", "")[:200],
    })
    if scores:
        mlflow.log_metric("avg_critique_score", round(sum(scores) / len(scores), 4))
        mlflow.log_metric("min_critique_score", round(min(scores), 4))
    mlflow.log_metric("chapters_written", len(chapters))


# ── Stream interface (for Streamlit live feed) ────────────────────────────

def stream_pipeline(
    seed_prompt:    str,
    total_chapters: int = 6,
    max_revisions:  int = 2,
    model_id:       str | None = None,
) -> Iterator[tuple[str, StoryState]]:
    """
    Stream intermediate states from the pipeline.
    Each yield is (node_name, full_accumulated_state) after the node completes.

    Yields the full accumulated state (not partial updates) so callers can
    safely read any field — including title/logline set by the planner — from
    every event, regardless of which node last ran.

    Usage in Streamlit:
        for node_name, state in stream_pipeline(prompt, chapters=6):
            st.write(f"Status: {state['status']} | Chapter: {state['current_chapter']}")
    """
    if model_id:
        from agents.model_loader import set_agent_model
        set_agent_model(model_id)

    state       = initial_state(seed_prompt, total_chapters, max_revisions)
    config      = {"configurable": {"thread_id": _thread_id(seed_prompt)}}
    accumulated = dict(state)

    for snapshot in get_app().stream(state, config=config):
        for node_name, node_state in snapshot.items():
            accumulated.update(node_state)
            yield node_name, dict(accumulated)


# ── Blocking interface (for CLI and pre-generation scripts) ───────────────

def run_pipeline(
    seed_prompt:    str,
    total_chapters: int = 6,
    max_revisions:  int = 2,
    mlflow_run_name: str | None = None,
) -> dict:
    """
    Run the full pipeline to completion and return the story output dict.
    Saves to data/stories/<story_id>/story.json automatically.
    """
    state    = initial_state(seed_prompt, total_chapters, max_revisions)
    config   = {"configurable": {"thread_id": _thread_id(seed_prompt)}}
    run_name = mlflow_run_name or f"v3_{seed_prompt[:30].replace(' ', '_')}"

    mlflow.end_run()
    with mlflow.start_run(run_name=run_name):
        print(f"\n🚀 Starting pipeline | Chapters: {total_chapters} | Max revisions: {max_revisions}")
        print(f"   Seed: {seed_prompt[:80]}...\n")

        final_state: StoryState = get_app().invoke(state, config=config)

        log_run_metrics(final_state)
        save_path = save_story(final_state)

        status = final_state.get("status", "unknown")
        error  = final_state.get("error")

        if error:
            print(f"\n⚠️  Pipeline finished with error: {error}")
        else:
            print(f"\n✅ Pipeline complete | Status: {status}")

        print(f"   Saved → {save_path}")

    return state_to_output(final_state)


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SciFi Forge V3 — pipeline runner")
    parser.add_argument("--prompt",    type=str, required=True)
    parser.add_argument("--chapters",  type=int, default=6)
    parser.add_argument("--revisions", type=int, default=2)
    args = parser.parse_args()

    result = run_pipeline(
        seed_prompt=args.prompt,
        total_chapters=args.chapters,
        max_revisions=args.revisions,
    )

    print(f"\n── Story: {result['title']} ──────────────────────────────────")
    print(f"   {result['logline']}")
    print(f"   Chapters: {len(result['chapters'])} | Error: {result['error']}")
