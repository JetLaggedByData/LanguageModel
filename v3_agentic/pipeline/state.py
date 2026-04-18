"""
v3_agentic/pipeline/state.py
Shared state schema for the LangGraph multi-agent pipeline.

StoryState is passed between every node in the graph.
Each agent receives the full state and returns a partial dict
with only the keys it modified — LangGraph merges these automatically.

Key design decisions:
  - coherence_score stored separately from consistency/style so the
    Critic can score three dimensions independently
  - revision_count tracked globally (not per-chapter) so the
    should_revise edge condition stays simple
  - story_id is a slug derived from the title, used as the
    filesystem key for StoryBible and ChapterMemory
"""

from typing import TypedDict, Optional
from datetime import datetime


class StoryState(TypedDict):
    # ── User input ────────────────────────────────────────────────────────
    seed_prompt:        str

    # ── Story Bible (set by Planner, read-only for all other agents) ──────
    story_id:           str             # slug: "the-last-colony-2024-..."
    title:              str
    logline:            str
    acts:               list[dict]      # [{act: 1, summary: "...", chapters: [...]}]
    characters:         list[dict]      # [{name, role, traits, voice_style}]
    world_rules:        list[str]       # hard sci-fi constraints
    technology:         list[str]       # named tech/ships/weapons

    # ── Generation progress ───────────────────────────────────────────────
    current_chapter:    int             # 1-indexed
    current_act:        int             # 1-indexed
    chapters_written:   list[dict]      # [{num, title, content, critique_score}]

    # ── Critic feedback (reset each chapter, accumulated across revisions) ─
    critique:                   Optional[str]
    consistency_score:          Optional[float]
    style_score:                Optional[float]
    coherence_score:            Optional[float]
    revision_needed:            bool
    revision_instructions:      Optional[str]

    # ── Pipeline control ──────────────────────────────────────────────────
    total_chapters:     int
    max_revisions:      int
    revision_count:     int
    status:             str     # "planning"|"writing"|"critiquing"|"editing"|"done"
    error:              Optional[str]   # set by any agent on failure


def initial_state(
    seed_prompt:    str,
    total_chapters: int = 6,
    max_revisions:  int = 2,
) -> StoryState:
    """
    Return a fully initialised StoryState for a new story run.
    All optional fields set to None; lists empty; counters at zero.

    Args:
        seed_prompt:    Raw user text, e.g. "A dying Earth colony ship..."
        total_chapters: How many chapters to generate (1–10)
        max_revisions:  Max Editor passes per chapter before moving on
    """
    if not seed_prompt.strip():
        raise ValueError("seed_prompt must not be empty")
    if not (1 <= total_chapters <= 10):
        raise ValueError("total_chapters must be between 1 and 10")
    if not (0 <= max_revisions <= 3):
        raise ValueError("max_revisions must be between 0 and 3")

    return StoryState(
        seed_prompt=seed_prompt.strip(),

        # Story Bible — filled by Planner
        story_id="",
        title="",
        logline="",
        acts=[],
        characters=[],
        world_rules=[],
        technology=[],

        # Generation progress
        current_chapter=0,
        current_act=1,
        chapters_written=[],

        # Critic feedback
        critique=None,
        consistency_score=None,
        style_score=None,
        coherence_score=None,
        revision_needed=False,
        revision_instructions=None,

        # Control
        total_chapters=total_chapters,
        max_revisions=max_revisions,
        revision_count=0,
        status="planning",
        error=None,
    )


def make_story_id(title: str) -> str:
    """
    Generate a filesystem-safe story ID from the title.
    Example: "The Last Colony" → "the-last-colony-20241201"
    """
    slug = title.lower().strip()
    slug = "".join(c if c.isalnum() or c == " " else "" for c in slug)
    slug = "-".join(slug.split())[:40]
    date = datetime.now().strftime("%Y%m%d")
    return f"{slug}-{date}" if slug else f"story-{date}"
