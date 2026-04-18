"""
v3_agentic/agents/_stubs.py
Temporary stub implementations for all four agents.
These make graph.py importable and the pipeline runnable for skeleton testing
BEFORE the real agents are built in Steps 5–8.

Each stub advances the pipeline state minimally so you can run:
  python v3_agentic/pipeline/runner.py --prompt "test" --chapters 2

and see the full graph execute end-to-end.

DELETE this file and replace imports in each agent module once real agents are built.
"""

from v3_agentic.pipeline.state import StoryState, make_story_id


def planner_node(state: StoryState) -> dict:
    """STUB: Planner — replaced in Step 5 (agents/planner.py)."""
    title    = f"Story: {state['seed_prompt'][:40]}"
    story_id = make_story_id(title)
    return {
        "story_id":    story_id,
        "title":       title,
        "logline":     "A gripping science fiction adventure.",
        "acts":        [
            {"act": 1, "summary": "Setup",      "chapters": [1, 2]},
            {"act": 2, "summary": "Confrontation", "chapters": [3, 4]},
            {"act": 3, "summary": "Resolution", "chapters": [5, 6]},
        ],
        "characters":  [{"name": "Protagonist", "role": "hero", "traits": [], "voice_style": ""}],
        "world_rules": ["Faster-than-light travel exists"],
        "technology":  ["Hyperdrive", "Neural interface"],
        "current_chapter": 1,
        "status":      "writing",
    }


def writer_node(state: StoryState) -> dict:
    """STUB: Writer — replaced in Step 6 (agents/writer.py)."""
    chapter_num = state.get("current_chapter", 1)
    chapter = {
        "num":           chapter_num,
        "title":         f"Chapter {chapter_num}",
        "content":       f"[STUB] Chapter {chapter_num} content goes here.",
        "critique_score": None,
    }
    existing = list(state.get("chapters_written", []))
    existing.append(chapter)
    return {
        "chapters_written": existing,
        "status":           "critiquing",
    }


def critic_node(state: StoryState) -> dict:
    """STUB: Critic — replaced in Step 7 (agents/critic.py)."""
    return {
        "consistency_score":     0.8,
        "style_score":           0.8,
        "coherence_score":       0.8,
        "critique":              "STUB: Looks good.",
        "revision_needed":       False,
        "revision_instructions": None,
        "status":                "writing",
    }


def editor_node(state: StoryState) -> dict:
    """STUB: Editor — replaced in Step 8 (agents/editor.py)."""
    return {
        "revision_count": state.get("revision_count", 0) + 1,
        "status":         "critiquing",
    }
