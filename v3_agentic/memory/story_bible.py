"""
v3_agentic/memory/story_bible.py
Persistent JSON storage for the story bible produced by the Planner agent.

One StoryBible instance per story run, keyed by story_id.
All agents call get_summary() to get a prompt-injectable condensed version
that stays within ~500 tokens to avoid overflowing the context window.
"""

import json
from pathlib import Path


STORIES_DIR = Path(__file__).resolve().parent.parent.parent / "data/stories"

# Approximate token budget for summary injected into agent prompts.
# Using 4 chars/token estimate → 500 tokens ≈ 2000 chars.
SUMMARY_MAX_CHARS = 2_000


class StoryBible:
    """
    Reads and writes the story bible for a single story.

    File layout:
        data/stories/<story_id>/bible.json

    Expected bible dict schema (produced by Planner):
        {
            "title":       str,
            "logline":     str,
            "acts":        [{act, summary, chapters: [...]}, ...],
            "characters":  [{name, role, traits, voice_style}, ...],
            "world_rules": [str, ...],
            "technology":  [str, ...]
        }
    """

    def __init__(self, story_id: str) -> None:
        self.story_id = story_id
        self.path     = STORIES_DIR / story_id / "bible.json"
        self._cache: dict | None = None

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, bible: dict) -> None:
        """Write bible dict to disk and update in-memory cache."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(bible, indent=2, ensure_ascii=False))
        self._cache = bible

    def load(self) -> dict:
        """Load bible from disk. Uses cache after first load."""
        if self._cache is not None:
            return self._cache
        if not self.path.exists():
            raise FileNotFoundError(
                f"Story bible not found: {self.path}\n"
                "Planner must run before any other agent."
            )
        self._cache = json.loads(self.path.read_text(encoding="utf-8"))
        return self._cache

    def exists(self) -> bool:
        return self.path.exists()

    # ── Summary for prompt injection ──────────────────────────────────────

    def get_summary(self, max_chars: int = SUMMARY_MAX_CHARS) -> str:
        """
        Return a condensed story bible string for injection into agent prompts.
        Truncates to max_chars to stay within context window budget.

        Format:
            TITLE: ...
            LOGLINE: ...
            CHARACTERS: name (role) — traits | voice
            WORLD RULES: rule1; rule2
            TECHNOLOGY: item1, item2
            ACTS: Act 1: summary | Act 2: summary ...
        """
        bible = self.load()

        # Characters — one line each
        char_lines = []
        for c in bible.get("characters", [])[:5]:   # cap at 5 characters
            traits = ", ".join(c.get("traits", []))
            voice  = c.get("voice_style", "")
            char_lines.append(
                f"  {c.get('name','?')} ({c.get('role','?')}) — {traits} | voice: {voice}"
            )

        # Acts — one line each
        act_lines = []
        for a in bible.get("acts", []):
            act_lines.append(f"  Act {a.get('act','?')}: {a.get('summary','')}")

        # World rules + tech — semicolon joined
        world = "; ".join(bible.get("world_rules", [])[:6])
        tech  = ", ".join(bible.get("technology",  [])[:8])

        summary = "\n".join([
            f"TITLE: {bible.get('title', '')}",
            f"LOGLINE: {bible.get('logline', '')}",
            "CHARACTERS:",
            *char_lines,
            f"WORLD RULES: {world}",
            f"TECHNOLOGY: {tech}",
            "ACTS:",
            *act_lines,
        ])

        # Hard truncate with ellipsis if over budget
        if len(summary) > max_chars:
            summary = summary[:max_chars - 3] + "..."

        return summary

    def get_character_names(self) -> list[str]:
        """Convenience helper used by Critic for consistency checking."""
        bible = self.load()
        return [c.get("name", "") for c in bible.get("characters", [])]

    def get_world_rules(self) -> list[str]:
        """Convenience helper used by Critic."""
        return self.load().get("world_rules", [])
