"""
v3_agentic/memory/character_memory.py
Character voice embedding store.

Embeds each character's canonical voice profile (traits + voice_style)
from the story bible so agents can perform semantic similarity checks
against generated chapter text — catching voice drift before the Critic scores.

Primary consumers:
  Writer   → get_voice_reminder() injects a targeted voice prompt for a
             named character before generating dialogue-heavy chapters
  Critic   → check_voice_consistency() returns a per-character similarity
             score that feeds into consistency_score calculation

Embeddings run on CPU (all-MiniLM-L6-v2) — no VRAM cost.
One CharacterMemory instance per story run, keyed by story_id.
"""

import json
from pathlib import Path
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings


STORIES_DIR = Path(__file__).resolve().parent.parent.parent / "data/stories"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# Minimum cosine similarity to consider a passage "in character"
VOICE_SIMILARITY_THRESHOLD = 0.45


class CharacterMemory:
    """
    Stores and retrieves character voice embeddings for a single story.

    Usage:
        memory = CharacterMemory(story_id="the-last-colony-20241201")
        memory.build_from_bible(characters)          # called by Planner
        reminder = memory.get_voice_reminder("Zara") # called by Writer
        scores   = memory.check_voice_consistency(   # called by Critic
            chapter_content, top_k=3
        )
    """

    def __init__(self, story_id: str) -> None:
        self.story_id   = story_id
        self.index_path = STORIES_DIR / story_id / "character_voices.json"
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        # {name: {"profile_text": str, "embedding": list[float], "traits": list, "voice_style": str}}
        self._store: dict[str, dict] = {}
        self._load_if_exists()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _profile_text(self, character: dict) -> str:
        """
        Build a canonical text description of a character for embedding.
        Combines role, traits, and voice_style into a single sentence.
        More descriptive = better embedding quality.
        """
        name   = character.get("name", "Unknown")
        role   = character.get("role", "character")
        traits = ", ".join(character.get("traits", []))
        voice  = character.get("voice_style", "")
        parts  = [f"{name} is a {role}."]
        if traits:
            parts.append(f"Personality traits: {traits}.")
        if voice:
            parts.append(f"Voice and speech style: {voice}.")
        return " ".join(parts)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two normalised embedding vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        # Vectors are already L2-normalised by encode_kwargs, so dot = cosine
        return round(float(dot), 4)

    def _save(self) -> None:
        """Persist character store to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        # Don't serialise embedding arrays in the JSON — recompute on load
        serialisable = {
            name: {
                "profile_text": data["profile_text"],
                "traits":       data["traits"],
                "voice_style":  data["voice_style"],
            }
            for name, data in self._store.items()
        }
        self.index_path.write_text(json.dumps(serialisable, indent=2))

    def _load_if_exists(self) -> None:
        """Load character profiles from disk and recompute embeddings."""
        if not self.index_path.exists():
            return
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            for name, data in raw.items():
                profile_text = data.get("profile_text", "")
                embedding    = self._embeddings.embed_query(profile_text)
                self._store[name] = {
                    "profile_text": profile_text,
                    "embedding":    embedding,
                    "traits":       data.get("traits", []),
                    "voice_style":  data.get("voice_style", ""),
                }
        except Exception:
            pass   # corrupt file — start fresh

    # ── Public interface ──────────────────────────────────────────────────

    def build_from_bible(self, characters: list[dict]) -> None:
        """
        Embed all characters from the Planner's story bible.
        Called once by the Planner after saving the bible JSON.

        Args:
            characters: list of character dicts from StoryState["characters"]
        """
        if not characters:
            return

        for char in characters:
            name = char.get("name", "").strip()
            if not name:
                continue

            profile_text = self._profile_text(char)
            embedding    = self._embeddings.embed_query(profile_text)

            self._store[name] = {
                "profile_text": profile_text,
                "embedding":    embedding,
                "traits":       char.get("traits", []),
                "voice_style":  char.get("voice_style", ""),
            }

        self._save()
        print(f"   CharacterMemory: embedded {len(self._store)} characters")

    def get_voice_reminder(self, character_name: str) -> str:
        """
        Return a concise voice reminder string for injection into the Writer prompt.
        Used to prevent the model from flattening all characters into the same voice.

        Args:
            character_name: Must match a name in the story bible exactly.

        Returns:
            One-sentence reminder, e.g. "Zara speaks tersely and technically,
            with clipped sentences and no small talk."
            Empty string if character not found.
        """
        data = self._store.get(character_name)
        if not data:
            # Try case-insensitive match
            lower_map = {k.lower(): k for k in self._store}
            canonical = lower_map.get(character_name.lower())
            if canonical:
                data = self._store[canonical]
            else:
                return ""

        voice  = data.get("voice_style", "")
        traits = ", ".join(data.get("traits", []))
        parts  = []
        if voice:
            parts.append(f"{character_name} speaks: {voice}.")
        if traits:
            parts.append(f"Key traits: {traits}.")
        return " ".join(parts)

    def get_all_voice_reminders(self) -> str:
        """
        Return voice reminders for ALL characters in one string.
        Used by Writer when multiple characters appear in a chapter.
        """
        if not self._store:
            return ""
        lines = []
        for name in self._store:
            reminder = self.get_voice_reminder(name)
            if reminder:
                lines.append(f"- {reminder}")
        return "\n".join(lines)

    def check_voice_consistency(
        self,
        chapter_content: str,
        top_k: int = 3,
    ) -> dict[str, float]:
        """
        Compute cosine similarity between chapter text and each character's
        voice profile embedding. Returns per-character similarity scores.

        Used by the Critic as a signal for the consistency_score dimension.
        Higher score = chapter text is more semantically aligned with the
        character's established voice and traits.

        Args:
            chapter_content: Full chapter text to check.
            top_k:           Only return the top_k most relevant characters.

        Returns:
            Dict of {character_name: similarity_score}, sorted descending.
            Returns empty dict if no characters embedded.
        """
        if not self._store or not chapter_content.strip():
            return {}

        # Embed the chapter (truncate to first 600 chars for speed)
        chapter_embedding = self._embeddings.embed_query(chapter_content[:600])

        scores = {}
        for name, data in self._store.items():
            sim = self._cosine_similarity(chapter_embedding, data["embedding"])
            scores[name] = sim

        # Sort descending, return top_k
        sorted_scores = dict(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )
        return sorted_scores

    def character_names(self) -> list[str]:
        """Return list of all embedded character names."""
        return list(self._store.keys())

    def is_built(self) -> bool:
        """Return True if at least one character has been embedded."""
        return len(self._store) > 0
