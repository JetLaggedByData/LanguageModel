"""
v3_agentic/agents/planner.py
Planner agent — generates the story bible from the seed prompt.

Uses the active agent model (default: Qwen2.5-1.5B + LoRA adapters) to produce
a structured JSON story bible, then saves it via StoryBible memory.

The story bible is the single source of truth for all subsequent agents:
  Writer  → reads acts/chapter briefs + get_summary()
  Critic  → reads get_summary() for consistency checking
  Editor  → reads get_summary() for revision grounding
"""

import json
import torch
from pathlib import Path

from pipeline.state  import StoryState, make_story_id
from memory.story_bible      import StoryBible
from memory.character_memory import CharacterMemory
from agents.model_loader import load_agent_model
from agents.utils import extract_json


_ROOT       = Path(__file__).resolve().parent.parent.parent
MAX_TOKENS  = 1_024


PLANNER_PROMPT = """You are a master science fiction story architect.
Given a seed prompt, create a detailed story bible in JSON format.

Return ONLY a JSON object with these exact keys:
{{
  "title": "Story title",
  "logline": "One sentence story summary",
  "acts": [
    {{"act": 1, "summary": "Act summary", "chapters": [1, 2]}},
    {{"act": 2, "summary": "Act summary", "chapters": [3, 4]}},
    {{"act": 3, "summary": "Act summary", "chapters": [5, 6]}}
  ],
  "characters": [
    {{
      "name": "Character name",
      "role": "protagonist|antagonist|supporting",
      "traits": ["trait1", "trait2"],
      "voice_style": "How they speak, e.g. terse and technical"
    }}
  ],
  "world_rules": [
    "Rule about physics or society",
    "Rule about technology limits"
  ],
  "technology": [
    "Ship or weapon name",
    "Named technology"
  ]
}}

Seed prompt: {seed_prompt}

Respond ONLY with valid JSON. No preamble, no explanation, no markdown fences."""


def _generate_raw(prompt: str) -> str:
    with load_agent_model() as (model, tokeniser):
        inputs = tokeniser(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=0.4,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokeniser.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokeniser.decode(new_tokens, skip_special_tokens=True).strip()


def _parse_bible_json(raw: str, seed_prompt: str, total_chapters: int) -> dict:
    """
    Parse JSON with graceful fallback.
    If the model output is malformed, return a minimal valid bible so
    the pipeline can continue rather than hard-crashing.
    """
    json_str = extract_json(raw)
    try:
        bible = json.loads(json_str)
    except json.JSONDecodeError:
        bible = {
            "title":       f"Story: {seed_prompt[:50]}",
            "logline":     seed_prompt[:120],
            "acts":        [
                {"act": i, "summary": f"Act {i}", "chapters": []}
                for i in range(1, 4)
            ],
            "characters":  [{"name": "Protagonist", "role": "protagonist",
                              "traits": [], "voice_style": ""}],
            "world_rules": ["Standard science fiction setting"],
            "technology":  ["Spacecraft", "Advanced computers"],
        }

    bible.setdefault("title",      f"Story: {seed_prompt[:50]}")
    bible.setdefault("logline",    seed_prompt[:120])
    bible.setdefault("acts",       [])
    bible.setdefault("characters", [])
    bible.setdefault("world_rules",[])
    bible.setdefault("technology", [])

    chapters_per_act = max(1, total_chapters // max(len(bible["acts"]), 1))
    for i, act in enumerate(bible["acts"]):
        if not act.get("chapters"):
            start = i * chapters_per_act + 1
            act["chapters"] = list(range(start, start + chapters_per_act))

    return bible


def planner_node(state: StoryState) -> dict:
    """
    LangGraph node: generate story bible and save to disk.

    Returns partial state dict with all Story Bible fields populated.
    On any failure, sets state["error"] and returns minimal defaults
    so the pipeline degrades gracefully rather than crashing.
    """
    seed_prompt    = state["seed_prompt"]
    total_chapters = state.get("total_chapters", 6)

    print(f"\n🗺️  Planner: generating story bible for '{seed_prompt[:60]}...'")

    try:
        prompt  = PLANNER_PROMPT.format(seed_prompt=seed_prompt)
        raw     = _generate_raw(prompt)
        bible   = _parse_bible_json(raw, seed_prompt, total_chapters)

        story_id = make_story_id(bible["title"])
        bible["story_id"] = story_id

        StoryBible(story_id).save(bible)

        try:
            CharacterMemory(story_id).build_from_bible(bible.get("characters", []))
        except Exception as emb_exc:
            print(f"   ⚠️  CharacterMemory build failed (non-fatal): {emb_exc}")

        print(f"   ✅ Story bible saved | Title: {bible['title']} | story_id: {story_id}")

        return {
            "story_id":        story_id,
            "title":           bible["title"],
            "logline":         bible["logline"],
            "acts":            bible["acts"],
            "characters":      bible["characters"],
            "world_rules":     bible["world_rules"],
            "technology":      bible["technology"],
            "current_chapter": 1,
            "current_act":     1,
            "status":          "writing",
            "error":           None,
        }

    except Exception as exc:
        error_msg = f"Planner failed: {type(exc).__name__}: {exc}"
        print(f"   ❌ {error_msg}")
        fallback_id = make_story_id(seed_prompt[:40])
        fallback_bible = {
            "story_id":   fallback_id,
            "title":      seed_prompt[:60],
            "logline":    seed_prompt[:120],
            "acts":       [{"act": i, "summary": f"Act {i}", "chapters": list(range((i-1)*2+1, i*2+1))} for i in range(1, 4)],
            "characters": [],
            "world_rules": [],
            "technology": [],
        }
        try:
            StoryBible(fallback_id).save(fallback_bible)
        except Exception:
            pass
        return {
            "story_id":        fallback_id,
            "title":           fallback_bible["title"],
            "logline":         fallback_bible["logline"],
            "acts":            fallback_bible["acts"],
            "characters":      [],
            "world_rules":     [],
            "technology":      [],
            "current_chapter": 1,
            "current_act":     1,
            "status":          "writing",
            "error":           error_msg,
        }
