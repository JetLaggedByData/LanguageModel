"""
v3_agentic/agents/critic.py
Critic agent — scores each chapter on three dimensions and decides
whether a revision pass is needed.

Pipeline position:  writer → critic → (editor → critic) → advance_chapter

Scores three dimensions (0–1 each):
  consistency_score  — characters, world rules, named technology
  style_score        — genuine SciFi prose quality
  coherence_score    — narrative flow and logical continuity

revision_needed = True only if ANY score < 0.6 (skill spec threshold).
All scores are logged to MLflow per chapter for the benchmark report.
"""

import json
import mlflow
import torch

from pipeline.state   import StoryState
from memory.character_memory import CharacterMemory
from agents.model_loader import load_agent_model
from agents.tools import get_story_bible_summary
from agents.utils import extract_json


MAX_TOKENS  = 512
REVISION_THRESHOLD = 0.6


CRITIC_PROMPT = """You are a science fiction editor. Review this chapter and respond in JSON.

STORY BIBLE:
{story_bible_summary}

CHAPTER:
{chapter_content}

Score the chapter on these criteria and return ONLY a JSON object:
{{
  "consistency_score": <float 0.0-1.0>,
  "style_score": <float 0.0-1.0>,
  "coherence_score": <float 0.0-1.0>,
  "revision_needed": <true or false>,
  "critique": "<one sentence overall assessment>",
  "revision_instructions": "<specific list of what to fix, or empty string if no revision needed>"
}}

Scoring guide:
- consistency_score: Do characters behave consistently? Are world rules respected? Is named technology used correctly?
- style_score: Is this genuine SciFi prose — vivid, purposeful, not generic? Does dialogue fit character voice styles?
- coherence_score: Does the narrative flow logically from the previous chapter? Is cause and effect clear?
- revision_needed: Set true ONLY if any single score is below {threshold}.

Respond ONLY with valid JSON. No preamble, no markdown fences."""


def _generate_critique(prompt: str, model, tokeniser) -> str:
    inputs = tokeniser(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokeniser.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokeniser.decode(new_tokens, skip_special_tokens=True).strip()


def _parse_critique(raw: str) -> dict:
    """
    Parse critique JSON with graceful fallback.
    If parsing fails, default to scores that trigger revision so parse failures
    don't silently pass weak chapters through.
    """
    json_str = extract_json(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return {
            "consistency_score":     0.5,
            "style_score":           0.5,
            "coherence_score":       0.5,
            "revision_needed":       True,
            "critique":              "Could not parse critique — defaulting to revision.",
            "revision_instructions": "Improve prose quality, character consistency, and narrative flow.",
        }

    for key in ("consistency_score", "style_score", "coherence_score"):
        val = data.get(key, 0.5)
        data[key] = max(0.0, min(1.0, float(val)))

    # Recompute revision_needed from actual scores rather than trusting model output
    scores = [data["consistency_score"], data["style_score"], data["coherence_score"]]
    data["revision_needed"] = any(s < REVISION_THRESHOLD for s in scores)

    data.setdefault("critique", "")
    data.setdefault("revision_instructions", "")

    return data


def _blend_voice_signal(critique: dict, story_id: str, chapter_content: str) -> dict:
    """
    Blend embedding-based voice consistency into the LLM's consistency_score.

    CharacterMemory.check_voice_consistency() returns cosine similarities between
    the chapter text and each character's embedded voice profile. These sit in
    roughly [0.2, 0.65] for all-MiniLM-L6-v2 normalised vectors. We normalise
    to [0, 1] and blend at 25% weight so the score is grounded in something
    independent of the writer model's own tendencies.
    """
    if not story_id:
        return critique
    try:
        char_mem = CharacterMemory(story_id)
        if not char_mem.is_built():
            return critique

        voice_scores = char_mem.check_voice_consistency(chapter_content[:600], top_k=3)
        if not voice_scores:
            return critique

        raw_max = max(voice_scores.values())
        voice_signal = max(0.0, min(1.0, (raw_max - 0.20) / 0.45))

        blended = round(0.75 * critique["consistency_score"] + 0.25 * voice_signal, 4)
        critique = {**critique, "consistency_score": blended}

        scores = [critique["consistency_score"], critique["style_score"], critique["coherence_score"]]
        critique["revision_needed"] = any(s < REVISION_THRESHOLD for s in scores)

    except Exception:
        pass

    return critique


def _composite_score(critique: dict) -> float:
    """Weighted average: consistency 40%, style 35%, coherence 25%."""
    return round(
        critique["consistency_score"] * 0.40
        + critique["style_score"]     * 0.35
        + critique["coherence_score"] * 0.25,
        4,
    )


def _log_to_mlflow(critique: dict, chapter_num: int) -> None:
    try:
        mlflow.log_metrics({
            f"ch{chapter_num}_consistency":  critique["consistency_score"],
            f"ch{chapter_num}_style":        critique["style_score"],
            f"ch{chapter_num}_coherence":    critique["coherence_score"],
            f"ch{chapter_num}_composite":    _composite_score(critique),
        }, step=chapter_num)
    except Exception:
        pass


def critic_node(state: StoryState) -> dict:
    """
    LangGraph node: score the latest chapter and decide on revision.

    Updates:
      - consistency_score, style_score, coherence_score
      - critique, revision_needed, revision_instructions
      - chapters_written[-1]["critique_score"] with composite score
    """
    chapter_num = state.get("current_chapter", 1)
    story_id    = state.get("story_id", "")
    chapters    = state.get("chapters_written", [])

    print(f"\n🔍  Critic: scoring Chapter {chapter_num} (revision #{state.get('revision_count', 0)})")

    current_chapter_content = ""
    if chapters:
        current_chapter_content = chapters[-1].get("content", "")

    if not current_chapter_content or current_chapter_content.startswith("[Generation failed"):
        print("   ⚠️  Empty or failed chapter — skipping critique, marking as pass.")
        return {
            "consistency_score":     None,
            "style_score":           None,
            "coherence_score":       None,
            "critique":              "Chapter generation failed — skipped critique.",
            "revision_needed":       False,
            "revision_instructions": None,
            "status":                "writing",
        }

    try:
        bible_summary   = get_story_bible_summary.invoke(story_id)
        chapter_excerpt = current_chapter_content[:1500]

        prompt = CRITIC_PROMPT.format(
            story_bible_summary=bible_summary,
            chapter_content=chapter_excerpt,
            threshold=REVISION_THRESHOLD,
        )

        with load_agent_model() as (model, tokeniser):
            raw = _generate_critique(prompt, model, tokeniser)

        critique  = _parse_critique(raw)
        critique  = _blend_voice_signal(critique, story_id, current_chapter_content)
        composite = _composite_score(critique)

        _log_to_mlflow(critique, chapter_num)

        updated_chapters = list(chapters)
        if updated_chapters:
            updated_chapters[-1] = {**updated_chapters[-1], "critique_score": composite}

        revision = critique["revision_needed"]
        print(
            f"   {'🔄 Revision needed' if revision else '✅ Chapter accepted'} | "
            f"consistency={critique['consistency_score']:.2f} "
            f"style={critique['style_score']:.2f} "
            f"coherence={critique['coherence_score']:.2f} "
            f"composite={composite:.2f}"
        )

        return {
            "consistency_score":     critique["consistency_score"],
            "style_score":           critique["style_score"],
            "coherence_score":       critique["coherence_score"],
            "critique":              critique["critique"],
            "revision_needed":       revision,
            "revision_instructions": critique["revision_instructions"],
            "chapters_written":      updated_chapters,
            "status":                "editing" if revision else "writing",
            "error":                 None,
        }

    except Exception as exc:
        error_msg = f"Critic failed (ch{chapter_num}): {type(exc).__name__}: {exc}"
        print(f"   ❌ {error_msg}")
        return {
            "consistency_score":     None,
            "style_score":           None,
            "coherence_score":       None,
            "critique":              f"Critique failed: {exc}",
            "revision_needed":       False,
            "revision_instructions": None,
            "status":                "writing",
            "error":                 error_msg,
        }
