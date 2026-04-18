"""
v3_agentic/agents/editor.py
Editor agent — rewrites a chapter based on Critic feedback.

Pipeline position:  critic → editor → critic (re-score)

The Editor receives:
  - The original chapter content (chapters_written[-1]["content"])
  - revision_instructions from Critic
  - The story bible summary for grounding

It rewrites the chapter in full, replacing the last entry in
chapters_written so the Critic re-scores the improved version.
revision_count is incremented on both the global state counter and the
per-chapter record so evaluation metrics have the actual data.
"""

import torch

from pipeline.state   import StoryState
from agents.model_loader import load_agent_model
from agents.tools import get_story_bible_summary, update_chapter_in_memory
from agents.utils import trim_to_sentence


MAX_TOKENS  = 900


EDITOR_PROMPT = """You are a science fiction editor revising a chapter.

STORY BIBLE:
{story_bible_summary}

ORIGINAL CHAPTER:
{chapter_content}

REVISION INSTRUCTIONS:
{revision_instructions}

Rewrite the chapter in full, incorporating ALL revision instructions.
Maintain the author's narrative voice and prose style.
Keep approximately the same length (~600 words).
Write prose only — no chapter headings, no editor notes, no commentary."""


def _generate_revision(prompt: str, model, tokeniser) -> str:
    inputs = tokeniser(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.65,
            top_p=0.90,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokeniser.eos_token_id,
            eos_token_id=tokeniser.eos_token_id,
        )

    new_tokens = output_ids[0][prompt_len:]
    raw = tokeniser.decode(new_tokens, skip_special_tokens=True).strip()
    return trim_to_sentence(raw)


def editor_node(state: StoryState) -> dict:
    """
    LangGraph node: rewrite the latest chapter based on Critic feedback.

    Updates:
      - chapters_written[-1]["content"] replaced with revised version
      - chapters_written[-1]["critique_score"] reset to None (Critic will re-score)
      - chapters_written[-1]["revision_count"] incremented (persisted to story.json)
      - revision_count (global counter) incremented
      - status → "critiquing"
    """
    chapter_num           = state.get("current_chapter", 1)
    story_id              = state.get("story_id", "")
    revision_count        = state.get("revision_count", 0)
    revision_instructions = state.get("revision_instructions") or "Improve prose quality and consistency."
    chapters              = list(state.get("chapters_written", []))

    print(f"\n✏️   Editor: revising Chapter {chapter_num} (revision pass {revision_count + 1})")
    print(f"   Instructions: {revision_instructions[:120]}...")

    if not chapters:
        print("   ⚠️  No chapters to revise — skipping.")
        return {
            "revision_count": revision_count + 1,
            "status":         "critiquing",
        }

    original_content = chapters[-1].get("content", "")
    chapter_revision_count = chapters[-1].get("revision_count", 0)

    if not original_content or original_content.startswith("[Generation failed"):
        print("   ⚠️  Original chapter empty or failed — skipping revision.")
        return {
            "revision_count": revision_count + 1,
            "status":         "critiquing",
        }

    try:
        bible_summary   = get_story_bible_summary.invoke(story_id)
        chapter_excerpt = original_content[:1200]

        prompt = EDITOR_PROMPT.format(
            story_bible_summary=bible_summary,
            chapter_content=chapter_excerpt,
            revision_instructions=revision_instructions,
        )

        with load_agent_model() as (model, tokeniser):
            revised_content = _generate_revision(prompt, model, tokeniser)

        if not revised_content.strip():
            print("   ⚠️  Empty revision output — keeping original chapter.")
            revised_content = original_content

        print(f"   ✅ Revision complete | {len(revised_content.split())} words")

        # Replace last chapter record; persist the per-chapter revision count
        chapters[-1] = {
            **chapters[-1],
            "content":        revised_content,
            "critique_score": None,
            "revision_count": chapter_revision_count + 1,
        }

        update_chapter_in_memory.invoke({
            "story_id": story_id, "chapter_num": chapter_num, "content": revised_content
        })

        return {
            "chapters_written": chapters,
            "revision_count":   revision_count + 1,
            "status":           "critiquing",
            "error":            None,
        }

    except Exception as exc:
        error_msg = f"Editor failed (ch{chapter_num}): {type(exc).__name__}: {exc}"
        print(f"   ❌ {error_msg}")
        return {
            "revision_count": revision_count + 1,
            "status":         "critiquing",
            "error":          error_msg,
        }
