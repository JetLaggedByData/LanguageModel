"""
v3_agentic/agents/writer.py
Writer agent — generates each chapter of the story.

Pipeline position:  planner → [writer → critic → (editor →) critic] × N chapters

For each chapter the Writer:
  1. Extracts the chapter brief from the story bible acts
  2. Retrieves semantically relevant past chapter excerpts via FAISS tool
  3. Summarises the immediately previous chapter (last 300 chars as proxy)
  4. Builds the full prompt and generates ~600 words of prose
  5. Stores the new chapter in FAISS via memory tool
  6. Appends the chapter dict to state["chapters_written"]
"""

import torch
from pipeline.state    import StoryState
from agents.model_loader import load_agent_model
from agents.tools import (
    get_story_bible_summary,
    get_relevant_chapter_context,
    get_character_voice_reminders,
    store_chapter_in_memory,
)
from agents.utils import trim_to_sentence


MAX_TOKENS  = 900


WRITER_PROMPT = """You are a science fiction author writing Chapter {chapter_num}.

STORY BIBLE:
{story_bible_summary}

PREVIOUS CHAPTER SUMMARY:
{prev_chapter_summary}

RELEVANT PAST CONTEXT:
{faiss_context}

CHARACTER VOICES:
{character_voices}

CHAPTER BRIEF:
{chapter_brief}

Write a compelling, immersive chapter of approximately 600 words.
Stay true to each character's voice style and all world rules.
Close the chapter at a deliberate narrative moment — a resolved beat, a quiet reflection, or an intentional cliffhanger.
Never end mid-action or in a way that reads as an abrupt cut.
Write prose only — no chapter headings, no author notes."""


def _get_chapter_brief(state: StoryState) -> str:
    """
    Extract the brief for the current chapter from the story bible acts.
    Falls back to act summary if no per-chapter brief exists.
    """
    chapter_num = state.get("current_chapter", 1)
    acts        = state.get("acts", [])

    for act in acts:
        chapters_in_act = act.get("chapters", [])
        if chapter_num in chapters_in_act:
            idx     = chapters_in_act.index(chapter_num)
            total   = len(chapters_in_act)
            section = "opening" if idx == 0 else ("closing" if idx == total - 1 else "middle")
            return (
                f"Act {act.get('act', '?')} — {section} chapter.\n"
                f"Act theme: {act.get('summary', 'Continue the story.')}"
            )

    return f"Continue the story naturally into chapter {chapter_num}."


def _last_paragraph(text: str, fallback_chars: int = 300) -> str:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras[-1] if paras else text[-fallback_chars:].strip()


def _get_prev_chapter_summary(state: StoryState) -> str:
    chapters_written = state.get("chapters_written", [])
    if not chapters_written:
        return "This is the opening chapter."

    prev = chapters_written[-1]
    content = prev.get("content", "")
    if not content:
        return "The previous chapter ended without recorded content."

    closing = _last_paragraph(content)
    return f"The previous chapter concluded with:\n\"{closing}\""


def _generate_chapter(prompt: str, model, tokeniser) -> str:
    inputs = tokeniser(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.75,
            top_p=0.92,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokeniser.eos_token_id,
            eos_token_id=tokeniser.eos_token_id,
        )

    new_tokens = output_ids[0][prompt_len:]
    raw = tokeniser.decode(new_tokens, skip_special_tokens=True).strip()
    return trim_to_sentence(raw)


def _infer_chapter_title(content: str, chapter_num: int) -> str:
    if not content:
        return f"Chapter {chapter_num}"
    first_sentence = content.split(".")[0].strip()
    title = first_sentence[:50].strip()
    return title if title else f"Chapter {chapter_num}"


def writer_node(state: StoryState) -> dict:
    """
    LangGraph node: generate one chapter and store in FAISS + state.

    Returns partial state dict updating chapters_written and status.
    On failure, appends a chapter with error content so the pipeline
    can continue rather than blocking at this node indefinitely.
    """
    chapter_num = state.get("current_chapter", 1)
    story_id    = state.get("story_id", "")

    print(f"\n✍️  Writer: generating Chapter {chapter_num} (story: {story_id})")

    try:
        # Retrieve all context via @tool interface before loading the model
        bible_summary    = get_story_bible_summary.invoke(story_id)
        chapter_brief    = _get_chapter_brief(state)
        prev_summary     = _get_prev_chapter_summary(state)
        faiss_context    = get_relevant_chapter_context.invoke({
            "story_id": story_id, "query": chapter_brief, "k": 3
        })
        character_voices = get_character_voice_reminders.invoke(story_id)

        prompt = WRITER_PROMPT.format(
            chapter_num=chapter_num,
            story_bible_summary=bible_summary,
            prev_chapter_summary=prev_summary,
            faiss_context=faiss_context,
            character_voices=character_voices,
            chapter_brief=chapter_brief,
        )

        with load_agent_model() as (model, tokeniser):
            content = _generate_chapter(prompt, model, tokeniser)

        title = _infer_chapter_title(content, chapter_num)
        print(f"   ✅ Chapter {chapter_num} written | {len(content.split())} words | '{title}'")

        store_chapter_in_memory.invoke({
            "story_id": story_id, "chapter_num": chapter_num, "content": content
        })

        chapter_record = {
            "num":            chapter_num,
            "title":          title,
            "content":        content,
            "critique_score": None,
            "revision_count": 0,
        }

        existing = list(state.get("chapters_written", []))
        existing.append(chapter_record)

        return {
            "chapters_written": existing,
            "status":           "critiquing",
            "error":            None,
        }

    except Exception as exc:
        error_msg = f"Writer failed (ch{chapter_num}): {type(exc).__name__}: {exc}"
        print(f"   ❌ {error_msg}")

        fallback_record = {
            "num":            chapter_num,
            "title":          f"Chapter {chapter_num}",
            "content":        f"[Generation failed: {exc}]",
            "critique_score": 0.0,
            "revision_count": 0,
        }
        existing = list(state.get("chapters_written", []))
        existing.append(fallback_record)

        return {
            "chapters_written": existing,
            "status":           "critiquing",
            "error":            error_msg,
        }
