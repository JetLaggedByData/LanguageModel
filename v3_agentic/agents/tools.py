"""
v3_agentic/agents/tools.py
LangChain @tool definitions for all memory operations used by the agents.

Wrapping memory calls as @tool-decorated functions gives each operation a
well-defined schema (name, description, args) that any LLM-driven agent can
invoke via tool-use — the same interface used by ReAct loops, ToolNode, and
LangGraph's built-in tool execution. Agents in this pipeline call these tools
directly (not through an LLM's tool-selection step), but the abstraction makes
every memory access traceable, reusable, and schema-validated.
"""

from langchain_core.tools import tool

from memory.story_bible      import StoryBible
from memory.chapter_store    import ChapterMemory
from memory.character_memory import CharacterMemory


@tool
def get_story_bible_summary(story_id: str) -> str:
    """
    Retrieve the condensed story bible summary for a given story.
    Returns a prompt-injectable string with title, logline, characters,
    world rules, technology, and act summaries (~500 token budget).
    Returns empty string if the bible has not been saved yet.
    """
    if not story_id:
        return ""
    try:
        return StoryBible(story_id).get_summary()
    except FileNotFoundError:
        return ""
    except Exception as exc:
        return f"[Bible unavailable: {exc}]"


@tool
def get_relevant_chapter_context(story_id: str, query: str, k: int = 3) -> str:
    """
    Search FAISS for the k most semantically relevant past chapter excerpts.
    Used by the Writer to retrieve prior narrative context before generating
    the next chapter. Returns formatted excerpts or a no-context message.
    """
    if not story_id:
        return "No previous chapters yet."
    try:
        memory = ChapterMemory(story_id)
        context = memory.get_relevant_context(query=query, k=k)
        return context if context else "No relevant prior context retrieved."
    except Exception:
        return "No relevant prior context retrieved."


@tool
def get_character_voice_reminders(story_id: str) -> str:
    """
    Retrieve voice reminders for all characters in a story.
    Returns a bullet-list string injected into the Writer prompt to prevent
    character voice drift across chapters. Returns empty string if no
    character embeddings have been built yet.
    """
    if not story_id:
        return "No character voice profiles available."
    try:
        mem = CharacterMemory(story_id)
        if not mem.is_built():
            return "No character voice profiles available."
        return mem.get_all_voice_reminders() or "No character voice profiles available."
    except Exception:
        return "No character voice profiles available."


@tool
def store_chapter_in_memory(story_id: str, chapter_num: int, content: str) -> str:
    """
    Embed and store a completed chapter in the FAISS index for future retrieval.
    Called by the Writer after each chapter is generated. Returns a status string.
    """
    if not story_id or not content.strip():
        return "Skipped: no story_id or empty content."
    try:
        ChapterMemory(story_id).add_chapter(chapter_num, content)
        return f"Chapter {chapter_num} stored in FAISS."
    except Exception as exc:
        return f"FAISS store failed (non-fatal): {exc}"


@tool
def update_chapter_in_memory(story_id: str, chapter_num: int, content: str) -> str:
    """
    Replace all FAISS chunks for chapter_num with re-embedded revised content.
    Called by the Editor after a revision to prevent stale pre-revision vectors
    from being retrieved by subsequent chapters. Returns a status string.
    """
    if not story_id or not content.strip():
        return "Skipped: no story_id or empty content."
    try:
        ChapterMemory(story_id).update_chapter(chapter_num, content)
        return f"Chapter {chapter_num} updated in FAISS."
    except Exception as exc:
        return f"FAISS update failed (non-fatal): {exc}"
