"""
v3_agentic/pipeline/graph.py
LangGraph state machine for the 4-agent SciFi generation pipeline.

Flow:
  planner → writer → critic ──(revision needed)──→ editor → critic
                         └───(no revision / max reached)──→ writer (next chapter)
                                                       └──(all chapters done)──→ END

Node stubs (agents/planner.py etc.) are imported here.
If an agent raises, the error is caught and stored in state["error"]
so the pipeline fails gracefully rather than crashing the graph.
"""

from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from pipeline.state import StoryState
import agents.planner as _planner_mod
import agents.writer  as _writer_mod
import agents.critic  as _critic_mod
import agents.editor  as _editor_mod

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


# Indirection through lambdas so monkey-patching the module attribute
# (e.g. _planner_mod.planner_node = stub) takes effect at call time.
def planner_node(state): return _planner_mod.planner_node(state)
def writer_node(state):  return _writer_mod.writer_node(state)
def critic_node(state):  return _critic_mod.critic_node(state)
def editor_node(state):  return _editor_mod.editor_node(state)


# ── Routing functions ─────────────────────────────────────────────────────

def route_after_critic(state: StoryState) -> str:
    """
    After Critic scores a chapter, decide:
      - "editor"       → revision needed AND budget remaining
      - "next_chapter" → accept chapter, advance counter
    Also routes to "next_chapter" if an error was set (fail-safe).
    """
    if state.get("error"):
        return "next_chapter"

    needs_revision = state.get("revision_needed", False)
    budget_left    = state.get("revision_count", 0) < state.get("max_revisions", 2)

    if needs_revision and budget_left:
        return "editor"
    return "next_chapter"


def route_after_advance(state: StoryState) -> str:
    """
    After the chapter counter is incremented (via advance_chapter node):
      - "writer" → more chapters to write
      - END      → all chapters complete
    """
    if state.get("error"):
        return END
    # current_chapter is the NEXT chapter to write; stop when it exceeds total
    if state.get("current_chapter", 0) > state.get("total_chapters", 1):
        return END
    return "writer"


# ── Chapter advance node ──────────────────────────────────────────────────

def advance_chapter_node(state: StoryState) -> dict:
    """
    Micro-node between critic acceptance and next writer call.
    Increments current_chapter, resets per-chapter critique fields,
    and updates current_act based on chapter position.

    Keeping this logic in a dedicated node (rather than inside writer_node)
    makes the graph edges explicit and easier to debug in LangGraph Studio.
    """
    chapter      = state.get("current_chapter", 0) + 1
    total        = state.get("total_chapters", 1)
    acts         = state.get("acts", [])

    chapters_per_act = max(1, total // max(len(acts), 1))
    current_act  = min(len(acts), (chapter - 1) // chapters_per_act + 1)

    return {
        "current_chapter":      chapter,
        "current_act":          current_act,
        "revision_count":       0,
        "critique":             None,
        "consistency_score":    None,
        "style_score":          None,
        "coherence_score":      None,
        "revision_needed":      False,
        "revision_instructions": None,
        "status":               "writing" if chapter <= total else "done",
    }


# ── Graph assembly ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assemble and return the compiled LangGraph application.

    Uses MemorySaver for in-process checkpointing — the graph can resume from
    the last completed node if interrupted within the same process.
    For cross-process crash recovery, swap MemorySaver for SqliteSaver:
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver.from_conn_string(str(CHECKPOINT_DIR / "langgraph.db"))
    """
    graph = StateGraph(StoryState)

    graph.add_node("planner",         planner_node)
    graph.add_node("writer",          writer_node)
    graph.add_node("critic",          critic_node)
    graph.add_node("editor",          editor_node)
    graph.add_node("advance_chapter", advance_chapter_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner",  "writer")
    graph.add_edge("writer",   "critic")
    graph.add_edge("editor",   "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "editor":       "editor",
            "next_chapter": "advance_chapter",
        },
    )

    graph.add_conditional_edges(
        "advance_chapter",
        route_after_advance,
        {
            "writer": "writer",
            END:      END,
        },
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ── Singleton app ─────────────────────────────────────────────────────────
_app = None

def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app
