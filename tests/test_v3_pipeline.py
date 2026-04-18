"""
tests/test_v3_pipeline.py
Unit tests for the V3 agentic pipeline.

Tests cover every function that can be exercised without loading a GPU model:
  - State helpers     (make_story_id, initial_state)
  - Routing functions (route_after_critic, route_after_advance)
  - Advance node      (advance_chapter_node)
  - Utility functions (extract_json, trim_to_sentence)
  - Graph execution   (full graph with stub agents, no real model)
"""

import pytest

from pipeline.state import StoryState, make_story_id, initial_state
from pipeline.graph import (
    route_after_critic,
    route_after_advance,
    advance_chapter_node,
    build_graph,
)
from agents.utils import extract_json, trim_to_sentence
from langgraph.graph import END


# ── Helpers ───────────────────────────────────────────────────────────────

def _base_state(**overrides) -> StoryState:
    """Return a minimal valid StoryState for testing, with optional overrides."""
    state = initial_state("A lone explorer discovers alien ruins on Mars.", total_chapters=4)
    state.update({
        "story_id":   "test-story-20240101",
        "title":      "Test Story",
        "logline":    "A short test story.",
        "acts":       [{"act": 1, "summary": "Setup", "chapters": [1, 2]},
                       {"act": 2, "summary": "Climax", "chapters": [3, 4]}],
        "characters": [],
        "current_chapter": 1,
        "current_act":     1,
    })
    state.update(overrides)
    return state


# ══════════════════════════════════════════════════════════════════════════
# make_story_id
# ══════════════════════════════════════════════════════════════════════════

class TestMakeStoryId:
    def test_slug_lowercased(self):
        result = make_story_id("The Last Colony")
        assert result.startswith("the-last-colony")

    def test_spaces_become_hyphens(self):
        result = make_story_id("Red Planet Blues")
        assert "red-planet-blues" in result

    def test_special_chars_stripped(self):
        result = make_story_id("Zara's! Quest: Part 1")
        assert "!" not in result
        assert ":" not in result

    def test_appends_date(self):
        import re
        result = make_story_id("Some Title")
        assert re.search(r"-\d{8}$", result), "Expected date suffix like -20241201"

    def test_empty_title_fallback(self):
        result = make_story_id("")
        assert result.startswith("story-")

    def test_long_title_truncated(self):
        long_title = "A" * 100
        result = make_story_id(long_title)
        # slug part must not exceed 40 chars (plus date suffix)
        slug = result.rsplit("-", 1)[0]
        assert len(slug) <= 40


# ══════════════════════════════════════════════════════════════════════════
# initial_state
# ══════════════════════════════════════════════════════════════════════════

class TestInitialState:
    def test_valid_prompt_accepted(self):
        state = initial_state("A crew discovers a derelict spaceship.")
        assert state["seed_prompt"] == "A crew discovers a derelict spaceship."

    def test_whitespace_stripped_from_prompt(self):
        state = initial_state("  trimmed prompt  ")
        assert state["seed_prompt"] == "trimmed prompt"

    def test_empty_prompt_raises(self):
        with pytest.raises(ValueError, match="seed_prompt"):
            initial_state("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            initial_state("   ")

    def test_chapters_below_minimum_raises(self):
        with pytest.raises(ValueError, match="total_chapters"):
            initial_state("valid prompt", total_chapters=0)

    def test_chapters_above_maximum_raises(self):
        with pytest.raises(ValueError, match="total_chapters"):
            initial_state("valid prompt", total_chapters=11)

    def test_revisions_below_minimum_raises(self):
        with pytest.raises(ValueError, match="max_revisions"):
            initial_state("valid prompt", max_revisions=-1)

    def test_revisions_above_maximum_raises(self):
        with pytest.raises(ValueError, match="max_revisions"):
            initial_state("valid prompt", max_revisions=4)

    def test_counters_initialised_to_zero(self):
        state = initial_state("valid prompt")
        assert state["current_chapter"] == 0
        assert state["revision_count"] == 0
        assert state["chapters_written"] == []

    def test_status_set_to_planning(self):
        state = initial_state("valid prompt")
        assert state["status"] == "planning"

    def test_optional_fields_none(self):
        state = initial_state("valid prompt")
        assert state["error"] is None
        assert state["critique"] is None
        assert state["consistency_score"] is None


# ══════════════════════════════════════════════════════════════════════════
# route_after_critic
# ══════════════════════════════════════════════════════════════════════════

class TestRouteAfterCritic:
    def test_routes_to_editor_when_revision_needed_and_budget_left(self):
        state = _base_state(revision_needed=True, revision_count=0, max_revisions=2)
        assert route_after_critic(state) == "editor"

    def test_routes_to_next_chapter_when_no_revision_needed(self):
        state = _base_state(revision_needed=False, revision_count=0, max_revisions=2)
        assert route_after_critic(state) == "next_chapter"

    def test_routes_to_next_chapter_when_budget_exhausted(self):
        state = _base_state(revision_needed=True, revision_count=2, max_revisions=2)
        assert route_after_critic(state) == "next_chapter"

    def test_fails_safe_to_next_chapter_on_error(self):
        state = _base_state(revision_needed=True, revision_count=0, max_revisions=2,
                            error="Something broke")
        assert route_after_critic(state) == "next_chapter"

    def test_routes_to_editor_at_zero_revisions_with_budget(self):
        state = _base_state(revision_needed=True, revision_count=0, max_revisions=1)
        assert route_after_critic(state) == "editor"

    def test_routes_to_next_chapter_when_budget_is_zero(self):
        state = _base_state(revision_needed=True, revision_count=0, max_revisions=0)
        assert route_after_critic(state) == "next_chapter"


# ══════════════════════════════════════════════════════════════════════════
# route_after_advance
# ══════════════════════════════════════════════════════════════════════════

class TestRouteAfterAdvance:
    def test_routes_to_writer_when_more_chapters_remain(self):
        state = _base_state(current_chapter=1, total_chapters=4)
        assert route_after_advance(state) == "writer"

    def test_routes_to_end_when_chapter_exceeds_total(self):
        # current_chapter after advance = N+1; stop when N+1 > total_chapters
        state = _base_state(current_chapter=5, total_chapters=4)
        assert route_after_advance(state) == END

    def test_still_writes_final_chapter(self):
        # chapter=4 with total=4 means the 4th chapter hasn't been written yet
        state = _base_state(current_chapter=4, total_chapters=4)
        assert route_after_advance(state) == "writer"

    def test_fails_safe_to_end_on_error(self):
        state = _base_state(current_chapter=2, total_chapters=4, error="Pipeline error")
        assert route_after_advance(state) == END

    def test_routes_to_writer_at_chapter_zero(self):
        # Edge case: just after planner, before first writer call
        state = _base_state(current_chapter=0, total_chapters=4)
        assert route_after_advance(state) == "writer"


# ══════════════════════════════════════════════════════════════════════════
# advance_chapter_node
# ══════════════════════════════════════════════════════════════════════════

class TestAdvanceChapterNode:
    def test_increments_chapter_counter(self):
        state = _base_state(current_chapter=1, total_chapters=4)
        result = advance_chapter_node(state)
        assert result["current_chapter"] == 2

    def test_resets_revision_count(self):
        state = _base_state(current_chapter=1, revision_count=2)
        result = advance_chapter_node(state)
        assert result["revision_count"] == 0

    def test_resets_all_critique_fields(self):
        state = _base_state(
            current_chapter=1,
            consistency_score=0.8,
            style_score=0.7,
            coherence_score=0.9,
            revision_needed=True,
            revision_instructions="Fix this.",
        )
        result = advance_chapter_node(state)
        assert result["consistency_score"] is None
        assert result["style_score"] is None
        assert result["coherence_score"] is None
        assert result["revision_needed"] is False
        assert result["revision_instructions"] is None

    def test_sets_status_to_writing_mid_story(self):
        state = _base_state(current_chapter=1, total_chapters=4)
        result = advance_chapter_node(state)
        assert result["status"] == "writing"

    def test_sets_status_to_done_on_final_chapter(self):
        state = _base_state(current_chapter=4, total_chapters=4)
        result = advance_chapter_node(state)
        assert result["status"] == "done"

    def test_act_calculation_single_act(self):
        state = _base_state(current_chapter=0, total_chapters=4,
                            acts=[{"act": 1, "summary": "all", "chapters": [1, 2, 3, 4]}])
        result = advance_chapter_node(state)
        assert result["current_act"] == 1

    def test_act_calculation_two_acts(self):
        state = _base_state(
            current_chapter=2, total_chapters=4,
            acts=[{"act": 1, "summary": "A", "chapters": [1, 2]},
                  {"act": 2, "summary": "B", "chapters": [3, 4]}],
        )
        result = advance_chapter_node(state)
        assert result["current_act"] == 2

    def test_handles_empty_acts_gracefully(self):
        state = _base_state(current_chapter=1, total_chapters=4, acts=[])
        result = advance_chapter_node(state)
        assert "current_chapter" in result


# ══════════════════════════════════════════════════════════════════════════
# extract_json
# ══════════════════════════════════════════════════════════════════════════

class TestExtractJson:
    def test_extracts_clean_json(self):
        raw = '{"key": "value"}'
        assert extract_json(raw) == '{"key": "value"}'

    def test_strips_markdown_fences(self):
        raw = '```json\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == '{"key": "value"}'

    def test_strips_plain_code_fences(self):
        raw = '```\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == '{"key": "value"}'

    def test_extracts_json_with_prose_before(self):
        raw = 'Here is the JSON: {"key": "value"}'
        assert extract_json(raw) == '{"key": "value"}'

    def test_extracts_json_with_prose_after(self):
        raw = '{"key": "value"} and some trailing text'
        assert extract_json(raw) == '{"key": "value"}'

    def test_handles_nested_braces(self):
        raw = '{"outer": {"inner": "value"}}'
        assert extract_json(raw) == '{"outer": {"inner": "value"}}'

    def test_returns_raw_when_no_json_found(self):
        raw = "no json here"
        assert extract_json(raw) == "no json here"

    def test_handles_empty_string(self):
        result = extract_json("")
        assert result == ""


# ══════════════════════════════════════════════════════════════════════════
# trim_to_sentence
# ══════════════════════════════════════════════════════════════════════════

class TestTrimToSentence:
    def test_complete_sentence_unchanged(self):
        text = "The ship landed safely."
        assert trim_to_sentence(text) == text

    def test_complete_exclamation_unchanged(self):
        text = "Watch out!"
        assert trim_to_sentence(text) == text

    def test_complete_question_unchanged(self):
        text = "What is out there?"
        assert trim_to_sentence(text) == text

    def test_trims_incomplete_fragment(self):
        text = "The ship landed safely. She walked to the"
        result = trim_to_sentence(text)
        assert result == "The ship landed safely."

    def test_trims_mid_word_cutoff(self):
        text = "She opened the airlock. The cold air rush"
        result = trim_to_sentence(text)
        assert result == "She opened the airlock."

    def test_empty_string_returns_empty(self):
        assert trim_to_sentence("") == ""

    def test_no_sentence_boundary_returns_original(self):
        text = "no sentence boundary here at all"
        assert trim_to_sentence(text) == text

    def test_preserves_closing_quote(self):
        text = '"Run," she said.'
        assert trim_to_sentence(text) == text

    def test_multiple_sentences_trims_to_last_complete(self):
        text = "First sentence. Second sentence. Incomplete fragm"
        result = trim_to_sentence(text)
        assert result == "First sentence. Second sentence."


# ══════════════════════════════════════════════════════════════════════════
# Full graph with stub agents (no real model)
# ══════════════════════════════════════════════════════════════════════════

class TestGraphWithStubs:
    """
    Exercises the full LangGraph compile → stream loop using the stub agents
    from agents/_stubs.py. No GPU required.

    The monkeypatch fixture swaps the production node functions on the imported
    module objects that graph.py references through its lambda indirection, so
    the graph calls our stubs without touching model weights.
    """

    @pytest.fixture(autouse=True)
    def patch_agents(self, monkeypatch):
        import agents.planner as planner_mod
        import agents.writer  as writer_mod
        import agents.critic  as critic_mod
        import agents.editor  as editor_mod

        def stub_planner(state):
            from pipeline.state import make_story_id
            title    = f"Story: {state['seed_prompt'][:40]}"
            story_id = make_story_id(title)
            return {
                "story_id":        story_id,
                "title":           title,
                "logline":         "A gripping sci-fi adventure.",
                "acts":            [
                    {"act": 1, "summary": "Setup",       "chapters": [1, 2]},
                    {"act": 2, "summary": "Confrontation","chapters": [3, 4]},
                ],
                "characters":      [],
                "world_rules":     ["FTL travel exists"],
                "technology":      ["Hyperdrive"],
                "current_chapter": 1,
                "current_act":     1,
                "status":          "writing",
                "error":           None,
            }

        def stub_writer(state):
            chapter_num = state.get("current_chapter", 1)
            existing    = list(state.get("chapters_written", []))
            existing.append({
                "num":            chapter_num,
                "title":          f"Chapter {chapter_num}",
                "content":        f"Chapter {chapter_num} content.",
                "critique_score": None,
                "revision_count": 0,
            })
            return {"chapters_written": existing, "status": "critiquing", "error": None}

        def stub_critic(state):
            chapters = list(state.get("chapters_written", []))
            if chapters:
                chapters[-1] = {**chapters[-1], "critique_score": 0.85}
            return {
                "consistency_score":     0.85,
                "style_score":           0.85,
                "coherence_score":       0.85,
                "critique":              "Looks good.",
                "revision_needed":       False,
                "revision_instructions": None,
                "chapters_written":      chapters,
                "status":                "writing",
                "error":                 None,
            }

        def stub_editor(state):
            return {
                "revision_count": state.get("revision_count", 0) + 1,
                "status":         "critiquing",
            }

        monkeypatch.setattr(planner_mod, "planner_node", stub_planner)
        monkeypatch.setattr(writer_mod,  "writer_node",  stub_writer)
        monkeypatch.setattr(critic_mod,  "critic_node",  stub_critic)
        monkeypatch.setattr(editor_mod,  "editor_node",  stub_editor)

    def test_graph_compiles(self):
        app = build_graph()
        assert app is not None

    def test_graph_runs_to_completion(self):
        state = initial_state("A test prompt.", total_chapters=2, max_revisions=1)
        config = {"configurable": {"thread_id": "test-thread-001"}}
        app    = build_graph()
        result = app.invoke(state, config=config)
        assert result is not None

    def test_all_chapters_written(self):
        state  = initial_state("A test prompt.", total_chapters=2, max_revisions=1)
        config = {"configurable": {"thread_id": "test-thread-002"}}
        app    = build_graph()
        result = app.invoke(state, config=config)
        assert len(result.get("chapters_written", [])) == 2

    def test_all_chapters_written_matches_total(self):
        # After the final chapter, advance increments to total+1 before END
        state  = initial_state("A test prompt.", total_chapters=3, max_revisions=0)
        config = {"configurable": {"thread_id": "test-thread-003"}}
        app    = build_graph()
        result = app.invoke(state, config=config)
        assert len(result.get("chapters_written", [])) == 3

    def test_graph_streams_node_names(self):
        from pipeline.runner import stream_pipeline
        nodes_seen = []
        for node_name, _ in stream_pipeline("A test prompt.", total_chapters=1, max_revisions=0):
            nodes_seen.append(node_name)
        assert "planner" in nodes_seen
        assert "writer"  in nodes_seen
        assert "critic"  in nodes_seen

    def test_stream_yields_full_accumulated_state(self):
        """
        Regression test for the stream_pipeline partial-state bug.
        Every yielded state must include title and logline (set by planner),
        even when the last node was writer or critic.
        """
        from pipeline.runner import stream_pipeline
        for _, state in stream_pipeline("A test prompt.", total_chapters=1, max_revisions=0):
            # title is set by planner — must be present in ALL subsequent yields
            assert "title"   in state, f"'title' missing from state after a node ran"
            assert "logline" in state, f"'logline' missing from state after a node ran"

    def test_revision_loop_respects_max(self):
        """When Critic forces revision, Editor+Critic cycle runs exactly max_revisions times."""
        import agents.critic as critic_mod

        # Override critic to always request revision
        original = critic_mod.critic_node
        call_count = {"n": 0}

        def always_revise(state):
            call_count["n"] += 1
            chapters = list(state.get("chapters_written", []))
            if chapters:
                chapters[-1] = {**chapters[-1], "critique_score": 0.4}
            return {
                "consistency_score":     0.4,
                "style_score":           0.4,
                "coherence_score":       0.4,
                "critique":              "Needs work.",
                "revision_needed":       True,
                "revision_instructions": "Improve everything.",
                "chapters_written":      chapters,
                "status":                "editing",
                "error":                 None,
            }

        critic_mod.critic_node = always_revise
        try:
            state  = initial_state("Revision test.", total_chapters=1, max_revisions=2)
            config = {"configurable": {"thread_id": "test-revision-001"}}
            app    = build_graph()
            result = app.invoke(state, config=config)
            # Critic called once initially + once per revision = 1 + max_revisions
            assert call_count["n"] == 3
            assert result.get("revision_count") == 0  # reset by advance_chapter_node
        finally:
            critic_mod.critic_node = original
