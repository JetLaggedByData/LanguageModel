# V3 — LangGraph Agentic Pipeline

Four specialised agents orchestrated as a LangGraph state machine.
Each chapter passes through Planner → Writer → Critic → (Editor →) Critic until
quality thresholds are met or the revision budget is exhausted.

## Structure

```
agents/       Planner, Writer, Critic, Editor + shared model loader
memory/       StoryBible (JSON), ChapterMemory (FAISS), CharacterMemory (embeddings)
pipeline/     LangGraph graph, TypedDict state, CLI + streaming runner
evaluate/     Cross-version benchmark, story-level consistency scorer, chart exporter
```

## Key design decisions

- **Shared model loader** — context manager loads/unloads the model per node; VRAM freed after every agent call
- **FAISS chapter memory** — semantic retrieval with 50% sentence overlap; revised chapters replace their old embeddings in-place
- **Critic blending** — consistency score = 75% LLM score + 25% embedding-based voice signal
- **`advance_chapter` micro-node** — dedicated node for counter/reset to keep state transitions explicit

## Run
```bash
# Single story via CLI
python v3_agentic/pipeline/runner.py \
  --prompt "A dying colony ship discovers an alien signal" \
  --chapters 4 --revisions 2

# Pre-generate 5 stories for the deployed app
python scripts/pregenerate_stories.py

# Benchmark all three versions
python v3_agentic/evaluate/benchmark.py
```
