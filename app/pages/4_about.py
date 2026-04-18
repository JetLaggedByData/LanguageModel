"""
app/pages/4_about.py
About — project overview, evolution diagram, architecture diagrams, demo video.
"""

import os
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[2]
LITE_MODE     = os.environ.get("LITE_MODE", "0") == "1"
DEMO_VIDEO_URL = os.environ.get("DEMO_VIDEO_URL", "")   # set in deployment env or .streamlit/secrets.toml
GITHUB_URL    = os.environ.get("GITHUB_URL", "https://github.com/JetLaggedByData/LanguageModel")
HF_SPACE_URL  = os.environ.get("HF_SPACE_URL", "https://huggingface.co/spaces/JetLaggedByData/scifi-forge")

st.set_page_config(page_title="About | SciFi Forge", page_icon="ℹ️", layout="wide")

st.markdown("# ℹ️ About SciFi Forge")
st.divider()

# ── Project overview ──────────────────────────────────────────────────────
st.markdown("## Project Overview")
st.markdown("""
SciFi Forge is a three-generation language model project that evolves from a
character-level LSTM trained from scratch, through a fine-tuned large language model,
to a full multi-agent agentic pipeline capable of generating complete, self-critiqued
science fiction stories. Each version builds directly on the previous, adding a
distinct capability layer.
""")

st.divider()

# ── Demo video (GPU pipeline) ─────────────────────────────────────────────
if DEMO_VIDEO_URL:
    st.markdown("## 🎬 Full V3 Pipeline Demo")
    st.markdown("*Planner → Writer → Critic → Editor running on a local 8GB GPU.*")
    st.video(DEMO_VIDEO_URL)
    st.divider()
elif LITE_MODE:
    st.info("Head to **The Forge** to generate your own SciFi story.")
    st.divider()

# ── Interactive architecture + evolution diagram ──────────────────────────
st.markdown("## Architecture & Evolution")

pipeline_html = ROOT / "assets" / "architecture" / "pipeline.html"
if pipeline_html.exists():
    st.caption("Interactive diagram — click tabs to switch between V3 Pipeline and Evolution views. Hover nodes for details.")
    components.html(pipeline_html.read_text(encoding="utf-8"), height=860, scrolling=False)
else:
    # ── Fallback: static evolution columns ───────────────────────────────
    st.markdown("### Project Evolution")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""**V1 — LSTM** *(MSc coursework)*
- TensorFlow / Keras → PyTorch port
- Char-level, 75-char vocab
- 149M chars SciFi corpus
- Google Colab GPU""")
    with col2:
        st.markdown("""**V2 — QLoRA** *(self-directed)*
- Qwen2.5-1.5B-Instruct
- 4-bit NF4 QLoRA fine-tuning
- 50k instruction samples
- 8GB consumer GPU""")
    with col3:
        st.markdown("""**V3 — Agentic** *(portfolio project)*
- LangGraph state machine + MemorySaver
- 4 agents with shared model loader
- `@tool` memory abstraction (FAISS)
- MLflow · 60 pytest tests""")

    st.markdown("### Architecture")
    tab_v1, tab_v2, tab_v3 = st.tabs(["V1 LSTM", "V2 QLoRA", "V3 Agentic"])

    with tab_v1:
        v1_img = ROOT / "assets" / "architecture" / "v1.png"
        if v1_img.exists():
            st.image(str(v1_img), use_column_width=True)
        else:
            st.markdown("""
**V1 char-level LSTM** — trained from scratch on 149M characters of SciFi text.

```
Input chars → Embedding(75, 256) → LSTM(1024) → Dense(75) → Softmax
```

Key design:
- Stateful LSTM: hidden state passed between batches to learn long-range patterns
- Temperature sampling at inference (T=0.5) for controlled randomness
- AMP training + gradient clipping for stability on consumer GPU
- Evaluated on: char perplexity, bits/char, BLEU-2
            """)

    with tab_v2:
        v2_img = ROOT / "assets" / "architecture" / "v2.png"
        if v2_img.exists():
            st.image(str(v2_img), use_column_width=True)
        else:
            st.markdown("""
**V2 QLoRA fine-tuning** — adapts Qwen2.5-1.5B-Instruct on 50k SciFi instruction pairs.

```
Qwen2.5-1.5B (frozen, 4-bit NF4) + LoRA adapters (r=16, α=32) on q_proj & v_proj
```

Key design:
- 4-bit NF4 quantisation: reduces VRAM from ~6 GB to ~2 GB
- Only 2.75M trainable parameters (0.18% of total)
- `paged_adamw_8bit` + gradient checkpointing for 8GB VRAM budget
- `load_best_model_at_end=True` saves the eval-loss-minimising checkpoint
            """)

    with tab_v3:
        v3_img = ROOT / "assets" / "architecture" / "v3.png"
        if v3_img.exists():
            st.image(str(v3_img), use_column_width=True)
        else:
            st.markdown("""
**V3 LangGraph agentic pipeline** — 4 specialised agents orchestrated as a state machine.

```
planner → writer → critic ──(revision needed)──→ editor → critic
                       └───(accepted / budget exhausted)──→ advance_chapter → writer*
                                                                          └──(done)──→ END
```

Key design:
- `TypedDict` state passed between nodes — typed, diff'd, merged by LangGraph automatically
- `MemorySaver` checkpointing — graph can resume from last completed node within a session
- `@tool` decorated memory operations — FAISS retrieval, StoryBible lookup, character voices
- Shared `load_agent_model()` context manager — VRAM freed after every node, no singleton leaks
- Critic blends LLM score (75%) with embedding-based voice consistency signal (25%)
- `advance_chapter_node` is a dedicated micro-node — keeps state transitions explicit and debuggable
- 60 pytest tests covering routing logic, state validation, and full graph execution with stubs
            """)

st.divider()

# ── Links ─────────────────────────────────────────────────────────────────
st.markdown("## 🔗 Links")
st.markdown(f"- **GitHub**: [{GITHUB_URL}]({GITHUB_URL})")
if HF_SPACE_URL and "JetLaggedByData" in HF_SPACE_URL:
    st.markdown(f"- **HF Space**: [{HF_SPACE_URL}]({HF_SPACE_URL})")
if DEMO_VIDEO_URL:
    st.markdown(f"- **Demo Video**: [Watch full V3 pipeline]({DEMO_VIDEO_URL})")

st.divider()
st.caption("SciFi Forge · V1 LSTM → V2 QLoRA → V3 Agentic · Built on 8GB VRAM · 100% free tools")
