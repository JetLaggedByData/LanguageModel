"""
app/main.py
SciFi Forge — unified entry point for both GPU (local) and CPU (HF Spaces) deployments.

GPU detection is automatic: if torch.cuda.is_available() the full V1/V2/V3 pipeline
is offered; otherwise the app runs in lite mode (Qwen2.5-0.5B, CPU).

The LITE_MODE env var overrides auto-detection — set it to "1" to force CPU mode
regardless of hardware (used by the Dockerfile for HF Spaces).

Run locally:
  streamlit run app/main.py              # auto-detects GPU
  LITE_MODE=1 streamlit run app/main.py  # force CPU/lite mode
"""

import ctypes
import os
from pathlib import Path

# bitsandbytes needs libnvJitLink.so.13 on LD_LIBRARY_PATH.
# When launched via `streamlit run` directly (not scripts/run.sh) the library
# isn't picked up automatically even though it ships inside the venv.
# Preload it with ctypes so the CUDA kernels can resolve the symbol at runtime.
# Safe on CPU — the file simply won't exist.
_bnb_cuda_lib = (
    Path(__file__).resolve().parent.parent
    / ".venv/lib/python3.12/site-packages/nvidia/cu13/lib/libnvJitLink.so.13"
)
if _bnb_cuda_lib.exists():
    try:
        ctypes.CDLL(str(_bnb_cuda_lib), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
else:
    _cuda_lib_dir = str(_bnb_cuda_lib.parent)
    os.environ["LD_LIBRARY_PATH"] = (
        _cuda_lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )

import torch
import streamlit as st

# ── Mode detection ────────────────────────────────────────────────────────
# Respect explicit env var; fall back to hardware detection.
_gpu_available = torch.cuda.is_available()
LITE_MODE = os.environ.get("LITE_MODE", "0") == "1" or not _gpu_available

# Propagate to child pages (1_forge.py reads this env var).
if LITE_MODE:
    os.environ["LITE_MODE"] = "1"

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SciFi Forge",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — dark sci-fi terminal aesthetic ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

:root {
    --bg:        #090e1a;
    --surface:   #0f1829;
    --border:    #1e3a5f;
    --amber:     #f5a623;
    --cyan:      #00d4ff;
    --green:     #39ff14;
    --text:      #c8d8e8;
    --muted:     #5a7a9a;
    --danger:    #ff4b4b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Exo 2', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3 { font-family: 'Share Tech Mono', monospace !important; color: var(--cyan) !important; }
h4, h5, h6 { font-family: 'Exo 2', sans-serif !important; color: var(--amber) !important; }

code, pre { font-family: 'Share Tech Mono', monospace !important; }

[data-testid="stMetricValue"]  { color: var(--cyan) !important; font-family: 'Share Tech Mono', monospace !important; }
[data-testid="stMetricLabel"]  { color: var(--muted) !important; }

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.08em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--cyan) !important;
    color: var(--bg) !important;
}

.stTextArea textarea, .stTextInput input {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

.stSlider [data-baseweb="slider"] { color: var(--cyan) !important; }

.stAlert { border-left: 3px solid var(--amber) !important; }

div[data-testid="stMarkdownContainer"] p { color: var(--text) !important; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🚀 SciFi Forge")
    st.markdown("*Multi-Agent Story Generation*")
    st.divider()

    if LITE_MODE:
        st.info(
            "🖥️ **CPU mode** — Qwen2.5-0.5B.\n\n"
            "🎬 **Full V3 demo** (4-agent GPU pipeline) — see the About page.",
            icon="ℹ️",
        )
    else:
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.metric("GPU VRAM", f"{used:.1f} / {total:.0f} GB")
        if used > 6.5:
            st.warning("⚠️ VRAM high — avoid loading additional models")

    st.divider()
    st.caption("V1 LSTM → V2 QLoRA → V3 Agentic")


# ── Landing page ──────────────────────────────────────────────────────────
st.markdown("# 🚀 SciFi Forge")
st.markdown(
    "##### A multi-agent AI system that evolves a student LSTM into a "
    "LangGraph-orchestrated story generation pipeline."
)
st.markdown("---")

col1, col2, col3 = st.columns(3)
col1.metric("Model Versions", "3", "LSTM → QLoRA → Agentic")
col2.metric("Pipeline Agents", "4", "Plan · Write · Critique · Edit")
if LITE_MODE:
    col3.metric("Live Model", "0.5B", "CPU · Qwen2.5-Instruct")
else:
    col3.metric("VRAM Budget", "8 GB", "4-bit QLoRA")

st.markdown("""
---
**Navigate using the sidebar** to explore the project:

| Page | Description |
|---|---|
| 🔥 **The Forge** | Generate SciFi prose — live agent feed, chapter-by-chapter |
| 🏆 **Model Arena** | Benchmark charts: V1 vs V2 vs V3 perplexity, BLEU, critic scores |
| 📚 **Story Library** | Browse pre-generated stories, read chapters, see critique scores |
| ℹ️ **About** | Project evolution, architecture diagram, links |
""")
