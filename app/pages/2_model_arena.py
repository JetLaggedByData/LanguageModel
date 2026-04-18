"""
app/pages/2_model_arena.py
Model Arena — three columns, one per model version.
"""

import sys
import math
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

st.set_page_config(page_title="Model Arena | SciFi Forge", page_icon="🏆", layout="wide")

# ── Paths ─────────────────────────────────────────────────────────────────
V1_RESULTS      = ROOT / "v1_baseline"  / "eval_results_v1.json"
V2_RESULTS      = ROOT / "v2_finetuned" / "eval_results_v2.json"
ADAPTER_WEIGHTS = ROOT / "v2_finetuned" / "adapters" / "adapter_model.safetensors"
STORIES_DIR     = ROOT / "data" / "stories"

COLORS = {"v1": "#5a7a9a", "v2": "#f5a623", "v3": "#00d4ff"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#090e1a", plot_bgcolor="#0f1829",
    font=dict(color="#c8d8e8", size=12),
    xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
    margin=dict(l=40, r=20, t=40, b=30),
)


# ── Data loaders ──────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _load_v3_stats() -> dict:
    stories_total = stories_ok = chapters_total = words_total = 0
    scores: list[float] = []
    per_story: list[dict] = []

    for p in sorted(STORIES_DIR.glob("*/story.json")):
        data = _load_json(p)
        if not data:
            continue
        stories_total += 1
        chapters = data.get("chapters", [])
        if chapters:
            stories_ok += 1
        chapters_total += len(chapters)
        ch_scores = []
        for ch in chapters:
            s = ch.get("critique_score")
            if s is not None:
                scores.append(float(s))
                ch_scores.append(float(s))
            words_total += len(ch.get("content", "").split())
        if ch_scores:
            per_story.append({
                "title": data.get("title", "?")[:25],
                "avg":   round(sum(ch_scores) / len(ch_scores), 3),
            })

    return {
        "stories_total":  stories_total,
        "stories_ok":     stories_ok,
        "chapters_total": chapters_total,
        "words_total":    words_total,
        "avg_score":      round(sum(scores) / len(scores), 3) if scores else None,
        "min_score":      round(min(scores), 3) if scores else None,
        "max_score":      round(max(scores), 3) if scores else None,
        "all_scores":     scores,
        "per_story":      per_story,
    }


def _adapter_mb() -> str:
    if ADAPTER_WEIGHTS.exists():
        return f"{ADAPTER_WEIGHTS.stat().st_size / 1e6:.1f} MB"
    return "—"


def _f(val, fmt="{}"):
    return fmt.format(val) if val is not None else "—"


# ── Load data ─────────────────────────────────────────────────────────────

v1 = _load_json(V1_RESULTS)
v2 = _load_json(V2_RESULTS)
v3 = _load_v3_stats()


# ── Page header ───────────────────────────────────────────────────────────

st.markdown("# 🏆 Model Arena")
st.markdown("*V1 LSTM · V2 QLoRA · V3 Agentic — each evaluated on its own metrics*")
st.divider()

col1, col2, col3 = st.columns(3)

# ═══════════════════════════════════════════════════════
# V1 LSTM
# ═══════════════════════════════════════════════════════
with col1:
    st.markdown(
        '<div style="font-family:monospace;color:#5a7a9a;font-size:0.75em;">V1</div>'
        '<div style="font-family:monospace;color:#5a7a9a;font-size:1.3em;font-weight:bold;">LSTM</div>',
        unsafe_allow_html=True,
    )
    st.caption("Character-level · TensorFlow/Keras")
    st.divider()

    if v1:
        bpc = math.log2(v1["char_perplexity"])
        rows = [
            ("Char Perplexity",  f"{v1.get('char_perplexity', 0):.4f}"),
            ("Bits/Char (BPC)",  f"{bpc:.3f}"),
            ("BLEU-2 (char)",    f"{v1.get('bleu2', 0):.4f}"),
            ("Speed",            f"{v1.get('inference_chars_per_sec', 0):.0f} c/s"),
            ("Vocab size",       "75 chars"),
            ("Parameters",       "~5.4 M"),
            ("Architecture",     "Embedding → LSTM(1024) → Dense"),
            ("Trained on",       "149M chars SciFi"),
            ("Epochs",           "15"),
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["Metric", "Value"]),
            hide_index=True, width="stretch",
        )

        # Mini bar chart
        fig = go.Figure(go.Bar(
            x=["Char Perplexity", "BLEU-2", "BPC"],
            y=[v1.get("char_perplexity", 0), v1.get("bleu2", 0), bpc],
            marker_color=COLORS["v1"],
            text=[f"{v:.3f}" for v in [v1.get("char_perplexity",0), v1.get("bleu2",0), bpc]],
            textposition="outside", textfont=dict(color="#c8d8e8"),
        ))
        fig.update_layout(title="V1 Metrics", showlegend=False,
                          height=260, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")

    else:
        st.info("Run `python v1_baseline/evaluate.py` to populate.")


# ═══════════════════════════════════════════════════════
# V2 QLoRA
# ═══════════════════════════════════════════════════════
with col2:
    st.markdown(
        '<div style="font-family:monospace;color:#f5a623;font-size:0.75em;">V2</div>'
        '<div style="font-family:monospace;color:#f5a623;font-size:1.3em;font-weight:bold;">QLoRA</div>',
        unsafe_allow_html=True,
    )
    st.caption("Word-level · Qwen2.5-1.5B fine-tuned")
    st.divider()

    if v2:
        bpw = math.log2(v2["word_perplexity"])
        rows = [
            ("Word Perplexity",   f"{v2.get('word_perplexity', 0):.4f}"),
            ("Bits/Word (BPW)",   f"{bpw:.3f}"),
            ("BLEU-2 (word)",     f"{v2.get('bleu2', 0):.4f}"),
            ("BLEU-4 (word)",     f"{v2.get('bleu4', 0):.4f}"),
            ("Speed",             f"{v2.get('inference_tokens_per_sec', 0):.1f} t/s"),
            ("Base model",        "Qwen2.5-1.5B-Instruct"),
            ("Trainable params",  "2.75 M (0.18%)"),
            ("Adapter size",      _adapter_mb()),
            ("Quantization",      "4-bit NF4 QLoRA"),
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["Metric", "Value"]),
            hide_index=True, width="stretch",
        )

        fig = go.Figure(go.Bar(
            x=["Word Perplexity", "BLEU-2", "BLEU-4", "BPW"],
            y=[v2.get("word_perplexity",0), v2.get("bleu2",0), v2.get("bleu4",0), bpw],
            marker_color=COLORS["v2"],
            text=[f"{v:.3f}" for v in [v2.get("word_perplexity",0), v2.get("bleu2",0), v2.get("bleu4",0), bpw]],
            textposition="outside", textfont=dict(color="#c8d8e8"),
        ))
        fig.update_layout(title="V2 Metrics", showlegend=False,
                          height=260, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")

    else:
        st.info("Run `python v2_finetuned/evaluate.py` to populate.")


# ═══════════════════════════════════════════════════════
# V3 Agentic
# ═══════════════════════════════════════════════════════
with col3:
    st.markdown(
        '<div style="font-family:monospace;color:#00d4ff;font-size:0.75em;">V3</div>'
        '<div style="font-family:monospace;color:#00d4ff;font-size:1.3em;font-weight:bold;">Agentic</div>',
        unsafe_allow_html=True,
    )
    st.caption("4-agent LangGraph pipeline")
    st.divider()

    if v3["stories_total"]:
        rows = [
            ("Avg Critique Score", _f(v3["avg_score"],  "{:.3f}")),
            ("Min Score",          _f(v3["min_score"],  "{:.3f}")),
            ("Max Score",          _f(v3["max_score"],  "{:.3f}")),
            ("Stories generated",  str(v3["stories_total"])),
            ("Stories w/ chapters",str(v3["stories_ok"])),
            ("Chapters written",   str(v3["chapters_total"])),
            ("Total words",        f"{v3['words_total']:,}"),
            ("LLM backend",        "User-selected · default: Qwen2.5-1.5B + LoRA"),
            ("Framework",          "LangGraph"),
            ("Memory",             "FAISS + MiniLM-L6"),
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["Metric", "Value"]),
            hide_index=True, width="stretch",
        )

        if v3["per_story"]:
            titles = [s["title"] for s in v3["per_story"]]
            avgs   = [s["avg"]   for s in v3["per_story"]]
            fig = go.Figure(go.Bar(
                x=titles, y=avgs,
                marker_color=COLORS["v3"],
                text=[f"{v:.2f}" for v in avgs],
                textposition="outside", textfont=dict(color="#c8d8e8"),
            ))
            overall = sum(avgs) / len(avgs)
            fig.add_hline(y=overall, line_dash="dash", line_color=COLORS["v2"],
                          annotation_text=f"avg {overall:.2f}",
                          annotation_font_color=COLORS["v2"])
            layout = {
                **PLOTLY_LAYOUT,
                "yaxis":  {**PLOTLY_LAYOUT["yaxis"], "range": [0, 1.1]},
                "xaxis":  {**PLOTLY_LAYOUT["xaxis"], "tickangle": -35,
                           "tickfont": dict(size=10)},
                "margin": dict(l=40, r=20, t=40, b=80),
            }
            fig.update_layout(title="Critique Score per Story", showlegend=False,
                              height=300, **layout)
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Run `python v3_agentic/pipeline/runner.py --prompt ...` to populate.")


# ── Model / Agent breakdown row ───────────────────────────────────────────
st.divider()
st.markdown("#### Model Architecture")

m1, m2, m3 = st.columns(3)

with m1:
    st.caption("V1 LSTM — Components")
    st.dataframe(pd.DataFrame([
        ("LSTM",        "PyTorch (custom)",           "Single-layer char-level LM"),
        ("Embedding",   "75 × 256",                   "Maps each char to a dense vector"),
        ("LSTM layer",  "1024 units",                 "Learns sequential char patterns"),
        ("Output",      "Dense(75) + Softmax",         "Predicts next character"),
    ], columns=["Component", "Config", "Role"]), hide_index=True, width="stretch")

with m2:
    st.caption("V2 QLoRA — Components")
    st.dataframe(pd.DataFrame([
        ("Base LLM",  "Qwen2.5-1.5B-Instruct",    "Pre-trained transformer, weights frozen"),
        ("LoRA A/B",  "r=16, α=32",                "Trainable low-rank adapter matrices"),
        ("Targets",   "q_proj, v_proj",             "Attention query & value projections"),
        ("Quant",     "4-bit NF4 (bitsandbytes)",   "Reduces VRAM from ~6 GB to ~2 GB"),
    ], columns=["Component", "Config", "Role"]), hide_index=True, width="stretch")

with m3:
    st.caption("V3 Agentic — Agents")
    st.dataframe(pd.DataFrame([
        ("🗺️ Planner", "Selected model (default: Qwen2.5-1.5B + LoRA)", "Generates story bible — title, acts, characters, world rules"),
        ("✍️ Writer",  "Selected model (default: Qwen2.5-1.5B + LoRA)", "Drafts each chapter using bible + FAISS context"),
        ("🔍 Critic",  "Selected model (default: Qwen2.5-1.5B + LoRA)", "Scores draft 0–1 on consistency, style, coherence"),
        ("✏️ Editor",  "Selected model (default: Qwen2.5-1.5B + LoRA)", "Rewrites weak chapters based on Critic instructions"),
    ], columns=["Agent", "Model", "Role"]), hide_index=True, width="stretch")
