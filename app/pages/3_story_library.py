"""
app/pages/3_story_library.py
Story Library — browse and read all generated stories.

Loads stories from data/stories/*/story.json.
Supports filter by title search, chapter count, and min critique score.
Shows critique scores per chapter inline.
"""

import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

st.set_page_config(page_title="Story Library | SciFi Forge", page_icon="📚", layout="wide")

STORIES_DIR = ROOT / "data" / "stories"


# ── Loaders ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_all_stories() -> list[dict]:
    stories = []
    for path in sorted(STORIES_DIR.glob("*/story.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_path"] = str(path)
            scores = [
                c.get("critique_score")
                for c in data.get("chapters", [])
                if c.get("critique_score") is not None
            ]
            data["_avg_score"] = round(sum(scores) / len(scores), 3) if scores else None
            stories.append(data)
        except Exception:
            continue
    return stories


# ── Page layout ───────────────────────────────────────────────────────────

st.markdown("# 📚 Story Library")
st.markdown("*All V3-generated stories, browsable with per-chapter critique scores.*")
st.divider()

stories = load_all_stories()

if not stories:
    st.info(
        "No stories found yet.\n\n"
        "Generate one from **The Forge** page, or run:\n"
        "```bash\npython v3_agentic/pipeline/runner.py --prompt 'Your seed' --chapters 4\n```"
    )
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filter Stories")
    search       = st.text_input("Search title", placeholder="colony, mars, alien...")
    min_chapters = st.slider("Min chapters", 0, 10, 0)
    min_score    = st.slider("Min avg score", 0.0, 1.0, 0.0, step=0.05)

filtered = [
    s for s in stories
    if (not search or search.lower() in s.get("title", "").lower())
    and len(s.get("chapters", [])) >= min_chapters
    and (s["_avg_score"] is None or s["_avg_score"] >= min_score)
]

st.markdown(f"**{len(filtered)} stories** ({len(stories)} total)")
st.divider()

# ── Story grid ────────────────────────────────────────────────────────────
if not filtered:
    st.warning("No stories match the current filters.")
    st.stop()

cols = st.columns(3)
for i, story in enumerate(filtered):
    title      = story.get("title", "Untitled")
    logline    = story.get("logline", "")
    chapters   = story.get("chapters", [])
    avg_score  = story.get("_avg_score")
    n_chapters = len(chapters)

    score_str = f"⭐ {avg_score:.2f}" if avg_score is not None else "unscored"

    with cols[i % 3]:
        st.markdown(
            f"""<div style="border:1px solid #1e3a5f;border-radius:6px;padding:14px;
            margin-bottom:12px;background:#0f1829;">
            <div style="font-family:'Share Tech Mono',monospace;color:#00d4ff;
            font-size:1em;margin-bottom:6px;">{title}</div>
            <div style="color:#5a7a9a;font-size:0.8em;margin-bottom:8px;">{logline[:100]}...</div>
            <div style="color:#f5a623;font-size:0.8em;">{n_chapters} chapters &nbsp;|&nbsp; {score_str}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("Read →", key=f"read_{i}"):
            st.session_state["selected_story_path"] = story["_path"]

st.divider()

# ── Story reader ──────────────────────────────────────────────────────────
selected_path = st.session_state.get("selected_story_path")

all_titles = [s.get("title", f"Story {i}") for i, s in enumerate(filtered)]
chosen = st.selectbox("Or select a story to read:", ["— select —"] + all_titles)
if chosen != "— select —":
    idx = all_titles.index(chosen)
    st.session_state["selected_story_path"] = filtered[idx]["_path"]
    selected_path = filtered[idx]["_path"]

if selected_path:
    matched = [s for s in filtered if s.get("_path") == selected_path]
    if not matched:
        st.stop()

    story    = matched[0]
    title    = story.get("title", "Untitled")
    logline  = story.get("logline", "")
    chapters = story.get("chapters", [])
    chars    = story.get("characters", [])
    rules    = story.get("world_rules", [])

    st.markdown(f"## {title}")
    st.caption(logline)

    tab_story, tab_bible, tab_scores = st.tabs(["📖 Chapters", "📜 Story Bible", "📊 Scores"])

    with tab_story:
        if not chapters:
            st.warning("No chapters were generated for this story.")
        for ch in chapters:
            score = ch.get("critique_score")
            badge = f"  ⭐ {score:.2f}" if score is not None else ""
            with st.expander(f"**Chapter {ch['num']}: {ch['title']}**{badge}", expanded=False):
                st.markdown(ch.get("content", "_No content_"))
                if score is not None:
                    colour = "green" if score >= 0.75 else "orange" if score >= 0.6 else "red"
                    st.markdown(f"*Critique score: :{colour}[{score:.2f}]*")

    with tab_bible:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Characters**")
            if chars:
                for c in chars:
                    traits = ", ".join(c.get("traits", []))
                    st.markdown(f"- **{c.get('name')}** ({c.get('role')}) — {traits}")
            else:
                st.caption("No character data.")
        with col2:
            st.markdown("**World Rules**")
            if rules:
                for r in rules:
                    st.markdown(f"- {r}")
            else:
                st.caption("No world rules recorded.")
        tech = story.get("technology", [])
        if tech:
            st.markdown("**Technology**")
            st.markdown(", ".join(tech))

    with tab_scores:
        scored = [(ch["num"], ch.get("critique_score")) for ch in chapters
                  if ch.get("critique_score") is not None]

        score_rows = [
            {
                "Chapter": ch["num"],
                "Title":   ch.get("title", ""),
                "Score":   f"{ch['critique_score']:.3f}" if ch.get("critique_score") is not None else "—",
            }
            for ch in chapters
        ]
        st.dataframe(pd.DataFrame(score_rows), width="stretch", hide_index=True)

        if scored:
            import plotly.graph_objects as go
            fig = go.Figure(go.Scatter(
                x=[c[0] for c in scored],
                y=[c[1] for c in scored],
                mode="lines+markers",
                line=dict(color="#00d4ff", width=2),
                marker=dict(size=8, color="#f5a623"),
            ))
            fig.update_layout(
                title="Critique Score by Chapter",
                xaxis_title="Chapter", yaxis_title="Score",
                paper_bgcolor="#090e1a", plot_bgcolor="#0f1829",
                font=dict(color="#c8d8e8"),
                yaxis=dict(range=[0, 1], gridcolor="#1e3a5f"),
                xaxis=dict(gridcolor="#1e3a5f"),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No critique scores recorded for this story.")
