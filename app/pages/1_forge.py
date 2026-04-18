"""
app/pages/1_forge.py
The Forge — main story generation page.

Features:
  - Seed prompt input + chapter/revision sliders
  - Model selector: V1 LSTM | V2 Fast | V3 Agentic
  - Live agent activity feed (streaming via st.status blocks)
  - Chapter-by-chapter progressive display
  - Download as .txt or .md
"""

import os
import re
import sys
import json
import time
from pathlib import Path

import streamlit as st
import torch

# ── Path setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT), str(ROOT / "v3_agentic"), str(ROOT / "v1_baseline")])

# LITE_MODE=1 is set by the Dockerfile for HF Spaces (CPU-only deployment).
# In lite mode the page shows Qwen2.5-0.5B on CPU instead of the full GPU pipeline.
LITE_MODE = os.environ.get("LITE_MODE", "0") == "1"

# Curated CPU-safe models. ram_gb is the approximate float32 footprint.
# gated=True means the model requires a HuggingFace token (Settings → Access Tokens).
CURATED_MODELS: dict[str, dict] = {
    "Qwen/Qwen2.5-0.5B-Instruct":                   {"ram_gb": 1.0,  "gated": False},
    "Qwen/Qwen2.5-1.5B-Instruct":                   {"ram_gb": 3.0,  "gated": False},
    "HuggingFaceTB/SmolLM2-1.7B-Instruct":          {"ram_gb": 3.5,  "gated": False},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":    {"ram_gb": 3.0,  "gated": False},
    "microsoft/Phi-3.5-mini-instruct":               {"ram_gb": 7.0,  "gated": False},
    "meta-llama/Llama-3.2-1B-Instruct":             {"ram_gb": 2.5,  "gated": True},
    "custom": {"ram_gb": None, "gated": False},
}

# One model per company, all ≤5B params (4-bit VRAM footprint shown).
# Qwen2.5-1.5B is the only entry with a fine-tuned LoRA adapter.
# Mistral has no sub-5B instruct model; their 7B is included as the closest option.
V3_MODELS: dict[str, dict] = {
    "Qwen/Qwen2.5-1.5B-Instruct":                {"vram_gb": 1.5, "adapter": True,  "gated": False},
    "meta-llama/Llama-3.2-3B-Instruct":           {"vram_gb": 2.5, "adapter": False, "gated": True},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":  {"vram_gb": 1.5, "adapter": False, "gated": False},
    "mistralai/Mistral-7B-Instruct-v0.3":          {"vram_gb": 5.0, "adapter": False, "gated": False},
    "custom":                                       {"vram_gb": None,"adapter": False, "gated": False},
}


def _available_ram_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return None

st.set_page_config(page_title="The Forge | SciFi Forge", page_icon="🔥", layout="wide")


# ── Cached model loaders ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading V1 LSTM (first run only)...")
def load_v1_model():
    """Load V1 char-level LSTM.  Returns (model, chartoindex, indextochar)."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "v1_baseline"))
    from train import load_and_preprocess
    from lstm_model import build_lstm
    import torch, numpy as np

    DATA_PATH  = ROOT / "data" / "raw" / "internet_archive_scifi_v3.txt"
    CKPT_PATH  = ROOT / "tests" / "checkpoints" / "lstm_checkpoints" / "checkpt_best.pt"
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, vocab, chartoindex, indextochar = load_and_preprocess(DATA_PATH)
    state = torch.load(str(CKPT_PATH), map_location=DEVICE)
    model = build_lstm(vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    return model, chartoindex, indextochar


@st.cache_resource(show_spinner="Loading V2 model (first run only)...")
def load_v2_generator():
    from v2_finetuned.generate import load_model
    return load_model()


@st.cache_resource(show_spinner="Loading V3 pipeline (first run only)...")
def load_v3_pipeline():
    from v3_agentic.pipeline.graph import build_graph
    return build_graph()


@st.cache_resource(show_spinner="Loading model — downloading if not cached (~30s–5min)...")
def load_lite_model(model_id: str):
    """Load any HF causal-LM on CPU. Cached separately per model_id."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    hf_token = os.environ.get("HF_TOKEN") or None
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True, token=hf_token,
    )
    model.eval()
    return model, tok


def _trim_to_sentence(text: str) -> str:
    text = text.strip()
    if not text or text[-1] in ".!?\"'":
        return text
    matches = list(re.finditer(r'[.!?]["\']?', text))
    return text[:matches[-1].end()].strip() if matches else text


def _last_paragraph(text: str, fallback_chars: int = 300) -> str:
    """Return the last non-empty paragraph, falling back to a char slice."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras[-1] if paras else text[-fallback_chars:].strip()


def run_lite(prompt: str, chapters: int, model_id: str, max_tokens: int = 300) -> list[dict]:
    """Generate chapters with the selected CPU model using its chat template."""
    model, tok = load_lite_model(model_id)
    results = []
    for i in range(chapters):
        with st.status(f"✍️ Lite: generating chapter {i+1}/{chapters}...", expanded=False):
            t0 = time.perf_counter()

            if i == 0:
                user_msg = (
                    f"Write Chapter 1 of a science fiction story based on this premise:\n\n{prompt}\n\n"
                    f"Write approximately {max_tokens // 2} words. "
                    "Close the chapter at a complete, satisfying narrative moment — "
                    "a resolved beat or a deliberate cliffhanger. Never end mid-action."
                )
            else:
                ending = _last_paragraph(results[-1]["content"])
                user_msg = (
                    f"Write Chapter {i + 1} of a science fiction story. "
                    f"The previous chapter ended with:\n\n\"{ending}\"\n\n"
                    f"Open Chapter {i + 1} with a fresh scene — a new moment, location, or perspective. "
                    f"Write approximately {max_tokens // 2} words. "
                    "Close the chapter at a complete, satisfying narrative moment."
                )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a science fiction author. Every chapter you write has a clear arc: "
                        "an opening that establishes the scene, a development, and a deliberate close. "
                        "Never begin a chapter mid-sentence or as raw continuation of prior prose."
                    ),
                },
                {"role": "user", "content": user_msg},
            ]
            full_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs     = tok(full_prompt, return_tensors="pt", truncation=True, max_length=512)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=0.7, top_p=0.9, repetition_penalty=1.1,
                    do_sample=True, pad_token_id=tok.eos_token_id,
                )
            raw     = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
            content = _trim_to_sentence(raw)
            elapsed = time.perf_counter() - t0
            st.write(f"Chapter {i+1} complete — {len(content.split())} words · {elapsed:.1f}s")
        results.append({"num": i+1, "title": f"Chapter {i+1}", "content": content})
    return results


# ── VRAM sidebar ──────────────────────────────────────────────────────────

def _vram_sidebar():
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.sidebar.metric("GPU VRAM", f"{used:.1f} / {total:.0f} GB")
        if used > 6.5:
            st.sidebar.warning("⚠️ VRAM high")


# ── V1 generation (char-level LSTM) ──────────────────────────────────────

def run_v1(prompt: str, chapters: int, chars_per_chapter: int = 1200) -> list[dict]:
    """Generate `chapters` continuations via V1 char-level LSTM."""
    import torch, numpy as np

    model, chartoindex, indextochar = load_v1_model()
    DEVICE = next(model.parameters()).device

    # Filter seed to known vocabulary — V1 only knows 75 chars
    clean_seed = "".join(c for c in prompt if c in chartoindex)
    if not clean_seed:
        clean_seed = "the "   # safe fallback

    results = []
    for i in range(chapters):
        with st.status(f"🔤 V1 LSTM: generating chapter {i+1}/{chapters}...", expanded=False):
            model.reset_states()
            text_result = []

            input_indices = torch.tensor(
                [[chartoindex[c] for c in clean_seed]], dtype=torch.long
            ).to(DEVICE)

            with torch.no_grad():
                _, _ = model(input_indices)
                for _ in range(chars_per_chapter):
                    logits, _ = model(input_indices[:, -1:])
                    logits     = logits[:, -1, :] / 0.5
                    probs      = torch.softmax(logits, dim=-1)
                    predicted  = torch.multinomial(probs, num_samples=1).item()
                    char       = indextochar[predicted]
                    text_result.append(char)
                    input_indices = torch.tensor([[predicted]], dtype=torch.long).to(DEVICE)

            content = clean_seed + "".join(text_result)
            results.append({
                "num":   i + 1,
                "title": f"Chapter {i + 1}",
                "content": content,
            })
            st.write(f"Chapter {i+1} complete — {len(content)} characters")
    return results


# ── V2 generation (fast, single-shot) ─────────────────────────────────────

def run_v2(prompt: str, chapters: int) -> list[dict]:
    """Generate `chapters` independent continuations via V2 model."""
    from v2_finetuned.generate import generate_text
    results = []
    for i in range(chapters):
        with st.status(f"✍️ V2: generating chapter {i+1}/{chapters}...", expanded=False):
            content = generate_text(prompt, max_new_tokens=400)
            results.append({"num": i+1, "title": f"Chapter {i+1}", "content": content})
            st.write(f"Chapter {i+1} complete — {len(content.split())} words")
    return results


# ── V3 generation (agentic, streamed) ─────────────────────────────────────

def run_v3(prompt: str, chapters: int, revisions: int, model_id: str | None = None) -> dict:
    """Run full V3 pipeline with live agent status updates."""
    from v3_agentic.pipeline.state   import initial_state
    from v3_agentic.pipeline.runner  import stream_pipeline

    agent_icons = {
        "planner":         "🗺️  Planner",
        "writer":          "✍️  Writer",
        "critic":          "🔍 Critic",
        "editor":          "✏️  Editor",
        "advance_chapter": "⏭️  Advancing",
    }

    feed_placeholder = st.empty()
    activity_log: list[str] = []
    final_state: dict = {}

    for node_name, node_state in stream_pipeline(prompt, chapters, revisions, model_id=model_id):
        icon_label = agent_icons.get(node_name, f"⚙️ {node_name}")
        chapter    = node_state.get("current_chapter", "—")
        status_str = node_state.get("status", "")

        msg = f"`{time.strftime('%H:%M:%S')}` **{icon_label}** | chapter {chapter} | {status_str}"
        activity_log.append(msg)

        # Show last 8 events in the live feed
        feed_placeholder.markdown("\n\n".join(activity_log[-8:]))
        final_state = node_state

    return final_state


# ── Chapter display ───────────────────────────────────────────────────────

def display_chapters(chapters: list[dict]) -> None:
    for ch in chapters:
        score = ch.get("critique_score")
        score_str = f"⭐ {score:.2f}" if score is not None else ""
        with st.expander(f"Chapter {ch['num']}: {ch['title']}  {score_str}", expanded=False):
            st.markdown(ch.get("content", "_No content_"))


# ── Download helpers ──────────────────────────────────────────────────────

def _story_to_txt(title: str, chapters: list[dict]) -> str:
    lines = [f"{'='*60}", f"  {title.upper()}", f"{'='*60}\n"]
    for ch in chapters:
        lines += [f"\nCHAPTER {ch['num']}: {ch['title']}\n", ch.get("content",""), "\n"]
    return "\n".join(lines)


def _story_to_md(title: str, logline: str, chapters: list[dict]) -> str:
    lines = [f"# {title}", f"*{logline}*\n"]
    for ch in chapters:
        score = ch.get("critique_score")
        score_str = f" *(score: {score:.2f})*" if score else ""
        lines += [f"\n## Chapter {ch['num']}: {ch['title']}{score_str}\n", ch.get("content",""), ""]
    return "\n".join(lines)


# ── Page layout ───────────────────────────────────────────────────────────

st.markdown("# 🔥 The Forge")
st.divider()

if LITE_MODE:
    # ── LITE MODE: CPU generator with model selector ───────────────────────
    st.markdown("*Single-shot SciFi continuation on CPU — pick any open-source model below.*")
    st.caption("For the full 4-agent pipeline (Planner → Writer → Critic → Editor), see the About page.")

    with st.sidebar:
        st.markdown("### Model")

        curated_ids   = [k for k in CURATED_MODELS if k != "custom"]
        display_names = {
            mid: f"{mid.split('/')[-1]}  ·  ~{CURATED_MODELS[mid]['ram_gb']:.0f} GB RAM"
            for mid in curated_ids
        }
        display_names["custom"] = "Custom HF model ID..."

        selected_label = st.selectbox(
            "Open-source model",
            options=list(display_names.values()),
            index=0,
            help="Models are downloaded from HuggingFace Hub on first use and cached locally.",
        )
        # Reverse-lookup: label → model ID
        label_to_id = {v: k for k, v in display_names.items()}
        selected_id = label_to_id[selected_label]

        if selected_id == "custom":
            selected_id = st.text_input(
                "HuggingFace model ID",
                placeholder="org/model-name",
                help="Any CausalLM on HuggingFace Hub, e.g. Qwen/Qwen2.5-3B-Instruct",
            ).strip()
            meta = {"ram_gb": None, "gated": False}
        else:
            meta = CURATED_MODELS[selected_id]

        # ── RAM warning ───────────────────────────────────────────────────
        avail = _available_ram_gb()
        if meta["ram_gb"] and avail and meta["ram_gb"] > avail:
            st.warning(
                f"⚠️ This model needs ~{meta['ram_gb']:.0f} GB RAM but only "
                f"{avail:.1f} GB is free. Generation may be very slow or crash.",
                icon="⚠️",
            )
        elif meta["ram_gb"] is None:
            st.info("Unknown RAM requirement for custom model. Start small if unsure.", icon="ℹ️")

        # ── HF token notice for gated models ─────────────────────────────
        if meta["gated"] and not os.environ.get("HF_TOKEN"):
            st.warning(
                "This model is gated. Set the `HF_TOKEN` environment variable "
                "to your HuggingFace access token before launching.",
                icon="🔑",
            )

        st.divider()
        st.markdown("### Generation Settings")
        num_chapters = st.slider("Chapters", 1, 5, 2,
                                 help="Each chapter generated independently (CPU mode)")
        max_tokens   = st.slider("Words per chapter (approx)", 100, 400, 200)

    seed = st.text_area(
        "Seed Prompt",
        placeholder="A dying Earth colony ship discovers an alien signal emanating from an abandoned moon...",
        height=100,
    )
    can_generate = bool(seed.strip() and selected_id)
    generate_btn = st.button("🚀 Generate", type="primary", disabled=not can_generate)

    if generate_btn and can_generate:
        st.divider()
        st.markdown("### 📡 Generation Feed")
        try:
            chapters = run_lite(seed, num_chapters, selected_id, max_tokens)
        except Exception as exc:
            st.error(f"Failed to load or run `{selected_id}`: {exc}")
            st.stop()

        title   = f"Story: {seed[:50]}"
        logline = seed

        st.divider()
        st.markdown(f"## {title}")
        col1, col2 = st.columns(2)
        col1.metric("Chapters written", len(chapters))
        col2.metric("Total words", sum(len(c.get("content","").split()) for c in chapters))
        st.divider()
        display_chapters(chapters)

        st.divider()
        st.markdown("### 💾 Download")
        st.download_button(
            "📄 Download .txt",
            data=_story_to_txt(title, chapters),
            file_name=f"{title[:40].replace(' ','_')}.txt",
            mime="text/plain",
        )

else:
    # ── FULL MODE: V1 / V2 / V3 GPU pipeline ──────────────────────────────
    st.markdown("*Generate a full SciFi story using the agentic pipeline.*")

    _vram_sidebar()

    with st.sidebar:
        st.markdown("### Generation Settings")
        model_choice = st.radio(
            "Model",
            ["🔤 V1 LSTM (Character)", "⚡ V2 Fast (QLoRA)", "🤖 V3 Agentic (Quality)"],
            index=1,
            help=(
                "V1: char-level LSTM, raw statistical patterns.\n"
                "V2: fine-tuned QLoRA LLM, coherent prose.\n"
                "V3: full 4-agent pipeline with Critic revision loop."
            ),
        )
        num_chapters = st.slider("Chapters", 1, 10, 4)
        max_revisions = st.slider(
            "Max revisions per chapter", 0, 3, 1,
            disabled=(model_choice != "🤖 V3 Agentic (Quality)"),
            help="V3 only — how many Editor passes per chapter before moving on.",
        )

        # ── V3 model selector ─────────────────────────────────────────────
        v3_model_id = None
        if model_choice.startswith("🤖"):
            st.divider()
            st.markdown("#### V3 Agent Model")

            v3_ids = [k for k in V3_MODELS if k != "custom"]
            v3_labels = {}
            for mid in v3_ids:
                m = V3_MODELS[mid]
                tag = " ✦ fine-tuned" if m["adapter"] else ""
                v3_labels[mid] = f"{mid.split('/')[-1]}  ·  ~{m['vram_gb']:.0f} GB VRAM{tag}"
            v3_labels["custom"] = "Custom HF model ID..."

            v3_selected_label = st.selectbox(
                "Model",
                options=list(v3_labels.values()),
                index=0,
                help="Default uses the fine-tuned QLoRA adapter. Other models run vanilla (no adapter).",
                label_visibility="collapsed",
            )
            v3_label_to_id = {v: k for k, v in v3_labels.items()}
            v3_model_id    = v3_label_to_id[v3_selected_label]

            if v3_model_id == "custom":
                v3_model_id = st.text_input(
                    "HuggingFace model ID",
                    placeholder="org/model-name",
                ).strip() or None
                v3_meta = {"vram_gb": None, "adapter": False, "gated": False}
            else:
                v3_meta = V3_MODELS[v3_model_id]

            if v3_meta.get("vram_gb") and torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                if v3_meta["vram_gb"] > total_vram * 0.85:
                    st.warning(f"⚠️ ~{v3_meta['vram_gb']:.0f} GB VRAM needed, {total_vram:.0f} GB available.")
            if not v3_meta.get("adapter", True):
                st.caption("ℹ️ No fine-tuned adapter for this model — runs as vanilla base LLM.")
            if v3_meta.get("gated") and not os.environ.get("HF_TOKEN"):
                st.warning("Gated model — set `HF_TOKEN` env var before launching.", icon="🔑")

        if model_choice == "🔤 V1 LSTM (Character)":
            st.caption(
                "⚠️ V1 is character-level — it may produce fragmented or nonsensical "
                "text. That is intentional; it shows the baseline before fine-tuning."
            )

    seed = st.text_area(
        "Seed Prompt",
        placeholder="A dying Earth colony ship discovers an alien signal emanating from an abandoned moon...",
        height=100,
    )
    generate_btn = st.button("🚀 Generate Story", type="primary", disabled=not seed.strip())

    if generate_btn and seed.strip():
        st.divider()
        st.markdown("### 📡 Generation Feed")

        use_v1 = model_choice.startswith("🔤")
        use_v3 = model_choice.startswith("🤖")

        if use_v1:
            with st.spinner("Loading V1 LSTM checkpoint..."):
                load_v1_model()
            chapters = run_v1(seed, num_chapters)
            title, logline, error = f"Story: {seed[:50]}", seed, None
        elif use_v3:
            with st.spinner("Initialising V3 pipeline..."):
                load_v3_pipeline()
            result   = run_v3(seed, num_chapters, max_revisions, model_id=v3_model_id)
            chapters = result.get("chapters_written", [])
            title    = result.get("title", "Untitled")
            logline  = result.get("logline", "")
            error    = result.get("error")
        else:
            with st.spinner("Loading V2 model..."):
                load_v2_generator()
            chapters = run_v2(seed, num_chapters)
            title, logline, error = f"Story: {seed[:50]}", seed, None

        st.divider()

        if error:
            st.error(f"Pipeline error: {error}")

        if chapters:
            st.markdown(f"## {title}")
            if logline:
                st.caption(logline)

            scores = [c.get("critique_score") for c in chapters if c.get("critique_score") is not None]
            col1, col2, col3 = st.columns(3)
            col1.metric("Chapters written", len(chapters))
            col2.metric("Avg critique score", f"{sum(scores)/len(scores):.2f}" if scores else "N/A")
            col3.metric("Total words", sum(len(c.get("content","").split()) for c in chapters))

            st.divider()
            display_chapters(chapters)

            st.divider()
            st.markdown("### 💾 Download")
            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "📄 Download .txt",
                data=_story_to_txt(title, chapters),
                file_name=f"{title[:40].replace(' ','_')}.txt",
                mime="text/plain",
            )
            dl2.download_button(
                "📝 Download .md",
                data=_story_to_md(title, logline, chapters),
                file_name=f"{title[:40].replace(' ','_')}.md",
                mime="text/markdown",
            )
