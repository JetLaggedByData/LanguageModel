# App — Streamlit UI

## Entry point

`main.py` — unified entry point for both GPU (local) and CPU (HF Spaces) deployments.
Auto-detects GPU at startup; set `LITE_MODE=1` to force CPU mode.

## Pages

| Page | File | Description |
|---|---|---|
| The Forge | `pages/1_forge.py` | Live story generation — V1/V2/V3 on GPU, or model-selectable CPU |
| Model Arena | `pages/2_model_arena.py` | Benchmark charts: perplexity, BLEU, critique scores |
| Story Library | `pages/3_story_library.py` | Browse pre-generated stories with per-chapter critique scores |
| About | `pages/4_about.py` | Interactive architecture diagram, project evolution, links |

## Run locally

```bash
streamlit run app/main.py              # auto-detects GPU
LITE_MODE=1 streamlit run app/main.py  # force CPU/lite mode
```
