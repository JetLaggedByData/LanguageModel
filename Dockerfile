# Dockerfile
# SciFi Forge — lite app for Hugging Face Spaces (CPU, 16GB RAM)
#
# Builds the app/main.py Streamlit app (auto-detects CPU/GPU at runtime).
# Does NOT include GPU drivers — Qwen2.5-0.5B runs fine on CPU.
#
# Local test:
#   docker build -t scifi-forge .
#   docker run -p 7860:7860 scifi-forge
#   open http://localhost:7860
#
# HF Spaces auto-builds from this file (README.md sets sdk: docker).

FROM python:3.10-slim

# ── System deps ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps (layer-cached separately from app code) ──────────────────
COPY requirements-lite.txt .

# Install CPU-only PyTorch explicitly before everything else.
# Must come first — without the --index-url flag pip would pull the 3GB CUDA build.
RUN pip install --no-cache-dir \
    "torch==2.2.0" "torchvision==0.17.0" "torchaudio==2.2.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Install lite app dependencies (no CUDA, no TF, no bitsandbytes)
RUN pip install --no-cache-dir -r requirements-lite.txt

# ── App code ──────────────────────────────────────────────────────────────
COPY . .

# ── HF Spaces runs as non-root user ──────────────────────────────────────
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Pre-cache HuggingFace model weights at build time ────────────────────
# This avoids a slow first-request download on HF Spaces.
# Remove this block if you want a faster build at the cost of slower cold start.
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForCausalLM; \
import torch; \
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True); \
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', \
    device_map='cpu', torch_dtype=torch.float32, trust_remote_code=True, \
    low_cpu_mem_usage=True); \
print('Model cached successfully')"

# ── Runtime env ───────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV LITE_MODE=1
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
# Silence HuggingFace symlink warning on Spaces
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

EXPOSE 7860

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
