#!/usr/bin/env bash
# scripts/run.sh — launch the Streamlit app with the correct CUDA libraries on LD_LIBRARY_PATH.
#
# bitsandbytes requires libnvJitLink.so.13 which ships inside the venv but isn't
# automatically added to LD_LIBRARY_PATH by pip.  Without it, 4-bit NF4 quantization
# fails with "Missing dependency: libnvJitLink.so.13".
#
# Usage:
#   bash scripts/run.sh            # auto-detects GPU; full pipeline if available, CPU lite if not
#   LITE_MODE=1 bash scripts/run.sh  # force CPU/lite mode

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_SITE="$PROJECT_ROOT/.venv/lib/python3.12/site-packages"

export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_ROOT"

exec .venv/bin/streamlit run app/main.py \
    --server.port=8501 \
    --server.address=0.0.0.0
