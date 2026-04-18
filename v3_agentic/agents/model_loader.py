"""
v3_agentic/agents/model_loader.py
Shared LoRA model context manager used by all four agent nodes.

Eliminates the ~30-line load/unload boilerplate that was copy-pasted
across planner, writer, critic, and editor. The context manager guarantees
VRAM is released even when the node raises, without requiring explicit
finally blocks in each agent.
"""

import threading
import torch
from contextlib import contextmanager
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from agents.utils import bnb_config, clear_gpu


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
_ROOT            = Path(__file__).resolve().parent.parent.parent
ADAPTER_DIR      = _ROOT / "v2_finetuned/adapters"

# Per-thread model override — each Streamlit session runs in its own thread,
# so threading.local() isolates concurrent users rather than a shared global.
_local = threading.local()


def set_agent_model(model_id: str) -> None:
    """Called by the Streamlit UI (or CLI) to choose which model powers V3 agents."""
    _local.model_id = model_id


def _get_active_model_id() -> str:
    return getattr(_local, "model_id", DEFAULT_MODEL_ID)


@contextmanager
def load_agent_model(
    model_id: str | None = None,
    adapter_dir: Path = ADAPTER_DIR,
):
    """
    Load a causal-LM, yield (model, tokeniser), then unload on exit.

    If model_id is None, uses whatever was last set via set_agent_model().
    The LoRA adapter is applied only when the model matches DEFAULT_MODEL_ID
    and the adapter directory exists — other models run as vanilla base models.

    Usage:
        with load_agent_model() as (model, tok):
            output = model.generate(...)
        # VRAM freed here — even if an exception occurred inside the block
    """
    mid = model_id or _get_active_model_id()
    use_adapter = (mid == DEFAULT_MODEL_ID) and adapter_dir.is_dir()

    tok_source = str(adapter_dir) if use_adapter else mid
    tokeniser  = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        mid,
        quantization_config=bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, str(adapter_dir)) if use_adapter else base
    model.eval()

    try:
        yield model, tokeniser
    finally:
        del model
        clear_gpu()
