"""
v3_agentic/agents/model_loader.py
Shared LoRA model context manager used by all four agent nodes.

Eliminates the ~30-line load/unload boilerplate that was copy-pasted
across planner, writer, critic, and editor. The context manager guarantees
VRAM is released even when the node raises, without requiring explicit
finally blocks in each agent.
"""

import torch
from contextlib import contextmanager
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from agents.utils import bnb_config, clear_gpu


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
_ROOT            = Path(__file__).resolve().parent.parent.parent
ADAPTER_DIR      = _ROOT / "v2_finetuned/adapters"

# Module-level override set by the UI before the pipeline runs.
# Agents call load_agent_model() with no args and pick this up automatically.
_active_model_id: str = DEFAULT_MODEL_ID


def set_agent_model(model_id: str) -> None:
    """Called by the Streamlit UI (or CLI) to choose which model powers V3 agents."""
    global _active_model_id
    _active_model_id = model_id


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
    mid = model_id or _active_model_id
    use_adapter = (mid == DEFAULT_MODEL_ID) and adapter_dir.is_dir()

    tok_source = str(adapter_dir) if use_adapter else mid
    tokeniser  = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        mid,
        quantization_config=bnb_config(),
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, str(adapter_dir)) if use_adapter else base
    model.eval()

    try:
        yield model, tokeniser
    finally:
        del model
        clear_gpu()
