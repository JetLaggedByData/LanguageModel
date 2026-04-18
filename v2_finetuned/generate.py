"""
v2_finetuned/generate.py
Text generation using the fine-tuned V2 model (base + LoRA adapters).

Supports:
  - Interactive CLI mode
  - Programmatic use: generate_text(prompt, ...)

Run:
  python v2_finetuned/generate.py
"""

import gc
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from model_config import (
    MODEL_ID, ADAPTER_DIR,
    GEN_MAX_TOKENS, GEN_TEMPERATURE, GEN_TOP_P, GEN_REP_PENALTY,
)


# ── GPU helpers ───────────────────────────────────────────────────────────

def clear_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ── Singleton model loader ────────────────────────────────────────────────

_model = None
_tokeniser = None


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _load_base_model(model_id: str) -> "AutoModelForCausalLM":
    """Try 4-bit quantized load; fall back to bfloat16 if bitsandbytes unavailable."""
    try:
        print("  Attempting 4-bit quantized load...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
        )
        print("  4-bit load succeeded.")
        return model
    except Exception as e:
        print(f"  4-bit load failed ({e}); falling back to bfloat16.")
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )


def load_model(
    adapter_dir: Path = ADAPTER_DIR,
    model_id: str = MODEL_ID,
) -> tuple:
    """
    Load base model + LoRA adapters. Singleton — safe to call multiple times.
    Returns (model, tokeniser).
    """
    global _model, _tokeniser
    if _model is not None:
        return _model, _tokeniser

    if not adapter_dir.exists():
        sys.exit(
            f"Adapter directory not found: {adapter_dir}\n"
            "Run: python v2_finetuned/finetune.py"
        )

    print(f"Loading tokeniser from {adapter_dir}")
    _tokeniser = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if _tokeniser.pad_token is None:
        _tokeniser.pad_token = _tokeniser.eos_token

    print(f"Loading base model: {model_id}")
    base = _load_base_model(model_id)

    print(f"Merging LoRA adapters from {adapter_dir}")
    _model = PeftModel.from_pretrained(base, str(adapter_dir))
    _model.eval()

    print("Model ready.")
    return _model, _tokeniser


def unload_model() -> None:
    """Explicitly release model from VRAM. Call before loading another model."""
    global _model, _tokeniser
    _model = None
    _tokeniser = None
    clear_gpu()


# ── Prompt formatting ─────────────────────────────────────────────────────

def build_prompt(user_text: str) -> str:
    """Wrap user input in the same template used during fine-tuning."""
    return (
        f"### Instruction:\nContinue this science fiction story:\n\n"
        f"### Input:\n{user_text}\n\n"
        f"### Response:\n"
    )


# ── Generation ────────────────────────────────────────────────────────────

def generate_text(
    prompt: str,
    max_new_tokens: int = GEN_MAX_TOKENS,
    temperature: float = GEN_TEMPERATURE,
    top_p: float = GEN_TOP_P,
    repetition_penalty: float = GEN_REP_PENALTY,
    adapter_dir: Path = ADAPTER_DIR,
) -> str:
    """
    Generate a SciFi continuation from a seed prompt.

    Args:
        prompt:             Seed text (raw user input, no template needed)
        max_new_tokens:     Tokens to generate (default 300)
        temperature:        Sampling temperature (0.1=focused, 1.0=creative)
        top_p:              Nucleus sampling cutoff
        repetition_penalty: >1.0 penalises repeated phrases
        adapter_dir:        Path to saved LoRA adapters

    Returns:
        Generated continuation string (prompt NOT included).
    """
    model, tokeniser = load_model(adapter_dir)

    full_prompt = build_prompt(prompt)
    inputs = tokeniser(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokeniser.eos_token_id,
            eos_token_id=tokeniser.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][prompt_len:]
    return tokeniser.decode(new_tokens, skip_special_tokens=True).strip()


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SciFi Forge — V2 Generator (Qwen2.5-1.5B QLoRA)\n")
    seed = input("Enter your seed text: ").strip()
    if not seed:
        sys.exit("No input provided.")

    print("\nGenerating...\n")
    result = generate_text(seed)
    print("─" * 70)
    print(result)
    print("─" * 70)
