"""
v3_agentic/agents/utils.py
Shared utilities for all four agent modules.

Centralises GPU helpers, BitsAndBytes config, and JSON extraction so
a single fix propagates everywhere rather than requiring changes in
four separate files.
"""

import gc
import re
import torch
from transformers import BitsAndBytesConfig


# ── GPU helpers ───────────────────────────────────────────────────────────

def clear_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


# ── Text post-processing ──────────────────────────────────────────────────

def trim_to_sentence(text: str) -> str:
    """
    Trim generated text to end at the last complete sentence.

    When max_new_tokens is reached the model stops mid-word/mid-clause.
    This finds the last sentence-ending punctuation (.  !  ?) and discards
    the incomplete fragment after it, so every chapter ends cleanly.
    """
    import re
    text = text.strip()
    if not text:
        return text
    if text[-1] in ".!?\"'":
        return text
    matches = list(re.finditer(r'[.!?]["\']?', text))
    if matches:
        return text[:matches[-1].end()].strip()
    return text


# ── JSON extraction ───────────────────────────────────────────────────────

def extract_json(raw: str) -> str:
    """
    Strip markdown fences and extract the first {...} block from raw model output.
    Handles prose before/after the JSON and unclosed fences.
    """
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start = raw.find("{")
    if start == -1:
        return raw
    depth, end = 0, start
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0:
            end = i
            break
    return raw[start: end + 1]
