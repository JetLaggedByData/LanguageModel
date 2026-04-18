"""
v1_baseline/generate.py
Text generation — PyTorch port of original TF coursework generate_text().
Logic identical to notebook: categorical sampling scaled by temperature.

Run:
  python v1_baseline/generate.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import textwrap
import numpy as np
import torch
from pathlib import Path

from lstm_model import build_lstm, EMBEDDING_DIM, RNN_UNITS
from train      import load_and_preprocess, DATA_PATH, CHECKPOINT_DIR, DEVICE


# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT  = CHECKPOINT_DIR / "checkpt_best.pt"
DEFAULT_NUM_CHARS   = 1000
DEFAULT_TEMPERATURE = 0.5
DEFAULT_WRAP_WIDTH  = 80


def load_inference_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
) -> tuple:
    """Load trained weights for inference (batch_size=1)."""
    _, vocab, chartoindex, indextochar = load_and_preprocess(DATA_PATH)

    state = torch.load(checkpoint_path, map_location=DEVICE)
    model = build_lstm(vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    return model, chartoindex, indextochar


def generate_text(
    model:        object,
    chartoindex:  dict,
    indextochar:  np.ndarray,
    input_string: str,
    num:          int   = DEFAULT_NUM_CHARS,
    temperature:  float = DEFAULT_TEMPERATURE,
    wrap_width:   int   = DEFAULT_WRAP_WIDTH,
) -> str:
    """
    Generate `num` characters continuing from `input_string`.
    Identical sampling logic to original notebook generate_text().
    """
    model.reset_states()
    text_result = []

    # Warm up the hidden state on the input string
    input_indices = torch.tensor(
        [[chartoindex[c] for c in input_string]], dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():
        _, _ = model(input_indices)   # builds up hidden state

        for _ in range(num):
            # Feed only the last predicted character
            logits, _ = model(input_indices[:, -1:])
            logits     = logits[:, -1, :] / temperature   # (1, vocab_size)

            # Categorical sampling — matches tf.random.categorical
            probs      = torch.softmax(logits, dim=-1)
            predicted  = torch.multinomial(probs, num_samples=1).item()

            char = indextochar[predicted]
            text_result.append(char)

            input_indices = torch.tensor([[predicted]], dtype=torch.long).to(DEVICE)

            # Line break after complete word (matches original notebook)
            if char == " ":
                word = "".join(text_result).rsplit(" ", 1)[-1]
                if len(word) > 1 and word[-1] not in [".", "!", "?"]:
                    text_result.append("\n")

    generated = "".join(text_result)
    wrapped   = textwrap.fill(generated, wrap_width)
    return input_string + wrapped


if __name__ == "__main__":
    print("Loading model...")
    model, chartoindex, indextochar = load_inference_model()
    seed = input("Enter your text: ")
    print("\nPrediction:")
    print(generate_text(model, chartoindex, indextochar, seed, num=1000))
