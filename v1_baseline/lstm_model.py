"""
v1_baseline/lstm_model.py
LSTM model definition — PyTorch port of the original TensorFlow coursework.
Architecture is identical: Embedding(75) -> LSTM(1024) -> Linear(75)

Ported to PyTorch because the original TF build has no GPU support on
Python 3.12 + CUDA 13. The architecture, hyperparameters, and benchmark
metrics are preserved exactly — only the framework changed.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn


# Vocabulary constants (fixed from original training corpus)
VOCAB_SIZE    = 75
EMBEDDING_DIM = 256
RNN_UNITS     = 1024


class CharLSTM(nn.Module):
    """
    Character-level LSTM — exact port of the original Keras model.

    Keras layout:  Embedding → LSTM(stateful, return_sequences) → Dense
    PyTorch layout: Embedding → LSTM(batch_first)               → Linear

    Stateful behaviour is replicated by passing hidden state explicitly
    between generate() calls via reset_states() / forward().
    """

    def __init__(
        self,
        vocab_size:    int = VOCAB_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
        rnn_units:     int = RNN_UNITS,
    ) -> None:
        super().__init__()
        self.rnn_units = rnn_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm      = nn.LSTM(embedding_dim, rnn_units, batch_first=True)
        self.linear    = nn.Linear(rnn_units, vocab_size)

        # Hidden state held between calls (replicates Keras stateful=True)
        self._hidden: tuple | None = None

    def reset_states(self) -> None:
        """Zero the hidden + cell state — call before each new generation."""
        self._hidden = None

    def forward(
        self,
        x:      torch.Tensor,           # (batch, seq_len) int indices
        hidden: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            x:      Integer token indices (batch, seq_len)
            hidden: Optional (h, c) from previous step for stateful inference

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: (h, c) to pass into next step
        """
        use_hidden = hidden if hidden is not None else self._hidden
        emb  = self.embedding(x)                       # (B, T, embed_dim)
        out, new_hidden = self.lstm(emb, use_hidden)   # (B, T, rnn_units)
        logits = self.linear(out)                      # (B, T, vocab_size)
        self._hidden = new_hidden
        return logits, new_hidden


def build_lstm(
    vocab_size:    int = VOCAB_SIZE,
    embedding_dim: int = EMBEDDING_DIM,
    rnn_units:     int = RNN_UNITS,
    batch_size:    int = 1,            # kept for API compatibility with old callers
) -> CharLSTM:
    """
    Build and return a CharLSTM model.
    batch_size param is accepted but unused — PyTorch is batch-size agnostic.
    """
    return CharLSTM(vocab_size, embedding_dim, rnn_units)


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss matching the original Keras sparse_categorical_crossentropy.

    Args:
        logits:  (batch * seq_len, vocab_size)  — must be pre-flattened
        targets: (batch * seq_len,)              — integer class indices
    """
    return nn.functional.cross_entropy(logits, targets)
