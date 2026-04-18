"""
v1_baseline/train.py
Training script — PyTorch port of original TF coursework.
Hyperparameters unchanged: same sequence length, batch size, learning rate.

Run:
  python v1_baseline/train.py
  python v1_baseline/train.py --resume   # continue from last checkpoint
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from lstm_model import build_lstm, loss_fn, VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS


# ── Hyperparameters (original notebook values — do not change) ────────────
SEQUENCE_LENGTH    = 80
BATCH_SIZE         = 128
LEARNING_RATE      = 0.001
EPOCHS             = 15
MID_EPOCH_SAVE_N   = 50_000   # save a mid-epoch checkpoint every N batches
DATA_PATH          = Path("../data/raw/internet_archive_scifi_v3.txt")
CHECKPOINT_DIR     = Path("checkpoints/lstm_checkpoints")
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────

class CharDataset(Dataset):
    """Sliding-window character sequence dataset."""

    def __init__(self, int_text: np.ndarray, seq_len: int) -> None:
        self.data    = torch.tensor(int_text, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx: idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]   # input, target


# ── Data loading ──────────────────────────────────────────────────────────

def load_and_preprocess(
    data_path: Path = DATA_PATH,
) -> tuple[np.ndarray, list, dict, np.ndarray]:
    """
    Load corpus, strip header, collapse double spaces.
    Identical preprocessing to original notebook.
    """
    text        = data_path.read_text(encoding="utf-8")
    text        = text[580:149_322_961]
    text        = text.replace("  ", " ")
    vocab       = sorted(set(text))
    chartoindex = {v: i for i, v in enumerate(vocab)}
    indextochar = np.array(vocab)
    int_text    = np.array([chartoindex[c] for c in text])

    print(f"Characters: {len(text):,} | Unique: {len(vocab)}")
    return int_text, vocab, chartoindex, indextochar


# ── Training ──────────────────────────────────────────────────────────────

def train(resume: bool = False) -> None:
    """Full training run with early stopping and checkpoint saving."""
    print(f"Device: {DEVICE}")

    int_text, vocab, _, _ = load_and_preprocess()
    dataset    = CharDataset(int_text, SEQUENCE_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model     = build_lstm(vocab_size=len(vocab)).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=1, factor=0.5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    start_epoch  = 1
    best_loss    = float("inf")
    patience_ctr = 0
    PATIENCE     = 2   # matches original EarlyStopping(patience=2)

    # Resume from latest checkpoint if requested
    resume_batch = 0   # batch index to skip to when resuming mid-epoch
    if resume:
        mid = CHECKPOINT_DIR / "checkpt_mid.pt"
        epoch_ckpts = sorted(CHECKPOINT_DIR.glob("checkpt_[0-9]*.pt"))
        if mid.exists():
            state = torch.load(mid, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state["model"])
            optimiser.load_state_dict(state["optimiser"])
            if "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            start_epoch   = state["epoch"]
            resume_batch  = state["batch_idx"] + 1
            best_loss     = state["best_loss"]
            print(f"Resumed from mid-epoch checkpoint "
                  f"(epoch {start_epoch}, batch {state['batch_idx']})")
        elif epoch_ckpts:
            latest = epoch_ckpts[-1]
            state  = torch.load(latest, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state["model"])
            optimiser.load_state_dict(state["optimiser"])
            if "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            start_epoch = state["epoch"] + 1
            best_loss   = state["best_loss"]
            print(f"Resumed from {latest} (epoch {state['epoch']})")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
             # Skip batches already processed when resuming mid-epoch
            if epoch == start_epoch and batch_idx < resume_batch:
                continue
            resume_batch = 0   # only skip on the first (resumed) epoch

            inputs  = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            model.reset_states()
            optimiser.zero_grad()

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits, _ = model(inputs)
                # Flatten for cross_entropy: (B*T, vocab) vs (B*T,)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()

            epoch_loss += loss.item()
            n_batches  += 1

            if batch_idx % 500 == 0:
                print(f"  Epoch {epoch} | batch {batch_idx}/{len(dataloader)} "
                      f"| loss {loss.item():.4f}")

            # Mid-epoch checkpoint — survives crashes within an epoch
            if batch_idx > 0 and batch_idx % MID_EPOCH_SAVE_N == 0:
                torch.save({
                    "epoch":     epoch,
                    "batch_idx": batch_idx,
                    "model":     model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scaler":    scaler.state_dict(),
                    "best_loss": best_loss,
                }, CHECKPOINT_DIR / "checkpt_mid.pt")

        avg_loss = epoch_loss / n_batches
        scheduler.step(avg_loss)
        print(f"Epoch {epoch}/{EPOCHS} — avg loss: {avg_loss:.4f}")

        # Save checkpoint every epoch (remove mid-epoch checkpoint on success)
        ckpt_path = CHECKPOINT_DIR / f"checkpt_{epoch}.pt"
        torch.save({
            "epoch":      epoch,
            "loss":       avg_loss,   # epoch avg loss — used by the plot cell
            "model":      model.state_dict(),
            "optimiser":  optimiser.state_dict(),
            "scaler":     scaler.state_dict(),
            "best_loss":  best_loss,
            "vocab_size": len(vocab),
        }, ckpt_path)
        mid = CHECKPOINT_DIR / "checkpt_mid.pt"
        if mid.exists():
            mid.unlink()   # epoch complete — mid-epoch checkpoint no longer needed

        # Early stopping (matches original patience=2)
        if avg_loss < best_loss:
            best_loss    = avg_loss
            patience_ctr = 0
            # Also save a named "best" checkpoint
            torch.save(torch.load(ckpt_path, weights_only=False), CHECKPOINT_DIR / "checkpt_best.pt")
            print(f"  ✅ New best checkpoint saved.")
        else:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"\nTraining complete. Checkpoints in: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)
