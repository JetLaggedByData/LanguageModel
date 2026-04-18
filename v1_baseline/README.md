# V1 — Character-Level LSTM

Baseline model trained from scratch on 149M characters of science fiction text.
Serves as the benchmark anchor for V2 and V3 comparisons.

## Files

| File | Purpose |
|---|---|
| `lstm_model.py` | Embedding → LSTM(1024) → Dense architecture |
| `train.py` | Training loop with AMP and gradient clipping |
| `generate.py` | Temperature-sampled character generation |
| `evaluate.py` | Perplexity, bits/char, BLEU-2, inference speed |
| `org/` | Original TensorFlow/Keras coursework notebook (preserved as-is) |

## Key numbers
- Vocab: 75 characters · Corpus: 149M chars · Device: GPU (Colab)
- Checkpoint: `tests/checkpoints/lstm_checkpoints/checkpt_best.pt`

## Run
```bash
python v1_baseline/train.py       # train (checkpoint already included)
python v1_baseline/generate.py    # generate text
python v1_baseline/evaluate.py    # benchmark metrics
```
