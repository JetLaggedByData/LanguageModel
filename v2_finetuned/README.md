# V2 — QLoRA Fine-Tuned LLM

Qwen2.5-1.5B-Instruct fine-tuned on 50k science fiction instruction pairs using 4-bit QLoRA.
Runs on a single 8GB GPU. The resulting LoRA adapters are used by the V3 agentic pipeline.

## Files

| File | Purpose |
|---|---|
| `finetune.py` | 4-bit NF4 QLoRA training (3 epochs, ~4–6h on 8GB GPU) |
| `generate.py` | Inference with LoRA adapters loaded over the frozen base |
| `evaluate.py` | Word-level perplexity, BLEU-4, genre consistency score |
| `model_config.py` | LoRA config: r=16, α=32, target q_proj & v_proj |
| `adapters/` | LoRA weights — gitignored, push to HF Hub separately |

## Key numbers
- Base: Qwen2.5-1.5B-Instruct (frozen, 4-bit NF4)
- Trainable params: 2.75M (0.18% of total)
- Peak VRAM: ~4 GB · Optimiser: paged_adamw_8bit

## Run
```bash
python data/prepare_dataset.py    # build 50k instruction JSONL
python v2_finetuned/finetune.py   # fine-tune (~4–6h)
python v2_finetuned/generate.py   # test generation
python v2_finetuned/evaluate.py   # benchmark vs V1
```
