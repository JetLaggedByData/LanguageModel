"""
v2_finetuned/finetune.py
QLoRA fine-tuning of Qwen2.5-1.5B-Instruct on the SciFi corpus JSONL.

VRAM budget: ~5.5GB peak (4-bit model ~3GB + gradients + activations)
Safe on 8GB GPU with gradient_checkpointing + paged_adamw_8bit.

Run:
  python v2_finetuned/finetune.py
  python v2_finetuned/finetune.py --resume   # resume from last checkpoint
"""

import gc
import sys
import json
import argparse
import torch
import mlflow
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, str(Path(__file__).parent))
from model_config import (
    MODEL_ID, ADAPTER_DIR, TRAIN_JSONL, VAL_JSONL,
    MAX_LENGTH, PADDING_SIDE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS,
    BATCH_SIZE, GRAD_ACCUM, EPOCHS, LR,
    WARMUP_RATIO, LOGGING_STEPS, SAVE_STEPS, SAVE_TOTAL_LIMIT, WEIGHT_DECAY,
)


# ── GPU helpers ───────────────────────────────────────────────────────────

def clear_gpu() -> None:
    """Free all cached GPU memory. Call after any model unload."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def vram_used_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


# ── 4-bit quantisation config (mandatory for 8GB VRAM) ───────────────────

def bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# ── LoRA config (skill spec values) ──────────────────────────────────────

def lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ── Data loading ──────────────────────────────────────────────────────────

def load_jsonl_dataset(path: Path) -> Dataset:
    """Load JSONL into a HuggingFace Dataset."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return Dataset.from_list(records)


def format_prompt(sample: dict) -> str:
    """
    Format one sample into the instruction-tuning prompt template.
    Qwen2.5-Instruct uses its own chat template; we adapt to plain text
    so the same format works for the fallback GPT-2 model too.
    """
    return (
        f"### Instruction:\n{sample['instruction']}\n\n"
        f"### Input:\n{sample['input']}\n\n"
        f"### Response:\n{sample['output']}"
    )


# ── MLflow loss callback ──────────────────────────────────────────────────

class MLflowStepCallback(TrainerCallback):
    """Log train loss to MLflow at every logging step."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            mlflow.log_metric("train_loss", logs["loss"], step=step)
        if "eval_loss" in logs:
            mlflow.log_metric("eval_loss", logs["eval_loss"], step=step)
        if torch.cuda.is_available():
            mlflow.log_metric("vram_gb", vram_used_gb(), step=step)


# ── Training ──────────────────────────────────────────────────────────────

def build_training_args(resume: bool) -> TrainingArguments:
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    return SFTConfig(
        output_dir=str(ADAPTER_DIR),
        per_device_train_batch_size=BATCH_SIZE,       # VRAM hard limit
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,                  # VRAM hard limit
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="paged_adamw_8bit",                     # VRAM hard limit
        report_to="none",                             # MLflow handled manually
        dataloader_pin_memory=False,                  # saves RAM on Ubuntu
        resume_from_checkpoint=str(ADAPTER_DIR) if resume else None,
        max_length=MAX_LENGTH,
    )


def run_finetune(resume: bool = False) -> None:
    """Full QLoRA fine-tuning pipeline with MLflow tracking."""
    if not TRAIN_JSONL.exists():
        sys.exit(
            f"Training data not found: {TRAIN_JSONL}\n"
            "Run: python data/prepare_dataset.py"
        )

    print(f"Loading tokeniser: {MODEL_ID}")
    tokeniser = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokeniser.padding_side = PADDING_SIDE
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token

    print(f"Loading model in 4-bit: {MODEL_ID}")
    print(f"  VRAM before load: {vram_used_gb():.2f} GB")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config(),
        device_map="auto",                # CPU offload allowed via 32GB RAM
        trust_remote_code=True,
    )
    model.config.use_cache = False        # required for gradient_checkpointing
    print(f"  VRAM after load:  {vram_used_gb():.2f} GB")

    model = get_peft_model(model, lora_config())
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_ds = load_jsonl_dataset(TRAIN_JSONL)
    val_ds   = load_jsonl_dataset(VAL_JSONL)
    print(f"  Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples")

    training_args = build_training_args(resume)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokeniser,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=format_prompt,
        args=training_args,
        callbacks=[MLflowStepCallback()],
    )

    with mlflow.start_run(run_name="v2_qlora_finetune"):
        mlflow.log_params({
            "model_id":    MODEL_ID,
            "lora_r":      LORA_R,
            "lora_alpha":  LORA_ALPHA,
            "epochs":      EPOCHS,
            "lr":          LR,
            "batch_size":  BATCH_SIZE,
            "grad_accum":  GRAD_ACCUM,
            "max_length":  MAX_LENGTH,
            "train_size":  len(train_ds),
            "val_size":    len(val_ds),
        })

        print("\nStarting training...")
        trainer.train(resume_from_checkpoint=resume)

        print(f"\nSaving adapter weights → {ADAPTER_DIR}")
        trainer.save_model(str(ADAPTER_DIR))
        tokeniser.save_pretrained(str(ADAPTER_DIR))

        final_metrics = trainer.evaluate()
        mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()})
        print(f"Final eval metrics: {final_metrics}")

    clear_gpu()
    print(f"\n✅ Fine-tuning complete. Adapters saved to: {ADAPTER_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SciFi Forge V2 — QLoRA fine-tuning")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    run_finetune(resume=args.resume)
