"""QLoRA fine-tune of Gemma 4 E4B with Unsloth.

Heavy imports (unsloth / torch / trl) are deferred into functions so the rest of the package
stays importable on a CPU-only box. Run as a module on a CUDA GPU:

    python -m src.train.train_qlora
"""
from __future__ import annotations

import json
import math

from src import config


def _load_dataset(tokenizer):
    from datasets import Dataset

    rows = [json.loads(l) for l in open(config.TRAIN_PATH, encoding="utf-8")]

    def _to_text(row):
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = Dataset.from_list(rows).map(_to_text, remove_columns=Dataset.from_list(rows).column_names)
    return ds


def _load_model():
    from unsloth import FastModel

    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.MODEL_ID,
            max_seq_length=config.TRAIN.max_seq_len,
            load_in_4bit=config.LOAD_IN_4BIT,
            full_finetuning=False,
            token=config.get_secret("HF_TOKEN"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {config.MODEL_ID} failed ({exc}); trying {config.MODEL_ID_FALLBACK}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.MODEL_ID_FALLBACK,
            max_seq_length=config.TRAIN.max_seq_len,
            load_in_4bit=config.LOAD_IN_4BIT,
            full_finetuning=False,
            token=config.get_secret("HF_TOKEN"),
        )

    model = FastModel.get_peft_model(
        model,
        r=config.TRAIN.lora_r,
        lora_alpha=config.TRAIN.lora_alpha,
        lora_dropout=config.TRAIN.lora_dropout,
        target_modules=list(config.TRAIN.target_modules),
        use_gradient_checkpointing="unsloth",
        random_state=config.TRAIN.seed,
    )
    return model, tokenizer


def train() -> str:
    config.seed_everything(config.TRAIN.seed)
    config.ensure_dirs()
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = _load_model()
    ds = _load_dataset(tokenizer)

    bs, ga = config.TRAIN.per_device_batch_size, config.TRAIN.grad_accumulation
    steps_per_epoch = max(1, math.ceil(len(ds) / (bs * ga)))
    target_steps = math.ceil(config.TRAIN.epochs * steps_per_epoch)
    max_steps = min(config.TRAIN.max_steps, target_steps)
    print(f"Training {len(ds)} examples | {steps_per_epoch} steps/epoch | max_steps={max_steps}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=config.TRAIN.max_seq_len,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=ga,
            warmup_ratio=config.TRAIN.warmup_ratio,
            max_steps=max_steps,
            learning_rate=config.TRAIN.learning_rate,
            logging_steps=config.TRAIN.logging_steps,
            optim="adamw_8bit",
            weight_decay=config.TRAIN.weight_decay,
            lr_scheduler_type="linear",
            seed=config.TRAIN.seed,
            output_dir=str(config.OUTPUTS_DIR / "trainer"),
            report_to="none",
        ),
    )

    # Train on the model's responses only when the helper supports this template.
    try:
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[info] train_on_responses_only skipped: {exc}")

    trainer.train()

    model.save_pretrained(str(config.ADAPTER_DIR))
    tokenizer.save_pretrained(str(config.ADAPTER_DIR))
    print("Saved adapter to", config.ADAPTER_DIR)
    _maybe_push_to_hub(model, tokenizer)
    return str(config.ADAPTER_DIR)


def _maybe_push_to_hub(model, tokenizer) -> None:
    repo = config.get_secret("HF_PUSH_REPO")
    token = config.get_secret("HF_TOKEN")
    if not (repo and token):
        return
    try:
        model.push_to_hub(repo, token=token)
        tokenizer.push_to_hub(repo, token=token)
        print("Pushed adapter to", repo)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] hub push failed: {exc}")


if __name__ == "__main__":
    train()
