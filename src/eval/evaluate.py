"""Before/after evaluation: base vs fine-tuned Gemma 4 E4B.

Loads the base model, generates on the held-out val set, scores with the same validators used
for data, then repeats with the LoRA adapter applied (models are loaded one at a time to fit
free-tier GPU memory). Writes a metrics table and a side-by-side sample gallery.

    python -m src.eval.evaluate
"""
from __future__ import annotations

import gc
import json

from src import config
from src.data.format_chat import to_prompt_messages
from src.data.validate import validate_example

MAX_NEW_TOKENS = 1024
GALLERY_N = 6  # examples shown side-by-side in the markdown report


def _read_val() -> list[dict]:
    return [json.loads(l) for l in open(config.VAL_PATH, encoding="utf-8")]


def _load(model_name: str):
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.TRAIN.max_seq_len,
        load_in_4bit=config.LOAD_IN_4BIT,
        token=config.get_secret("HF_TOKEN"),
    )
    FastModel.for_inference(model)
    return model, tokenizer


def _generate(model, tokenizer, example: dict) -> str:
    inputs = tokenizer.apply_chat_template(
        to_prompt_messages(example),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    out = model.generate(input_ids=inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    text = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
    return text.strip()


def _score(model, tokenizer, val: list[dict]) -> tuple[dict, list[str]]:
    by_task: dict[str, list[bool]] = {}
    outputs: list[str] = []
    for ex in val:
        text = _generate(model, tokenizer, ex)
        ok, _ = validate_example(ex["task_type"], text)
        by_task.setdefault(ex["task_type"], []).append(ok)
        outputs.append(text)
    rates = {t: round(sum(v) / len(v), 3) for t, v in by_task.items()}
    flat = [ok for v in by_task.values() for ok in v]
    rates["overall"] = round(sum(flat) / len(flat), 3)
    return rates, outputs


def _free(model) -> None:
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def evaluate() -> dict:
    config.ensure_dirs()
    val = _read_val()

    base_model, base_tok = _load(config.MODEL_ID)
    base_rates, base_out = _score(base_model, base_tok, val)
    _free(base_model)

    tuned_model, tuned_tok = _load(str(config.ADAPTER_DIR))
    tuned_rates, tuned_out = _score(tuned_model, tuned_tok, val)
    _free(tuned_model)

    metrics = {"base": base_rates, "fine_tuned": tuned_rates}
    (config.OUTPUTS_DIR / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_report(val, base_out, tuned_out, metrics)
    print("Eval metrics:", json.dumps(metrics, indent=2))
    return metrics


def _trunc(text: str, n: int = 600) -> str:
    return text if len(text) <= n else text[:n] + "\n… (truncated)"


def _write_report(val, base_out, tuned_out, metrics) -> None:
    lines = ["# Before / after evaluation", "", "## Valid-output rate", "",
             "| task | base | fine-tuned |", "| --- | --- | --- |"]
    tasks = sorted(set(metrics["base"]) | set(metrics["fine_tuned"]))
    for t in tasks:
        b = metrics["base"].get(t, "-")
        f = metrics["fine_tuned"].get(t, "-")
        lines.append(f"| {t} | {b} | {f} |")
    lines += ["", "## Sample gallery", ""]
    for ex, b, f in list(zip(val, base_out, tuned_out))[:GALLERY_N]:
        lines += [
            f"### [{ex['task_type']}] {_trunc(ex['prompt'], 200)}",
            "", "**Base:**", "```", _trunc(b), "```",
            "**Fine-tuned:**", "```", _trunc(f), "```", "",
        ]
    (config.OUTPUTS_DIR / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", config.OUTPUTS_DIR / "eval_report.md")


if __name__ == "__main__":
    evaluate()
