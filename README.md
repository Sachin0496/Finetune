# Finetune — Gemma 4 E4B agent distillation (QLoRA, Unsloth, Kaggle free tier)

QLoRA fine-tune of [`google/gemma-4-E4B`](https://huggingface.co/google/gemma-4-E4B) with
[Unsloth](https://github.com/unslothai/unsloth), runnable end-to-end on a **free Kaggle GPU**
(T4 16GB / P100 16GB). The model is taught three behaviors of a slide-deck building agent
from a synthetically bootstrapped, deterministically validated dataset:

1. **Deck-plan JSON** — a topic → a schema-valid deck plan.
2. **Animation HTML** — a slide spec → self-contained animation HTML.
3. **Tool calls** — a build state → the next tool call the orchestrator should emit.

No external sandbox or browser is required: all data validation is pure-Python
(JSON-schema + HTML structural parsing).

## Layout

```
src/
  config.py            # paths, model id, hyper-params, free-tier budgets, seeding
  tools_schema.py      # deck-plan schema + agent tool argument schemas
  data/
    seeds.py           # procedural generators + committed seed exemplars
    teacher.py         # optional free teacher LLM (falls back to seeds offline)
    validate.py        # jsonschema + HTML structural validators
    format_chat.py     # render examples into the Gemma 4 chat template
    build_dataset.py   # generate -> author -> validate -> dedupe -> split -> JSONL
  train/train_qlora.py # Unsloth QLoRA SFT; saves the adapter
  eval/evaluate.py     # base vs fine-tuned valid-rates + a before/after gallery
Finetune               # Kaggle notebook entry point (clone + install + run)
```

## Run on Kaggle

The `Finetune` notebook is the entry point. It is a thin orchestrator: it installs deps,
clones this repo into the session (so `src/` is available even though Kaggle only syncs the
notebook itself), then builds the dataset, fine-tunes, and evaluates.

1. Open the notebook on Kaggle. In **Settings**, set **Accelerator = GPU** and **Internet = On**.
2. (Optional) **Add-ons → Secrets**: `HF_TOKEN` (if the base weights are gated) and
   `GOOGLE_API_KEY` (to let the free teacher LLM augment the dataset). Without them the run
   still works from the committed seed exemplars.
3. **Run All**. The adapter and an evaluation gallery are written to `/kaggle/working`
   (persisted on *Save & Run All*).

The in-notebook `git clone` expects this repo to be **public**. If it is private, attach the
repo as a Kaggle Dataset instead — the clone step is skipped automatically when `src/` is
already importable.

### Push/pull via the Kaggle CLI (optional)

`kernel-metadata.json` declares the kernel (GPU + internet on) for the
[`kaggle kernels`](https://github.com/Kaggle/kaggle-api) flow. Set the `id` field to
`<your-kaggle-username>/finetune-gemma4-e4b-qlora`, then:

```bash
kaggle kernels push      # upload + run
kaggle kernels pull <your-kaggle-username>/finetune-gemma4-e4b-qlora
```

## Run locally

```bash
pip install -r requirements.txt
python -m src.data.build_dataset --smoke      # tiny dataset + validator self-check
python -m src.train.train_qlora               # needs a CUDA GPU
python -m src.eval.evaluate                   # base vs fine-tuned report
```

## Reproducibility

A single `SEED` drives every generator; the teacher temperature is pinned; the dataset is
cached and regenerable; dataset size, training steps, and sequence length are hard-capped in
`src/config.py` to stay inside free-tier limits.
