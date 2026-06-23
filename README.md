# Finetune — Gemma 4 E4B agent distillation (QLoRA · Unsloth)

A single self-contained Kaggle notebook that QLoRA-fine-tunes
[`unsloth/gemma-4-E4B-it`](https://huggingface.co/unsloth/gemma-4-E4B-it) (ungated, 4-bit on
load) on a **free GPU**. The model is taught three behaviors of a slide-deck building agent
from a synthetically generated, deterministically validated dataset:

1. **Deck-plan JSON** — a topic → a schema-valid deck plan.
2. **Animation HTML** — a slide spec → self-contained animation HTML.
3. **Tool calls** — a build state → the next tool call the orchestrator should emit.

All data validation is pure-Python (JSON-schema + HTML structural parsing) — no sandbox or
browser required.

## Run it

Everything lives in the `Finetune` notebook. **Just press *Run All*** — GPU and Internet are
preset in the notebook settings, and no secrets or external files are needed. The notebook:

1. installs the stack (Unsloth, TRL, …),
2. builds and validates the dataset,
3. runs QLoRA,
4. prints a **before/after** valid-output comparison and a sample gallery.

The adapter and evaluation report are written to `/kaggle/working` (persisted on
*Save & Run All*).

_Optional:_ add an `Add-ons → Secrets` value `GOOGLE_API_KEY` to let a free teacher model
enrich the dataset. It runs fully offline without it.

## Push/pull via the Kaggle CLI (optional)

`kernel-metadata.json` declares the kernel (GPU + internet on) for the
[`kaggle kernels`](https://github.com/Kaggle/kaggle-api) flow:

```bash
kaggle kernels push                                    # upload
kaggle kernels pull sachin0496/finetune-gemma4-e4b-qlora
```

## Reproducibility

A single seed drives every generator; the dataset is regenerable; dataset size, training
steps, and sequence length are capped to stay within free-tier limits. `requirements.txt`
documents the dependencies the notebook installs.
