"""Central configuration: paths, model ids, hyper-parameters, and free-tier budgets.

Everything tunable lives here so the notebook and the CLI scripts stay thin and a run
is reproducible from a single fixed seed.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths. On Kaggle, /kaggle/working is the only directory persisted on "Save & Run All".
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_output_root() -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working
    return REPO_ROOT


OUTPUT_ROOT = Path(os.environ.get("FT_OUTPUT_ROOT", _default_output_root()))
DATA_DIR = OUTPUT_ROOT / "data"
OUTPUTS_DIR = OUTPUT_ROOT / "outputs"
ADAPTER_DIR = OUTPUTS_DIR / "adapter"
DATASET_PATH = DATA_DIR / "dataset.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"

SEED = 42

# --------------------------------------------------------------------------------------
# Model. Prefer the Unsloth 4-bit mirror; fall back to the upstream weights + load_in_4bit.
# --------------------------------------------------------------------------------------
MODEL_ID = os.environ.get("FT_MODEL_ID", "unsloth/gemma-4-E4B-it")
MODEL_ID_FALLBACK = "google/gemma-4-E4B"
LOAD_IN_4BIT = True


@dataclass
class TrainConfig:
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # Sequence / batch. E4B is tighter on 16GB than E2B: drop max_seq_len first if it OOMs.
    max_seq_len: int = 2048
    per_device_batch_size: int = 1
    grad_accumulation: int = 8
    # Schedule (free-tier budget)
    epochs: float = 2.0
    max_steps: int = 200          # hard cap; whichever of epochs/max_steps is hit first
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 5
    seed: int = SEED


@dataclass
class DataConfig:
    # Final dataset budget, balanced across the three task types.
    n_train: int = 300
    n_val: int = 60
    # Smoke mode produces a tiny dataset for fast end-to-end checks.
    smoke_n_train: int = 12
    smoke_n_val: int = 6
    seed: int = SEED
    # Mixing weights across task types (schema / html / tool_call).
    task_mix: dict[str, float] = field(
        default_factory=lambda: {"schema": 0.34, "html": 0.33, "tool_call": 0.33}
    )


TRAIN = TrainConfig()
DATA = DataConfig()


def seed_everything(seed: int = SEED) -> None:
    """Seed python / numpy / torch when available. Safe to call before heavy imports."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_secret(name: str) -> str | None:
    """Read a secret from the environment or Kaggle Secrets, returning None if absent."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def ensure_dirs() -> None:
    for d in (DATA_DIR, OUTPUTS_DIR, ADAPTER_DIR):
        d.mkdir(parents=True, exist_ok=True)
