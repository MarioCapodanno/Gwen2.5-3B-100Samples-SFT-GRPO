"""Small utility to auto-adjust training hyperparameters based on dataset size.

This module provides a conservative rule-based policy to adapt learning rate,
effective batch size (via gradient accumulation), and warmup steps when the
number of training samples grows (e.g. > 500). It's intentionally simple and
deterministic so it can be imported into notebooks or training scripts.

References / rationale:
- Pareja et al (2024) "Unveiling the Secret Recipe: A Guide For Supervised Fine-Tuning Small LLMs"
  (recommended lowering LR when training on larger datasets / longer runs)
- Continuum Labs training guide: "Rethinking learning rate tuning in the era of
  language models" (practical notes on reducing LR and increasing warmup/batch)
- Hugging Face / TRL and PEFT conceptual guidance about effective batch size
  and gradient accumulation for memory-limited LoRA training.

The rules below are conservative defaults. They are not a full hyperparameter
search: for best results run a small sweep (learning rate, warmup_ratio,
gradient_accumulation_steps) using a validation split.
"""

from math import ceil
from typing import Dict, Optional


def recommend_hyperparams(
    dataset_size: int,
    base_lr: float = 2e-4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 2,
    num_train_epochs: int = 2,
    base_warmup_steps: int = 5,
    target_effective_batch: int = 8,
) -> Dict[str, Optional[float]]:
    """Return a recommended training config tuned for dataset_size.

    Inputs:
    - dataset_size: integer count of training samples (approximate is fine)
    - base_lr, batch sizes, epochs: current/default settings
    - target_effective_batch: desired effective batch size (accum * per_device)

    Outputs (dict): keys include 'learning_rate', 'per_device_train_batch_size',
    'gradient_accumulation_steps', 'warmup_steps', 'num_train_epochs', plus a
    short 'notes' string describing what was changed.

    Strategy used (simple, conservative):
    - Reduce learning rate as dataset_size grows to avoid instability when
      training longer on more data. We cap the reduction to an absolute floor.
    - Increase effective batch size (via gradient accumulation) to stabilize
      updates for larger datasets when GPU memory is constrained.
    - Increase warmup_steps as a fraction of training steps when dataset grows.

    These heuristics are based on practical guidance from the literature and
    practitioner notes (see module docstring). They should be used as a
    starting point and validated with a small run or sweep.
    """

    if dataset_size <= 0:
        raise ValueError("dataset_size must be > 0")

    cfg = {
        "learning_rate": float(base_lr),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "warmup_steps": int(base_warmup_steps),
        "num_train_epochs": int(num_train_epochs),
        "notes": "no change",
    }

    # conservative floor for lr for longer runs / larger data
    lr_floor = 1e-6

    # if dataset is small (<= 100) keep defaults
    if dataset_size <= 100:
        cfg["notes"] = "dataset small: keep base hyperparameters"
        return cfg

    # For medium / larger datasets apply adjustments
    # Reduce LR proportionally but conservatively: lr = base_lr * min(1, 100/ds)
    scale = min(1.0, 100.0 / float(dataset_size))
    recommended_lr = max(lr_floor, float(base_lr) * scale)

    # For datasets >= 500 prefer a default LR around 1e-4 (practical guide)
    if dataset_size >= 500:
        recommended_lr = min(recommended_lr, 1e-4)

    # Increase effective batch size by moving toward target_effective_batch
    cur_effective = per_device_train_batch_size * gradient_accumulation_steps
    if cur_effective < target_effective_batch:
        new_grad_accum = ceil(target_effective_batch / per_device_train_batch_size)
    else:
        new_grad_accum = gradient_accumulation_steps

    # Warmup: use a small fraction of the (estimated) steps; we do a rough
    # estimate of steps = dataset_size / effective_batch * epochs
    est_effective_batch = per_device_train_batch_size * new_grad_accum
    est_steps = max(1, int(dataset_size / max(1, est_effective_batch) * num_train_epochs))
    # use warmup ratio between 0.01 and 0.1 depending on dataset size
    warmup_ratio = 0.03 if dataset_size < 1000 else 0.1
    recommended_warmup = max(base_warmup_steps, int(est_steps * warmup_ratio))

    cfg.update(
        {
            "learning_rate": float(recommended_lr),
            "gradient_accumulation_steps": int(new_grad_accum),
            "warmup_steps": int(recommended_warmup),
            "notes": "auto-adjusted for dataset_size",
        }
    )

    return cfg


if __name__ == "__main__":
    # quick manual smoke checks when invoked directly
    for ds in [50, 120, 500, 2000]:
        cfg = recommend_hyperparams(ds)
        print(f"dataset={ds:5d} -> lr={cfg['learning_rate']:g}, accum={cfg['gradient_accumulation_steps']}, warmup={cfg['warmup_steps']}, notes={cfg['notes']}")
