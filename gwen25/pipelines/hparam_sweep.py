"""Light hyperparameter sweep harness.

This module provides a tiny, dependency-free harness to run very short
validation runs (a few batches / steps) to pick between a couple of
learning rates and warmup ratios. It's intended to be fast and conservative.

Usage: import and call `quick_sweep(train_step, val_step, candidates, n_steps=2)`
where `train_step` and `val_step` are callables implemented by the caller that
execute a single training or validation batch and return a scalar loss.

Note: This harness is intentionally minimal—integrating it into large
frameworks requires adapting the train_step/val_step wrappers.
"""

from typing import Callable, Dict, List, Tuple


def quick_sweep(
    train_step: Callable[[], float],
    val_step: Callable[[], float],
    lr_candidates: List[float],
    warmup_ratios: List[float],
    n_steps: int = 2,
) -> Dict[str, float]:
    """Run a very short sweep and return the best (lr, warmup_ratio).

    - train_step(): should perform one training step (forward/backward/opt.step)
      and return the training loss as float.
    - val_step(): should perform one validation forward and return validation loss.
    - lr_candidates/warmup_ratios: lists of values to try.
    - n_steps: number of training steps per candidate (small, e.g. 1..3).

    Returns a dict with best_lr, best_warmup_ratio and a small results table
    keyed by (lr,warmup) -> validation loss.
    """

    results: Dict[Tuple[float, float], float] = {}

    best = None
    best_loss = float("inf")

    for lr in lr_candidates:
        for wr in warmup_ratios:
            # The caller should set the optimizer lr and warmup scheduler externally
            # before calling train_step. We just run n_steps of train then a val step.
            train_losses = []
            for _ in range(n_steps):
                loss_val = train_step()
                train_losses.append(loss_val)

            val_loss = val_step()
            results[(lr, wr)] = val_loss

            if val_loss < best_loss:
                best_loss = val_loss
                best = (lr, wr)

    if best is None:
        raise RuntimeError("Sweep failed to produce a best configuration")

    return {
        "best_lr": best[0],
        "best_warmup_ratio": best[1],
        "best_val_loss": best_loss,
        "results": results,
    }


if __name__ == "__main__":
    print("This module is a lightweight helper and should be called from training code.")
