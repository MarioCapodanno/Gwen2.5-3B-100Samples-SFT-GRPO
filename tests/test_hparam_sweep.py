import collections

from gwen25.pipelines.hparam_sweep import quick_sweep


def test_quick_sweep_selects_best():
    # Candidates and iteration order: for lr in lr_candidates: for wr in warmup_ratios
    lr_candidates = [1e-1, 1e-2]
    warmup_ratios = [0.03, 0.1]

    # We'll simulate train and validation losses for each candidate.
    # quick_sweep will call train_step n_steps times then val_step once per candidate.
    n_steps = 2

    # Create synthetic train losses (not used to pick best here) and val losses.
    # Order of candidates: (0.1,0.03), (0.1,0.1), (0.01,0.03), (0.01,0.1)
    train_losses_seq = [0.5, 0.45, 0.48, 0.44, 0.4, 0.39, 0.38, 0.37]
    # Choose validation losses so the best (lowest) is the last candidate (0.01, 0.1)
    val_losses_seq = [0.46, 0.45, 0.42, 0.35]

    train_losses = collections.deque(train_losses_seq)
    val_losses = collections.deque(val_losses_seq)


    def train_step():
        # consume and return next synthetic train loss
        return train_losses.popleft()


    def val_step():
        # return next synthetic validation loss for the candidate
        return val_losses.popleft()


    out = quick_sweep(train_step, val_step, lr_candidates, warmup_ratios, n_steps=n_steps)

    assert out["best_lr"] == 1e-2
    assert out["best_warmup_ratio"] == 0.1
    assert out["best_val_loss"] == 0.35
