from __future__ import annotations

import random
from statistics import mean


def bootstrap_ci(
    values: list[float],
    n_samples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))

    means.sort()
    low_idx = int((alpha / 2) * (n_samples - 1))
    high_idx = int((1 - alpha / 2) * (n_samples - 1))
    return {
        "mean": float(mean(values)),
        "ci_low": float(means[low_idx]),
        "ci_high": float(means[high_idx]),
    }


def paired_permutation_test(
    a: list[float],
    b: list[float],
    n_trials: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    if len(a) != len(b) or not a:
        return {"diff_mean": 0.0, "p_value": 1.0}

    rng = random.Random(seed)
    observed = mean(a) - mean(b)
    more_extreme = 0
    diffs = [x - y for x, y in zip(a, b, strict=False)]

    for _ in range(n_trials):
        flips = [1 if rng.random() < 0.5 else -1 for _ in diffs]
        sim = mean([d * f for d, f in zip(diffs, flips, strict=False)])
        if abs(sim) >= abs(observed):
            more_extreme += 1

    p_value = (more_extreme + 1) / (n_trials + 1)
    return {"diff_mean": float(observed), "p_value": float(p_value)}
