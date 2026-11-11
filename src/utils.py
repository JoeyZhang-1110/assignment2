"""Small, dependency-light helpers shared across the project.

Everything here is deterministic given a seed so that training runs and the
validation tests are reproducible (see tests/ for why that matters).
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch (CPU + CUDA) for reproducibility.

    Note we do NOT force deterministic cuDNN kernels because that makes the
    CNN training a lot slower; for the near-deterministic validation tests we
    use CPU-friendly tiny networks where kernel non-determinism is irrelevant.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(x, window: int = 100):
    """Trailing moving average of a 1-D sequence.

    For index i the value is the mean of x[max(0, i-window+1) .. i], so the
    curve is defined from the very first episode (no NaN warm-up gap). This is
    exactly the "moving average of the last 100 episodes" the assignment asks
    to overlay on the reward curve.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    out = np.empty_like(x)
    cumsum = np.cumsum(x)
    for i in range(x.size):
        lo = max(0, i - window + 1)
        total = cumsum[i] - (cumsum[lo - 1] if lo > 0 else 0.0)
        out[i] = total / (i - lo + 1)
    return out


@dataclass
class LinearSchedule:
    """Linear anneal from `start` to `end` over `duration` calls/steps.

    Used for the epsilon (exploration) schedule. Clamped at `end` afterwards so
    there is always a small floor of exploration during training.
    """
    start: float
    end: float
    duration: int

    def value(self, step: int) -> float:
        if self.duration <= 0:
            return self.end
        frac = min(max(step / self.duration, 0.0), 1.0)
        return self.start + frac * (self.end - self.start)


def theoretical_max_q(gamma: float, horizon: int, reward_per_step: float = 1.0) -> float:
    """Upper bound on the optimal Q-value for a constant-reward task.

    For CartPole every surviving step yields +1, so an optimal agent that keeps
    the pole up for `horizon` more steps has

        Q* = sum_{t=0}^{horizon-1} gamma^t * reward_per_step.

    With gamma=0.95 and horizon=500 this is ~20. Learned max-Q estimates should
    approach this ceiling from below; if they blow past it, the network is
    overestimating (a known DQN failure mode) -- we use this as a sanity bound.
    """
    if gamma >= 1.0:
        return reward_per_step * horizon
    geo = (1.0 - gamma ** horizon) / (1.0 - gamma)
    return reward_per_step * geo


class Logger:
    """Collects per-episode metrics and dumps them to a JSON file.

    Keeping the log as plain JSON (not a pickle) means plots.py and the tests
    can be re-run offline without re-training, and the numbers are inspectable.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.data: dict[str, list] = {
            "episode": [],
            "reward": [],
            "length": [],
            "max_q": [],       # max predicted Q at the episode's start state
            "epsilon": [],
            "loss": [],
            "total_steps": [],
        }

    def log(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.data.setdefault(k, []).append(v)

    def save(self, extra: dict | None = None) -> None:
        payload = dict(self.data)
        if extra:
            payload["meta"] = extra
        with open(self.path, "w") as f:
            json.dump(payload, f)

    @staticmethod
    def load(path: str) -> dict:
        with open(path) as f:
            return json.load(f)
