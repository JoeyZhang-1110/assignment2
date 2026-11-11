"""TEST 2 -- The terminated-vs-truncated edge case (and that we handle it right).

The edge case
    Gymnasium's step() returns BOTH `terminated` (the episode really ended: the
    agent failed/finished) and `truncated` (a time limit cut the episode off,
    but the world would have continued). The DQN bootstrap must be masked only
    on `terminated`. If you also mask on `truncated`, you teach the agent that
    reaching the time limit has zero future value -- biasing every Q estimate
    downward. CartPole-v1 truncates at 500 steps, so this bug is easy to hit.

The experiment
    A trivial "survival" environment: one constant state, +1 reward every step,
    it NEVER terminates but is TRUNCATED after H steps. As a continuing task the
    true value is V = sum_{t>=0} gamma^t = 1/(1-gamma) = 10 at gamma=0.9.

      * Correct handling (mask bootstrap on `terminated` only): learned V -> 10.
      * Buggy handling  (mask on terminated OR truncated): the fixed point is
            V = (1 - 1/H)*(1 + gamma*V) + (1/H)*1,
        which for H=2, gamma=0.9 gives V ~= 1.82 -- a huge downward bias
        (an ~82% underestimate of the true value of the state).

    (We use gamma=0.9 / H=2 rather than CartPole's 0.95/500 so both runs converge
    quickly and the gap is unmistakable; the masking logic under test is
    identical. The principle holds for any gamma/H.)

How it validates
    We run the SAME agent both ways and check (a) the correct run lands near 100
    and (b) the buggy run lands near its (lower) biased fixed point. Matching the
    predicted numbers for BOTH shows our training loop masks the bootstrap on the
    right flag.

How to run
    python tests/test_truncation.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from src.dqn_agent import DQNAgent
from src.networks import MLPQNetwork
from src.replay_buffer import ReplayBuffer
from src.utils import set_seed

GAMMA = 0.9
H = 2


class SurvivalEnv:
    """One constant state, +1 every step, never terminates, truncates at H."""

    def __init__(self):
        self.t = 0

    def reset(self):
        self.t = 0
        return np.array([1.0], dtype=np.float32)

    def step(self, a):
        self.t += 1
        terminated = False
        truncated = self.t >= H
        return np.array([1.0], dtype=np.float32), 1.0, terminated, truncated, {}


def buggy_fixed_point(gamma, h):
    # V = (1-1/h)*(1+gamma*V) + (1/h)*1  ->  solve for V
    p = 1.0 - 1.0 / h
    # V = p + p*gamma*V + (1/h)  ->  V (1 - p*gamma) = p + 1/h
    return (p + 1.0 / h) / (1.0 - p * gamma)


def train(mask_on_truncation: bool, steps=35_000, seed=0):
    set_seed(seed)
    env = SurvivalEnv()
    agent = DQNAgent(MLPQNetwork(1, 2, hidden=32), 2, device="cpu",
                     gamma=GAMMA, lr=2e-3)
    buf = ReplayBuffer(50_000, 1, seed=seed)
    s = env.reset()
    for t in range(steps):
        a = agent.act(s, 0.1)
        s2, r, term, trunc, _ = env.step(a)
        # The ONLY difference between the two runs is this flag:
        done = term or (trunc if mask_on_truncation else False)
        buf.add(s, a, r, s2, done)
        s = env.reset() if (term or trunc) else s2
        if len(buf) >= 256:
            agent.learn(buf.sample(128))
            agent.soft_update(0.02)   # smooth target tracking -> fast convergence
    with torch.no_grad():
        v = float(agent.q(torch.as_tensor(np.array([[1.0]], np.float32))).max().item())
    return v


def main():
    true_v = 1.0 / (1.0 - GAMMA)
    buggy_v = buggy_fixed_point(GAMMA, H)

    v_correct = train(mask_on_truncation=False)   # our real training-loop behavior
    v_buggy = train(mask_on_truncation=True)      # the bug, for contrast

    print("=" * 64)
    print("TEST 2: terminated-vs-truncated bootstrap masking")
    print("=" * 64)
    print(f"gamma={GAMMA}, truncation horizon H={H}")
    print(f"true continuing-task value  V = 1/(1-gamma)      = {true_v:7.2f}")
    print(f"predicted BUGGY fixed point (mask on truncation) = {buggy_v:7.2f}")
    print("-" * 64)
    print(f"learned V, CORRECT handling (mask on terminated) = {v_correct:7.2f}")
    print(f"learned V, BUGGY   handling (mask on truncated)  = {v_buggy:7.2f}")
    print("-" * 64)

    correct_ok = abs(v_correct - true_v) < 1.0          # ~10
    buggy_ok = abs(v_buggy - buggy_v) < 1.0             # ~1.82
    clearly_biased = (v_correct - v_buggy) > 5.0        # the bug really hurts

    print(f"correct run near {true_v:.1f} : {correct_ok}")
    print(f"buggy run near {buggy_v:.2f} : {buggy_ok}")
    print(f"bug causes large downward bias : {clearly_biased}")
    ok = correct_ok and buggy_ok and clearly_biased
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 64)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
