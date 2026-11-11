"""TEST 4 -- The trained policy must beat a random policy (by a wide margin).

A model can look "trained" (loss goes down, Q rises) yet have learned nothing
useful about the task. The most basic task-level sanity check is: does acting
greedily under the learned Q do dramatically better than acting randomly on the
*real* environment? We make this quantitative with an effect size.

How it validates
    Roll out the trained CartPole policy (greedy) and a uniform-random policy for
    N episodes each, then compare. PASS requires the trained mean to exceed the
    random mean by many random-policy standard deviations (a huge effect), which
    no amount of lucky noise could produce.

How to run (needs a trained model)
    python -m src.train_cartpole            # if results/cartpole/cartpole_model.pt is missing
    python tests/test_random_baseline.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from src.dqn_agent import DQNAgent
from src.envs import make_cartpole
from src.networks import MLPQNetwork

MODEL = os.path.join(ROOT, "results/cartpole/cartpole_model.pt")
N = 100


def rollout(agent, epsilon, seed0):
    env = make_cartpole()
    out = []
    for i in range(N):
        s, _ = env.reset(seed=seed0 + i)
        s = s.astype(np.float32)
        tot = 0.0
        while True:
            a = agent.act(s, epsilon) if agent is not None else env.action_space.sample()
            s, r, term, trunc, _ = env.step(a)
            s = s.astype(np.float32)
            tot += r
            if term or trunc:
                break
        out.append(tot)
    env.close()
    return np.asarray(out)


def main():
    if not os.path.exists(MODEL):
        print(f"FAIL: no trained model at {MODEL}. Train first with "
              f"`python -m src.train_cartpole`."); sys.exit(2)

    agent = DQNAgent(MLPQNetwork(4, 2), 2, device="cpu", gamma=0.95)
    agent.load(MODEL)

    trained = rollout(agent, epsilon=0.0, seed0=20_000)
    random = rollout(None, epsilon=1.0, seed0=20_000)

    gap = trained.mean() - random.mean()
    effect = gap / (random.std() + 1e-8)

    print("=" * 64)
    print("TEST 4: trained policy vs random baseline (CartPole)")
    print("=" * 64)
    print(f"random  : mean {random.mean():6.1f}  std {random.std():5.1f}  "
          f"(min {random.min():.0f}, max {random.max():.0f})")
    print(f"trained : mean {trained.mean():6.1f}  std {trained.std():5.1f}  "
          f"(min {trained.min():.0f}, max {trained.max():.0f})")
    print(f"gap = {gap:.1f}  = {effect:.1f} random-policy std-devs")
    ok = effect > 10.0 and trained.mean() > 5 * random.mean()
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 64)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
