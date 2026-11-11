"""TEST 5 -- Step-by-step trace of the first 10 steps (states/values/reward/policy).

The handout explicitly suggests: "isolate the first 10 steps and observe how the
states, values, rewards, and policy interact at each step." This test does that
for the trained CartPole policy and checks three interpretable properties that
must hold if the agent is behaving as designed.

Checks
    (a) policy consistency : the action taken equals argmax_a Q(s,a) at every step
        (confirms act() and the Q-values agree -- no indexing/argmax bug).
    (b) value calibration  : from these near-upright start states the greedy
        policy survives ~500 steps, so max-Q should sit just under the gamma=0.95
        ceiling of ~20 (we require 15 <= maxQ <= 20.5). A value far outside that
        band would signal under-training or over-estimation.
    (c) reactivity         : over a longer greedy episode the chosen action is
        positively correlated with the pole's "falling-right" indicator
        (angle + 0.5*angular_velocity): the controller pushes the cart under the
        falling pole instead of ignoring the state.

How to run (needs a trained model)
    python tests/test_first10_trace.py
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
ACTION_NAME = {0: "push-LEFT", 1: "push-RIGHT"}


def main():
    if not os.path.exists(MODEL):
        print(f"FAIL: no trained model at {MODEL}. Train first."); sys.exit(2)
    agent = DQNAgent(MLPQNetwork(4, 2), 2, device="cpu", gamma=0.95)
    agent.load(MODEL)
    env = make_cartpole()

    s, _ = env.reset(seed=777)
    s = s.astype(np.float32)

    print("=" * 78)
    print("TEST 5: first-10-step trace (trained CartPole, greedy)")
    print("=" * 78)
    print(f"{'t':>2} | {'cart_x':>7} {'cart_v':>7} {'angle':>7} {'ang_v':>7} | "
          f"{'Q_left':>7} {'Q_right':>7} | {'action':>10} {'reward':>6}")

    consistent = True
    maxqs = []
    for t in range(10):
        q = agent.q_values(s)
        a = int(np.argmax(q))
        a_act = agent.act(s, 0.0)
        consistent &= (a == a_act)
        maxqs.append(float(q.max()))
        s2, r, term, trunc, _ = env.step(a)
        print(f"{t:>2} | {s[0]:7.3f} {s[1]:7.3f} {s[2]:7.3f} {s[3]:7.3f} | "
              f"{q[0]:7.3f} {q[1]:7.3f} | {ACTION_NAME[a]:>10} {r:6.1f}")
        s = s2.astype(np.float32)
        if term or trunc:
            break

    # (c) reactivity over a longer greedy episode
    s, _ = env.reset(seed=778); s = s.astype(np.float32)
    acts, fall = [], []
    for _ in range(300):
        a = agent.act(s, 0.0)
        acts.append(a)
        fall.append(s[2] + 0.5 * s[3])       # falling-right indicator
        s, _, term, trunc, _ = env.step(a); s = s.astype(np.float32)
        if term or trunc:
            break
    corr = float(np.corrcoef(acts, fall)[0, 1]) if len(set(acts)) > 1 else 0.0

    maxq_ok = all(15.0 <= m <= 20.5 for m in maxqs)
    print("-" * 78)
    print(f"(a) action == argmax Q at every step : {consistent}")
    print(f"(b) max-Q in [15, 20.5] (near ceiling): {maxq_ok}  "
          f"(range {min(maxqs):.2f}..{max(maxqs):.2f})")
    print(f"(c) corr(action, falling-right)      : {corr:+.2f}  (>0 means it reacts)")
    ok = consistent and maxq_ok and corr > 0.1
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    env.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
