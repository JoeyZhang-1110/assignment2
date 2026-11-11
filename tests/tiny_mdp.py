"""A tiny, fully deterministic MDP whose optimal Q-function is known exactly.

Why this exists
---------------
On CartPole/MsPacman we never know the *true* optimal Q, so we cannot directly
check whether the Bellman update is implemented correctly. Here the environment
is small and deterministic, so we can compute Q*(s,a) exactly by tabular value
iteration and then check that the SAME DQN code (network + replay + target net)
converges to those exact numbers. If the discounting, the target bootstrap, the
action gather, or the terminal masking were wrong, the learned Q would not match.

The MDP: a 5-state corridor with a single goal reward.
    states 0..4 are non-terminal, state 5 is the terminal GOAL.
    action 1 ("right") -> s+1 ;  action 0 ("left") -> max(s-1, 0)
    reward is R_GOAL (=10) only on the transition that REACHES the goal, else 0.
Because the only reward is at the goal and every step discounts by gamma, the
optimal policy is unambiguously "always right" (reach the goal as fast as
possible); the optimal *values* fall off geometrically with distance to the
goal, so matching them tests discounting + bootstrapping, not just the argmax.
There is deliberately no per-step reward, so there is no reward-farming loop --
the optimum is clean and easy to state.
"""
from __future__ import annotations

import numpy as np

N_STATES = 5           # non-terminal states 0..4
TERMINAL = N_STATES    # state 5 (goal)
N_ACTIONS = 2          # 0 = left, 1 = right
R_GOAL = 10.0


def step_model(s: int, a: int):
    """Deterministic transition. Returns (next_state, reward, terminal)."""
    ns = s + 1 if a == 1 else max(s - 1, 0)
    terminal = ns == TERMINAL
    r = R_GOAL if terminal else 0.0
    return ns, r, terminal


def optimal_q_table(gamma: float) -> np.ndarray:
    """Exact Q*(s,a) via value iteration (deterministic MDP -> converges exactly)."""
    Q = np.zeros((N_STATES, N_ACTIONS))
    for _ in range(10_000):
        newQ = np.zeros_like(Q)
        for s in range(N_STATES):
            for a in range(N_ACTIONS):
                ns, r, terminal = step_model(s, a)
                boot = 0.0 if terminal else Q[ns].max()
                newQ[s, a] = r + gamma * boot
        if np.allclose(newQ, Q, atol=1e-12):
            Q = newQ
            break
        Q = newQ
    return Q


def one_hot(s: int) -> np.ndarray:
    v = np.zeros(N_STATES, dtype=np.float32)
    if s < N_STATES:
        v[s] = 1.0
    return v


class ChainEnv:
    """Minimal reset/step interface over the corridor MDP (one-hot observations)."""

    def __init__(self, seed: int = 0):
        self.s = 0
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.s = 0
        return one_hot(self.s)

    def step(self, a: int):
        ns, r, terminal = step_model(self.s, int(a))
        self.s = ns if ns < TERMINAL else 0  # (state reset handled by caller on done)
        return one_hot(ns), r, terminal, False, {}
