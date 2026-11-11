"""TEST 1 -- Correctness vs an analytically-known optimum.

What it does
    Trains the project's DQN (src/dqn_agent.py: network + replay buffer +
    target network) on the deterministic corridor MDP from tiny_mdp.py, then
    compares the learned Q(s,a) to the EXACT Q*(s,a) computed by value iteration.

How it validates
    If the Bellman target, the discount gamma, the per-action gather, or the
    terminal masking were wrong, the learned Q would not match Q*. We require:
      * max |Q_learned - Q*| < 0.15 over all (s,a), and
      * the greedy policy equals the optimal policy in every state.
    These are strong, quantitative conditions -- not a format/shape check.

How to run
    python tests/test_known_q.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from src.dqn_agent import DQNAgent
from src.networks import MLPQNetwork
from src.replay_buffer import ReplayBuffer
from src.utils import LinearSchedule, set_seed
from tiny_mdp import ChainEnv, N_ACTIONS, N_STATES, one_hot, optimal_q_table

GAMMA = 0.95


def train_on_chain(steps=20_000, seed=0):
    set_seed(seed)
    env = ChainEnv(seed=seed)
    agent = DQNAgent(MLPQNetwork(N_STATES, N_ACTIONS, hidden=64), N_ACTIONS,
                     device="cpu", gamma=GAMMA, lr=1e-3, double_dqn=False)
    buf = ReplayBuffer(20_000, N_STATES, seed=seed)
    eps = LinearSchedule(1.0, 0.1, steps // 2)

    s = env.reset()
    for t in range(steps):
        a = agent.act(s, eps.value(t))
        s2, r, term, _, _ = env.step(a)
        buf.add(s, a, r, s2, term)
        s = env.reset() if term else s2
        if len(buf) >= 256:
            agent.learn(buf.sample(128))
            if t % 200 == 0:
                agent.update_target()
    return agent


def main():
    Qstar = optimal_q_table(GAMMA)
    agent = train_on_chain()

    states = np.stack([one_hot(s) for s in range(N_STATES)])
    with torch.no_grad():
        Qlearned = agent.q(torch.as_tensor(states)).cpu().numpy()

    err = np.abs(Qlearned - Qstar)
    max_err = float(err.max())
    policy_learned = Qlearned.argmax(axis=1)
    policy_star = Qstar.argmax(axis=1)
    policy_ok = bool(np.array_equal(policy_learned, policy_star))

    print("=" * 64)
    print("TEST 1: DQN convergence to the exact optimal Q*")
    print("=" * 64)
    print(f"{'state':>5} | {'Q*_left':>8} {'Q*_right':>8} | "
          f"{'Qhat_left':>9} {'Qhat_right':>10} | {'max|err|':>8}")
    for s in range(N_STATES):
        print(f"{s:>5} | {Qstar[s,0]:8.3f} {Qstar[s,1]:8.3f} | "
              f"{Qlearned[s,0]:9.3f} {Qlearned[s,1]:10.3f} | {err[s].max():8.3f}")
    print("-" * 64)
    print(f"max |Q_learned - Q*|   = {max_err:.4f}   (require < 0.15)")
    print(f"greedy policy matches  = {policy_ok}  "
          f"(learned={policy_learned.tolist()}, optimal={policy_star.tolist()})")

    # save a comparison bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(N_STATES)
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.2
        ax.bar(x - 1.5 * w, Qstar[:, 1], w, label="Q*(s, right)", color="#4C72B0")
        ax.bar(x - 0.5 * w, Qlearned[:, 1], w, label="Qhat(s, right)", color="#8FB0DA")
        ax.bar(x + 0.5 * w, Qstar[:, 0], w, label="Q*(s, left)", color="#C44E52")
        ax.bar(x + 1.5 * w, Qlearned[:, 0], w, label="Qhat(s, left)", color="#E3A1A3")
        ax.set_xlabel("state"); ax.set_ylabel("Q-value")
        ax.set_title("DQN learned Q vs exact optimal Q*")
        ax.set_xticks(x); ax.legend(fontsize=8)
        fig.tight_layout()
        os.makedirs(os.path.join(ROOT, "results/tests"), exist_ok=True)
        fig.savefig(os.path.join(ROOT, "results/tests/test1_known_q.png"), dpi=130)
        plt.close(fig)
        print("saved -> results/tests/test1_known_q.png")
    except Exception as e:
        print("plot skipped:", e)

    ok = (max_err < 0.15) and policy_ok
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 64)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
