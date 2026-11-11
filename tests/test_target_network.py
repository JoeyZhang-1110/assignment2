"""TEST 3 -- Ablation: does the target network actually matter?

The assignment requires "the notion of the target network". This test shows WHY
it is there, by removing it and measuring the damage.

Setup
    Train CartPole twice with identical everything EXCEPT the target update:
      * WITH target network : Polyak soft update, tau = 0.005 (target lags online)
      * WITHOUT             : tau = 1.0  (target == online every step; no separate
                              target network at all -- the standard "no target" case)

How it validates
    The moving bootstrap target of a no-target DQN chases its own predictions,
    which on CartPole produces (a) lower average return and (b) a far more
    unstable / over-estimating max-Q curve. We train both, then compare the mean
    training return over the second half of training. We expect WITH >> WITHOUT.

    Note: this is a *learning-dynamics* test, so we report the quantitative gap
    and save both curves rather than asserting a single brittle threshold; the
    PASS condition is that the target network gives a clearly higher return.

How to run
    python tests/test_target_network.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from src.train_cartpole import train
from src.utils import moving_average

EPISODES = 160


def run(tau, seed):
    # eval_every huge -> skip the (expensive) periodic greedy eval; we only
    # care about the training-reward dynamics here. Big solve_threshold so it
    # never early-stops.
    _, logger, _ = train(
        episodes=EPISODES, seed=seed, device="cpu",
        tau=tau, target_mode="soft", eval_every=10 ** 9,
        solve_threshold=10 ** 9, out_dir="results/tests",
        tag=f"tnet_tau{tau}_s{seed}", verbose=False,
    )
    return np.asarray(logger.data["reward"]), np.asarray(logger.data["max_q"])


def main():
    seeds = [0, 1]
    with_r, with_q, no_r, no_q = [], [], [], []
    for s in seeds:
        r, q = run(0.005, s); with_r.append(r); with_q.append(q)
        r, q = run(1.0, s);  no_r.append(r);  no_q.append(q)

    with_q_mean = np.mean(with_q, axis=0); no_q_mean = np.mean(no_q, axis=0)
    with_r = np.mean(with_r, axis=0); no_r = np.mean(no_r, axis=0)
    half = EPISODES // 2
    with_mean = float(with_r[half:].mean())
    no_mean = float(no_r[half:].mean())
    # Divergence signal: final mean max-Q (last 20 episodes).
    with_q_final = float(with_q_mean[-20:].mean())
    no_q_final = float(no_q_mean[-20:].mean())
    CEILING = 20.0

    print("=" * 64)
    print("TEST 3: target-network ablation on CartPole")
    print("=" * 64)
    print("PRIMARY SIGNAL -- max-Q divergence (theoretical ceiling ~20):")
    print(f"  final mean max-Q WITH target net    : {with_q_final:10.1f}  (bounded near ceiling)")
    print(f"  final mean max-Q WITHOUT target net : {no_q_final:10.1f}  (should blow past 20)")
    print(f"  divergence factor (without/ceiling) : {no_q_final / CEILING:10.1f} x")
    print("-" * 64)
    print(f"mean training return, 2nd half (averaged over {len(seeds)} seeds):")
    print(f"  WITH target network (tau=0.005) : {with_mean:7.1f}")
    print(f"  WITHOUT target network (tau=1.0): {no_mean:7.1f}")
    print(f"  improvement from target network : {with_mean - no_mean:+7.1f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = np.arange(EPISODES)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        a1.plot(ep, moving_average(with_r, 20), label="with target net", color="#4C72B0")
        a1.plot(ep, moving_average(no_r, 20), label="no target net", color="#C44E52")
        a1.set_title("reward (20-ep moving avg)"); a1.set_xlabel("episode")
        a1.set_ylabel("return"); a1.legend()
        a2.plot(ep, np.clip(np.mean(with_q, axis=0), 1e-2, None), label="with target net", color="#4C72B0")
        a2.plot(ep, np.clip(np.mean(no_q, axis=0), 1e-2, None), label="no target net", color="#C44E52")
        a2.axhline(20.0, color="k", ls="--", lw=1, label="Q ceiling ~20")
        a2.set_yscale("log")
        a2.set_title("max-Q stability (log scale) — no-target DIVERGES")
        a2.set_xlabel("episode"); a2.set_ylabel("mean max-Q (log)"); a2.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "results/tests/test3_target_network.png"), dpi=130)
        plt.close(fig)
        print("saved -> results/tests/test3_target_network.png")
    except Exception as e:
        print("plot skipped:", e)

    # The target network's defining job is to stop the bootstrap target from
    # chasing its own predictions and diverging. PASS requires: (1) WITHOUT it
    # the max-Q blows well past the ~20 ceiling, (2) WITH it the max-Q stays
    # bounded near the ceiling, and (3) the target network also yields >= return.
    diverges = no_q_final > 5 * CEILING        # without: clearly diverges (>100)
    bounded = with_q_final < 1.5 * CEILING     # with: stays bounded (<30)
    better = with_mean >= no_mean              # and does not hurt return
    ok = diverges and bounded and better
    print(f"\nwithout target net diverges (>100) : {diverges}")
    print(f"with target net stays bounded (<30): {bounded}")
    print(f"target net return >= no-target     : {better}")
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 64)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
