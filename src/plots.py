"""Build the two per-episode plots required for both games:

  (i)  maximum Q-value vs training episode
  (ii) episode reward vs training episode, overlaid with the 100-episode
       moving average.

Both are generated from the JSON log written by the training scripts, so they
can be regenerated offline without retraining.

Usage:
    python -m src.plots --log results/cartpole/cartpole_log.json --out-dir results/cartpole
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import Logger, moving_average


def plot_max_q(log: dict, out_path: str) -> None:
    ep = np.asarray(log["episode"])
    mq = np.asarray(log["max_q"])
    meta = log.get("meta", {})

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ep, mq, color="#4C72B0", lw=1.4, label="mean max-Q (held-out states)")

    # Overlay the theoretical ceiling where we have one (CartPole).
    ceil = meta.get("theoretical_max_q")
    if ceil is not None:
        ax.axhline(ceil, color="#C44E52", ls="--", lw=1.5,
                   label=f"theoretical ceiling = {ceil:.2f}")
        ax.set_ylim(min(mq.min(), 0) - 0.5, ceil * 1.15)

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Max predicted Q-value")
    ax.set_title(f"{meta.get('game','')}: max Q-value vs training episode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_rewards(log: dict, out_path: str, window: int = 100) -> None:
    ep = np.asarray(log["episode"])
    r = np.asarray(log["reward"], dtype=float)
    ma = moving_average(r, window)
    meta = log.get("meta", {})

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ep, r, color="#CCCCCC", lw=1.0, label="episode reward")
    ax.plot(ep, ma, color="#C44E52", lw=2.0, label=f"moving avg (last {window})")

    # Overlay greedy-eval curve if present (CartPole): the clean solve signal.
    if "eval_episode" in log and log["eval_episode"]:
        ax.plot(log["eval_episode"], log["eval_reward"], "o-", color="#55A868",
                ms=4, lw=1.4, label="greedy eval (eps=0)")

    solved = meta.get("solved_episode")
    if solved is not None:
        ax.axvline(solved, color="#8172B2", ls=":", lw=1.5,
                   label=f"solved @ ep {solved}")

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward")
    ax.set_title(f"{meta.get('game','')}: reward vs training episode")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def build_all(log_path: str, out_dir: str, prefix: str | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log = Logger.load(log_path)
    prefix = prefix or os.path.splitext(os.path.basename(log_path))[0].replace("_log", "")
    plot_max_q(log, os.path.join(out_dir, f"{prefix}_maxq.png"))
    plot_rewards(log, os.path.join(out_dir, f"{prefix}_rewards.png"))
    print(f"Saved {prefix}_maxq.png and {prefix}_rewards.png to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", default=None)
    args = p.parse_args()
    build_all(args.log, args.out_dir, args.prefix)


if __name__ == "__main__":
    main()
