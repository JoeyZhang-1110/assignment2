"""Roll out a trained network for N episodes and summarize the returns.

Implements requirement (iii): after training, run 500 episodes with the trained
network, plot a histogram of episode rewards, and report mean and std.

Usage:
    python -m src.evaluate --game cartpole --episodes 500
    python -m src.evaluate --game mspacman --episodes 500 --epsilon 0.05
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .dqn_agent import DQNAgent


def rollout_cartpole(model_path, episodes=500, epsilon=0.0, seed=10_000, device="cpu"):
    from .envs import make_cartpole
    from .networks import MLPQNetwork

    env = make_cartpole()
    agent = DQNAgent(MLPQNetwork(4, env.action_space.n), env.action_space.n,
                     device=device, gamma=0.95)
    agent.load(model_path)
    returns = []
    for i in range(episodes):
        s, _ = env.reset(seed=seed + i)
        s = s.astype(np.float32)
        tot = 0.0
        while True:
            a = agent.act(s, epsilon)
            s, r, term, trunc, _ = env.step(a)
            s = s.astype(np.float32)
            tot += r
            if term or trunc:
                break
        returns.append(tot)
    env.close()
    return np.asarray(returns)


def rollout_mspacman(model_path, episodes=500, epsilon=0.05, seed=10_000, device="cuda"):
    from .envs import FrameStacker, make_mspacman, preprocess_observation
    from .networks import CNNQNetwork

    device = device if torch.cuda.is_available() else "cpu"
    env = make_mspacman()
    agent = DQNAgent(CNNQNetwork(env.action_space.n), env.action_space.n,
                     device=device, gamma=0.99, state_scale=1.0 / 128.0)
    agent.load(model_path)
    fs = FrameStacker(4, (88, 80))
    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        state = fs.reset(preprocess_observation(obs))
        tot = 0.0
        while True:
            a = agent.act(state, epsilon)
            obs, r, term, trunc, _ = env.step(a)
            state = fs.step(preprocess_observation(obs))
            tot += r
            if term or trunc:
                break
        returns.append(tot)
    env.close()
    return np.asarray(returns)


def plot_histogram(returns: np.ndarray, title: str, out_path: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean, std = float(returns.mean()), float(returns.std())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(returns, bins=30, color="#4C72B0", edgecolor="white", alpha=0.9)
    ax.axvline(mean, color="#C44E52", lw=2, label=f"mean = {mean:.1f}")
    ax.axvline(mean - std, color="#C44E52", lw=1, ls="--", alpha=0.7)
    ax.axvline(mean + std, color="#C44E52", lw=1, ls="--", alpha=0.7,
               label=f"std = {std:.1f}")
    ax.set_xlabel("Episode reward")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return {"mean": mean, "std": std, "n": int(returns.size),
            "min": float(returns.min()), "max": float(returns.max())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", choices=["cartpole", "mspacman"], required=True)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--epsilon", type=float, default=None,
                   help="exploration during rollout (default: 0 cartpole, 0.05 mspacman)")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    if args.game == "cartpole":
        model = args.model or "results/cartpole/cartpole_model.pt"
        out_dir = args.out_dir or "results/cartpole"
        eps = 0.0 if args.epsilon is None else args.epsilon
        returns = rollout_cartpole(model, args.episodes, eps)
        title = f"CartPole-v1: {args.episodes}-episode rollout (greedy)"
    else:
        model = args.model or "results/mspacman/mspacman_model.pt"
        out_dir = args.out_dir or "results/mspacman"
        eps = 0.05 if args.epsilon is None else args.epsilon
        returns = rollout_mspacman(model, args.episodes, eps)
        title = f"MsPacman-v5: {args.episodes}-episode rollout (eps={eps})"

    out_png = os.path.join(out_dir, f"{args.game}_rollout_hist.png")
    stats = plot_histogram(returns, title, out_png)
    np.save(os.path.join(out_dir, f"{args.game}_rollout_returns.npy"), returns)
    with open(os.path.join(out_dir, f"{args.game}_rollout_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"{args.game}: mean={stats['mean']:.2f} std={stats['std']:.2f} "
          f"(min={stats['min']:.0f}, max={stats['max']:.0f}) over {stats['n']} episodes")
    print(f"Saved histogram -> {out_png}")


if __name__ == "__main__":
    main()
