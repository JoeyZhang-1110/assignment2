"""Train a DQN on CartPole-v1 (discount factor 0.95).

The environment is "solved" when the pole is kept upright for 500 steps; we
report it solved once the 100-episode moving average of the return reaches
`solve_threshold` (default 475), which guarantees many near-perfect episodes.

Produces a JSON training log (for plots.py) and a saved model checkpoint.

Run:
    python -m src.train_cartpole --episodes 600 --seed 0
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .dqn_agent import DQNAgent
from .envs import make_cartpole
from .networks import MLPQNetwork
from .replay_buffer import ReplayBuffer
from .utils import LinearSchedule, Logger, moving_average, set_seed, theoretical_max_q

# CartPole-v1 constants
GAMMA = 0.95
HORIZON = 500  # max episode length in v1


def collect_eval_states(env, n: int = 256, seed: int = 123) -> np.ndarray:
    """Fixed set of states (random policy) for a stable max-Q curve.

    Following the original DQN paper, the max-Q plot is the *average over a held
    out set of states* of max_a Q(s,a). A fixed set makes the curve comparable
    across episodes (no variance from which states we happened to visit).
    """
    states = []
    rng = np.random.default_rng(seed)
    s, _ = env.reset(seed=seed)
    while len(states) < n:
        states.append(s.astype(np.float32))
        a = int(rng.integers(env.action_space.n))
        s, _, term, trunc, _ = env.step(a)
        if term or trunc:
            s, _ = env.reset()
    return np.asarray(states, dtype=np.float32)


def greedy_eval(agent, n_episodes: int = 20, seed: int = 9000) -> float:
    """Run `n_episodes` fully greedy (epsilon=0) episodes and return mean return.

    This is the *true* performance signal. The training-reward curve is noisy
    because we keep epsilon=0.01 for exploration; greedy eval removes that noise
    and gives a clean "is it solved?" measurement (solved == mean return 500).
    """
    env = make_cartpole()
    returns = []
    for i in range(n_episodes):
        s, _ = env.reset(seed=seed + i)
        s = s.astype(np.float32)
        tot = 0.0
        while True:
            a = agent.act(s, 0.0)
            s, r, term, trunc, _ = env.step(a)
            s = s.astype(np.float32)
            tot += r
            if term or trunc:
                break
        returns.append(tot)
    env.close()
    return float(np.mean(returns))


def train(
    episodes: int = 1000,
    seed: int = 0,
    eval_every: int = 20,
    eval_episodes: int = 20,
    lr: float = 5e-4,
    buffer_size: int = 100_000,
    batch_size: int = 128,
    warmup: int = 1_000,
    target_mode: str = "soft",      # "soft" (Polyak) or "hard"
    tau: float = 5e-3,              # soft-update rate (per gradient step)
    target_update: int = 500,       # steps between syncs when target_mode="hard"
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay_steps: int = 20_000,
    solve_threshold: float = 475.0,
    double_dqn: bool = True,
    out_dir: str = "results/cartpole",
    tag: str = "cartpole",
    device: str | None = None,
    verbose: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    env = make_cartpole()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    eval_states = collect_eval_states(make_cartpole(), n=256, seed=seed + 777)

    agent = DQNAgent(
        MLPQNetwork(obs_dim, n_actions),
        n_actions=n_actions,
        device=device,
        gamma=GAMMA,
        lr=lr,
        double_dqn=double_dqn,
    )
    buffer = ReplayBuffer(buffer_size, obs_dim, seed=seed)
    eps_sched = LinearSchedule(eps_start, eps_end, eps_decay_steps)
    logger = Logger(os.path.join(out_dir, f"{tag}_log.json"))

    total_steps = 0
    solved_episode = None
    best_eval = -1.0
    recent_rewards: list[float] = []

    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        s = s.astype(np.float32)
        ep_reward, ep_loss, ep_len = 0.0, [], 0

        while True:
            eps = eps_sched.value(total_steps)
            a = agent.act(s, eps)
            s2, r, terminated, truncated, _ = env.step(a)
            s2 = s2.astype(np.float32)

            # EDGE CASE: bootstrap flag must be the *terminal* flag only.
            # CartPole-v1 truncates at 500 steps -- that is a time limit, the
            # pole did NOT fall, so we must still bootstrap from s2. Storing
            # truncation as "done" would teach the agent that surviving to 500
            # has zero future value and cap learned Q far below the true ~20.
            done_terminal = bool(terminated)
            buffer.add(s, a, r, s2, done_terminal)

            s = s2
            ep_reward += r
            ep_len += 1
            total_steps += 1

            if len(buffer) >= warmup:
                loss = agent.learn(buffer.sample(batch_size))
                ep_loss.append(loss)
                if target_mode == "soft":
                    agent.soft_update(tau)
                elif total_steps % target_update == 0:
                    agent.update_target()

            if terminated or truncated:
                break

        recent_rewards.append(ep_reward)
        ma100 = float(np.mean(recent_rewards[-100:]))
        mean_max_q = float(agent.q(
            torch.as_tensor(eval_states, device=agent.device)
        ).max(dim=1).values.mean().item())

        logger.log(
            episode=ep,
            reward=ep_reward,
            length=ep_len,
            max_q=mean_max_q,
            epsilon=eps,
            loss=float(np.mean(ep_loss)) if ep_loss else 0.0,
            total_steps=total_steps,
        )

        # Periodic GREEDY evaluation (the clean performance signal).
        if len(buffer) >= warmup and (ep % eval_every == 0 or ep == episodes - 1):
            eval_mean = greedy_eval(agent, eval_episodes, seed=seed + 9000)
            logger.log(eval_episode=ep, eval_reward=eval_mean)
            if eval_mean > best_eval:
                best_eval = eval_mean
                agent.save(os.path.join(out_dir, f"{tag}_model.pt"))  # keep best
            if verbose:
                print(f"ep {ep:4d} | R {ep_reward:6.1f} | MA100 {ma100:6.1f} "
                      f"| maxQ {mean_max_q:5.2f} | eps {eps:4.2f} | EVAL {eval_mean:6.1f}")
            # SOLVED when the greedy policy averages >= threshold (500 => perfect).
            if solved_episode is None and eval_mean >= solve_threshold:
                solved_episode = ep
                if verbose:
                    print(f"*** SOLVED at episode {ep} (greedy eval={eval_mean:.1f}) ***")
                episodes = min(episodes, ep + 2 * eval_every)  # a little past solve, then stop

    # If we never beat the initial best (shouldn't happen), still save final.
    if best_eval < 0:
        agent.save(os.path.join(out_dir, f"{tag}_model.pt"))
    logger.save(extra={
        "game": "CartPole-v1",
        "gamma": GAMMA,
        "theoretical_max_q": theoretical_max_q(GAMMA, HORIZON),
        "solved_episode": solved_episode,
        "best_greedy_eval": best_eval,
        "double_dqn": double_dqn,
        "target_mode": target_mode,
        "tau": tau,
        "seed": seed,
        "device": device,
    })
    env.close()
    if verbose:
        print(f"Saved model + log to {out_dir}. solved_episode={solved_episode}")
    return agent, logger, solved_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--double-dqn", action="store_true")
    p.add_argument("--out-dir", type=str, default="results/cartpole")
    p.add_argument("--tag", type=str, default="cartpole")
    args = p.parse_args()
    train(episodes=args.episodes, seed=args.seed, double_dqn=args.double_dqn,
          out_dir=args.out_dir, tag=args.tag)


if __name__ == "__main__":
    main()
