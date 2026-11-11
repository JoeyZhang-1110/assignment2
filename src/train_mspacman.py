"""Train a DQN on ALE/MsPacman-v5 (discount factor 0.99).

States are stacks of 4 preprocessed 88x80 frames; the network is a Nature-DQN
CNN with 9 outputs (MsPacman's reduced action set). Advanced techniques enabled,
as suggested by the handout: Huber loss, an epsilon scheduler, a learning-rate
scheduler, and reward clipping for training stability.

The model genuinely needs thousands of episodes (~4000-6000) on a GPU. The
script checkpoints periodically so a long run can be resumed/inspected, and
writes the same JSON log format as CartPole so plots.py works unchanged.

Run (full):
    python -m src.train_mspacman --episodes 5000 --seed 0
Quick smoke test:
    python -m src.train_mspacman --episodes 5 --warmup 500 --buffer 5000
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .dqn_agent import DQNAgent
from .envs import FrameStacker, make_mspacman, preprocess_observation
from .networks import CNNQNetwork
from .replay_buffer import FrameStackReplayBuffer
from .utils import LinearSchedule, Logger, set_seed, theoretical_max_q

GAMMA = 0.99
STACK = 4
FRAME = (88, 80)


def collect_eval_states(n=256, seed=123, stack=STACK):
    """Fixed held-out states (random policy) for the max-Q curve."""
    env = make_mspacman()
    fs = FrameStacker(stack, FRAME)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    state = fs.reset(preprocess_observation(obs))
    states = []
    while len(states) < n:
        states.append(state.astype(np.int8))
        a = int(rng.integers(env.action_space.n))
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
            state = fs.reset(preprocess_observation(obs))
        else:
            state = fs.step(preprocess_observation(obs))
    env.close()
    return np.asarray(states, dtype=np.int8)


def train(
    episodes: int = 5000,
    seed: int = 0,
    lr: float = 1e-4,
    buffer_size: int = 100_000,
    batch_size: int = 32,
    warmup: int = 20_000,
    train_freq: int = 4,            # agent steps between gradient updates
    target_update: int = 2_500,     # agent steps between hard target syncs
    eps_start: float = 1.0,
    eps_end: float = 0.1,
    eps_decay_steps: int = 400_000,
    lr_step_episodes: int = 1_500,  # LR halves every this many episodes
    clip_rewards: bool = True,
    max_steps: int | None = None,
    checkpoint_every: int = 250,
    out_dir: str = "results/mspacman",
    tag: str = "mspacman",
    device: str | None = None,
    verbose: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    env = make_mspacman()
    n_actions = env.action_space.n
    assert n_actions == 9, f"expected 9 actions, got {n_actions}"

    eval_states = collect_eval_states(n=256, seed=seed + 777)

    agent = DQNAgent(
        CNNQNetwork(n_actions, stack=STACK, h=FRAME[0], w=FRAME[1]),
        n_actions=n_actions,
        device=device,
        gamma=GAMMA,
        lr=lr,
        state_scale=1.0 / 128.0,     # int8 frames -> ~[-1, 1]
        lr_step_size=lr_step_episodes,
        lr_gamma=0.5,
    )
    buffer = FrameStackReplayBuffer(buffer_size, FRAME, STACK, seed=seed)
    fs = FrameStacker(STACK, FRAME)
    eps_sched = LinearSchedule(eps_start, eps_end, eps_decay_steps)
    logger = Logger(os.path.join(out_dir, f"{tag}_log.json"))

    total_steps = 0
    recent = []
    t0 = time.time()

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        frame = preprocess_observation(obs)   # current frame (newest in the stack)
        state = fs.reset(frame)
        first = True

        ep_reward, ep_loss, ep_len = 0.0, [], 0
        while True:
            eps = eps_sched.value(total_steps)
            a = agent.act(state, eps)
            obs, r, terminated, truncated, _ = env.step(a)

            ep_reward += r                      # raw game score for plots
            train_r = np.sign(r) if clip_rewards else r  # clipped for TD updates

            # Store the CURRENT frame together with the action taken from it and
            # the resulting reward/done. The next-state for this transition is
            # the frame stored on the NEXT add (standard Atari replay layout);
            # bootstrap flag = terminal only (same edge case as CartPole).
            buffer.add(frame, a, train_r, bool(terminated), episode_start=first)
            first = False

            frame = preprocess_observation(obs)  # advance to the resulting frame
            state = fs.step(frame)
            ep_len += 1
            total_steps += 1

            if len(buffer) >= warmup and total_steps % train_freq == 0:
                loss = agent.learn(buffer.sample(batch_size))
                ep_loss.append(loss)
            if total_steps % target_update == 0:
                agent.update_target()

            if terminated or truncated:
                break
            if max_steps and total_steps >= max_steps:
                break

        agent.scheduler_step()
        recent.append(ep_reward)
        ma100 = float(np.mean(recent[-100:]))
        with torch.no_grad():
            st = torch.as_tensor(eval_states, device=agent.device, dtype=torch.float32) / 128.0
            mean_max_q = float(agent.q(st).max(dim=1).values.mean().item())

        logger.log(
            episode=ep, reward=ep_reward, length=ep_len, max_q=mean_max_q,
            epsilon=eps, loss=float(np.mean(ep_loss)) if ep_loss else 0.0,
            total_steps=total_steps,
        )

        if verbose and (ep % 10 == 0 or ep == episodes - 1):
            fps = total_steps / max(time.time() - t0, 1e-6)
            print(f"ep {ep:5d} | R {ep_reward:7.0f} | MA100 {ma100:7.1f} | maxQ {mean_max_q:6.2f} "
                  f"| eps {eps:4.2f} | steps {total_steps} | {fps:5.0f} fps")

        if (ep + 1) % checkpoint_every == 0:
            agent.save(os.path.join(out_dir, f"{tag}_model.pt"))
            logger.save(extra=_meta(seed, device))
        if max_steps and total_steps >= max_steps:
            break

    agent.save(os.path.join(out_dir, f"{tag}_model.pt"))
    logger.save(extra=_meta(seed, device))
    env.close()
    if verbose:
        print(f"Saved model + log to {out_dir}.")
    return agent, logger


def _meta(seed, device):
    return {
        "game": "ALE/MsPacman-v5",
        "gamma": GAMMA,
        "n_actions": 9,
        "reward_clipped_for_training": True,
        "theoretical_max_q_clipped": theoretical_max_q(GAMMA, 10_000, 1.0),
        "seed": seed,
        "device": device,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=20_000)
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out-dir", type=str, default="results/mspacman")
    p.add_argument("--tag", type=str, default="mspacman")
    args = p.parse_args()
    train(episodes=args.episodes, seed=args.seed, warmup=args.warmup,
          buffer_size=args.buffer, max_steps=args.max_steps,
          out_dir=args.out_dir, tag=args.tag)


if __name__ == "__main__":
    main()
