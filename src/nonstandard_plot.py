"""NON-STANDARD VISUALIZATION: the CartPole value landscape forming over training.

Standard DQN plots show scalar curves over time. This instead opens up *what the
network has actually learned as a function of state*. We slice CartPole's 4-D
state at cart position = 0 and cart velocity = 0, sweep a grid over

    pole angle  theta    in [-0.25, 0.25] rad   (the pole falls at |theta| ~ 0.2095)
    pole ang-vel thetadot in [-3, 3] rad/s

and, at snapshots taken *throughout* training, render:

  * left  panel : a heatmap of max_a Q(s,a) -- the learned "how safe am I here"
                  surface, which should rise toward the gamma-ceiling (~20) in
                  the central safe region;
  * right panel : the greedy decision map (push-left vs push-right), i.e. the
                  policy's decision boundary, which should sharpen into the
                  classic "oppose the fall" diagonal as learning proceeds.

The frames are stitched into an animated GIF so you literally watch the value
surface and decision boundary crystallize as the agent learns. A final static
figure overlays a real greedy episode's trajectory on the decision map to show
the policy keeps the state inside the region it has learned is safe.

Usage:
    python -m src.nonstandard_plot --episodes 300 --out-dir results/cartpole
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .dqn_agent import DQNAgent
from .envs import make_cartpole
from .networks import MLPQNetwork
from .replay_buffer import ReplayBuffer
from .utils import LinearSchedule, set_seed, theoretical_max_q

GAMMA = 0.95
FAIL_ANGLE = 0.2095  # ~12 degrees; CartPole terminates beyond this
THETA_RANGE = (-0.25, 0.25)
THETADOT_RANGE = (-3.0, 3.0)
GRID = 90


def _grid_states():
    thetas = np.linspace(*THETA_RANGE, GRID)
    thetadots = np.linspace(*THETADOT_RANGE, GRID)
    TH, THD = np.meshgrid(thetas, thetadots)           # (GRID, GRID)
    states = np.zeros((GRID * GRID, 4), dtype=np.float32)
    states[:, 2] = TH.ravel()                          # pole angle
    states[:, 3] = THD.ravel()                         # pole angular velocity
    return thetas, thetadots, states


@torch.no_grad()
def _snapshot(agent, states):
    q = agent.q(torch.as_tensor(states, device=agent.device))  # (N, 2)
    maxq = q.max(dim=1).values.cpu().numpy().reshape(GRID, GRID)
    action = q.argmax(dim=1).cpu().numpy().reshape(GRID, GRID)
    return maxq, action


def _render_frame(thetas, thetadots, maxq, action, ep, ceil):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
    extent = [thetas[0], thetas[-1], thetadots[0], thetadots[-1]]

    im = axL.imshow(maxq, origin="lower", aspect="auto", extent=extent,
                    cmap="viridis", vmin=0, vmax=ceil)
    axL.set_title(f"max-Q surface   (episode {ep})")
    axL.set_xlabel("pole angle  theta (rad)")
    axL.set_ylabel("pole angular velocity  (rad/s)")
    for xv in (-FAIL_ANGLE, FAIL_ANGLE):
        axL.axvline(xv, color="white", ls="--", lw=1, alpha=0.7)
    fig.colorbar(im, ax=axL, label="max Q")

    axR.imshow(action, origin="lower", aspect="auto", extent=extent,
               cmap="coolwarm", vmin=0, vmax=1)
    axR.set_title(f"greedy policy: push-left (blue) / push-right (red)")
    axR.set_xlabel("pole angle  theta (rad)")
    axR.set_ylabel("pole angular velocity  (rad/s)")
    for xv in (-FAIL_ANGLE, FAIL_ANGLE):
        axR.axvline(xv, color="black", ls="--", lw=1, alpha=0.6)
    fig.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = buf[:, :, :3].copy()
    plt.close(fig)
    return img


def run(episodes=300, seed=0, capture_every=12, out_dir="results/cartpole", device="cpu"):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    thetas, thetadots, grid_states = _grid_states()
    ceil = theoretical_max_q(GAMMA, 500)

    env = make_cartpole()
    agent = DQNAgent(MLPQNetwork(4, 2), 2, device=device, gamma=GAMMA,
                     lr=5e-4, double_dqn=True)
    buffer = ReplayBuffer(100_000, 4, seed=seed)
    eps_sched = LinearSchedule(1.0, 0.01, 15_000)

    frames, frame_eps = [], []
    total_steps = 0
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        s = s.astype(np.float32)
        while True:
            eps = eps_sched.value(total_steps)
            a = agent.act(s, eps)
            s2, r, term, trunc, _ = env.step(a)
            s2 = s2.astype(np.float32)
            buffer.add(s, a, r, s2, bool(term))   # terminal-only bootstrap flag
            s = s2
            total_steps += 1
            if len(buffer) >= 1000:
                agent.learn(buffer.sample(128))
                agent.soft_update(5e-3)
            if term or trunc:
                break
        if ep % capture_every == 0 or ep == episodes - 1:
            maxq, action = _snapshot(agent, grid_states)
            frames.append(_render_frame(thetas, thetadots, maxq, action, ep, ceil))
            frame_eps.append(ep)

    # ---- write animated GIF ----
    import imageio.v2 as imageio
    gif_path = os.path.join(out_dir, "cartpole_value_landscape.gif")
    # hold the converged surface longer by repeating the last frame; a scalar
    # duration (seconds per frame) keeps this robust across imageio versions.
    frames_out = frames + [frames[-1]] * 8
    imageio.mimsave(gif_path, frames_out, duration=0.3, loop=0)
    print(f"Saved animation -> {gif_path}  ({len(frames)} frames, episodes {frame_eps[0]}..{frame_eps[-1]})")

    # ---- final static figure with a real trajectory overlaid ----
    maxq, action = _snapshot(agent, grid_states)
    traj = _greedy_trajectory(agent, seed=4242)
    _static_figure(thetas, thetadots, maxq, action, traj, ceil,
                   os.path.join(out_dir, "cartpole_decision_landscape.png"))
    env.close()


def _greedy_trajectory(agent, seed=0, max_steps=500):
    env = make_cartpole()
    s, _ = env.reset(seed=seed)
    s = s.astype(np.float32)
    pts = []
    for _ in range(max_steps):
        pts.append((s[2], s[3]))   # (theta, thetadot)
        a = agent.act(s, 0.0)
        s, _, term, trunc, _ = env.step(a)
        s = s.astype(np.float32)
        if term or trunc:
            break
    env.close()
    return np.asarray(pts)


def _static_figure(thetas, thetadots, maxq, action, traj, ceil, out_path):
    extent = [thetas[0], thetas[-1], thetadots[0], thetadots[-1]]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    im = axL.imshow(maxq, origin="lower", aspect="auto", extent=extent,
                    cmap="viridis", vmin=0, vmax=ceil)
    axL.set_title("Converged max-Q surface")
    axL.set_xlabel("pole angle theta (rad)"); axL.set_ylabel("ang. vel (rad/s)")
    fig.colorbar(im, ax=axL, label="max Q")

    axR.imshow(action, origin="lower", aspect="auto", extent=extent,
               cmap="coolwarm", vmin=0, vmax=1)
    if traj.size:
        axR.plot(traj[:, 0], traj[:, 1], color="black", lw=1.2, alpha=0.9)
        axR.scatter(traj[0, 0], traj[0, 1], color="lime", s=40, zorder=5, label="start")
    axR.set_title("Decision map + a real 500-step greedy episode")
    axR.set_xlabel("pole angle theta (rad)"); axR.set_ylabel("ang. vel (rad/s)")
    axR.set_xlim(thetas[0], thetas[-1]); axR.set_ylim(thetadots[0], thetadots[-1])
    axR.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved static figure -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--capture-every", type=int, default=12)
    p.add_argument("--out-dir", type=str, default="results/cartpole")
    args = p.parse_args()
    run(episodes=args.episodes, seed=args.seed, capture_every=args.capture_every,
        out_dir=args.out_dir)


if __name__ == "__main__":
    main()
