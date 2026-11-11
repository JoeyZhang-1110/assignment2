"""The DQN agent: policy, target network, and the Bellman learning step.

This single class drives *both* games. The only differences between CartPole
and MsPacman are the network passed in and a couple of flags (how states are
converted to tensors, whether rewards are clipped), so the core algorithm --
replay sampling, target-network bootstrap, Huber loss -- lives here once.

The three mandatory DQN ingredients are all present:
  * a neural network                    -> `self.q` (online network)
  * a replay buffer                     -> supplied by the caller, sampled in `learn`
  * a target network                    -> `self.target`, periodically synced
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNAgent:
    def __init__(
        self,
        network: nn.Module,
        n_actions: int,
        device: str = "cpu",
        gamma: float = 0.99,
        lr: float = 1e-3,
        state_scale: float = 1.0,     # multiply raw stored state by this (1/128 for int8 frames)
        double_dqn: bool = False,
        grad_clip: float | None = 10.0,
        huber_delta: float = 1.0,
        lr_step_size: int | None = None,   # episodes/steps between LR decays (None = no scheduler)
        lr_gamma: float = 0.5,
    ):
        self.device = torch.device(device)
        self.q = network.to(self.device)
        self.target = copy.deepcopy(network).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()

        self.n_actions = n_actions
        self.gamma = gamma
        self.state_scale = state_scale
        self.double_dqn = double_dqn
        self.grad_clip = grad_clip
        self.huber_delta = huber_delta

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.scheduler = None
        if lr_step_size:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=lr_step_size, gamma=lr_gamma
            )

    # ------------------------------------------------------------------ #
    # state handling
    # ------------------------------------------------------------------ #
    def _to_tensor(self, states) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(states), device=self.device, dtype=torch.float32)
        if self.state_scale != 1.0:
            t = t * self.state_scale
        return t

    # ------------------------------------------------------------------ #
    # policy
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, state, epsilon: float) -> int:
        """Epsilon-greedy action selection for a single state."""
        if np.random.random() < epsilon:
            return int(np.random.randint(self.n_actions))
        q = self.q(self._to_tensor(state[None, ...]))
        return int(q.argmax(dim=1).item())

    @torch.no_grad()
    def max_q(self, state) -> float:
        """Max predicted Q at a single state (for the max-Q-vs-episode plot)."""
        q = self.q(self._to_tensor(state[None, ...]))
        return float(q.max().item())

    @torch.no_grad()
    def q_values(self, state) -> np.ndarray:
        return self.q(self._to_tensor(state[None, ...])).cpu().numpy()[0]

    # ------------------------------------------------------------------ #
    # learning
    # ------------------------------------------------------------------ #
    def learn(self, batch) -> float:
        states, actions, rewards, next_states, dones = batch
        states = self._to_tensor(states)
        next_states = self._to_tensor(next_states)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(dones, device=self.device, dtype=torch.float32)

        # Q(s,a) for the actions actually taken.
        q_sa = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                # action selection by online net, evaluation by target net
                next_actions = self.q(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target(next_states).max(dim=1).values
            # (1 - done) masks the bootstrap at TERMINAL states. `dones` must be
            # the terminal flag, not truncation -- see train loops / edge cases.
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.huber_loss(q_sa, target, delta=self.huber_delta)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self) -> None:
        """Hard update: copy online weights into the target network."""
        self.target.load_state_dict(self.q.state_dict())

    @torch.no_grad()
    def soft_update(self, tau: float) -> None:
        """Polyak (soft) update: target <- tau*online + (1-tau)*target.

        Applied every gradient step with a small tau, this tracks the online
        network smoothly instead of in periodic jumps. For CartPole it makes the
        bootstrap target far less jittery, which matters a lot here: with
        gamma=0.95 the optimal Q saturates near 20, so the policy depends on
        *small* Q-differences that periodic hard target jumps tend to wash out.
        """
        for tp, p in zip(self.target.parameters(), self.q.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)

    def scheduler_step(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.q.load_state_dict(state)
        self.target.load_state_dict(state)
