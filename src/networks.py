"""Q-network architectures.

* ``MLPQNetwork`` -- a small fully connected net for CartPole's 4-D state.
* ``CNNQNetwork`` -- the classic "Nature DQN" convolutional net for MsPacman,
  adapted to the 88x80 preprocessed frames (4 stacked).

Both output one Q-value per action; the greedy action is the argmax.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNQNetwork(nn.Module):
    """Nature-DQN CNN for stacked 88x80 int8 frames.

    Input is expected as float in roughly [-1, 1] (we scale the int8 frames by
    1/128 in the agent). Conv stack -> 512-unit FC -> n_actions.
    """

    def __init__(self, n_actions: int, stack: int = 4, h: int = 88, w: int = 80):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Infer the flattened conv-output size instead of hard-coding it, so the
        # net stays correct if the frame size ever changes.
        with torch.no_grad():
            dummy = torch.zeros(1, stack, h, w)
            n_flat = self.features(dummy).view(1, -1).size(1)
        self.head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.head(x)
