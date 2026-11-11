"""Environment construction and MsPacman image preprocessing.

CartPole needs no preprocessing. MsPacman uses the exact preprocessing function
from the assignment handout, wrapped so that the agent sees a stack of the last
4 preprocessed frames (DQN needs several frames to perceive motion/direction).
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

try:  # ale_py registers the ALE/* environment ids
    import ale_py
    gym.register_envs(ale_py)
except Exception:  # pragma: no cover - only needed for Atari
    ale_py = None


# --------------------------------------------------------------------------- #
# MsPacman preprocessing -- taken verbatim from the assignment handout.
# --------------------------------------------------------------------------- #
mspacman_color = 210 + 164 + 74


def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """Crop, downsample, greyscale and normalize a 210x160x3 frame to 88x80x1.

    Returns an int8 array in [-128, 127] exactly as specified by the handout.
    """
    img = obs[1:176:2, ::2]              # crop and downsize -> (88, 80, 3)
    img = img.sum(axis=2)                # to greyscale -> (88, 80)
    img[img == mspacman_color] = 0       # improve contrast
    img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    return img.reshape(88, 80, 1)


def make_cartpole(render_mode: str | None = None) -> gym.Env:
    return gym.make("CartPole-v1", render_mode=render_mode)


def make_mspacman(render_mode: str | None = None, repeat_action_probability: float | None = None) -> gym.Env:
    """Create ALE/MsPacman-v5 (9 actions, built-in frameskip=4).

    `repeat_action_probability` overrides the v5 default sticky-action prob of
    0.25. Setting it to 0 makes the environment (near-)deterministic, which the
    validation tests exploit to get repeatable rollouts.
    """
    kwargs = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    if repeat_action_probability is not None:
        kwargs["repeat_action_probability"] = repeat_action_probability
    return gym.make("ALE/MsPacman-v5", **kwargs)


class FrameStacker:
    """Maintains a rolling stack of the last `k` preprocessed frames.

    Not a gym wrapper on purpose -- the training loop also needs to push single
    frames into the FrameStackReplayBuffer, so we keep the stacking logic
    explicit and visible here.
    """

    def __init__(self, k: int = 4, frame_shape=(88, 80)):
        self.k = k
        self.h, self.w = frame_shape
        self.frames = deque(maxlen=k)

    def reset(self, first_frame: np.ndarray) -> np.ndarray:
        f = first_frame.reshape(self.h, self.w)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(f)   # pad by repeating the first frame
        return self.state()

    def step(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame.reshape(self.h, self.w))
        return self.state()

    def state(self) -> np.ndarray:
        return np.stack(self.frames, axis=0)  # (k, 88, 80) int8
