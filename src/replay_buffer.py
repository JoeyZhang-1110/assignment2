"""Experience replay buffers.

Two implementations:

* ``ReplayBuffer`` -- for low-dimensional *vector* states (CartPole). Stores
  full transitions (s, a, r, s', done) in pre-allocated NumPy arrays.

* ``FrameStackReplayBuffer`` -- for Atari. Storing a stacked 4x88x80 state for
  every transition would waste 4x memory, so instead we store *single* frames
  (uint8) in a ring and reconstruct the 4-frame stack on sampling. Crucially it
  respects episode boundaries: when reconstructing a stack we never reach back
  across a ``done`` into the previous episode -- those positions are zero-padded.
  Getting this boundary logic right is one of the documented edge cases.
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Uniform experience replay for vector observations."""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        # ``done`` here means *terminal* (environment ended because the agent
        # failed / finished), NOT truncated by a time limit. The training loop
        # is responsible for passing the right flag -- see dqn_agent / train_*.
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size


class FrameStackReplayBuffer:
    """Memory-efficient replay for stacked Atari frames.

    We store one preprocessed frame per transition plus the action/reward/done.
    A *state* is the stack of the last ``stack`` frames. On sampling we gather
    both the state (frames ending at i) and next-state (frames ending at i+1)
    by walking backwards, stopping at episode boundaries so stacks never mix
    two episodes.
    """

    def __init__(self, capacity: int, frame_shape=(88, 80), stack: int = 4, seed: int = 0):
        self.capacity = int(capacity)
        self.stack = stack
        self.h, self.w = frame_shape
        # frames stored as int8 exactly as produced by preprocess_observation.
        self.frames = np.zeros((self.capacity, self.h, self.w), dtype=np.int8)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        # ``first`` marks the frame that begins an episode: when building a stack
        # we must not reach *before* a frame flagged first (it belongs to the
        # previous episode), so we zero-pad instead.
        self.first = np.zeros(self.capacity, dtype=bool)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, frame, action, reward, done, episode_start) -> None:
        i = self.ptr
        self.frames[i] = frame.reshape(self.h, self.w)
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.first[i] = bool(episode_start)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_stack(self, end_idx: int) -> np.ndarray:
        """Return the `stack` frames ending at `end_idx` (newest last).

        Walk backwards; if we cross an episode start we stop and pad the
        remaining (older) slots by REPEATING that episode-start frame. This
        matches how FrameStacker.reset() pads the first few states of an episode
        at action-selection time, so the states the agent *acts on* and the
        states sampled for *training* follow the same padding convention and the
        stack never contains frames from a prior episode.
        """
        out = np.empty((self.stack, self.h, self.w), dtype=np.int8)
        idx = end_idx
        for s in range(self.stack - 1, -1, -1):
            out[s] = self.frames[idx]
            if self.first[idx]:
                for s2 in range(s - 1, -1, -1):   # repeat the boundary frame
                    out[s2] = self.frames[idx]
                break
            idx = (idx - 1) % self.capacity
        return out

    def sample(self, batch_size: int):
        # Valid sample indices: anything with a defined "next frame" in the same
        # episode. We sample i in [0, size-1), require frame i is not the last
        # stored frame and frame i+1 is not an episode start (else i was a
        # terminal transition whose next-state is irrelevant anyway because
        # done=1 masks the bootstrap).
        states = np.zeros((batch_size, self.stack, self.h, self.w), dtype=np.int8)
        next_states = np.zeros((batch_size, self.stack, self.h, self.w), dtype=np.int8)
        actions = np.zeros(batch_size, dtype=np.int64)
        rewards = np.zeros(batch_size, dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)

        count = 0
        while count < batch_size:
            i = int(self.rng.integers(0, self.size))
            nxt = (i + 1) % self.capacity
            # skip the single frame at the write head (its "next" is undefined)
            if nxt == self.ptr:
                continue
            states[count] = self._get_stack(i)
            next_states[count] = self._get_stack(nxt)
            actions[count] = self.actions[i]
            rewards[count] = self.rewards[i]
            dones[count] = self.dones[i]
            count += 1
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size
