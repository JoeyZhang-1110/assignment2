"""TEST 6 -- Frame-stack replay never mixes two episodes (MsPacman edge case).

Atari states are stacks of 4 consecutive frames. A reconstructed stack must
never reach back across an episode boundary into the previous episode, or the
agent would be trained on a physically impossible "state" (half of one game,
half of the next). `FrameStackReplayBuffer` stores single frames and rebuilds
stacks on sampling; this test feeds a hand-built two-episode sequence with known
frame ids and checks every reconstructed stack.

This is the one place where a cheap, deterministic check is the right tool, so we
assert exact frame layouts. It validates the episode-boundary edge case claimed
in the report, independently of any training.

How to run
    python tests/test_framestack_buffer.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from src.replay_buffer import FrameStackReplayBuffer


def main():
    buf = FrameStackReplayBuffer(capacity=50, frame_shape=(1, 1), stack=4, seed=0)
    # Episode A frames 10,11,12 (A starts at 10, ends terminal at 12);
    # Episode B frames 20,21,22,23 (B starts at 20).
    seq = [
        (10, True, 0, 0.0, False), (11, False, 1, 1.0, False), (12, False, 2, 2.0, True),
        (20, True, 0, 0.0, False), (21, False, 1, 1.0, False),
        (22, False, 2, 2.0, False), (23, False, 3, 3.0, True),
    ]
    for fid, first, a, r, done in seq:
        buf.add(np.array([[fid]], dtype=np.int8), a, r, done, episode_start=first)

    def stk(i):
        return buf._get_stack(i).reshape(-1).tolist()

    cases = {
        0: [10, 10, 10, 10],   # episode start -> repeat-pad
        1: [10, 10, 10, 11],
        2: [10, 10, 11, 12],   # within episode A
        3: [20, 20, 20, 20],   # episode B start -> must NOT contain A's 12
        4: [20, 20, 20, 21],
        5: [20, 20, 21, 22],
        6: [20, 21, 22, 23],   # full stack within episode B
    }

    print("=" * 64)
    print("TEST 6: frame-stack buffer boundary + alignment")
    print("=" * 64)
    ok = True
    for i, want in cases.items():
        got = stk(i)
        good = got == want
        ok &= good
        print(f"stack@{i}: got {got}  want {want}  {'OK' if good else 'WRONG'}")
    leak = 12 in stk(3)
    print(f"episode-boundary leak into stack@3 (A's frame 12): {leak} (must be False)")
    ok &= not leak
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 64)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
