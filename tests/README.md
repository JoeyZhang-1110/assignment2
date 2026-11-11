# Validation & correctness tests

These are **not** unit tests or format checks. Each one is an RL-aware experiment
that tries to answer *"beyond asking an AI, how do I know the algorithm is
actually correct and correctly tuned for the task?"* Every test prints a
quantitative result and a `PASS`/`FAIL`, and several save a figure to
`results/tests/`.

Run them all:

```bash
python tests/test_known_q.py          # Test 1
python tests/test_truncation.py       # Test 2
python tests/test_target_network.py   # Test 3
python tests/test_random_baseline.py  # Test 4  (needs a trained CartPole model)
python tests/test_first10_trace.py    # Test 5  (needs a trained CartPole model)
python tests/test_framestack_buffer.py # Test 6 (instant; no training)
```

Tests 4 and 5 load `results/cartpole/cartpole_model.pt`, so run
`python -m src.train_cartpole` first if it is missing.

| # | File | What it validates | How it validates |
|---|------|-------------------|------------------|
| 1 | `test_known_q.py` | The Bellman update, discounting, action-gather and terminal masking are correct. | Builds a tiny deterministic MDP (`tiny_mdp.py`) whose optimal Q\* is computed exactly by value iteration, trains the **real** DQN code on it, and requires `max|Q_learned − Q*| < 0.15` **and** an exact policy match. Converges to within ~0.004. |
| 2 | `test_truncation.py` | The bootstrap is masked on `terminated` only, **not** on `truncated` (the classic CartPole-500 trap). | On a "+1 forever, truncate at 50" environment the true value is `1/(1−γ)=100`. Correct masking learns ≈100; masking on truncation collapses to the predicted biased fixed point ≈33.6. Matching **both** numbers proves we mask the right flag. |
| 3 | `test_target_network.py` | The target network is load-bearing, not decorative. | Trains CartPole with (`τ=0.005`) and without (`τ=1.0`, target≡online) a target network and compares mean return over the 2nd half of training; the target network must win by a clear margin. Also plots max-Q stability vs the ~20 ceiling. |
| 4 | `test_random_baseline.py` | The trained policy learned something *useful about the task*, not just a decreasing loss. | Compares greedy rollouts to a uniform-random policy over 100 episodes; requires the gap to exceed 10 random-policy standard deviations. |
| 5 | `test_first10_trace.py` | States, Q-values, rewards and the policy interact as predicted, step by step. | Traces the first 10 greedy steps and checks (a) action = argmax Q, (b) max-Q sits just under the γ=0.95 ceiling (~20), (c) the action is positively correlated with the pole's "falling-right" indicator (the controller reacts to the pole). |
| 6 | `test_framestack_buffer.py` | The Atari frame-stack replay never mixes two episodes (the frame-stacking edge case). | Feeds a hand-built two-episode frame sequence and asserts the exact reconstructed 4-stack at every index, including that an episode-start stack is repeat-padded and never reaches back into the previous episode. |

### How these connect to the edge cases

* **Test 2** is the direct, quantitative demonstration of the `terminated` vs
  `truncated` edge case that `src/train_cartpole.py` and `src/train_mspacman.py`
  handle (store `done = terminated` only).
* **Test 1** indirectly exercises terminal masking too: state 4 → goal is a
  terminal transition whose Q\* equals the immediate reward with no bootstrap; a
  masking bug would show up as a wrong Q there.
* **Test 5 (b)** is the **Q-value ceiling** sanity check: with γ=0.95 and +1
  per step, no Q may meaningfully exceed `Σ γ^t ≈ 20`. The same ceiling is drawn
  on the max-Q training plots.
