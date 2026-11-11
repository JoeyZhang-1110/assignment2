# Assignment 2 — Deep Q-Network (CartPole-v1 & MsPacman-v5)

A single, shared DQN implementation (neural network + experience replay + target
network) applied to two environments:

* **CartPole-v1** — discount factor **γ = 0.95**; solved by balancing the pole for
  500 steps.
* **ALE/MsPacman-v5** — discount factor **γ = 0.99**; image-based, 9 actions; the
  goal is to maximize game score and, above all, to **show learning/improvement**
  over training.

> **GitHub:** username `JoeyZhang-1110` · repo `assignment2` ·
> link https://github.com/JoeyZhang-1110/assignment2

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

Requires Python 3.10+, a CUDA GPU for MsPacman (CartPole runs fine on CPU).
Atari ROMs ship with `ale-py` ≥ 0.10, so no extra ROM install is needed.

Verify the environments load:

```bash
python -c "import gymnasium as gym, ale_py; gym.register_envs(ale_py); \
print(gym.make('CartPole-v1').action_space, gym.make('ALE/MsPacman-v5').action_space)"
```

## 2. Repository layout

```
src/
  utils.py            seeding, epsilon schedule, moving average, Q-ceiling, JSON logger
  replay_buffer.py    ReplayBuffer (vectors) + FrameStackReplayBuffer (Atari, boundary-safe)
  networks.py         MLPQNetwork (CartPole) + CNNQNetwork (Nature-DQN for MsPacman)
  dqn_agent.py        the DQN agent: epsilon-greedy, target net (hard/soft), Huber learn step
  envs.py             env construction + the handout's preprocess_observation + FrameStacker
  train_cartpole.py   CartPole training (γ=0.95)   -> model + JSON log
  train_mspacman.py   MsPacman training (γ=0.99)   -> model + JSON log
  evaluate.py         requirement (iii): N-episode rollout + reward histogram + mean/std
  plots.py            requirements (i) & (ii): max-Q curve and reward+MA100 curve
  nonstandard_plot.py the non-standard visualization (value-landscape animation)
tests/                five validation experiments (see tests/README.md)
results/              logs, checkpoints and all generated figures
report/report.md      the written report (answers + figures + discussion)
report/report.pdf     the report exported to PDF (for Canvas); rebuild with report/build_pdf.py
```

## 3. How to run

### CartPole (fast — minutes on CPU)

```bash
python -m src.train_cartpole --episodes 1000 --seed 0     # train -> results/cartpole/
python -m src.plots  --log results/cartpole/cartpole_log.json --out-dir results/cartpole
python -m src.evaluate --game cartpole --episodes 500      # 500-episode rollout histogram
python -m src.nonstandard_plot --episodes 300              # value-landscape GIF + figure
```

### MsPacman (long — thousands of episodes on a GPU)

```bash
# Full training as the handout describes (checkpoints every 50 episodes so a long
# run can be inspected / resumed). Expect several hours; 4000-6000 episodes.
python -m src.train_mspacman --episodes 4000 --seed 0

# Smoke test (seconds) to confirm the pipeline end-to-end:
python -m src.train_mspacman --episodes 5 --warmup 500 --buffer 5000

python -m src.plots    --log results/mspacman/mspacman_log.json --out-dir results/mspacman
python -m src.evaluate --game mspacman --episodes 500 --epsilon 0.05
```

### Validation tests

```bash
python tests/test_known_q.py          # converge to an exact, known Q*
python tests/test_truncation.py       # terminated-vs-truncated edge case
python tests/test_target_network.py   # target-network ablation
python tests/test_random_baseline.py  # trained policy >> random
python tests/test_first10_trace.py    # first-10-step state/value/policy trace
```

See [tests/README.md](tests/README.md) for what each test checks and how.

## 4. Required outputs (both games)

* **(i) max Q-value vs episode** — `*_maxq.png` (CartPole overlays the γ=0.95
  theoretical ceiling ≈ 20).
* **(ii) reward vs episode + 100-episode moving average** — `*_rewards.png`
  (CartPole also overlays a clean greedy-evaluation curve).
* **(iii) 500-episode rollout histogram + mean/std** — `*_rollout_hist.png`
  and `*_rollout_stats.json`.
* **Non-standard plot** — `cartpole_value_landscape.gif` and
  `cartpole_decision_landscape.png`: the learned value surface and decision
  boundary forming over the (pole-angle × angular-velocity) state plane.

## 5. Key implementation choices (and edge cases)

* **Terminal vs. time-limit:** the TD target bootstraps on `terminated` only,
  never on `truncated`. CartPole-v1 *truncates* at 500 steps — that is a time
  limit, not a failure — so masking it would cap learned Q far below the true
  ~20. `tests/test_truncation.py` quantifies the bug this avoids.
* **Frame stacking across episode boundaries:** `FrameStackReplayBuffer` stores
  single frames and rebuilds 4-stacks on sampling, stopping at episode starts so
  a stack never mixes two episodes (and uses 4× less memory than storing stacks).
* **Q-value ceiling as a sanity bound:** with γ=0.95, `Σ γ^t ≈ 20`; the max-Q
  plot overlays this ceiling and learned values approach it from below.
* **Stability aids:** Huber loss, gradient-norm clipping, soft (Polyak) target
  updates for CartPole, and for MsPacman an ε-scheduler, an LR scheduler and
  reward clipping (raw score is still tracked for the reward plots).


