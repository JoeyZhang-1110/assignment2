"""Deep Q-Network implementation for CartPole-v1 and MsPacman-v5.

Modules
-------
utils           : seeding, schedules, moving average, logging helpers.
replay_buffer   : uniform replay buffer (vector states) + frame-stack buffer (Atari).
networks        : MLP Q-network (CartPole) and Nature-CNN Q-network (MsPacman).
dqn_agent       : the DQN agent (epsilon-greedy policy, target network, Huber learn step).
envs            : environment construction + MsPacman preprocessing / frame stacking.
train_cartpole  : CartPole training entry point.
train_mspacman  : MsPacman training entry point.
evaluate        : roll out N episodes with a trained network, build the reward histogram.
plots           : build the three required plots from a training log.
nonstandard_plot: the non-standard CartPole value-landscape visualization.
"""
