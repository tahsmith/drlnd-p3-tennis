import numpy as np
import torch

from replay_buffer import ReplayBuffer


def test_replay_buffer(device):
    replay_buffer = ReplayBuffer(device, 2, 3, 5)

    for i in range(10):
        replay_buffer.add(np.zeros(2), np.zeros(3), 0.0, np.zeros(2), 0)

    indices, (states, actions, rewards, next_states, dones,
              p) = replay_buffer.sample(5)
    assert (5, 2) == states.shape
    assert (5, 3) == actions.shape
    assert (5, 1) == rewards.shape
    assert (5, 2) == next_states.shape
    assert (5, 1) == dones.shape

    for i in range(10):
        replay_buffer.add(np.zeros(2), np.zeros(3), 0.0, np.zeros(2), 0)

    indices, (states, actions, rewards, next_states, dones,
              p) = replay_buffer.sample(5)
    assert (5, 2) == states.shape
    assert (5, 3) == actions.shape
    assert (5, 1) == rewards.shape
    assert (5, 2) == next_states.shape
    assert (5, 1) == dones.shape
