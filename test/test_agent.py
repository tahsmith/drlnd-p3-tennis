from unittest.mock import Mock

import numpy as np
import torch
from numpy.testing import assert_allclose

from agent import (
    pack_actors,
    unpack_actors,
    repeat_actors,
    swap_actors,
    select_actor,
    default_agent
)


def test_pack():
    unpacked = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])

    actual = pack_actors(torch.from_numpy(unpacked))

    expected = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8]
    ])

    assert (actual == torch.from_numpy(expected)).all()
    assert actual.shape == expected.shape


def test_unpack():
    packed = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8]
    ])

    actual = unpack_actors(torch.from_numpy(packed))

    expected = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])

    assert (actual == torch.from_numpy(expected)).all()
    assert actual.shape == expected.shape


def test_repeat_actors():
    single = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])

    actual = repeat_actors(torch.from_numpy(single))

    expected = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [5, 6, 7, 8]
    ])

    assert (actual == torch.from_numpy(expected)).all()
    assert actual.shape == expected.shape


def test_swap_actors():
    packed = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])

    actual = swap_actors(torch.from_numpy(packed))

    expected = np.array([
        [3, 4, 1, 2],
        [7, 8, 5, 6]
    ])

    assert (actual == torch.from_numpy(expected)).all()
    assert actual.shape == expected.shape


def test_select_actors():
    test = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ])

    actual = select_actor(0, torch.from_numpy(test))

    expected = np.array([
        [1, 2],
        [5, 6]
    ])

    assert (actual == torch.from_numpy(expected)).all()
    assert actual.shape == expected.shape

    actual = select_actor(1, torch.from_numpy(test))

    expected = np.array([
        [3, 4],
        [7, 8]
    ])

    assert (actual == torch.from_numpy(expected)).all()
    assert actual.shape == expected.shape


def test_agent():
    agent = default_agent(torch.device('cpu'), 2, 3)

    states = np.array([
        [1, 2],
        [3, 4]
    ])

    actions = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    next_states = states * 2

    rewards = np.array([0, 1])

    dones = np.array([0, 1])

    agent.step(
        states,
        actions,
        rewards,
        next_states,
        dones
    )

    expected_global_state = np.array(
        [1, 2, 3, 4]
    )

    expected_global_actions = np.array(
        [1, 2, 3, 4, 5, 6]
    )

    expected_global_next_state = expected_global_state * 2

    expected_global_rewards = np.array(
        [0, 1]
    )

    expected_global_dones = np.array(
        [0, 1]
    )

    assert_allclose(agent.replay_buffer.state[0, :],
                    expected_global_state)
    assert_allclose(agent.replay_buffer.action[0, :],
                    expected_global_actions)
    assert_allclose(agent.replay_buffer.next_state[0, :],
                    expected_global_next_state)
    assert_allclose(agent.replay_buffer.reward[0, :],
                    expected_global_rewards)
    assert_allclose(agent.replay_buffer.done[0, :],
                    expected_global_dones)
