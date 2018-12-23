import pytest
import torch
import numpy as np

from agent import (
    Agent
)


@pytest.fixture
def agent(device):
    return Agent(device, 2, 3, batch_size=10)


def test_policy(device, agent):
    state = np.random.randn(5, 2)
    action = agent.policy(state)
    assert action.shape == (5, 3)


def test_learn(agent: Agent):
    for i in range(20):
        agent.replay_buffer.add(np.zeros(2), np.zeros(3), 0.0, np.zeros(2), 1)
    agent.learn()


def test_save_restore(agent, tmpdir):
    path = str(tmpdir + '/model')
    agent.save(path)
    agent.restore(path)
