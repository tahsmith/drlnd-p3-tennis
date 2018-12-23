from unittest.mock import MagicMock

import pytest
from attr import attrs, attrib

from unity_env import unity_episode


@attrs
class EnvInfo:
    vector_observations = attrib()
    rewards = attrib()
    local_done = attrib()


@pytest.fixture
def env():
    mock_env = MagicMock()
    mock_env.step.return_value = {'brain':
        EnvInfo(
            vector_observations=[[0], [0]],
            rewards=[0, 1],
            local_done=[0, 1]
        )
    }

    return mock_env


@pytest.fixture
def agent():
    mock_agent = MagicMock()
    mock_agent.policy.return_value = [[1]]
    return mock_agent


def test_reacher_episode(env, agent):
    score = unity_episode(env, agent, 'brain')
    assert score == 1
