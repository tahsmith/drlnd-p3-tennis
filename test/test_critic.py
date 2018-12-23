import torch

from critic import Critic


def test_forward():
    critic = Critic(2, 3)
    state = torch.randn(7, 2)
    action = torch.randn(7, 3)
    value = critic(state, action)
    assert value.shape == (7, 1)
