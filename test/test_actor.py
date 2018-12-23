import torch

from actor import Actor


def test_forward():
    actor = Actor(2, 3)
    state = torch.randn(7, 2)
    action = actor(state)
    assert action.shape == (7, 3)
