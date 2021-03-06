from functools import partial
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F


def initialise(m):
    noise(1.0, m)


def noise(sigma, m):
    if isinstance(m, nn.Linear):
        scale = sqrt(m.weight.data.shape[1])
        noise = torch.randn_like(m.weight.data)
        m.weight.data.add_(noise * sigma / scale)


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.dropout = nn.Dropout()
        self.pi = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=400, bias=False),
            nn.BatchNorm1d(400),
            nn.ELU(),
            self.dropout,
            nn.Linear(in_features=400, out_features=300, bias=False),
            nn.BatchNorm1d(300),
            nn.ELU(),
            nn.Linear(in_features=300, out_features=action_size, bias=False),
            nn.BatchNorm1d(action_size)
        )

    def forward(self, state):
        x = self.pi(state)
        x = F.tanh(x)
        return x

    def noise(self, sigma):
        self.pi.apply(partial(noise, sigma))
