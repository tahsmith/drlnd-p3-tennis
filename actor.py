import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_size, action_size, action_range=None):
        super(Actor, self).__init__()
        hidden_units = 256

        if action_range is None:
            action_range = [[-1, 1] for _ in range(action_size)]
        action_range = np.array(action_range)
        self.action_0 = torch.from_numpy(action_range[:, 0]).float()
        self.action_range = torch.from_numpy(
            np.diff(action_range, axis=1)[:, 0]).float()

        self.dropout = nn.Dropout()
        self.pi = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=400),
            nn.ELU(),
            self.dropout,
            nn.Linear(in_features=400, out_features=300),
            nn.ELU(),
            nn.Linear(in_features=300, out_features=action_size),
        )

    def forward(self, state):
        x = self.pi(state)
        x = F.tanh(x)
        return x

    def to(self, *args, **kwargs):
        self.action_0 = self.action_0.to(*args, **kwargs)
        self.action_range = self.action_range.to(*args, **kwargs)
        return super(Actor, self).to(*args, **kwargs)
