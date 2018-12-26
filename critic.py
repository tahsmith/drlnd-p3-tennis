import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.dropout = nn.Dropout()

        self.state_layer = nn.Sequential(
            nn.Linear(in_features=state_size * 2, out_features=800),
            nn.ELU(),
            self.dropout,
        )
        self.qa = nn.Sequential(
            nn.Linear(in_features=800 + action_size,
                      out_features=600, bias=False),
            nn.BatchNorm1d(600),
            nn.ELU(),
            self.dropout,

            nn.Linear(in_features=600,
                      out_features=300, bias=False),
            nn.BatchNorm1d(300),
            nn.ELU(),
            nn.Linear(in_features=300, out_features=1),
        )

    def forward(self, state, action):
        x = self.state_layer(state)
        x = torch.cat([x, action], dim=1)
        return self.qa(x)
