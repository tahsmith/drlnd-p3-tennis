from typing import Dict

import torch

import ddpg
import a2c


def agent_map(
        brain_map: Dict[str, tuple],
        device: torch.device
):
    def agent_fn(action_type):
        if action_type == 'continuous':
            return ddpg.default_agent
        else:
            return a2c.default_agent

    return {
        k: agent_fn(action_type)(device, n_agent, state_size, action_size)
        for k, (n_agent, state_size, action_type, action_size)
        in brain_map.items()
    }
