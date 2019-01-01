import torch


def agents_to_global(x):
    assert len(x.shape) == 3
    width = x.shape[2]
    n_agents = x.shape[1]
    reversed_agents = list(reversed(range(n_agents)))
    result = torch.cat([
        x.reshape(-1, width * 2),
        x[:, reversed_agents, :].reshape(-1, width * 2)
    ], dim=0)
    assert result.shape == (x.shape[0] * 2, width * 2)
    return result


def global_to_agents(x):
    x = x.reshape(2, -1)
    x = x.transpose(0, 1)
    x = x.reshape(-1, 2, 1)
    return x


def unpack_agents(x):
    states_size = x.shape[2]
    return x.reshape(-1, states_size)


def pack_agents(n, x):
    return x.reshape(-1, n, x.shape[1])
