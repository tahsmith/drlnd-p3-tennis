import sys
from logging import warning

import torch
from unityagents import UnityEnvironment

from agent import default_agent
from unity_env import get_agent_requirements, unity_episode


def main(argv):
    env_path = argv[1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = UnityEnvironment(file_name=env_path)
    brain_map = get_agent_requirements(env)

    agent_map = {
        k: default_agent(device, n_agent, state_size, action_size)
        for k, (n_agent, state_size, action_size)
        in brain_map.items()
    }

    for name, agent in agent_map.items():
        try:
            agent.restore('{name}-best'.format(name=name))
        except FileNotFoundError:
            warning('Missing model data for {name}'.format(name=name))

    return run(env, agent_map)


def run(env, agent_map):
    scores = []
    while True:
        try:
            scores.append(unity_episode(env, agent_map, train=False))
        except KeyboardInterrupt:
            break
    return scores


if __name__ == '__main__':
    main(sys.argv)
