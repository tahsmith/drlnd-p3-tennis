import sys

import torch
from unityagents import UnityEnvironment

from agent import default_agent
from unity_env import wrap_env, get_agent_requirements


def main(argv):
    env_path = argv[1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = UnityEnvironment(file_name=env_path)
    brain_name, state_size, action_size = get_agent_requirements(env)

    agents = [
        default_agent(device, state_size, action_size)
        for _ in range(2)
    ]

    for i, agent in enumerate(agents):
        agent.restore('best-{i}'.format(i=i))

    episode_fn = wrap_env(env, brain_name, train=False)

    return run(episode_fn, agents)


def run(episode_fn, agent):
    scores = []
    while True:
        try:
            scores.append(episode_fn(agent))
        except KeyboardInterrupt:
            break
    return scores


if __name__ == '__main__':
    main(sys.argv)
