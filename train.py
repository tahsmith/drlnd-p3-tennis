import sys

import torch
from unityagents import UnityEnvironment

from agent import default_agent
from unity_env import wrap_env, get_agent_requirements, unity_episode


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

    return train(env, agent_map)


def train(env, agent_map, window_size=100, max_eps=int(2e5),
          min_score=1.0):
    scores = []
    best_score = float('-inf')
    steps = 0

    for i in range(max_eps):
        t, score = unity_episode(env, agent_map)
        steps += t
        scores.append(score)

        print_stats(i, score, steps, best_score, False)

        if (i + 1) % window_size == 0:
            avg_score = sum(scores[-window_size:]) / window_size

            if avg_score > best_score:
                best_score = avg_score
                for name, agent in agent_map.items():
                    agent.save('{name}-best'.format(name=name))

            print_stats(i, avg_score, steps, best_score, True)

            if avg_score > min_score:
                break
    return scores


def print_stats(i, score, steps, best_score, endl):
    print('\rep = {i:9d}, total steps = {t:9d}, score = {score:5.2f}, best = '
          '{best:5.2f} {star}'.
          format(i=i + 1, t=steps, score=score, best=best_score,
                 star='*' if best_score == score else ' '),
          end='\n' if endl else '')


if __name__ == '__main__':
    main(sys.argv)
