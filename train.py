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

    episode_fn = wrap_env(env, brain_name)

    agent = default_agent(device, state_size, action_size)

    return train(episode_fn, agent)


def train(episode_fn, agent, window_size=100, max_eps=int(2e5),
          min_score=0.5):
    scores = []
    best_score = float('-inf')
    steps = 0

    for i in range(max_eps):
        t, score = episode_fn(agent)
        steps += t
        scores.append(score)

        print_stats(i, score, steps, best_score, False)

        if (i + 1) % window_size == 0:
            avg_score = sum(scores[-window_size:]) / window_size

            print_stats(i, avg_score, steps, best_score, True)

            if avg_score > best_score:
                best_score = avg_score
                agent.save('best')

            if avg_score > min_score:
                break
    return scores


def print_stats(i, score, steps, best_score, endl):
    print('\rep = {i:9d}, total steps = {t:9d}, score = {score:5.2f}, best = '
          '{best:5.2f}'.
          format(i=i + 1, t=steps, score=score, best=best_score),
          end='\n' if endl else '')


if __name__ == '__main__':
    main(sys.argv)
