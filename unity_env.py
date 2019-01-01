from functools import partial
from typing import Dict

import numpy as np

from ddpg import Agent


def get_agent_requirements(env):
    env_info = env.reset(train_mode=True)

    agent_requirements = {}

    for brain_name in env.brain_names:
        info = env_info[brain_name]
        brain = env.brains[brain_name]

        action_type = brain.vector_action_space_type
        action_size = brain.vector_action_space_size
        n_agents = len(info.vector_observations)
        state_size = len(info.vector_observations[0])

        agent_requirements[brain_name] = (n_agents, state_size, action_type,
                                          action_size)

    return agent_requirements


def apply(f, agent_map, env_info):
    return {
        k: f(agent_map[k], v)
        for k, v in env_info.items()
    }


policy = partial(apply, lambda a, v: a.policy(v.vector_observations))


def split_experience(state, action, env_info):
    experience = {}
    for k, v in env_info.items():
        next_state = np.array(v.vector_observations)
        reward = np.array(v.rewards)
        done = np.array(v.local_done, dtype=np.uint8)
        experience[k] = (state[k], action[k], reward, next_state, done)

    return experience


step = partial(apply, lambda a, v: a.step(*v))

end_of_episode = partial(apply, lambda a, v: a.end_of_episode(v))


def unity_episode(env, agent_map: Dict[str, Agent], max_t=10000,
                  train=True):
    assert max_t > 0
    env_info = env.reset(train_mode=train)
    state = {k: v.vector_observations for k, v in env_info.items()}
    score = {k: 0.0 for k in agent_map.keys()}
    for t in range(max_t):
        action = policy(agent_map, env_info)
        env_info = env.step(action.copy())
        experience = split_experience(state, action, env_info)
        if train:
            step(agent_map, experience)
        state = {k: v[3] for k, v in experience.items()}
        score = {k: v + experience[k][2] for k, v in score.items()}

        if any(v[4].any() for v in experience.values()):
            break

    end_of_episode(agent_map, score)
    return t + 1, max(brain_values.max() for brain_values in score.values())


def wrap_env(env, brain_name, train=True):
    return partial(unity_episode, env, agent_map=brain_name, train=train)
