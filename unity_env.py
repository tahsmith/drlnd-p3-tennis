from functools import partial

import numpy as np

from agent import Agent


def get_agent_requirements(env):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    assert len(env_info.vector_observations) == 2, "Must be a two player game"
    state = env_info.vector_observations[0]
    state_size = len(state)

    return brain_name, state_size, action_size

def augment_state(state):
    # ids = np.array([[-1], [1]], dtype=np.float64)
    # return np.concatenate([state, ids], axis=1)
    return state


def unity_episode(env, agent: Agent, brain_name, max_t=10000, train=True):
    assert max_t > 0
    env_info = env.reset(train_mode=False)[brain_name]
    state = augment_state(np.array(env_info.vector_observations))
    score = np.array([0.0, 0.0])
    for t in range(max_t):
        action = agent.policy(state, train)
        env_info = env.step(action)[brain_name]
        next_state = augment_state(np.array(env_info.vector_observations))
        reward = np.array(env_info.rewards)
        done = np.array(env_info.local_done, dtype=np.uint8)
        if train:
            agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done.any():
            break

    agent.end_of_episode(score)
    return t + 1, score.max()


def wrap_env(env, brain_name, train=True):
    return partial(unity_episode, env, brain_name=brain_name, train=train)
