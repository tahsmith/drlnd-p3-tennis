import numpy as np


class Agent:
    def __init__(self, action_size):
        self.action_size = action_size

    def policy(self, state, train=False):
        return np.random.uniform(-1, 1, size=(state.shape[0], self.action_size))

    def step(self, state, action, reward, next_state, done):
        pass

    def end_of_episode(self, score):
        pass

    def save(self, path):
        pass

    def restore(self, path):
        pass


def default_agent(device, state_size, action_size):
    return Agent(action_size)