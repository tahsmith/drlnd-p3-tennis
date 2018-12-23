import torch
import numpy as np


def normalise(x):
    # y = np.exp(x - np.max(x))
    # return y / y.sum()
    return x / x.sum()


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, device, state_size, action_size, buffer_size):
        self.action_size = action_size

        self.last = 0
        self.p = np.zeros((buffer_size,), dtype=np.float32)
        self.state = np.empty((buffer_size, state_size), dtype=np.float32)
        self.action = np.empty((buffer_size, action_size), dtype=np.float32)
        self.reward = np.empty((buffer_size,), dtype=np.float32)
        self.next_state = np.empty((buffer_size, state_size), dtype=np.float32)
        self.done = np.empty((buffer_size,), dtype=np.uint8)

        self.device = device
        self.cursor = 0

    @property
    def buffer_size(self):
        return self.p.shape[0]

    def add(self, state, action, reward, next_state, done, p=1.0):
        """Add a new experience to memory."""
        if self.last < self.buffer_size:
            i = self.last
            self.last += 1
        else:
            i = np.argmin(self.p)
        #
        # i = self.cursor
        # self.cursor += 1
        # self.cursor %= self.buffer_size


        self.p[i] = p
        self.state[i, :] = state
        self.action[i, :] = action
        self.reward[i] = reward
        self.next_state[i, :] = next_state
        self.done[i] = done

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        p = normalise(self.p[:self.last])
        choices = np.random.choice(np.arange(self.last),
                                   size=(batch_size,),
                                   p=p)

        states = torch.from_numpy(self.state[choices, :]).float().to(
            self.device)
        actions = torch.from_numpy(self.action[choices, :]).float().to(
            self.device)
        rewards = torch.from_numpy(self.reward[choices, np.newaxis]).float().to(
            self.device)
        next_states = torch.from_numpy(self.next_state[choices, :]).float().to(
            self.device)
        dones = torch.from_numpy(self.done[choices, np.newaxis]).float().to(
            self.device)
        p = torch.from_numpy(p[choices, np.newaxis]).float().to(
            self.device)

        return choices, (states, actions, rewards, next_states, dones, p)

    def update(self, indices, p):
        self.p[indices] = p[:, 0]

    def __len__(self):
        """Return the current size of internal memory."""
        return self.last
