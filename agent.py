import copy
import random

import torch
import torch.optim
import numpy as np

from critic import Critic
from actor import Actor
from replay_buffer import ReplayBuffer


def pack_actors(x):
    return x.reshape(x.shape[0] // 2, -1)


def unpack_actors(x):
    return x.reshape(x.shape[0] * 2, -1)


def repeat_actors(x):
    return unpack_actors(x.repeat(1, 2))


def swap_actors(x):
    width = x.shape[1]
    x = x.reshape(-1, 2, width // 2)
    x = x[:, [1, 0], :]
    return x.reshape(-1, width)


def select_actor(n, x):
    m = x.shape[0]
    x = x.reshape(m // 2, 2, -1)
    x = x[:, n, :]
    return x


class Agent:
    def __init__(self, device, state_size, action_size, buffer_size=10,
                 batch_size=10,
                 actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3,
                 discount_rate=0.99,
                 tau=0.1,
                 steps_per_update=4,
                 action_range=None,
                 dropout_p=0.0,
                 weight_decay=0.0001,
                 noise_max=0.2,
                 noise_decay=1.0,
                 n_agents=1
                 ):
        self.device: torch.device = device
        self.state_size = state_size
        self.action_size = action_size

        self.critic_control = Critic(state_size, action_size).to(device)
        self.critic_control.dropout.p = dropout_p
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_target.eval()
        self.critic_optimizer = torch.optim.Adam(
            self.critic_control.parameters(),
            weight_decay=weight_decay,
            lr=critic_learning_rate)

        self.actor_control = Actor(state_size, action_size).to(
            device)
        self.actor_control.dropout.p = dropout_p
        self.actor_target = Actor(state_size, action_size).to(
            device)
        self.actor_target.eval()
        self.actor_optimizer = torch.optim.Adam(
            self.actor_control.parameters(),
            weight_decay=weight_decay,
            lr=actor_learning_rate)

        self.batch_size = batch_size
        self.min_buffer_size = batch_size
        self.replay_buffer = ReplayBuffer(device,
                                          state_size * 2, action_size * 2,
                                          buffer_size)

        self.discount_rate = discount_rate

        self.tau = tau

        self.step_count = 0
        self.steps_per_update = steps_per_update

        self.noise_max = noise_max
        self.noise = OUNoise([n_agents, action_size], 15071988, sigma=self.noise_max)
        self.noise_decay = noise_decay
        self.last_score = float('-inf')

    def policy(self, state, training=True):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_control.eval()
        with torch.no_grad():
            action = self.actor_control(state).cpu().numpy()
        self.actor_control.train()
        if training:
            noise = self.noise.sample()
            action += noise
        return action

    def step(self, state, action, reward, next_state, done):
        # p = self.calculate_p(state, action, reward, next_state, done)

        self.replay_buffer.add(
            pack_actors(state),
            pack_actors(action),
            pack_actors(reward[:, np.newaxis]),
            pack_actors(next_state),
            pack_actors(done[:, np.newaxis])
        )
        if self.step_count % self.steps_per_update == 0:
            self.learn()
        self.step_count += 1

    def learn(self):
        if len(self.replay_buffer) < self.min_buffer_size:
            return
        indicies, (states, actions, rewards, next_states, dones, p) = \
            self.replay_buffer.sample(self.batch_size)

        self.actor_control.eval()
        error = self.bellman_eqn_error(
            states, actions, rewards, next_states, dones)
        self.actor_control.train()

        self.critic_optimizer.zero_grad()
        loss = (error ** 2).sum() / self.batch_size
        loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        unpacked_states = unpack_actors(states)
        expected_actions = self.actor_control(unpacked_states)
        critic_scores = []
        for i in range(0, 2):
            critic_scores.append(
                self.critic_control(states, select_actor(i, expected_actions))
            )
            states = swap_actors(states)

        critic_scores = torch.cat(critic_scores, dim=-1)
        loss = -1 * critic_scores.sum() / self.batch_size
        loss.backward()
        self.actor_optimizer.step()

        self.update_target(self.critic_control, self.critic_target)
        self.update_target(self.actor_control, self.actor_target)

        # self.replay_buffer.update(indicies, error.detach().abs().cpu() + 1e-3)

    def bellman_eqn_error_for_actor(self, all_states, actions, rewards,
                                    all_next_states, next_actions, dones):
        target_action_values = self.critic_target(all_next_states, next_actions)

        target_rewards = (
                rewards
                + self.discount_rate * (1 - dones) * target_action_values
        )

        current_rewards = self.critic_control(all_states, actions)
        error = current_rewards - target_rewards
        return error

    def bellman_eqn_error(self, states, actions, rewards, next_states, dones):
        """Double DQN error - use the control network to get the best action
        and apply the target network to it to get the target reward which is
        used for the bellman eqn error.
        """
        actions = unpack_actors(actions)
        rewards = unpack_actors(rewards)
        next_states_unpacked = unpack_actors(next_states)
        dones = unpack_actors(dones)

        next_actions = self.actor_target(next_states_unpacked)
        errors = []

        for i in range(0, 2):
            error = self.bellman_eqn_error_for_actor(
                states,
                select_actor(i, actions),
                select_actor(i, rewards),
                next_states,
                select_actor(i, next_actions),
                select_actor(i, dones),
            )

            errors.append(error)
            states = swap_actors(states)
            next_states = swap_actors(next_states)

        return torch.cat(errors, dim=-1)

    def calculate_p(self, state, action, reward, next_state, done):
        next_state = torch.from_numpy(next_state).float().to(
            self.device)
        state = torch.from_numpy(state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        done = torch.from_numpy(done).float().to(
            self.device)

        done = done.unsqueeze(1)
        reward = reward.unsqueeze(1)

        self.actor_control.eval()
        self.critic_control.eval()

        with torch.no_grad():
            retval = abs(
                self.bellman_eqn_error(state, action, reward, next_state,
                                       done)) + 1e-3
        self.critic_control.train()
        self.actor_control.train()
        return retval

    def update_target(self, control, target):
        for target_param, control_param in zip(
                target.parameters(),
                control.parameters()):
            target_param.data.copy_(
                self.tau * control_param.data + (1.0 - self.tau) *
                target_param.data)

    def end_of_episode(self, final_score):
        self.step_count = 0

        self.noise.sigma *= self.noise_decay
        self.last_score = final_score
        self.noise.reset()

    def save(self, path):
        torch.save(self.critic_control.state_dict(), path + '-critic.p')
        torch.save(self.actor_control.state_dict(), path + '-actor.p')

    def restore(self, path):
        self.critic_control.load_state_dict(
            torch.load(path + '-critic.p', map_location='cpu'))
        self.actor_control.load_state_dict(
            torch.load(path + '-actor.p', map_location='cpu'))


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(
            size=self.mu.shape)
        self.state = x + dx
        return self.state


def default_agent(device, state_size, action_size):
    return Agent(
        device,
        state_size,
        action_size,
        buffer_size=int(1e5),
        batch_size=1028,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-2,
        discount_rate=0.99,
        tau=1e-2,
        steps_per_update=5,
        weight_decay=0.001,
        noise_decay=1.0,
        noise_max=0.2,
        dropout_p=0.0,
        n_agents=2
    )
