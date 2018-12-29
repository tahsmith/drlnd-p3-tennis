import copy
import random

import torch
import torch.optim
import numpy as np

from critic import Critic
from actor import Actor
from replay_buffer import ReplayBuffer


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


class Agent:
    def __init__(self, device, state_size, action_size, buffer_size=10,
                 batch_size=10,
                 actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3,
                 discount_rate=0.99,
                 tau=0.1,
                 steps_per_update=4,
                 dropout_p=0.0,
                 weight_decay=0.0001,
                 noise_max=0.2,
                 noise_decay=1.0,
                 n_agents=1
                 ):
        self.device: torch.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents

        def make_critic():
            critic = Critic(state_size * n_agents, action_size * n_agents)
            critic = critic.to(device)
            return critic

        self.critic_control = make_critic()
        self.critic_control.dropout.p = dropout_p
        self.critic_target = make_critic()
        self.critic_target.eval()
        self.critic_optimizer = torch.optim.Adam(
            self.critic_control.parameters(),
            weight_decay=weight_decay,
            lr=critic_learning_rate)

        self.actor_control = Actor(state_size, action_size).to(device)
        self.actor_control.dropout.p = dropout_p
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_target.eval()
        self.actor_optimizer = torch.optim.Adam(
            self.actor_control.parameters(),
            weight_decay=weight_decay,
            lr=actor_learning_rate)

        self.batch_size = batch_size
        self.min_buffer_size = batch_size
        self.replay_buffer = ReplayBuffer(device, state_size, action_size,
                                          buffer_size, n_agents)

        self.discount_rate = discount_rate

        self.tau = tau

        self.step_count = 0
        self.steps_per_update = steps_per_update

        self.noise_max = noise_max
        self.noise = OUNoise([n_agents, action_size], 15071988, sigma=self.noise_max)
        self.noise_decay = noise_decay

    def policy(self, state, training=True):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_control.eval()
        with torch.no_grad():
            action = self.actor_control(state).cpu().numpy()
        self.actor_control.train()
        if training:
            noise = self.noise.sample()
            action += noise
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        self.actor_control.noise(self.noise.sigma)
        # p = self.calculate_p(state, action, reward, next_state, done)

        self.replay_buffer.add(state, action, reward, next_state, done)
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

        importance_scaling = (self.replay_buffer.buffer_size
                              * p.unsqueeze(1).repeat(1, 2, 1)) ** -1
        importance_scaling /= importance_scaling.max()
        self.critic_optimizer.zero_grad()
        loss = (importance_scaling * (error ** 2)).sum() / self.batch_size
        loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        expected_actions = self.actor_control(unpack_agents(states))
        expected_actions = pack_agents(self.n_agents, expected_actions)
        critic_score = self.critic_control(
            agents_to_global(states),
            agents_to_global(expected_actions)
        )
        critic_score = global_to_agents(critic_score)
        loss = -1 * (importance_scaling * critic_score).sum() / self.batch_size
        loss.backward()
        self.actor_optimizer.step()

        self.update_target(self.critic_control, self.critic_target)
        self.update_target(self.actor_control, self.actor_target)

        # self.replay_buffer.update(indicies, error.detach().abs().cpu() + 1e-3)

    def bellman_eqn_error(self, states, actions, rewards, next_states, dones):
        """Double DQN error - use the control network to get the best action
        and apply the target network to it to get the target reward which is
        used for the bellman eqn error.
        """
        next_actions = self.actor_control(unpack_agents(next_states))
        next_actions = pack_agents(self.n_agents, next_actions)
        next_states_global = agents_to_global(next_states)
        next_actions_global = agents_to_global(next_actions)

        target_action_values = self.critic_target(next_states_global,
                                                  next_actions_global)
        target_action_values = global_to_agents(target_action_values)

        target_rewards = (
                rewards
                + self.discount_rate * (1 - dones) * target_action_values
        )

        states = agents_to_global(states)
        actions = agents_to_global(actions)

        current_rewards = self.critic_control(states, actions)
        current_rewards = global_to_agents(current_rewards)

        error = current_rewards - target_rewards
        return error

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
        batch_size=64,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        discount_rate=0.99,
        tau=1e-2,
        steps_per_update=1,
        weight_decay=0.00,
        noise_decay=0.9995,
        noise_max=0.2,
        dropout_p=0.0,
        n_agents=2
    )
