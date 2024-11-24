import numpy as np
import gymnasium as gym
import random

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


class Network(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.layers

        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = nn.LeakyReLU()(x)
            x = self.layers[i](x)
        return x


class DQN:
    def __init__(
            self,
            state_dim: int,
            action_n: int,
            epsilon_decrease: float,
            gamma: float = 0.99,
            batch_size: int = 64,
            lr: float = 1e-4,
            epsilon_min: float = 1e-2):
        self.state_dim = state_dim
        self.action_n = action_n
        self.q_model = Network(self.state_dim, self.action_n, [64, 64])
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_min = epsilon_min
        self.epsilon = 1
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = []
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=lr)

    def get_action(self, state):
        q_values = self.q_model(torch.FloatTensor(state)).data.numpy()
        max_action = np.argmax(q_values)
        probs = np.ones(self.action_n) * self.epsilon / self.action_n
        probs[max_action] += 1 - self.epsilon
        return np.random.choice(np.arange(self.action_n), p=probs)

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.tensor, zip(*batch))

            targets = rewards + (1 - dones) * self.gamma * torch.max(self.q_model(next_states), dim=1).values
            q_values = self.q_model(states)[torch.arange(self.batch_size), actions]
            loss = torch.mean((q_values - targets) ** 2)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.epsilon = max(self.epsilon - self.epsilon_decrease, self.epsilon_min)


def evaluate(env: gym.Env, agent: DQN, t_max: int = 500, count: int = 100) -> float:
    total_reward = 0
    for _ in range(count):
        state, _ = env.reset()
        for _ in range(t_max):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            state = next_state
    return total_reward / count


def main():
    env = gym.make('Acrobot-v1')
    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n

    agent = DQN(state_dim, action_n, epsilon_decrease=1e-4)

    trajectory_n = 5000
    trajectory_len = 500

    output = []
    for trajectory in tqdm(range(trajectory_n)):
        total_reward = 0
        state, _ = env.reset()
        for t in range(trajectory_len):
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            agent.fit(state, action, reward, done, next_state)

            state = next_state

            if done:
                break

        if trajectory % 100 == 0:
            mean_reward = evaluate(env, agent, 500, 100)
            output.append({
                "epoch": trajectory,
                "reward": mean_reward
            })

    pd.DataFrame(output).to_csv("task1.csv", index=False)


if __name__ == "__main__":
    main()
