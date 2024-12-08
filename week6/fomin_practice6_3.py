import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.99, batch_size=128,
                 epsilon=0.2, epoch_n=30, pi_lr=1e-4, v_lr=5e-4, q_advantage: bool = False):
        super().__init__()

        self.pi_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU(),
                                      nn.Linear(128, action_dim), nn.Softmax())

        self.v_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, 1))

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)
        self.q_advantage = q_advantage

    def get_action(self, state):
        probs = self.pi_model(torch.FloatTensor(state).unsqueeze(0))
        dist = Categorical(probs)
        action = dist.sample()
        return action.numpy()[0]

    def fit(self, states, actions, rewards, dones):
        states, actions, rewards, dones = map(np.array, [states, actions, rewards, dones])
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        returns = np.zeros(rewards.shape)
        returns[-1] = rewards[-1]
        for t in range(returns.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

        states, actions, returns, rewards = map(torch.FloatTensor, [states, actions, returns, rewards])

        probs = self.pi_model(states)
        dist = Categorical(probs)
        old_log_probs = dist.log_prob(actions).detach()

        for epoch in range(self.epoch_n):
            indices = np.random.permutation(returns.shape[0])
            for i in range(0, returns.shape[0], self.batch_size):
                b_indices = indices[i: i + self.batch_size]
                b_states = states[b_indices]
                b_actions = actions[b_indices]
                b_returns = returns[b_indices]
                b_rewards = rewards[b_indices]
                b_old_log_probs = old_log_probs[b_indices]

                if self.q_advantage:
                    v_next = torch.FloatTensor([self.v_model(states[index + 1]) if not dones[index] and index + 1 < len(states) else 0 for index in b_indices]).unsqueeze(1)
                    b_advantage = b_rewards + self.gamma * v_next - self.v_model(b_states)
                else:
                    b_advantage = b_returns.detach() - self.v_model(b_states)

                b_probs = self.pi_model(b_states)
                b_dist = Categorical(b_probs)
                b_new_log_probs = b_dist.log_prob(b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantage.detach()
                pi_loss_2 = torch.clamp(b_ratio, 1. - self.epsilon, 1. + self.epsilon) * b_advantage.detach()
                pi_loss = -torch.mean(torch.min(pi_loss_1, pi_loss_2))

                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = torch.mean(b_advantage ** 2)

                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()


def evaluate(env: gym.Env, agent: PPO, episodes_n: int = 100, t_max: int = 100) -> float:
    total_reward = 0
    for episode in range(episodes_n):
        state, _ = env.reset()
        for t in range(t_max):
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
            if episode == episodes_n - 1:
                env.render()
    return total_reward / episodes_n


def train(env: gym.Env, agent: PPO) -> list[float]:
    episode_n = 50
    trajectory_n = 20
    t_max = 500

    total_rewards = []

    for episode in range(episode_n):
        states, actions, rewards, dones = [], [], [], []
        for _ in range(trajectory_n):
            state, _ = env.reset()
            for t in range(t_max):
                states.append(state)
                action = agent.get_action(state)
                actions.append(action)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                rewards.append(reward)
                dones.append(done)
                if done:
                    break
        agent.fit(states, actions, rewards, dones)
        reward = evaluate(env, agent, 100, t_max)
        total_rewards.append(reward)
        print(f"Episode: {episode}, reward: {reward}")
    return total_rewards


def main():
    env = gym.make('Acrobot-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    output = []
    rewards = train(env, PPO(state_dim, action_dim, q_advantage=False))
    for epoch, reward in enumerate(rewards):
        output.append({
            "epoch": epoch,
            "reward": reward,
            "type": "Monte Carlo"
        })
    rewards = train(env, PPO(state_dim, action_dim, q_advantage=True))
    for epoch, reward in enumerate(rewards):
        output.append({
            "epoch": epoch,
            "reward": reward,
            "type": "TD"
        })
    pd.DataFrame(output).to_csv("task3.csv", index=False)


if __name__ == "__main__":
    main()
