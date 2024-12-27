from copy import deepcopy
import random

import gymnasium as gym
import pandas as pd

import torch
from torch import nn
from torch.distributions import Normal


class SAC(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=1e-3, tau=1e-2,
                 batch_size=64, pi_lr=1e-3, q_lr=1e-3):
        super().__init__()

        self.pi_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU(),
                                      nn.Linear(128, 2 * action_dim), nn.Tanh())

        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU(),
                                      nn.Linear(128, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU(),
                                      nn.Linear(128, 1))

        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.memory = []

        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), pi_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1_model.parameters(), q_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2_model.parameters(), q_lr)
        self.q1_target_model = deepcopy(self.q1_model)
        self.q2_target_model = deepcopy(self.q2_model)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.predict_actions(state)
        return action.squeeze(1).detach().numpy()

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch))
            rewards, dones = rewards.unsqueeze(1), dones.unsqueeze(1)

            next_actions, next_log_probs = self.predict_actions(next_states)
            next_states_and_actions = torch.concatenate((next_states, next_actions), dim=1)
            next_q1_values = self.q1_target_model(next_states_and_actions)
            next_q2_values = self.q2_target_model(next_states_and_actions)
            next_min_q_values = torch.min(next_q1_values, next_q2_values)
            targets = rewards + self.gamma * (1 - dones) * (next_min_q_values - self.alpha * next_log_probs)

            states_and_actions = torch.concatenate((states, actions), dim=1)
            q1_loss = torch.mean((self.q1_model(states_and_actions) - targets.detach()) ** 2)
            q2_loss = torch.mean((self.q2_model(states_and_actions) - targets.detach()) ** 2)
            self.update_model(q1_loss, self.q1_optimizer, self.q1_model, self.q1_target_model)
            self.update_model(q2_loss, self.q2_optimizer, self.q2_model, self.q2_target_model)

            pred_actions, log_probs = self.predict_actions(states)
            states_and_pred_actions = torch.concatenate((states, pred_actions), dim=1)
            q1_values = self.q1_model(states_and_pred_actions)
            q2_values = self.q2_model(states_and_pred_actions)
            min_q_values = torch.min(q1_values, q2_values)
            pi_loss = - torch.mean(min_q_values - self.alpha * log_probs)
            self.update_model(pi_loss, self.pi_optimizer)

    def update_model(self, loss, optimizer, model=None, target_model=None):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if model is not None and target_model is not None:
            for param, target_param in zip(model.parameters(), target_model.parameters()):
                new_target_param = (1 - self.tau) * target_param + self.tau * param
                target_param.data.copy_(new_target_param)

    def predict_actions(self, states):
        means, log_stds = self.pi_model(states).T
        means, log_stds = means.unsqueeze(1), log_stds.unsqueeze(1)
        dists = Normal(means, torch.exp(log_stds))
        actions = dists.rsample()
        log_probs = dists.log_prob(actions)
        return actions, log_probs


def evaluate(env: gym.Env, agent: SAC, episodes_n: int = 1000) -> float:
    total_reward = 0
    for episode in range(episodes_n):
        state, _ = env.reset()
        for t in range(200):
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(2 * action)
            done = terminated or truncated
            if done:
                break
            total_reward += reward
    return total_reward / episodes_n


def main():
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SAC(state_dim, action_dim)

    episode_n = 100

    output = []
    for epoch in range(episode_n):
        state, _ = env.reset()
        for t in range(200):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(2 * action)
            done = terminated or truncated
            agent.fit(state, action, reward, done, next_state)
            if done:
                break
            state = next_state
        reward = evaluate(env, agent, 100)
        print(f"{epoch}: {reward}")
        output.append({
            "epoch": epoch,
            "reward": reward,
        })
    pd.DataFrame(output).to_csv("task1.csv", index=False)


if __name__ == "__main__":
    main()
