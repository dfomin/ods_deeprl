from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch.optim
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Step:
    state: int
    action: int
    reward: float


class Trajectory:
    def __init__(self, steps: list[Step]):
        self.steps = steps
        self.total_reward = sum([step.reward for step in self.steps])


class ActionStateDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        return state, action


class Network(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], lr=1e-4):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
        self.optimizer = torch.optim.Adam(self.parameters(), lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.layers

        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            # x = nn.ReLU()(x)
            x = nn.Tanh()(x)
            x = self.layers[i](x)
        return nn.Softmax(dim=1)(x)


class DeepCEMAgent:
    def __init__(self, state_dim: int, actions: int, quantile: float, epsilon_min: float = 1e-2, epsilon_decay: float = 0.99):
        self.actions = actions
        self.model = Network(state_dim, actions, [512, 512])
        self.epsilon = 1
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.loss = nn.CrossEntropyLoss()
        self.elite_quantile = quantile

    def get_action(self, state: np.ndarray, eval: bool = False) -> int:
        state = np.array([state])
        pred = self.model(torch.tensor(state))[0].detach().numpy()
        if eval:
            # print(pred)
            # return pred.argmax()
            return np.random.choice(range(self.actions), p=pred)
        probs = (1 - self.epsilon) * pred + self.epsilon / self.actions
        probs /= probs.sum()
        return np.random.choice(range(self.actions), p=probs)

    def fit(self, trajectories: list[Trajectory]):
        threshold = np.quantile([t.total_reward for t in trajectories], self.elite_quantile)
        elite_trajectories = [t for t in trajectories if t.total_reward > threshold]
        if not elite_trajectories:
            elite_trajectories = [t for t in trajectories if t.total_reward >= threshold]

        states = []
        actions = []
        for t in elite_trajectories:
            states.extend([s.state for s in t.steps])
            actions.extend([s.action for s in t.steps])

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))

        dataset = ActionStateDataset(states, actions)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        for states_batch, actions_batch in dataloader:
            self.model.optimizer.zero_grad()
            loss = self.loss(self.model(states_batch), actions_batch)
            loss.backward()
            self.model.optimizer.step()

        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon = 1 / (1 / self.epsilon + 1)


def train(env: gym.Env, agent: DeepCEMAgent, trajectories_count: int, max_steps: int, epochs: int):
    for epoch in range(epochs):
        trajectories = [play(env, agent, max_steps) for _ in range(trajectories_count)]
        agent.fit(trajectories)
        mean_reward = np.mean([evaluate(env, agent, max_steps) for _ in range(10)])
        print(f"epoch: {epoch}, mean reward: {mean_reward}")
        # play(gym.make("LunarLander-v2", render_mode="human"), agent, max_steps, True)


def play(env: gym.Env, agent: DeepCEMAgent, max_steps: int, greedy: bool = False) -> Trajectory:
    state = env.reset()[0]

    steps = []

    for _ in range(max_steps):
        action = agent.get_action(state, greedy)
        next_state, reward, terminated, truncated, _ = env.step(action)
        steps.append(Step(state, action, reward))

        if terminated or truncated:
            break
        state = next_state

    return Trajectory(steps)


def evaluate(env: gym.Env, agent: DeepCEMAgent, max_steps: int) -> float:
    state = env.reset()[0]
    total_reward = 0
    for _ in range(max_steps):
        action = agent.get_action(state, True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
        state = next_state
    return total_reward


def main():
    env = gym.make("LunarLander-v2")
    agent = DeepCEMAgent(env.observation_space.shape[0], env.action_space.n, 0.8)
    train(env, agent, 1000, 100, 100)


if __name__ == "__main__":
    main()
