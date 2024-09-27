from abc import ABC, abstractmethod
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class Step:
    state: int
    action: int
    reward: float


class Trajectory:
    def __init__(self, steps: list[Step]):
        self.steps = steps
        self.total_reward = sum([step.reward for step in self.steps])


class Updater(ABC):
    @abstractmethod
    def update(self, old_policy: np.ndarray, statistics: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Normalizer(Updater):
    def __init__(self, ignore_zeros: bool = False):
        self.ignore_zeros = ignore_zeros

    def update(self, old_policy: np.ndarray, statistics: np.ndarray) -> np.ndarray:
        new_policy = np.zeros_like(old_policy)
        for s in range(len(statistics)):
            if np.count_nonzero(statistics[s]) == 0:
                if not self.ignore_zeros:
                    new_policy[s] = 1 / len(statistics[s])
            else:
                new_policy[s] = statistics[s] / statistics[s].sum()
        return new_policy


class LaplaceSmoothing(Updater):
    def __init__(self, alpha: float):
        assert alpha > 0

        self.alpha = alpha

    def update(self, old_policy: np.ndarray, statistics: np.ndarray) -> np.ndarray:
        new_policy = np.zeros_like(old_policy)
        for s in range(len(statistics)):
            for a in range(len(statistics[s])):
                new_policy[s, a] = (statistics[s, a] + self.alpha) / (sum(statistics[s]) + self.alpha * len(statistics[s]))
        return new_policy


class PolicySmoothing(Updater):
    def __init__(self, alpha: float):
        assert 0 < alpha <= 1

        self.alpha = alpha

    def update(self, old_policy: np.ndarray, statistics: np.ndarray) -> np.ndarray:
        new_policy = np.zeros_like(old_policy)
        statistics = Normalizer(True).update(old_policy, statistics)
        for s in range(len(old_policy)):
            if np.count_nonzero(statistics[s]) == 0:
                new_policy[s, :] = old_policy[s, :]
                continue

            new_policy[s, :] = self.alpha * statistics[s, :] + (1 - self.alpha) * old_policy[s, :]
        return new_policy


class CEMAgent:
    def __init__(self, states: int, actions: int, elite_quantile: float, updater: Updater):
        assert actions > 0
        assert 0 < elite_quantile < 1

        self.actions = actions
        self.updater = updater
        self.elite_quantile = elite_quantile
        self.policy = np.ones((states, actions)) / actions

    def get_action(self, state: int) -> int:
        return np.random.choice(self.actions, p=self.policy[state])

    def fit(self, trajectories: list[Trajectory]):
        threshold = np.quantile([t.total_reward for t in trajectories], self.elite_quantile)
        elite_trajectories = [t for t in trajectories if t.total_reward > threshold]
        if not elite_trajectories:
            elite_trajectories = [t for t in trajectories if t.total_reward >= threshold]
        statistics = np.zeros_like(self.policy)
        for trajectory in elite_trajectories:
            for step in trajectory.steps:
                statistics[step.state, step.action] += 1
        self.policy = self.updater.update(self.policy, statistics)


def train(env: gym.Env, agent: CEMAgent, trajectories_count: int, max_steps: int, epochs: int) -> list:
    result = []
    for epoch in range(epochs):
        trajectories = [play(env, agent, max_steps) for _ in range(trajectories_count)]
        agent.fit(trajectories)
        evaluation = [evaluate(env, agent, max_steps) for _ in range(100)]
        result.append((epoch, np.mean([e[0] for e in evaluation]), sum(e[1] for e in evaluation)))
    return result


def play(env: gym.Env, agent: CEMAgent, max_steps: int) -> Trajectory:
    state = env.reset()[0]

    steps = []

    for _ in range(max_steps):
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        steps.append(Step(state, action, reward))

        if terminated or truncated:
            break
        state = next_state

    return Trajectory(steps)


def evaluate(env: gym.Env, agent: CEMAgent, max_steps: int) -> (float, bool):
    state = env.reset()[0]
    total_reward = 0
    delivered = False
    for _ in range(max_steps):
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if reward == 20:
            delivered = True
        if terminated or truncated:
            break
        state = next_state
    return total_reward, delivered


def main():
    env = gym.make("Taxi-v3")
    elite_quantile = 0.2
    trajectories_count = 1000
    output = []
    # Laplace
    alphas = [0.1, 0.5, 1, 2]
    for alpha in tqdm(alphas):
        agent = CEMAgent(env.observation_space.n, env.action_space.n, elite_quantile, LaplaceSmoothing(alpha))
        result = train(env, agent, trajectories_count, 100, 100)
        for item in result:
            output.append({
                "epoch": item[0],
                "reward": item[1],
                "delivered": item[2],
                "quantile": elite_quantile,
                "trajectories": trajectories_count,
                "smoothing": "Laplace",
                "alpha": alpha
            })
    # Policy
    alphas = [0.25, 0.5, 0.75, 0.9]
    for alpha in tqdm(alphas):
        agent = CEMAgent(env.observation_space.n, env.action_space.n, elite_quantile, PolicySmoothing(alpha))
        result = train(env, agent, trajectories_count, 100, 100)
        for item in result:
            output.append({
                "epoch": item[0],
                "reward": item[1],
                "delivered": item[2],
                "quantile": elite_quantile,
                "trajectories": trajectories_count,
                "smoothing": "Policy",
                "alpha": alpha
            })
    pd.DataFrame(output).to_csv("task2.csv", index=False)


if __name__ == "__main__":
    main()
