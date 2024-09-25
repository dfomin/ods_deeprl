import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import gymnasium as gym
import numpy as np


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
    def update(self, old_policy: np.ndarray, new_policy: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Normalizer(Updater):
    def update(self, old_policy: np.ndarray, statistics: np.ndarray) -> np.ndarray:
        new_policy = np.zeros_like(old_policy)
        for s in range(len(statistics)):
            if np.count_nonzero(statistics[s]) == 0:
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
        statistics = Normalizer().update(old_policy, statistics)
        for s in range(len(old_policy)):
            for a in range(len(old_policy[s])):
                new_policy[s, a] = self.alpha * statistics[s, a] + (1 - self.alpha) * old_policy[s, a]
        return new_policy


class Policy(Protocol):
    def get_action(self, state: int) -> int:
        raise NotImplementedError


class DeterministicPolicy(Policy):
    def __init__(self, policy: list[int]):
        self.policy = policy

    def get_action(self, state: int) -> int:
        return self.policy[state]


class CEMAgent(Policy):
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

    def sample_policy(self) -> DeterministicPolicy:
        return DeterministicPolicy([np.random.choice(self.actions, p=self.policy[state]) for state in range(len(self.policy))])


def train(env: gym.Env, agent: CEMAgent, trajectories_count: int, max_steps: int, epochs: int, sample_count: int):
    for epoch in range(epochs):
        if sample_count > 0:
            trajectories = []
            for _ in range(trajectories_count):
                trajectories_batch = []
                policy = agent.sample_policy()
                for _ in range(sample_count):
                    trajectories_batch.append(play(env, policy, max_steps))
                mean_reward = np.mean([t.total_reward for t in trajectories_batch])
                for trajectory in trajectories_batch:
                    trajectory.total_reward = mean_reward
                trajectories.extend(trajectories_batch)
        else:
            trajectories = [play(env, agent, max_steps) for _ in range(trajectories_count)]
        print(f"epoch: {epoch}, mean reward: {np.mean([t.total_reward for t in trajectories])}")
        agent.fit(trajectories)


def play(env: gym.Env, agent: Policy, max_steps: int) -> Trajectory:
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


def main():
    env = gym.make("Taxi-v3")
    agent = CEMAgent(env.observation_space.n, env.action_space.n, 0.25, PolicySmoothing(0.8))
    train(env, agent, 1000, 100, 100, 10)


if __name__ == "__main__":
    main()
