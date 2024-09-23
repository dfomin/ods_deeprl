from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class Step:
    state: int
    action: int
    reward: float


@dataclass
class Trajectory:
    steps: list[Step]


class CEMAgent:
    def __init__(self, states: int, actions: int, elite_quantile: float):
        assert actions > 0
        assert 0 < elite_quantile < 1

        self.actions = actions
        self.elite_quantile = elite_quantile
        self.probs = np.ones((states, actions)) / actions

    def get_action(self, state: int) -> int:
        return np.random.choice(self.actions, p=self.probs[state])

    def fit(self, trajectories: list[Trajectory]):
        pass


def train(env: gym.Env, agent: CEMAgent, trajectories_count: int, max_steps: int, epochs: int):
    for epoch in range(epochs):
        trajectories = [play(env, agent, max_steps) for _ in range(trajectories_count)]
        agent.fit(trajectories)


def play(env: gym.Env, agent: CEMAgent, max_steps: int) -> Trajectory:
    state = env.reset()[0]

    steps = []

    for _ in range(max_steps):
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        steps.append(Step(state, action, float(reward)))
        if terminated or truncated:
            break
        state = next_state

    return Trajectory(steps)


def main():
    env = gym.make("Taxi-v3")
    agent = CEMAgent(env.observation_space.n, env.action_space.n, 0.5)
    train(env, agent, 100, 10_000, 10)


if __name__ == "__main__":
    main()
