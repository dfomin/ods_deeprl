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


class CEMAgent:
    def __init__(self, states: int, actions: int, elite_quantile: float):
        assert actions > 0
        assert 0 < elite_quantile < 1

        self.actions = actions
        self.elite_quantile = elite_quantile
        self.policy = np.ones((states, actions)) / actions

    def get_action(self, state: int) -> int:
        return np.random.choice(self.actions, p=self.policy[state])

    def fit(self, trajectories: list[Trajectory]):
        threshold = np.quantile([t.total_reward for t in trajectories], self.elite_quantile)
        elite_trajectories = [t for t in trajectories if t.total_reward > threshold]
        if not elite_trajectories:
            elite_trajectories = [t for t in trajectories if t.total_reward >= threshold]
        new_policy = np.zeros_like(self.policy)
        for trajectory in elite_trajectories:
            for step in trajectory.steps:
                new_policy[step.state, step.action] += 1
        for state in range(len(new_policy)):
            if np.count_nonzero(new_policy[state]) == 0:
                new_policy[state] = 1 / self.actions
            else:
                new_policy[state] /= new_policy[state].sum()
        self.policy = new_policy


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
    elite_quantiles = [x / 10 for x in range(1, 10)]
    trajectories_counts = [30, 100, 300, 1000, 3000]
    output = []
    for elite_quantile in tqdm(elite_quantiles):
        for trajectories_count in tqdm(trajectories_counts, leave=False):
            agent = CEMAgent(env.observation_space.n, env.action_space.n, elite_quantile)
            result = train(env, agent, trajectories_count, 100, 100)
            for item in result:
                output.append({
                    "epoch": item[0],
                    "reward": item[1],
                    "delivered": item[2],
                    "quantile": elite_quantile,
                    "trajectories": trajectories_count
                })
    pd.DataFrame(output).to_csv("task1.csv", index=False)


if __name__ == "__main__":
    main()
