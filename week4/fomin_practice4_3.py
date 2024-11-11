from collections import defaultdict
from enum import Enum, auto
from typing import Callable

import numpy as np
import gymnasium as gym
import pandas as pd


class EpsilonStrategy(str, Enum):
    LINEAR = "Linear"
    EXPONENTIAL = "Exponential"
    DECAYING = "Decaying"


def get_epsilon_greedy_action(q_values: np.ndarray, epsilon: float, action_n: int) -> int:
    prob = np.ones(action_n) * epsilon / action_n
    argmax_action = np.argmax(q_values)
    prob[argmax_action] += 1 - epsilon
    action = np.random.choice(np.arange(action_n), p=prob)
    return action


def monte_carlo(
        env: gym.Env,
        strategy: EpsilonStrategy,
        episode_n: int = 100000,
        trajectory_len: int = 100,
        gamma: float = 0.99) -> dict[int, float]:
    state_n = env.observation_space.n
    action_n = env.action_space.n

    q_values = np.zeros((state_n, action_n))
    counters = np.zeros((state_n, action_n))

    total_rewards = defaultdict(float)
    epsilon = 1
    for episode in range(episode_n):
        match strategy:
            case EpsilonStrategy.LINEAR:
                epsilon = 1 - episode / episode_n
            case EpsilonStrategy.EXPONENTIAL:
                epsilon = epsilon * 0.99996
            case EpsilonStrategy.DECAYING:
                epsilon = 1 / ((episode // 1000) + 1)

        trajectory = {'states': [], 'actions': [], 'rewards': []}

        state, _ = env.reset()
        for t in range(trajectory_len):
            action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
            next_state, reward, terminated, truncated, _ = env.step(action)

            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)

            state = next_state

            if terminated or truncated:
                break

        if episode % 1000 == 0:
            total_rewards[episode // 1000] = evaluate(env, lambda x: get_epsilon_greedy_action(q_values[x], epsilon, action_n))

        real_trajectory_len = len(trajectory['rewards'])
        returns = np.zeros(real_trajectory_len + 1)
        for t in range(real_trajectory_len - 1, -1, -1):
            returns[t] = trajectory['rewards'][t] + gamma * returns[t + 1]

        for t in range(real_trajectory_len):
            state = trajectory['states'][t]
            action = trajectory['actions'][t]
            counters[state][action] += 1
            q_values[state][action] += (returns[t] - q_values[state][action]) / counters[state][action]

    return total_rewards


def evaluate(env: gym.Env, policy: Callable[[int], int], t_max: int = 100, count: int = 100) -> float:
    total_reward = 0
    for _ in range(count):
        state, _ = env.reset()
        for _ in range(t_max):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            state = next_state
    return total_reward / count


def main():
    output = []
    for strategy in [EpsilonStrategy.LINEAR, EpsilonStrategy.EXPONENTIAL, EpsilonStrategy.DECAYING]:
        result = monte_carlo(gym.make("Taxi-v3"), strategy=strategy)
        for epoch, reward in result.items():
            output.append({
                "epoch": epoch,
                "reward": reward,
                "strategy": strategy.value
            })
    pd.DataFrame(output).to_csv("task3.csv", index=False)


if __name__ == "__main__":
    main()
