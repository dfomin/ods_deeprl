from collections import defaultdict
from typing import Callable

import numpy as np
import gymnasium as gym
import pandas as pd


def get_epsilon_greedy_action(q_values: np.ndarray, epsilon: float, action_n: int) -> int:
    prob = np.ones(action_n) * epsilon / action_n
    argmax_action = np.argmax(q_values)
    prob[argmax_action] += 1 - epsilon
    action = np.random.choice(np.arange(action_n), p=prob)
    return action


def encode_state(env: gym.Env, state: np.ndarray, bins_count: int) -> int:
    index = 0
    for i in range(len(state)):
        index *= bins_count
        value = state[i]
        low = max(env.observation_space.low[i], -3)
        high = min(env.observation_space.high[i], 3)
        bins = np.linspace(low, high, bins_count + 1)
        bin_index = np.digitize(value, bins) - 1
        index += bin_index
    return index


def monte_carlo(
        env: gym.Env,
        bins: int,
        episode_n: int = 100000,
        trajectory_len: int = 500,
        gamma: float = 0.99) -> dict[int, float]:
    state_n = bins ** (env.observation_space.shape[0] + 1)
    action_n = env.action_space.n

    q_values = np.zeros((state_n, action_n))
    counters = np.zeros((state_n, action_n))

    total_rewards = defaultdict(float)
    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n

        trajectory = {'states': [], 'actions': [], 'rewards': []}

        state, _ = env.reset()
        state = encode_state(env, state, bins)
        for t in range(trajectory_len):
            action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = encode_state(env, next_state, bins)

            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)

            state = next_state

            if terminated or truncated:
                break

        if episode % 1000 == 0:
            total_rewards[episode // 1000] = evaluate(env, bins, lambda x: get_epsilon_greedy_action(q_values[x], epsilon, action_n))
            print(episode // 1000, total_rewards[episode // 1000])

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


def sarsa(
        env: gym.Env,
        bins: int,
        episode_n: int = 100000,
        gamma: float = 0.99,
        trajectory_len: int = 500,
        alpha: float = 0.5) -> dict[int, float]:
    state_n = bins ** (env.observation_space.shape[0] + 1)
    action_n = env.action_space.n

    q_values = np.zeros((state_n, action_n))

    total_rewards = defaultdict(float)
    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n

        total_reward = 0

        state, _ = env.reset()
        state = encode_state(env, state, bins)
        action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
        for t in range(trajectory_len):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = encode_state(env, next_state, bins)
            next_action = get_epsilon_greedy_action(q_values[next_state], epsilon, action_n)

            q_values[state][action] += alpha * (reward + gamma * q_values[next_state][next_action] - q_values[state][action])

            total_reward += reward

            state = next_state
            action = next_action

            if terminated or truncated:
                break

        if episode % 1000 == 0:
            total_rewards[episode // 1000] = evaluate(env, bins, lambda x: get_epsilon_greedy_action(q_values[x], epsilon, action_n))
            print(episode // 1000, total_rewards[episode // 1000])

    return total_rewards


def q_learning(
        env: gym.Env,
        bins: int,
        episode_n: int = 100000,
        gamma: float = 0.99,
        t_max: int = 500,
        alpha: float = 0.5) -> dict[int, float]:
    state_n = bins ** (env.observation_space.shape[0] + 1)
    action_n = env.action_space.n

    q_values = np.zeros((state_n, action_n))

    total_rewards = defaultdict(float)
    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n

        total_reward = 0

        state, _ = env.reset()
        state = encode_state(env, state, bins)
        action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
        for t in range(t_max):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = encode_state(env, next_state, bins)
            next_action = get_epsilon_greedy_action(q_values[next_state], epsilon, action_n)

            q_values[state][action] += alpha * (reward + gamma * q_values[next_state].max() - q_values[state][action])

            total_reward += reward

            state = next_state
            action = next_action

            if terminated or truncated:
                break

        if episode % 1000 == 0:
            total_rewards[episode // 1000] = evaluate(env, bins, lambda x: get_epsilon_greedy_action(q_values[x], epsilon, action_n))
            print(episode // 1000, total_rewards[episode // 1000])

    return total_rewards


def evaluate(env: gym.Env, bins: int, policy: Callable[[int], int], t_max: int = 500, count: int = 10) -> float:
    total_reward = 0
    for _ in range(count):
        state, _ = env.reset()
        state = encode_state(env, state, bins)
        for _ in range(t_max):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = encode_state(env, next_state, bins)
            total_reward += reward
            if terminated or truncated:
                break
            state = next_state
    return total_reward / count


def main():
    algorithms = [
        ("MonteCarlo", monte_carlo),
        ("SARSA", sarsa),
        ("Q Learning", q_learning),
    ]

    output = []
    for algorithm_name, algorithm in algorithms:
        result = algorithm(gym.make("CartPole-v1"), 20, episode_n=100000)
        for epoch, reward in result.items():
            output.append({
                "epoch": epoch,
                "reward": reward,
                "algorithm": algorithm_name
            })
    pd.DataFrame(output).to_csv("task2_2.csv", index=False)


if __name__ == "__main__":
    main()
