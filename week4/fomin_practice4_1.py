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


def cross_entropy(
        env: gym.Env,
        episode_n: int = 100,
        trajectories_count: int = 1000,
        elite_quantile: float = 0.2,
        trajectory_len: int = 100,
        alpha: float = 0.5) -> dict[int, float]:
    state_n = env.observation_space.n
    action_n = env.action_space.n

    policy = np.ones((state_n, action_n)) / action_n

    total_rewards = defaultdict(float)
    for epoch in range(episode_n):
        trajectories = []
        for _ in range(trajectories_count):
            state = env.reset()[0]

            steps = []

            for _ in range(trajectory_len):
                action = np.random.choice(action_n, p=policy[state])
                next_state, reward, terminated, truncated, _ = env.step(action)
                steps.append((state, action, reward))

                if terminated or truncated:
                    break
                state = next_state
            trajectories.append(steps)

        total_rewards[epoch] = evaluate(env, lambda x: int(policy[x].argmax()))

        threshold = np.quantile([sum([s[2] for s in t]) for t in trajectories], elite_quantile)
        elite_trajectories = [t for t in trajectories if sum([s[2] for s in t]) > threshold]
        if not elite_trajectories:
            elite_trajectories = [t for t in trajectories if sum([s[2] for s in t]) >= threshold]
        new_policy = np.zeros_like(policy)
        for trajectory in elite_trajectories:
            for step in trajectory:
                new_policy[step[0], step[1]] += 1
        for state in range(len(new_policy)):
            if np.count_nonzero(new_policy[state]) == 0:
                new_policy[state] = 1 / action_n
            else:
                new_policy[state] /= new_policy[state].sum()
        policy = alpha * policy + (1 - alpha) * new_policy
    return total_rewards


def monte_carlo(
        env: gym.Env,
        episode_n: int = 100000,
        trajectory_len: int = 100,
        gamma: float = 0.99) -> dict[int, float]:
    state_n = env.observation_space.n
    action_n = env.action_space.n

    q_values = np.zeros((state_n, action_n))
    counters = np.zeros((state_n, action_n))

    total_rewards = defaultdict(float)
    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n

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
            total_rewards[episode // 1000] = evaluate(env, lambda x: get_epsilon_greedy_action(q_values[x], 0, action_n))

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
        episode_n: int = 100000,
        gamma: float = 0.99,
        trajectory_len: int = 100,
        alpha: float = 0.5) -> dict[int, float]:
    state_n = env.observation_space.n
    action_n = env.action_space.n

    q_values = np.zeros((state_n, action_n))

    total_rewards = defaultdict(float)
    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n

        total_reward = 0

        state, _ = env.reset()
        action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
        for t in range(trajectory_len):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = get_epsilon_greedy_action(q_values[next_state], epsilon, action_n)

            q_values[state][action] += alpha * (reward + gamma * q_values[next_state][next_action] - q_values[state][action])

            total_reward += reward

            state = next_state
            action = next_action

            if terminated or truncated:
                break

        if episode % 1000 == 0:
            total_rewards[episode // 1000] = evaluate(env, lambda x: get_epsilon_greedy_action(q_values[x], 0, action_n))

    return total_rewards


def q_learning(
        env: gym.Env,
        episode_n: int = 100000,
        gamma: float = 0.99,
        t_max: int = 100,
        alpha: float = 0.5) -> dict[int, float]:
    state_n = env.observation_space.n
    action_n = env.action_space.n

    q_values = np.zeros((state_n, action_n))

    total_rewards = defaultdict(float)
    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n

        total_reward = 0

        state, _ = env.reset()
        action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
        for t in range(t_max):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = get_epsilon_greedy_action(q_values[next_state], epsilon, action_n)

            q_values[state][action] += alpha * (reward + gamma * q_values[next_state].max() - q_values[state][action])

            total_reward += reward

            state = next_state
            action = next_action

            if terminated or truncated:
                break

        if episode % 1000 == 0:
            total_rewards[episode // 1000] = evaluate(env, lambda x: get_epsilon_greedy_action(q_values[x], 0, action_n))

    return total_rewards


def evaluate(env: gym.Env, policy: Callable[[int], int], t_max: int = 100, count: int = 1000) -> float:
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
    algorithms = [
        ("Cross-entropy", cross_entropy),
        ("MonteCarlo", monte_carlo),
        ("SARSA", sarsa),
        ("Q Learning", q_learning),
    ]

    output = []
    for algorithm_name, algorithm in algorithms:
        result = algorithm(gym.make("Taxi-v3"))
        for epoch, reward in result.items():
            output.append({
                "epoch": epoch,
                "reward": reward,
                "algorithm": algorithm_name
            })
    pd.DataFrame(output).to_csv("task1.csv", index=False)


if __name__ == "__main__":
    main()
