from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from Frozen_Lake import FrozenLakeEnv


# These types are the same but logically Policy sums to 1 for each state, QTable doesn't
type QTable = dict[int, dict[int, float]]
type Policy = dict[int, dict[int, float]]


def get_q_values(env: FrozenLakeEnv, values: dict[int, float], gamma: float) -> Policy:
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
            for next_state in env.get_next_states(state, action):
                q_values[state][action] += env.get_transition_prob(state, action, next_state) * (env.get_reward(state, action, next_state) + gamma * values[next_state])
    return q_values


def init_policy(env: FrozenLakeEnv) -> Policy:
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        for action in env.get_possible_actions(state):
            policy[state][action] = 1 / len(env.get_possible_actions(state))
    return policy


def init_values(env: FrozenLakeEnv) -> dict[int, float]:
    values = {}
    for state in env.get_all_states():
        values[state] = 0
    return values


def policy_iteration(env: FrozenLakeEnv, gamma: float, iter_n: int) -> dict[int, float]:
    values = init_values(env)
    for i in range(iter_n):
        new_values = init_values(env)
        for state in env.get_all_states():
            for action in env.get_possible_actions(state):
                v = 0
                for next_state in env.get_next_states(state, action):
                    v += env.get_transition_prob(state, action, next_state) * (env.get_reward(state, action, next_state) + gamma * values[next_state])
                new_values[state] = max(new_values[state], v)
        values = new_values
    return values


def greedy_policy(env: FrozenLakeEnv, values: dict[int, float], gamma: float) -> Policy:
    policy = init_policy(env)
    for state in env.get_all_states():
        if len(env.get_possible_actions(state)) > 0:
            result = [
                sum([
                    env.get_transition_prob(state, action, next_state) * (env.get_reward(state, action, next_state) + gamma * values[next_state])
                    for next_state in env.get_next_states(state, action)
                ])
                for action in env.get_possible_actions(state)
            ]
            max_i = np.argmax(result)
            max_action = env.get_possible_actions(state)[max_i]
            for action in env.get_possible_actions(state):
                policy[state][action] = 1 if action == max_action else 0
    return policy


def train(env: FrozenLakeEnv, epochs: int, policy_evaluation_steps: int, gamma: float) -> Policy:
    values = policy_iteration(env, gamma, policy_evaluation_steps * epochs)
    policy = greedy_policy(env, values, gamma)
    return policy


def evaluate(env: FrozenLakeEnv, policy: Policy, iterations: int = 1000, max_len: int = 1000) -> float:
    total_rewards = []
    for _ in range(iterations):
        state = env.reset()
        total_reward = 0
        for i in range(max_len):
            action_i = np.random.choice(np.arange(len(policy[state])), p=list(policy[state].values()))
            action = env.get_possible_actions(state)[action_i]
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        total_rewards.append(total_reward)
    return np.mean(total_rewards)


def main():
    env = FrozenLakeEnv()
    epochs = 20
    policy_evaluation_steps = 20
    output = []
    for gamma in tqdm(range(1, 101, 1)):
        gamma /= 100
        policy = train(env, epochs, policy_evaluation_steps, 1)
        mean_reward = evaluate(env, policy)
        output.append({
            "gamma": gamma,
            "reward": mean_reward,
        })
    pd.DataFrame(output).to_csv("task3.csv", index=False)


if __name__ == "__main__":
    main()
