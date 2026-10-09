import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.config import MarketConfig
from environment.market_env import MarketEnvironment
from agents.momentum_agent import MomentumAgent
from agents.value_agent import ValueAgent


def run_random_baseline_experiment(episodes: int = 10, episode_length: int = 100):
    print("==================================================")
    print("        BASELINE 1: RANDOM TRADER EXPERIMENT      ")
    print("==================================================")

    market_config = MarketConfig(episode_length=episode_length, seed=42)
    env = MarketEnvironment(config=market_config)
    rng = np.random.default_rng(42)

    actions_list = ["HOLD", "BUY", "SELL"]
    rewards_history = []
    portfolio_history = []

    for ep in range(episodes):
        states = env.reset(seed=100 + ep)
        ep_reward = 0.0
        done = False

        while not done:
            random_action = rng.choice(actions_list)
            actions = {
                "momentum": "HOLD",
                "value": "HOLD",
                "rl_trader": random_action,
            }
            next_states, rewards, done, info = env.step(actions)
            ep_reward += rewards["rl_trader"]

        rewards_history.append(ep_reward)
        portfolio_history.append(info["portfolio_values"]["rl_trader"])

    mean_reward = float(np.mean(rewards_history))
    mean_pv = float(np.mean(portfolio_history))

    print("\n===== RANDOM BASELINE COMPLETED =====")
    print(f"Episodes: {episodes}")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Mean Final Portfolio Value: {mean_pv:.2f}")

    return {
        "mean_reward": mean_reward,
        "mean_portfolio_value": mean_pv,
        "episodes": episodes,
    }


if __name__ == "__main__":
    run_random_baseline_experiment(episodes=10)
