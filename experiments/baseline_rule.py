import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.config import MarketConfig
from environment.market_env import MarketEnvironment
from agents.momentum_agent import MomentumAgent
from agents.value_agent import ValueAgent


def run_rule_baseline_experiment(episodes: int = 10, episode_length: int = 100):
    print("==================================================")
    print("      BASELINE 2: RULE-BASED TRADERS EXPERIMENT   ")
    print("==================================================")

    market_config = MarketConfig(episode_length=episode_length, seed=42)
    env = MarketEnvironment(config=market_config)

    momentum = MomentumAgent()
    value = ValueAgent()

    rewards_history = {aid: [] for aid in env.agent_ids}
    portfolio_history = {aid: [] for aid in env.agent_ids}

    for ep in range(episodes):
        states = env.reset(seed=200 + ep)
        momentum.reset()
        ep_rewards = {aid: 0.0 for aid in env.agent_ids}
        done = False
        info: dict = {}

        while not done:
            actions = {
                "momentum": momentum.act(states["momentum"]),
                "value": value.act(states["value"]),
                "rl_trader": "HOLD",
            }
            next_states, rewards, done, info = env.step(actions)
            for aid in env.agent_ids:
                ep_rewards[aid] += rewards[aid]
            states = next_states

        for aid in env.agent_ids:
            rewards_history[aid].append(ep_rewards[aid])
            pv = info.get("portfolio_values", {}).get(aid, 0.0) if info else 0.0
            portfolio_history[aid].append(pv)

    print("\n===== RULE BASELINE COMPLETED =====")
    for aid in env.agent_ids:
        print(f"[{aid}] Mean Reward: {np.mean(rewards_history[aid]):.2f} | Final Portfolio Value: {np.mean(portfolio_history[aid]):.2f}")

    return {
        "mean_rewards": {aid: float(np.mean(rewards_history[aid])) for aid in env.agent_ids},
        "mean_portfolio_values": {aid: float(np.mean(portfolio_history[aid])) for aid in env.agent_ids},
        "episodes": episodes,
    }


if __name__ == "__main__":
    run_rule_baseline_experiment(episodes=10)
