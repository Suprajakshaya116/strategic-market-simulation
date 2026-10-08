import numpy as np
from typing import Dict, Any, List
from environment.market_env import MarketEnvironment
from agents.momentum_agent import MomentumAgent
from agents.value_agent import ValueAgent


class Evaluator:
    """Evaluates trained agents on the MarketEnvironment and computes financial metrics."""

    def __init__(self, env: MarketEnvironment):
        self.env = env

    def evaluate_agent(
        self,
        ppo_agent,
        num_episodes: int = 5,
        target_agent_id: str = "rl_trader",
        seed: int | None = 100
    ) -> Dict[str, Any]:
        """Run deterministic evaluation episodes and compute comprehensive performance metrics."""
        momentum_agent = MomentumAgent()
        value_agent = ValueAgent()

        episode_rewards = []
        portfolio_histories = []
        pnl_series = []
        total_tx_costs = []
        total_impact_costs = []

        for ep in range(num_episodes):
            ep_seed = seed + ep if seed is not None else None
            states = self.env.reset(seed=ep_seed)
            momentum_agent.reset()

            ep_reward = 0.0
            done = False
            initial_val = self.env.portfolios[target_agent_id].portfolio_value
            pv_history = [initial_val]

            tx_cost = 0.0
            impact_cost = 0.0

            while not done:
                # Deterministic action from PPO agent
                rl_action, _, _ = ppo_agent.select_action(
                    states[target_agent_id],
                    update_encoder_stats=False,
                    deterministic=True
                )
                env_rl_action = ppo_agent.action_space.to_env_action(rl_action)

                actions = {
                    "momentum": momentum_agent.act(states["momentum"]),
                    "value": value_agent.act(states["value"]),
                    target_agent_id: env_rl_action
                }

                states, rewards, done, info = self.env.step(actions)
                ep_reward += rewards[target_agent_id]

                current_val = info["portfolio_values"][target_agent_id]
                pv_history.append(current_val)

                tx_cost += info["transaction_costs"].get(target_agent_id, 0.0)
                impact_cost += info["market_impact_costs"].get(target_agent_id, 0.0)

            episode_rewards.append(ep_reward)
            portfolio_histories.append(pv_history)

            ret = (pv_history[-1] - initial_val) / initial_val
            pnl_series.append(ret)
            total_tx_costs.append(tx_cost)
            total_impact_costs.append(impact_cost)

        # Financial Performance Metrics
        pnl_arr = np.array(pnl_series)
        mean_return = float(np.mean(pnl_arr))
        std_return = float(np.std(pnl_arr)) + 1e-8
        sharpe_ratio = float((mean_return / std_return) * np.sqrt(252))

        # Max Drawdown calculation across histories
        all_drawdowns = []
        for pv_hist in portfolio_histories:
            pv_arr = np.array(pv_hist)
            peak = np.maximum.accumulate(pv_arr)
            dd = (peak - pv_arr) / (peak + 1e-8)
            all_drawdowns.append(np.max(dd))
        max_drawdown = float(np.mean(all_drawdowns))

        win_rate = float(np.mean(pnl_arr > 0))

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "cumulative_return": mean_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "mean_transaction_cost": float(np.mean(total_tx_costs)),
            "mean_market_impact_cost": float(np.mean(total_impact_costs)),
            "num_episodes": num_episodes
        }
