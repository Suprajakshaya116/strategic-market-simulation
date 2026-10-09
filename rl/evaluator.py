import numpy as np
from typing import Dict, Any, List
from environment.market_env import MarketEnvironment
from agents.momentum_agent import MomentumAgent
from agents.value_agent import ValueAgent


class Evaluator:
    """Evaluates trained agents on the MarketEnvironment and computes financial metrics."""

    def __init__(
        self,
        env: MarketEnvironment,
        strategic_layer: Any = None,
        enable_opponent_modeling: bool = False,
        strategic_adapter: Any = None
    ):
        self.env = env
        self.strategic_layer = strategic_layer
        self.enable_opponent_modeling = enable_opponent_modeling
        self.strategic_adapter = strategic_adapter

    def evaluate_agent(
        self,
        ppo_agent,
        num_episodes: int = 5,
        target_agent_id: str = "rl_trader",
        seed: int | None = 100,
        deterministic: bool = True,
        strategic_layer: Any = None,
        enable_opponent_modeling: bool | None = None,
        strategic_adapter: Any = None,
        freeze_opponent_model: bool = True
    ) -> Dict[str, Any]:
        """Run evaluation episodes ensuring 100% training/evaluation parity.

        Applies the strategic layer, leader decisions, matching observation construction,
        and frozen opponent models by default.
        """
        strat_layer = strategic_layer if strategic_layer is not None else self.strategic_layer
        enable_opp = enable_opponent_modeling if enable_opponent_modeling is not None else self.enable_opponent_modeling
        strat_adapter = strategic_adapter if strategic_adapter is not None else self.strategic_adapter

        momentum_agent = MomentumAgent()
        value_agent = ValueAgent()

        episode_rewards = []
        portfolio_histories = []
        pnl_series = []
        total_tx_costs = []
        total_impact_costs = []
        total_trades = []
        total_notionals = []
        episode_spreads = []
        episode_volatilities = []
        episode_liquidities = []
        episode_imbalances = []

        for ep in range(num_episodes):
            ep_seed = seed + ep if seed is not None else None
            states = self.env.reset(seed=ep_seed)
            momentum_agent.reset()

            # If not freezing opponent model, reset strategic layer per episode
            if strat_layer is not None and hasattr(strat_layer, "reset") and not freeze_opponent_model:
                strat_layer.reset()

            ep_reward = 0.0
            done = False
            initial_val = self.env.portfolios[target_agent_id].portfolio_value
            pv_history = [initial_val]

            tx_cost = 0.0
            impact_cost = 0.0
            trades = 0
            turnover = 0.0
            spreads = []
            volatilities = []
            liquidities = []
            imbalances = []

            while not done:
                market_state = self.env.get_market_state()
                leader_decision = None
                strat_dict = None

                # Generate strategic context identical to training loop
                if strat_layer is not None:
                    leader_decision = strat_layer.solve_leader(market_state)
                    strat_dict = strat_layer.build_observation(market_state, leader_decision)
                elif enable_opp and strat_adapter is not None:
                    strat_dict = strat_adapter.build_strategic_dict()

                # Action selection with identical encoder configuration (update_stats=False for eval)
                rl_action, _, _ = ppo_agent.select_action(
                    states[target_agent_id],
                    strategic_dict=strat_dict,
                    update_encoder_stats=False,
                    deterministic=deterministic
                )
                env_rl_action = ppo_agent.action_space.to_env_action(rl_action)

                actions = {}
                for aid in self.env.agent_ids:
                    if aid == target_agent_id:
                        actions[aid] = env_rl_action
                    elif aid == "momentum":
                        actions[aid] = momentum_agent.act(states["momentum"])
                        if not freeze_opponent_model:
                            if strat_layer is not None:
                                strat_layer.update_opponent(aid, market_state, actions[aid])
                            elif enable_opp and strat_adapter and strat_adapter.opponent_model:
                                strat_adapter.opponent_model.update(aid, actions[aid], states[aid])
                    elif aid == "value":
                        actions[aid] = value_agent.act(states["value"])
                        if not freeze_opponent_model:
                            if strat_layer is not None:
                                strat_layer.update_opponent(aid, market_state, actions[aid])
                            elif enable_opp and strat_adapter and strat_adapter.opponent_model:
                                strat_adapter.opponent_model.update(aid, actions[aid], states[aid])
                    else:
                        actions[aid] = "HOLD"

                # Advance environment with leader market control
                states, rewards, done, info = self.env.step(actions, market_control=leader_decision)
                ep_reward += rewards[target_agent_id]

                current_val = info["portfolio_values"][target_agent_id]
                pv_history.append(current_val)

                tx_cost += info["transaction_costs"].get(target_agent_id, 0.0)
                impact_cost += info["market_impact_costs"].get(target_agent_id, 0.0)

                # Record trade metrics
                for order_res in info.get("orders", []):
                    if order_res.get("agent_id") == target_agent_id and order_res.get("status") == "EXECUTED":
                        trades += 1
                        turnover += order_res.get("notional", 0.0)

                spreads.append(info.get("spread", self.env.spread))
                volatilities.append(info.get("volatility", self.env.volatility))
                liquidities.append(info.get("liquidity", self.env.liquidity))
                imbalances.append(abs(info.get("order_imbalance", self.env.order_imbalance)))

            episode_rewards.append(ep_reward)
            portfolio_histories.append(pv_history)

            ret = (pv_history[-1] - initial_val) / (initial_val + 1e-8)
            pnl_series.append(ret)
            total_tx_costs.append(tx_cost)
            total_impact_costs.append(impact_cost)
            total_trades.append(trades)
            total_notionals.append(turnover)

            episode_spreads.append(float(np.mean(spreads)) if spreads else 0.0)
            episode_volatilities.append(float(np.mean(volatilities)) if volatilities else 0.0)
            episode_liquidities.append(float(np.mean(liquidities)) if liquidities else 0.0)
            episode_imbalances.append(float(np.mean(imbalances)) if imbalances else 0.0)

        # Financial Performance Metrics
        pnl_arr = np.array(pnl_series)
        mean_return = float(np.mean(pnl_arr))
        std_return = float(np.std(pnl_arr))
        sharpe_ratio = float((mean_return / (std_return + 1e-8)) * np.sqrt(252)) if std_return > 1e-7 else 0.0

        # Max Drawdown calculation across histories
        all_drawdowns = []
        for pv_hist in portfolio_histories:
            pv_arr = np.array(pv_hist)
            peak = np.maximum.accumulate(pv_arr)
            dd = (peak - pv_arr) / (peak + 1e-8)
            all_drawdowns.append(np.max(dd))
        max_drawdown = float(np.mean(all_drawdowns)) if all_drawdowns else 0.0

        win_rate = float(np.mean(pnl_arr > 0)) if len(pnl_arr) > 0 else 0.0

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "cumulative_return": mean_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "mean_transaction_cost": float(np.mean(total_tx_costs)),
            "mean_market_impact_cost": float(np.mean(total_impact_costs)),
            "mean_trade_count": float(np.mean(total_trades)),
            "mean_turnover": float(np.mean(total_notionals)),
            "mean_spread": float(np.mean(episode_spreads)),
            "mean_market_volatility": float(np.mean(episode_volatilities)),
            "mean_liquidity": float(np.mean(episode_liquidities)),
            "mean_order_imbalance": float(np.mean(episode_imbalances)),
            "num_episodes": num_episodes
        }
