from .config import MarketConfig
from .liquidity import LiquidityModel
from .order_manager import OrderManager
from .portfolio import Portfolio
from .price_model import PriceModel
from .execution_engine import ExecutionEngine

class MarketEnvironment:
    """Member 1 market simulator and stable integration boundary.

    The public interface follows the project specification:
        state -> agents -> actions -> environment -> next_state + reward + done + info

    Trader actions are either ``"BUY"``, ``"HOLD"``, ``"SELL"`` or
    ``("BUY", quantity)`` / ``("SELL", quantity)``.
    """

    def __init__(self, config=None, agent_ids=None):
        self.config = config or MarketConfig()
        self.config.validate()
        self.agent_ids = tuple(agent_ids or self.config.agent_ids)
        if not self.agent_ids:
            raise ValueError("At least one agent is required")
        if len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("agent_ids must be unique")

        self.order_manager = OrderManager(self.config.max_order_size)
        self.price_model = PriceModel(
            self.config.drift,
            self.config.imbalance_impact,
            self.config.noise_scale,
            self.config.fundamental_reversion,
            self.config.seed,
        )
        self.liquidity_model = LiquidityModel(
            self.config.initial_liquidity,
            self.config.initial_spread,
            self.config.liquidity_impact,
            self.config.liquidity_recovery,
        )
        self.reset()

    def reset(self, seed=None):
        """Reset the market and all portfolios to their initial conditions."""
        if seed is not None:
            self.price_model.reset(seed)
        else:
            self.price_model.reset()

        self.step_count = 0
        self.price = float(self.config.initial_price)
        self.previous_price = self.price
        self.fundamental_value = float(self.config.initial_fundamental_value)
        self.volume = float(self.config.initial_volume)
        self.liquidity = float(self.config.initial_liquidity)
        self.spread = float(self.config.initial_spread)
        self.volatility = float(self.config.initial_volatility)
        self.order_imbalance = 0.0
        self.portfolios = {
            aid: Portfolio(
                float(self.config.initial_cash),
                int(self.config.initial_inventory),
                self.price,
            )
            for aid in self.agent_ids
        }

        self.execution_engine = ExecutionEngine(
            self.portfolios,
            self.config,
        )

        return self.get_all_states()

    @property
    def current_return(self):
        if self.previous_price == 0:
            return 0.0
        return (self.price - self.previous_price) / self.previous_price

    @property
    def bid_price(self):
        return max(self.price - self.spread / 2.0, 1e-8)

    @property
    def ask_price(self):
        return self.price + self.spread / 2.0

    def get_market_state(self):
        """Return the shared market fields from the project's interface."""
        return {
            "price": float(self.price),
            "return": float(self.current_return),
            "volume": float(self.volume),
            "spread": float(self.spread),
            "liquidity": float(self.liquidity),
            "volatility": float(self.volatility),
            "order_imbalance": float(self.order_imbalance),
            "fundamental_value": float(self.fundamental_value),
        }

    def get_quotes(self):
        """Return bid/ask quotes; kept separate so the Point 26 state stays stable."""
        return {"bid": float(self.bid_price), "ask": float(self.ask_price)}

    def get_state(self, agent_id):
        if agent_id not in self.portfolios:
            raise KeyError(f"Unknown agent: {agent_id}")
        state = self.get_market_state()
        portfolio = self.portfolios[agent_id]
        portfolio.mark_to_market(self.price)
        state.update({
            "inventory": int(portfolio.inventory),
            "cash": float(portfolio.cash),
        })
        return state

    def get_all_states(self):
        return {aid: self.get_state(aid) for aid in self.agent_ids}

    def _validate_actions(self, actions):
        if not isinstance(actions, dict):
            raise TypeError("actions must be a dictionary keyed by agent id")

        provided = set(actions)
        expected = set(self.agent_ids)
        missing = expected - provided
        extra = provided - expected
        if missing:
            raise ValueError(f"Missing actions for agents: {sorted(missing)}")
        if extra:
            raise ValueError(f"Unknown agents in actions: {sorted(extra)}")


    def execute_orders(self, actions):
        """Validate, normalize and execute trader orders against current quotes."""
        self._validate_actions(actions)

        orders = self.order_manager.create_orders(actions)

        executed = self.execution_engine.execute(
            orders,
            self.price,
            self.liquidity,
            bid_price=self.bid_price,
            ask_price=self.ask_price,
        )

        return orders, executed

    def update_price(self, buy_volume, sell_volume):
        """Update order flow, price, liquidity, spread and volatility."""
        self.order_imbalance = self.price_model.order_imbalance(
            buy_volume, sell_volume
        )
        self.previous_price = self.price
        self.price = self.price_model.update(
            self.price,
            self.fundamental_value,
            self.order_imbalance,
            self.volatility,
        )
        self.volume = float(buy_volume + sell_volume)

        self.liquidity, self.spread = self.liquidity_model.update(
            self.liquidity,
            self.spread,
            self.order_imbalance,
        )

        target_volatility = (
            self.config.initial_volatility
            + self.config.volatility_imbalance_sensitivity
            * abs(self.order_imbalance)
        )
        self.volatility = max(
            self.config.volatility_decay * self.volatility
            + (1.0 - self.config.volatility_decay) * target_volatility,
            1e-8,
        )

    def calculate_rewards(self, previous_values, previous_costs):
        """Implement R_i = ΔV_i - λRisk_i - cCost_i - ηImpact_i.

        ΔV is measured before newly accumulated trading costs, avoiding double
        counting while retaining the cost terms explicitly in the reward.
        """
        rewards = {}
        for aid in self.agent_ids:
            portfolio = self.portfolios[aid]
            portfolio.mark_to_market(self.price)

            total_cost = portfolio.total_transaction_cost - previous_costs[aid][0]
            total_impact = (
                portfolio.total_market_impact_cost - previous_costs[aid][1]
            )
            gross_value_change = (
                portfolio.portfolio_value
                + total_cost
                + total_impact
                - previous_values[aid]
            )
            risk_penalty = (
                self.config.risk_aversion
                * abs(portfolio.inventory)
                * self.volatility
            )

            rewards[aid] = float(
                gross_value_change
                - risk_penalty
                - self.config.reward_transaction_cost_weight * total_cost
                - self.config.reward_market_impact_weight * total_impact
            )
        return rewards

    def step(self, actions, market_control=None):
        """Advance one market timestep.

        Returns ``(next_state, reward, done, info)`` to match the project's
        Point 26 integration contract.

        The strategic leader's ``market_control`` action (TIGHT, MEDIUM, WIDE)
        is applied to the market conditions (spread and liquidity multipliers)
        BEFORE follower order execution, establishing the causal Stackelberg sequence.
        """
        self._validate_actions(actions)
        previous_values = {
            aid: self.portfolios[aid].portfolio_value for aid in self.agent_ids
        }
        previous_costs = {
            aid: (
                self.portfolios[aid].total_transaction_cost,
                self.portfolios[aid].total_market_impact_cost,
            )
            for aid in self.agent_ids
        }

        # 1. Apply strategic market control from Leader BEFORE order execution
        applied_control = None
        if market_control is not None:
            spread_mult = 1.0
            liq_mult = 1.0
            action_name = str(market_control)

            if isinstance(market_control, dict):
                action_name = market_control.get("leader_action", "MEDIUM")
                spread_mult = float(market_control.get("spread_multiplier", 1.0))
                liq_mult = float(market_control.get("liquidity_multiplier", 1.0))
            elif hasattr(market_control, "selected_action"):
                action_name = market_control.selected_action.value
            elif hasattr(market_control, "value"):
                action_name = market_control.value

            action_upper = str(action_name).upper()
            if action_upper == "TIGHT":
                spread_mult = spread_mult if isinstance(market_control, dict) and "spread_multiplier" in market_control else 0.7
                liq_mult = liq_mult if isinstance(market_control, dict) and "liquidity_multiplier" in market_control else 1.3
            elif action_upper == "WIDE":
                spread_mult = spread_mult if isinstance(market_control, dict) and "spread_multiplier" in market_control else 1.4
                liq_mult = liq_mult if isinstance(market_control, dict) and "liquidity_multiplier" in market_control else 0.7

            self.spread = float(self.spread * spread_mult)
            self.liquidity = float(self.liquidity * liq_mult)
            applied_control = {
                "leader_action": action_upper,
                "spread_multiplier": spread_mult,
                "liquidity_multiplier": liq_mult,
            }

        # 2. Execute orders under the resulting market conditions
        orders, executed = self.execute_orders(actions)
        volumes = self.execution_engine.aggregate_executed(executed)

        # 3. Update market price and dynamics
        self.update_price(volumes["buy_volume"], volumes["sell_volume"])

        rewards = self.calculate_rewards(previous_values, previous_costs)

        self.step_count += 1
        done = self.step_count >= self.config.episode_length
        info = {
            "step": self.step_count,
            "orders": executed,
            "buy_volume": volumes["buy_volume"],
            "sell_volume": volumes["sell_volume"],
            "order_imbalance": float(self.order_imbalance),
            "price": float(self.price),
            "bid": float(self.bid_price),
            "ask": float(self.ask_price),
            "spread": float(self.spread),
            "liquidity": float(self.liquidity),
            "volatility": float(self.volatility),
            "volume": volumes["buy_volume"] + volumes["sell_volume"],
            "market_control": applied_control,
            "portfolio_values": {
                aid: float(self.portfolios[aid].portfolio_value)
                for aid in self.agent_ids
            },
            "transaction_costs": {
                aid: float(
                    self.portfolios[aid].total_transaction_cost
                    - previous_costs[aid][0]
                )
                for aid in self.agent_ids
            },
            "market_impact_costs": {
                aid: float(
                    self.portfolios[aid].total_market_impact_cost
                    - previous_costs[aid][1]
                )
                for aid in self.agent_ids
            },
        }
        return self.get_all_states(), rewards, done, info
