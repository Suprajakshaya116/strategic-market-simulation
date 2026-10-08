from dataclasses import dataclass


@dataclass
class MarketConfig:
    """Configuration for the Member 1 artificial market simulator."""

    initial_price: float = 100.0
    initial_fundamental_value: float = 100.0
    initial_volume: float = 0.0
    initial_liquidity: float = 1.0
    initial_spread: float = 0.02
    initial_volatility: float = 0.01

    drift: float = 0.0
    imbalance_impact: float = 0.05
    noise_scale: float = 0.01
    fundamental_reversion: float = 0.01

    liquidity_impact: float = 0.10
    liquidity_recovery: float = 0.05
    volatility_decay: float = 0.90
    volatility_imbalance_sensitivity: float = 0.10

    transaction_cost_rate: float = 0.001
    market_impact_rate: float = 0.0005

    # Coefficients c and eta in the specification's reward formula.
    reward_transaction_cost_weight: float = 1.0
    reward_market_impact_weight: float = 1.0
    risk_aversion: float = 0.01

    initial_cash: float = 100_000.0
    initial_inventory: int = 0
    max_order_size: int = 100
    episode_length: int = 100
    seed: int | None = 42

    agent_ids: tuple[str, ...] = ("momentum", "value", "rl_trader")

    def validate(self):
        positive = {
            "initial_price": self.initial_price,
            "initial_fundamental_value": self.initial_fundamental_value,
            "initial_liquidity": self.initial_liquidity,
            "initial_volatility": self.initial_volatility,
            "max_order_size": self.max_order_size,
            "episode_length": self.episode_length,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0")

        non_negative = {
            "initial_cash": self.initial_cash,
            "initial_volume": self.initial_volume,
            "initial_spread": self.initial_spread,
            "drift": self.drift,
            "noise_scale": self.noise_scale,
            "fundamental_reversion": self.fundamental_reversion,
            "liquidity_impact": self.liquidity_impact,
            "liquidity_recovery": self.liquidity_recovery,
            "transaction_cost_rate": self.transaction_cost_rate,
            "market_impact_rate": self.market_impact_rate,
            "reward_transaction_cost_weight": self.reward_transaction_cost_weight,
            "reward_market_impact_weight": self.reward_market_impact_weight,
            "risk_aversion": self.risk_aversion,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0")

        if not 0.0 <= self.volatility_decay <= 1.0:
            raise ValueError("volatility_decay must be between 0 and 1")
        if not self.agent_ids:
            raise ValueError("agent_ids must contain at least one agent")
        if len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("agent_ids must be unique")
