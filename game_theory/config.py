from dataclasses import dataclass
from typing import Optional


@dataclass
class GameTheoryConfig:
    """Configuration parameters for the Game-Theoretic Strategic Layer."""

    # Leader Utility Weights
    leader_liquidity_weight: float = 1.0    # alpha
    leader_activity_weight: float = 0.5     # beta
    leader_volatility_penalty: float = 0.5   # gamma
    leader_inventory_penalty: float = 0.2    # delta
    leader_adverse_flow_penalty: float = 0.5 # epsilon

    # Follower Utility Weights
    follower_return_weight: float = 1.0
    follower_risk_penalty: float = 0.01
    follower_transaction_cost_penalty: float = 1.0

    # Opponent Modeling & Pressure Estimation Parameters
    smoothing_parameter: float = 1.0        # Laplace smoothing parameter
    expected_order_size: float = 100.0

    # Leader Action Parameters (Spread & Liquidity Multipliers)
    tight_spread_multiplier: float = 0.7
    tight_liquidity_multiplier: float = 1.3
    medium_spread_multiplier: float = 1.0
    medium_liquidity_multiplier: float = 1.0
    wide_spread_multiplier: float = 1.4
    wide_liquidity_multiplier: float = 0.7

    # Reproducibility
    seed: Optional[int] = 42

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.leader_liquidity_weight < 0:
            raise ValueError("leader_liquidity_weight must be >= 0")
        if self.leader_activity_weight < 0:
            raise ValueError("leader_activity_weight must be >= 0")
        if self.leader_volatility_penalty < 0:
            raise ValueError("leader_volatility_penalty must be >= 0")
        if self.leader_inventory_penalty < 0:
            raise ValueError("leader_inventory_penalty must be >= 0")
        if self.leader_adverse_flow_penalty < 0:
            raise ValueError("leader_adverse_flow_penalty must be >= 0")

        if self.follower_return_weight < 0:
            raise ValueError("follower_return_weight must be >= 0")
        if self.follower_risk_penalty < 0:
            raise ValueError("follower_risk_penalty must be >= 0")
        if self.follower_transaction_cost_penalty < 0:
            raise ValueError("follower_transaction_cost_penalty must be >= 0")

        if self.smoothing_parameter <= 0:
            raise ValueError("smoothing_parameter must be > 0")
        if self.expected_order_size <= 0:
            raise ValueError("expected_order_size must be > 0")

        if self.tight_spread_multiplier <= 0 or self.medium_spread_multiplier <= 0 or self.wide_spread_multiplier <= 0:
            raise ValueError("Spread multipliers must be > 0")
        if self.tight_liquidity_multiplier <= 0 or self.medium_liquidity_multiplier <= 0 or self.wide_liquidity_multiplier <= 0:
            raise ValueError("Liquidity multipliers must be > 0")
