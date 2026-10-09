from typing import Dict, Any, Optional
from .config import GameTheoryConfig
from .types import LeaderAction


def calculate_leader_utility(
    state: Dict[str, Any],
    leader_action: LeaderAction,
    expected_buy_pressure: float,
    expected_sell_pressure: float,
    expected_order_imbalance: float,
    config: Optional[GameTheoryConfig] = None,
    leader_inventory: float = 0.0,
) -> float:
    """Calculate the Leader (Market Maker / Liquidity Controller) utility.

    Formulation:
        U_L = alpha * liquidity
            + beta * trading_activity
            - gamma * volatility_risk
            - delta * inventory_risk
            - epsilon * adverse_flow_risk
    """
    cfg = config or GameTheoryConfig()

    base_liquidity = float(state.get("liquidity", 1.0))
    volatility = float(state.get("volatility", 0.01))

    # Apply spread/liquidity multipliers according to the candidate leader action
    if leader_action == LeaderAction.TIGHT:
        effective_liquidity = base_liquidity * cfg.tight_liquidity_multiplier
    elif leader_action == LeaderAction.WIDE:
        effective_liquidity = base_liquidity * cfg.wide_liquidity_multiplier
    else:
        effective_liquidity = base_liquidity * cfg.medium_liquidity_multiplier

    trading_activity = expected_buy_pressure + expected_sell_pressure
    volatility_risk = volatility
    inventory_risk = abs(leader_inventory) * volatility
    adverse_flow_risk = abs(expected_order_imbalance)

    u_l = (
        cfg.leader_liquidity_weight * effective_liquidity
        + cfg.leader_activity_weight * trading_activity
        - cfg.leader_volatility_penalty * volatility_risk
        - cfg.leader_inventory_penalty * inventory_risk
        - cfg.leader_adverse_flow_penalty * adverse_flow_risk
    )
    return float(u_l)


def calculate_follower_utility(
    follower_id: str,
    action: str,
    state: Dict[str, Any],
    leader_action: LeaderAction,
    config: Optional[GameTheoryConfig] = None,
) -> float:
    """Calculate simple configurable Follower utility.

    Formulation:
        U_i = follower_return_weight * expected_return
            - follower_transaction_cost_penalty * transaction_cost
            - follower_risk_penalty * risk_penalty
    """
    cfg = config or GameTheoryConfig()
    action = action.upper()

    if action == "HOLD":
        return 0.0

    price = float(state.get("price", 100.0))
    base_spread = float(state.get("spread", 0.02))
    volatility = float(state.get("volatility", 0.01))

    # Leader action impacts effective spread / transaction friction for followers
    if leader_action == LeaderAction.TIGHT:
        effective_spread = base_spread * cfg.tight_spread_multiplier
    elif leader_action == LeaderAction.WIDE:
        effective_spread = base_spread * cfg.wide_spread_multiplier
    else:
        effective_spread = base_spread * cfg.medium_spread_multiplier

    transaction_cost = (effective_spread / 2.0) / max(price, 1e-8)
    risk_penalty = volatility

    # Calculate expected return based on agent type and action
    ret = float(state.get("return", 0.0))
    fundamental = float(state.get("fundamental_value", price))
    mispricing = (fundamental - price) / max(fundamental, 1e-8)

    if follower_id == "momentum":
        # Positive price trend encourages BUY, negative trend encourages SELL
        if action == "BUY":
            expected_return = max(ret, 0.005)
        elif action == "SELL":
            expected_return = max(-ret, 0.005)
        else:
            expected_return = 0.0
    elif follower_id == "value":
        # Mispricing encourages BUY when undervalued, SELL when overvalued
        if action == "BUY":
            expected_return = max(mispricing, 0.005)
        elif action == "SELL":
            expected_return = max(-mispricing, 0.005)
        else:
            expected_return = 0.0
    else:
        # Default / RL trader response fallback
        expected_return = 0.01 if action in ("BUY", "SELL") else 0.0

    u_i = (
        cfg.follower_return_weight * expected_return
        - cfg.follower_transaction_cost_penalty * transaction_cost
        - cfg.follower_risk_penalty * risk_penalty
    )
    return float(u_i)
