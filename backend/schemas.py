from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class MarketStateSchema(BaseModel):
    price: float
    return_rate: float = Field(alias="return")
    volume: float
    spread: float
    liquidity: float
    volatility: float
    order_imbalance: float
    fundamental_value: float
    bid_price: float
    ask_price: float


class AgentStateSchema(BaseModel):
    agent_id: str
    action: str  # BUY, SELL, HOLD
    action_status: str  # EXECUTED, REJECTED, HOLD
    quantity: int
    fill_price: float
    cash: float
    inventory: int
    portfolio_value: float
    reward: float = 0.0
    cumulative_pnl: float = 0.0
    transaction_cost: float = 0.0
    market_impact_cost: float = 0.0
    signal_metric: Optional[float] = None
    signal_label: Optional[str] = None


class CandidateEvaluationSchema(BaseModel):
    leader_action: str
    predicted_follower_actions: Dict[str, str]
    predicted_follower_utilities: Dict[str, float]
    expected_buy_pressure: float
    expected_sell_pressure: float
    expected_order_imbalance: float
    leader_utility: float


class LeaderDecisionSchema(BaseModel):
    selected_action: str
    leader_utility: float
    spread_multiplier: float
    liquidity_multiplier: float
    explanation: str
    candidate_evaluations: List[CandidateEvaluationSchema]


class OpponentBeliefSchema(BaseModel):
    agent_id: str
    buy_prob: float
    hold_prob: float
    sell_prob: float
    observation_count: int
    state_bucket: str
    last_observed_action: Optional[str] = None


class SimulationFrameSchema(BaseModel):
    step: int
    episode: int
    status: str
    timestamp: float
    market_state: MarketStateSchema
    leader_decision: LeaderDecisionSchema
    agents: Dict[str, AgentStateSchema]
    opponent_beliefs: Dict[str, OpponentBeliefSchema]
    recent_orders: List[Dict[str, Any]]
    encoded_state_dim: int
    strategic_feature_dim: int
    active_variant: str


class EventLogSchema(BaseModel):
    id: str
    timestamp: str
    step: int
    category: str  # MARKET, STACKELBERG, OPPONENT, AGENT, ORDER, PPO
    message: str
    details: Optional[Dict[str, Any]] = None


class SimulationConfigSchema(BaseModel):
    episode_length: int = 100
    initial_price: float = 100.0
    initial_cash: float = 100000.0
    initial_spread: float = 0.02
    initial_liquidity: float = 100.0
    seed: int = 42
    delay_ms: int = 250
    variant: str = "proposed_full"
