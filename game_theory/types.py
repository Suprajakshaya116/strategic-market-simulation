from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any


class LeaderAction(Enum):
    """Discrete action space for the Market-Maker / Strategic Liquidity Controller."""
    TIGHT = "TIGHT"
    MEDIUM = "MEDIUM"
    WIDE = "WIDE"

    @classmethod
    def from_str(cls, val: str) -> "LeaderAction":
        val_upper = str(val).upper()
        if val_upper not in cls.__members__:
            raise ValueError(f"Invalid LeaderAction: '{val}'. Must be one of {[e.value for e in cls]}")
        return cls[val_upper]


@dataclass
class CandidateEvaluation:
    """Evaluation result for a single candidate leader action during Stackelberg solve."""
    leader_action: LeaderAction
    predicted_follower_actions: Dict[str, str]
    predicted_follower_utilities: Dict[str, float]
    expected_buy_pressure: float
    expected_sell_pressure: float
    expected_order_imbalance: float
    leader_utility: float


@dataclass
class LeaderDecision:
    """Final decision outcome produced by the Stackelberg game solver."""
    selected_action: LeaderAction
    leader_utility: float
    predicted_follower_actions: Dict[str, str]
    predicted_follower_utilities: Dict[str, float]
    candidate_evaluations: List[CandidateEvaluation] = field(default_factory=list)


@dataclass
class StrategicObservation:
    """Strategic observation structure designed for Member 3's RL agent consumption."""
    leader_action: str
    leader_utility: float
    momentum_buy_probability: float
    momentum_hold_probability: float
    momentum_sell_probability: float
    value_buy_probability: float
    value_hold_probability: float
    value_sell_probability: float
    expected_buy_pressure: float
    expected_sell_pressure: float
    expected_order_imbalance: float

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate probabilities and bounds."""
        m_sum = self.momentum_buy_probability + self.momentum_hold_probability + self.momentum_sell_probability
        if not (0.99 <= m_sum <= 1.01):
            raise ValueError(f"Momentum probabilities must sum to 1.0, got {m_sum:.4f}")

        v_sum = self.value_buy_probability + self.value_hold_probability + self.value_sell_probability
        if not (0.99 <= v_sum <= 1.01):
            raise ValueError(f"Value probabilities must sum to 1.0, got {v_sum:.4f}")

    def to_feature_vector(self) -> List[float]:
        """Convert the strategic observation into a deterministic float feature vector for RL policies.

        Feature Indexing Order:
        0: leader_action (TIGHT=0.0, MEDIUM=1.0, WIDE=2.0)
        1: leader_utility
        2: momentum_buy_probability
        3: momentum_hold_probability
        4: momentum_sell_probability
        5: value_buy_probability
        6: value_hold_probability
        7: value_sell_probability
        8: expected_buy_pressure
        9: expected_sell_pressure
        10: expected_order_imbalance
        """
        action_map = {
            LeaderAction.TIGHT.value: 0.0,
            LeaderAction.MEDIUM.value: 1.0,
            LeaderAction.WIDE.value: 2.0,
        }
        action_val = action_map.get(str(self.leader_action).upper(), 1.0)

        return [
            float(action_val),
            float(self.leader_utility),
            float(self.momentum_buy_probability),
            float(self.momentum_hold_probability),
            float(self.momentum_sell_probability),
            float(self.value_buy_probability),
            float(self.value_hold_probability),
            float(self.value_sell_probability),
            float(self.expected_buy_pressure),
            float(self.expected_sell_pressure),
            float(self.expected_order_imbalance),
        ]
