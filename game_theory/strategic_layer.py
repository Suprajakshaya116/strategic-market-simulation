from typing import Dict, Any, Optional
from .config import GameTheoryConfig
from .types import LeaderAction, LeaderDecision, StrategicObservation, CandidateEvaluation
from .stackelberg import StackelbergGame
from .opponent_model import OpponentModel, calculate_expected_order_pressure


class StrategicLayer:
    """Unified Game-Theoretic Strategic Layer for Member 2.

    Combines Stackelberg Decision Engine and Opponent Model cleanly.
    Supports modular toggles for baseline comparisons and ablation studies (Section 27).
    """

    def __init__(
        self,
        config: Optional[GameTheoryConfig] = None,
        use_stackelberg: bool = True,
        use_opponent_model: bool = True,
    ):
        self.config = config or GameTheoryConfig()
        self.config.validate()
        self.use_stackelberg = use_stackelberg
        self.use_opponent_model = use_opponent_model

        self.stackelberg_game = StackelbergGame(self.config) if use_stackelberg else None
        self.opponent_model = OpponentModel(self.config) if use_opponent_model else None

    def reset(self) -> None:
        """Reset internal state (e.g. opponent model statistics)."""
        if self.opponent_model is not None:
            self.opponent_model.reset()

    def update_opponent(self, opponent_id: str, state: Dict[str, Any], action: str) -> None:
        """Update opponent model with observed action."""
        if self.use_opponent_model and self.opponent_model is not None:
            self.opponent_model.update(opponent_id, state, action)

    def solve_leader(self, state: Dict[str, Any]) -> LeaderDecision:
        """Solve leader decision using Stackelberg Game if enabled, or baseline default."""
        if self.use_stackelberg and self.stackelberg_game is not None:
            return self.stackelberg_game.solve(state)

        # Baseline: default MEDIUM action with 0 utility
        return LeaderDecision(
            selected_action=LeaderAction.MEDIUM,
            leader_utility=0.0,
            predicted_follower_actions={"momentum": "HOLD", "value": "HOLD"},
            predicted_follower_utilities={"momentum": 0.0, "value": 0.0},
            candidate_evaluations=[
                CandidateEvaluation(
                    leader_action=LeaderAction.MEDIUM,
                    predicted_follower_actions={"momentum": "HOLD", "value": "HOLD"},
                    predicted_follower_utilities={"momentum": 0.0, "value": 0.0},
                    expected_buy_pressure=0.0,
                    expected_sell_pressure=0.0,
                    expected_order_imbalance=0.0,
                    leader_utility=0.0,
                )
            ],
        )

    def build_observation(
        self,
        state: Dict[str, Any],
        leader_decision: Optional[LeaderDecision] = None,
    ) -> StrategicObservation:
        """Build structured StrategicObservation according to Section 18 & 26 pipeline.

        Combines leader action, leader utility, opponent probabilities, and expected order pressure.
        """
        if leader_decision is None:
            leader_decision = self.solve_leader(state)

        # Opponent probabilities
        if self.use_opponent_model and self.opponent_model is not None:
            mom_probs = self.opponent_model.predict("momentum", state)
            val_probs = self.opponent_model.predict("value", state)
        else:
            # Uniform prior baseline
            mom_probs = {"BUY": 1.0 / 3.0, "HOLD": 1.0 / 3.0, "SELL": 1.0 / 3.0}
            val_probs = {"BUY": 1.0 / 3.0, "HOLD": 1.0 / 3.0, "SELL": 1.0 / 3.0}

        # Expected order pressure calculation
        beliefs = {"momentum": mom_probs, "value": val_probs}
        buy_p, sell_p, imb = calculate_expected_order_pressure(
            beliefs=beliefs,
            expected_quantity=self.config.expected_order_size,
        )

        return StrategicObservation(
            leader_action=leader_decision.selected_action.value,
            leader_utility=float(leader_decision.leader_utility),
            momentum_buy_probability=float(mom_probs["BUY"]),
            momentum_hold_probability=float(mom_probs["HOLD"]),
            momentum_sell_probability=float(mom_probs["SELL"]),
            value_buy_probability=float(val_probs["BUY"]),
            value_hold_probability=float(val_probs["HOLD"]),
            value_sell_probability=float(val_probs["SELL"]),
            expected_buy_pressure=float(buy_p),
            expected_sell_pressure=float(sell_p),
            expected_order_imbalance=float(imb),
        )
