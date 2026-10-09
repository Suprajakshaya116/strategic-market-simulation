from typing import Dict, Any, List, Optional
from .config import GameTheoryConfig
from .types import LeaderAction, CandidateEvaluation, LeaderDecision
from .utility import calculate_leader_utility, calculate_follower_utility
from .strategies import FollowerResponseModel


class StackelbergGame:
    """Computational Discrete Stackelberg Game Solver.

    Theoretical Formulation:
        Follower Best Response: B_i(a_L) = argmax_a U_i(a, a_L)
        Leader Decision:        a_L* = argmax_{a_L} U_L(a_L, B(a_L))
    """

    def __init__(
        self,
        config: Optional[GameTheoryConfig] = None,
        follower_response_model: Optional[FollowerResponseModel] = None,
        opponent_model: Optional[Any] = None,
    ):
        self.config = config or GameTheoryConfig()
        self.config.validate()
        self.follower_model = follower_response_model or FollowerResponseModel(self.config)
        self.opponent_model = opponent_model

    def solve(self, state: Dict[str, Any]) -> LeaderDecision:
        """Solve the discrete Stackelberg game for the given market state."""
        candidate_evaluations: List[CandidateEvaluation] = []
        followers = ["momentum", "value"]

        for action in LeaderAction:
            # 1. Predict follower responses
            predicted_actions: Dict[str, str] = {}
            predicted_utilities: Dict[str, float] = {}

            for f_id in followers:
                pred_act = self.follower_model.predict_response(f_id, state, action)
                pred_util = calculate_follower_utility(f_id, pred_act, state, action, self.config)
                predicted_actions[f_id] = pred_act
                predicted_utilities[f_id] = pred_util

            # 2. Estimate aggregate order pressure
            buy_count = sum(1 for act in predicted_actions.values() if act == "BUY")
            sell_count = sum(1 for act in predicted_actions.values() if act == "SELL")

            buy_pressure = buy_count * self.config.expected_order_size
            sell_pressure = sell_count * self.config.expected_order_size

            denom = buy_pressure + sell_pressure + 1e-8
            order_imbalance = (buy_pressure - sell_pressure) / denom

            # 3. Calculate leader utility
            leader_util = calculate_leader_utility(
                state=state,
                leader_action=action,
                expected_buy_pressure=buy_pressure,
                expected_sell_pressure=sell_pressure,
                expected_order_imbalance=order_imbalance,
                config=self.config,
            )

            cand = CandidateEvaluation(
                leader_action=action,
                predicted_follower_actions=predicted_actions,
                predicted_follower_utilities=predicted_utilities,
                expected_buy_pressure=buy_pressure,
                expected_sell_pressure=sell_pressure,
                expected_order_imbalance=order_imbalance,
                leader_utility=leader_util,
            )
            candidate_evaluations.append(cand)

        # 4. Select leader action with maximum leader utility (deterministic tie-breaking by enum order)
        best_cand = max(candidate_evaluations, key=lambda c: (c.leader_utility, -c.leader_action.value.encode()[0]))

        return LeaderDecision(
            selected_action=best_cand.leader_action,
            leader_utility=best_cand.leader_utility,
            predicted_follower_actions=best_cand.predicted_follower_actions,
            predicted_follower_utilities=best_cand.predicted_follower_utilities,
            candidate_evaluations=candidate_evaluations,
        )
