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
            predicted_actions: Dict[str, str] = {}
            predicted_utilities: Dict[str, float] = {}
            buy_pressure = 0.0
            sell_pressure = 0.0

            for f_id in followers:
                if self.opponent_model is not None:
                    # 1A. Opponent-Model Informed Response: modulate learned baseline by candidate payoffs
                    base_probs = self.opponent_model.predict(f_id, state)
                    temp = max(getattr(self.config, "temperature", 0.5), 1e-4)

                    u_buy = calculate_follower_utility(f_id, "BUY", state, action, self.config)
                    u_hold = calculate_follower_utility(f_id, "HOLD", state, action, self.config)
                    u_sell = calculate_follower_utility(f_id, "SELL", state, action, self.config)

                    import math
                    w_buy = base_probs["BUY"] * math.exp(max(min(u_buy / temp, 20.0), -20.0))
                    w_hold = base_probs["HOLD"] * math.exp(max(min(u_hold / temp, 20.0), -20.0))
                    w_sell = base_probs["SELL"] * math.exp(max(min(u_sell / temp, 20.0), -20.0))
                    w_sum = w_buy + w_hold + w_sell + 1e-12

                    p_buy = w_buy / w_sum
                    p_hold = w_hold / w_sum
                    p_sell = w_sell / w_sum

                    cond_probs = {"BUY": p_buy, "HOLD": p_hold, "SELL": p_sell}
                    pred_act = max(cond_probs, key=cond_probs.get)
                    pred_util = p_buy * u_buy + p_hold * u_hold + p_sell * u_sell

                    buy_pressure += p_buy * self.config.expected_order_size
                    sell_pressure += p_sell * self.config.expected_order_size
                else:
                    # 1B. Baseline Stackelberg: Rule-based best response estimation
                    pred_act = self.follower_model.predict_response(f_id, state, action)
                    pred_util = calculate_follower_utility(f_id, pred_act, state, action, self.config)
                    if pred_act == "BUY":
                        buy_pressure += self.config.expected_order_size
                    elif pred_act == "SELL":
                        sell_pressure += self.config.expected_order_size

                predicted_actions[f_id] = pred_act
                predicted_utilities[f_id] = pred_util

            denom = buy_pressure + sell_pressure + 1e-8
            order_imbalance = (buy_pressure - sell_pressure) / denom

            # 2. Calculate leader utility
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

        # 3. Select leader action with maximum leader utility (deterministic tie-breaking by enum order)
        best_cand = max(candidate_evaluations, key=lambda c: (c.leader_utility, -c.leader_action.value.encode()[0]))

        return LeaderDecision(
            selected_action=best_cand.leader_action,
            leader_utility=best_cand.leader_utility,
            predicted_follower_actions=best_cand.predicted_follower_actions,
            predicted_follower_utilities=best_cand.predicted_follower_utilities,
            candidate_evaluations=candidate_evaluations,
        )
