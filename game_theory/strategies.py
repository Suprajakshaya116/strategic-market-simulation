from typing import Dict, Any, Optional
from .config import GameTheoryConfig
from .types import LeaderAction


class FollowerResponseModel:
    """Strategic abstraction to estimate follower responses under candidate leader actions.

    Does not duplicate agent code; leverages MomentumAgent and ValueAgent logic while adjusting
    for strategic conditions created by the Leader's action (e.g., TIGHT, MEDIUM, WIDE spreads).
    """

    def __init__(self, config: Optional[GameTheoryConfig] = None):
        from agents.momentum_agent import MomentumAgent
        from agents.value_agent import ValueAgent

        self.config = config or GameTheoryConfig()
        self.momentum_agent = MomentumAgent()
        self.value_agent = ValueAgent()

    def predict_response(
        self,
        follower_id: str,
        state: Dict[str, Any],
        leader_action: LeaderAction,
    ) -> str:
        """Predict follower action ('BUY', 'HOLD', or 'SELL') given state and leader action."""
        follower_id = follower_id.lower()
        adjusted_state = self._adjust_state_for_leader_action(state, leader_action)

        if follower_id == "momentum":
            return self.momentum_agent.act(adjusted_state)
        elif follower_id == "value":
            return self.value_agent.act(adjusted_state)
        elif follower_id == "rl_trader":
            # For strategic prediction before RL policy is attached, return neutral HOLD or rule heuristic
            return "HOLD"
        else:
            return "HOLD"

    def _adjust_state_for_leader_action(
        self,
        state: Dict[str, Any],
        leader_action: LeaderAction,
    ) -> Dict[str, Any]:
        """Produce an adjusted state dict reflecting the leader's action on spread and price perceived by followers."""
        adjusted = dict(state)
        base_spread = float(state.get("spread", 0.02))

        if leader_action == LeaderAction.TIGHT:
            multiplier = self.config.tight_spread_multiplier
        elif leader_action == LeaderAction.WIDE:
            multiplier = self.config.wide_spread_multiplier
        else:
            multiplier = self.config.medium_spread_multiplier

        adjusted["spread"] = base_spread * multiplier
        return adjusted
