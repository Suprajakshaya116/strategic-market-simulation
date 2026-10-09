from typing import Dict, Any, Optional
from game_theory.config import GameTheoryConfig
from game_theory.types import LeaderDecision, LeaderAction
from game_theory.strategic_layer import StrategicLayer


class MarketMaker:
    """Strategic Leader / Market Maker Agent for Member 2.

    Uses the Game-Theoretic Strategic Layer to make Stackelberg leader decisions
    and model follower opponents.
    """

    def __init__(
        self,
        config: Optional[GameTheoryConfig] = None,
        use_stackelberg: bool = True,
        use_opponent_model: bool = True,
    ):
        self.config = config or GameTheoryConfig()
        self.strategic_layer = StrategicLayer(
            config=self.config,
            use_stackelberg=use_stackelberg,
            use_opponent_model=use_opponent_model,
        )

    def reset(self) -> None:
        """Reset internal strategic layer state."""
        self.strategic_layer.reset()

    def act(self, state: Dict[str, Any]) -> LeaderDecision:
        """Select strategic leader action given current market state."""
        return self.strategic_layer.solve_leader(state)

    def update_opponent(self, opponent_id: str, state: Dict[str, Any], action: str) -> None:
        """Update opponent model with observed follower action."""
        self.strategic_layer.update_opponent(opponent_id, state, action)

    def get_strategic_observation(
        self,
        state: Dict[str, Any],
        leader_decision: Optional[LeaderDecision] = None,
    ):
        """Build strategic observation feature vector for MARL policy consumption."""
        return self.strategic_layer.build_observation(state, leader_decision)
