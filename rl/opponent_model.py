import numpy as np


class OpponentModel:
    """Base interface for tracking and modeling opponent trader behavior."""

    def update(self, agent_id: str, action: str, state: dict) -> None:
        raise NotImplementedError

    def get_action_probs(self, agent_id: str) -> np.ndarray:
        raise NotImplementedError


class EmpiricalOpponentModel(OpponentModel):
    """Tracks empirical action distribution P(a_opp) for each opponent agent."""

    def __init__(self, agent_ids: tuple[str, ...], num_actions: int = 3):
        self.agent_ids = tuple(aid for aid in agent_ids if aid != "rl_trader")
        self.num_actions = num_actions
        self.action_counts = {aid: np.zeros(num_actions, dtype=np.float64) for aid in self.agent_ids}
        self.total_counts = {aid: 0.0 for aid in self.agent_ids}

        self.act_map = {"HOLD": 0, "BUY": 1, "SELL": 2}

    def update(self, agent_id: str, action: str | tuple, state: dict) -> None:
        if agent_id not in self.action_counts:
            return
        act_str = action[0] if isinstance(action, (tuple, list)) else action
        act_str = str(act_str).upper()

        if act_str in self.act_map:
            idx = self.act_map[act_str]
            self.action_counts[agent_id][idx] += 1.0
            self.total_counts[agent_id] += 1.0

    def get_action_probs(self, agent_id: str) -> np.ndarray:
        if agent_id not in self.action_counts or self.total_counts[agent_id] == 0:
            return np.ones(self.num_actions, dtype=np.float32) / float(self.num_actions)
        probs = self.action_counts[agent_id] / self.total_counts[agent_id]
        return probs.astype(np.float32)

    def get_combined_opponent_probs(self) -> np.ndarray:
        """Returns average opponent action probability distribution across all opponents."""
        if not self.agent_ids:
            return np.ones(self.num_actions, dtype=np.float32) / float(self.num_actions)
        all_probs = [self.get_action_probs(aid) for aid in self.agent_ids]
        return np.mean(all_probs, axis=0).astype(np.float32)


class StrategicStateAdapter:
    """Combines environment state and game-theoretic signals into a strategic state dictionary."""

    def __init__(self, opponent_model: OpponentModel | None = None):
        self.opponent_model = opponent_model

    def build_strategic_dict(
        self,
        leader_action: str | int | None = None,
        expected_response: float | None = None
    ) -> dict:
        strat = {}
        if leader_action is not None:
            strat["leader_action"] = leader_action
        if self.opponent_model is not None and isinstance(self.opponent_model, EmpiricalOpponentModel):
            strat["opponent_probs"] = self.opponent_model.get_combined_opponent_probs()
        if expected_response is not None:
            strat["expected_response"] = expected_response
        return strat
