from collections import defaultdict
from typing import Dict, Any, Tuple, Optional
from .config import GameTheoryConfig


class OpponentModel:
    """Lightweight, statistical frequency-based opponent model.

    Estimates P(action | discretized_state) for each opponent using Laplace smoothing.
    No deep learning dependencies.
    """

    def __init__(self, config: Optional[GameTheoryConfig] = None):
        self.config = config or GameTheoryConfig()
        self.config.validate()
        self.reset()

    def reset(self) -> None:
        """Reset all state-action frequency counts."""
        # Structure: counts[opponent_id][state_key][action] = int
        self.counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"BUY": 0, "HOLD": 0, "SELL": 0})
        )
        self.total_observations: Dict[str, int] = defaultdict(int)

    def _discretize_state(self, state: Dict[str, Any]) -> str:
        """Deterministic discretization of market state into compact state key."""
        ret = float(state.get("return", 0.0))
        vol = float(state.get("volatility", 0.01))
        price = float(state.get("price", 100.0))
        fund = float(state.get("fundamental_value", price))
        imbalance = float(state.get("order_imbalance", 0.0))

        # 1. Return direction
        if ret > 0.001:
            ret_reg = "POS"
        elif ret < -0.001:
            ret_reg = "NEG"
        else:
            ret_reg = "FLAT"

        # 2. Volatility regime
        vol_reg = "HIGH_VOL" if vol > 0.02 else "LOW_VOL"

        # 3. Mispricing direction
        mispricing = (fund - price) / max(fund, 1e-8)
        if mispricing > 0.01:
            misp_reg = "UNDERVALUED"
        elif mispricing < -0.01:
            misp_reg = "OVERVALUED"
        else:
            misp_reg = "FAIR"

        # 4. Imbalance regime
        if imbalance > 0.2:
            imb_reg = "BUY_HEAVY"
        elif imbalance < -0.2:
            imb_reg = "SELL_HEAVY"
        else:
            imb_reg = "BALANCED"

        return f"{ret_reg}|{vol_reg}|{misp_reg}|{imb_reg}"

    def update(self, opponent_id: str, state: Dict[str, Any], action: str) -> None:
        """Update action frequency counts for a given opponent and state."""
        opp_id = opponent_id.lower()
        act = action.upper()
        if act not in ("BUY", "HOLD", "SELL"):
            raise ValueError(f"Invalid opponent action: '{action}'. Must be BUY, HOLD, or SELL.")

        state_key = self._discretize_state(state)
        self.counts[opp_id][state_key][act] += 1
        self.total_observations[opp_id] += 1

    def predict(self, opponent_id: str, state: Dict[str, Any]) -> Dict[str, float]:
        """Predict action probability distribution P(BUY/HOLD/SELL | state) for opponent.

        Uses Laplace smoothing: P(a | s) = (count(s, a) + alpha) / (total_count(s) + 3 * alpha).
        Unseen states produce uniform prior (1/3, 1/3, 1/3).
        """
        opp_id = opponent_id.lower()
        state_key = self._discretize_state(state)
        alpha = self.config.smoothing_parameter

        if opp_id not in self.counts or state_key not in self.counts[opp_id]:
            # Unseen state: return uniform prior
            return {"BUY": 1.0 / 3.0, "HOLD": 1.0 / 3.0, "SELL": 1.0 / 3.0}

        s_counts = self.counts[opp_id][state_key]
        total_s = sum(s_counts.values())

        if total_s == 0:
            return {"BUY": 1.0 / 3.0, "HOLD": 1.0 / 3.0, "SELL": 1.0 / 3.0}

        denom = total_s + 3.0 * alpha
        probs = {
            act: float(s_counts[act] + alpha) / denom
            for act in ("BUY", "HOLD", "SELL")
        }

        # Normalize to ensure sum is exactly 1.0
        prob_sum = sum(probs.values())
        return {k: v / prob_sum for k, v in probs.items()}

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics of opponent models."""
        return {
            opp_id: {
                "total_observations": self.total_observations[opp_id],
                "states_observed": len(states_dict),
            }
            for opp_id, states_dict in self.counts.items()
        }


def calculate_expected_order_pressure(
    beliefs: Dict[str, Dict[str, float]],
    expected_quantity: float = 100.0,
) -> Tuple[float, float, float]:
    """Calculate aggregate expected buy pressure, sell pressure, and order imbalance from opponent beliefs.

    Returns:
        (expected_buy_pressure, expected_sell_pressure, expected_order_imbalance)
    """
    buy_pressure = 0.0
    sell_pressure = 0.0

    for opp_id, belief in beliefs.items():
        p_buy = float(belief.get("BUY", 1.0 / 3.0))
        p_sell = float(belief.get("SELL", 1.0 / 3.0))
        buy_pressure += p_buy * expected_quantity
        sell_pressure += p_sell * expected_quantity

    denom = buy_pressure + sell_pressure + 1e-8
    imbalance = (buy_pressure - sell_pressure) / denom

    return float(buy_pressure), float(sell_pressure), float(imbalance)
