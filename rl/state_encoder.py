import math
import numpy as np
from typing import Any
from .config import StateEncoderConfig


class RunningMeanStd:
    """Tracks running mean and variance using Welford's algorithm for online normalization."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip_limit: float = 10.0) -> np.ndarray:
        norm = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        if clip_limit is not None and clip_limit > 0:
            norm = np.clip(norm, -clip_limit, clip_limit)
        return norm.astype(np.float32)


class StateEncoder:
    """Encodes raw environment market states + strategic state into normalized features.

    Raw Market State Schema (10 features):
      - price: float
      - return: float
      - volume: float
      - spread: float
      - liquidity: float
      - volatility: float
      - order_imbalance: float
      - fundamental_value: float
      - inventory: int
      - cash: float

    Domain Feature Transformations:
      1. Relative mispricing: (price - fundamental_value) / (fundamental_value + eps)
      2. Price return: return
      3. Volume scale: volume / 100.0
      4. Relative spread: spread / initial_spread
      5. Liquidity level: liquidity
      6. Volatility level: volatility
      7. Order imbalance: order_imbalance
      8. Normalized inventory: inventory / max_order_size
      9. Normalized cash ratio: (cash - initial_cash) / initial_cash
     10. Normalized price level: (price - initial_price) / initial_price
    """

    def __init__(self, config: StateEncoderConfig | None = None, strategic_dim: int = 0):
        self.config = config or StateEncoderConfig()
        self.market_dim = 10
        self.strategic_dim = strategic_dim
        self.total_dim = self.market_dim + self.strategic_dim

        self.running_ms = RunningMeanStd(shape=(self.market_dim,))

    def encode_market_dict(self, state: dict) -> np.ndarray:
        """Transform raw market state dictionary into structured feature vector."""
        price = float(state.get("price", self.config.initial_price))
        ret = float(state.get("return", 0.0))
        volume = float(state.get("volume", 0.0))
        spread = float(state.get("spread", self.config.initial_spread))
        liquidity = float(state.get("liquidity", 1.0))
        volatility = float(state.get("volatility", 0.01))
        imbalance = float(state.get("order_imbalance", 0.0))
        fundamental = float(state.get("fundamental_value", self.config.initial_price))
        inventory = float(state.get("inventory", 0))
        cash = float(state.get("cash", self.config.initial_cash))

        # Domain relative scaling
        rel_mispricing = (price - fundamental) / (fundamental + self.config.epsilon)
        scaled_volume = volume / 100.0
        rel_spread = spread / (self.config.initial_spread + self.config.epsilon)
        scaled_inventory = inventory / float(self.config.max_order_size)
        scaled_cash = (cash - self.config.initial_cash) / (self.config.initial_cash + self.config.epsilon)
        rel_price = (price - self.config.initial_price) / (self.config.initial_price + self.config.epsilon)

        raw_vec = np.array([
            rel_mispricing,
            ret,
            scaled_volume,
            rel_spread,
            liquidity,
            volatility,
            imbalance,
            scaled_inventory,
            scaled_cash,
            rel_price
        ], dtype=np.float32)

        return raw_vec

    def encode_strategic(self, strategic_dict: Any) -> np.ndarray:
        """Encode optional strategic game theory information.

        Supports:
          - Member 2 StrategicObservation instance (via to_feature_vector())
          - 11-dimensional feature vector (list / np.ndarray)
          - Dictionary containing leader_action, opponent_probs, expected_response
        """
        if self.strategic_dim == 0 or strategic_dict is None:
            return np.zeros((self.strategic_dim,), dtype=np.float32)

        if hasattr(strategic_dict, "to_feature_vector"):
            strat_vec = np.array(strategic_dict.to_feature_vector(), dtype=np.float32)
        elif isinstance(strategic_dict, (list, np.ndarray)):
            strat_vec = np.array(strategic_dict, dtype=np.float32)
        elif isinstance(strategic_dict, dict) and "feature_vector" in strategic_dict:
            strat_vec = np.array(strategic_dict["feature_vector"], dtype=np.float32)
        elif isinstance(strategic_dict, dict):
            feats = []
            if "leader_action" in strategic_dict:
                act = strategic_dict["leader_action"]
                one_hot = np.zeros(3, dtype=np.float32)
                if isinstance(act, int) and 0 <= act < 3:
                    one_hot[act] = 1.0
                elif isinstance(act, str):
                    act_map = {"HOLD": 0, "BUY": 1, "SELL": 2, "TIGHT": 0, "MEDIUM": 1, "WIDE": 2}
                    if act.upper() in act_map:
                        one_hot[act_map[act.upper()]] = 1.0
                feats.append(one_hot)

            if "opponent_probs" in strategic_dict:
                probs = np.array(strategic_dict["opponent_probs"], dtype=np.float32)
                feats.append(probs)

            if "expected_response" in strategic_dict:
                resp = np.array([float(strategic_dict["expected_response"])], dtype=np.float32)
                feats.append(resp)

            strat_vec = np.concatenate(feats, axis=0) if feats else np.zeros((self.strategic_dim,), dtype=np.float32)
        else:
            strat_vec = np.zeros((self.strategic_dim,), dtype=np.float32)

        # Pad or truncate to self.strategic_dim
        if len(strat_vec) < self.strategic_dim:
            strat_vec = np.pad(strat_vec, (0, self.strategic_dim - len(strat_vec)))
        elif len(strat_vec) > self.strategic_dim:
            strat_vec = strat_vec[:self.strategic_dim]

        return strat_vec.astype(np.float32)

    def encode(
        self,
        state: dict,
        strategic_dict: Any = None,
        update_stats: bool = False
    ) -> np.ndarray:
        """Return full encoded and normalized feature vector."""
        m_vec = self.encode_market_dict(state)

        if update_stats and self.config.use_running_stats:
            self.running_ms.update(m_vec[None, :])

        if self.config.use_running_stats:
            m_norm = self.running_ms.normalize(m_vec, clip_limit=self.config.clip_observation)
        else:
            m_norm = m_vec

        if self.strategic_dim > 0:
            s_vec = self.encode_strategic(strategic_dict)
            return np.concatenate([m_norm, s_vec], axis=0).astype(np.float32)

        return m_norm.astype(np.float32)

    def get_state_dict(self) -> dict:
        return {
            "mean": self.running_ms.mean,
            "var": self.running_ms.var,
            "count": self.running_ms.count
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.running_ms.mean = state_dict["mean"]
        self.running_ms.var = state_dict["var"]
        self.running_ms.count = state_dict["count"]
