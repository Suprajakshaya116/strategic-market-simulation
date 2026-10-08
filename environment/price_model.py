import math

import numpy as np


class PriceModel:
    """Price dynamics based on drift, order imbalance and volatility noise."""

    def __init__(
        self,
        drift,
        imbalance_impact,
        noise_scale,
        fundamental_reversion,
        seed=None,
    ):
        self.drift = float(drift)
        self.imbalance_impact = float(imbalance_impact)
        self.noise_scale = float(noise_scale)
        self.fundamental_reversion = float(fundamental_reversion)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        """Reset the random generator for reproducible episode starts."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def order_imbalance(buy_volume, sell_volume, eps=1e-8):
        return (buy_volume - sell_volume) / (buy_volume + sell_volume + eps)

    def update(self, price, fundamental_value, imbalance, volatility):
        """Apply the simplified exponential price-impact model from the specification."""
        if price <= 0:
            raise ValueError("price must be > 0")

        fundamental_term = self.fundamental_reversion * (
            (fundamental_value - price) / price
        )
        noise = self.noise_scale * max(float(volatility), 0.0) * self.rng.normal()
        log_return = (
            self.drift
            + self.imbalance_impact * imbalance
            + fundamental_term
            + noise
        )
        return max(price * math.exp(log_return), 1e-8)
