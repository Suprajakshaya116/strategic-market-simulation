class LiquidityModel:
    """Update liquidity and bid-ask spread from order-flow pressure."""

    def __init__(self, initial_liquidity, initial_spread, impact, recovery):
        self.initial_liquidity = float(initial_liquidity)
        self.initial_spread = float(initial_spread)
        self.impact = float(impact)
        self.recovery = float(recovery)

    def update(self, liquidity, spread, imbalance):
        pressure = abs(float(imbalance))
        new_liquidity = liquidity * (1.0 - self.impact * pressure)
        new_liquidity += self.recovery * (self.initial_liquidity - new_liquidity)
        new_liquidity = max(new_liquidity, 1e-8)

        # Spread widens as order-flow pressure rises.
        new_spread = max(self.initial_spread * (1.0 + pressure), 1e-8)
        return new_liquidity, new_spread
