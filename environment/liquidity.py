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

        # Spread widens as order-flow pressure rises, with recovery towards initial_spread.
        new_spread = spread * (1.0 + pressure)
        new_spread += self.recovery * (self.initial_spread - new_spread)
        new_spread = max(new_spread, 1e-8)
        return new_liquidity, new_spread
