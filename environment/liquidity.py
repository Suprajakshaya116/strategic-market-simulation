class LiquidityModel:
    """Update liquidity and bid-ask spread from order-flow pressure with economic bounds.

    Market Microstructure Formulation:
      - Liquidity: Depletes under order flow pressure, recovers toward initial_liquidity.
      - Bid-Ask Spread: Expands under order flow pressure relative to initial spread,
        with mean reversion toward baseline and an explicit economic maximum bound.
    """

    def __init__(
        self,
        initial_liquidity: float,
        initial_spread: float,
        impact: float,
        recovery: float,
        max_spread_multiplier: float = 20.0
    ):
        self.initial_liquidity = float(initial_liquidity)
        self.initial_spread = float(initial_spread)
        self.impact = float(impact)
        self.recovery = float(recovery)
        # Economic upper bound: spread cannot exceed a reasonable multiple of initial spread (default 20x)
        self.max_spread = float(self.initial_spread * max_spread_multiplier)

    def update(self, liquidity: float, spread: float, imbalance: float) -> tuple[float, float]:
        pressure = abs(float(imbalance))
        new_liquidity = liquidity * (1.0 - self.impact * pressure)
        new_liquidity += self.recovery * (self.initial_liquidity - new_liquidity)
        new_liquidity = max(new_liquidity, 1e-8)

        # Target spread widens with order-flow imbalance pressure anchored to initial spread
        target_spread = self.initial_spread * (1.0 + pressure * 2.0)

        # Autoregressive adjustment toward target spread with recovery
        new_spread = (1.0 - self.recovery) * spread + self.recovery * target_spread

        # Enforce economic bounds: non-negative and bounded ceiling
        new_spread = max(min(new_spread, self.max_spread), 1e-8)
        return new_liquidity, new_spread
