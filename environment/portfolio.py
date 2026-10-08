from dataclasses import dataclass


@dataclass
class Portfolio:
    """Cash/inventory accounting for one market participant."""

    cash: float
    inventory: int = 0
    last_price: float = 0.0
    total_transaction_cost: float = 0.0
    total_market_impact_cost: float = 0.0

    @property
    def portfolio_value(self):
        return self.cash + self.inventory * self.last_price

    def mark_to_market(self, price):
        self.last_price = float(price)

    def buy(self, quantity, price, transaction_cost, market_impact_cost):
        if quantity <= 0:
            raise ValueError("Buy quantity must be positive.")

        gross = quantity * price
        total_cost = gross + transaction_cost + market_impact_cost

        if total_cost > self.cash:
            raise ValueError(
                f"Insufficient cash for BUY: required {total_cost:.2f}, "
                f"available {self.cash:.2f}"
            )

        self.cash -= total_cost
        self.inventory += quantity
        self.total_transaction_cost += transaction_cost
        self.total_market_impact_cost += market_impact_cost

    def sell(self, quantity, price, transaction_cost, market_impact_cost):
        if quantity <= 0:
            raise ValueError("Sell quantity must be positive.")

        if quantity > self.inventory:
            raise ValueError(
                f"Insufficient inventory for SELL: requested {quantity}, "
                f"available {self.inventory}"
            )

        gross = quantity * price
        self.cash += gross - transaction_cost - market_impact_cost
        self.inventory -= quantity
        self.total_transaction_cost += transaction_cost
        self.total_market_impact_cost += market_impact_cost

    def snapshot(self, price):
        self.mark_to_market(price)
        return {
            "cash": float(self.cash),
            "inventory": int(self.inventory),
            "portfolio_value": float(self.portfolio_value),
            "transaction_cost": float(self.total_transaction_cost),
            "market_impact_cost": float(self.total_market_impact_cost),
        }