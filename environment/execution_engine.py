class ExecutionEngine:
    """Execute validated orders against portfolios.

    This is the single execution boundary for all agents.
    Agents submit intent; the execution engine decides whether
    that intent can actually be executed.
    """

    def __init__(self, portfolios, config):
        self.portfolios = portfolios
        self.config = config

    def trade_costs(self, quantity, price, liquidity):
        notional = abs(float(quantity) * float(price))

        transaction_cost = (
            notional * self.config.transaction_cost_rate
        )

        market_impact_cost = (
            notional
            * self.config.market_impact_rate
            / max(float(liquidity), 1e-8)
        )

        return transaction_cost, market_impact_cost

    def execute_order(self, order, price, liquidity):
        """Execute one validated order.

        Returns an execution record. A rejected order does not
        modify the portfolio or market volume.
        """

        if order.quantity == 0:
            return {
                "agent_id": order.agent_id,
                "action": order.action,
                "requested_quantity": 0,
                "quantity": 0,
                "status": "HOLD",
                "reason": None,
                "transaction_cost": 0.0,
                "market_impact_cost": 0.0,
                "notional": 0.0,
            }

        portfolio = self.portfolios[order.agent_id]

        transaction_cost, market_impact_cost = self.trade_costs(
            order.quantity,
            price,
            liquidity,
        )

        notional = float(order.quantity * price)

        try:
            if order.action == "BUY":
                portfolio.buy(
                    order.quantity,
                    price,
                    transaction_cost,
                    market_impact_cost,
                )

            elif order.action == "SELL":
                portfolio.sell(
                    order.quantity,
                    price,
                    transaction_cost,
                    market_impact_cost,
                )

            else:
                raise ValueError(
                    f"Unsupported execution action: {order.action}"
                )

        except ValueError as exc:
            return {
                "agent_id": order.agent_id,
                "action": order.action,
                "requested_quantity": int(order.quantity),
                "quantity": 0,
                "status": "REJECTED",
                "reason": str(exc),
                "transaction_cost": 0.0,
                "market_impact_cost": 0.0,
                "notional": 0.0,
            }

        return {
            "agent_id": order.agent_id,
            "action": order.action,
            "requested_quantity": int(order.quantity),
            "quantity": int(order.quantity),
            "status": "EXECUTED",
            "reason": None,
            "transaction_cost": float(transaction_cost),
            "market_impact_cost": float(market_impact_cost),
            "notional": notional,
        }

    def execute(self, orders, price, liquidity):
        """Execute a batch of validated orders."""

        return [
            self.execute_order(order, price, liquidity)
            for order in orders
        ]

    @staticmethod
    def aggregate_executed(executions):
        return {
            "buy_volume": float(
                sum(
                    item["quantity"]
                    for item in executions
                    if item["status"] == "EXECUTED"
                    and item["action"] == "BUY"
                )
            ),
            "sell_volume": float(
                sum(
                    item["quantity"]
                    for item in executions
                    if item["status"] == "EXECUTED"
                    and item["action"] == "SELL"
                )
            ),
        }