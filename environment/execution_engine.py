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

    def execute_order(self, order, price, liquidity, bid_price=None, ask_price=None):
        """Execute one validated order using market bid/ask quotes.

        BUY orders execute at the ask_price (price + spread / 2).
        SELL orders execute at the bid_price (price - spread / 2).
        Falls back to price if quotes are not provided for backward compatibility.
        """
        if order.quantity == 0:
            return {
                "agent_id": order.agent_id,
                "action": order.action,
                "requested_quantity": 0,
                "quantity": 0,
                "status": "HOLD",
                "reason": None,
                "fill_price": float(price),
                "transaction_cost": 0.0,
                "market_impact_cost": 0.0,
                "notional": 0.0,
            }

        portfolio = self.portfolios[order.agent_id]

        # Determine execution fill price based on bid/ask convention
        if order.action == "BUY":
            fill_price = float(ask_price) if ask_price is not None else float(price)
        elif order.action == "SELL":
            fill_price = float(bid_price) if bid_price is not None else float(price)
        else:
            fill_price = float(price)

        transaction_cost, market_impact_cost = self.trade_costs(
            order.quantity,
            fill_price,
            liquidity,
        )

        notional = float(order.quantity * fill_price)

        try:
            if order.action == "BUY":
                portfolio.buy(
                    order.quantity,
                    fill_price,
                    transaction_cost,
                    market_impact_cost,
                )

            elif order.action == "SELL":
                portfolio.sell(
                    order.quantity,
                    fill_price,
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
                "fill_price": fill_price,
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
            "fill_price": fill_price,
            "transaction_cost": float(transaction_cost),
            "market_impact_cost": float(market_impact_cost),
            "notional": notional,
        }

    def execute(self, orders, price, liquidity, bid_price=None, ask_price=None):
        """Execute a batch of validated orders against bid/ask quotes."""
        return [
            self.execute_order(
                order,
                price,
                liquidity,
                bid_price=bid_price,
                ask_price=ask_price
            )
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