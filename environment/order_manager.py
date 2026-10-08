from dataclasses import dataclass


VALID_ACTIONS = {"BUY", "HOLD", "SELL"}


@dataclass(frozen=True)
class Order:
    agent_id: str
    action: str
    quantity: int


class OrderManager:
    """Normalize trader actions into validated orders."""

    def __init__(self, max_order_size):
        if max_order_size <= 0:
            raise ValueError("max_order_size must be > 0")
        self.max_order_size = int(max_order_size)

    def create_orders(self, actions):
        orders = []
        for agent_id, raw in actions.items():
            if isinstance(raw, (tuple, list)):
                if len(raw) != 2:
                    raise ValueError(
                        f"Action for {agent_id} must be 'ACTION' or ('ACTION', quantity)"
                    )
                action, quantity = raw
            else:
                action, quantity = raw, self.max_order_size

            action = str(action).upper()
            if action not in VALID_ACTIONS:
                raise ValueError(f"Invalid action {action!r} for {agent_id}")

            if isinstance(quantity, bool):
                raise ValueError(f"Invalid quantity for {agent_id}: {quantity}")
            try:
                quantity = int(quantity)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid quantity for {agent_id}: {quantity}") from exc

            if not 0 <= quantity <= self.max_order_size:
                raise ValueError(f"Invalid quantity for {agent_id}: {quantity}")
            if action == "HOLD":
                quantity = 0

            orders.append(Order(str(agent_id), action, quantity))
        return orders

    @staticmethod
    def aggregate(orders):
        return {
            "buy_volume": float(sum(o.quantity for o in orders if o.action == "BUY")),
            "sell_volume": float(sum(o.quantity for o in orders if o.action == "SELL")),
        }
