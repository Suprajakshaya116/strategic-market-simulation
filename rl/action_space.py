from typing import Union, Tuple


class DiscreteActionSpace:
    """Discrete action space mapping integer indices <-> environment actions.

    Mapping:
      0 -> "HOLD"
      1 -> "BUY"
      2 -> "SELL"
    """

    INDEX_TO_ACTION = {0: "HOLD", 1: "BUY", 2: "SELL"}
    ACTION_TO_INDEX = {"HOLD": 0, "BUY": 1, "SELL": 2}

    def __init__(self, num_actions: int = 3, max_order_size: int = 100):
        self.num_actions = num_actions
        self.max_order_size = max_order_size

    def to_env_action(
        self,
        action_idx: int,
        quantity: int | None = None
    ) -> Union[str, Tuple[str, int]]:
        """Convert integer index to environment action string or tuple."""
        action_str = self.INDEX_TO_ACTION.get(int(action_idx), "HOLD")
        if quantity is not None and action_str in ("BUY", "SELL"):
            q = max(0, min(int(quantity), self.max_order_size))
            return (action_str, q)
        return action_str

    def to_action_index(self, env_action: Union[str, Tuple[str, int]]) -> int:
        """Convert environment action to integer index."""
        if isinstance(env_action, (tuple, list)):
            action_str = env_action[0]
        else:
            action_str = env_action
        return self.ACTION_TO_INDEX.get(str(action_str).upper(), 0)
