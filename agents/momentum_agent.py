from collections import deque


class MomentumAgent:
    def __init__(self, lookback=3):
        self.prices = deque(maxlen=lookback + 1)

    def reset(self):
        self.prices.clear()

    def act(self, state):
        self.prices.append(float(state["price"]))

        if len(self.prices) < 2:
            return "HOLD"

        move = self.prices[-1] - self.prices[0]

        if move > 0:
            return "BUY"

        if move < 0:
            return "SELL"

        return "HOLD"