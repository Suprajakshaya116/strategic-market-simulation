class ValueAgent:
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def act(self, state):
        price = float(state["price"])
        fundamental = float(state["fundamental_value"])

        if fundamental <= 0:
            return "HOLD"

        mispricing = (fundamental - price) / fundamental

        if mispricing > self.threshold:
            return "BUY"

        if mispricing < -self.threshold:
            return "SELL"

        return "HOLD"