from environment.market_env import MarketEnvironment


env = MarketEnvironment()

# Start with the normal initial state.
env.reset()

print("===== INITIAL PORTFOLIO =====")
print(env.portfolios["momentum"].snapshot(env.price))

# First BUY 100 so that Momentum has inventory to sell.
print("\n===== BUY 100 =====")

_, _, _, buy_info = env.step({
    "momentum": ("BUY", 100),
    "value": "HOLD",
    "rl_trader": "HOLD",
})

print("Order:", buy_info["orders"][0])
print("Buy volume:", buy_info["buy_volume"])
print("Sell volume:", buy_info["sell_volume"])
print("Inventory:", env.portfolios["momentum"].inventory)
print("Cash:", env.portfolios["momentum"].cash)


# Now SELL 100.
print("\n===== SELL 100 =====")

_, _, _, sell_info = env.step({
    "momentum": ("SELL", 100),
    "value": "HOLD",
    "rl_trader": "HOLD",
})

print("Order:", sell_info["orders"][0])
print("Buy volume:", sell_info["buy_volume"])
print("Sell volume:", sell_info["sell_volume"])
print("Inventory:", env.portfolios["momentum"].inventory)
print("Cash:", env.portfolios["momentum"].cash)


# Try another SELL with zero inventory.
print("\n===== INVALID SELL 100 =====")

_, _, _, rejected_info = env.step({
    "momentum": ("SELL", 100),
    "value": "HOLD",
    "rl_trader": "HOLD",
})

print("Order:", rejected_info["orders"][0])
print("Buy volume:", rejected_info["buy_volume"])
print("Sell volume:", rejected_info["sell_volume"])
print("Inventory:", env.portfolios["momentum"].inventory)
print("Cash:", env.portfolios["momentum"].cash)