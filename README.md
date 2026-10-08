# Game-Theoretic Multi-Agent Reinforcement Learning for Strategic Trading

Repository structure follows the project specification.

## Member 1 status

Implemented only the Member 1 market-simulation scope:

- `environment/market_env.py` — market environment and integration boundary
- `environment/price_model.py` — price dynamics, order imbalance and fundamental-value term
- `environment/order_manager.py` — BUY/HOLD/SELL order normalization and aggregation
- `environment/liquidity.py` — liquidity and spread dynamics
- `environment/portfolio.py` — cash, inventory and mark-to-market accounting
- `environment/config.py` — market configuration and validation
- `agents/momentum_agent.py` — rule-based momentum baseline
- `agents/value_agent.py` — rule-based value baseline

The final repository structure also contains placeholder files for the other team members' modules, but those modules are intentionally not implemented by Member 1.

## Integration contract

Environment state uses the project's Point 26 interface:

```python
{
    "price": ...,
    "return": ...,
    "volume": ...,
    "spread": ...,
    "liquidity": ...,
    "volatility": ...,
    "order_imbalance": ...,
    "inventory": ...,
    "cash": ...,
    "fundamental_value": ...,
}
```

Actions use:

```python
{
    "momentum": "BUY",
    "value": "SELL",
    "rl_trader": "HOLD",
}
```

Position-sized actions are also supported:

```python
{
    "momentum": ("BUY", 25),
    "value": ("SELL", 10),
    "rl_trader": ("HOLD", 0),
}
```

The environment returns:

```python
next_state, reward, done, info = env.step(actions)
```

`info` additionally exposes bid/ask quotes, executed-order costs, market metrics and portfolio values without changing the stable Point 26 state schema.

## Member 1 scope intentionally not implemented

- Stackelberg/game-theory policies
- opponent modeling
- PPO/MAPPO/RL training
- multi-agent learning
- experiment runners
- final analysis/dashboard
