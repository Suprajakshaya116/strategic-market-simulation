# Game-Theoretic Strategic Layer (Member 2)

## Overview

The `game_theory` package implements **Member 2 — Game-Theoretic Strategic Layer** for the strategic market simulation. It combines a **Stackelberg Game Solver** and a **Statistical Opponent Model** to provide a decision mechanism for the strategic Leader (Market Maker / Liquidity Controller) and a feature vector interface for Member 3's future MARL / RL policy agent.

---

## Architecture Diagram

```mermaid
graph TD
    subgraph Market Environment (Member 1)
        ME[MarketEnvironment] -->|Market State: price, return, spread, etc.| SG
    end

    subgraph Game-Theoretic Strategic Layer (Member 2)
        SG[StackelbergGame] -->|1. Candidate Leader Actions: TIGHT/MEDIUM/WIDE| FRM[FollowerResponseModel]
        FRM -->|2. Predict Best Responses: B_i(a_L)| Util[Utility Calculator]
        Util -->|3. Compute Leader & Follower Utilities| Decision[LeaderDecision: a_L*]
        
        OM[OpponentModel] -->|4. Frequency Probabilities: P(action|state)| SO[StrategicObservation]
        Decision --> SO
        SO -->|5. to_feature_vector()| RL
    end

    subgraph MARL Policy (Member 3 - Future)
        RL[PPO / MAPPO Agent]
    end

    Decision -->|market_control: spread & liquidity multipliers| ME
```

---

## Key Components

### 1. Stackelberg Game (`stackelberg.py`)

Formulated as a computational discrete Stackelberg game:
- **Follower Best Response**: \( B_i(a_L) = \arg\max_{a} U_i(a, a_L) \)
- **Leader Decision**: \( a_L^* = \arg\max_{a_L} U_L(a_L, B(a_L)) \)

For each candidate Leader action (`TIGHT`, `MEDIUM`, `WIDE`), the game solver:
1. Predicts expected follower responses (`BUY`, `HOLD`, `SELL`).
2. Calculates follower utilities.
3. Estimates aggregate order pressure and order imbalance.
4. Computes leader utility \( U_L \).
5. Selects the Leader action maximizing \( U_L \).

All candidate evaluations remain stored and accessible in `LeaderDecision.candidate_evaluations` for debugging, visualization, and ablation studies.

### 2. Leader Action Space (`types.py`)

Implemented via `LeaderAction` Enum:
- `TIGHT`: Lower spread multiplier (0.7x), higher liquidity provision (1.3x). Encourages trading activity.
- `MEDIUM`: Baseline market conditions (1.0x spread, 1.0x liquidity).
- `WIDE`: Higher spread multiplier (1.4x), reduced liquidity provision (0.7x). Protects market maker during high volatility.

### 3. Utility Functions (`utility.py`)

- **Leader Utility**:
  \[
  U_L = \alpha \cdot \text{liquidity} + \beta \cdot \text{trading\_activity} - \gamma \cdot \text{volatility\_risk} - \delta \cdot \text{inventory\_risk} - \epsilon \cdot \text{adverse\_flow\_risk}
  \]
  Configured via `GameTheoryConfig` (`leader_liquidity_weight`, `leader_activity_weight`, `leader_volatility_penalty`, `leader_inventory_penalty`, `leader_adverse_flow_penalty`).

- **Follower Utility**:
  \[
  U_i = w_{\text{return}} \cdot \text{expected\_return} - w_{\text{cost}} \cdot \text{transaction\_cost} - w_{\text{risk}} \cdot \text{risk\_penalty}
  \]

### 4. Opponent Model (`opponent_model.py`)

Lightweight, frequency-based statistical model with Laplace smoothing.
- Discretizes market state into compact state keys: `return_direction | volatility_regime | mispricing_direction | order_imbalance_regime`.
- Estimates \( P(\text{BUY} \mid s) \), \( P(\text{HOLD} \mid s) \), \( P(\text{SELL} \mid s) \).
- Returns uniform prior \( (1/3, 1/3, 1/3) \) for unseen state keys.
- Completely decoupled from neural networks (no PyTorch/TensorFlow dependencies).

### 5. Strategic Observation Interface (`types.py`)

Provides a 11-dimensional feature vector `to_feature_vector()` for Member 3's RL policy:
```python
[
    0: leader_action (TIGHT=0.0, MEDIUM=1.0, WIDE=2.0),
    1: leader_utility,
    2: momentum_buy_probability,
    3: momentum_hold_probability,
    4: momentum_sell_probability,
    5: value_buy_probability,
    6: value_hold_probability,
    7: value_sell_probability,
    8: expected_buy_pressure,
    9: expected_sell_pressure,
    10: expected_order_imbalance
]
```

---

## Integration Contract for Member 3

Member 3's RL / MARL policy agent can easily consume `StrategicObservation` alongside standard market state:

```python
from game_theory import StrategicLayer

strategic_layer = StrategicLayer()

# 1. Solve leader decision
leader_decision = strategic_layer.solve_leader(market_state)

# 2. Build strategic observation
strat_obs = strategic_layer.build_observation(market_state, leader_decision)

# 3. Convert to float feature vector for PyTorch/RL policy
feature_vector = strat_obs.to_feature_vector()
# Member 3 will do:
# torch_tensor = torch.tensor(feature_vector, dtype=torch.float32)
```

---

## Ablation Study Configurations (Section 27)

To support scientific research and baseline comparisons, components can be enabled/disabled independently via `StrategicLayer`:

- **Baseline 1 (No Strategic Layer)**: `StrategicLayer(use_stackelberg=False, use_opponent_model=False)`
- **Baseline 2 (Stackelberg Only)**: `StrategicLayer(use_stackelberg=True, use_opponent_model=False)`
- **Baseline 3 (Opponent Model Only)**: `StrategicLayer(use_stackelberg=False, use_opponent_model=True)`
- **Proposed (Stackelberg + Opponent Model)**: `StrategicLayer(use_stackelberg=True, use_opponent_model=True)`

---

## Verification & Testing

Run all unit and integration tests:
```bash
python -m pytest tests/
```
