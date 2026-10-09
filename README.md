# Game-Theoretic Multi-Agent Reinforcement Learning for Strategic Trading

A comprehensive research framework integrating **Artificial Financial Market Simulation**, **Stackelberg Game Theory**, **Statistical Opponent Modeling**, and **Deep Reinforcement Learning (PPO/MAPPO)** for strategic algorithmic trading.

---

## 1. System Architecture

The project couples three distinct subsystems into a unified sequential decision loop:

1. **Artificial Financial Market Engine (Member 1)**
   - Microstructure simulation with order execution, bid/ask quotes, price dynamics, and liquidity depletion.
   - Cash and portfolio mark-to-market accounting.
   - Heuristic baseline trading agents: `MomentumAgent` and `ValueAgent`.

2. **Game-Theoretic Strategic Layer (Member 2)**
   - Two-stage Stackelberg game with a strategic market maker leader selecting market-control actions (`TIGHT`, `MEDIUM`, `WIDE`).
   - Statistical opponent modeling with Laplace smoothing over discretized market states.
   - Quantal Response Equilibrium (QRE) integration combining empirical opponent action probabilities with follower candidate utilities.
   - 11-dimensional `StrategicObservation` feature vector generation.

3. **Reinforcement Learning Subsystem (Member 3)**
   - Production-grade Proximal Policy Optimization (PPO) agent consuming a unified 21-dimensional state space (10 market features + 11 strategic features).
   - Generalized Advantage Estimation (GAE-$\lambda$) with normalized advantages.
   - Evaluation harness ensuring 100% training/evaluation parity and frozen opponent model evaluation.
   - Strict checkpoint serialization and state dimension validation.

---

## 2. Quickstart & Installation

Clone the repository and install dependencies in editable mode:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install in editable mode with development dependencies
pip install -e .
```

---

## 3. Verified Execution Commands

### 3.1 Run Full Test Suite
Executes all 44 unit, integration, and regression tests:
```bash
pytest tests/
```

### 3.2 Run End-to-End Pipeline Demonstration
Executes a live 5-step simulation exercising all three members simultaneously (market engine, leader decisions, opponent updates, RL action selection, and a genuine PPO optimization update):
```bash
python examples/end_to_end_example.py
```

### 3.3 Run Controlled Multi-Seed Experiments
Runs matched, fair ablation experiments across 4 variants (`rl_only`, `stackelberg_only`, `opponent_model_only`, `proposed_full`) across 3 seeds, saving JSON and CSV metrics:
```bash
python experiments/run_controlled_experiments.py --seeds 42 100 2026 --timesteps 1000 --eval_episodes 5
```

### 3.4 Launch Interactive Research Terminal & Live Frontend
Start the FastAPI WebSocket backend and React Vite frontend:
```bash
# Terminal 1: Launch FastAPI simulation server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Launch Vite React development terminal
cd frontend
npm run dev
```
Navigate to `http://localhost:5173` to access the live dashboard.

---

## 4. Empirical Findings Summary

Controlled multi-seed experiments demonstrate:
- **Spread Regulation**: In uncontrolled baselines (`rl_only`), order imbalance triggers catastrophic spread divergence ($> 1000.0$). The proposed Stackelberg strategic leader regulates spreads to $2.10 \pm 1.20$.
- **Downside Protection**: Strategic market-control prevents severe adverse selection and portfolio account collapses (e.g., preventing the $-84.77\%$ drawdown observed in `rl_only` on Seed 42).
- **Full Parity**: Training and evaluation utilize identical state encoders, strategic features, and execution timing.

For complete mathematical definitions, empirical tables, and audit details, refer to:
- [`INTEGRATION_AUDIT.md`](./INTEGRATION_AUDIT.md)
- [`RESEARCH_VALIDATION.md`](./RESEARCH_VALIDATION.md)
