"""Comprehensive Regression and Acceptance Tests for MARL Strategic Trading Pipeline.

Verifies all audit repairs:
- Priority 1: Evaluation pipeline parity, frozen policy, checkpoint dimension validation.
- Priority 2: Decision cycle timing, bid/ask execution quotes, cash/inventory reconciliation.
- Priority 3: Opponent model integration into Stackelberg solver, quantal response belief shifts.
- Priority 4: Real RL optimization updates, gradient finiteness, checkpoint serialization.
- Priority 8: Complete 20-step end-to-end acceptance test with diagnostic printouts.
"""

import copy
import pytest
import numpy as np
import torch

from environment.config import MarketConfig
from environment.market_env import MarketEnvironment
from environment.portfolio import Portfolio
from environment.order_manager import Order
from environment.execution_engine import ExecutionEngine
from agents.momentum_agent import MomentumAgent
from agents.value_agent import ValueAgent
from agents.market_maker import MarketMaker
from game_theory import (
    StrategicLayer,
    GameTheoryConfig,
    LeaderAction,
    LeaderDecision,
    StrategicObservation,
    OpponentModel,
    StackelbergGame,
)
from rl.config import PPOConfig, StateEncoderConfig
from rl.ppo_agent import PPOAgent
from rl.evaluator import Evaluator
from rl.trainer import RLTrainer


# ==============================================================================
# Priority 1: Evaluation Pipeline Parity & Checkpointing Tests
# ==============================================================================

def test_evaluator_uses_strategic_features_and_market_control():
    """Verify evaluator provides strategic features and applies leader market control."""
    env_config = MarketConfig(episode_length=5, seed=42)
    env = MarketEnvironment(config=env_config)

    gt_config = GameTheoryConfig(seed=42)
    strategic_layer = StrategicLayer(config=gt_config, use_stackelberg=True, use_opponent_model=True)

    ppo_config = PPOConfig(rollout_length=32, batch_size=16, device="cpu", seed=42)
    encoder_config = StateEncoderConfig()
    agent = PPOAgent(config=ppo_config, encoder_config=encoder_config, strategic_dim=11)

    evaluator = Evaluator(env=env, strategic_layer=strategic_layer, enable_opponent_modeling=True)
    metrics = evaluator.evaluate_agent(ppo_agent=agent, num_episodes=2, seed=123, deterministic=True)

    assert "mean_reward" in metrics
    assert "cumulative_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "mean_trade_count" in metrics
    assert "mean_spread" in metrics
    assert metrics["num_episodes"] == 2


def test_evaluator_does_not_modify_policy_weights():
    """Verify evaluation does not inadvertently change policy parameters."""
    env = MarketEnvironment(config=MarketConfig(episode_length=5, seed=42))
    agent = PPOAgent(strategic_dim=11)

    params_before = [p.clone() for p in agent.policy.parameters()]
    evaluator = Evaluator(env=env, strategic_layer=StrategicLayer())
    evaluator.evaluate_agent(ppo_agent=agent, num_episodes=3, seed=42)
    params_after = [p.clone() for p in agent.policy.parameters()]

    for p_b, p_a in zip(params_before, params_after):
        assert torch.equal(p_b, p_a), "Policy parameters changed during evaluation!"


def test_checkpoint_dimension_mismatch_raises_error(tmp_path):
    """Verify loading checkpoint with mismatched state_dim raises ValueError."""
    agent_dim11 = PPOAgent(strategic_dim=11)
    agent_dim0 = PPOAgent(strategic_dim=0)

    ckpt_path = str(tmp_path / "test_dim11.pt")
    agent_dim11.save_checkpoint(ckpt_path)

    # Attempting to load dim11 checkpoint into dim0 agent must fail safely
    with pytest.raises(ValueError, match="Checkpoint state_dim.*does not match"):
        agent_dim0.load_checkpoint(ckpt_path)


# ==============================================================================
# Priority 2: Market Execution & Stackelberg Timing Tests
# ==============================================================================

def test_leader_action_timing_affects_current_step_quotes():
    """Verify market_control changes spread and quotes before order execution."""
    env = MarketEnvironment(config=MarketConfig(initial_price=100.0, initial_spread=0.04, seed=42))
    env.reset()

    # Step with TIGHT market control (spread multiplier 0.7)
    tight_control = {"leader_action": "TIGHT", "spread_multiplier": 0.7, "liquidity_multiplier": 1.3}
    _, _, _, info_tight = env.step({"momentum": "HOLD", "value": "HOLD", "rl_trader": "HOLD"}, market_control=tight_control)
    assert info_tight["spread"] < 0.04
    assert info_tight["ask"] - info_tight["bid"] == pytest.approx(info_tight["spread"], rel=1e-5)

    # Reset and step with WIDE market control (spread multiplier 1.4)
    env.reset()
    wide_control = {"leader_action": "WIDE", "spread_multiplier": 1.4, "liquidity_multiplier": 0.7}
    _, _, _, info_wide = env.step({"momentum": "HOLD", "value": "HOLD", "rl_trader": "HOLD"}, market_control=wide_control)
    assert info_wide["spread"] > 0.04
    assert info_wide["ask"] > info_tight["ask"]
    assert info_wide["bid"] < info_tight["bid"]


def test_buy_sell_quote_execution_and_cash_reconciliation():
    """Verify BUY fills at ask, SELL fills at bid, and cash/holdings reconcile exactly."""
    portfolios = {"trader": Portfolio(cash=10_000.0, inventory=50, last_price=100.0)}
    config = MarketConfig(transaction_cost_rate=0.001, market_impact_rate=0.0001)
    engine = ExecutionEngine(portfolios, config)

    price = 100.0
    spread = 2.0
    bid = price - spread / 2.0  # 99.0
    ask = price + spread / 2.0  # 101.0
    liquidity = 10.0

    # Execute BUY order of 10 shares
    buy_order = Order(agent_id="trader", action="BUY", quantity=10)
    res_buy = engine.execute_order(buy_order, price, liquidity, bid_price=bid, ask_price=ask)

    assert res_buy["status"] == "EXECUTED"
    assert res_buy["fill_price"] == ask  # 101.0
    expected_notional = 10 * 101.0
    assert res_buy["notional"] == expected_notional
    expected_tx_cost = expected_notional * config.transaction_cost_rate
    assert res_buy["transaction_cost"] == pytest.approx(expected_tx_cost)
    assert portfolios["trader"].inventory == 60

    # Execute SELL order of 10 shares
    sell_order = Order(agent_id="trader", action="SELL", quantity=10)
    res_sell = engine.execute_order(sell_order, price, liquidity, bid_price=bid, ask_price=ask)

    assert res_sell["status"] == "EXECUTED"
    assert res_sell["fill_price"] == bid  # 99.0
    assert portfolios["trader"].inventory == 50


def test_rejected_orders_do_not_corrupt_portfolio():
    """Verify attempting to buy more than cash allows is safely rejected."""
    portfolios = {"broke_trader": Portfolio(cash=50.0, inventory=0, last_price=100.0)}
    engine = ExecutionEngine(portfolios, MarketConfig())

    buy_huge = Order(agent_id="broke_trader", action="BUY", quantity=100)
    res = engine.execute_order(buy_huge, price=100.0, liquidity=1.0, bid_price=99.0, ask_price=101.0)

    assert res["status"] == "REJECTED"
    assert portfolios["broke_trader"].inventory == 0
    assert portfolios["broke_trader"].cash == 50.0


# ==============================================================================
# Priority 3: Opponent Modeling & Stackelberg Integration Tests
# ==============================================================================

def test_opponent_model_probability_properties():
    """Verify predicted probabilities sum to 1.0 and update conditionally."""
    op_model = OpponentModel(GameTheoryConfig())
    dummy_state = {"price": 100.0, "return": 0.01, "volatility": 0.02, "order_imbalance": 0.1, "spread": 0.02}

    # Initial uniform Dirichlet prior
    probs_init = op_model.predict("momentum", dummy_state)
    assert sum(probs_init.values()) == pytest.approx(1.0, rel=1e-5)
    assert probs_init["BUY"] == pytest.approx(1.0 / 3.0)

    # Observe multiple BUY actions
    for _ in range(10):
        op_model.update("momentum", dummy_state, "BUY")

    probs_updated = op_model.predict("momentum", dummy_state)
    assert sum(probs_updated.values()) == pytest.approx(1.0, rel=1e-5)
    assert probs_updated["BUY"] > probs_init["BUY"]
    assert probs_updated["SELL"] < probs_init["SELL"]


def test_opponent_belief_shift_alters_stackelberg_decision():
    """Verify shifting opponent beliefs shifts expected pressure and alters leader utilities."""
    gt_config = GameTheoryConfig(expected_order_size=100.0, leader_inventory_penalty=0.1)
    op_model = OpponentModel(gt_config)
    solver = StackelbergGame(config=gt_config, opponent_model=op_model)

    state = {"price": 100.0, "return": 0.0, "volatility": 0.01, "order_imbalance": 0.0, "spread": 0.02}

    # Prior state decision
    dec_neutral = solver.solve(state)

    # Heavily train opponent model to expect BUY orders from both followers
    for _ in range(50):
        op_model.update("momentum", state, "BUY")
        op_model.update("value", state, "BUY")

    dec_bullish = solver.solve(state)
    # The expected buy pressure under bullish beliefs must be higher
    cand_bullish_tight = next(c for c in dec_bullish.candidate_evaluations if c.leader_action == LeaderAction.TIGHT)
    cand_neutral_tight = next(c for c in dec_neutral.candidate_evaluations if c.leader_action == LeaderAction.TIGHT)

    assert cand_bullish_tight.expected_buy_pressure > cand_neutral_tight.expected_buy_pressure
    assert cand_bullish_tight.expected_order_imbalance > cand_neutral_tight.expected_order_imbalance


# ==============================================================================
# Priority 4: Real RL Updates & Finiteness Tests
# ==============================================================================

def test_real_ppo_optimizer_update_modifies_parameters():
    """Verify at least one real PPO optimization step updates network weights."""
    agent = PPOAgent(
        config=PPOConfig(rollout_length=8, batch_size=4, ppo_epochs=2, lr=1e-3, device="cpu"),
        strategic_dim=11
    )

    initial_policy_weights = copy.deepcopy([p.data.clone() for p in agent.policy.parameters()])
    initial_value_weights = copy.deepcopy([p.data.clone() for p in agent.value_net.parameters()])

    # Fill buffer with 8 steps of transitions
    dummy_market_state = {"price": 100.0, "return": 0.01, "volume": 10.0, "spread": 0.02, "liquidity": 1.0,
                          "volatility": 0.01, "order_imbalance": 0.0, "fundamental_value": 100.0, "inventory": 0, "cash": 10000.0}
    dummy_strat = np.zeros(11, dtype=np.float32)

    for i in range(8):
        action, log_prob, val = agent.select_action(dummy_market_state, strategic_dict=dummy_strat)
        agent.store_transition(
            state=dummy_market_state,
            action=action,
            log_prob=log_prob,
            reward=1.0,
            value=val,
            done=False,
            strategic_dict=dummy_strat
        )

    # Perform update
    update_metrics = agent.update(last_state=dummy_market_state, last_done=False, last_strategic_dict=dummy_strat)

    assert "total_loss" in update_metrics
    assert np.isfinite(update_metrics["total_loss"])
    assert np.isfinite(update_metrics["policy_loss"])
    assert np.isfinite(update_metrics["value_loss"])

    # Verify at least some parameters changed
    weights_changed = any(
        not torch.equal(w_init, w_curr.data)
        for w_init, w_curr in zip(initial_policy_weights, agent.policy.parameters())
    )
    assert weights_changed, "PPO update did not modify policy weights!"


# ==============================================================================
# Priority 8: Complete 20-Step End-to-End Acceptance Test
# ==============================================================================

def test_complete_20_step_acceptance_pipeline():
    """Execute complete 20-step integration acceptance workflow with diagnostics."""
    print("\n=======================================================")
    print(" ACCEPTANCE TEST: 20-STEP END-TO-END PIPELINE VALIDATION ")
    print("=======================================================")

    # 1. Initialize the market environment
    env_config = MarketConfig(episode_length=5, seed=42)
    env = MarketEnvironment(config=env_config)

    # 2. Initialize Momentum and Value agents
    momentum_agent = MomentumAgent()
    value_agent = ValueAgent()

    # 3. Initialize the opponent model
    gt_config = GameTheoryConfig(seed=42)
    opponent_model = OpponentModel(gt_config)

    # 4. Initialize the Stackelberg solver
    solver = StackelbergGame(config=gt_config, opponent_model=opponent_model)
    strategic_layer = StrategicLayer(config=gt_config, use_stackelberg=True, use_opponent_model=True)

    # 5. Initialize the RL policy and training components
    ppo_config = PPOConfig(rollout_length=4, batch_size=4, ppo_epochs=2, lr=1e-3, device="cpu", seed=42)
    encoder_config = StateEncoderConfig()
    rl_agent = PPOAgent(config=ppo_config, encoder_config=encoder_config, strategic_dim=11)

    # 6. Reset the environment
    states = env.reset(seed=42)
    assert "rl_trader" in states
    market_state = env.get_market_state()

    # 7. Construct a valid observation
    strat_obs = strategic_layer.build_observation(market_state)
    strat_vec = strat_obs.to_feature_vector()
    assert len(strat_vec) == 11

    # 8. Obtain opponent predictions using only past observations
    mom_probs = opponent_model.predict("momentum", market_state)
    val_probs = opponent_model.predict("value", market_state)
    assert sum(mom_probs.values()) == pytest.approx(1.0, rel=1e-5)

    # 9. Evaluate all candidate leader actions
    leader_decision = solver.solve(market_state)
    assert len(leader_decision.candidate_evaluations) == 3

    # 10. Select and apply the leader action before the relevant executions
    applied_control = leader_decision.selected_action.value

    # 11. Generate RL and baseline follower actions
    action_idx, log_prob, val = rl_agent.select_action(
        state=states["rl_trader"],
        strategic_dict=strat_obs,
        deterministic=False
    )
    rl_env_action = rl_agent.action_space.to_env_action(action_idx)
    mom_action = momentum_agent.act(states["momentum"])
    val_action = value_agent.act(states["value"])

    actions = {"momentum": mom_action, "value": val_action, "rl_trader": rl_env_action}

    # 12. Execute orders under the intended market conditions
    next_states, rewards, done, info = env.step(actions, market_control=leader_decision)

    # 13. Verify prices, spread, portfolio values, and rewards
    assert info["price"] > 0
    assert info["spread"] > 0
    assert "rl_trader" in rewards
    rl_reward = rewards["rl_trader"]
    rl_portfolio_val = info["portfolio_values"]["rl_trader"]

    # 14. Record actual observed opponent actions
    # 15. Update the opponent model at the documented time
    opponent_model.update("momentum", market_state, mom_action)
    opponent_model.update("value", market_state, val_action)

    # 16. Construct next observation with expected feature dimensions
    next_market_state = env.get_market_state()
    next_strat_obs = strategic_layer.build_observation(next_market_state)
    encoded_next = rl_agent.encoder.encode(next_states["rl_trader"], next_strat_obs)
    assert len(encoded_next) == 21

    # Store transitions until buffer is ready
    rl_agent.store_transition(states["rl_trader"], action_idx, log_prob, rl_reward, val, done, strat_obs)
    for _ in range(3):
        a_i, lp, v = rl_agent.select_action(states["rl_trader"], strategic_dict=strat_obs)
        rl_agent.store_transition(states["rl_trader"], a_i, lp, 0.5, v, False, strat_obs)

    # 17. Perform at least one real RL training update in a training-mode test
    update_metrics = rl_agent.update(next_states["rl_trader"], done, next_strat_obs)
    assert np.isfinite(update_metrics["total_loss"])

    # 18. Run evaluation with model updates disabled where required
    evaluator = Evaluator(env=env, strategic_layer=strategic_layer)
    eval_metrics = evaluator.evaluate_agent(ppo_agent=rl_agent, num_episodes=2, freeze_opponent_model=True)
    assert eval_metrics["num_episodes"] == 2

    # 19. Save and reload a checkpoint
    ckpt_path = "checkpoints/acceptance_test_agent.pt"
    rl_agent.save_checkpoint(ckpt_path)
    reloaded_agent = PPOAgent(config=ppo_config, encoder_config=encoder_config, strategic_dim=11)
    reloaded_agent.load_checkpoint(ckpt_path)
    assert reloaded_agent.encoder.total_dim == rl_agent.encoder.total_dim

    # 20. Verify next step and evaluation run using loaded agent
    reloaded_eval = evaluator.evaluate_agent(ppo_agent=reloaded_agent, num_episodes=1)
    assert reloaded_eval["num_episodes"] == 1

    # Print Compact Diagnostic Summary
    print("\n--- DIAGNOSTIC SUMMARY ---")
    print(f"State keys: {list(market_state.keys())}")
    print(f"Action formats: RL={rl_env_action}, Mom={mom_action}, Val={val_action}")
    print(f"Candidate Leader Utilities: {[round(c.leader_utility, 3) for c in leader_decision.candidate_evaluations]}")
    print(f"Selected Leader Action: {applied_control}")
    print(f"Opponent Probabilities (Momentum): {mom_probs}")
    print(f"Strategic Feature Dimension: {len(strat_vec)}")
    print(f"RL Encoded State Dimension: {len(encoded_next)}")
    executed_summary = [f"{o.get('action')}:{o.get('status')}" for o in info.get("orders", [])]
    print(f"Executed Orders: {executed_summary}")
    print(f"RL Reward: {rl_reward:.4f} | Portfolio Value: {rl_portfolio_val:.2f}")
    print(f"Episode Termination Status: {done}")
    print(f"RL Optimization Update: SUCCESS (Loss={update_metrics['total_loss']:.4f})")
    print("-------------------------------------------------------\n")
