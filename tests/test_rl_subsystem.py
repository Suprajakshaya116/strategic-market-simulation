import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import numpy as np
import torch

from environment.config import MarketConfig
from environment.market_env import MarketEnvironment
from rl.config import PPOConfig, MARLConfig, StateEncoderConfig
from rl.state_encoder import StateEncoder, RunningMeanStd
from rl.action_space import DiscreteActionSpace
from rl.policy import PolicyNetwork
from rl.value_network import ValueNetwork
from rl.critic import CentralizedCritic
from rl.rollout_buffer import RolloutBuffer
from rl.ppo_loss import compute_ppo_loss
from rl.ppo_agent import PPOAgent
from rl.marl import MAPPOAgent
from rl.evaluator import Evaluator
from rl.checkpoint import CheckpointManager
from agents.rl_trader import RLTraderAgent


def test_state_encoder():
    raw_state = {
        "price": 105.0,
        "return": 0.05,
        "volume": 200.0,
        "spread": 0.04,
        "liquidity": 0.9,
        "volatility": 0.02,
        "order_imbalance": 0.5,
        "fundamental_value": 100.0,
        "inventory": 25,
        "cash": 95000.0
    }
    encoder = StateEncoder(strategic_dim=4)
    strat_dict = {"leader_action": "BUY", "opponent_probs": [0.2, 0.7, 0.1]}

    encoded = encoder.encode(raw_state, strategic_dict=strat_dict, update_stats=True)

    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (14,)
    assert not np.isnan(encoded).any()
    print("PASS: test_state_encoder")


def test_action_space():
    space = DiscreteActionSpace()
    assert space.to_env_action(0) == "HOLD"
    assert space.to_env_action(1) == "BUY"
    assert space.to_env_action(2) == "SELL"
    assert space.to_env_action(1, quantity=50) == ("BUY", 50)

    assert space.to_action_index("BUY") == 1
    assert space.to_action_index(("SELL", 20)) == 2
    print("PASS: test_action_space")


def test_policy_network():
    policy = PolicyNetwork(state_size=10, action_size=3, hidden_dim=32)
    state_tensor = torch.randn(4, 10)

    dist = policy(state_tensor)
    action, log_prob, entropy = policy.act(state_tensor)

    assert action.shape == (4,)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)

    details = policy.get_action_details(state_tensor)
    assert "logits" in details
    assert "action_probs" in details
    assert details["action_probs"].shape == (4, 3)
    print("PASS: test_policy_network")


def test_rollout_buffer_and_gae():
    buffer = RolloutBuffer(rollout_length=10, state_dim=5)
    for i in range(10):
        buffer.add(
            state=np.random.randn(5),
            action=np.random.randint(0, 3),
            log_prob=-0.5,
            reward=1.0,
            value=0.8,
            done=(i == 9)
        )

    buffer.compute_returns_and_advantages(last_value=0.0, last_done=True)

    assert len(buffer.advantages) == 10
    assert len(buffer.returns) == 10

    batches = list(buffer.get_batches(batch_size=4))
    assert len(batches) == 3
    assert "states" in batches[0]
    print("PASS: test_rollout_buffer_and_gae")


def test_ppo_agent_step_and_checkpoint():
    config = PPOConfig(rollout_length=16, batch_size=8, ppo_epochs=2)
    agent = PPOAgent(config=config, strategic_dim=0)

    raw_state = {
        "price": 100.0, "return": 0.0, "volume": 0.0, "spread": 0.02,
        "liquidity": 1.0, "volatility": 0.01, "order_imbalance": 0.0,
        "fundamental_value": 100.0, "inventory": 0, "cash": 100000.0
    }

    action, log_prob, val = agent.select_action(raw_state)
    assert action in (0, 1, 2)

    for _ in range(16):
        agent.store_transition(raw_state, action, log_prob, 1.0, val, False)

    metrics = agent.update(raw_state, False)
    assert "policy_loss" in metrics

    test_dir = "test_checkpoints"
    ckpt_path = agent.save_checkpoint(os.path.join(test_dir, "test.pt"))
    assert os.path.exists(ckpt_path)

    agent.load_checkpoint(ckpt_path)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    print("PASS: test_ppo_agent_step_and_checkpoint")


def test_rl_trader_agent_and_market_env():
    market_config = MarketConfig(episode_length=5, seed=42)
    env = MarketEnvironment(config=market_config)

    ppo_agent = PPOAgent()
    rl_trader = RLTraderAgent(ppo_agent=ppo_agent)

    states = env.reset()
    actions = {
        "momentum": "HOLD",
        "value": "HOLD",
        "rl_trader": rl_trader.act(states["rl_trader"])
    }

    next_states, rewards, done, info = env.step(actions)
    assert "rl_trader" in next_states
    assert "rl_trader" in rewards
    print("PASS: test_rl_trader_agent_and_market_env")


def test_mappo_agent():
    env = MarketEnvironment()
    marl_config = MARLConfig(agent_ids=env.agent_ids, ppo_config=PPOConfig(rollout_length=16))
    mappo = MAPPOAgent(marl_config=marl_config)

    states = env.reset()
    actions, log_probs, global_val = mappo.select_actions(states)

    assert len(actions) == 3
    assert isinstance(global_val, float)
    print("PASS: test_mappo_agent")


if __name__ == "__main__":
    print("\n===== RUNNING MEMBER-3 RL SUBSYSTEM UNIT TESTS =====\n")
    test_state_encoder()
    test_action_space()
    test_policy_network()
    test_rollout_buffer_and_gae()
    test_ppo_agent_step_and_checkpoint()
    test_rl_trader_agent_and_market_env()
    test_mappo_agent()
    print("\n===== ALL MEMBER-3 RL SUBSYSTEM TESTS PASSED! =====\n")
