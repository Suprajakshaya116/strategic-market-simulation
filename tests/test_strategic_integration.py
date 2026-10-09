from environment.config import MarketConfig
from environment.market_env import MarketEnvironment
from agents.momentum_agent import MomentumAgent
from agents.value_agent import ValueAgent
from agents.market_maker import MarketMaker
from game_theory import (
    StrategicLayer,
    GameTheoryConfig,
    StrategicObservation,
    LeaderAction,
)


def test_strategic_layer_pipeline():
    env_config = MarketConfig(episode_length=5, seed=42)
    env = MarketEnvironment(config=env_config)

    strategic_layer = StrategicLayer(GameTheoryConfig(seed=42))

    states = env.reset()
    market_state = env.get_market_state()

    # 1. Leader Decision
    decision = strategic_layer.solve_leader(market_state)
    assert decision.selected_action in LeaderAction

    # 2. Opponent Model Update
    strategic_layer.update_opponent("momentum", market_state, "BUY")
    strategic_layer.update_opponent("value", market_state, "HOLD")

    # 3. Strategic Observation Construction
    obs = strategic_layer.build_observation(market_state, decision)
    assert isinstance(obs, StrategicObservation)

    vec = obs.to_feature_vector()
    assert len(vec) == 11
    assert all(isinstance(x, float) for x in vec)


def test_market_environment_integration_with_leader():
    env_config = MarketConfig(episode_length=5, seed=42)
    env = MarketEnvironment(config=env_config)

    mm_agent = MarketMaker()
    momentum_agent = MomentumAgent()
    value_agent = ValueAgent()

    states = env.reset()

    for step in range(5):
        market_state = env.get_market_state()

        # Leader acts first (Stackelberg)
        leader_decision = mm_agent.act(market_state)

        # Followers observe state and act
        mom_act = momentum_agent.act(states["momentum"])
        val_act = value_agent.act(states["value"])

        actions = {
            "momentum": mom_act,
            "value": val_act,
            "rl_trader": "HOLD",
        }

        # Step environment with optional market_control from Leader
        states, rewards, done, info = env.step(actions, market_control=leader_decision)

        # Update opponent model
        mm_agent.update_opponent("momentum", market_state, mom_act)
        mm_agent.update_opponent("value", market_state, val_act)

        # Build strategic observation for Member 3's RL policy
        strat_obs = mm_agent.get_strategic_observation(market_state, leader_decision)
        feat_vec = strat_obs.to_feature_vector()

        assert info["market_control"] is not None
        assert info["market_control"]["leader_action"] in ("TIGHT", "MEDIUM", "WIDE")
        assert len(feat_vec) == 11

        if done:
            break


def test_ablation_study_configurations():
    env_config = MarketConfig(episode_length=3, seed=42)
    env = MarketEnvironment(config=env_config)
    state = env.get_market_state()

    # BASELINE 1: No strategic layer
    layer_b1 = StrategicLayer(use_stackelberg=False, use_opponent_model=False)
    obs_b1 = layer_b1.build_observation(state)

    # BASELINE 2: Stackelberg only
    layer_b2 = StrategicLayer(use_stackelberg=True, use_opponent_model=False)
    obs_b2 = layer_b2.build_observation(state)

    # BASELINE 3: Opponent modeling only
    layer_b3 = StrategicLayer(use_stackelberg=False, use_opponent_model=True)
    obs_b3 = layer_b3.build_observation(state)

    # PROPOSED: Stackelberg + Opponent modeling
    layer_prop = StrategicLayer(use_stackelberg=True, use_opponent_model=True)
    obs_prop = layer_prop.build_observation(state)

    for obs in (obs_b1, obs_b2, obs_b3, obs_prop):
        assert len(obs.to_feature_vector()) == 11
