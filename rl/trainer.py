import os
import numpy as np
from typing import Dict, Any

from environment.market_env import MarketEnvironment
from agents.momentum_agent import MomentumAgent
from agents.value_agent import ValueAgent

from .ppo_agent import PPOAgent
from .evaluator import Evaluator
from .metrics import MetricsTracker
from .opponent_model import EmpiricalOpponentModel, StrategicStateAdapter


class RLTrainer:
    """Trainer orchestrating PPO training loop on MarketEnvironment."""

    def __init__(
        self,
        env: MarketEnvironment,
        agent: PPOAgent,
        target_agent_id: str = "rl_trader",
        enable_opponent_modeling: bool = True,
        strategic_layer: Any = None
    ):
        self.env = env
        self.agent = agent
        self.target_agent_id = target_agent_id

        self.opponent_ids = [aid for aid in self.env.agent_ids if aid != target_agent_id]
        self.opponents = {
            "momentum": MomentumAgent(),
            "value": ValueAgent()
        }

        self.strategic_layer = strategic_layer
        self.enable_opponent_modeling = enable_opponent_modeling
        if strategic_layer is not None:
            self.opponent_model = getattr(strategic_layer, "opponent_model", None)
            self.strategic_adapter = None
        elif enable_opponent_modeling:
            self.opponent_model = EmpiricalOpponentModel(agent_ids=self.env.agent_ids)
            self.strategic_adapter = StrategicStateAdapter(opponent_model=self.opponent_model)
        else:
            self.opponent_model = None
            self.strategic_adapter = StrategicStateAdapter(opponent_model=None)

        self.evaluator = Evaluator(env=self.env)
        self.metrics_tracker = MetricsTracker()

    def train(
        self,
        total_timesteps: int = 20_000,
        eval_interval: int = 5_000,
        checkpoint_interval: int = 10_000,
        checkpoint_dir: str = "checkpoints"
    ) -> Dict[str, Any]:
        """Run complete single-agent PPO training loop."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        step = 0
        episode_count = 0

        states = self.env.reset()
        for opp in self.opponents.values():
            if hasattr(opp, "reset"):
                opp.reset()
        if self.strategic_layer is not None and hasattr(self.strategic_layer, "reset"):
            self.strategic_layer.reset()

        ep_reward = 0.0
        ep_len = 0

        while step < total_timesteps:
            market_state = self.env.get_market_state()
            leader_decision = None

            if self.strategic_layer is not None:
                leader_decision = self.strategic_layer.solve_leader(market_state)
                strat_dict = self.strategic_layer.build_observation(market_state, leader_decision)
            elif self.enable_opponent_modeling:
                strat_dict = self.strategic_adapter.build_strategic_dict()
            else:
                strat_dict = None

            # 1. Action selection
            action_idx, log_prob, val = self.agent.select_action(
                state=states[self.target_agent_id],
                strategic_dict=strat_dict,
                update_encoder_stats=True,
                deterministic=False
            )
            env_rl_action = self.agent.action_space.to_env_action(action_idx)

            actions = {}
            for aid in self.env.agent_ids:
                if aid == self.target_agent_id:
                    actions[aid] = env_rl_action
                elif aid in self.opponents:
                    actions[aid] = self.opponents[aid].act(states[aid])
                    if self.strategic_layer is not None:
                        self.strategic_layer.update_opponent(aid, market_state, actions[aid])
                    elif self.enable_opponent_modeling:
                        self.opponent_model.update(aid, actions[aid], states[aid])
                else:
                    actions[aid] = "HOLD"

            # 2. Environment step with optional leader market_control
            next_states, rewards, done, info = self.env.step(actions, market_control=leader_decision)
            reward = rewards[self.target_agent_id]

            # 3. Store transition
            self.agent.store_transition(
                state=states[self.target_agent_id],
                action=action_idx,
                log_prob=log_prob,
                reward=reward,
                value=val,
                done=done,
                strategic_dict=strat_dict
            )

            ep_reward += reward
            ep_len += 1
            step += 1

            # 4. Trigger PPO Update when rollout buffer is full
            if len(self.agent.buffer.states) >= self.agent.config.rollout_length:
                next_m_state = self.env.get_market_state()
                if self.strategic_layer is not None:
                    next_ld = self.strategic_layer.solve_leader(next_m_state)
                    next_strat = self.strategic_layer.build_observation(next_m_state, next_ld)
                elif self.enable_opponent_modeling:
                    next_strat = self.strategic_adapter.build_strategic_dict()
                else:
                    next_strat = None

                update_metrics = self.agent.update(
                    last_state=next_states[self.target_agent_id],
                    last_done=done,
                    last_strategic_dict=next_strat
                )
                self.metrics_tracker.log_update(update_metrics)

            if done:
                episode_count += 1
                pv = info["portfolio_values"][self.target_agent_id]
                self.metrics_tracker.log_episode(ep_reward, ep_len, final_portfolio_value=pv)

                states = self.env.reset()
                for opp in self.opponents.values():
                    if hasattr(opp, "reset"):
                        opp.reset()
                ep_reward = 0.0
                ep_len = 0
            else:
                states = next_states

            # 5. Periodic Evaluation
            if step % eval_interval == 0:
                eval_stats = self.evaluator.evaluate_agent(
                    ppo_agent=self.agent,
                    num_episodes=5,
                    target_agent_id=self.target_agent_id,
                    seed=200 + step
                )
                print(
                    f"[Step {step}/{total_timesteps}] "
                    f"Eval Reward: {eval_stats['mean_reward']:.2f} | "
                    f"Sharpe: {eval_stats['sharpe_ratio']:.2f} | "
                    f"Win Rate: {eval_stats['win_rate']*100:.1f}% | "
                    f"Max DD: {eval_stats['max_drawdown']*100:.1f}%"
                )

            # 6. Periodic Checkpointing
            if step % checkpoint_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"ppo_agent_step_{step}.pt")
                self.agent.save_checkpoint(ckpt_path)

        # Final evaluation
        final_eval = self.evaluator.evaluate_agent(
            ppo_agent=self.agent,
            num_episodes=10,
            target_agent_id=self.target_agent_id,
            seed=999
        )
        final_ckpt = os.path.join(checkpoint_dir, "ppo_agent_final.pt")
        self.agent.save_checkpoint(final_ckpt)

        return {
            "total_timesteps": step,
            "episodes": episode_count,
            "final_eval": final_eval,
            "metrics": self.metrics_tracker.get_summary()
        }
