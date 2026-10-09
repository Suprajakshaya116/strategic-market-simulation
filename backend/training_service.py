import os
import glob
import threading
import time
from typing import Dict, Any, List, Optional

from environment.config import MarketConfig
from environment.market_env import MarketEnvironment
from game_theory import StrategicLayer, GameTheoryConfig
from rl.config import PPOConfig, StateEncoderConfig
from rl.ppo_agent import PPOAgent
from rl.trainer import RLTrainer


class TrainingService:
    """Manages asynchronous RL training jobs and training telemetry."""

    def __init__(self):
        self.is_training = False
        self.current_step = 0
        self.total_timesteps = 1000
        self.variant = "proposed_full"
        self.seed = 42
        self.recent_metrics: Dict[str, Any] = {}
        self.training_history: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_training": self.is_training,
            "current_step": self.current_step,
            "total_timesteps": self.total_timesteps,
            "variant": self.variant,
            "seed": self.seed,
            "metrics": self.recent_metrics,
            "history": self.training_history[-50:]  # last 50 data points
        }

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        ckpts = []
        for path in glob.glob("checkpoints/**/*.pt", recursive=True):
            stat = os.stat(path)
            ckpts.append({
                "path": path.replace("\\", "/"),
                "filename": os.path.basename(path),
                "size_kb": round(stat.st_size / 1024, 2),
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
            })
        return sorted(ckpts, key=lambda x: x["modified"], reverse=True)

    def start_training(self, timesteps: int = 1000, variant: str = "proposed_full", seed: int = 42):
        if self.is_training:
            return {"status": "error", "message": "Training already in progress."}

        self.is_training = True
        self.current_step = 0
        self.total_timesteps = timesteps
        self.variant = variant
        self.seed = seed
        self.training_history.clear()
        self._stop_requested = False

        self._thread = threading.Thread(target=self._run_training_job, daemon=True)
        self._thread.start()
        return {"status": "success", "message": f"Training started for {timesteps} steps."}

    def stop_training(self):
        if self.is_training:
            self._stop_requested = True
            self.is_training = False
            return {"status": "success", "message": "Training stop signal sent."}
        return {"status": "idle", "message": "No active training process."}

    def _run_training_job(self):
        try:
            market_config = MarketConfig(episode_length=50, seed=self.seed)
            env = MarketEnvironment(config=market_config)

            gt_config = GameTheoryConfig(seed=self.seed)
            use_stackelberg = self.variant in ("stackelberg_only", "proposed_full")
            use_opponent = self.variant in ("opponent_model_only", "proposed_full")

            strat_layer = (
                StrategicLayer(config=gt_config, use_stackelberg=use_stackelberg, use_opponent_model=use_opponent)
                if self.variant != "rl_only" else None
            )

            strat_dim = 11 if self.variant != "rl_only" else 0
            ppo_config = PPOConfig(
                rollout_length=128,
                batch_size=32,
                ppo_epochs=3,
                lr=3e-4,
                device="cpu",
                seed=self.seed
            )
            encoder_config = StateEncoderConfig(
                initial_price=market_config.initial_price,
                initial_cash=market_config.initial_cash,
                max_order_size=market_config.max_order_size
            )
            agent = PPOAgent(
                config=ppo_config,
                encoder_config=encoder_config,
                strategic_dim=strat_dim
            )

            trainer = RLTrainer(
                env=env,
                agent=agent,
                target_agent_id="rl_trader",
                enable_opponent_modeling=use_opponent,
                strategic_layer=strat_layer
            )

            ckpt_dir = f"checkpoints/{self.variant}_run"
            os.makedirs(ckpt_dir, exist_ok=True)

            step = 0
            states = env.reset()
            ep_reward = 0.0

            while step < self.total_timesteps and not self._stop_requested:
                market_state = env.get_market_state()
                leader_decision = strat_layer.solve_leader(market_state) if strat_layer else None
                strat_dict = strat_layer.build_observation(market_state, leader_decision) if strat_layer else None

                action_idx, log_prob, val = agent.select_action(
                    states["rl_trader"],
                    strategic_dict=strat_dict,
                    deterministic=False
                )
                rl_action = agent.action_space.to_env_action(action_idx)

                actions = {
                    "momentum": trainer.opponents["momentum"].act(states["momentum"]),
                    "value": trainer.opponents["value"].act(states["value"]),
                    "rl_trader": rl_action
                }

                next_states, rewards, done, info = env.step(actions, market_control=leader_decision)
                reward = rewards["rl_trader"]
                ep_reward += reward

                agent.store_transition(
                    state=states["rl_trader"],
                    action=action_idx,
                    log_prob=log_prob,
                    reward=reward,
                    value=val,
                    done=done,
                    strategic_dict=strat_dict
                )

                step += 1
                self.current_step = step

                # Update PPO
                if len(agent.buffer.states) >= agent.config.rollout_length:
                    next_m = env.get_market_state()
                    next_ld = strat_layer.solve_leader(next_m) if strat_layer else None
                    next_sd = strat_layer.build_observation(next_m, next_ld) if strat_layer else None
                    metrics = agent.update(next_states["rl_trader"], done, next_sd)
                    self.recent_metrics = metrics
                    self.training_history.append({
                        "step": step,
                        "policy_loss": metrics.get("policy_loss", 0.0),
                        "value_loss": metrics.get("value_loss", 0.0),
                        "total_loss": metrics.get("total_loss", 0.0),
                        "entropy": metrics.get("entropy", 0.0),
                        "approx_kl": metrics.get("approx_kl", 0.0),
                        "reward": float(ep_reward)
                    })

                if done:
                    states = env.reset()
                    ep_reward = 0.0
                else:
                    states = next_states

                # Small sleep to allow async loop responsiveness
                if step % 20 == 0:
                    time.sleep(0.01)

            # Save final checkpoint
            final_ckpt = os.path.join(ckpt_dir, f"{self.variant}_final.pt")
            agent.save_checkpoint(final_ckpt)

        except Exception as e:
            self.recent_metrics["error"] = str(e)
        finally:
            self.is_training = False


training_service = TrainingService()
