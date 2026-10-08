from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class StateEncoderConfig:
    """Configuration for raw state preprocessing and normalization."""
    initial_price: float = 100.0
    initial_cash: float = 100_000.0
    initial_spread: float = 0.02
    max_order_size: int = 100
    clip_observation: float = 10.0
    use_running_stats: bool = True
    epsilon: float = 1e-8


@dataclass
class PPOConfig:
    """Configuration for PPO agent hyper-parameters."""
    lr: float = 3e-4
    lr_decay: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    c1_value_loss_coeff: float = 0.5
    c2_entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    batch_size: int = 64
    rollout_length: int = 2048
    hidden_dim: int = 64
    use_orthogonal_init: bool = True
    use_layer_norm: bool = True
    clip_value_loss: bool = True
    target_kl: Optional[float] = 0.015
    device: str = "cpu"
    seed: Optional[int] = 42


@dataclass
class MARLConfig:
    """Configuration for Multi-Agent PPO (MAPPO) with CTDE."""
    agent_ids: Tuple[str, ...] = ("momentum", "value", "rl_trader")
    target_agent_id: str = "rl_trader"
    use_ctde: bool = True
    centralized_critic_dim: int = 30
    share_policy: bool = False
    ppo_config: PPOConfig = field(default_factory=PPOConfig)
