import torch
import torch.nn as nn
from .networks import CriticNetwork


class CentralizedCritic(nn.Module):
    """Centralized Critic V(S_global) for Multi-Agent PPO (MAPPO CTDE architecture).

    Computes global state-value V(S_global) during training, using concatenated agent states,
    shared market parameters, and strategic information.
    """

    def __init__(
        self,
        global_state_dim: int,
        hidden_dim: int = 128,
        use_layer_norm: bool = True,
        use_orthogonal_init: bool = True
    ):
        super().__init__()
        self.global_state_dim = global_state_dim
        self.critic = CriticNetwork(
            input_dim=global_state_dim,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            use_orthogonal_init=use_orthogonal_init
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Estimate global state value V(S_global)."""
        return self.critic(global_state)
