import torch
import torch.nn as nn
from .networks import CriticNetwork


class ValueNetwork(nn.Module):
    """Single-agent State Value Network estimating V(s)."""

    def __init__(
        self,
        state_size: int = 10,
        hidden_dim: int = 64,
        use_layer_norm: bool = True,
        use_orthogonal_init: bool = True
    ):
        super().__init__()
        self.critic = CriticNetwork(
            input_dim=state_size,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            use_orthogonal_init=use_orthogonal_init
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate state value V(s)."""
        return self.critic(state)
