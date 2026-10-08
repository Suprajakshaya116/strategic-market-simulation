import torch
import torch.nn as nn
from torch.distributions import Categorical
from .utils import orthogonal_init


class FeatureExtractor(nn.Module):
    """Deep feature extraction network with LayerNorm, GELU/Tanh, and Orthogonal Init."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        use_layer_norm: bool = True,
        use_orthogonal_init: bool = True,
        activation: str = "tanh"
    ):
        super().__init__()

        act_cls = nn.Tanh if activation.lower() == "tanh" else nn.GELU

        layers = []
        # Layer 1
        l1 = nn.Linear(input_dim, hidden_dim)
        if use_orthogonal_init:
            orthogonal_init(l1, gain=math.sqrt(2))
        layers.append(l1)
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_cls())

        # Layer 2
        l2 = nn.Linear(hidden_dim, hidden_dim)
        if use_orthogonal_init:
            orthogonal_init(l2, gain=math.sqrt(2))
        layers.append(l2)
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_cls())

        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


import math


class CategoricalActor(nn.Module):
    """Categorical policy head mapping features to action distribution."""

    def __init__(
        self,
        hidden_dim: int = 64,
        action_dim: int = 3,
        use_orthogonal_init: bool = True
    ):
        super().__init__()
        self.action_head = nn.Linear(hidden_dim, action_dim)
        if use_orthogonal_init:
            # Small gain for policy output layer to ensure near-uniform initial policy
            orthogonal_init(self.action_head, gain=0.01)

    def forward(self, features: torch.Tensor) -> Categorical:
        logits = self.action_head(features)
        return Categorical(logits=logits)


class CriticNetwork(nn.Module):
    """Value network estimating state value V(s)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        use_layer_norm: bool = True,
        use_orthogonal_init: bool = True
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            use_orthogonal_init=use_orthogonal_init,
            activation="tanh"
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        if use_orthogonal_init:
            orthogonal_init(self.value_head, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(state)
        value = self.value_head(feats)
        return value.squeeze(-1)
