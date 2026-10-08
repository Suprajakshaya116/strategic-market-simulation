import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int | None) -> None:
    """Set random seeds across python, numpy, and torch for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(data, device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
    """Convert numpy array or python numeric sequence to PyTorch tensor."""
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    return torch.tensor(np.array(data), device=device, dtype=dtype)


def orthogonal_init(layer: nn.Module, gain: float = 1.0) -> nn.Module:
    """Apply orthogonal initialization to linear layers (standard PPO practice)."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    return layer
