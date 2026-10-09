import os
import torch
from typing import Dict, Any


class CheckpointManager:
    """Manages saving, loading, and resuming training state for PPO/MAPPO agents."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        filepath: str,
        policy_state_dict: dict,
        value_state_dict: dict,
        optimizer_state_dict: dict,
        encoder_state_dict: dict,
        step_count: int,
        config_dict: dict | None = None
    ) -> str:
        """Save training checkpoint to disk."""
        if not filepath.endswith(".pt") and not filepath.endswith(".pth"):
            filepath = filepath + ".pt"

        full_path = os.path.join(self.checkpoint_dir, os.path.basename(filepath)) if not os.path.isabs(filepath) else filepath

        dir_name = os.path.dirname(full_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        payload = {
            "policy": policy_state_dict,
            "value": value_state_dict,
            "optimizer": optimizer_state_dict,
            "encoder": encoder_state_dict,
            "step_count": step_count,
            "config": config_dict or {}
        }
        torch.save(payload, full_path)
        return full_path

    def load_checkpoint(self, filepath: str, device: str = "cpu") -> Dict[str, Any]:
        """Load training checkpoint from disk."""
        if not os.path.exists(filepath):
            # Try looking in checkpoint_dir
            alt_path = os.path.join(self.checkpoint_dir, os.path.basename(filepath))
            if os.path.exists(alt_path):
                filepath = alt_path
            else:
                raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        payload = torch.load(filepath, map_location=device, weights_only=False)
        return payload
