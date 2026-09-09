"""Configuration loading for reproducible TinyLM experiments."""
from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
import yaml

@dataclass
class TrainConfig:
    data_path: str | None = None
    text: str | None = None
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.0
    learning_rate: float = 0.0003
    batch_size: int = 8
    max_steps: int = 1000
    eval_interval: int = 100
    eval_batches: int = 10
    validation_fraction: float = 0.1
    seed: int = 42
    device: str = "auto"
    checkpoint_path: str = "checkpoints/tiny_lm.pt"
    log_path: str = "logs/tiny_lm.csv"
    curve_path: str = "logs/tiny_lm_loss.svg"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown configuration fields: {unknown}")
        return cls(**payload)

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device