"""Checkpoint persistence helpers."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import torch
from .model import TinyGPT, TinyGPTConfig
from .tokenizer import CharacterTokenizer

def save_checkpoint(path: str | Path, model: TinyGPT, optimizer: torch.optim.Optimizer | None, step: int, tokenizer: CharacterTokenizer, metadata: dict | None = None) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "metadata": metadata or {},
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_random_state"] = torch.cuda.get_rng_state_all()
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)

def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[TinyGPT, torch.optim.Optimizer, CharacterTokenizer, int]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = TinyGPTConfig(**payload["config"])
    model = TinyGPT(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    if payload["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    tokenizer = CharacterTokenizer.from_dict(payload["tokenizer"])
    return model, optimizer, tokenizer, int(payload["step"])