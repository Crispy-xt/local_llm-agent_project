"""Data loading and train/validation splitting helpers."""
from __future__ import annotations
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .dataset import LanguageModelingDataset

def load_text(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if not text:
        raise ValueError("training text must not be empty")
    return text

def split_token_ids(token_ids: list[int] | torch.Tensor, validation_fraction: float, block_size: int, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    tokens = torch.as_tensor(token_ids, dtype=torch.long)
    if tokens.ndim != 1 or tokens.numel() <= 2 * block_size:
        raise ValueError("token sequence must contain more than two block sizes")
    split_index = int(tokens.numel() * (1.0 - validation_fraction))
    split_index = max(block_size + 1, min(split_index, tokens.numel() - block_size - 1))
    return tokens[:split_index], tokens[split_index:]

def create_dataloaders(train_tokens: torch.Tensor, validation_tokens: torch.Tensor, block_size: int, batch_size: int, seed: int = 42) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    train_dataset = LanguageModelingDataset(train_tokens, block_size)
    validation_dataset = LanguageModelingDataset(validation_tokens, block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader