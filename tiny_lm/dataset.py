"""Autoregressive next-token dataset."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset):
    """Returns an input window and the same window shifted by one token."""

    def __init__(self, token_ids: list[int] | torch.Tensor, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.tokens = torch.as_tensor(token_ids, dtype=torch.long)
        if self.tokens.ndim != 1:
            raise ValueError("token_ids must be one-dimensional")
        if self.tokens.numel() <= block_size:
            raise ValueError("token_ids must contain more than block_size tokens")
        self.block_size = block_size

    def __len__(self) -> int:
        return self.tokens.numel() - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index
        end = start + self.block_size
        return self.tokens[start:end], self.tokens[start + 1:end + 1]