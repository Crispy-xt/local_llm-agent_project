"""Causal multi-head self-attention."""

from __future__ import annotations

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, n_embd = x.shape
        if sequence_length > self.block_size:
            raise ValueError("sequence length exceeds block_size")
        q, k, v = self.qkv(x).split(n_embd, dim=-1)
        q = q.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        attention_scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        attention_scores = attention_scores.masked_fill(
            ~self.causal_mask[:, :, :sequence_length, :sequence_length],
            torch.finfo(attention_scores.dtype).min,
        )
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        output = attention_weights @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, n_embd)
        return self.resid_dropout(self.proj(output))