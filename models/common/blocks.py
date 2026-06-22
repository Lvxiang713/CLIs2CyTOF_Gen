from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class TimestepBlock(nn.Module):
    """Base class for modules that consume timestep embeddings."""

    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ConditioningBlock(nn.Module):
    """Base class for modules that consume conditional embeddings."""

    @abstractmethod
    def forward(self, x: torch.Tensor, cond: tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock, ConditioningBlock):
    """Sequential container that forwards timestep and condition only when needed."""

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        cond: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, ConditioningBlock):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


def norm_layer(dims: int) -> nn.LayerNorm:
    """Return the normalization layer used across the backbone."""
    return nn.LayerNorm(dims)


class ResidualBlock(TimestepBlock):
    """Residual MLP block with additive timestep conditioning.

    The attribute names intentionally follow the original training script so that
    legacy checkpoints can be loaded without renaming state dict keys.
    """

    def __init__(self, in_dims: int, out_dims: int, time_dims: int, dropout: float) -> None:
        super().__init__()
        self.Linear1 = nn.Sequential(
            norm_layer(in_dims),
            nn.SiLU(),
            nn.Linear(in_dims, out_dims),
        )
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dims, out_dims),
        )
        self.Linear2 = nn.Sequential(
            norm_layer(out_dims),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_dims, out_dims),
        )
        self.shortcut = nn.Linear(in_dims, out_dims) if in_dims != out_dims else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.Linear1(x)
        h = h + self.time_emb(emb)[:, None, :]
        h = self.Linear2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self attention block with original attribute names for checkpoint compatibility."""

    def __init__(self, dims: int, num_heads: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        if dims % num_heads != 0:
            raise ValueError(f"dims={dims} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.normq = nn.LayerNorm(dims)
        self.normk = nn.LayerNorm(dims)
        self.normv = nn.LayerNorm(dims)
        self.att = nn.MultiheadAttention(dims, num_heads, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.normq(x)
        k = self.normk(x)
        v = self.normv(x)
        h, _ = self.att(q, k, v)
        return h + x


class CrossAttentionBlock(ConditioningBlock):
    """Cross attention block with original attribute names for checkpoint compatibility."""

    def __init__(self, query_dim: int, key_value_dim: int, num_heads: int = 1) -> None:
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim={query_dim} must be divisible by num_heads={num_heads}")
        if key_value_dim % num_heads != 0:
            raise ValueError(f"key_value_dim={key_value_dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.normq = nn.LayerNorm(query_dim)
        self.prok = nn.LayerNorm(key_value_dim)
        self.prov = nn.LayerNorm(key_value_dim)
        self.att = nn.MultiheadAttention(
            query_dim,
            num_heads,
            batch_first=True,
            kdim=key_value_dim,
            vdim=key_value_dim,
        )
        self.analysis_recorder = None

    def forward(self, query: torch.Tensor, cond: tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
        key_value_embeds, attention_mask = cond
        q = self.normq(query)
        k = self.prok(key_value_embeds)
        v = self.prov(key_value_embeds)

        key_padding_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
            key_padding_mask = ~attention_mask

        record_weights = self.analysis_recorder is not None and getattr(self.analysis_recorder, "enabled", False)
        h, attn_weights = self.att(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            need_weights=record_weights,
            average_attn_weights=False if record_weights else True,
        )
        if record_weights and attn_weights is not None:
            self.analysis_recorder.add(attn_weights)
        return h + query


__all__ = [
    "TimestepBlock",
    "ConditioningBlock",
    "TimestepEmbedSequential",
    "norm_layer",
    "ResidualBlock",
    "AttentionBlock",
    "CrossAttentionBlock",
]
