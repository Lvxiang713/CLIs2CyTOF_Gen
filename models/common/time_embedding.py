from __future__ import annotations

import math

import torch


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal embeddings for continuous or discrete timesteps.

    Args:
        timesteps: Tensor of shape [batch]. Values may be integer or float.
        dim: Embedding dimension.
        max_period: Controls the minimum frequency in the sinusoidal embedding.

    Returns:
        Tensor of shape [batch, dim].
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


__all__ = ["timestep_embedding"]
