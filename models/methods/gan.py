from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base_method import BaseGenerativeMethod


class GANMethod(BaseGenerativeMethod):
    """Sampling wrapper for the GAN generator."""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def compute_loss(
        self,
        model: nn.Module,
        batch_x: torch.Tensor,
        cond: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        raise NotImplementedError("GAN uses a dedicated trainer and does not support generic compute_loss.")

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        cell_num: int,
        dims: int,
        cond: tuple[torch.Tensor, torch.Tensor | None],
        **kwargs: Any,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        z = torch.randn(batch_size, cell_num, self.latent_dim, device=device)
        return model(z, cond)


__all__ = ["GANMethod"]
