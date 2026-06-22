from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..backbones.cvae import vae_loss
from .base_method import BaseGenerativeMethod


class VAEMethod(BaseGenerativeMethod):
    """VAE wrapper exposing compute_loss and sample methods."""

    def __init__(self, kl_weight: float = 0.1) -> None:
        super().__init__()
        self.kl_weight = kl_weight

    def compute_loss(
        self,
        model: nn.Module,
        batch_x: torch.Tensor,
        cond: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        x_recon, mu, logvar = model(batch_x, cond)
        total, _, _ = vae_loss(x_recon, batch_x, mu, logvar, kl_weight=self.kl_weight)
        return total

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
        return model.sample_from_prior(batch_size=batch_size, cell_num=cell_num, cd=cond)


__all__ = ["VAEMethod"]
