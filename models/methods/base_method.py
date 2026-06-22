from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseGenerativeMethod(nn.Module, ABC):
    """Common interface for future generative methods.

    The training engine only depends on this interface. New methods such as DDPM,
    VAE, GAN, or GAN can be added by implementing this base class without
    changing the trainer.
    """

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        batch_x: torch.Tensor,
        cond: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError


__all__ = ["BaseGenerativeMethod"]
