from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


AggregateMode = Literal["mean", "sum", "first", "last"]


@dataclass
class CrossAttentionRecorder:
    """Collect attention weights from CrossAttentionBlock instances during forward passes."""

    enabled: bool = False
    records: list[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        self.records.clear()

    def enable(self, enabled: bool = True, clear: bool = True) -> None:
        self.enabled = bool(enabled)
        if clear:
            self.clear()

    def add(self, weights: torch.Tensor) -> None:
        if not self.enabled:
            return
        self.records.append(weights.detach())

    def collect(self, aggregate: AggregateMode = "mean", clear: bool = True) -> torch.Tensor | None:
        if not self.records:
            return None
        mats: list[torch.Tensor] = []
        for w in self.records:
            if w.dim() == 4:
                # (B, H, Q, K) -> average over heads
                w = w.mean(dim=1)
            mats.append(w.float())
        stack = torch.stack(mats, dim=0)  # (N_blocks, B, Q, K)
        if aggregate == "mean":
            out = stack.mean(dim=0)
        elif aggregate == "sum":
            out = stack.sum(dim=0)
        elif aggregate == "first":
            out = stack[0]
        elif aggregate == "last":
            out = stack[-1]
        else:
            raise ValueError(f"Unknown aggregate={aggregate}")
        if clear:
            self.clear()
        return out


def attach_cross_attention_recorder(module: torch.nn.Module, recorder: CrossAttentionRecorder) -> None:
    """Attach a recorder to every child CrossAttentionBlock if present."""
    for child in module.modules():
        if child.__class__.__name__ == "CrossAttentionBlock":
            setattr(child, "analysis_recorder", recorder)
