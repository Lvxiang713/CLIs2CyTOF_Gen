from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
try:
    from torchdiffeq import odeint
except Exception:
    odeint = None

from .base_method import BaseGenerativeMethod


class FlowMatchingMethod(BaseGenerativeMethod):
    """Flow Matching method wrapper.

    This class stores the method specific loss and sampling logic, while the
    backbone remains method agnostic.
    """

    def __init__(self, time_log_path: str | None = None) -> None:
        super().__init__()
        self.time_log_path = time_log_path

    def _log_time(
        self,
        method_name: str,
        elapsed: float,
        batch_size: int,
        cell_num: int,
        dims: int,
        extra: str = "",
    ) -> None:
        if self.time_log_path is None:
            return
        os.makedirs(os.path.dirname(self.time_log_path), exist_ok=True)
        with open(self.time_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{method_name}\t"
                f"time={elapsed:.6f}\t"
                f"batch={batch_size}\t"
                f"cell_num={cell_num}\t"
                f"dims={dims}"
            )
            if extra:
                f.write(f"\t{extra}")
            f.write("\n")

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device)

    def mix(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        view_shape = (-1,) + (1,) * (x0.ndim - 1)
        return (1 - t).view(view_shape) * x0 + t.view(view_shape) * x1

    def velocity(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return x1 - x0

    def compute_loss(
        self,
        model: nn.Module,
        batch_x: torch.Tensor,
        cond: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        device = batch_x.device
        batch_size = batch_x.size(0)
        x1 = torch.randn_like(batch_x)
        t = self.sample_t(batch_size, device)
        x_t = self.mix(batch_x, x1, t)
        v_gt = self.velocity(batch_x, x1)
        v_pred = model(x_t, t, cond)
        return F.mse_loss(v_pred, v_gt)

    @torch.no_grad()
    def sample_euler(
        self,
        model: nn.Module,
        batch_size: int,
        cell_num: int,
        dims: int,
        cond: tuple[torch.Tensor, torch.Tensor | None],
        steps: int = 100,
    ) -> torch.Tensor:
        if odeint is None:
            raise ImportError("torchdiffeq is required for flow_method=odeint but is not installed.")
        start = time.time()
        device = next(model.parameters()).device
        x = torch.randn((batch_size, cell_num, dims), device=device)
        t_vals = torch.linspace(1.0, 0.0, steps, device=device)
        dt = t_vals[1] - t_vals[0]
        for t in t_vals:
            t_batch = t.expand(batch_size)
            v = model(x, t_batch, cond)
            x = x + v * dt

        self._log_time(
            method_name="flow_sample_euler",
            elapsed=time.time() - start,
            batch_size=batch_size,
            cell_num=cell_num,
            dims=dims,
            extra=f"steps={steps}",
        )
        return x

    @torch.no_grad()
    def sample_odeint(
        self,
        model: nn.Module,
        batch_size: int,
        cell_num: int,
        dims: int,
        cond: tuple[torch.Tensor, torch.Tensor | None],
        atol: float = 1e-5,
        rtol: float = 1e-5,
        solver: str = "dopri5",
    ) -> torch.Tensor:
        start = time.time()
        device = next(model.parameters()).device
        x0 = torch.randn((batch_size, cell_num, dims), device=device)

        def ode_func(t: torch.Tensor, x_flat: torch.Tensor) -> torch.Tensor:
            x = x_flat.view(batch_size, cell_num, dims)
            t_batch = torch.full((batch_size,), t, device=device)
            v = model(x, t_batch, cond)
            return v.view(-1, dims)

        x0_flat = x0.view(-1, dims)
        t_span = torch.linspace(1.0, 0.0, steps=2, device=device)
        sol = odeint(ode_func, x0_flat, t_span, rtol=rtol, atol=atol, method=solver)
        x_final = sol[-1].view(batch_size, cell_num, dims)

        self._log_time(
            method_name="flow_sample_odeint",
            elapsed=time.time() - start,
            batch_size=batch_size,
            cell_num=cell_num,
            dims=dims,
            extra=f"atol={atol},rtol={rtol},solver={solver}",
        )
        return x_final

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
        sampler = kwargs.get("sampler", "euler")
        if sampler == "euler":
            return self.sample_euler(
                model=model,
                batch_size=batch_size,
                cell_num=cell_num,
                dims=dims,
                cond=cond,
                steps=int(kwargs.get("steps", 100)),
            )
        if sampler == "odeint":
            return self.sample_odeint(
                model=model,
                batch_size=batch_size,
                cell_num=cell_num,
                dims=dims,
                cond=cond,
                atol=float(kwargs.get("atol", 1e-5)),
                rtol=float(kwargs.get("rtol", 1e-5)),
                solver=str(kwargs.get("solver", "dopri5")),
            )
        raise ValueError(f"Unknown Flow Matching sampler: {sampler}")


__all__ = ["FlowMatchingMethod"]
