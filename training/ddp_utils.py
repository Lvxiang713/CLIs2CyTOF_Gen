from __future__ import annotations

import os
import platform

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(rank: int, world_size: int, master_addr: str = "127.0.0.1", master_port: str = "12355") -> None:
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    backend = "gloo" if platform.system() == "Windows" else "nccl"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """Destroy the distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_rank_zero() -> bool:
    return (not is_dist_initialized()) or dist.get_rank() == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Recursively unwrap DDP and torch.compile wrappers."""
    while True:
        if isinstance(model, DDP):
            model = model.module
            continue
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        return model


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """All reduce a tensor with sum semantics if DDP is initialized."""
    if is_dist_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


__all__ = [
    "setup_ddp",
    "cleanup_ddp",
    "is_dist_initialized",
    "is_rank_zero",
    "unwrap_model",
    "all_reduce_sum",
]
