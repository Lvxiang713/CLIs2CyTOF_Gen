from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..models.methods.base_method import BaseGenerativeMethod


@dataclass
class SamplingConfig:
    """Configuration for donor level chunked sampling."""

    num_cells_total: int
    cell_num_per_sample: int
    feature_num: int
    output_dir: str
    time_log_path: str | None = None
    sampler: str = "euler"
    euler_steps: int = 100
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-5
    ode_solver: str = "dopri5"


class ChunkSampler:
    """Generate donor level synthetic cells in chunks and save them to disk."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _normalize_donor_ids(self, donor_ids: Iterable[str]) -> list[str]:
        seen = set()
        ordered: list[str] = []
        for donor_id in donor_ids:
            donor_id = str(donor_id)
            if donor_id not in seen:
                seen.add(donor_id)
                ordered.append(donor_id)
        return ordered

    def _extract_donor_ids_from_loader(self, loader: DataLoader) -> list[str]:
        dataset = loader.dataset

        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            indices = dataset.indices
            donor_ids = [base_dataset.samples[idx][0] for idx in indices]
            return self._normalize_donor_ids(donor_ids)

        if hasattr(dataset, "samples"):
            donor_ids = [sample[0] for sample in dataset.samples]
            return self._normalize_donor_ids(donor_ids)

        raise TypeError(
            "Unable to infer donor IDs from loader.dataset. "
            "Please pass donor_ids explicitly through sample_by_donor()."
        )

    def sample_from_loader(
        self,
        *,
        model: torch.nn.Module,
        method: BaseGenerativeMethod,
        donor_condition_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
        loader: DataLoader,
        cfg: SamplingConfig,
    ) -> list[str]:
        donor_ids = self._extract_donor_ids_from_loader(loader)
        self.sample_by_donor(
            model=model,
            method=method,
            donor_condition_dict=donor_condition_dict,
            donor_ids=donor_ids,
            cfg=cfg,
        )
        return donor_ids

    def sample_by_donor(
        self,
        *,
        model: torch.nn.Module,
        method: BaseGenerativeMethod,
        donor_condition_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
        donor_ids: Iterable[str],
        cfg: SamplingConfig,
    ) -> list[str]:
        donor_ids = self._normalize_donor_ids(donor_ids)
        os.makedirs(cfg.output_dir, exist_ok=True)

        log_file = None
        log_writer = None
        if cfg.time_log_path is not None:
            os.makedirs(os.path.dirname(cfg.time_log_path), exist_ok=True)
            log_file = open(cfg.time_log_path, "a", newline="", encoding="utf-8")
            log_writer = csv.writer(log_file)
            if log_file.tell() == 0:
                log_writer.writerow(["donor_id", "kind", "call_idx", "elapsed_sec", "num_cells", "extra"])

        self._synchronize()
        global_start = time.time()
        model.eval()

        with torch.no_grad():
            for donor_id in donor_ids:
                if donor_id not in donor_condition_dict:
                    print(f"[Warning] donor_id={donor_id} not found in donor_condition_dict. Skipped.")
                    continue

                cond_embeds, cond_mask = donor_condition_dict[donor_id]
                cond = (cond_embeds.to(self.device), cond_mask.to(self.device))
                generated_chunks: list[list[float]] = []

                self._synchronize()
                donor_start = time.time()
                call_idx = 0

                while len(generated_chunks) < cfg.num_cells_total:
                    self._synchronize()
                    call_start = time.time()
                    gen = method.sample(
                        model=model,
                        batch_size=1,
                        cell_num=cfg.cell_num_per_sample,
                        dims=cfg.feature_num,
                        cond=cond,
                        sampler=cfg.sampler,
                        steps=cfg.euler_steps,
                        rtol=cfg.ode_rtol,
                        atol=cfg.ode_atol,
                        solver=cfg.ode_solver,
                    )
                    self._synchronize()
                    elapsed = time.time() - call_start
                    call_idx += 1

                    print(f"[Sampling-{cfg.sampler.upper()}] donor {donor_id}: call {call_idx} took {elapsed:.3f} s")
                    if log_writer is not None:
                        extra = (
                            f"sampler={cfg.sampler},steps={cfg.euler_steps}"
                            if cfg.sampler == "euler"
                            else f"sampler={cfg.sampler},solver={cfg.ode_solver},rtol={cfg.ode_rtol},atol={cfg.ode_atol}"
                        )
                        log_writer.writerow([donor_id, "per_call", call_idx, f"{elapsed:.6f}", cfg.cell_num_per_sample, extra])

                    generated_chunks.extend(gen.squeeze(0).cpu().tolist())

                donor_elapsed = time.time() - donor_start
                print(
                    f"[Sampling-{cfg.sampler.upper()}] donor {donor_id}: "
                    f"generated {cfg.num_cells_total} cells in {donor_elapsed:.3f} s"
                )
                if log_writer is not None:
                    log_writer.writerow([donor_id, "donor_total", "", f"{donor_elapsed:.6f}", cfg.num_cells_total, cfg.sampler])

                arr = np.asarray(generated_chunks[: cfg.num_cells_total], dtype=np.float32)
                np.save(os.path.join(cfg.output_dir, f"{donor_id}.npy"), arr)
                print(f"Saved donor {donor_id} to {cfg.output_dir}")

        self._synchronize()
        total_elapsed = time.time() - global_start
        print(f"[Sampling-{cfg.sampler.upper()}] total wall time: {total_elapsed:.3f} s")
        if log_writer is not None:
            log_writer.writerow(["ALL", "all_total", "", f"{total_elapsed:.6f}", "", cfg.sampler])
            log_file.close()

        return donor_ids


__all__ = ["SamplingConfig", "ChunkSampler"]
