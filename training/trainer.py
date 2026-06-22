from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, DistributedSampler

from ..models.conditions.clip_condition import ClipConditionEncoder
from ..models.methods.base_method import BaseGenerativeMethod
from .ddp_utils import all_reduce_sum, is_dist_initialized, is_rank_zero, unwrap_model


@dataclass
class TrainerOutput:
    """Outputs returned by the trainer."""

    best_val_loss: float
    best_ckpt_path: str


class GenerativeTrainer:
    """Generic trainer for conditional generative methods with a single optimizer."""

    def __init__(
        self,
        device: torch.device,
        ckpt_dir: str,
        num_epochs: int = 1000,
        early_stop: int = 20,
    ) -> None:
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.num_epochs = num_epochs
        self.early_stop = early_stop
        self.best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _run_one_epoch(
        self,
        *,
        model: torch.nn.Module,
        method: BaseGenerativeMethod,
        condition_encoder: ClipConditionEncoder,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        train: bool,
        epoch: int,
    ) -> float:
        if train:
            model.train()
            if is_dist_initialized() and isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(epoch)
        else:
            model.eval()

        local_loss_sum = 0.0
        local_count = 0

        for sc_data, ehr_data, _, _ in loader:
            sc_data = sc_data.to(self.device, non_blocking=True)
            ehr_data = ehr_data.to(self.device, non_blocking=True)

            with torch.no_grad():
                cond = condition_encoder.encode_batch(ehr_data)

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss = method.compute_loss(model, sc_data, cond)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    loss = method.compute_loss(model, sc_data, cond)

            batch_size = sc_data.size(0)
            local_loss_sum += loss.item() * batch_size
            local_count += batch_size

        loss_tensor = torch.tensor(local_loss_sum, dtype=torch.float64, device=self.device)
        count_tensor = torch.tensor(local_count, dtype=torch.float64, device=self.device)
        loss_tensor = all_reduce_sum(loss_tensor)
        count_tensor = all_reduce_sum(count_tensor)
        return (loss_tensor / count_tensor.clamp(min=1)).item()

    def fit(
        self,
        *,
        model: torch.nn.Module,
        method: BaseGenerativeMethod,
        condition_encoder: ClipConditionEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    ) -> TrainerOutput:
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._run_one_epoch(
                model=model,
                method=method,
                condition_encoder=condition_encoder,
                loader=train_loader,
                optimizer=optimizer,
                train=True,
                epoch=epoch,
            )
            val_loss = self._run_one_epoch(
                model=model,
                method=method,
                condition_encoder=condition_encoder,
                loader=val_loader,
                optimizer=None,
                train=False,
                epoch=epoch,
            )

            if scheduler is not None:
                scheduler.step(val_loss)

            if is_rank_zero():
                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(now)
                print(f"Epoch {epoch}/{self.num_epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if is_rank_zero():
                    torch.save(unwrap_model(model).state_dict(), self.best_ckpt_path)
                    print("Validation improved. Saved checkpoint.")
            else:
                epochs_no_improve += 1
                if is_rank_zero():
                    print(f"No improvement for {epochs_no_improve} epoch(s).")
                if epochs_no_improve >= self.early_stop:
                    if is_rank_zero():
                        print("Early stopping triggered.")
                    break

        if is_dist_initialized():
            torch.distributed.barrier()

        unwrap_model(model).load_state_dict(torch.load(self.best_ckpt_path, map_location=self.device, weights_only=True))
        if is_rank_zero():
            print("Loaded the best model checkpoint.")

        return TrainerOutput(best_val_loss=best_val_loss, best_ckpt_path=self.best_ckpt_path)


__all__ = ["GenerativeTrainer", "TrainerOutput"]
