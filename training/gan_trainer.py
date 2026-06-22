from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, DistributedSampler

from ..models.backbones.gan import binned_reconstruction_loss, gradient_penalty
from ..models.conditions.clip_condition import ClipConditionEncoder
from .ddp_utils import all_reduce_sum, is_dist_initialized, is_rank_zero, unwrap_model


@dataclass
class GANTrainerOutput:
    best_val_loss: float
    best_ckpt_path: str


class GANTrainer:
    """Dedicated trainer for the GAN generator and discriminator."""

    def __init__(
        self,
        device: torch.device,
        ckpt_dir: str,
        num_epochs: int = 1000,
        latent_dim: int = 128,
        feature_dims: int = 36,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        lambda_recon: float = 10.0,
        recon_num_bins: int = 32,
        recon_sigma_factor: float = 0.5,
        recon_mode: str = "js",
    ) -> None:
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.num_epochs = num_epochs
        self.latent_dim = latent_dim
        self.feature_dims = feature_dims
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.lambda_recon = lambda_recon
        self.recon_num_bins = recon_num_bins
        self.recon_sigma_factor = recon_sigma_factor
        self.recon_mode = recon_mode
        self.best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
        self.last_ckpt_path = os.path.join(ckpt_dir, "last_epoch_model.pth")
        os.makedirs(ckpt_dir, exist_ok=True)

    def _allreduce_scalar(self, value: float) -> float:
        tensor = torch.tensor(value, device=self.device, dtype=torch.float64)
        tensor = all_reduce_sum(tensor)
        return tensor.item()

    def fit(
        self,
        *,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        condition_encoder: ClipConditionEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
        vae_init_ckpt: str = "",
    ) -> GANTrainerOutput:
        if vae_init_ckpt:
            state_obj = torch.load(vae_init_ckpt, map_location=self.device)
            if isinstance(state_obj, dict) and 'model_state_dict' in state_obj:
                state_obj = state_obj['model_state_dict']
            decoder_state = {}
            for key, value in state_obj.items():
                new_key = key
                while new_key.startswith('module.') or new_key.startswith('_orig_mod.'):
                    if new_key.startswith('module.'):
                        new_key = new_key[len('module.'):]
                    if new_key.startswith('_orig_mod.'):
                        new_key = new_key[len('_orig_mod.'):]
                if new_key.startswith('decoder.'):
                    decoder_state[new_key[len('decoder.'):]] = value
            unwrap_model(generator).load_state_dict(decoder_state, strict=False)
            if is_rank_zero():
                print(f"Loaded VAE decoder weights from {vae_init_ckpt}")

        best_val_loss = float('inf')
        for epoch in range(1, self.num_epochs + 1):
            if is_dist_initialized() and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            generator.train()
            discriminator.train()

            train_g_total = train_g_count = 0.0
            train_d_total = train_d_count = 0.0

            for step, (sc_data, ehr_data, _, _) in enumerate(train_loader):
                sc_data = sc_data.to(self.device, non_blocking=True)
                ehr_data = ehr_data.to(self.device, non_blocking=True)
                with torch.no_grad():
                    cond = condition_encoder.encode_batch(ehr_data)
                    cond_vec = condition_encoder.pool_condition(cond)
                batch_size = sc_data.size(0)
                cell_num = sc_data.size(1) if sc_data.size(2) == self.feature_dims else sc_data.size(2)
                real_for_d = sc_data if sc_data.size(-1) == self.feature_dims else sc_data.transpose(1, 2).contiguous()

                for p in discriminator.parameters():
                    p.requires_grad_(True)
                for p in generator.parameters():
                    p.requires_grad_(False)

                with torch.no_grad():
                    z_d = torch.randn(batch_size, cell_num, self.latent_dim, device=self.device)
                    fake_d = generator(z_d, cond)

                score_real = discriminator(real_for_d, cond_vec)
                score_fake = discriminator(fake_d.detach(), cond_vec)
                wasserstein_d = score_fake.mean() - score_real.mean()
                gp = gradient_penalty(discriminator, real_for_d, fake_d.detach(), cond_vec, lambda_gp=self.lambda_gp)
                loss_d = wasserstein_d + gp
                optimizer_d.zero_grad(set_to_none=True)
                loss_d.backward()
                optimizer_d.step()
                train_d_total += loss_d.item() * batch_size
                train_d_count += batch_size

                if step % self.n_critic == 0:
                    for p in discriminator.parameters():
                        p.requires_grad_(False)
                    for p in generator.parameters():
                        p.requires_grad_(True)
                    z = torch.randn(batch_size, cell_num, self.latent_dim, device=self.device)
                    fake = generator(z, cond)
                    score_fake_g = discriminator(fake, cond_vec)
                    adv_g = -score_fake_g.mean()
                    recon_g = binned_reconstruction_loss(
                        real=sc_data,
                        fake=fake,
                        feature_dims=self.feature_dims,
                        num_bins=self.recon_num_bins,
                        sigma_factor=self.recon_sigma_factor,
                        mode=self.recon_mode,
                    )
                    loss_g = adv_g + self.lambda_recon * recon_g
                    optimizer_g.zero_grad(set_to_none=True)
                    loss_g.backward()
                    optimizer_g.step()
                    train_g_total += loss_g.item() * batch_size
                    train_g_count += batch_size

            generator.eval()
            discriminator.eval()
            val_g_total = val_g_count = 0.0
            with torch.no_grad():
                for sc_data, ehr_data, _, _ in val_loader:
                    sc_data = sc_data.to(self.device, non_blocking=True)
                    ehr_data = ehr_data.to(self.device, non_blocking=True)
                    cond = condition_encoder.encode_batch(ehr_data)
                    cond_vec = condition_encoder.pool_condition(cond)
                    batch_size = sc_data.size(0)
                    cell_num = sc_data.size(1) if sc_data.size(2) == self.feature_dims else sc_data.size(2)
                    z = torch.randn(batch_size, cell_num, self.latent_dim, device=self.device)
                    fake = generator(z, cond)
                    adv_g = -discriminator(fake, cond_vec).mean()
                    recon_g = binned_reconstruction_loss(
                        real=sc_data,
                        fake=fake,
                        feature_dims=self.feature_dims,
                        num_bins=self.recon_num_bins,
                        sigma_factor=self.recon_sigma_factor,
                        mode=self.recon_mode,
                    )
                    loss_g = adv_g + self.lambda_recon * recon_g
                    val_g_total += loss_g.item() * batch_size
                    val_g_count += batch_size

            train_g_loss = self._allreduce_scalar(train_g_total) / max(self._allreduce_scalar(train_g_count if train_g_count > 0 else 1.0), 1.0)
            train_d_loss = self._allreduce_scalar(train_d_total) / max(self._allreduce_scalar(train_d_count if train_d_count > 0 else 1.0), 1.0)
            val_g_loss = self._allreduce_scalar(val_g_total) / max(self._allreduce_scalar(val_g_count if val_g_count > 0 else 1.0), 1.0)

            if scheduler_g is not None:
                scheduler_g.step(val_g_loss)

            if is_rank_zero():
                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(now)
                print(f"Epoch {epoch}/{self.num_epochs} | train_G={train_g_loss:.6f} | train_D={train_d_loss:.6f} | val_G={val_g_loss:.6f}")
                if val_g_loss < best_val_loss:
                    best_val_loss = val_g_loss
                    torch.save(unwrap_model(generator).state_dict(), self.best_ckpt_path)
                    print("Validation improved. Saved GAN generator checkpoint.")

        if is_rank_zero():
            torch.save(unwrap_model(generator).state_dict(), self.last_ckpt_path)
            print(f"Saved last-epoch GAN generator checkpoint to {self.last_ckpt_path}")
        if is_dist_initialized():
            torch.distributed.barrier()
        unwrap_model(generator).load_state_dict(torch.load(self.best_ckpt_path, map_location=self.device, weights_only=True))
        if is_rank_zero():
            print("Loaded the best GAN generator checkpoint.")
        return GANTrainerOutput(best_val_loss=best_val_loss, best_ckpt_path=self.best_ckpt_path)


__all__ = ["GANTrainer", "GANTrainerOutput"]
