from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .base_method import BaseGenerativeMethod


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DDPMMethod(BaseGenerativeMethod):
    """DDPM wrapper supporting both epsilon and x0 prediction."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        pred_type: str = "x0",
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        if pred_type not in {"eps", "x0"}:
            raise ValueError(f"pred_type must be eps or x0, got {pred_type}")
        self.pred_type = pred_type
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1.0 - betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, axis=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / self.alphas_cumprod - 1))
        posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_alpha_t * x_start) / torch.clamp(sqrt_one_minus_alpha_t, min=1e-12)

    def model_predictions(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: tuple[torch.Tensor, torch.Tensor | None],
        clip_denoised: bool = True,
    ) -> dict[str, torch.Tensor]:
        if self.pred_type == 'eps':
            eps_pred = model(x_t, t, cond)
            x0_pred = self.predict_start_from_noise(x_t, t, eps_pred)
        else:
            x0_pred = model(x_t, t, cond)
            eps_pred = self.predict_noise_from_start(x_t, t, x0_pred)
        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, min=0.0, max=10.0)
        return {"x0_pred": x0_pred, "eps_pred": eps_pred}

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        posterior_mean = self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: tuple[torch.Tensor, torch.Tensor | None],
        clip_denoised: bool = True,
    ):
        preds = self.model_predictions(model, x_t, t, cond, clip_denoised=clip_denoised)
        x0 = preds["x0_pred"]
        return self.q_posterior_mean_variance(x0, x_t, t)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, cond: tuple[torch.Tensor, torch.Tensor | None], clip_denoised: bool = True) -> torch.Tensor:
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, cond: tuple[torch.Tensor, torch.Tensor | None], shape: tuple[int, int, int], clip_denoised: bool = True) -> torch.Tensor:
        batch_size = shape[0]
        device = next(model.parameters()).device
        sample_data = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.timesteps)), desc='DDPM sampling loop', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            sample_data = self.p_sample(model, sample_data, t, cond, clip_denoised=clip_denoised)
        return sample_data

    @torch.no_grad()
    def ddim_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        cond: tuple[torch.Tensor, torch.Tensor | None],
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        preds = self.model_predictions(model, x_t, t, cond, clip_denoised=clip_denoised)
        x0 = preds["x0_pred"]
        eps = preds["eps_pred"]
        alpha_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)
        sigma_t = eta * torch.sqrt((1.0 - alpha_prev) / torch.clamp(1.0 - alpha_t, min=1e-12) * (1.0 - alpha_t / torch.clamp(alpha_prev, min=1e-12)))
        pred_dir = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma_t ** 2, min=0.0))
        noise = torch.randn_like(x_t)
        return torch.sqrt(alpha_prev) * x0 + pred_dir * eps + sigma_t * noise

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: nn.Module,
        cond: tuple[torch.Tensor, torch.Tensor | None],
        shape: tuple[int, int, int],
        ddim_steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        batch_size = shape[0]
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        time_pairs = torch.linspace(self.timesteps - 1, 0, steps=ddim_steps, device=device).long()
        prev_times = torch.cat([time_pairs[1:], torch.zeros(1, device=device, dtype=torch.long)])
        for t_now, t_prev in zip(time_pairs, prev_times):
            t = torch.full((batch_size,), t_now.item(), device=device, dtype=torch.long)
            tp = torch.full((batch_size,), t_prev.item(), device=device, dtype=torch.long)
            x = self.ddim_step(model, x, t, tp, cond, eta=eta, clip_denoised=clip_denoised)
        return x

    def compute_loss(
        self,
        model: nn.Module,
        batch_x: torch.Tensor,
        cond: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        batch_size = batch_x.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=batch_x.device).long()
        noise = torch.randn_like(batch_x)
        x_noisy = self.q_sample(batch_x, t, noise=noise)
        if self.pred_type == 'eps':
            predicted_noise = model(x_noisy, t, cond)
            return F.mse_loss(noise, predicted_noise, reduction='mean')
        x0_pred = model(x_noisy, t, cond)
        return F.mse_loss(batch_x, x0_pred, reduction='mean')

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
        sampler = str(kwargs.get('sampler', 'ddpm')).lower()
        shape = (batch_size, cell_num, dims)
        if sampler == 'ddim':
            return self.ddim_sample_loop(
                model=model,
                cond=cond,
                shape=shape,
                ddim_steps=int(kwargs.get('steps', kwargs.get('ddim_steps', 50))),
                eta=float(kwargs.get('eta', 0.0)),
                clip_denoised=bool(kwargs.get('clip_denoised', True)),
            )
        if sampler == 'ddpm':
            return self.p_sample_loop(
                model=model,
                cond=cond,
                shape=shape,
                clip_denoised=bool(kwargs.get('clip_denoised', True)),
            )
        raise ValueError(f"Unknown DDPM sampler: {sampler}")


__all__ = ["DDPMMethod"]
