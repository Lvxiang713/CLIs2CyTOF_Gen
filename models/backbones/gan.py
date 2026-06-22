from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils import spectral_norm

from .cvae import SingleCellDecoder


class ViTBackbone(nn.Module):
    """Vision discriminator backbone used by the original GAN script."""

    def __init__(
        self,
        numFeas: int,
        quantileLen: int = 32,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        noise_std: float = 0.1,
        cut_len: int = 16,
        cut_stepsize: int = 16,
        num_CNN_layers: int = 4,
        concat_CNNlayer=(2, 4),
        hidden_channels: int = 32,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.numFeas = numFeas
        self.quantileLen = quantileLen
        self.noise_std = noise_std
        self.cut_len = cut_len
        self.cut_stepsize = cut_stepsize
        self.concat_CNNlayer = list(concat_CNNlayer)
        self.embed_dim = embed_dim

        sobel_kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer('sobel_kernel_x', sobel_kernel_x.repeat(numFeas, 1, 1, 1))
        self.register_buffer('sobel_kernel_y', sobel_kernel_y.repeat(numFeas, 1, 1, 1))

        cnn_layers = []
        current_channels = numFeas
        for _ in range(num_CNN_layers):
            cnn_layers.append(nn.Conv2d(current_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            cnn_layers.append(nn.InstanceNorm2d(hidden_channels))
            cnn_layers.append(nn.ReLU())
            current_channels = hidden_channels
        self.cnnList = nn.ModuleList(cnn_layers)
        self.q = torch.linspace(0.01, 0.99, quantileLen)
        self.patch_embed = nn.Linear(hidden_channels * cut_len ** 2, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0)

    def grad(self, x: torch.Tensor, conv_kernel: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, conv_kernel, bias=None, stride=1, padding=1, groups=x.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        x_q = x.quantile(q=self.q.to(x.device), dim=1, keepdim=False)
        x_q = x_q.permute(1, 0, 2)
        x_q_2d = x_q.unsqueeze(1) - x_q.unsqueeze(2)
        x_q_2d = x_q_2d.permute(0, 3, 1, 2)
        grad_x = self.grad(x_q_2d, self.sobel_kernel_x)
        grad_y = self.grad(x_q_2d, self.sobel_kernel_y)
        x_q_2d = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        cnn_middle_x_list = []
        for layer in self.cnnList:
            x_q_2d = layer(x_q_2d)
            if isinstance(layer, nn.Conv2d):
                cnn_middle_x_list.append(x_q_2d)

        x_q_1d_list = []
        for feat_map in cnn_middle_x_list:
            feat_map = feat_map.permute(0, 2, 3, 1)
            feat_map = feat_map.unfold(1, self.cut_len, self.cut_stepsize).unfold(2, self.cut_len, self.cut_stepsize).flatten(-3)
            l0 = feat_map.size(1)
            feat_map = feat_map.reshape(n, l0 * l0, -1)
            x_q_1d_list.append(feat_map)

        x_q_1d = torch.cat([t for idx, t in enumerate(x_q_1d_list) if (idx + 1) in self.concat_CNNlayer], dim=1)
        x_tokens = self.patch_embed(x_q_1d)
        cls_tokens = self.cls_token.expand(n, -1, -1).clone()
        x_tokens = torch.cat((cls_tokens, x_tokens), dim=1)
        x_tokens = self.pos_drop(x_tokens)
        x_tokens = self.transformer_encoder(x_tokens)
        return self.norm(x_tokens[:, 0])


class ViTCondDiscriminator(nn.Module):
    """Conditional discriminator used by the original GAN script."""

    def __init__(self, numFeas: int, cond_dim: int, vit_embed_dim: int = 128) -> None:
        super().__init__()
        self.vit = ViTBackbone(numFeas=numFeas, quantileLen=32, embed_dim=vit_embed_dim, depth=6, num_heads=8)
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(vit_embed_dim + cond_dim, 256)),
            nn.SiLU(),
            spectral_norm(nn.Linear(256, 1)),
        )

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        feat = self.vit(x)
        h = torch.cat([feat, cond_vec], dim=-1)
        return self.fc(h).squeeze(-1)


def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    cond_vec: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """WGAN GP gradient penalty."""
    batch_size = real.size(0)
    device = real.device
    alpha = torch.rand(batch_size, 1, 1, device=device).expand_as(real)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "sdp_kernel"):
        from torch.backends.cuda import sdp_kernel
        sdp_ctx = sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False)
    else:
        sdp_ctx = nullcontext()
    with sdp_ctx:
        d_interpolates = discriminator(interpolates, cond_vec)
    gradients = torch.autograd.grad(
        outputs=d_interpolates.sum(),
        inputs=interpolates,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean() * lambda_gp


def _to_bcf(x: torch.Tensor, feature_dims: int) -> torch.Tensor:
    """Normalize tensors to (B, C, F)."""
    if x.dim() != 3:
        raise AssertionError(f"Expect 3D tensor, got {tuple(x.shape)}")
    _, d1, d2 = x.shape
    if d2 == feature_dims:
        return x.contiguous()
    if d1 == feature_dims:
        return x.transpose(1, 2).contiguous()
    raise AssertionError(f"Cannot infer feature dimension from shape={tuple(x.shape)} and feature_dims={feature_dims}")


def _make_featurewise_bin_centers(real_bcf: torch.Tensor, num_bins: int, eps: float = 1e-6):
    with torch.no_grad():
        rmin = real_bcf.amin(dim=(0, 1))
        rmax = real_bcf.amax(dim=(0, 1))
        span = (rmax - rmin).clamp_min(eps)
        edges01 = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=real_bcf.device, dtype=real_bcf.dtype)
        edges = rmin[:, None] + span[:, None] * edges01[None, :]
        centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
        bin_width = (edges[:, 1:] - edges[:, :-1]).mean(dim=1, keepdim=True).clamp_min(eps)
    return centers[None, None, :, :], bin_width[None, None, :, :]


def soft_histogram(
    x: torch.Tensor,
    centers: torch.Tensor,
    bin_width: torch.Tensor,
    sigma_factor: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    if bin_width.dim() == 5 and bin_width.shape[-1] == 1:
        bin_width = bin_width.squeeze(-1)
    if x.dim() != 3 or centers.dim() != 4 or bin_width.dim() != 4:
        raise RuntimeError("Invalid tensor shape for soft_histogram")
    feature_dims = centers.shape[2]
    x = _to_bcf(x, feature_dims)
    x4 = x.unsqueeze(-1)
    sigma = (bin_width * sigma_factor).clamp_min(eps)
    w = torch.exp(-0.5 * ((x4 - centers) / sigma) ** 2)
    h = w.sum(dim=1)
    h = h / (h.sum(dim=-1, keepdim=True) + eps)
    return h


def binned_reconstruction_loss(
    real: torch.Tensor,
    fake: torch.Tensor,
    feature_dims: int = 36,
    num_bins: int = 32,
    sigma_factor: float = 0.5,
    mode: str = "js",
    eps: float = 1e-8,
) -> torch.Tensor:
    real_bcf = _to_bcf(real, feature_dims)
    fake_bcf = _to_bcf(fake, feature_dims)
    centers, bin_width = _make_featurewise_bin_centers(real_bcf, num_bins=num_bins)
    p = soft_histogram(real_bcf, centers, bin_width, sigma_factor=sigma_factor, eps=eps)
    q = soft_histogram(fake_bcf, centers, bin_width, sigma_factor=sigma_factor, eps=eps)
    mode = mode.lower()
    if mode == "l2":
        return ((p - q) ** 2).mean()
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    if mode == "kl":
        kl = (p * (p.log() - q.log())).sum(dim=-1)
        return kl.mean()
    m = 0.5 * (p + q)
    m = m.clamp_min(eps)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm).mean()


__all__ = [
    "SingleCellDecoder",
    "ViTBackbone",
    "ViTCondDiscriminator",
    "gradient_penalty",
    "binned_reconstruction_loss",
]
