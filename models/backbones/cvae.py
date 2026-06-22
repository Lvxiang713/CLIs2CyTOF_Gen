from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import AttentionBlock, CrossAttentionBlock, ConditioningBlock, norm_layer
from ..common.condition_utils import apply_condition_token_dropout, expand_condition_for_cells
from ...analysis.attention_recording import CrossAttentionRecorder, attach_cross_attention_recorder


class CondSequential(nn.Sequential, ConditioningBlock):
    """Sequential container that forwards conditioning only when needed."""

    def forward(self, x: torch.Tensor, cd: tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ConditioningBlock):
                x = layer(x, cd)
            else:
                x = layer(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block without timestep conditioning for VAE style models."""

    def __init__(self, in_dims: int, out_dims: int, dropout: float) -> None:
        super().__init__()
        self.Linear1 = nn.Sequential(
            norm_layer(in_dims),
            nn.SiLU(),
            nn.Linear(in_dims, out_dims),
        )
        self.Linear2 = nn.Sequential(
            norm_layer(out_dims),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_dims, out_dims),
        )
        self.shortcut = nn.Linear(in_dims, out_dims) if in_dims != out_dims else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.Linear1(x)
        h = self.Linear2(h)
        return h + self.shortcut(x)


class SingleCellEncoder(nn.Module):
    """Conditional encoder that maps x and EHR conditions into mu and logvar."""

    def __init__(
        self,
        feature_dims: int = 36,
        EHR_embdims: int = 128,
        model_dims: int = 512,
        dims_mult=(1, 2, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions=(2, 4, 8, 16),
        dropout: float = 0.0,
        dropoutAtt: float = 0.1,
        num_heads: int = 4,
        latent_dim: int = 128,
        condition_seq_len: int = 1,
        condition_token_keep_all_max: float | None = None,
        condition_token_dropout_enabled: bool = False,
        EHR_fdims: int | None = None,
    ) -> None:
        super().__init__()
        if EHR_fdims is not None:
            condition_seq_len = EHR_fdims
        self.feature_dims = feature_dims
        self.model_dims = model_dims
        self.latent_dim = latent_dim
        self.condition_token_keep_all_max = condition_token_keep_all_max
        self.condition_token_dropout_enabled = condition_token_dropout_enabled

        self.proteinEmb = nn.Sequential(
            nn.Linear(1, model_dims),
            nn.LayerNorm(model_dims),
            nn.SiLU(),
            nn.Linear(model_dims, model_dims),
        )
        self.InitEmb = nn.Sequential(
            nn.Linear(model_dims, model_dims),
            nn.LayerNorm(model_dims),
            nn.SiLU(),
            nn.Linear(model_dims, model_dims),
        )
        self.position_emb = nn.Parameter(torch.zeros(feature_dims, model_dims))
        self.ehr_position_emb = nn.Parameter(torch.zeros(condition_seq_len, EHR_embdims))

        self.down_blocks = nn.ModuleList([CondSequential(nn.Linear(model_dims, model_dims))])
        down_block_dims = [model_dims]
        ch = model_dims
        ds = 1
        for level, mult in enumerate(dims_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_dims, dropout)]
                ch = mult * model_dims
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads, dropout=dropoutAtt))
                    layers.append(CrossAttentionBlock(ch, EHR_embdims, num_heads=num_heads))
                self.down_blocks.append(CondSequential(*layers))
                down_block_dims.append(ch)
            if level != len(dims_mult) - 1:
                ds *= 2

        self.middle_block = CondSequential(
            ResidualBlock(ch, ch, dropout),
            AttentionBlock(ch, num_heads=num_heads, dropout=dropoutAtt),
            CrossAttentionBlock(ch, EHR_embdims, num_heads=num_heads),
            ResidualBlock(ch, ch, dropout),
        )
        self.enc_out_dim = ch
        self.to_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.to_logvar = nn.Linear(self.enc_out_dim, latent_dim)

    def forward(
        self,
        x_sc: torch.Tensor,
        cd: tuple[torch.Tensor, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond_embeds, attention_mask = cd
        batch_size, cell_num, feat_dim = x_sc.shape
        assert feat_dim == self.feature_dims, "feature_dims mismatch in encoder"

        x = x_sc.reshape(batch_size * cell_num, feat_dim).unsqueeze(-1)
        x = self.proteinEmb(x)
        x = self.InitEmb(x + self.position_emb.unsqueeze(0))
        cond_rep, mask_rep = expand_condition_for_cells(cond_embeds, attention_mask, cell_num)
        if cond_rep.size(1) == self.ehr_position_emb.size(0):
            cond_rep = cond_rep + self.ehr_position_emb.unsqueeze(0)
        cond_rep, mask_rep, _ = apply_condition_token_dropout(
            cond_rep,
            mask_rep,
            training=self.training,
            enabled=self.condition_token_dropout_enabled,
            keep_all_max=self.condition_token_keep_all_max,
        )

        h = x
        for module in self.down_blocks:
            h = module(h, (cond_rep, mask_rep))
        h = self.middle_block(h, (cond_rep, mask_rep))

        h = h.view(batch_size, cell_num, feat_dim, self.enc_out_dim)
        h_cell = h.mean(dim=2)
        mu = self.to_mu(h_cell)
        logvar = self.to_logvar(h_cell)
        return mu, logvar


class SingleCellDecoder(nn.Module):
    """Conditional decoder that maps latent z and EHR conditions into x_hat."""

    def __init__(
        self,
        feature_dims: int = 36,
        EHR_embdims: int = 128,
        model_dims: int = 512,
        dims_mult=(1, 2, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions=(2, 4, 8, 16),
        dropout: float = 0.0,
        dropoutAtt: float = 0.1,
        num_heads: int = 4,
        latent_dim: int = 128,
        condition_seq_len: int = 1,
        condition_token_keep_all_max: float | None = None,
        condition_token_dropout_enabled: bool = False,
        EHR_fdims: int | None = None,
    ) -> None:
        super().__init__()
        if EHR_fdims is not None:
            condition_seq_len = EHR_fdims
        self.feature_dims = feature_dims
        self.model_dims = model_dims
        self.latent_dim = latent_dim
        self.condition_token_keep_all_max = condition_token_keep_all_max
        self.condition_token_dropout_enabled = condition_token_dropout_enabled

        self.latent2feat = nn.Linear(latent_dim, feature_dims)
        self.proteinEmb = nn.Sequential(
            nn.Linear(1, model_dims),
            nn.LayerNorm(model_dims),
            nn.SiLU(),
            nn.Linear(model_dims, model_dims),
        )
        self.InitEmb = nn.Sequential(
            nn.Linear(model_dims, model_dims),
            nn.LayerNorm(model_dims),
            nn.SiLU(),
            nn.Linear(model_dims, model_dims),
        )
        self.position_emb = nn.Parameter(torch.zeros(feature_dims, model_dims))
        self.ehr_position_emb = nn.Parameter(torch.zeros(condition_seq_len, EHR_embdims))

        self.down_blocks = nn.ModuleList([CondSequential(nn.Linear(model_dims, model_dims))])
        down_block_dims = [model_dims]
        ch = model_dims
        ds = 1
        for level, mult in enumerate(dims_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_dims, dropout)]
                ch = mult * model_dims
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads, dropout=dropoutAtt))
                    layers.append(CrossAttentionBlock(ch, EHR_embdims, num_heads=num_heads))
                self.down_blocks.append(CondSequential(*layers))
                down_block_dims.append(ch)
            if level != len(dims_mult) - 1:
                ds *= 2

        self.middle_block = CondSequential(
            ResidualBlock(ch, ch, dropout),
            AttentionBlock(ch, num_heads=num_heads, dropout=dropoutAtt),
            CrossAttentionBlock(ch, EHR_embdims, num_heads=num_heads),
            ResidualBlock(ch, ch, dropout),
        )

        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(dims_mult))[::-1]:
            for i in range(num_res_blocks):
                layers = [ResidualBlock(ch + down_block_dims.pop(), model_dims * mult, dropout)]
                ch = model_dims * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads, dropout=dropoutAtt))
                    layers.append(CrossAttentionBlock(ch, EHR_embdims, num_heads=num_heads))
                if level and i == num_res_blocks:
                    ds //= 2
                self.up_blocks.append(CondSequential(*layers))

        layers = [ResidualBlock(ch + down_block_dims.pop(), model_dims * mult, dropout)]
        self.up_blocks.append(CondSequential(*layers))
        self.out = nn.Sequential(
            nn.Linear(model_dims, model_dims),
            norm_layer(model_dims),
            nn.SiLU(),
            nn.Linear(model_dims, model_dims),
            norm_layer(model_dims),
            nn.SiLU(),
            nn.Linear(model_dims, 1),
        )
        self._analysis_attn_recorder = CrossAttentionRecorder(enabled=False)
        attach_cross_attention_recorder(self, self._analysis_attn_recorder)


    def set_cross_attn_recording(self, enabled: bool, clear: bool = True) -> None:
        self._analysis_attn_recorder.enable(enabled=enabled, clear=clear)

    def collect_cross_attn(self, aggregate: str = "mean", clear: bool = True) -> torch.Tensor | None:
        return self._analysis_attn_recorder.collect(aggregate=aggregate, clear=clear)

    def forward(
        self,
        z: torch.Tensor,
        cd: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        cond_embeds, attention_mask = cd
        batch_size, cell_num, latent_dim = z.shape
        assert latent_dim == self.latent_dim
        feat_values = self.latent2feat(z)
        x = feat_values.reshape(batch_size * cell_num, self.feature_dims).unsqueeze(-1)
        x = self.proteinEmb(x)
        x = self.InitEmb(x + self.position_emb.unsqueeze(0))
        cond_rep, mask_rep = expand_condition_for_cells(cond_embeds, attention_mask, cell_num)
        if cond_rep.size(1) == self.ehr_position_emb.size(0):
            cond_rep = cond_rep + self.ehr_position_emb.unsqueeze(0)
        cond_rep, mask_rep, _ = apply_condition_token_dropout(
            cond_rep,
            mask_rep,
            training=self.training,
            enabled=self.condition_token_dropout_enabled,
            keep_all_max=self.condition_token_keep_all_max,
        )

        h = x
        hs = []
        for module in self.down_blocks:
            h = module(h, (cond_rep, mask_rep))
            hs.append(h)
        h = self.middle_block(h, (cond_rep, mask_rep))
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, (cond_rep, mask_rep))
        out = self.out(h).squeeze(-1).view(batch_size, cell_num, self.feature_dims)
        return out


class SingleCellCVAE(nn.Module):
    """Conditional VAE main module."""

    def __init__(
        self,
        feature_dims: int = 36,
        EHR_embdims: int = 128,
        model_dims: int = 512,
        dims_mult=(1, 2, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions=(2, 4, 8, 16),
        dropout: float = 0.0,
        dropoutAtt: float = 0.1,
        num_heads: int = 4,
        latent_dim: int = 128,
        condition_seq_len: int = 1,
        condition_token_keep_all_max: float | None = None,
        condition_token_dropout_enabled: bool = False,
        EHR_fdims: int | None = None,
    ) -> None:
        super().__init__()
        if EHR_fdims is not None:
            condition_seq_len = EHR_fdims
        self.latent_dim = latent_dim
        self.encoder = SingleCellEncoder(
            feature_dims=feature_dims,
            EHR_embdims=EHR_embdims,
            model_dims=model_dims,
            dims_mult=dims_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            dropoutAtt=dropoutAtt,
            num_heads=num_heads,
            latent_dim=latent_dim,
            condition_seq_len=condition_seq_len,
            condition_token_keep_all_max=condition_token_keep_all_max,
            condition_token_dropout_enabled=condition_token_dropout_enabled,
        )
        self.decoder = SingleCellDecoder(
            feature_dims=feature_dims,
            EHR_embdims=EHR_embdims,
            model_dims=model_dims,
            dims_mult=dims_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            dropoutAtt=dropoutAtt,
            num_heads=num_heads,
            latent_dim=latent_dim,
            condition_seq_len=condition_seq_len,
            condition_token_keep_all_max=condition_token_keep_all_max,
            condition_token_dropout_enabled=condition_token_dropout_enabled,
        )


    def set_cross_attn_recording(self, enabled: bool, clear: bool = True) -> None:
        self.decoder.set_cross_attn_recording(enabled=enabled, clear=clear)

    def collect_cross_attn(self, aggregate: str = "mean", clear: bool = True) -> torch.Tensor | None:
        return self.decoder.collect_cross_attn(aggregate=aggregate, clear=clear)

    def encode(self, x_sc: torch.Tensor, cd: tuple[torch.Tensor, torch.Tensor | None]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x_sc, cd)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cd: tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
        return self.decoder(z, cd)

    def forward(self, x_sc: torch.Tensor, cd: tuple[torch.Tensor, torch.Tensor | None]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x_sc, cd)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, cd)
        return x_recon, mu, logvar

    def sample_from_prior(
        self,
        batch_size: int,
        cell_num: int,
        cd: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        z = torch.randn(batch_size, cell_num, self.latent_dim, device=device)
        return self.decode(z, cd)


def vae_loss(
    x_recon: torch.Tensor,
    x_target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard VAE loss."""
    recon_loss = F.mse_loss(x_recon, x_target, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss


__all__ = [
    "CondSequential",
    "ResidualBlock",
    "SingleCellEncoder",
    "SingleCellDecoder",
    "SingleCellCVAE",
    "vae_loss",
]
