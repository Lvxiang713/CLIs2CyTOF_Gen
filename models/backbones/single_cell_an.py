from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..common.blocks import (
    AttentionBlock,
    CrossAttentionBlock,
    ResidualBlock,
    TimestepEmbedSequential,
    norm_layer,
)
from ..common.condition_utils import apply_condition_token_dropout, expand_condition_for_cells
from ..common.time_embedding import timestep_embedding
from ...analysis.attention_recording import CrossAttentionRecorder, attach_cross_attention_recorder


class SingleCellAN(nn.Module):
    """Conditional time aware backbone.

    The internal attribute names intentionally match the original
    FM based implementation so that previously trained checkpoints can
    still be loaded with strict=False.
    """

    def __init__(
        self,
        feature_dims: int = 36,
        ehr_emb_dims: int = 128,
        model_dims: int = 512,
        dims_mult: Sequence[int] = (1, 2, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (2, 4, 8, 16),
        dropout: float = 0.0,
        dropout_attn: float = 0.1,
        num_heads: int = 4,
        condition_seq_len: int = 1,
        condition_token_keep_all_max: float | None = None,
        condition_token_dropout_enabled: bool = False,
        EHR_embdims: int | None = None,
        EHR_fdims: int | None = None,
    ) -> None:
        super().__init__()
        if EHR_embdims is not None:
            ehr_emb_dims = EHR_embdims
        if EHR_fdims is not None:
            condition_seq_len = EHR_fdims

        self.feature_dims = feature_dims
        self.model_dims = model_dims
        self.condition_seq_len = condition_seq_len
        self.condition_token_keep_all_max = condition_token_keep_all_max
        self.condition_token_dropout_enabled = condition_token_dropout_enabled
        time_embed_dim = model_dims * 4

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
        self.ehr_position_emb = nn.Parameter(torch.zeros(condition_seq_len, ehr_emb_dims))

        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Linear(model_dims, model_dims))
        ])

        self.time_embed = nn.Sequential(
            nn.Linear(model_dims, time_embed_dim),
            nn.LayerNorm(time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        down_block_dims = [model_dims]
        ch = model_dims
        ds = 1

        for level, mult in enumerate(dims_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [ResidualBlock(ch, mult * model_dims, time_embed_dim, dropout)]
                ch = mult * model_dims
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads, dropout=dropout_attn))
                    layers.append(CrossAttentionBlock(ch, ehr_emb_dims, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_dims.append(ch)

            if level != len(dims_mult) - 1:
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads, dropout=dropout_attn),
            CrossAttentionBlock(ch, ehr_emb_dims, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout),
        )

        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(dims_mult))[::-1]:
            for i in range(num_res_blocks):
                layers = [
                    ResidualBlock(
                        ch + down_block_dims.pop(),
                        model_dims * mult,
                        time_embed_dim,
                        dropout,
                    )
                ]
                ch = model_dims * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads, dropout=dropout_attn))
                    layers.append(CrossAttentionBlock(ch, ehr_emb_dims, num_heads=num_heads))
                if level and i == num_res_blocks:
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        layers = [
            ResidualBlock(
                ch + down_block_dims.pop(),
                model_dims * mult,
                time_embed_dim,
                dropout,
            )
        ]
        self.up_blocks.append(TimestepEmbedSequential(*layers))

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
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        batch_size, cell_num, _ = x.shape
        cond_embeds, attention_mask = cond

        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_dims))
        time_emb = time_emb.unsqueeze(1).expand(-1, cell_num, -1).reshape(batch_size * cell_num, -1)

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

        h = x.reshape(batch_size * cell_num, -1).unsqueeze(-1)
        h = self.proteinEmb(h)
        h = self.InitEmb(h + self.position_emb.unsqueeze(0))

        hs: list[torch.Tensor] = []
        for module in self.down_blocks:
            h = module(h, time_emb, (cond_rep, mask_rep))
            hs.append(h)

        h = self.middle_block(h, time_emb, (cond_rep, mask_rep))

        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, time_emb, (cond_rep, mask_rep))

        out = self.out(h).squeeze(-1)
        return out.reshape(batch_size, cell_num, -1)


__all__ = ["SingleCellAN"]
