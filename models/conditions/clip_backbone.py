from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class LayerNorm(nn.LayerNorm):
    """LayerNorm that safely handles fp16 inputs by normalizing in fp32."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.to(torch.float32))
        return ret.to(orig_type)


class QuickGELU(nn.Module):
    """Approximate GELU activation used by CLIP style transformers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Transformer residual block with pre norm attention and MLP."""

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.attn(x, x, x, need_weights=False, key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Minimal transformer stack used by the EHR text encoder."""

    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return x


class ViTModel(nn.Module):
    """Vision branch that converts grouped single cell matrices into embeddings."""

    def __init__(
        self,
        out_dim: int,
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
        concat_CNNlayer: list[int] | tuple[int, ...] = (2, 4),
        hidden_channels: int = 32,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
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

        cnn_layers: list[nn.Module] = []
        current_channels = numFeas
        for _ in range(num_CNN_layers):
            cnn_layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            cnn_layers.append(nn.InstanceNorm2d(hidden_channels))
            cnn_layers.append(nn.ReLU())
            current_channels = hidden_channels
        self.cnnList = nn.ModuleList(cnn_layers)

        self.quantileLen = quantileLen
        self.noise_std = noise_std
        self.cut_len = cut_len
        self.cut_stepsize = cut_stepsize
        self.concat_CNNlayer = list(concat_CNNlayer)
        self.q = torch.linspace(0.01, 0.99, quantileLen)

        num_patches_per_axis = 1 + (quantileLen - cut_len) // cut_stepsize
        total_patches = num_patches_per_axis ** 2
        self.patch_embed = nn.Linear(hidden_channels * cut_len ** 2, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + total_patches, embed_dim))
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
        self.head = nn.Linear(embed_dim, out_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
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
        for cnnlayer in self.cnnList:
            x_q_2d = cnnlayer(x_q_2d)
            if isinstance(cnnlayer, nn.Conv2d):
                cnn_middle_x_list.append(x_q_2d)

        x_q_1d_list = []
        for feat_map in cnn_middle_x_list:
            feat_map = feat_map.permute(0, 2, 3, 1)
            feat_map = feat_map.unfold(1, self.cut_len, self.cut_stepsize).unfold(2, self.cut_len, self.cut_stepsize)
            feat_map = feat_map.flatten(-3)
            l0 = feat_map.size(1)
            feat_map = feat_map.reshape(n, l0 * l0, -1)
            x_q_1d_list.append(feat_map)

        x_q_1d = torch.cat([i for idx, i in enumerate(x_q_1d_list) if idx + 1 in self.concat_CNNlayer], dim=1)
        x_tokens = self.patch_embed(x_q_1d)
        cls_tokens = self.cls_token.expand(n, -1, -1)
        x_tokens = torch.cat((cls_tokens, x_tokens), dim=1)
        x_tokens = self.pos_drop(x_tokens)
        x_tokens = self.transformer_encoder(x_tokens)
        cls_out = self.norm(x_tokens[:, 0])
        return self.head(cls_out)


class CLIP(nn.Module):
    """CLIP style dual encoder for single cell data and EHR data."""

    def __init__(
        self,
        out_dim: int,
        width: int,
        ScVision_heads: int,
        ScVitquantileLen: int,
        ScCutLen: int,
        ScCutStep: int,
        ScCNN_depth: int,
        ScCNN_used_depth: list[int],
        ScCNN_hidden_channel: int,
        ScCNN_kernelSize: int,
        vision_inchannel: int,
        ehr_feature_dim: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ) -> None:
        super().__init__()
        self.ehr_feature_dim = ehr_feature_dim
        self.transformer_width = transformer_width
        self.visual = ViTModel(
            out_dim=out_dim,
            numFeas=vision_inchannel,
            quantileLen=ScVitquantileLen,
            embed_dim=width,
            depth=ScCNN_depth,
            num_heads=ScVision_heads,
            mlp_ratio=4.0,
            dropout=0.1,
            cut_len=ScCutLen,
            cut_stepsize=ScCutStep,
            num_CNN_layers=ScCNN_depth,
            concat_CNNlayer=ScCNN_used_depth,
            hidden_channels=ScCNN_hidden_channel,
            kernel_size=ScCNN_kernelSize,
        )
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
        )
        self.ehr_projection = nn.Linear(1, transformer_width)
        self.ehr_cls_token = nn.Parameter(torch.zeros(1, 1, transformer_width))
        self.ehr_pos_embed = nn.Parameter(torch.zeros(1, 1 + ehr_feature_dim, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, out_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    @property
    def dtype(self) -> torch.dtype:
        return self.text_projection.dtype

    def initialize_parameters(self) -> None:
        nn.init.trunc_normal_(self.ehr_cls_token, std=0.02)
        nn.init.trunc_normal_(self.ehr_pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.ehr_projection.weight)
        if self.ehr_projection.bias is not None:
            nn.init.constant_(self.ehr_projection.bias, 0)
        nn.init.xavier_uniform_(self.text_projection)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image.to(self.dtype))

    def _encode_ehr_tokens(self, ehr_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ehr_data = ehr_data.view(ehr_data.size(0), -1)
        batch_size, _ = ehr_data.shape
        device = ehr_data.device

        missing_mask = torch.isnan(ehr_data)
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([cls_mask, missing_mask], dim=1)

        ehr_data_clean = torch.where(missing_mask, torch.zeros_like(ehr_data), ehr_data)
        x = self.ehr_projection(ehr_data_clean.unsqueeze(-1))
        cls_tokens = self.ehr_cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.ehr_pos_embed
        x = self.transformer(x, key_padding_mask=key_padding_mask)
        x = self.ln_final(x)
        attn_mask = ~key_padding_mask
        return x, attn_mask

    def encode_text(self, ehr_data: torch.Tensor) -> torch.Tensor:
        x, _ = self._encode_ehr_tokens(ehr_data)
        x = x[:, 0, :]
        x = x @ self.text_projection
        return x

    def encode_text_tokens(
        self,
        ehr_data: torch.Tensor,
        use_cls_token: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, attn_mask = self._encode_ehr_tokens(ehr_data)
        if use_cls_token:
            return x, attn_mask
        return x[:, 1:, :], attn_mask[:, 1:]

    def forward(self, image: torch.Tensor, ehr_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(image)
        text_features = self.encode_text(ehr_data)
        image_features = image_features / image_features.norm(dim=1, keepdim=True).clamp_min(1e-12)
        text_features = text_features / text_features.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return image_features, text_features


__all__ = [
    'LayerNorm',
    'QuickGELU',
    'ResidualAttentionBlock',
    'Transformer',
    'ViTModel',
    'CLIP',
]
