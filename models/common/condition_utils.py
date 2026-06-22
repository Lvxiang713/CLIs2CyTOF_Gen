from __future__ import annotations

import torch


def expand_condition_for_cells(
    cond_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cell_num: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Expand donor level condition tokens to per cell condition tokens."""
    batch_size, seq_len, emb_dim = cond_embeds.shape
    cond_rep = (
        cond_embeds.unsqueeze(1)
        .expand(-1, cell_num, -1, -1)
        .reshape(batch_size * cell_num, seq_len, emb_dim)
    )
    if attention_mask is None:
        return cond_rep, None
    if attention_mask.dim() == 3 and attention_mask.size(-1) == 1:
        attention_mask = attention_mask.squeeze(-1)
    mask_rep = (
        attention_mask.unsqueeze(1)
        .expand(-1, cell_num, -1)
        .reshape(batch_size * cell_num, seq_len)
    )
    return cond_rep, mask_rep


def keep_prob_from_tau(tau: float | None, seq_len: int) -> float:
    """Convert a cap on P(all tokens kept) into per token keep probability."""
    if tau is None:
        return 1.0
    tau = float(tau)
    if tau >= 1.0:
        return 1.0
    if tau <= 0.0:
        return 0.0
    return tau ** (1.0 / float(seq_len))


def apply_condition_token_dropout(
    cond_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    training: bool,
    enabled: bool,
    keep_all_max: float | None,
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    """Randomly mask condition tokens during training.

    Parameters
    ----------
    cond_embeds:
        Tensor of shape (N, L, D).
    attention_mask:
        Bool like mask of shape (N, L). True means valid token.
    training:
        Whether the caller is in training mode.
    enabled:
        Global switch for token dropout.
    keep_all_max:
        Upper bound for the probability that all tokens remain visible.
    """
    if (not training) or (not enabled) or (keep_all_max is None):
        return cond_embeds, attention_mask, 1.0

    num_batch, seq_len, _ = cond_embeds.shape
    keep_prob = keep_prob_from_tau(keep_all_max, seq_len)

    if attention_mask is None:
        base_mask = torch.ones(num_batch, seq_len, device=cond_embeds.device, dtype=torch.bool)
        mask_dtype = None
    else:
        if attention_mask.dim() == 3 and attention_mask.size(-1) == 1:
            attention_mask = attention_mask.squeeze(-1)
        base_mask = attention_mask.to(torch.bool)
        mask_dtype = attention_mask.dtype

    sampled_keep = torch.rand(num_batch, seq_len, device=cond_embeds.device) < keep_prob
    final_keep = sampled_keep & base_mask
    cond_embeds = cond_embeds * final_keep.unsqueeze(-1).to(cond_embeds.dtype)

    if mask_dtype is None:
        new_mask = final_keep
    else:
        new_mask = final_keep.to(mask_dtype)
    return cond_embeds, new_mask, keep_prob


def masked_mean_pool(cond_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Pool token conditions into a single vector with a valid token mask."""
    if attention_mask is None:
        return cond_embeds.mean(dim=1)

    if attention_mask.dim() == 3 and attention_mask.size(-1) == 1:
        attention_mask = attention_mask.squeeze(-1)
    weights = attention_mask.to(cond_embeds.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (cond_embeds * weights).sum(dim=1) / denom


__all__ = [
    "expand_condition_for_cells",
    "keep_prob_from_tau",
    "apply_condition_token_dropout",
    "masked_mean_pool",
]
