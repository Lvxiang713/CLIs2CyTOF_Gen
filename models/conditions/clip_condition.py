from __future__ import annotations

import json
from typing import Iterable

import torch

from singlecell_generative_unified.models.common.condition_utils import masked_mean_pool
from singlecell_generative_unified.models.conditions.clip_backbone import CLIP


class ClipConditionEncoder:
    """Wrap the frozen CLIP text encoder used for EHR conditioning."""

    def __init__(
        self,
        cfg_path: str,
        ckpt_path: str,
        device: torch.device,
        condition_mode: str = "cls",
    ) -> None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if condition_mode not in {"cls", "all_tokens", "feature_tokens"}:
            raise ValueError(f"Unsupported condition_mode: {condition_mode}")
        self.cfg = cfg
        self.device = device
        self.condition_mode = condition_mode
        self.model = CLIP(**cfg).to(device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def output_dim(self) -> int:
        return int(self.cfg["out_dim"])

    @property
    def condition_dim(self) -> int:
        if self.condition_mode == "cls":
            return int(self.cfg["out_dim"])
        return int(self.cfg["transformer_width"])

    @property
    def condition_seq_len(self) -> int:
        ehr_feature_dim = int(self.cfg["ehr_feature_dim"])
        if self.condition_mode == "cls":
            return 1
        if self.condition_mode == "all_tokens":
            return 1 + ehr_feature_dim
        return ehr_feature_dim

    @staticmethod
    def pool_condition(cond: tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
        cond_embeds, cond_mask = cond
        return masked_mean_pool(cond_embeds, cond_mask)

    @torch.no_grad()
    def encode_batch(self, ehr_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of EHR vectors into condition tokens and masks."""
        ehr_data = ehr_data.to(self.device)
        if self.condition_mode == "cls":
            text_emb = self.model.encode_text(ehr_data)
            text_emb = text_emb / text_emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cond_embeds = text_emb.unsqueeze(1)
            cond_mask = torch.ones(cond_embeds.size(0), 1, dtype=torch.bool, device=self.device)
            return cond_embeds, cond_mask
        if self.condition_mode == "all_tokens":
            return self.model.encode_text_tokens(ehr_data, use_cls_token=True)
        return self.model.encode_text_tokens(ehr_data, use_cls_token=False)

    @torch.no_grad()
    def build_donor_condition_dict(
        self,
        dataset,
        donor_ids: Iterable[str],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Precompute donor level condition tensors for sampling."""
        donor_condition: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for donor_id in donor_ids:
            ehr = torch.as_tensor(dataset.ehr_dict[donor_id], dtype=torch.float32, device=self.device).unsqueeze(0)
            cond_embeds, cond_mask = self.encode_batch(ehr)
            donor_condition[donor_id] = (
                cond_embeds.detach().cpu().clone(),
                cond_mask.detach().cpu().clone(),
            )
        return donor_condition


__all__ = ["ClipConditionEncoder"]
