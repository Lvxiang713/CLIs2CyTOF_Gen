from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EHRSingleCellDataset(Dataset):
    """Dataset that pairs donor level EHR features with grouped single cell data.

    Expected CSV formats:
    - EHR CSV: first column named ``donor_ID``, remaining columns are numeric EHR features.
    - Single cell CSV: first column named ``sample`` with values such as ``HPAP-167-0``.
      The donor ID is parsed from the first two dash separated parts and the final
      part is treated as the group index.

    Each item returns:
    - sc_data: tensor with shape ``(cell_num, sc_feature_dim)``
    - ehr_data: tensor with shape ``(ehr_feature_dim,)``
    - label: donor class index
    - donor_id: original donor identifier string
    """

    def __init__(self, ehr_csv_path: str, sc_csv_path: str) -> None:
        super().__init__()

        ehr_df = pd.read_csv(ehr_csv_path)
        self.ehr_dict: dict[str, np.ndarray] = {}
        donor_list: list[str] = []
        for _, row in ehr_df.iterrows():
            donor_id = row['donor_ID']
            features = row.drop(labels=['donor_ID']).to_numpy(dtype=np.float32)
            self.ehr_dict[donor_id] = features
            donor_list.append(donor_id)

        unique_donors = sorted(set(donor_list))
        self.donor2label = {donor: i for i, donor in enumerate(unique_donors)}

        sc_df = pd.read_csv(sc_csv_path)
        self.sc_dict: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
        for sample_id, group_df in sc_df.groupby('sample'):
            parts = str(sample_id).split('-')
            donor_id = '-'.join(parts[:2])
            group_idx = int(parts[2]) if len(parts) > 2 else 0
            sc_features = group_df.drop(['sample'], axis=1).to_numpy(dtype=np.float32)
            self.sc_dict[donor_id][group_idx] = sc_features

        self.samples: list[tuple[str, int, int]] = []
        for donor_id, groups in self.sc_dict.items():
            if donor_id not in self.ehr_dict:
                continue
            for group_idx in groups.keys():
                label = self.donor2label[donor_id]
                self.samples.append((donor_id, group_idx, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        donor_id, group_idx, label = self.samples[idx]
        ehr_data = torch.from_numpy(self.ehr_dict[donor_id])
        sc_data = torch.from_numpy(self.sc_dict[donor_id][group_idx])
        label_tensor = torch.tensor(label, dtype=torch.long)
        return sc_data, ehr_data, label_tensor, donor_id


__all__ = ['EHRSingleCellDataset']
