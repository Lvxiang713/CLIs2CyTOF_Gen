from __future__ import annotations

import csv

import numpy as np
import torch
from torch.utils.data import Dataset


class SingleCellProteinDataset(Dataset):
    """Dataset for sample level single cell protein matrices.

    The CSV is expected to store the sample identifier in the first column and
    all remaining columns as numeric features for each cell.
    """

    def __init__(self, csv_path: str, cell_num_per_sample: int = 10, mode: str = 'train') -> None:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)

        data = np.loadtxt(csv_path, delimiter=',', skiprows=1, dtype=str)
        self.samples = np.unique(data[:, 0])
        self.data = data[:, 1:].astype(np.float32)
        self.cell_num_per_sample = cell_num_per_sample
        self.mode = mode

        self.sample_to_indices: dict[str, np.ndarray] = {}
        for i, sample in enumerate(data[:, 0]):
            self.sample_to_indices.setdefault(sample, []).append(i)
        for sample in self.sample_to_indices:
            self.sample_to_indices[sample] = np.asarray(self.sample_to_indices[sample])

        self.sample_to_probs = {
            sample: np.ones(len(indices), dtype=np.float64) / len(indices)
            for sample, indices in self.sample_to_indices.items()
        }
        self.indices = list(range(len(self.samples)))

    def set_mode(self, mode: str, cell_num_per_sample: int | None = None) -> None:
        """Switch dataset behavior between train, val, and test modes."""
        self.mode = mode
        if mode in ['val', 'test']:
            self.cell_num_per_sample = None
        elif mode == 'train' and cell_num_per_sample is not None:
            self.cell_num_per_sample = cell_num_per_sample

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample = self.samples[self.indices[idx]]
        indices = self.sample_to_indices[sample]
        probs = self.sample_to_probs[sample]

        if self.cell_num_per_sample is not None:
            replace = len(indices) < self.cell_num_per_sample
            sampled_indices = np.random.choice(
                indices,
                size=self.cell_num_per_sample,
                replace=replace,
                p=probs,
            )
            sampled_data = self.data[sampled_indices]
        else:
            sampled_data = self.data[indices]

        return sample, torch.tensor(sampled_data, dtype=torch.float32)


__all__ = ['SingleCellProteinDataset']
