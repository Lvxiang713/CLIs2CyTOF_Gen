from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from singlecell_generative_unified.data.ehr_singlecell_dataset import EHRSingleCellDataset


@dataclass
class DatasetBundle:
    """Container returned by the data builder."""

    dataset: EHRSingleCellDataset
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    train_donor_ids: list[str]
    val_donor_ids: list[str]
    test_donor_ids: list[str]


def set_global_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _unique_donor_ids(dataset: EHRSingleCellDataset, indices: list[int]) -> list[str]:
    donor_ids = []
    seen = set()
    for idx in indices:
        donor_id = dataset.samples[idx][0]
        if donor_id not in seen:
            seen.add(donor_id)
            donor_ids.append(donor_id)
    return donor_ids


def build_datasets(
    ehr_csv: str,
    sc_csv: str,
    train_val_limit: int = 1400,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> DatasetBundle:
    """Build train, validation, and test subsets using absolute dataset indices."""
    dataset = EHRSingleCellDataset(ehr_csv, sc_csv)
    total_len = len(dataset)

    train_val_limit = min(train_val_limit, total_len)
    base_indices = list(range(train_val_limit))
    test_indices = list(range(train_val_limit, total_len))

    rng = random.Random(seed)
    rng.shuffle(base_indices)

    train_len = int(len(base_indices) * train_ratio)
    train_indices = sorted(base_indices[:train_len])
    val_indices = sorted(base_indices[train_len:])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return DatasetBundle(
        dataset=dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_donor_ids=_unique_donor_ids(dataset, train_indices),
        val_donor_ids=_unique_donor_ids(dataset, val_indices),
        test_donor_ids=_unique_donor_ids(dataset, test_indices),
    )


def build_dataloaders(
    bundle: DatasetBundle,
    batch_size: int,
    world_size: int,
    rank: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build distributed data loaders for train, validation, and test subsets."""
    train_sampler = DistributedSampler(
        bundle.train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    val_sampler = DistributedSampler(
        bundle.val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    test_sampler = DistributedSampler(
        bundle.test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        bundle.val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        bundle.test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


__all__ = [
    "DatasetBundle",
    "set_global_seed",
    "build_datasets",
    "build_dataloaders",
]
