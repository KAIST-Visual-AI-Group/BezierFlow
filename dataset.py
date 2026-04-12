"""
Dataset utilities for BézierFlow distillation.
"""

from typing import List, Optional, Tuple
import os
import torch
from torch.utils.data import Dataset


def load_data_from_dir(
    data_folder: str, limit: int = 200
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
    """Load pre-generated (latent, image) pairs from .pt files."""
    latents, targets, conditions, unconditions = [], [], [], []
    pt_files = [f for f in os.listdir(data_folder) if f.endswith('pt')]
    for file_name in sorted(pt_files)[:limit]:
        file_path = os.path.join(data_folder, file_name)
        data = torch.load(file_path)
        latents.append(data["latent"])
        targets.append(data["img"])
        conditions.append(data.get("c", None))
        unconditions.append(data.get("uc", None))
    return latents, targets, conditions, unconditions


class BFDataset(Dataset):
    """Dataset storing latents, targets, and conditions for BézierFlow distillation."""

    def __init__(
        self,
        latent: List[torch.Tensor],
        target: List[torch.Tensor],
        condition: List[Optional[torch.Tensor]],
        uncondition: List[Optional[torch.Tensor]],
    ):
        self.latent = latent
        self.target = target
        self.condition = condition
        self.uncondition = uncondition

    def __len__(self) -> int:
        return len(self.latent)

    def __getitem__(self, idx: int):
        return (self.target[idx], self.latent[idx],
                self.condition[idx], self.uncondition[idx])
