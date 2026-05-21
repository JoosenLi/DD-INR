from __future__ import annotations

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler


def create_spatiotemporal_grid_3dt(num_t: int, num_x: int, num_y: int, num_z: int):
    """Return normalized spatial coordinates `(X,Y,Z,3)` and temporal coordinates `(T,1)`."""
    coords_t = torch.linspace(0.0, 1.0, steps=num_t).unsqueeze(-1)
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.linspace(0.0, 1.0, steps=num_x),
        torch.linspace(0.0, 1.0, steps=num_y),
        torch.linspace(0.0, 1.0, steps=num_z),
        indexing="ij",
    )
    coords_spatial = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    return coords_spatial, coords_t


class FrameNufftDataset(Dataset):
    """One sample is one fMRI frame: k-space, NUFFT operator, time coordinate, frame index."""

    def __init__(self, kspace: torch.Tensor, nufft_operators: list, t_coords: torch.Tensor):
        if kspace.shape[0] != len(nufft_operators):
            raise ValueError(f"kspace has {kspace.shape[0]} frames but got {len(nufft_operators)} NUFFT operators")
        if t_coords.shape[0] != kspace.shape[0]:
            raise ValueError("t_coords length must match kspace frame count")
        self.kspace = kspace
        self.nufft_operators = nufft_operators
        self.t_coords = t_coords

    def __len__(self) -> int:
        return int(self.kspace.shape[0])

    def __getitem__(self, frame_idx: int):
        return self.kspace[frame_idx], self.nufft_operators[frame_idx], self.t_coords[frame_idx], frame_idx


def nufft_collate_fn(batch):
    kspace, ops, t_coords, indices = zip(*batch)
    return torch.stack(kspace), list(ops), torch.stack(t_coords), torch.tensor(indices, dtype=torch.long)


def make_frame_loader(dataset: FrameNufftDataset, batch_size: int, num_workers: int = 0, pin_memory: bool = True):
    if num_workers != 0:
        raise ValueError("FrameNufftDataset stores Python/CUDA NUFFT operator objects, so num_workers must be 0")
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=int(batch_size), drop_last=False)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=nufft_collate_fn,
        num_workers=0,
        pin_memory=pin_memory,
    )

