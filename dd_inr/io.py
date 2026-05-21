from __future__ import annotations

from pathlib import Path

import numpy as np


def load_case(data_dir: str | Path):
    data_dir = Path(data_dir)
    kspace = np.load(data_dir / "kspace.npy")
    traj = np.load(data_dir / "traj.npy")
    background = np.load(data_dir / "background.npy")
    csm_path = data_dir / "csm.npy"
    csm = np.load(csm_path) if csm_path.exists() else None
    return kspace, traj, background, csm


def flatten_shots(kspace: np.ndarray, traj: np.ndarray):
    """Convert `(T,C,shots,S)` and `(T,shots,S,3)` arrays to `(T,C,N)` and `(T,N,3)`."""
    if kspace.ndim == 4:
        t, c, shots, samples = kspace.shape
        kspace = kspace.reshape(t, c, shots * samples)
    if traj.ndim == 4:
        t, shots, samples, dim = traj.shape
        traj = traj.reshape(t, shots * samples, dim)
    if kspace.ndim != 3:
        raise ValueError(f"Expected kspace shape (T,C,N) after flattening, got {kspace.shape}")
    if traj.ndim != 3 or traj.shape[-1] != 3:
        raise ValueError(f"Expected traj shape (T,N,3) after flattening, got {traj.shape}")
    if kspace.shape[0] != traj.shape[0]:
        raise ValueError("kspace and traj frame counts differ")
    return kspace, traj


def background_to_complex(background: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(background):
        return background.astype(np.complex64)
    if background.shape[-1] != 2:
        raise ValueError("Real-valued background must have final real/imag channel of size 2")
    return (background[..., 0] + 1j * background[..., 1]).astype(np.complex64)


def ensure_output_dir(path: str | Path) -> Path:
    path = Path(path)
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (path / "images").mkdir(parents=True, exist_ok=True)
    return path

