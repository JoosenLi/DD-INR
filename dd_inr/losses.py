from __future__ import annotations

import torch


def complex_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target) ** 2)


def charbonnier(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(x * x + eps)


def lp_penalty(x: torch.Tensor, p: float, eps: float = 1e-12) -> torch.Tensor:
    return torch.pow(torch.abs(x) + eps, p)


def l2_grad3d(u: torch.Tensor) -> torch.Tensor:
    """Squared finite-difference gradient for `(B,X,Y,Z)` tensors."""
    dx = u[:, 1:, :, :] - u[:, :-1, :, :]
    dy = u[:, :, 1:, :] - u[:, :, :-1, :]
    dz = u[:, :, :, 1:] - u[:, :, :, :-1]
    return (dx * dx).mean() + (dy * dy).mean() + (dz * dz).mean()


def tv3d(u: torch.Tensor, eps: float = 1e-12, isotropic: bool = True) -> torch.Tensor:
    """Spatial total variation for `(B,X,Y,Z)` tensors."""
    dx = u[:, 1:, :, :] - u[:, :-1, :, :]
    dy = u[:, :, 1:, :] - u[:, :, :-1, :]
    dz = u[:, :, :, 1:] - u[:, :, :, :-1]

    if isotropic:
        dx = dx[:, :, :-1, :-1]
        dy = dy[:, :-1, :, :-1]
        dz = dz[:, :-1, :-1, :]
        return torch.sqrt(dx * dx + dy * dy + dz * dz + eps).mean()
    return (dx.abs().mean() + dy.abs().mean() + dz.abs().mean()) / 3.0


def ema_update(ema_val: float | None, x: float, beta: float = 0.98) -> float:
    x = float(x)
    return x if ema_val is None else beta * ema_val + (1.0 - beta) * x


def ramp_factor(epoch: int, start: int, ramp: int) -> float:
    if epoch < start:
        return 0.0
    if ramp <= 0:
        return 1.0
    return float(min(1.0, (epoch - start) / float(ramp)))


def infer_time_scale_to_index(tcoord_t: torch.Tensor, num_t: int, eps: float = 1e-6) -> float:
    tmin = float(tcoord_t.min().item())
    tmax = float(tcoord_t.max().item())
    if abs(tmin - 0.0) < eps and abs(tmax - 1.0) < eps:
        return 1.0 / max(num_t - 1, 1)
    if abs(tmin + 1.0) < eps and abs(tmax - 1.0) < eps:
        return 2.0 / max(num_t - 1, 1)
    return 1.0


def safe_scalar(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def sanitize_loss(x: torch.Tensor | None, device: torch.device, name: str = "loss"):
    if x is None:
        return torch.tensor(0.0, device=device), True
    if not torch.isfinite(x).all():
        print(f"[WARN] {name} is NaN/Inf; replacing with zero and skipping its EMA update.")
        return torch.zeros((), device=device, dtype=x.dtype), True
    return x, False

