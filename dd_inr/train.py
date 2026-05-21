from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .data import FrameNufftDataset, create_spatiotemporal_grid_3dt, make_frame_loader
from .io import background_to_complex, ensure_output_dir
from .losses import (
    charbonnier,
    complex_mse_loss,
    ema_update,
    infer_time_scale_to_index,
    l2_grad3d,
    lp_penalty,
    ramp_factor,
    safe_scalar,
    sanitize_loss,
    tv3d,
)
from .models import SIREN, SpatioTemporalEncoder
from .nufft import apply_nufft, build_frame_operators


@dataclass
class ReconstructionResult:
    output_dir: Path
    last_checkpoint: Path | None
    last_reconstruction: Path | None


def _grad_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.data.norm(norm_type).item() ** norm_type)
    return total ** (1.0 / norm_type)


def _maybe_writer(log_dir: Path):
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        return None
    return SummaryWriter(str(log_dir))


def _enabled(reg_cfg: dict, key: str) -> bool:
    return bool(reg_cfg.get(key, {}).get("enable", False))


def reconstruct_dynamic_3d(
    kspaces: np.ndarray,
    trajs: np.ndarray,
    background: np.ndarray,
    config: dict,
    case_name: str,
    csm=None,
) -> ReconstructionResult:
    """Run DD-INR reconstruction for preloaded 3D+T non-Cartesian k-space data."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cudnn.benchmark = True

    output_root = Path(config.get("output_path", "outputs"))
    run_name = config.get("exp_idx", "dd_inr")
    output_dir = ensure_output_dir(output_root / case_name / run_name)
    writer = _maybe_writer(output_dir / "logs")

    num_t, num_x, num_y, num_z, _ = config["img_size"]
    n_coils = int(kspaces.shape[1])

    model = SIREN(config["net"]).to(device).train()
    encoder = SpatioTemporalEncoder(config["SpatioTemporal_Encoder"]).to(device)
    encoder.train(bool(config["SpatioTemporal_Encoder"].get("learnable_temporal", True)))

    params = list(model.parameters()) + [p for p in encoder.parameters() if p.requires_grad]
    optim = torch.optim.Adam(
        params,
        lr=float(config["lr"]),
        betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    num_epochs = int(config["num_epoch"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs, eta_min=float(config.get("eta_min", 1e-7)))

    coords_spatial, coords_temporal = create_spatiotemporal_grid_3dt(num_t, num_x, num_y, num_z)
    coords_spatial = coords_spatial.to(device)
    coords_temporal_t = coords_temporal.squeeze(-1).to(torch.float32)
    spatial_features = encoder.spatial_encode(coords_spatial).unsqueeze(0)

    kspaces_t = torch.as_tensor(kspaces, dtype=torch.complex64)
    background_complex = torch.as_tensor(background_to_complex(background), dtype=torch.complex64, device=device)
    if not bool(config.get("dynamics", True)):
        background_complex = torch.zeros_like(background_complex)

    csm_for_ops = csm
    if isinstance(csm_for_ops, np.ndarray):
        csm_for_ops = torch.as_tensor(csm_for_ops, dtype=torch.complex64, device=device)

    ops = build_frame_operators(
        trajs,
        shape=(num_x, num_y, num_z),
        n_coils=n_coils,
        csm=csm_for_ops,
        backend=str(config.get("nufft_backend", "cufinufft")),
        density=bool(config.get("density", True)),
    )
    dataset = FrameNufftDataset(kspaces_t, ops, coords_temporal_t)
    recon_buffer = torch.zeros((num_t, num_x, num_y, num_z, 2), dtype=torch.float32)

    reg_cfg = config.get("reg", {})
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    time_scale = infer_time_scale_to_index(coords_temporal_t, num_t) if bool(reg_cfg.get("time_scale_to_index", True)) else 1.0

    warm = {key: int(reg_cfg.get(key, {}).get("warm", 100)) for key in ["dt", "tv_xyz", "tvxyz_dt", "dtt", "sparse_dyn"]}
    ramp = {key: int(reg_cfg.get(key, {}).get("ramp", 100)) for key in ["dt", "tv_xyz", "tvxyz_dt", "dtt", "sparse_dyn"]}
    alpha = {
        "dt": float(reg_cfg.get("dt", {}).get("alpha", 0.01)),
        "tv_xyz": float(reg_cfg.get("tv_xyz", {}).get("alpha", 0.01)),
        "tvxyz_dt": float(reg_cfg.get("tvxyz_dt", {}).get("alpha", 0.002)),
        "dtt": float(reg_cfg.get("dtt", {}).get("alpha", 0.002)),
        "sparse_dyn": float(reg_cfg.get("sparse_dyn", {}).get("alpha", 0.002)),
    }
    enabled = {key: _enabled(reg_cfg, key) for key in warm}
    ema_vals = {"k": None, **{key: None for key in warm}}
    lam_target = {key: None for key in warm}

    batch_size = int(config["batch_size"])
    log_interval = int(config.get("log_interval", 20))
    save_interval = int(config.get("save_interval", 50))
    grad_clip = float(config.get("grad_clip", 0.0))
    last_checkpoint = None
    last_reconstruction = None

    for epoch in range(num_epochs):
        loader = make_frame_loader(dataset, batch_size=batch_size, num_workers=0)
        sums = {key: 0.0 for key in ["loss", "k", "dt", "tv_xyz", "tvxyz_dt", "dtt", "sparse_dyn"]}
        n_batches = 0
        last_lams = {key: 0.0 for key in warm}

        for kspace_batch, op_batch, t_coord, t_index in loader:
            n_batches += 1
            kspace_batch = kspace_batch.to(device, non_blocking=True)
            t_coord = t_coord.to(device, non_blocking=True)
            batch = int(t_index.numel())
            optim.zero_grad(set_to_none=True)

            need_t_field = any(enabled[key] and epoch >= warm[key] for key in ["dt", "tvxyz_dt", "dtt"])
            with torch.amp.autocast("cuda", enabled=use_amp):
                if need_t_field:
                    t_field = t_coord.view(batch, 1, 1, 1, 1).expand(batch, num_x, num_y, num_z, 1).clone().detach()
                    t_field.requires_grad_(True)
                    temporal_features = encoder.temporal_encode(t_field)
                else:
                    t_field = None
                    temporal_features = encoder.temporal_encode(t_coord.unsqueeze(-1))

                spatial = spatial_features.expand(batch, -1, -1, -1, -1)
                if temporal_features.dim() == 2:
                    temporal = temporal_features.view(batch, 1, 1, 1, -1).expand(batch, num_x, num_y, num_z, -1)
                else:
                    temporal = temporal_features
                dynamic_ri = model(torch.cat([spatial, temporal], dim=-1))

            raw = {key: torch.tensor(0.0, device=device) for key in warm}
            bad = {key: False for key in warm}

            with torch.amp.autocast("cuda", enabled=False):
                real = dynamic_ri[..., 0].float()
                imag = dynamic_ri[..., 1].float()
                mag = torch.sqrt(real * real + imag * imag + 1e-12)

                if enabled["sparse_dyn"] and epoch >= warm["sparse_dyn"]:
                    sparse_norm = str(reg_cfg.get("sparse_dyn", {}).get("norm", "charb")).lower()
                    if sparse_norm == "l1":
                        raw["sparse_dyn"] = mag.mean()
                    elif sparse_norm == "charb":
                        raw["sparse_dyn"] = charbonnier(mag).mean()
                    elif sparse_norm == "lp":
                        raw["sparse_dyn"] = lp_penalty(mag, float(reg_cfg.get("sparse_dyn", {}).get("p", 0.5))).mean()
                    else:
                        raise ValueError(f"Unknown sparse_dyn.norm: {sparse_norm}")

                if enabled["tv_xyz"] and epoch >= warm["tv_xyz"]:
                    mag_norm = mag / (mag.detach().mean() + 1e-3 * mag.detach().mean() + 1e-12)
                    tv_type = str(reg_cfg.get("tv_xyz", {}).get("type", "l2grad")).lower()
                    raw["tv_xyz"] = tv3d(mag_norm) if tv_type == "tv" else l2_grad3d(mag_norm)

                compute_dt = enabled["dt"] and epoch >= warm["dt"]
                compute_tv_dt = enabled["tvxyz_dt"] and epoch >= warm["tvxyz_dt"]
                compute_dtt = enabled["dtt"] and epoch >= warm["dtt"]
                denom = (mag.detach() + 1e-2 * mag.detach().mean() + 1e-12).clamp_min(1e-12)

                if compute_dt or compute_tv_dt:
                    dt_real = torch.autograd.grad(real.sum(), t_field, create_graph=True, retain_graph=True, only_inputs=True)[0].squeeze(-1)
                    dt_imag = torch.autograd.grad(imag.sum(), t_field, create_graph=True, retain_graph=True, only_inputs=True)[0].squeeze(-1)
                    dt_rel = torch.sqrt(dt_real * dt_real + dt_imag * dt_imag + 1e-12) * time_scale / denom
                    if compute_dt:
                        dt_norm = str(reg_cfg.get("dt", {}).get("norm", "l2")).lower()
                        raw["dt"] = dt_rel.abs().mean() if dt_norm == "l1" else (dt_rel * dt_rel).mean()
                    if compute_tv_dt:
                        tv_dt_type = str(reg_cfg.get("tvxyz_dt", {}).get("type", "tv")).lower()
                        raw["tvxyz_dt"] = tv3d(dt_rel) if tv_dt_type == "tv" else l2_grad3d(dt_rel)

                if compute_dtt:
                    dmag_dt = torch.autograd.grad(mag.sum(), t_field, create_graph=True, retain_graph=True, only_inputs=True)[0].squeeze(-1)
                    d2mag_dt2 = torch.autograd.grad(dmag_dt.sum(), t_field, create_graph=True, retain_graph=True, only_inputs=True)[0].squeeze(-1)
                    dtt_rel = torch.clamp(d2mag_dt2 * (time_scale ** 2) / denom, -float(reg_cfg.get("dtt", {}).get("clamp", 20.0)), float(reg_cfg.get("dtt", {}).get("clamp", 20.0)))
                    raw["dtt"] = charbonnier(dtt_rel).mean()

                losses = {}
                for key, value in raw.items():
                    losses[key], bad[key] = sanitize_loss(value, device, key)

                dynamic_complex = torch.complex(real, imag)
                image_pred = dynamic_complex + background_complex
                k_loss = torch.tensor(0.0, device=device)
                for i in range(batch):
                    k_loss = k_loss + complex_mse_loss(apply_nufft(image_pred[i].to(torch.complex64), op_batch[i]), kspace_batch[i])
                k_loss = k_loss / batch
                k_loss, bad_k = sanitize_loss(k_loss, device, "kspace")

            ema_vals["k"] = ema_update(ema_vals["k"], safe_scalar(k_loss), float(reg_cfg.get("ema_beta", 0.98)))
            for key in warm:
                if enabled[key] and epoch >= warm[key] and not bad[key]:
                    ema_vals[key] = ema_update(ema_vals[key], safe_scalar(losses[key]), float(reg_cfg.get("ema_beta", 0.98)))

            for key in warm:
                if enabled[key] and lam_target[key] is None and epoch >= warm[key] and ema_vals[key] and ema_vals[key] > 1e-12:
                    if bool(reg_cfg.get("auto_scale", True)):
                        lam_target[key] = alpha[key] * (ema_vals["k"] / (ema_vals[key] + 1e-12))
                    else:
                        lam_target[key] = float(reg_cfg.get(key, {}).get("lambda", 0.0))

            total = k_loss
            for key in warm:
                lam = 0.0 if lam_target[key] is None else float(lam_target[key] * ramp_factor(epoch, warm[key], ramp[key]))
                last_lams[key] = lam
                if lam > 0.0 and not bad[key]:
                    total = total + lam * losses[key]
            total, bad_total = sanitize_loss(total, device, "total")
            if bad_total or bad_k:
                continue

            if use_amp:
                scaler.scale(total).backward()
                scaler.unscale_(optim)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                total.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                optim.step()

            sums["loss"] += safe_scalar(total)
            sums["k"] += safe_scalar(k_loss)
            for key in warm:
                sums[key] += safe_scalar(losses[key])

            if epoch % log_interval == 0:
                with torch.no_grad():
                    recon_buffer[t_index.cpu().numpy()] = dynamic_ri.detach().cpu().float()

        scheduler.step()

        if epoch % log_interval == 0:
            avg = {key: sums[key] / max(n_batches, 1) for key in sums}
            grad = _grad_norm(model)
            print(
                f"[ep {epoch:04d}] loss={avg['loss']:.4e} k={avg['k']:.4e} "
                f"tvxyz={avg['tv_xyz']:.3e} tv(dt)={avg['tvxyz_dt']:.3e} grad={grad:.3e}"
            )
            if writer is not None:
                writer.add_scalar("train/loss", avg["loss"], epoch)
                writer.add_scalar("loss/kspace", avg["k"], epoch)
                writer.add_scalar("train/grad_norm", grad, epoch)
                for key in warm:
                    writer.add_scalar(f"loss/{key}", avg[key], epoch)
                    writer.add_scalar(f"lambda/{key}", last_lams[key], epoch)

        if save_interval and epoch % save_interval == 0:
            last_checkpoint = output_dir / "checkpoints" / f"ckpt_epoch{epoch}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "encoder": encoder.state_dict(),
                    "optim": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "config": config,
                },
                last_checkpoint,
            )
            last_reconstruction = output_dir / "images" / f"dynamic_residual_epoch{epoch}.npy"
            np.save(last_reconstruction, recon_buffer.numpy())

    if writer is not None:
        writer.close()
    return ReconstructionResult(output_dir=output_dir, last_checkpoint=last_checkpoint, last_reconstruction=last_reconstruction)

