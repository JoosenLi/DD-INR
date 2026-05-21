"""SNAKE simulation recipe used to prepare DD-INR release-format data.

This script is intentionally explicit because trajectory, frame grouping, and
background reconstruction conventions are easy places to introduce silent errors.
It requires SNAKE, MRI-NUFFT, and a trajectory file compatible with
`mrinufft.io.read_trajectory`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a MICCAI-style SNAKE simulation for DD-INR.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--trajectory-bin", required=True)
    parser.add_argument("--subject-id", type=int, default=4)
    parser.add_argument("--snr", type=float, default=800.0)
    parser.add_argument("--shots-per-frame", type=int, default=20)
    parser.add_argument("--max-sim-time", type=int, default=240)
    args = parser.parse_args()

    from mrinufft import get_operator
    from mrinufft.extras.optim import loss_l2_reg
    from mrinufft.io import read_trajectory
    from snake.core import NufftAcquisitionEngine
    from snake.core.handlers import BlockActivationHandler
    from snake.core.phantom import Phantom
    from snake.core.sampling.samplers import LoadMultiTrajectorySampler
    from snake.core.simulation import GreConfig, SimConfig, default_hardware
    from snake.core.transform import apply_affine
    from snake.mrd_utils import NonCartesianFrameDataLoader

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_conf = SimConfig(max_sim_time=args.max_sim_time, seq=GreConfig(TR=50, TE=30, FA=12), hardware=default_hardware)
    sim_conf.hardware.n_coils = 1
    sim_conf.hardware.field_strength = 3
    sim_conf.fov.res_mm = (3, 3, 3)
    sim_conf.fov.size = (192, 192, 120)
    sim_conf.fov.offset = (-90, -110, -40)
    sim_conf.fov.angles = (0, 0, 0)
    image_shape = sim_conf.fov.shape

    activation = BlockActivationHandler(
        block_off=20,
        block_on=20,
        duration=args.max_sim_time,
        atlas="hardvard-oxford__cort-maxprob-thr0-1mm",
        atlas_label=48,
    )

    raw_traj = read_trajectory(grad_filename=args.trajectory_bin, dwell_time=0.002, raster_time=0.01)
    np.save(out_dir / "raw_traj.npy", raw_traj[0].reshape(args.max_sim_time * 20, 1, 15000, 3))
    sampler = LoadMultiTrajectorySampler(path=str(out_dir / "raw_traj.npy"))

    phantom_original = Phantom.from_brainweb(sub_id=args.subject_id, sim_conf=sim_conf, tissue_file="tissue_3T", output_res=1)
    phantom = phantom_original.resample(new_affine=sim_conf.fov.affine, new_shape=image_shape, use_gpu=True)
    static_phantom = activation.get_static(phantom.copy(), sim_conf)
    roi = static_phantom.masks[static_phantom.labels_idx["ROI"]]
    roi_resampled = apply_affine(roi, new_affine=sim_conf.fov.affine, old_affine=phantom.affine, new_shape=image_shape)

    engine = NufftAcquisitionEngine(model="T2s", snr=args.snr, slice_2d=False)
    engine(
        str(out_dir / "dataloader.mrd"),
        sampler,
        phantom,
        sim_conf,
        handlers=[activation],
        worker_chunk_size=20,
        n_workers=10,
        nufft_backend="cufinufft",
    )

    traj_frames = []
    kspace_frames = []
    with NonCartesianFrameDataLoader(str(out_dir / "dataloader.mrd"), squeeze_dims=True) as loader:
        for _, traj, kspace_data in loader.iter_frames(shot_dim=True):
            traj_frames.append(np.asarray(traj, dtype=np.float32))
            kspace_frames.append(np.ascontiguousarray(kspace_data).astype(np.complex64))
        n_coils = loader.n_coils
        smaps = loader.get_smaps(resample=False)

    traj_frames = np.stack(traj_frames)
    kspace_frames = np.stack(kspace_frames)
    if n_coils == 1:
        kspace_frames = kspace_frames[:, 0, 0, :]
    else:
        kspace_frames = kspace_frames[:, :, 0, :]

    num_frame = traj_frames.shape[0] // args.shots_per_frame
    samples_per_shot = traj_frames.shape[2]
    traj = traj_frames.reshape(num_frame, args.shots_per_frame, samples_per_shot, 3)
    kspace = kspace_frames.reshape(num_frame, n_coils, args.shots_per_frame, samples_per_shot)

    samples_all = traj.reshape(-1, 3)
    kspace_all = np.transpose(kspace.reshape(num_frame, n_coils, -1), (1, 0, 2)).reshape(n_coils, -1)
    op_bg = get_operator("cufinufft")(samples=samples_all, shape=image_shape, n_coils=n_coils, density=True, smaps=smaps)
    recon_bg = op_bg.pinv_solver(kspace_all, optim="cg", max_iter=25, callback=loss_l2_reg)[0][0, 0]
    background = np.stack([recon_bg.real, recon_bg.imag], axis=-1).astype(np.float32)

    np.save(out_dir / "kspace.npy", kspace)
    np.save(out_dir / "traj.npy", traj)
    np.save(out_dir / "background.npy", background)
    np.save(out_dir / "roi.npy", roi_resampled)
    if smaps is not None:
        np.save(out_dir / "csm.npy", smaps)
    print(f"Wrote DD-INR case to {out_dir}")


if __name__ == "__main__":
    main()

