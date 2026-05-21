from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from dd_inr.io import flatten_shots, load_case
from dd_inr.train import reconstruct_dynamic_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DD-INR 3D+time fMRI reconstruction.")
    parser.add_argument("--config", required=True, help="Path to a DD-INR yaml config.")
    parser.add_argument("--data-dir", required=True, help="Directory containing kspace.npy, traj.npy, background.npy, optional csm.npy.")
    parser.add_argument("--case-name", default=None, help="Name used under the output directory.")
    parser.add_argument("--scale", type=float, default=1.0, help="Optional multiplicative scale for k-space and background.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kspace, traj, background, csm = load_case(args.data_dir)
    kspace, traj = flatten_shots(kspace, traj)
    case_name = args.case_name or Path(args.data_dir).name
    result = reconstruct_dynamic_3d(kspace * args.scale, traj, background * args.scale, config, case_name=case_name, csm=csm)
    print(f"Output directory: {result.output_dir}")
    if result.last_checkpoint:
        print(f"Last checkpoint: {result.last_checkpoint}")
    if result.last_reconstruction:
        print(f"Last reconstruction: {result.last_reconstruction}")


if __name__ == "__main__":
    main()

