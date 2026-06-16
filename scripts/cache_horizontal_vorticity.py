#!/usr/bin/env python3
"""Cache time-mean horizontal-plane vorticity from CRM velocity outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from case_selection import resolve_case_dirs
from plot_horizontal_mean_profiles import (
    DEFAULT_OUTPUT_DIR,
    resolve_field_files,
    select_last_files,
    validate_variables,
)


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASE_REGEX = r"jup_crm3d_H2O-NH3-H2S_F10_nu0\.01"
DEFAULT_LAST = 20
FIELD = "out1"
VELOCITY_X2 = "vel2"
VELOCITY_X3 = "vel3"
CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and cache horizontal-plane vertical vorticity, "
            "zeta = d(vel3)/d(x2) - d(vel2)/d(x3), averaged over latest snapshots."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-regex", default=DEFAULT_CASE_REGEX)
    parser.add_argument("--last", type=int, default=DEFAULT_LAST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def cache_path(output_dir: Path, case_name: str, last: int) -> Path:
    return (
        output_dir
        / "horizontal_vorticity_cache"
        / f"{case_name}_horizontal_vorticity_last{last}.npz"
    )


def read_velocity_snapshot(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        vel2 = ds[VELOCITY_X2]
        vel3 = ds[VELOCITY_X3]
        if "time" in vel2.dims:
            if vel2.sizes["time"] != 1 or vel3.sizes["time"] != 1:
                raise ValueError(f"{path} must contain exactly one time per velocity field")
            vel2 = vel2.isel(time=0)
            vel3 = vel3.isel(time=0)
        vel2_values = vel2.transpose("x1", "x3", "x2").values.astype(np.float64)
        vel3_values = vel3.transpose("x1", "x3", "x2").values.astype(np.float64)
        x1 = ds["x1"].values.astype(np.float64)
        x2 = ds["x2"].values.astype(np.float64)
        x3 = ds["x3"].values.astype(np.float64)
        time_seconds = np.asarray(ds["time"].values).reshape(-1)
        time_seconds = float(time_seconds[0]) if time_seconds.size else np.nan
    return x1, x2, x3, vel2_values, vel3_values, np.asarray(time_seconds)


def uniform_spacing(coordinate: np.ndarray, name: str) -> float:
    spacing = np.diff(coordinate)
    if spacing.size == 0:
        raise ValueError(f"{name} must have at least two points")
    if not np.allclose(spacing, spacing[0], rtol=1.0e-6, atol=1.0e-9):
        raise ValueError(f"{name} is not uniformly spaced; periodic finite difference needs uniform grid")
    return float(spacing[0])


def periodic_centered_difference(values: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (2.0 * spacing)


def horizontal_vorticity(
    vel2: np.ndarray,
    vel3: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
) -> np.ndarray:
    dx2 = uniform_spacing(x2, "x2")
    dx3 = uniform_spacing(x3, "x3")
    dvel3_dx2 = periodic_centered_difference(vel3, axis=2, spacing=dx2)
    dvel2_dx3 = periodic_centered_difference(vel2, axis=1, spacing=dx3)
    return dvel3_dx2 - dvel2_dx3


def build_cache(case_dir: Path, output_dir: Path, last: int) -> Path:
    files = select_last_files(resolve_field_files(case_dir, FIELD), last)
    validate_variables(files[-1].path, [VELOCITY_X2, VELOCITY_X3])

    reference_x1: np.ndarray | None = None
    reference_x2: np.ndarray | None = None
    reference_x3: np.ndarray | None = None
    vorticity_sum: np.ndarray | None = None
    vorticity_square_sum: np.ndarray | None = None
    times_seconds = []

    for file_info in files:
        x1, x2, x3, vel2, vel3, time_seconds = read_velocity_snapshot(file_info.path)
        if reference_x1 is None:
            reference_x1, reference_x2, reference_x3 = x1, x2, x3
            vorticity_sum = np.zeros_like(vel2, dtype=np.float64)
            vorticity_square_sum = np.zeros_like(vel2, dtype=np.float64)
        elif (
            not np.allclose(reference_x1, x1)
            or not np.allclose(reference_x2, x2)
            or not np.allclose(reference_x3, x3)
        ):
            raise ValueError(f"Grid coordinates changed in {file_info.path}")

        zeta = horizontal_vorticity(vel2, vel3, x2, x3)
        vorticity_sum += zeta
        vorticity_square_sum += zeta * zeta
        times_seconds.append(time_seconds)

    assert reference_x1 is not None
    assert reference_x2 is not None
    assert reference_x3 is not None
    assert vorticity_sum is not None
    assert vorticity_square_sum is not None

    vorticity_mean = vorticity_sum / len(files)
    vorticity_std = np.sqrt(np.maximum(vorticity_square_sum / len(files) - vorticity_mean**2, 0.0))
    column_mean_vorticity = np.nanmean(vorticity_mean, axis=0)

    output_path = cache_path(output_dir, case_dir.name, last)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        cache_version=np.array(CACHE_VERSION),
        case_name=np.array(case_dir.name),
        snapshots=np.asarray([item.snapshot for item in files]),
        times_seconds=np.asarray(times_seconds),
        x1_m=reference_x1,
        x2_m=reference_x2,
        x3_m=reference_x3,
        x2_km=reference_x2 / 1.0e3,
        x3_km=reference_x3 / 1.0e3,
        vorticity_mean_s_inv=vorticity_mean.astype(np.float32),
        vorticity_std_s_inv=vorticity_std.astype(np.float32),
        column_mean_vorticity_s_inv=column_mean_vorticity.astype(np.float32),
        formula=np.array("d(vel3)/d(x2) - d(vel2)/d(x3)"),
    )
    return output_path


def main() -> None:
    args = parse_args()
    if args.last <= 0:
        raise ValueError("--last must be positive")

    case_dirs = resolve_case_dirs(args.root.expanduser(), args.case_regex)
    for case_dir in case_dirs:
        output_path = build_cache(case_dir, args.output_dir, args.last)
        with np.load(output_path, allow_pickle=False) as cache:
            zeta = cache["column_mean_vorticity_s_inv"]
            snapshots = cache["snapshots"].astype(str)
        print(
            f"{case_dir.name}: snapshots {snapshots[0]}..{snapshots[-1]}; "
            f"column-mean zeta range {np.nanmin(zeta):.6e}..{np.nanmax(zeta):.6e} s^-1; "
            f"wrote {output_path}"
        )


if __name__ == "__main__":
    main()
