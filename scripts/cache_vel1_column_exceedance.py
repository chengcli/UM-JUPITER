#!/usr/bin/env python3
"""Cache horizontal locations whose vertical column exceeds a vel1 threshold."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from case_selection import resolve_case_dirs
from plot_horizontal_mean_profiles import resolve_field_files, select_last_files, validate_variables


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASE_REGEX = r"jup_crm3d_H2O-NH3-H2S_F10_nu0\.01"
DEFAULT_OUTPUT_DIR = Path("diagnostics")
DEFAULT_LAST = 20
DEFAULT_THRESHOLD = 1.0
DEFAULT_THRESHOLD_STEP = 0.5
DEFAULT_MAX_LOCATIONS = 500
FIELD = "out1"
VARIABLE = "vel1"
CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache locations where vel1 exceeds a threshold anywhere in the "
            "vertical column for the latest snapshots."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-regex", default=DEFAULT_CASE_REGEX)
    parser.add_argument("--last", type=int, default=DEFAULT_LAST)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--threshold-step", type=float, default=DEFAULT_THRESHOLD_STEP)
    parser.add_argument("--max-locations", type=int, default=DEFAULT_MAX_LOCATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def threshold_tag(threshold: float) -> str:
    return f"{threshold:g}".replace("-", "m").replace(".", "p")


def cache_path(output_dir: Path, case_name: str, last: int, threshold: float) -> Path:
    return (
        output_dir
        / "vel1_column_exceedance_cache"
        / f"{case_name}_vel1_gt_{threshold_tag(threshold)}_last{last}.npz"
    )


def read_column_exceedance(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        data = ds[VARIABLE]
        if "time" in data.dims:
            if data.sizes["time"] != 1:
                raise ValueError(f"{VARIABLE!r} in {path} must contain one time")
            data = data.isel(time=0)
        values = data.transpose("x1", "x3", "x2").values
        column_max = np.nanmax(values, axis=0).astype(np.float32)
        x2 = ds["x2"].values.astype(np.float64) / 1.0e3
        x3 = ds["x3"].values.astype(np.float64) / 1.0e3
        time_seconds = np.asarray(ds["time"].values).reshape(-1)
        time_seconds = np.float64(time_seconds[0]) if time_seconds.size else np.nan
    return x2, x3, column_max, np.asarray(time_seconds)


def build_cache(
    case_dir: Path,
    output_dir: Path,
    last: int,
    threshold: float,
    threshold_step: float,
    max_locations: int,
) -> Path:
    files = select_last_files(resolve_field_files(case_dir, FIELD), last)
    validate_variables(files[-1].path, [VARIABLE])

    column_maxima = []
    times_seconds = []
    reference_x2: np.ndarray | None = None
    reference_x3: np.ndarray | None = None
    for file_info in files:
        x2, x3, column_max, time_seconds = read_column_exceedance(file_info.path)
        if reference_x2 is None:
            reference_x2, reference_x3 = x2, x3
        elif not np.allclose(reference_x2, x2) or not np.allclose(reference_x3, x3):
            raise ValueError(f"Horizontal coordinates changed in {file_info.path}")
        column_maxima.append(column_max)
        times_seconds.append(time_seconds)

    assert reference_x2 is not None and reference_x3 is not None
    column_maxima_array = np.stack(column_maxima)
    max_vel1 = np.nanmax(column_maxima_array, axis=0)
    effective_threshold = threshold
    while np.count_nonzero(max_vel1 > effective_threshold) > max_locations:
        effective_threshold += threshold_step
    snapshot_masks = column_maxima_array > effective_threshold
    exceedance_count = np.count_nonzero(snapshot_masks, axis=0).astype(np.int16)
    output_path = cache_path(output_dir, case_dir.name, last, threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        cache_version=np.array(CACHE_VERSION),
        case_name=np.array(case_dir.name),
        variable=np.array(VARIABLE),
        requested_threshold_m_s=np.array(threshold),
        effective_threshold_m_s=np.array(effective_threshold),
        max_locations=np.array(max_locations),
        x2_km=reference_x2,
        x3_km=reference_x3,
        snapshots=np.asarray([item.snapshot for item in files]),
        times_seconds=np.asarray(times_seconds),
        snapshot_exceedance=snapshot_masks,
        exceedance_count=exceedance_count,
        exceedance_fraction=exceedance_count.astype(np.float32) / len(files),
        any_exceedance=exceedance_count > 0,
        max_vel1=max_vel1,
    )
    return output_path


def main() -> None:
    args = parse_args()
    if args.last <= 0:
        raise ValueError("--last must be positive")
    if args.threshold_step <= 0.0:
        raise ValueError("--threshold-step must be positive")
    if args.max_locations <= 0:
        raise ValueError("--max-locations must be positive")

    case_dirs = resolve_case_dirs(args.root.expanduser(), args.case_regex)
    for case_dir in case_dirs:
        output_path = build_cache(
            case_dir,
            args.output_dir,
            args.last,
            args.threshold,
            args.threshold_step,
            args.max_locations,
        )
        with np.load(output_path, allow_pickle=False) as cache:
            marked = int(np.count_nonzero(cache["any_exceedance"]))
            total = int(cache["any_exceedance"].size)
            effective_threshold = float(cache["effective_threshold_m_s"])
        print(
            f"{case_dir.name}: threshold {effective_threshold:g} m/s; "
            f"cached {marked}/{total} marked locations in {output_path}"
        )


if __name__ == "__main__":
    main()
