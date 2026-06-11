#!/usr/bin/env python3
"""Plot a time-pressure contour of horizontal-mean potential temperature."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-theta-contour")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np

from case_selection import resolve_case_dirs
from plot_horizontal_mean_profiles import read_pressure_profile, read_profile_samples
from plot_rms_vel1_time_pressure_contour import (
    filter_and_average_time as filter_rms_time,
    interpolate_to_pressure_grid as interpolate_rms,
    read_cache as read_rms_cache,
)
from plot_time_evolution import (
    DEFAULT_OUTPUT_DIR,
    PRESSURE_REFERENCE_PA,
    read_time_days,
    resolve_field_files,
    select_snapshots,
    validate_variables,
)


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASE_REGEX = r"jup_crm2d_H2O-NH3_F100_nu0\.01"
FIELD = "out2"
VARIABLE = "theta"
CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot horizontal-mean potential temperature as a filled contour "
            "versus time and log pressure for one CRM case."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory containing case folders. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--case-regex",
        default=DEFAULT_CASE_REGEX,
        help=f"Full-match regex selecting exactly one case. Default: {DEFAULT_CASE_REGEX}",
    )
    parser.add_argument("--first", type=int, default=None, help="Use the first N snapshots.")
    parser.add_argument("--last", type=int, default=None, help="Use the last N snapshots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output and cache files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory. Default: <output-dir>/theta_time_pressure_cache",
    )
    parser.add_argument(
        "--rms-cache-dir",
        type=Path,
        default=None,
        help="RMS-vel1 cache directory. Default: <output-dir>/rms_vel1_time_pressure_cache",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-read NetCDF snapshots and overwrite the cache.",
    )
    parser.add_argument(
        "--max-pressure",
        type=float,
        default=100.0,
        help="Maximum displayed pressure in bar. Default: 100",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Maximum displayed time in days. Default: no limit",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=158.0,
        help="Minimum contour value in K. Default: 158",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=166.0,
        help="Maximum contour value in K. Default: 166",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.5,
        help="Discrete contour spacing in K. Default: 0.5",
    )
    parser.add_argument(
        "--cmap",
        default="YlOrRd",
        help="Matplotlib colormap. Default: YlOrRd",
    )
    parser.add_argument(
        "--rms-contour-min",
        type=float,
        default=0.6,
        help="Minimum labeled RMS-vel1 contour in m/s. Default: 0.6",
    )
    parser.add_argument(
        "--rms-contour-max",
        type=float,
        default=2.5,
        help="Maximum labeled RMS-vel1 contour in m/s. Default: 2.5",
    )
    parser.add_argument(
        "--rms-contour-spacing",
        type=float,
        default=0.2,
        help="RMS-vel1 contour spacing in m/s. Default: 0.2",
    )
    parser.add_argument(
        "--rms-running-average",
        type=float,
        default=3.0,
        help="Centered RMS-vel1 running-average window in days. Default: 3",
    )
    args = parser.parse_args()
    if args.first is not None and args.last is not None:
        parser.error("--first and --last cannot be specified together")
    if args.max_pressure <= 0.0:
        parser.error("--max-pressure must be positive")
    if args.max_time is not None and args.max_time <= 0.0:
        parser.error("--max-time must be positive")
    if args.vmin >= args.vmax:
        parser.error("--vmin must be less than --vmax")
    if args.spacing <= 0.0:
        parser.error("--spacing must be positive")
    if args.rms_contour_min <= 0.0 or args.rms_contour_spacing <= 0.0:
        parser.error("--rms-contour-min and --rms-contour-spacing must be positive")
    if args.rms_contour_min >= args.rms_contour_max:
        parser.error("--rms-contour-min must be less than --rms-contour-max")
    if args.rms_running_average <= 0.0:
        parser.error("--rms-running-average must be positive")
    return args


def snapshot_suffix(first: int | None, last: int | None) -> str:
    if first is not None:
        return f"first{first}"
    if last is not None:
        return f"last{last}"
    return "all"


def cache_path(
    cache_dir: Path,
    case_name: str,
    first: int | None,
    last: int | None,
) -> Path:
    return cache_dir / f"{case_name}_theta_time_pressure_{snapshot_suffix(first, last)}.npz"


def read_case_data(
    case_dir: Path,
    first: int | None,
    last: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    files = select_snapshots(resolve_field_files(case_dir, FIELD), first, last)
    validate_variables(files[-1].path, [VARIABLE])

    times = []
    pressures = []
    theta_profiles = []
    snapshots = []
    for file_info in files:
        _, values = read_profile_samples(file_info.path, VARIABLE)
        pressure_x1, pressure_pa = read_pressure_profile([file_info])
        if values.shape[0] != pressure_x1.size:
            raise ValueError(f"Vertical coordinate mismatch in {file_info.path}")
        times.append(read_time_days(file_info.path))
        pressures.append(pressure_pa / PRESSURE_REFERENCE_PA)
        theta_profiles.append(np.nanmean(values, axis=(1, 2)))
        snapshots.append(file_info.snapshot)

    return (
        np.asarray(times, dtype=np.float64),
        np.stack(pressures),
        np.stack(theta_profiles),
        snapshots,
    )


def write_cache(
    path: Path,
    case_name: str,
    times: np.ndarray,
    pressures: np.ndarray,
    theta_profiles: np.ndarray,
    snapshots: list[str],
) -> None:
    np.savez_compressed(
        path,
        cache_version=np.array(CACHE_VERSION),
        case_name=np.array(case_name),
        times_days=times,
        pressure_bar=pressures,
        theta=theta_profiles,
        snapshots=np.asarray(snapshots),
    )


def read_cache(path: Path, case_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as data:
        if int(data["cache_version"]) != CACHE_VERSION:
            raise ValueError(f"Unsupported cache version in {path}")
        if str(data["case_name"]) != case_name:
            raise ValueError(f"Cache case mismatch in {path}")
        return (
            data["times_days"],
            data["pressure_bar"],
            data["theta"],
            data["snapshots"].astype(str).tolist(),
        )


def interpolate_to_pressure_grid(
    pressure_bar: np.ndarray,
    theta_profiles: np.ndarray,
    max_pressure: float,
) -> tuple[np.ndarray, np.ndarray]:
    positive = pressure_bar[np.isfinite(pressure_bar) & (pressure_bar > 0.0)]
    pressure_min = float(np.nanmax(np.nanmin(pressure_bar, axis=1)))
    pressure_max = min(max_pressure, float(np.nanmin(np.nanmax(pressure_bar, axis=1))))
    if positive.size == 0 or pressure_min >= pressure_max:
        raise ValueError("Snapshots do not share a valid positive pressure range")

    pressure_grid = np.geomspace(pressure_min, pressure_max, pressure_bar.shape[1])
    interpolated = []
    for pressure, theta in zip(pressure_bar, theta_profiles):
        keep = np.isfinite(pressure) & np.isfinite(theta) & (pressure > 0.0)
        order = np.argsort(np.log(pressure[keep]))
        interpolated.append(
            np.interp(
                np.log(pressure_grid),
                np.log(pressure[keep][order]),
                theta[keep][order],
            )
        )
    return pressure_grid, np.stack(interpolated)


def filter_time(
    times_days: np.ndarray,
    theta_grid: np.ndarray,
    max_time: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times_days)
    times_days = times_days[order]
    theta_grid = theta_grid[order]
    keep = np.isfinite(times_days)
    if max_time is not None:
        keep &= times_days <= max_time
    times_days = times_days[keep]
    theta_grid = theta_grid[keep]
    if times_days.size < 2:
        raise ValueError("Time selection must contain at least two snapshots")
    return times_days, theta_grid


def plot_contour(
    path: Path,
    times_days: np.ndarray,
    pressure_grid: np.ndarray,
    theta_grid: np.ndarray,
    vmin: float,
    vmax: float,
    spacing: float,
    cmap: str,
    rms_grid: np.ndarray,
    rms_contour_min: float,
    rms_contour_max: float,
    rms_contour_spacing: float,
) -> None:
    contour_levels = np.arange(vmin, vmax + spacing / 2.0, spacing)
    fig, ax = plt.subplots(figsize=(10.0, 5.5), constrained_layout=True)
    contour = ax.contourf(
        times_days,
        pressure_grid,
        theta_grid.T,
        levels=contour_levels,
        cmap=cmap,
        extend="both",
    )
    rms_levels = np.arange(
        rms_contour_min,
        rms_contour_max + rms_contour_spacing / 2.0,
        rms_contour_spacing,
    )
    rms_contour = ax.contour(
        times_days,
        pressure_grid,
        rms_grid.T,
        levels=rms_levels,
        colors="black",
        linewidths=1.0,
    )
    ax.clabel(rms_contour, inline=True, fontsize=8, fmt="%g")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Pressure [bar]")
    colorbar = fig.colorbar(contour, ax=ax, pad=0.02)
    colorbar.set_label(r"Potential temperature, $\theta$ [K]")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    case_dirs = resolve_case_dirs(args.root.expanduser(), args.case_regex)
    if len(case_dirs) != 1:
        raise ValueError(
            f"--case-regex must match exactly one case, matched {len(case_dirs)}: "
            + ", ".join(path.name for path in case_dirs)
        )
    case_dir = case_dirs[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or args.output_dir / "theta_time_pressure_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_path(cache_dir, case_dir.name, args.first, args.last)
    if cache.exists() and not args.refresh_cache:
        times, pressures, theta_profiles, snapshots = read_cache(cache, case_dir.name)
        action = "Read"
    else:
        times, pressures, theta_profiles, snapshots = read_case_data(
            case_dir, args.first, args.last
        )
        write_cache(cache, case_dir.name, times, pressures, theta_profiles, snapshots)
        action = "Wrote"

    rms_cache_dir = (
        args.rms_cache_dir
        if args.rms_cache_dir is not None
        else args.output_dir / "rms_vel1_time_pressure_cache"
    )
    rms_cache = rms_cache_dir / f"{case_dir.name}_rms_vel1_time_pressure_all.npz"
    if not rms_cache.exists():
        raise FileNotFoundError(
            f"Missing RMS-vel1 cache: {rms_cache}\nGenerate it with:\n"
            "  python scripts/plot_rms_vel1_time_pressure_contour.py "
            f"--case-regex '^{case_dir.name}$'"
        )
    rms_times, rms_pressures, rms_profiles, _ = read_rms_cache(rms_cache, case_dir.name)

    pressure_grid, theta_grid = interpolate_to_pressure_grid(
        pressures, theta_profiles, args.max_pressure
    )
    plot_times, plot_theta = filter_time(times, theta_grid, args.max_time)
    rms_pressure_grid, rms_grid = interpolate_rms(
        rms_pressures, rms_profiles, args.max_pressure
    )
    rms_plot_times, rms_plot_grid = filter_rms_time(
        rms_times, rms_grid, args.max_time, args.rms_running_average
    )
    if not np.array_equal(plot_times, rms_plot_times):
        raise ValueError("Theta and RMS-vel1 caches do not contain matching time coordinates")
    if not np.allclose(pressure_grid, rms_pressure_grid):
        rms_plot_grid = np.stack(
            [
                np.interp(np.log(pressure_grid), np.log(rms_pressure_grid), profile)
                for profile in rms_plot_grid
            ]
        )
    suffix = snapshot_suffix(args.first, args.last)
    if args.max_time is not None:
        suffix += f"_through_day{args.max_time:g}"
    plot_path = args.output_dir / f"{case_dir.name}_theta_time_pressure_{suffix}.png"
    plot_contour(
        plot_path,
        plot_times,
        pressure_grid,
        plot_theta,
        args.vmin,
        args.vmax,
        args.spacing,
        args.cmap,
        rms_plot_grid,
        args.rms_contour_min,
        args.rms_contour_max,
        args.rms_contour_spacing,
    )

    print(f"{action} cache: {cache}")
    print(f"Read RMS-vel1 cache: {rms_cache}")
    print(f"Used snapshots {snapshots[0]}..{snapshots[-1]} ({len(snapshots)})")
    print(f"Plotted {plot_times.size} snapshots through day {plot_times[-1]:g}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
