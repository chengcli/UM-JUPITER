#!/usr/bin/env python3
"""Compose cached RMS-vel1 and theta time-pressure contours."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-rms-theta-stack")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np

from case_selection import resolve_case_dirs
from plot_rms_vel1_time_pressure_contour import (
    filter_and_average_time,
    interpolate_to_pressure_grid as interpolate_rms,
    read_cache as read_rms_cache,
)
from plot_theta_time_pressure_contour import (
    filter_time,
    interpolate_to_pressure_grid as interpolate_theta,
    read_cache as read_theta_cache,
)


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASE_REGEX = r"jup_crm2d_H2O-NH3_F100_nu0\.01"
DEFAULT_OUTPUT_DIR = Path("diagnostics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a stacked RMS-vel1/theta time-pressure contour plot from "
            "existing per-case caches."
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing caches and receiving output. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=300.0,
        help="Maximum displayed time in days. Default: 300",
    )
    parser.add_argument(
        "--max-pressure",
        type=float,
        default=100.0,
        help="Maximum displayed pressure in bar. Default: 100",
    )
    parser.add_argument(
        "--rms-vmax",
        type=float,
        default=2.5,
        help="Maximum RMS-vel1 contour value in m/s. Default: 2.5",
    )
    parser.add_argument(
        "--theta-vmin",
        type=float,
        default=158.0,
        help="Minimum theta contour value in K. Default: 158",
    )
    parser.add_argument(
        "--theta-vmax",
        type=float,
        default=166.0,
        help="Maximum theta contour value in K. Default: 166",
    )
    parser.add_argument(
        "--rms-cmap",
        default="YlOrRd",
        help="Matplotlib colormap for RMS-vel1. Default: YlOrRd",
    )
    parser.add_argument(
        "--theta-cmap",
        default="YlGnBu",
        help="Matplotlib colormap for theta. Default: YlGnBu",
    )
    args = parser.parse_args()
    if args.max_time <= 0.0 or args.max_pressure <= 0.0:
        parser.error("--max-time and --max-pressure must be positive")
    if args.rms_vmax <= 0.0:
        parser.error("--rms-vmax must be positive")
    if args.theta_vmin >= args.theta_vmax:
        parser.error("--theta-vmin must be less than --theta-vmax")
    return args


def required_caches(output_dir: Path, case_name: str) -> tuple[Path, Path]:
    rms = (
        output_dir
        / "rms_vel1_time_pressure_cache"
        / f"{case_name}_rms_vel1_time_pressure_all.npz"
    )
    theta = (
        output_dir
        / "theta_time_pressure_cache"
        / f"{case_name}_theta_time_pressure_all.npz"
    )
    missing = [path for path in (rms, theta) if not path.exists()]
    if missing:
        commands = [
            (
                "python scripts/plot_rms_vel1_time_pressure_contour.py "
                f"--case-regex '^{case_name}$'"
            ),
            (
                "python scripts/plot_theta_time_pressure_contour.py "
                f"--case-regex '^{case_name}$'"
            ),
        ]
        raise FileNotFoundError(
            "Missing required cache(s):\n  "
            + "\n  ".join(str(path) for path in missing)
            + "\nGenerate them with:\n  "
            + "\n  ".join(commands)
        )
    return rms, theta


def prepare_data(
    case_name: str,
    rms_cache: Path,
    theta_cache: Path,
    max_time: float,
    max_pressure: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rms_times, rms_pressures, rms_profiles, _ = read_rms_cache(rms_cache, case_name)
    theta_times, theta_pressures, theta_profiles, _ = read_theta_cache(
        theta_cache, case_name
    )

    rms_pressure_grid, rms_grid = interpolate_rms(
        rms_pressures, rms_profiles, max_pressure
    )
    theta_pressure_grid, theta_grid = interpolate_theta(
        theta_pressures, theta_profiles, max_pressure
    )
    rms_times, rms_grid = filter_and_average_time(rms_times, rms_grid, max_time, None)
    theta_times, theta_grid = filter_time(theta_times, theta_grid, max_time)
    if not np.array_equal(rms_times, theta_times):
        raise ValueError("RMS-vel1 and theta caches do not contain matching time coordinates")
    return rms_times, rms_pressure_grid, rms_grid, theta_pressure_grid, theta_grid


def plot_stack(
    path: Path,
    times: np.ndarray,
    rms_pressure: np.ndarray,
    rms_grid: np.ndarray,
    theta_pressure: np.ndarray,
    theta_grid: np.ndarray,
    rms_vmax: float,
    theta_vmin: float,
    theta_vmax: float,
    rms_cmap: str,
    theta_cmap: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 6.0),
        sharex=True,
        constrained_layout=True,
    )
    rms_levels = np.arange(0.0, rms_vmax + 0.125, 0.25)
    theta_levels = np.arange(theta_vmin, theta_vmax + 0.25, 0.5)
    rms_contour = axes[0].contourf(
        times,
        rms_pressure,
        rms_grid.T,
        levels=rms_levels,
        cmap=rms_cmap,
        extend="max",
    )
    theta_contour = axes[1].contourf(
        times,
        theta_pressure,
        theta_grid.T,
        levels=theta_levels,
        cmap=theta_cmap,
        extend="both",
    )

    for ax in axes:
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_ylabel("Pressure [bar]")
    axes[1].set_xlabel("Time [days]")
    rms_colorbar = fig.colorbar(
        rms_contour, ax=axes[0], pad=0.015, fraction=0.03, aspect=18
    )
    rms_colorbar.set_label(r"RMS vel1 [m s$^{-1}$]", fontsize=12)
    rms_colorbar.ax.tick_params(labelsize=10)
    theta_colorbar = fig.colorbar(
        theta_contour, ax=axes[1], pad=0.015, fraction=0.03, aspect=18
    )
    theta_colorbar.set_label(r"$\theta$ [K]", fontsize=12)
    theta_colorbar.ax.tick_params(labelsize=10)
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
    case_name = case_dirs[0].name
    rms_cache, theta_cache = required_caches(args.output_dir, case_name)
    times, rms_pressure, rms_grid, theta_pressure, theta_grid = prepare_data(
        case_name,
        rms_cache,
        theta_cache,
        args.max_time,
        args.max_pressure,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = (
        args.output_dir
        / f"{case_name}_rms_vel1_theta_time_pressure_stack_through_day{args.max_time:g}.png"
    )
    plot_stack(
        path,
        times,
        rms_pressure,
        rms_grid,
        theta_pressure,
        theta_grid,
        args.rms_vmax,
        args.theta_vmin,
        args.theta_vmax,
        args.rms_cmap,
        args.theta_cmap,
    )
    print(f"Read RMS-vel1 cache: {rms_cache}")
    print(f"Read theta cache: {theta_cache}")
    print(f"Plotted {times.size} snapshots through day {times[-1]:g}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
