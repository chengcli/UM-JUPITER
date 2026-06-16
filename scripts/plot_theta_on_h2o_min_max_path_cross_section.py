#!/usr/bin/env python3
"""Plot potential-temperature anomaly on the H2O path-extrema cross-section."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-theta-h2o-section")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cache_cross_section_dynamics import dynamics_cache_path
from case_selection import resolve_case_dirs
from custom_colormaps import diverging_with_white_plateau
from periodic_cross_section import sample_periodic_bilinear
from plot_h2o_on_h2o_min_max_path_cross_section import (
    DEFAULT_CASE_REGEX,
    DEFAULT_LAST,
    DEFAULT_ROOT,
    MAX_PRESSURE_BAR,
    NEGATIVE_PERP_VELOCITY_LEVELS,
    POSITIVE_PERP_VELOCITY_LEVELS,
    cache_path as h2o_section_cache_path,
    read_dynamics_cache,
    symmetric_robust_levels,
)
from plot_horizontal_mean_profiles import (
    DEFAULT_OUTPUT_DIR,
    resolve_field_files,
    validate_variables,
)


CACHE_VERSION = 1
FIELD = "out2"
THETA_VARIABLE = "theta"
CLOUD_LEVELS = np.geomspace(1.0e-5, 1.0e-3, 5)
THETA_WHITE_HALF_WIDTH_K = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot potential-temperature anomaly on the H2O min/max path "
            "cross-section with wind and H2O cloud overlays."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-regex", default=DEFAULT_CASE_REGEX)
    parser.add_argument("--last", type=int, default=DEFAULT_LAST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dynamics-cache", type=Path, default=None)
    parser.add_argument("--section-cache", type=Path, default=None)
    parser.add_argument("--theta-cache", type=Path, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--show-wind",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay perpendicular wind contours. Default: enabled.",
    )
    return parser.parse_args()


def theta_cache_path(output_dir: Path, case_name: str, last: int) -> Path:
    return (
        output_dir
        / "theta_min_max_path_cross_section_cache"
        / f"{case_name}_theta_on_H2O_min_max_path_cross_section_last{last}.npz"
    )


def output_path(output_dir: Path, case_name: str, last: int) -> Path:
    return (
        output_dir
        / f"{case_name}_theta_on_H2O_min_max_path_cross_section_wind_cloud_last{last}.png"
    )


def read_theta_section(
    path: Path,
    section_x2_km: np.ndarray,
    section_x3_km: np.ndarray,
) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        data = ds[THETA_VARIABLE]
        if "time" in data.dims:
            if data.sizes["time"] != 1:
                raise ValueError(f"{THETA_VARIABLE!r} in {path} must contain one time")
            data = data.isel(time=0)
        x2_km = ds["x2"].values.astype(np.float64) / 1.0e3
        x3_km = ds["x3"].values.astype(np.float64) / 1.0e3
        values = data.transpose("x1", "x3", "x2").values.astype(np.float64)
    return sample_periodic_bilinear(values, x2_km, x3_km, section_x2_km, section_x3_km)


def build_theta_cache(case_dir: Path, section_path: Path, output: Path, last: int) -> None:
    if not section_path.exists():
        raise FileNotFoundError(
            f"Missing H2O section cache: {section_path}\nGenerate it with:\n"
            "  python scripts/plot_h2o_on_h2o_min_max_path_cross_section.py "
            f"--case-regex '{case_dir.name}' --last {last} --refresh-cache --cache-only"
        )

    with np.load(section_path, allow_pickle=False) as section:
        case_name = str(section["case_name"])
        snapshots = section["snapshots"].astype(str)
        section_x2_km = section["section_x2_wrapped_km"]
        section_x3_km = section["section_x3_wrapped_km"]
        distance_km = section["section_distance_km"]

    if case_name != case_dir.name:
        raise ValueError(f"Section cache case mismatch: {section_path}")

    out2_by_snapshot = {item.snapshot: item for item in resolve_field_files(case_dir, FIELD)}
    missing = [snapshot for snapshot in snapshots if snapshot not in out2_by_snapshot]
    if missing:
        raise FileNotFoundError(
            f"Missing matching out2 theta snapshots in {case_dir}: {', '.join(missing)}"
        )
    selected_files = [out2_by_snapshot[snapshot] for snapshot in snapshots]
    validate_variables(selected_files[-1].path, [THETA_VARIABLE])

    theta_sum = None
    theta_square_sum = None
    for file_info in selected_files:
        theta = read_theta_section(file_info.path, section_x2_km, section_x3_km)
        if theta_sum is None:
            theta_sum = np.zeros_like(theta)
            theta_square_sum = np.zeros_like(theta)
        theta_sum += theta
        theta_square_sum += theta * theta

    assert theta_sum is not None
    assert theta_square_sum is not None
    theta_mean = theta_sum / len(selected_files)
    theta_std = np.sqrt(np.maximum(theta_square_sum / len(selected_files) - theta_mean**2, 0.0))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        cache_version=np.array(CACHE_VERSION),
        case_name=np.array(case_name),
        snapshots=snapshots,
        section_distance_km=distance_km,
        theta_mean=theta_mean.astype(np.float32),
        theta_std=theta_std.astype(np.float32),
    )


def read_theta_cache(path: Path, section_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing theta section cache: {path}")
    with np.load(section_path, allow_pickle=False) as section, np.load(path, allow_pickle=False) as cache:
        if int(cache["cache_version"]) != CACHE_VERSION:
            raise ValueError(f"Unsupported theta cache version in {path}")
        if str(cache["case_name"]) != str(section["case_name"]):
            raise ValueError(f"Theta cache case mismatch: {path}")
        if not np.array_equal(cache["snapshots"].astype(str), section["snapshots"].astype(str)):
            raise ValueError(f"Theta cache snapshots do not match section cache: {path}")
        if not np.allclose(cache["section_distance_km"], section["section_distance_km"]):
            raise ValueError(f"Theta cache section coordinates do not match: {path}")
        return cache["theta_mean"].astype(np.float64), cache["theta_std"].astype(np.float64)


def theta_anomaly(theta: np.ndarray) -> np.ndarray:
    horizontal_mean = np.nanmean(theta, axis=1, keepdims=True)
    return theta - horizontal_mean


def plot_cache(
    section_path: Path,
    theta_path: Path,
    dynamics_path: Path,
    figure_path: Path,
    show_wind: bool,
) -> None:
    with np.load(section_path, allow_pickle=False) as section:
        case_name = str(section["case_name"])
        snapshots = section["snapshots"].astype(str)
        distance_km = section["section_distance_km"]
        pressure_bar = section["pressure_cross_section_bar"]
        if "cloud_cross_section" in section:
            cloud = section["cloud_cross_section"]
        elif "h2o_cloud_cross_section" in section:
            cloud = section["h2o_cloud_cross_section"]
        else:
            raise ValueError(
                f"Section cache lacks H2O cloud cross-section data: {section_path}"
            )

    theta, _ = read_theta_cache(theta_path, section_path)
    anomaly = theta_anomaly(theta)
    displayed = pressure_bar <= MAX_PRESSURE_BAR
    theta_levels = symmetric_robust_levels(anomaly, displayed, count=11)

    wind = None
    if show_wind:
        wind, _ = read_dynamics_cache(
            dynamics_path,
            case_name,
            snapshots,
            distance_km,
            anomaly.shape,
        )

    fig = plt.figure(figsize=(10.0, 4.5), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 0.02),
        height_ratios=(1.0, 1.0),
        hspace=0.04,
    )
    ax = fig.add_subplot(grid[:, 0])
    theta_colorbar_ax = fig.add_subplot(grid[0, 1])
    cloud_colorbar_ax = fig.add_subplot(grid[1, 1])
    distance_grid = np.broadcast_to(distance_km, pressure_bar.shape)

    theta_cmap = diverging_with_white_plateau(
        "PiYG",
        float(theta_levels[0]),
        float(theta_levels[-1]),
        min(THETA_WHITE_HALF_WIDTH_K, 0.5 * float(theta_levels[-1] - theta_levels[0])),
    )
    theta_contour = ax.contourf(
        distance_grid,
        pressure_bar,
        anomaly,
        levels=theta_levels,
        cmap=theta_cmap,
        extend="both",
    )

    cloud_contour = ax.contourf(
        distance_grid,
        pressure_bar,
        np.ma.masked_less(cloud, CLOUD_LEVELS[0]),
        levels=CLOUD_LEVELS,
        cmap="Blues",
        extend="max",
    )
    ax.contour(
        distance_grid,
        pressure_bar,
        cloud,
        levels=[CLOUD_LEVELS[0]],
        colors="tab:blue",
        linewidths=0.8,
    )

    if show_wind:
        assert wind is not None
        negative = ax.contour(
            distance_grid,
            pressure_bar,
            wind,
            levels=NEGATIVE_PERP_VELOCITY_LEVELS,
            colors="black",
            linestyles="dashed",
            linewidths=1.1,
        )
        positive = ax.contour(
            distance_grid,
            pressure_bar,
            wind,
            levels=POSITIVE_PERP_VELOCITY_LEVELS,
            colors="black",
            linestyles="solid",
            linewidths=1.1,
        )
        ax.clabel(negative, levels=NEGATIVE_PERP_VELOCITY_LEVELS, fmt="%.0f", fontsize=7)
        ax.clabel(positive, levels=POSITIVE_PERP_VELOCITY_LEVELS, fmt="%.0f", fontsize=7)

    ax.set_xlabel("Distance along x-section [km]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_yscale("log")
    ax.set_ylim(MAX_PRESSURE_BAR, np.nanmin(pressure_bar))

    theta_colorbar = fig.colorbar(theta_contour, cax=theta_colorbar_ax)
    theta_colorbar.set_label(r"$\theta$ anomaly [K]")
    theta_colorbar.set_ticks([theta_levels[0], 0.0, theta_levels[-1]])
    theta_colorbar.ax.tick_params(labelsize=9)
    theta_colorbar.ax.yaxis.label.set_size(9)

    cloud_colorbar = fig.colorbar(cloud_contour, cax=cloud_colorbar_ax)
    cloud_colorbar.set_label("H$_2$O cloud mass fraction")
    cloud_colorbar.set_ticks(CLOUD_LEVELS)
    cloud_colorbar.set_ticklabels([f"{value:.1e}" for value in CLOUD_LEVELS])
    cloud_colorbar.ax.tick_params(labelsize=9)
    cloud_colorbar.ax.yaxis.label.set_size(9)

    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.last <= 0:
        raise ValueError("--last must be positive")
    cases = resolve_case_dirs(args.root.expanduser(), args.case_regex)
    if len(cases) != 1:
        raise ValueError("--case-regex must match exactly one case")
    case_dir = cases[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    section = args.section_cache or h2o_section_cache_path(args.output_dir, case_dir.name, args.last)
    theta = args.theta_cache or theta_cache_path(args.output_dir, case_dir.name, args.last)
    dynamics = args.dynamics_cache or dynamics_cache_path(args.output_dir, case_dir.name, args.last)
    figure = output_path(args.output_dir, case_dir.name, args.last)

    if args.refresh_cache or not theta.exists():
        build_theta_cache(case_dir, section, theta, args.last)
        print(f"Wrote cache {theta}")
    else:
        print(f"Using cache {theta}")
    plot_cache(section, theta, dynamics, figure, args.show_wind)
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
