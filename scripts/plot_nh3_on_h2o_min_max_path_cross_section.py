#!/usr/bin/env python3
"""Plot NH3 on the periodic cross-section defined by H2O path extrema."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-nh3-cross-section")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np

from cache_cross_section_dynamics import dynamics_cache_path
from cache_horizontal_vorticity import cache_path as horizontal_vorticity_cache_path
from case_selection import resolve_case_dirs
from custom_colormaps import diverging_with_white_plateau
from plot_h2o_on_h2o_min_max_path_cross_section import (
    DEFAULT_CASE_REGEX,
    DEFAULT_LAST,
    DEFAULT_ROOT,
    MAX_PRESSURE_BAR,
    NEGATIVE_PERP_VELOCITY_LEVELS,
    NEGATIVE_VORTICITY_LEVELS,
    POSITIVE_PERP_VELOCITY_LEVELS,
    POSITIVE_VORTICITY_LEVELS,
    VAPOR_FRACTIONAL_DEVIATION_LIMIT,
    VAPOR_FRACTIONAL_WHITE_HALF_WIDTH,
    VORTICITY_SCALE,
    cache_path,
    read_dynamics_cache,
    read_vorticity_cache,
    symmetric_robust_levels,
)
from plot_horizontal_mean_profiles import DEFAULT_OUTPUT_DIR


CLOUD_LEVELS = np.geomspace(1.0e-5, 1.0e-3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot NH3 vapor fractional deviation and NH3 cloud on the periodic "
            "cross-section defined by the H2O vapor-path extrema."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-regex", default=DEFAULT_CASE_REGEX)
    parser.add_argument("--last", type=int, default=DEFAULT_LAST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dynamics-cache",
        type=Path,
        default=None,
        help="Projected velocity and streamfunction cache. Default: inferred.",
    )
    parser.add_argument(
        "--vorticity-cache",
        type=Path,
        default=None,
        help="Horizontal vorticity cache. Default: inferred.",
    )
    parser.add_argument(
        "--show-perp-velocity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay perpendicular-velocity contours. Default: enabled.",
    )
    parser.add_argument(
        "--show-vorticity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay horizontal vorticity contours scaled by 1e5. Default: enabled.",
    )
    parser.add_argument(
        "--show-streamfunction",
        action="store_true",
        help="Overlay dashed mass-streamfunction contours. Default: disabled.",
    )
    return parser.parse_args()


def output_path(output_dir: Path, case_name: str, last: int) -> Path:
    return (
        output_dir
        / f"{case_name}_NH3_on_H2O_min_max_path_cross_section_wind_vorticity_last{last}.png"
    )


def fractional_deviation(values: np.ndarray) -> np.ndarray:
    horizontal_mean = np.nanmean(values, axis=1, keepdims=True)
    return np.divide(
        values - horizontal_mean,
        horizontal_mean,
        out=np.full_like(values, np.nan),
        where=horizontal_mean > 0.0,
    )


def plot_cache(
    section_path: Path,
    dynamics_path: Path,
    vorticity_path: Path,
    figure_path: Path,
    show_perp_velocity: bool,
    show_vorticity: bool,
    show_streamfunction: bool,
) -> None:
    with np.load(section_path, allow_pickle=False) as cache:
        required = {
            "nh3_vapor_cross_section",
            "nh3_cloud_cross_section",
            "pressure_cross_section_bar",
        }
        missing = sorted(required.difference(cache.files))
        if missing:
            raise ValueError(
                f"Section cache is missing {', '.join(missing)}: {section_path}\n"
                "Rebuild it with:\n"
                "  python scripts/plot_h2o_on_h2o_min_max_path_cross_section.py "
                f"--case-regex '{str(cache['case_name'])}' "
                f"--last {len(cache['snapshots'])} --refresh-cache --cache-only"
            )
        case_name = str(cache["case_name"])
        snapshots = cache["snapshots"].astype(str)
        distance_km = cache["section_distance_km"]
        section_x2_km = cache["section_x2_wrapped_km"]
        section_x3_km = cache["section_x3_wrapped_km"]
        pressure_bar = cache["pressure_cross_section_bar"]
        nh3_vapor = fractional_deviation(cache["nh3_vapor_cross_section"])
        nh3_cloud = cache["nh3_cloud_cross_section"]

    perpendicular_velocity = None
    streamfunction = None
    vorticity = None
    if show_perp_velocity or show_streamfunction:
        perpendicular_velocity, streamfunction = read_dynamics_cache(
            dynamics_path,
            case_name,
            snapshots,
            distance_km,
            nh3_vapor.shape,
        )
    if show_vorticity:
        vorticity = read_vorticity_cache(
            vorticity_path,
            case_name,
            snapshots,
            section_x2_km,
            section_x3_km,
            nh3_vapor.shape,
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
    vapor_colorbar_ax = fig.add_subplot(grid[0, 1])
    cloud_colorbar_ax = fig.add_subplot(grid[1, 1])
    distance_grid = np.broadcast_to(distance_km, pressure_bar.shape)

    vapor_levels = np.linspace(
        -VAPOR_FRACTIONAL_DEVIATION_LIMIT,
        VAPOR_FRACTIONAL_DEVIATION_LIMIT,
        11,
    )
    vapor_cmap = diverging_with_white_plateau(
        "PiYG",
        -VAPOR_FRACTIONAL_DEVIATION_LIMIT,
        VAPOR_FRACTIONAL_DEVIATION_LIMIT,
        VAPOR_FRACTIONAL_WHITE_HALF_WIDTH,
    )
    vapor_contour = ax.contourf(
        distance_grid,
        pressure_bar,
        nh3_vapor,
        levels=vapor_levels,
        cmap=vapor_cmap,
        extend="both",
    )
    cloud_contour = ax.contourf(
        distance_grid,
        pressure_bar,
        np.ma.masked_less(nh3_cloud, CLOUD_LEVELS[0]),
        levels=CLOUD_LEVELS,
        cmap="Blues",
        extend="max",
    )
    ax.contour(
        distance_grid,
        pressure_bar,
        nh3_cloud,
        levels=[CLOUD_LEVELS[0]],
        colors="tab:blue",
        linewidths=0.8,
    )

    displayed = pressure_bar <= MAX_PRESSURE_BAR
    if show_perp_velocity:
        assert perpendicular_velocity is not None
        negative_contour = ax.contour(
            distance_grid,
            pressure_bar,
            perpendicular_velocity,
            levels=NEGATIVE_PERP_VELOCITY_LEVELS,
            colors="black",
            linestyles="dashed",
            linewidths=1.1,
        )
        positive_contour = ax.contour(
            distance_grid,
            pressure_bar,
            perpendicular_velocity,
            levels=POSITIVE_PERP_VELOCITY_LEVELS,
            colors="black",
            linestyles="solid",
            linewidths=1.1,
        )
        ax.clabel(
            negative_contour,
            levels=NEGATIVE_PERP_VELOCITY_LEVELS,
            fmt="%.0f",
            fontsize=7,
            inline=True,
        )
        ax.clabel(
            positive_contour,
            levels=POSITIVE_PERP_VELOCITY_LEVELS,
            fmt="%.0f",
            fontsize=7,
            inline=True,
        )
    if show_vorticity:
        assert vorticity is not None
        scaled_vorticity = vorticity * VORTICITY_SCALE
        negative_contour = ax.contour(
            distance_grid,
            pressure_bar,
            scaled_vorticity,
            levels=NEGATIVE_VORTICITY_LEVELS,
            colors="black",
            linestyles="dashed",
            linewidths=1.0,
        )
        positive_contour = ax.contour(
            distance_grid,
            pressure_bar,
            scaled_vorticity,
            levels=POSITIVE_VORTICITY_LEVELS,
            colors="black",
            linestyles="solid",
            linewidths=1.0,
        )
        ax.clabel(
            negative_contour,
            levels=NEGATIVE_VORTICITY_LEVELS,
            fmt="%.0f",
            fontsize=7,
            inline=True,
        )
        ax.clabel(
            positive_contour,
            levels=POSITIVE_VORTICITY_LEVELS,
            fmt="%.0f",
            fontsize=7,
            inline=True,
        )
    if show_streamfunction:
        assert streamfunction is not None
        streamfunction_anomaly = streamfunction - np.nanmean(streamfunction[displayed])
        streamfunction_levels = symmetric_robust_levels(
            streamfunction_anomaly, displayed
        )
        streamfunction_contour = ax.contour(
            distance_grid,
            pressure_bar,
            streamfunction_anomaly,
            levels=streamfunction_levels,
            colors="black",
            linestyles="dashed",
            linewidths=1.1,
        )
        ax.clabel(
            streamfunction_contour,
            levels=streamfunction_levels[[1, 3, 5]],
            fmt="%.1e",
            fontsize=7,
            inline=True,
        )

    ax.set_xlabel("Distance along x-section [km]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_yscale("log")
    ax.set_ylim(MAX_PRESSURE_BAR, np.nanmin(pressure_bar))

    cloud_colorbar = fig.colorbar(
        cloud_contour,
        cax=cloud_colorbar_ax,
        orientation="vertical",
    )
    cloud_colorbar.set_label("NH$_3$ cloud mass fraction")
    cloud_colorbar.set_ticks(CLOUD_LEVELS)
    cloud_colorbar.set_ticklabels([f"{value:.1e}" for value in CLOUD_LEVELS])
    cloud_colorbar.ax.tick_params(labelsize=9)
    cloud_colorbar.ax.yaxis.label.set_size(9)

    vapor_colorbar = fig.colorbar(
        vapor_contour,
        cax=vapor_colorbar_ax,
        orientation="vertical",
    )
    vapor_colorbar.set_label("NH$_3$ vapor fractional deviation")
    vapor_colorbar.set_ticks([vapor_levels[0], 0.0, vapor_levels[-1]])
    vapor_colorbar.set_ticklabels(
        [f"{vapor_levels[0]:.1e}", "0.0", f"{vapor_levels[-1]:.1e}"]
    )
    vapor_colorbar.ax.tick_params(labelsize=9)
    vapor_colorbar.ax.yaxis.label.set_size(9)

    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.last <= 0:
        raise ValueError("--last must be positive")
    case_dirs = resolve_case_dirs(args.root.expanduser(), args.case_regex)
    if len(case_dirs) != 1:
        raise ValueError("--case-regex must match exactly one case")
    case_dir = case_dirs[0]
    section = cache_path(args.output_dir, case_dir.name, args.last)
    if not section.exists():
        raise FileNotFoundError(
            f"Missing section cache: {section}\nGenerate it with:\n"
            "  python scripts/plot_h2o_on_h2o_min_max_path_cross_section.py "
            f"--case-regex '{args.case_regex}' --last {args.last} "
            "--refresh-cache --cache-only"
        )
    dynamics = args.dynamics_cache or dynamics_cache_path(
        args.output_dir, case_dir.name, args.last
    )
    vorticity = args.vorticity_cache or horizontal_vorticity_cache_path(
        args.output_dir, case_dir.name, args.last
    )
    figure = output_path(args.output_dir, case_dir.name, args.last)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_cache(
        section,
        dynamics,
        vorticity,
        figure,
        args.show_perp_velocity,
        args.show_vorticity,
        args.show_streamfunction,
    )
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
