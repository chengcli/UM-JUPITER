#!/usr/bin/env python3
"""Cache and plot a periodic H2O cross-section through path extrema."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-h2o-cross-section")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from case_selection import resolve_case_dirs
from cache_cross_section_dynamics import dynamics_cache_path
from custom_colormaps import diverging_with_white_plateau
from plot_horizontal_mean_profiles import (
    DEFAULT_OUTPUT_DIR,
    resolve_field_files,
    select_last_files,
    validate_variables,
)
from periodic_cross_section import build_periodic_section, sample_periodic_bilinear


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASE_REGEX = r"jup_crm3d_H2O-NH3-H2S_F10_nu0\.01"
DEFAULT_LAST = 20
CACHE_VERSION = 4
PATH_VARIABLE = "path_H2O"
VARIABLES = {
    "vapor": "H2O",
    "nh3_vapor": "NH3",
    "cloud": "H2O_l_",
    "pressure": "press",
}
PRESSURE_REFERENCE_PA = 1.0e5
MAX_PRESSURE_BAR = 50.0
VAPOR_FRACTIONAL_DEVIATION_LIMIT = 0.5
VAPOR_FRACTIONAL_WHITE_HALF_WIDTH = 0.05
NEGATIVE_PERP_VELOCITY_LEVELS = np.array([-20.0, -16.0, -12.0, -8.0, -4.0])
POSITIVE_PERP_VELOCITY_LEVELS = np.array([4.0, 8.0, 12.0, 16.0, 20.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the minimum and maximum time-mean H2O vapor path, cache the "
            "periodic vertical cross-section through both, and plot H2O vapor/cloud."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-regex", default=DEFAULT_CASE_REGEX)
    parser.add_argument("--last", type=int, default=DEFAULT_LAST)
    parser.add_argument("--levels", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dynamics-cache",
        type=Path,
        default=None,
        help="Projected velocity and streamfunction cache. Default: inferred from output directory.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-read NetCDF snapshots and overwrite an existing cache.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Build or reuse the section cache without rendering the figure.",
    )
    parser.add_argument(
        "--show-perp-velocity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay perpendicular-velocity contours. Default: enabled.",
    )
    parser.add_argument(
        "--show-streamfunction",
        action="store_true",
        help="Overlay dashed mass-streamfunction contours. Default: disabled.",
    )
    return parser.parse_args()


def cache_path(output_dir: Path, case_name: str, last: int) -> Path:
    return (
        output_dir
        / "h2o_min_max_path_cross_section_cache"
        / f"{case_name}_H2O_min_max_path_cross_section_last{last}.npz"
    )


def output_path(output_dir: Path, case_name: str, last: int) -> Path:
    return output_dir / f"{case_name}_H2O_min_max_path_cross_section_last{last}.png"


def symmetric_robust_levels(values: np.ndarray, mask: np.ndarray, count: int = 7) -> np.ndarray:
    finite = np.abs(values[mask & np.isfinite(values)])
    if finite.size == 0:
        raise ValueError("No finite displayed values are available for contour levels")
    limit = float(np.quantile(finite, 0.98))
    if limit == 0.0:
        limit = float(np.max(finite))
    if limit == 0.0:
        limit = 1.0e-12
    return np.linspace(-limit, limit, count)


def read_dynamics_cache(
    path: Path,
    case_name: str,
    snapshots: np.ndarray,
    distance_km: np.ndarray,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cross-section dynamics cache: {path}\nGenerate it with:\n"
            f"  python scripts/cache_cross_section_dynamics.py "
            f"--case-regex '{case_name}' --last {len(snapshots)}"
        )
    with np.load(path, allow_pickle=False) as cache:
        if int(cache["cache_version"]) != 1:
            raise ValueError(f"Unsupported dynamics cache version in {path}")
        if str(cache["case_name"]) != case_name:
            raise ValueError(f"Dynamics cache case mismatch in {path}")
        if not np.array_equal(cache["snapshots"].astype(str), snapshots.astype(str)):
            raise ValueError(f"Dynamics cache snapshots do not match section cache: {path}")
        if not np.allclose(cache["distance_km"], distance_km):
            raise ValueError(f"Dynamics cache section coordinates do not match: {path}")
        perpendicular = cache["perpendicular_velocity"]
        streamfunction = cache["mass_streamfunction"]
    if perpendicular.shape != expected_shape or streamfunction.shape != expected_shape:
        raise ValueError(f"Dynamics cache field shape does not match section cache: {path}")
    return perpendicular, streamfunction


def read_path(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        data = ds[PATH_VARIABLE]
        if "time" in data.dims:
            data = data.isel(time=0)
        x2_km = ds["x2"].values.astype(np.float64) / 1.0e3
        x3_km = ds["x3"].values.astype(np.float64) / 1.0e3
        values = data.transpose("x3", "x2").values.astype(np.float64)
    return x2_km, x3_km, values


def read_cross_section(
    path: Path,
    section_x2: np.ndarray,
    section_x3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with xr.open_dataset(path) as ds:
        x1_km = ds["x1"].values.astype(np.float64) / 1.0e3
        x2_km = ds["x2"].values.astype(np.float64) / 1.0e3
        x3_km = ds["x3"].values.astype(np.float64) / 1.0e3
        fields = {}
        for kind, variable in VARIABLES.items():
            data = ds[variable]
            if "time" in data.dims:
                data = data.isel(time=0)
            values = data.transpose("x1", "x3", "x2").values.astype(np.float64)
            fields[kind] = sample_periodic_bilinear(
                values,
                x2_km,
                x3_km,
                section_x2,
                section_x3,
            )
    return x1_km, x2_km, fields


def build_cache(case_dir: Path, path: Path, last: int) -> None:
    out2_files = select_last_files(resolve_field_files(case_dir, "out2"), last)
    validate_variables(out2_files[-1].path, [PATH_VARIABLE])

    path_sum = None
    reference_x2 = None
    reference_x3 = None
    for file_info in out2_files:
        x2_km, x3_km, values = read_path(file_info.path)
        if path_sum is None:
            reference_x2, reference_x3 = x2_km, x3_km
            path_sum = np.zeros_like(values)
        elif not np.allclose(reference_x2, x2_km) or not np.allclose(
            reference_x3, x3_km
        ):
            raise ValueError(f"Horizontal coordinates changed in {file_info.path}")
        path_sum += values

    assert path_sum is not None and reference_x2 is not None and reference_x3 is not None
    mean_path = path_sum / len(out2_files)
    min_x3_index, min_x2_index = np.unravel_index(
        np.nanargmin(mean_path), mean_path.shape
    )
    max_x3_index, max_x2_index = np.unravel_index(
        np.nanargmax(mean_path), mean_path.shape
    )
    section = build_periodic_section(
        reference_x2,
        reference_x3,
        (reference_x2[min_x2_index], reference_x3[min_x3_index]),
        (reference_x2[max_x2_index], reference_x3[max_x3_index]),
    )

    out1_by_snapshot = {
        item.snapshot: item for item in resolve_field_files(case_dir, "out1")
    }
    missing = [item.snapshot for item in out2_files if item.snapshot not in out1_by_snapshot]
    if missing:
        raise FileNotFoundError(
            f"Missing matching out1 snapshots in {case_dir}: {', '.join(missing)}"
        )
    selected_out1 = [out1_by_snapshot[item.snapshot] for item in out2_files]
    validate_variables(selected_out1[-1].path, list(VARIABLES.values()))

    sums: dict[str, np.ndarray] = {}
    x1_km = None
    for file_info in selected_out1:
        current_x1, _, fields = read_cross_section(
            file_info.path,
            section.x2_wrapped,
            section.x3_wrapped,
        )
        if x1_km is None:
            x1_km = current_x1
            sums = {kind: np.zeros_like(values) for kind, values in fields.items()}
        elif not np.allclose(x1_km, current_x1):
            raise ValueError(f"Cross-section coordinates changed in {file_info.path}")
        for kind, values in fields.items():
            sums[kind] += values

    assert x1_km is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_version=np.array(CACHE_VERSION),
        case_name=np.array(case_dir.name),
        snapshots=np.asarray([item.snapshot for item in out2_files]),
        x1_km=x1_km,
        x2_km=reference_x2,
        x3_km=reference_x3,
        mean_h2o_path=mean_path,
        min_x2_index=np.array(min_x2_index),
        min_x3_index=np.array(min_x3_index),
        max_x2_index=np.array(max_x2_index),
        max_x3_index=np.array(max_x3_index),
        min_x2_km=np.array(reference_x2[min_x2_index]),
        min_x3_km=np.array(reference_x3[min_x3_index]),
        max_x2_km=np.array(reference_x2[max_x2_index]),
        max_x3_km=np.array(reference_x3[max_x3_index]),
        min_path_kg_m2=np.array(mean_path[min_x3_index, min_x2_index]),
        max_path_kg_m2=np.array(mean_path[max_x3_index, max_x2_index]),
        section_distance_km=section.distance,
        section_x2_unwrapped_km=section.x2_unwrapped,
        section_x3_unwrapped_km=section.x3_unwrapped,
        section_x2_wrapped_km=section.x2_wrapped,
        section_x3_wrapped_km=section.x3_wrapped,
        min_distance_km=np.array(section.start_distance),
        max_distance_km=np.array(section.end_distance),
        x2_period_km=np.array(section.x2_period),
        x3_period_km=np.array(section.x3_period),
        vapor_cross_section=sums["vapor"] / len(selected_out1),
        nh3_vapor_cross_section=sums["nh3_vapor"] / len(selected_out1),
        cloud_cross_section=sums["cloud"] / len(selected_out1),
        pressure_cross_section_bar=(
            sums["pressure"] / len(selected_out1) / PRESSURE_REFERENCE_PA
        ),
    )


def plot_cache(
    path: Path,
    dynamics_path: Path,
    figure_path: Path,
    levels_count: int,
    show_perp_velocity: bool,
    show_streamfunction: bool,
) -> None:
    with np.load(path, allow_pickle=False) as cache:
        case_name = str(cache["case_name"])
        snapshots = cache["snapshots"].astype(str)
        distance_km = cache["section_distance_km"]
        pressure_bar = cache["pressure_cross_section_bar"]
        min_x2_km = float(cache["min_x2_km"])
        min_x3_km = float(cache["min_x3_km"])
        max_x2_km = float(cache["max_x2_km"])
        max_x3_km = float(cache["max_x3_km"])
        min_path = float(cache["min_path_kg_m2"])
        max_path = float(cache["max_path_kg_m2"])
        min_distance = float(cache["min_distance_km"])
        max_distance = float(cache["max_distance_km"])
        fields = {
            "vapor": cache["vapor_cross_section"],
            "nh3_vapor": cache["nh3_vapor_cross_section"],
            "cloud": cache["cloud_cross_section"],
        }
    perpendicular_velocity = None
    streamfunction = None
    if show_perp_velocity or show_streamfunction:
        perpendicular_velocity, streamfunction = read_dynamics_cache(
            dynamics_path,
            case_name,
            snapshots,
            distance_km,
            fields["vapor"].shape,
        )
    vapor_horizontal_mean = np.nanmean(fields["vapor"], axis=1, keepdims=True)
    fields["vapor"] = np.divide(
        fields["vapor"] - vapor_horizontal_mean,
        vapor_horizontal_mean,
        out=np.full_like(fields["vapor"], np.nan),
        where=vapor_horizontal_mean > 0.0,
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
        fields["vapor"],
        levels=vapor_levels,
        cmap=vapor_cmap,
        extend="both",
    )
    cloud_levels = np.geomspace(1.0e-5, 1.0e-3, 5)
    cloud_contour = ax.contourf(
        distance_grid,
        pressure_bar,
        np.ma.masked_less(fields["cloud"], cloud_levels[0]),
        levels=cloud_levels,
        cmap="Blues",
        extend="max",
    )
    ax.contour(
        distance_grid,
        pressure_bar,
        fields["cloud"],
        levels=[1.0e-5],
        colors="tab:blue",
        linewidths=0.8,
    )
    displayed = pressure_bar <= MAX_PRESSURE_BAR
    if show_perp_velocity:
        assert perpendicular_velocity is not None
        negative_velocity_contour = ax.contour(
            distance_grid,
            pressure_bar,
            perpendicular_velocity,
            levels=NEGATIVE_PERP_VELOCITY_LEVELS,
            colors="black",
            linestyles="dashed",
            linewidths=1.1,
        )
        positive_velocity_contour = ax.contour(
            distance_grid,
            pressure_bar,
            perpendicular_velocity,
            levels=POSITIVE_PERP_VELOCITY_LEVELS,
            colors="black",
            linestyles="solid",
            linewidths=1.1,
        )
        ax.clabel(
            negative_velocity_contour,
            levels=NEGATIVE_PERP_VELOCITY_LEVELS,
            fmt="%.0f",
            fontsize=7,
            inline=True,
        )
        ax.clabel(
            positive_velocity_contour,
            levels=POSITIVE_PERP_VELOCITY_LEVELS,
            fmt="%.0f",
            fontsize=7,
            inline=True,
        )
    if show_streamfunction:
        assert streamfunction is not None
        streamfunction_display_mean = np.nanmean(streamfunction[displayed])
        streamfunction_anomaly = streamfunction - streamfunction_display_mean
        streamfunction_levels = symmetric_robust_levels(streamfunction_anomaly, displayed)
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
    cloud_colorbar.set_label("H$_2$O cloud mass fraction")
    cloud_colorbar.set_ticks(cloud_levels)
    cloud_colorbar.set_ticklabels([f"{value:.1e}" for value in cloud_levels])
    cloud_colorbar.ax.tick_params(labelsize=9)
    cloud_colorbar.ax.yaxis.label.set_size(9)
    vapor_colorbar = fig.colorbar(
        vapor_contour,
        cax=vapor_colorbar_ax,
        orientation="vertical",
    )
    vapor_colorbar.set_label("H$_2$O vapor fractional deviation")
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
    if args.levels < 2:
        raise ValueError("--levels must be at least 2")

    case_dirs = resolve_case_dirs(args.root.expanduser(), args.case_regex)
    if len(case_dirs) != 1:
        raise ValueError("--case-regex must match exactly one case")
    case_dir = case_dirs[0]
    cache = cache_path(args.output_dir, case_dir.name, args.last)
    dynamics = args.dynamics_cache or dynamics_cache_path(
        args.output_dir, case_dir.name, args.last
    )
    figure = output_path(args.output_dir, case_dir.name, args.last)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_cache or not cache.exists():
        build_cache(case_dir, cache, args.last)
        print(f"Wrote cache {cache}")
    else:
        print(f"Using cache {cache}")
    if args.cache_only:
        return
    plot_cache(
        cache,
        dynamics,
        figure,
        args.levels,
        args.show_perp_velocity,
        args.show_streamfunction,
    )
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
