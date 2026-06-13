"""Reusable cache and plot helpers for species path-defined cross-sections."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from custom_colormaps import diverging_with_white_plateau
from periodic_cross_section import build_periodic_section, sample_periodic_bilinear
from plot_h2o_on_h2o_min_max_path_cross_section import (
    MAX_PRESSURE_BAR,
    NEGATIVE_PERP_VELOCITY_LEVELS,
    POSITIVE_PERP_VELOCITY_LEVELS,
    VAPOR_FRACTIONAL_DEVIATION_LIMIT,
    VAPOR_FRACTIONAL_WHITE_HALF_WIDTH,
    read_dynamics_cache,
    symmetric_robust_levels,
)
from plot_horizontal_mean_profiles import (
    resolve_field_files,
    select_last_files,
    validate_variables,
)


CACHE_VERSION = 1
PRESSURE_REFERENCE_PA = 1.0e5
FIELD_VARIABLES = {
    "h2o_vapor": "H2O",
    "h2o_cloud": "H2O_l_",
    "nh3_vapor": "NH3",
    "nh3_cloud": "NH3_s_",
    "pressure": "press",
}
CLOUD_LEVELS = np.geomspace(1.0e-5, 1.0e-3, 5)
DISPLAY_NAMES = {"H2O": "H$_2$O", "NH3": "NH$_3$"}


def section_cache_path(
    output_dir: Path, case_name: str, path_species: str, last: int
) -> Path:
    lower = path_species.lower()
    return (
        output_dir
        / f"{lower}_min_max_path_cross_section_cache"
        / f"{case_name}_{path_species}_min_max_path_cross_section_last{last}.npz"
    )


def section_dynamics_cache_path(
    output_dir: Path, case_name: str, path_species: str, last: int
) -> Path:
    return (
        output_dir
        / "cross_section_dynamics_cache"
        / f"{case_name}_{path_species}_min_max_path_dynamics_last{last}.npz"
    )


def read_path(path: Path, variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        data = ds[variable]
        if "time" in data.dims:
            data = data.isel(time=0)
        x2_km = ds["x2"].values.astype(np.float64) / 1.0e3
        x3_km = ds["x3"].values.astype(np.float64) / 1.0e3
        values = data.transpose("x3", "x2").values.astype(np.float64)
    return x2_km, x3_km, values


def read_cross_section(
    path: Path, section_x2: np.ndarray, section_x3: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with xr.open_dataset(path) as ds:
        x1_km = ds["x1"].values.astype(np.float64) / 1.0e3
        x2_km = ds["x2"].values.astype(np.float64) / 1.0e3
        x3_km = ds["x3"].values.astype(np.float64) / 1.0e3
        fields = {}
        for kind, variable in FIELD_VARIABLES.items():
            data = ds[variable]
            if "time" in data.dims:
                data = data.isel(time=0)
            fields[kind] = sample_periodic_bilinear(
                data.transpose("x1", "x3", "x2").values.astype(np.float64),
                x2_km,
                x3_km,
                section_x2,
                section_x3,
            )
    return x1_km, fields


def build_cache(case_dir: Path, path: Path, path_species: str, last: int) -> None:
    path_variable = f"path_{path_species}"
    out2_files = select_last_files(resolve_field_files(case_dir, "out2"), last)
    validate_variables(out2_files[-1].path, [path_variable])

    path_sum = None
    reference_x2 = None
    reference_x3 = None
    for file_info in out2_files:
        x2_km, x3_km, values = read_path(file_info.path, path_variable)
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
    min_x3_index, min_x2_index = np.unravel_index(np.nanargmin(mean_path), mean_path.shape)
    max_x3_index, max_x2_index = np.unravel_index(np.nanargmax(mean_path), mean_path.shape)
    section = build_periodic_section(
        reference_x2,
        reference_x3,
        (reference_x2[min_x2_index], reference_x3[min_x3_index]),
        (reference_x2[max_x2_index], reference_x3[max_x3_index]),
    )

    out1_by_snapshot = {item.snapshot: item for item in resolve_field_files(case_dir, "out1")}
    missing = [item.snapshot for item in out2_files if item.snapshot not in out1_by_snapshot]
    if missing:
        raise FileNotFoundError(f"Missing matching out1 snapshots: {', '.join(missing)}")
    selected_out1 = [out1_by_snapshot[item.snapshot] for item in out2_files]
    validate_variables(selected_out1[-1].path, list(FIELD_VARIABLES.values()))

    sums: dict[str, np.ndarray] = {}
    x1_km = None
    for file_info in selected_out1:
        current_x1, fields = read_cross_section(
            file_info.path, section.x2_wrapped, section.x3_wrapped
        )
        if x1_km is None:
            x1_km = current_x1
            sums = {kind: np.zeros_like(values) for kind, values in fields.items()}
        elif not np.allclose(x1_km, current_x1):
            raise ValueError(f"Cross-section coordinates changed in {file_info.path}")
        for kind, values in fields.items():
            sums[kind] += values

    assert x1_km is not None
    means = {kind: total / len(selected_out1) for kind, total in sums.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_version=np.array(CACHE_VERSION),
        case_name=np.array(case_dir.name),
        path_species=np.array(path_species),
        snapshots=np.asarray([item.snapshot for item in out2_files]),
        x1_km=x1_km,
        mean_path=mean_path,
        section_distance_km=section.distance,
        section_x2_unwrapped_km=section.x2_unwrapped,
        section_x3_unwrapped_km=section.x3_unwrapped,
        section_x2_wrapped_km=section.x2_wrapped,
        section_x3_wrapped_km=section.x3_wrapped,
        min_x2_km=np.array(reference_x2[min_x2_index]),
        min_x3_km=np.array(reference_x3[min_x3_index]),
        max_x2_km=np.array(reference_x2[max_x2_index]),
        max_x3_km=np.array(reference_x3[max_x3_index]),
        min_path_kg_m2=np.array(mean_path[min_x3_index, min_x2_index]),
        max_path_kg_m2=np.array(mean_path[max_x3_index, max_x2_index]),
        h2o_vapor_cross_section=means["h2o_vapor"],
        h2o_cloud_cross_section=means["h2o_cloud"],
        nh3_vapor_cross_section=means["nh3_vapor"],
        nh3_cloud_cross_section=means["nh3_cloud"],
        pressure_cross_section_bar=means["pressure"] / PRESSURE_REFERENCE_PA,
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
    figure_path: Path,
    plot_species: str,
    show_cloud: bool,
    show_perp_velocity: bool,
    show_streamfunction: bool,
) -> None:
    prefix = plot_species.lower()
    with np.load(section_path, allow_pickle=False) as cache:
        case_name = str(cache["case_name"])
        snapshots = cache["snapshots"].astype(str)
        distance_km = cache["section_distance_km"]
        pressure_bar = cache["pressure_cross_section_bar"]
        vapor = fractional_deviation(cache[f"{prefix}_vapor_cross_section"])
        cloud = cache[f"{prefix}_cloud_cross_section"]

    perpendicular_velocity = None
    streamfunction = None
    if show_perp_velocity or show_streamfunction:
        perpendicular_velocity, streamfunction = read_dynamics_cache(
            dynamics_path, case_name, snapshots, distance_km, vapor.shape
        )

    fig = plt.figure(figsize=(10.0, 4.5), constrained_layout=True)
    rows = 2 if show_cloud else 1
    grid = fig.add_gridspec(
        rows, 2, width_ratios=(1.0, 0.02), hspace=0.04
    )
    ax = fig.add_subplot(grid[:, 0])
    vapor_colorbar_ax = fig.add_subplot(grid[0, 1])
    cloud_colorbar_ax = fig.add_subplot(grid[1, 1]) if show_cloud else None
    distance_grid = np.broadcast_to(distance_km, pressure_bar.shape)
    vapor_levels = np.linspace(
        -VAPOR_FRACTIONAL_DEVIATION_LIMIT, VAPOR_FRACTIONAL_DEVIATION_LIMIT, 11
    )
    vapor_cmap = diverging_with_white_plateau(
        "PiYG",
        -VAPOR_FRACTIONAL_DEVIATION_LIMIT,
        VAPOR_FRACTIONAL_DEVIATION_LIMIT,
        VAPOR_FRACTIONAL_WHITE_HALF_WIDTH,
    )
    vapor_contour = ax.contourf(
        distance_grid, pressure_bar, vapor, levels=vapor_levels, cmap=vapor_cmap, extend="both"
    )
    cloud_contour = None
    if show_cloud:
        cloud_contour = ax.contourf(
            distance_grid,
            pressure_bar,
            np.ma.masked_less(cloud, CLOUD_LEVELS[0]),
            levels=CLOUD_LEVELS,
            cmap="Blues",
            extend="max",
        )
        ax.contour(
            distance_grid, pressure_bar, cloud, levels=[CLOUD_LEVELS[0]],
            colors="tab:blue", linewidths=0.8,
        )

    displayed = pressure_bar <= MAX_PRESSURE_BAR
    if show_perp_velocity:
        assert perpendicular_velocity is not None
        negative = ax.contour(
            distance_grid, pressure_bar, perpendicular_velocity,
            levels=NEGATIVE_PERP_VELOCITY_LEVELS, colors="black",
            linestyles="dashed", linewidths=1.1,
        )
        positive = ax.contour(
            distance_grid, pressure_bar, perpendicular_velocity,
            levels=POSITIVE_PERP_VELOCITY_LEVELS, colors="black",
            linestyles="solid", linewidths=1.1,
        )
        ax.clabel(negative, levels=NEGATIVE_PERP_VELOCITY_LEVELS, fmt="%.0f", fontsize=7)
        ax.clabel(positive, levels=POSITIVE_PERP_VELOCITY_LEVELS, fmt="%.0f", fontsize=7)
    if show_streamfunction:
        assert streamfunction is not None
        anomaly = streamfunction - np.nanmean(streamfunction[displayed])
        levels = symmetric_robust_levels(anomaly, displayed)
        contour = ax.contour(
            distance_grid, pressure_bar, anomaly, levels=levels,
            colors="black", linestyles="dashed", linewidths=1.1,
        )
        ax.clabel(contour, levels=levels[[1, 3, 5]], fmt="%.1e", fontsize=7)

    ax.set_xlabel("Distance along x-section [km]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_yscale("log")
    ax.set_ylim(MAX_PRESSURE_BAR, np.nanmin(pressure_bar))
    display = DISPLAY_NAMES[plot_species]
    vapor_colorbar = fig.colorbar(vapor_contour, cax=vapor_colorbar_ax)
    vapor_colorbar.set_label(f"{display} vapor fractional deviation")
    vapor_colorbar.set_ticks([vapor_levels[0], 0.0, vapor_levels[-1]])
    vapor_colorbar.ax.tick_params(labelsize=9)
    vapor_colorbar.ax.yaxis.label.set_size(9)
    if show_cloud:
        assert cloud_contour is not None and cloud_colorbar_ax is not None
        cloud_colorbar = fig.colorbar(cloud_contour, cax=cloud_colorbar_ax)
        cloud_colorbar.set_label(f"{display} cloud mass fraction")
        cloud_colorbar.set_ticks(CLOUD_LEVELS)
        cloud_colorbar.set_ticklabels([f"{value:.1e}" for value in CLOUD_LEVELS])
        cloud_colorbar.ax.tick_params(labelsize=9)
        cloud_colorbar.ax.yaxis.label.set_size(9)
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
