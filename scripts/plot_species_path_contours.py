#!/usr/bin/env python3
"""Plot time-mean vapor, cloud, and precipitation path maps."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-species-path")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from case_selection import resolve_case_dirs
from custom_colormaps import diverging_with_white_plateau
from plot_case_vapor_profiles import VAPOR_SPECS
from plot_horizontal_mean_profiles import (
    DEFAULT_OUTPUT_DIR,
    resolve_field_files,
    select_last_files,
    validate_variables,
)


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASE_REGEX = r"jup_crm3d_H2O-NH3-H2S_F10_nu(0\.01|0\.1|1\.0)"
DEFAULT_LAST = 20
DEFAULT_SPECIES = "H2O"
DEFAULT_CMAP = "PiYG"
DEFAULT_WHITE_PLATEAU_FRACTION = 0.08
DEFAULT_FRACTIONAL_ANOMALY = True
FIELD = "out2"
PATH_KINDS = ("vapor", "cloud", "precipitation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one 1x3 vapor/cloud/precipitation path contour figure per "
            "case, averaged over the latest out2 snapshots."
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
        help=f"Full-match regex selecting case directories. Default: {DEFAULT_CASE_REGEX}",
    )
    parser.add_argument(
        "--species",
        choices=tuple(VAPOR_SPECS),
        default=DEFAULT_SPECIES,
        help=f"Species to plot. Default: {DEFAULT_SPECIES}",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=DEFAULT_LAST,
        help=f"Number of latest snapshots to average. Default: {DEFAULT_LAST}",
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP,
        help=f"Matplotlib colormap. Default: {DEFAULT_CMAP}",
    )
    parser.add_argument(
        "--white-plateau-fraction",
        type=float,
        default=DEFAULT_WHITE_PLATEAU_FRACTION,
        help=(
            "Fraction of each panel data range mapped to white at the center. "
            "Use with diverging colormaps such as PiYG. "
            f"Default: {DEFAULT_WHITE_PLATEAU_FRACTION}"
        ),
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=16,
        help="Number of filled contour levels per panel. Default: 16",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--vel1-exceedance-cache",
        type=Path,
        default=None,
        help="Optional NPZ cache whose any_exceedance locations are marked with X symbols.",
    )
    parser.add_argument(
        "--absolute-path",
        action="store_true",
        help=(
            "Plot absolute path values instead of fractional anomaly from each "
            "panel's spatial mean. Default: fractional anomaly."
        ),
    )
    return parser.parse_args()


def path_variables(species: str) -> dict[str, str]:
    spec = VAPOR_SPECS[species]
    return {
        "vapor": f"path_{spec.variable}",
        "cloud": f"path_{spec.cloud_variable}",
        "precipitation": f"path_{spec.precipitation_variable}",
    }


def read_path_fields(
    path: Path,
    variables: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with xr.open_dataset(path) as ds:
        x2 = ds["x2"].values.astype(np.float64) / 1.0e3
        x3 = ds["x3"].values.astype(np.float64) / 1.0e3
        fields = {}
        for kind, variable in variables.items():
            data = ds[variable]
            if "time" in data.dims:
                if data.sizes["time"] != 1:
                    raise ValueError(
                        f"{variable!r} in {path} has {data.sizes['time']} times; expected one"
                    )
                data = data.isel(time=0)
            fields[kind] = data.transpose("x3", "x2").values.astype(np.float64)
    return x2, x3, fields


def read_time_mean_paths(
    case_dir: Path,
    variables: dict[str, str],
    last: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    files = select_last_files(resolve_field_files(case_dir, FIELD), last)
    validate_variables(files[-1].path, list(variables.values()))

    sums: dict[str, np.ndarray] = {}
    reference_x2: np.ndarray | None = None
    reference_x3: np.ndarray | None = None
    for file_info in files:
        x2, x3, fields = read_path_fields(file_info.path, variables)
        if reference_x2 is None:
            reference_x2, reference_x3 = x2, x3
            sums = {kind: np.zeros_like(field) for kind, field in fields.items()}
        elif not np.allclose(reference_x2, x2) or not np.allclose(reference_x3, x3):
            raise ValueError(f"Horizontal coordinates changed in {file_info.path}")
        for kind, field in fields.items():
            sums[kind] += field

    assert reference_x2 is not None and reference_x3 is not None
    means = {kind: total / len(files) for kind, total in sums.items()}
    return reference_x2, reference_x3, means, [item.snapshot for item in files]


def path_fractional_anomaly(values: np.ndarray) -> np.ndarray:
    spatial_mean = float(np.nanmean(values))
    if not np.isfinite(spatial_mean) or np.isclose(spatial_mean, 0.0):
        raise ValueError("Cannot form fractional anomaly with zero/nonfinite spatial mean")
    return values / spatial_mean - 1.0


def contour_levels(values: np.ndarray, count: int, symmetric: bool = False) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Path field contains no finite values")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if symmetric:
        limit = max(abs(lower), abs(upper))
        if np.isclose(limit, 0.0):
            limit = 1.0e-12
        lower = -limit
        upper = limit
    if np.isclose(lower, upper):
        upper = lower + max(abs(lower) * 0.01, 1.0e-12)
    return np.linspace(lower, upper, count)


def panel_colormap(
    cmap: str,
    levels: np.ndarray,
    white_plateau_fraction: float,
) -> str | mpl.colors.ListedColormap:
    if white_plateau_fraction <= 0.0:
        return cmap
    lower = float(levels[0])
    upper = float(levels[-1])
    center = 0.5 * (lower + upper)
    half_width = 0.5 * white_plateau_fraction * (upper - lower)
    return diverging_with_white_plateau(
        cmap,
        lower,
        upper,
        half_width,
        center=center,
        name=f"{cmap}_white_mid_{white_plateau_fraction:g}",
    )


def plot_paths(
    output_path: Path,
    case_name: str,
    species: str,
    x2_km: np.ndarray,
    x3_km: np.ndarray,
    means: dict[str, np.ndarray],
    cmap: str,
    white_plateau_fraction: float,
    fractional_anomaly: bool,
    level_count: int,
    last: int,
    marked_locations: tuple[np.ndarray, np.ndarray] | None,
    marker_threshold: float | None,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.5, 4.8),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(wspace=0.16)
    display_species = {"H2O": "H$_2$O", "NH3": "NH$_3$", "H2S": "H$_2$S"}[species]
    plot_means = (
        {kind: path_fractional_anomaly(values) for kind, values in means.items()}
        if fractional_anomaly
        else means
    )
    for index, (ax, kind) in enumerate(zip(axes, PATH_KINDS)):
        levels = contour_levels(plot_means[kind], level_count, symmetric=fractional_anomaly)
        contour = ax.contourf(
            x2_km,
            x3_km,
            plot_means[kind],
            levels=levels,
            cmap=panel_colormap(cmap, levels, white_plateau_fraction),
        )
        colorbar = fig.colorbar(
            contour,
            ax=ax,
            location="right",
            orientation="vertical",
            fraction=0.046,
            pad=0.018,
        )
        colorbar.set_ticks(np.linspace(levels[0], levels[-1], 5))
        colorbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2g"))
        colorbar.ax.tick_params(labelsize=8, width=1.2, length=3, pad=2)
        ax.set_title(f"{display_species} {kind}")
        ax.set_xlabel("X [km]")
        ax.set_aspect("equal")
        if marked_locations is not None:
            marked_x2, marked_x3 = marked_locations
            ax.scatter(
                marked_x2,
                marked_x3,
                marker="x",
                color="black",
                s=22,
                linewidths=1.0,
                zorder=5,
            )

    axes[0].set_ylabel("Y [km]")
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def read_vel1_exceedance_cache(
    path: Path,
    case_name: str,
    x2_km: np.ndarray,
    x3_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as cache:
        if str(cache["case_name"]) != case_name:
            raise ValueError(f"Velocity cache case mismatch in {path}")
        if not np.allclose(cache["x2_km"], x2_km) or not np.allclose(
            cache["x3_km"], x3_km
        ):
            raise ValueError(f"Velocity cache coordinates do not match {case_name}")
        mask = cache["any_exceedance"].astype(bool)
        threshold = float(
            cache["effective_threshold_m_s"]
            if "effective_threshold_m_s" in cache
            else cache["threshold_m_s"]
        )
    x2_grid, x3_grid = np.meshgrid(x2_km, x3_km)
    return x2_grid[mask], x3_grid[mask], threshold


def main() -> None:
    args = parse_args()
    if args.last <= 0:
        raise ValueError("--last must be positive")
    if args.levels < 2:
        raise ValueError("--levels must be at least 2")
    if args.white_plateau_fraction < 0.0 or args.white_plateau_fraction > 1.0:
        raise ValueError("--white-plateau-fraction must be between 0 and 1")

    root = args.root.expanduser()
    case_dirs = resolve_case_dirs(root, args.case_regex)
    variables = path_variables(args.species)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Matched {len(case_dirs)} cases from {root}")
    for case_dir in case_dirs:
        x2_km, x3_km, means, snapshots = read_time_mean_paths(
            case_dir,
            variables,
            args.last,
        )
        output_path = (
            args.output_dir
            / f"{case_dir.name}_{args.species}_path_contours_last{args.last}.png"
        )
        marked_locations = None
        marker_threshold = None
        if args.vel1_exceedance_cache is not None:
            marked_x2, marked_x3, marker_threshold = read_vel1_exceedance_cache(
                args.vel1_exceedance_cache,
                case_dir.name,
                x2_km,
                x3_km,
            )
            marked_locations = (marked_x2, marked_x3)
        plot_paths(
            output_path,
            case_dir.name,
            args.species,
            x2_km,
            x3_km,
            means,
            args.cmap,
            args.white_plateau_fraction,
            not args.absolute_path,
            args.levels,
            args.last,
            marked_locations,
            marker_threshold,
        )
        print(
            f"{case_dir.name}: snapshots {snapshots[0]}..{snapshots[-1]} "
            f"({len(snapshots)}); wrote {output_path}"
        )


if __name__ == "__main__":
    main()
