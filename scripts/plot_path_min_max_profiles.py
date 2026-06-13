#!/usr/bin/env python3
"""Plot H2O, NH3, and theta profiles at species path extrema."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-path-extrema-profiles")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D

from case_selection import resolve_case_dirs
from initial_profile_cache import read_or_create_initial_profile
from plot_h2o_on_h2o_min_max_path_cross_section import (
    DEFAULT_CASE_REGEX,
    DEFAULT_LAST,
    DEFAULT_ROOT,
    cache_path as section_cache_path,
)
from plot_horizontal_mean_profiles import (
    DEFAULT_OUTPUT_DIR,
    PRESSURE_MARKERS_BAR,
    PRESSURE_REFERENCE_PA,
    resolve_field_files,
)
from species_path_cross_section import section_cache_path as generic_section_cache_path


CACHE_VERSION = 2
MAX_PRESSURE_BAR = 50.0
LOG_XMIN = 1.0e-8
LOCATIONS = ("min", "max")
OUT1_VARIABLES = (
    "press",
    "H2O",
    "H2O_l_",
    "H2O_l_p_",
    "NH3",
    "NH3_s_",
    "NH3_s_p_",
)
OUT2_VARIABLES = ("theta",)
SPECIES = {
    "H2O": ("H2O", "H2O_l_", "H2O_l_p_"),
    "NH3": ("NH3", "NH3_s_", "NH3_s_p_"),
}
FIELD_STYLES = ("solid", "--", ":")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot H2O vapor/cloud/precipitation, NH3 vapor/cloud/precipitation, "
            "and potential-temperature profiles at selected species path extrema."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-regex", default=DEFAULT_CASE_REGEX)
    parser.add_argument("--last", type=int, default=DEFAULT_LAST)
    parser.add_argument(
        "--path-species",
        choices=("H2O", "NH3"),
        default="H2O",
        help="Species whose path minimum and maximum define the locations. Default: H2O.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--section-cache", type=Path, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--max-pressure", type=float, default=MAX_PRESSURE_BAR)
    return parser.parse_args()


def endpoint_cache_path(
    output_dir: Path, case_name: str, path_species: str, last: int
) -> Path:
    return (
        output_dir
        / "path_extrema_profile_cache"
        / f"{case_name}_{path_species}_path_min_max_profiles_last{last}.npz"
    )


def output_path(
    output_dir: Path, case_name: str, path_species: str, last: int
) -> Path:
    return output_dir / f"{case_name}_{path_species}_path_min_max_profiles_last{last}.png"


def cache_is_current(path: Path, path_species: str) -> bool:
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as cache:
        return (
            "path_species" in cache.files
            and int(cache["cache_version"]) == CACHE_VERSION
            and str(cache["path_species"]) == path_species
        )


def read_column(path: Path, variables: tuple[str, ...], x3: int, x2: int) -> dict[str, np.ndarray]:
    with xr.open_dataset(path) as ds:
        columns = {}
        for variable in variables:
            data = ds[variable]
            if "time" in data.dims:
                data = data.isel(time=0)
            columns[variable] = data.isel(x3=x3, x2=x2).values.astype(np.float64)
    return columns


def build_cache(
    case_dir: Path, section_path: Path, path: Path, path_species: str
) -> None:
    if not section_path.exists():
        raise FileNotFoundError(
            f"Missing {path_species} section cache: {section_path}\nGenerate it with:\n"
            f"  python scripts/plot_{path_species.lower()}_on_{path_species.lower()}_min_max_path_cross_section.py "
            f"--case-regex '{case_dir.name}' --refresh-cache --cache-only"
        )
    with np.load(section_path, allow_pickle=False) as section:
        snapshots = section["snapshots"].astype(str).tolist()
        coordinates = {
            location: (
                float(section[f"{location}_x2_km"]),
                float(section[f"{location}_x3_km"]),
            )
            for location in LOCATIONS
        }
        if {"min_x3_index", "min_x2_index", "max_x3_index", "max_x2_index"}.issubset(
            section.files
        ):
            indices = {
                "min": (int(section["min_x3_index"]), int(section["min_x2_index"])),
                "max": (int(section["max_x3_index"]), int(section["max_x2_index"])),
            }
        else:
            indices = None

    files_by_field = {
        field: {item.snapshot: item.path for item in resolve_field_files(case_dir, field)}
        for field in ("out1", "out2")
    }
    for field in files_by_field:
        missing = [snapshot for snapshot in snapshots if snapshot not in files_by_field[field]]
        if missing:
            raise FileNotFoundError(f"Missing {field} snapshots: {', '.join(missing)}")
    if indices is None:
        with xr.open_dataset(files_by_field["out1"][snapshots[0]]) as ds:
            x2_km = ds["x2"].values.astype(np.float64) / 1.0e3
            x3_km = ds["x3"].values.astype(np.float64) / 1.0e3
        indices = {
            location: (
                int(np.argmin(np.abs(x3_km - x3_coordinate))),
                int(np.argmin(np.abs(x2_km - x2_coordinate))),
            )
            for location, (x2_coordinate, x3_coordinate) in coordinates.items()
        }

    sums: dict[str, np.ndarray] = {}
    for snapshot in snapshots:
        for location, (x3_index, x2_index) in indices.items():
            current = {}
            current.update(
                read_column(files_by_field["out1"][snapshot], OUT1_VARIABLES, x3_index, x2_index)
            )
            current.update(
                read_column(files_by_field["out2"][snapshot], OUT2_VARIABLES, x3_index, x2_index)
            )
            for variable, values in current.items():
                key = f"{location}_{variable}"
                if key not in sums:
                    sums[key] = np.zeros_like(values)
                sums[key] += values

    means = {key: values / len(snapshots) for key, values in sums.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_version=np.array(CACHE_VERSION),
        case_name=np.array(case_dir.name),
        path_species=np.array(path_species),
        section_cache=np.array(str(section_path)),
        snapshots=np.asarray(snapshots),
        min_x2_km=np.array(coordinates["min"][0]),
        min_x3_km=np.array(coordinates["min"][1]),
        max_x2_km=np.array(coordinates["max"][0]),
        max_x3_km=np.array(coordinates["max"][1]),
        **means,
    )


def positive(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def configure_pressure_axis(ax: plt.Axes, max_pressure: float) -> None:
    ax.set_yscale("log")
    ax.invert_yaxis()
    bottom, top = ax.get_ylim()
    ax.set_ylim(min(max_pressure, bottom), top)
    ax.grid(True, which="both", alpha=0.25)
    for pressure in PRESSURE_MARKERS_BAR:
        ax.axhline(pressure, color="0.55", lw=1.5, alpha=0.55, zorder=0)


def plot_cache(
    path: Path,
    figure_path: Path,
    max_pressure: float,
    initial_profiles: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    with np.load(path, allow_pickle=False) as cache:
        path_species = str(cache["path_species"])
        profiles = {key: cache[key] for key in cache.files if key.startswith(("min_", "max_"))}

    cmap = plt.get_cmap("PiYG")
    colors = {"min": cmap(0.0), "max": cmap(1.0)}
    display_species = {"H2O": "H$_2$O", "NH3": "NH$_3$"}[path_species]
    labels = {
        "min": f"{display_species} path minimum",
        "max": f"{display_species} path maximum",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), sharey=True, constrained_layout=True)

    for ax, species in zip(axes[:2], ("H2O", "NH3")):
        initial_pressure, initial_profile = initial_profiles[species]
        ax.plot(
            positive(initial_profile),
            initial_pressure,
            color="black",
            lw=2.0,
            zorder=1,
        )
        for location in LOCATIONS:
            pressure = profiles[f"{location}_press"] / PRESSURE_REFERENCE_PA
            for variable, linestyle in zip(SPECIES[species], FIELD_STYLES):
                ax.plot(
                    positive(profiles[f"{location}_{variable}"]),
                    pressure,
                    color=colors[location],
                    linestyle=linestyle,
                    lw=2.0,
                )
        ax.set_xscale("log")
        ax.set_xlim(left=LOG_XMIN)
        ax.set_xlabel(f"{species.replace('H2O', 'H$_2$O').replace('NH3', 'NH$_3$')} mass fraction")
        configure_pressure_axis(ax, max_pressure)

    theta_ax = axes[2]
    initial_pressure, initial_theta = initial_profiles["theta"]
    theta_ax.plot(initial_theta, initial_pressure, color="black", lw=2.0, zorder=1)
    for location in LOCATIONS:
        pressure = profiles[f"{location}_press"] / PRESSURE_REFERENCE_PA
        theta_ax.plot(
            profiles[f"{location}_theta"],
            pressure,
            color=colors[location],
            lw=2.0,
        )
    theta_ax.set_xlabel("Potential temperature [K]")
    configure_pressure_axis(theta_ax, max_pressure)
    axes[0].set_ylabel("Pressure [bar]")

    location_handles = [
        Line2D([], [], color=colors[location], lw=2.0, label=labels[location])
        for location in LOCATIONS
    ]
    style_handles = [
        Line2D([], [], color="0.15", lw=2.0, linestyle="solid", label="vapor"),
        Line2D([], [], color="0.15", lw=2.0, linestyle="--", label="cloud"),
        Line2D([], [], color="0.15", lw=2.0, linestyle=":", label="precipitation"),
    ]
    axes[1].legend(handles=style_handles, loc="best", fontsize="small")
    axes[2].legend(handles=location_handles, loc="best", fontsize="small")
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.last <= 0:
        raise ValueError("--last must be positive")
    if args.max_pressure <= 0.0:
        raise ValueError("--max-pressure must be positive")
    cases = resolve_case_dirs(args.root.expanduser(), args.case_regex)
    if len(cases) != 1:
        raise ValueError("--case-regex must match exactly one case")
    case_dir = cases[0]
    if args.section_cache is not None:
        section = args.section_cache
    elif args.path_species == "H2O":
        section = section_cache_path(args.output_dir, case_dir.name, args.last)
    else:
        section = generic_section_cache_path(
            args.output_dir, case_dir.name, args.path_species, args.last
        )
    cache = endpoint_cache_path(
        args.output_dir, case_dir.name, args.path_species, args.last
    )
    figure = output_path(args.output_dir, case_dir.name, args.path_species, args.last)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.refresh_cache or not cache_is_current(cache, args.path_species):
        build_cache(case_dir, section, cache, args.path_species)
        print(f"Wrote cache {cache}")
    else:
        print(f"Using cache {cache}")
    if args.cache_only:
        return
    initial_profiles = {}
    for variable in ("H2O", "NH3", "theta"):
        pressure, profile, initial_cache = read_or_create_initial_profile(
            case_dir, variable, args.output_dir
        )
        initial_profiles[variable] = (pressure, profile)
        print(f"Initial profile cache: {initial_cache}")
    plot_cache(cache, figure, args.max_pressure, initial_profiles)
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
