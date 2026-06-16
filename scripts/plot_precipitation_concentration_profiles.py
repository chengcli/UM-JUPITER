#!/usr/bin/env python3
"""Plot horizontal-mean precipitation concentrations for one CRM case."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-precip-conc")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from case_selection import resolve_case_dirs
from plot_horizontal_mean_profiles import (
    DEFAULT_OUTPUT_DIR,
    PRESSURE_MARKERS_BAR,
    PRESSURE_REFERENCE_PA,
    resolve_field_files,
    select_last_files,
    validate_variables,
)


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASE_REGEX = r"jup_crm3d_H2O-NH3-H2S_F10_nu0\.01"
DEFAULT_LAST = 20
DEFAULT_MAX_PRESSURE_BAR = 50.0
FIELD = "out1"
RHO_VARIABLE = "rho"
PRESSURE_VARIABLE = "press"
LOG_XMIN = 1.0e-12


@dataclass(frozen=True)
class SpeciesSpec:
    precipitation_variable: str
    molar_mass_kg_mol: float
    label: str
    color: str


SPECIES = {
    "H2O": SpeciesSpec("H2O_l_p_", 18.01528e-3, "H$_2$O", "tab:blue"),
    "NH3": SpeciesSpec("NH3_s_p_", 17.03052e-3, "NH$_3$", "tab:orange"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot two horizontal-mean precipitation concentration profiles "
            "[mol m^-3] for H2O and NH3."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-regex", default=DEFAULT_CASE_REGEX)
    parser.add_argument("--last", type=int, default=DEFAULT_LAST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pressure", type=float, default=DEFAULT_MAX_PRESSURE_BAR)
    parser.add_argument("--no-std", action="store_true")
    return parser.parse_args()


def output_path(output_dir: Path, case_name: str, last: int) -> Path:
    return output_dir / f"{case_name}_precipitation_concentration_profiles_last{last}.png"


def cache_path(output_dir: Path, case_name: str, last: int) -> Path:
    return (
        output_dir
        / "precipitation_concentration_profile_cache"
        / f"{case_name}_precipitation_concentration_profiles_last{last}.npz"
    )


def read_snapshot(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with xr.open_dataset(path) as ds:
        rho = ds[RHO_VARIABLE]
        pressure = ds[PRESSURE_VARIABLE]
        if "time" in rho.dims:
            rho = rho.isel(time=0)
            pressure = pressure.isel(time=0)
        pressure_profile = pressure.mean(dim=("x3", "x2")).values.astype(np.float64)
        profiles = {}
        for species, spec in SPECIES.items():
            precip = ds[spec.precipitation_variable]
            if "time" in precip.dims:
                precip = precip.isel(time=0)
            concentration = rho * precip / spec.molar_mass_kg_mol
            profiles[species] = (
                concentration.mean(dim=("x3", "x2")).values.astype(np.float64)
            )
    return pressure_profile, profiles


def build_cache(case_dir: Path, path: Path, last: int) -> None:
    files = select_last_files(resolve_field_files(case_dir, FIELD), last)
    validate_variables(
        files[-1].path,
        [RHO_VARIABLE, PRESSURE_VARIABLE]
        + [spec.precipitation_variable for spec in SPECIES.values()],
    )

    pressure_profiles = []
    species_profiles = {species: [] for species in SPECIES}
    for file_info in files:
        pressure, profiles = read_snapshot(file_info.path)
        pressure_profiles.append(pressure)
        for species, profile in profiles.items():
            species_profiles[species].append(profile)

    pressure_bar = np.mean(pressure_profiles, axis=0) / PRESSURE_REFERENCE_PA
    output = {
        "case_name": np.array(case_dir.name),
        "snapshots": np.asarray([item.snapshot for item in files]),
        "pressure_bar": pressure_bar,
    }
    for species, profiles in species_profiles.items():
        stacked = np.stack(profiles, axis=0)
        output[f"{species}_mean"] = np.mean(stacked, axis=0)
        output[f"{species}_std"] = np.std(stacked, axis=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)


def positive(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def plot_cache(path: Path, figure_path: Path, max_pressure: float, draw_std: bool) -> None:
    with np.load(path, allow_pickle=False) as cache:
        pressure_bar = cache["pressure_bar"]
        snapshots = cache["snapshots"].astype(str)
        profiles = {
            species: (cache[f"{species}_mean"], cache[f"{species}_std"])
            for species in SPECIES
        }

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.0), sharey=True, constrained_layout=True)
    for ax, (species, (mean, std)) in zip(axes, profiles.items()):
        spec = SPECIES[species]
        mean_plot = positive(mean)
        if draw_std:
            lower = positive(mean - std)
            upper = positive(mean + std)
            ax.fill_betweenx(
                pressure_bar,
                lower,
                upper,
                color=spec.color,
                alpha=0.16,
                linewidth=0.0,
            )
        ax.plot(mean_plot, pressure_bar, color=spec.color, lw=2.0)
        ax.set_xscale("log")
        ax.set_xlim(left=LOG_XMIN)
        ax.set_yscale("log")
        ax.set_ylim(max_pressure, np.nanmin(pressure_bar))
        ax.grid(True, which="both", alpha=0.25)
        for pressure in PRESSURE_MARKERS_BAR:
            ax.axhline(pressure, color="0.55", lw=1.5, alpha=0.55, zorder=0)
        ax.set_title(f"{spec.label} precipitation")
        ax.set_xlabel("Concentration [mol m$^{-3}$]")
    axes[0].set_ylabel("Pressure [bar]")
    fig.suptitle(f"Snapshots {snapshots[0]}-{snapshots[-1]}")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_path(args.output_dir, case_dir.name, args.last)
    figure = output_path(args.output_dir, case_dir.name, args.last)
    if not cache.exists():
        build_cache(case_dir, cache, args.last)
        print(f"Wrote cache {cache}")
    else:
        print(f"Using cache {cache}")
    plot_cache(cache, figure, args.max_pressure, not args.no_std)
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
