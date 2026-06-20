#!/usr/bin/env python3
"""Plot fast-limit precipitation evaporation timescales along a Jovian adiabat."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-evap-timescale")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml

from case_selection import resolve_case_dirs
from plot_horizontal_mean_profiles import DEFAULT_OUTPUT_DIR, PRESSURE_REFERENCE_PA
from plot_horizontal_mean_profiles import (
    PRESSURE_VARIABLE,
    TEMPERATURE_VARIABLE,
    read_pressure_profile,
    read_profile_samples,
    resolve_field_files,
    select_last_files,
    validate_variables,
)
from plot_precipitation_evaporation_rate_coefficients import (
    pure_python_evaporation_rate,
)


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASE_REGEX = r"jup_crm3d_H2O-NH3-H2S_F10_nu0\.01"
DEFAULT_CONFIG = Path("jup_crm3d_H2O-NH3-H2S_F10_nu0.01.yaml")
DEFAULT_LAST = 20
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "precipitation_evaporation_timescales_realistic_temperature.png"
DEFAULT_CSV = DEFAULT_OUTPUT_DIR / "precipitation_evaporation_timescales_realistic_temperature.csv"
DEFAULT_TEMPERATURE_CACHE_DIR = DEFAULT_OUTPUT_DIR / "evaporation_timescale_temperature_cache"
DEFAULT_REALISTIC_CACHE = (
    DEFAULT_OUTPUT_DIR
    / "precipitation_evaporation_rate_coefficient_cache"
    / "jup_crm3d_H2O-NH3-H2S_F10_nu0.01_precipitation_evaporation_rate_coefficients_last20.npz"
)
DEFAULT_DIAMETERS_MM = (1.0, 2.0, 5.0, 10.0)
DEFAULT_PMIN_BAR = 0.1
DEFAULT_PMAX_BAR = 100.0
DEFAULT_NPRESSURE = 300
DEFAULT_ADIABAT_KAPPA = 2.0 / 7.0
DEFAULT_SEDIMENTATION_TIMESCALE_S = 25.0 * 60.0


@dataclass(frozen=True)
class EvaporationParameters:
    formula: str
    diff_c: float
    diff_T: float
    diff_P: float
    vm: float
    Tref: float
    Pref: float


SPECIES_LABELS = {
    "H2O": "H$_2$O",
    "NH3": "NH$_3$",
}

REACTIONS = {
    "H2O": "H2O(l,p) => H2O",
    "NH3": "NH3(s,p) => NH3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot precipitation evaporation timescales for H2O and NH3 along "
            "a dry Jovian adiabat in the maximally subsaturated limit chi_v=0."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-regex", default=DEFAULT_CASE_REGEX)
    parser.add_argument("--last", type=int, default=DEFAULT_LAST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--temperature-cache-dir",
        type=Path,
        default=DEFAULT_TEMPERATURE_CACHE_DIR,
        help="Cache directory for realistic temperature profiles.",
    )
    parser.add_argument("--refresh-temperature-cache", action="store_true")
    parser.add_argument(
        "--use-adiabat",
        action="store_true",
        help="Use the analytic dry Jovian adiabat instead of last-snapshot simulation temperature.",
    )
    parser.add_argument(
        "--realistic-cache",
        type=Path,
        default=DEFAULT_REALISTIC_CACHE,
        help=(
            "NPZ cache from plot_precipitation_evaporation_rate_coefficients.py "
            "to overlay realistic last-snapshot evaporation timescales."
        ),
    )
    parser.add_argument("--no-realistic-overlay", action="store_true")
    parser.add_argument(
        "--diameters-mm",
        type=float,
        nargs="+",
        default=list(DEFAULT_DIAMETERS_MM),
        help="Particle diameters to plot in mm. Default: 1 2 5 10.",
    )
    parser.add_argument("--pmin", type=float, default=DEFAULT_PMIN_BAR)
    parser.add_argument("--pmax", type=float, default=DEFAULT_PMAX_BAR)
    parser.add_argument("--npressure", type=int, default=DEFAULT_NPRESSURE)
    parser.add_argument(
        "--adiabat-kappa",
        type=float,
        default=DEFAULT_ADIABAT_KAPPA,
        help="Dry-adiabat exponent R/cp. Default: 2/7.",
    )
    parser.add_argument(
        "--sedimentation-timescale",
        type=float,
        default=DEFAULT_SEDIMENTATION_TIMESCALE_S,
        help="Reference sedimentation timescale in seconds. Default: 1500.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file does not contain a mapping: {path}")
    return data


def reaction_rate_constants(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reactions = config.get("reactions")
    if not isinstance(reactions, list):
        raise ValueError("YAML config is missing a reactions list")
    result = {}
    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        equation = reaction.get("equation")
        constants = reaction.get("rate-constant")
        if reaction.get("type") == "evaporation" and isinstance(equation, str):
            if not isinstance(constants, dict):
                raise ValueError(f"Evaporation reaction {equation!r} has no rate-constant map")
            result[equation] = constants
    return result


def evaporation_parameters(config: dict[str, Any]) -> dict[str, EvaporationParameters]:
    constants_by_reaction = reaction_rate_constants(config)
    parameters = {}
    for species, reaction in REACTIONS.items():
        if reaction not in constants_by_reaction:
            raise ValueError(f"YAML config is missing evaporation reaction {reaction!r}")
        constants = constants_by_reaction[reaction]
        parameters[species] = EvaporationParameters(
            formula=str(constants["formula"]),
            diff_c=float(constants["diff_c"]),
            diff_T=float(constants.get("diff_T", 0.0)),
            diff_P=float(constants.get("diff_P", 0.0)),
            vm=float(constants["vm"]),
            Tref=float(constants.get("Tref", 300.0)),
            Pref=float(constants.get("Pref", PRESSURE_REFERENCE_PA)),
        )
    return parameters


def problem_state(config: dict[str, Any]) -> tuple[float, float]:
    problem = config.get("problem")
    if not isinstance(problem, dict):
        raise ValueError("YAML config is missing a problem section")
    try:
        surface_pressure = float(problem["Ps"])
        surface_temperature = float(problem["Ts"])
    except KeyError as exc:
        raise ValueError("YAML problem section must contain Ps and Ts") from exc
    return surface_pressure, surface_temperature


def temperature_cache_path(cache_dir: Path, case_name: str, last: int) -> Path:
    return cache_dir / f"{case_name}_realistic_temperature_profile_last{last}.npz"


def build_temperature_cache(case_dir: Path, path: Path, last: int) -> None:
    out2_files = select_last_files(resolve_field_files(case_dir, "out2"), last)
    validate_variables(out2_files[-1].path, [TEMPERATURE_VARIABLE])
    validate_variables(out2_files[-1].path.with_name(
        f"{out2_files[-1].prefix}.out1.{out2_files[-1].snapshot}.nc"
    ), [PRESSURE_VARIABLE])
    _, pressure_pa = read_pressure_profile(out2_files)

    temperature_profiles = []
    snapshots = []
    for file_info in out2_files:
        _, temperature = read_profile_samples(file_info.path, TEMPERATURE_VARIABLE)
        temperature_profiles.append(np.nanmean(temperature, axis=(1, 2)))
        snapshots.append(file_info.snapshot)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        case_name=np.array(case_dir.name),
        snapshots=np.asarray(snapshots),
        pressure_bar=pressure_pa / PRESSURE_REFERENCE_PA,
        temperature_K=np.nanmean(np.stack(temperature_profiles, axis=0), axis=0),
        temperature_std_K=np.nanstd(np.stack(temperature_profiles, axis=0), axis=0),
    )


def read_temperature_cache(path: Path, case_name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        if str(cache["case_name"]) != case_name:
            raise ValueError(f"Temperature cache case mismatch in {path}")
        return cache["pressure_bar"].astype(np.float64), cache["temperature_K"].astype(np.float64)


def jovian_adiabat(
    pressure_pa: np.ndarray,
    surface_pressure_pa: float,
    surface_temperature_k: float,
    kappa: float,
) -> np.ndarray:
    return surface_temperature_k * (pressure_pa / surface_pressure_pa) ** kappa


def evaporation_timescale_seconds(
    temperature: np.ndarray,
    pressure_pa: np.ndarray,
    diameter_m: float,
    parameters: EvaporationParameters,
) -> np.ndarray:
    rate_coefficient = pure_python_evaporation_rate(
        temperature,
        pressure_pa,
        np.zeros_like(temperature, dtype=np.float64),
        {
            "diff_c": parameters.diff_c,
            "diff_T": parameters.diff_T,
            "diff_P": parameters.diff_P,
            "vm": parameters.vm,
            "diameter": diameter_m,
            "Tref": parameters.Tref,
            "Pref": parameters.Pref,
            "formula": parameters.formula,
            "runtime_formula": parameters.formula,
        },
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rate_coefficient > 0.0, 1.0 / rate_coefficient, np.nan)


def time_label(seconds: float) -> str:
    minutes = seconds / 60.0
    if minutes < 120.0:
        return f"{minutes:g} min"
    hours = minutes / 60.0
    if hours < 72.0:
        return f"{hours:g} hr"
    days = hours / 24.0
    return f"{days:g} d"


def write_csv(
    path: Path,
    pressure_bar: np.ndarray,
    temperature: np.ndarray,
    diameters_mm: list[float],
    timescales: dict[str, dict[float, np.ndarray]],
    realistic: dict[str, np.ndarray] | None,
    realistic_pressure_bar: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        columns = ["pressure_bar", "temperature_K"]
        for species in SPECIES_LABELS:
            for diameter in diameters_mm:
                columns.append(f"{species}_D{diameter:g}mm_timescale_s")
        if realistic is not None:
            columns.extend(["realistic_pressure_bar"])
            for species in SPECIES_LABELS:
                columns.append(f"{species}_realistic_timescale_s")
        handle.write(",".join(columns) + "\n")
        for index in range(pressure_bar.size):
            row = [pressure_bar[index], temperature[index]]
            for species in SPECIES_LABELS:
                for diameter in diameters_mm:
                    row.append(timescales[species][diameter][index])
            if realistic is not None and realistic_pressure_bar is not None:
                if index < realistic_pressure_bar.size:
                    row.append(realistic_pressure_bar[index])
                    for species in SPECIES_LABELS:
                        row.append(realistic[species][index])
                else:
                    row.extend([np.nan] * (1 + len(SPECIES_LABELS)))
            handle.write(",".join(f"{value:.8e}" for value in row) + "\n")


def read_realistic_timescales(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing realistic evaporation cache: {path}\n"
            "Generate it with:\n"
            "python scripts/plot_precipitation_evaporation_rate_coefficients.py "
            "--case-regex 'jup_crm3d_H2O-NH3-H2S_F10_nu0\\.01' --last 20"
        )
    with np.load(path, allow_pickle=False) as cache:
        pressure_bar = cache["pressure_bar"].astype(np.float64)
        timescales = {}
        for species in SPECIES_LABELS:
            key = f"{species}_python_mean"
            if key not in cache:
                raise KeyError(f"Realistic evaporation cache is missing {key!r}: {path}")
            coefficient = cache[key].astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                timescales[species] = np.where(coefficient > 0.0, 1.0 / coefficient, np.nan)
    return pressure_bar, timescales


def plot_timescales(
    output: Path,
    pressure_bar: np.ndarray,
    timescales: dict[str, dict[float, np.ndarray]],
    diameters_mm: list[float],
    sedimentation_timescale: float,
    realistic_pressure_bar: np.ndarray | None,
    realistic_timescales: dict[str, np.ndarray] | None,
) -> None:
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(diameters_mm)))
    linestyles = {"H2O": "--", "NH3": "-"}
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for species in SPECIES_LABELS:
        for color, diameter in zip(colors, diameters_mm):
            seconds = timescales[species][diameter]
            ax.plot(
                seconds / 60.0,
                pressure_bar,
                color=color,
                ls=linestyles[species],
            )
    if realistic_timescales is not None and realistic_pressure_bar is not None:
        for species in SPECIES_LABELS:
            ax.plot(
                realistic_timescales[species] / 60.0,
                realistic_pressure_bar,
                color="crimson",
                ls=linestyles[species],
                lw=2.8,
            )
    ax.axvline(
        sedimentation_timescale / 60.0,
        color="black",
        lw=2.0,
        ls=":",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(np.nanmax(pressure_bar), np.nanmin(pressure_bar))
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("Evaporation timescale [min]")
    ax.set_ylabel("Pressure [bar]")

    diameter_handles = [
        Line2D([0], [0], color=color, lw=2.0, label=f"{diameter:g} mm")
        for color, diameter in zip(colors, diameters_mm)
    ]
    species_handles = [
        Line2D([0], [0], color="black", lw=2.0, ls="--", label="H$_2$O"),
        Line2D([0], [0], color="black", lw=2.0, ls="-", label="NH$_3$"),
        Line2D([0], [0], color="crimson", lw=2.8, ls="-", label="realistic last 20"),
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.0,
            ls=":",
            label=f"{time_label(sedimentation_timescale)} sedimentation",
        ),
    ]
    ax.legend(
        handles=diameter_handles + species_handles,
        loc="best",
        frameon=False,
        fontsize="small",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.pmin <= 0.0 or args.pmax <= 0.0 or args.pmin >= args.pmax:
        raise ValueError("--pmin and --pmax must be positive with pmin < pmax")
    if args.npressure < 2:
        raise ValueError("--npressure must be at least 2")
    if any(diameter <= 0.0 for diameter in args.diameters_mm):
        raise ValueError("--diameters-mm values must be positive")
    if args.adiabat_kappa <= 0.0:
        raise ValueError("--adiabat-kappa must be positive")
    if args.last <= 0:
        raise ValueError("--last must be positive")

    config = read_yaml(args.config)
    parameters = evaporation_parameters(config)
    if args.use_adiabat:
        surface_pressure, surface_temperature = problem_state(config)
        pressure_bar = np.geomspace(args.pmin, args.pmax, args.npressure)
        pressure_pa = pressure_bar * PRESSURE_REFERENCE_PA
        temperature = jovian_adiabat(
            pressure_pa, surface_pressure, surface_temperature, args.adiabat_kappa
        )
    else:
        cases = resolve_case_dirs(args.root.expanduser(), args.case_regex)
        if len(cases) != 1:
            raise ValueError("--case-regex must match exactly one case")
        case_dir = cases[0]
        temp_cache = temperature_cache_path(
            args.temperature_cache_dir, case_dir.name, args.last
        )
        if args.refresh_temperature_cache or not temp_cache.exists():
            build_temperature_cache(case_dir, temp_cache, args.last)
            print(f"Wrote temperature cache {temp_cache}")
        else:
            print(f"Using temperature cache {temp_cache}")
        pressure_bar, temperature = read_temperature_cache(temp_cache, case_dir.name)
        keep = (
            np.isfinite(pressure_bar)
            & np.isfinite(temperature)
            & (pressure_bar >= args.pmin)
            & (pressure_bar <= args.pmax)
            & (pressure_bar > 0.0)
        )
        if not np.any(keep):
            raise ValueError("Realistic temperature profile has no levels in requested pressure range")
        pressure_bar = pressure_bar[keep]
        temperature = temperature[keep]
        pressure_pa = pressure_bar * PRESSURE_REFERENCE_PA

    diameters_mm = [float(item) for item in args.diameters_mm]
    timescales = {
        species: {
            diameter: evaporation_timescale_seconds(
                temperature, pressure_pa, diameter * 1.0e-3, species_parameters
            )
            for diameter in diameters_mm
        }
        for species, species_parameters in parameters.items()
    }
    if args.no_realistic_overlay:
        realistic_pressure_bar = None
        realistic_timescales = None
    else:
        realistic_pressure_bar, realistic_timescales = read_realistic_timescales(
            args.realistic_cache
        )

    plot_timescales(
        args.output,
        pressure_bar,
        timescales,
        diameters_mm,
        args.sedimentation_timescale,
        realistic_pressure_bar,
        realistic_timescales,
    )
    write_csv(
        args.csv,
        pressure_bar,
        temperature,
        diameters_mm,
        timescales,
        realistic_timescales,
        realistic_pressure_bar,
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
