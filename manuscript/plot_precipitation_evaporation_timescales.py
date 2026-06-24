#!/usr/bin/env python3
"""Plot fast-limit precipitation evaporation timescales along a Jovian adiabat."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-evap-timescale")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml

from plot_horizontal_mean_profiles import PRESSURE_REFERENCE_PA
from plot_precipitation_evaporation_rate_coefficients import (
    pure_python_evaporation_rate,
)


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASE_REGEX = r"jup_crm3d_H2O-NH3-H2S_F10_nu0\.01"
DEFAULT_CONFIG = Path("jup_crm3d_H2O-NH3-H2S_F10_nu0.01.yaml")
DEFAULT_LAST = 20
DEFAULT_OUTPUT = Path("manuscript") / "precipitation_evaporation_timescales_adiabat.png"
DEFAULT_CSV = Path("manuscript") / "precipitation_evaporation_timescales_adiabat.csv"
DEFAULT_DIAMETERS_MM = (1.0, 2.0, 5.0, 10.0)
DEFAULT_PMIN_BAR = 0.1
DEFAULT_PMAX_BAR = 20.0
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        columns = ["pressure_bar", "temperature_K"]
        for species in SPECIES_LABELS:
            for diameter in diameters_mm:
                columns.append(f"{species}_D{diameter:g}mm_timescale_s")
        handle.write(",".join(columns) + "\n")
        for index in range(pressure_bar.size):
            row = [pressure_bar[index], temperature[index]]
            for species in SPECIES_LABELS:
                for diameter in diameters_mm:
                    row.append(timescales[species][diameter][index])
            handle.write(",".join(f"{value:.8e}" for value in row) + "\n")


def plot_timescales(
    output: Path,
    pressure_bar: np.ndarray,
    timescales: dict[str, dict[float, np.ndarray]],
    diameters_mm: list[float],
    sedimentation_timescale: float,
) -> None:
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(diameters_mm)))
    linestyles = {"H2O": "--", "NH3": "-"}
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    for species in SPECIES_LABELS:
        for color, diameter in zip(colors, diameters_mm):
            seconds = timescales[species][diameter]
            ax.plot(
                seconds,
                pressure_bar,
                color=color,
                ls=linestyles[species],
            )
    ax.axvline(
        sedimentation_timescale,
        color="black",
        lw=2.0,
        ls=":",
    )
    ax.set_xscale("log")
    ax.set_xlim(1.0e-2, 1.0e10)
    ax.set_yscale("log")
    ax.set_ylim(np.nanmax(pressure_bar), np.nanmin(pressure_bar))
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("Evaporation timescale [s]")
    ax.set_ylabel("Pressure [bar]")

    diameter_handles = [
        Line2D([0], [0], color=color, lw=2.0, label=f"D = {diameter:g} mm")
        for color, diameter in zip(colors, diameters_mm)
    ]
    species_handles = [
        Line2D([0], [0], color="black", lw=2.0, ls="--", label="H$_2$O"),
        Line2D([0], [0], color="black", lw=2.0, ls="-", label="NH$_3$"),
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.0,
            ls=":",
            label=f"{time_label(sedimentation_timescale)}",
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
    surface_pressure, surface_temperature = problem_state(config)
    pressure_bar = np.geomspace(args.pmin, args.pmax, args.npressure)
    pressure_pa = pressure_bar * PRESSURE_REFERENCE_PA
    temperature = jovian_adiabat(
        pressure_pa, surface_pressure, surface_temperature, args.adiabat_kappa
    )

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
    plot_timescales(
        args.output,
        pressure_bar,
        timescales,
        diameters_mm,
        args.sedimentation_timescale,
    )
    write_csv(
        args.csv,
        pressure_bar,
        temperature,
        diameters_mm,
        timescales,
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
