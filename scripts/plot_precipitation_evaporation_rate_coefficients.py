#!/usr/bin/env python3
"""Plot Kintera precipitation evaporation rate coefficients for one CRM case."""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-evap-rate")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from kintera import Kinetics, KineticsOptions, ThermoOptions, ThermoX

from case_selection import resolve_case_dirs
from plot_horizontal_mean_profiles import (
    DEFAULT_OUTPUT_DIR,
    PRESSURE_MARKERS_BAR,
    PRESSURE_REFERENCE_PA,
    OutputFile,
    resolve_field_files,
    select_last_files,
    validate_variables,
)


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASE_REGEX = r"jup_crm3d_H2O-NH3-H2S_F10_nu0\.01"
DEFAULT_CONFIG = Path("jup_crm3d_H2O-NH3-H2S_F10_nu0.01.yaml")
DEFAULT_LAST = 20
DEFAULT_MAX_PRESSURE_BAR = 50.0
LOG_XMIN = 1.0e-8

OUT1_FIELD = "out1"
OUT2_FIELD = "out2"
RHO_VARIABLE = "rho"
PRESSURE_VARIABLE = "press"
TEMPERATURE_VARIABLE = "temp"

SPECIES_TO_VARIABLE = {
    "H2O": "H2O",
    "NH3": "NH3",
    "H2S": "H2S",
    "H2O(l)": "H2O_l_",
    "H2O(l,p)": "H2O_l_p_",
    "NH3(s)": "NH3_s_",
    "NH3(s,p)": "NH3_s_p_",
    "NH4SH(s)": "NH4SH_s_",
    "NH4SH(s,p)": "NH4SH_s_p_",
}


@dataclass(frozen=True)
class EvaporationSpec:
    reaction: str
    vapor_species: str
    precip_species: str
    label: str
    color: str


SPECIES = {
    "H2O": EvaporationSpec(
        reaction="H2O(l,p) => H2O",
        vapor_species="H2O",
        precip_species="H2O(l,p)",
        label="H$_2$O precipitation",
        color="tab:blue",
    ),
    "NH3": EvaporationSpec(
        reaction="NH3(s,p) => NH3",
        vapor_species="NH3",
        precip_species="NH3(s,p)",
        label="NH$_3$ precipitation",
        color="tab:orange",
    ),
}

RGAS = 8.31446


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call Kintera on CRM snapshots and plot horizontal-mean "
            "evaporation rate coefficients [s^-1] for H2O and NH3 precipitation."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-regex", default=DEFAULT_CASE_REGEX)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--last", type=int, default=DEFAULT_LAST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pressure", type=float, default=DEFAULT_MAX_PRESSURE_BAR)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-std", action="store_true")
    return parser.parse_args()


def output_path(output_dir: Path, case_name: str, last: int) -> Path:
    return output_dir / f"{case_name}_precipitation_evaporation_rate_coefficients_last{last}.png"


def cache_path(output_dir: Path, case_name: str, last: int) -> Path:
    return (
        output_dir
        / "precipitation_evaporation_rate_coefficient_cache"
        / f"{case_name}_precipitation_evaporation_rate_coefficients_last{last}.npz"
    )


def snapshot_map(files: list[OutputFile]) -> dict[str, OutputFile]:
    return {item.snapshot: item for item in files}


def match_out2_files(out1_files: list[OutputFile], out2_files: list[OutputFile]) -> list[OutputFile]:
    by_snapshot = snapshot_map(out2_files)
    missing = [item.snapshot for item in out1_files if item.snapshot not in by_snapshot]
    if missing:
        raise FileNotFoundError(
            "Missing matching out2 temperature file(s) for snapshot(s): "
            + ", ".join(missing)
        )
    return [by_snapshot[item.snapshot] for item in out1_files]


def require_time_singleton(data_array: xr.DataArray, path: Path) -> xr.DataArray:
    if "time" not in data_array.dims:
        return data_array
    if data_array.sizes["time"] != 1:
        raise ValueError(
            f"{data_array.name!r} in {path} has time length "
            f"{data_array.sizes['time']}; expected one time per file"
        )
    return data_array.isel(time=0)


def read_transposed(ds: xr.Dataset, variable: str, path: Path) -> np.ndarray:
    data = require_time_singleton(ds[variable], path)
    for dim in ("x1", "x3", "x2"):
        if dim not in data.dims:
            raise ValueError(f"{variable!r} in {path.name} is missing dimension {dim!r}")
    return data.transpose("x1", "x3", "x2").values.astype(np.float64)


def kintera_indices(
    kinetics_options: KineticsOptions,
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    species_names = list(kinetics_options.species())
    reactions = [reaction_name(item) for item in kinetics_options.reactions()]
    species_index = {name: index for index, name in enumerate(species_names)}
    reaction_index = {reaction: index for index, reaction in enumerate(reactions)}
    missing_species = sorted(set(SPECIES_TO_VARIABLE) - set(species_index))
    if missing_species:
        raise ValueError(
            "Kintera config is missing expected kinetic species: "
            + ", ".join(missing_species)
        )
    missing_reactions = [
        spec.reaction for spec in SPECIES.values() if spec.reaction not in reaction_index
    ]
    if missing_reactions:
        raise ValueError(
            "Kintera config is missing expected evaporation reaction(s): "
            + ", ".join(missing_reactions)
        )
    return species_names, species_index, reaction_index


def reaction_name(reaction: object) -> str:
    text = str(reaction)
    if text.startswith("Reaction(") and text.endswith(")"):
        return text[len("Reaction(") : -1]
    return text


def kinetic_molar_masses(config_path: Path) -> np.ndarray:
    thermo = ThermoX(ThermoOptions.from_yaml(str(config_path)))
    molar_masses = np.asarray(thermo.get_buffer("mu"), dtype=np.float64)
    if molar_masses.size <= len(SPECIES_TO_VARIABLE):
        raise ValueError(
            "Thermo molar-mass buffer does not include all kinetic species. "
            f"Found {molar_masses.size} entries."
        )
    return molar_masses[1:]


def logsvp_ideal(
    temperature_ratio: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    return (1.0 - 1.0 / temperature_ratio) * beta - gamma * np.log(temperature_ratio)


def h2o_ideal_logsvp(temperature: np.ndarray) -> np.ndarray:
    beta_liquid = 24.845
    gamma_liquid = 4.986009
    beta_solid = 22.98
    gamma_solid = 0.52
    triple_temperature = 273.16
    triple_pressure = 611.7
    ratio = temperature / triple_temperature
    return (
        np.where(
            temperature > triple_temperature,
            logsvp_ideal(ratio, beta_liquid, gamma_liquid),
            logsvp_ideal(ratio, beta_solid, gamma_solid),
        )
        + np.log(triple_pressure)
    )


def nh3_ideal_logsvp(temperature: np.ndarray) -> np.ndarray:
    beta_liquid = 20.08
    gamma_liquid = 5.62
    beta_solid = 20.64
    gamma_solid = 1.43
    triple_temperature = 195.4
    triple_pressure = 6060.0
    ratio = temperature / triple_temperature
    return (
        np.where(
            temperature > triple_temperature,
            logsvp_ideal(ratio, beta_liquid, gamma_liquid),
            logsvp_ideal(ratio, beta_solid, gamma_solid),
        )
        + np.log(triple_pressure)
    )


def evaporation_options_by_reaction(
    kinetics_options: KineticsOptions,
) -> dict[str, dict[str, float | str]]:
    evaporation = kinetics_options.evaporation()
    reactions = [reaction_name(item) for item in evaporation.reactions()]
    formulas = list(evaporation.logsvp())
    runtime_formula = formulas[0]
    return {
        reaction: {
            "diff_c": evaporation.diff_c()[index],
            "diff_T": evaporation.diff_T()[index],
            "diff_P": evaporation.diff_P()[index],
            "vm": evaporation.vm()[index],
            "diameter": evaporation.diameter()[index],
            "Tref": evaporation.Tref(),
            "Pref": evaporation.Pref(),
            "formula": formulas[index],
            # Match Kintera's current non-temperature-evolving runtime path:
            # temperature is passed with a singleton reaction dimension, so
            # LogSVP broadcasts the first evaporation formula over all columns.
            "runtime_formula": runtime_formula,
        }
        for index, reaction in enumerate(reactions)
    }


def logsvp_for_formula(formula: str, temperature: np.ndarray) -> np.ndarray:
    if formula == "h2o_ideal":
        return h2o_ideal_logsvp(temperature)
    if formula == "nh3_ideal":
        return nh3_ideal_logsvp(temperature)
    raise ValueError(f"No pure-Python vapor-pressure formula for {formula!r}")


def pure_python_evaporation_rate(
    temperature: np.ndarray,
    pressure: np.ndarray,
    vapor_concentration: np.ndarray,
    parameters: dict[str, float | str],
) -> np.ndarray:
    logsvp = logsvp_for_formula(str(parameters["runtime_formula"]), temperature)

    diffusivity = (
        parameters["diff_c"]
        * (temperature / parameters["Tref"]) ** parameters["diff_T"]
        * (pressure / parameters["Pref"]) ** parameters["diff_P"]
    )
    kappa = 12.0 * diffusivity * parameters["vm"] / parameters["diameter"] ** 2
    saturation_concentration = np.exp(logsvp - np.log(RGAS * temperature))
    deficit = np.maximum(saturation_concentration - vapor_concentration, 0.0)
    return kappa * deficit


def build_concentrations(
    ds: xr.Dataset,
    path: Path,
    rho: np.ndarray,
    species_names: list[str],
    molar_masses: np.ndarray,
) -> np.ndarray:
    concentration = np.empty(rho.shape + (len(species_names),), dtype=np.float64)
    for index, species_name in enumerate(species_names):
        variable = SPECIES_TO_VARIABLE[species_name]
        mass_fraction = read_transposed(ds, variable, path)
        concentration[..., index] = rho * mass_fraction / molar_masses[index]
    return concentration


def read_snapshot_coefficients(
    out1_path: Path,
    out2_path: Path,
    kinetics: Kinetics,
    species_names: list[str],
    species_index: dict[str, int],
    reaction_index: dict[str, int],
    evaporation_parameters: dict[str, dict[str, float]],
    molar_masses: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    with xr.open_dataset(out1_path) as ds1, xr.open_dataset(out2_path) as ds2:
        pressure = read_transposed(ds1, PRESSURE_VARIABLE, out1_path)
        rho = read_transposed(ds1, RHO_VARIABLE, out1_path)
        temperature = read_transposed(ds2, TEMPERATURE_VARIABLE, out2_path)
        concentrations = build_concentrations(
            ds1, out1_path, rho, species_names, molar_masses
        )

    with torch.no_grad():
        rates, _, _ = kinetics.forward_nogil(
            torch.as_tensor(temperature),
            torch.as_tensor(pressure),
            torch.as_tensor(concentrations),
        )
    rates_np = rates.detach().cpu().numpy()

    kintera_profiles = {}
    python_profiles = {}
    stats = {}
    for name, spec in SPECIES.items():
        reactant = concentrations[..., species_index[spec.precip_species]]
        vapor = concentrations[..., species_index[spec.vapor_species]]
        kintera_rate = rates_np[..., reaction_index[spec.reaction]]
        python_rate_coefficient = pure_python_evaporation_rate(
            temperature,
            pressure,
            vapor,
            evaporation_parameters[spec.reaction],
        )
        python_rate = python_rate_coefficient * reactant

        difference = python_rate - kintera_rate
        abs_kintera_rate = np.abs(kintera_rate)
        max_rate = float(np.nanmax(abs_kintera_rate))
        stats[f"{name}_max_abs_rate_difference"] = float(np.nanmax(np.abs(difference)))
        with np.errstate(divide="ignore", invalid="ignore"):
            significant = abs_kintera_rate > max(max_rate * 1.0e-12, 1.0e-300)
            relative = np.abs(difference[significant]) / abs_kintera_rate[significant]
        stats[f"{name}_max_relative_rate_difference"] = (
            float(np.nanmax(relative)) if relative.size else 0.0
        )

        kintera_coefficient = np.full(reactant.shape, np.nan, dtype=np.float64)
        python_coefficient = np.full(reactant.shape, np.nan, dtype=np.float64)
        mask = reactant > 0.0
        kintera_coefficient[mask] = kintera_rate[mask] / reactant[mask]
        python_coefficient[mask] = python_rate_coefficient[mask]
        kintera_profiles[name] = quiet_nanmean(kintera_coefficient, axis=(1, 2))
        python_profiles[name] = quiet_nanmean(python_coefficient, axis=(1, 2))

    pressure_profile = quiet_nanmean(pressure, axis=(1, 2))
    return pressure_profile, kintera_profiles, python_profiles, stats


def build_cache(case_dir: Path, config_path: Path, path: Path, last: int) -> None:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Kintera YAML config does not exist: {config_path}")

    out1_files = select_last_files(resolve_field_files(case_dir, OUT1_FIELD), last)
    out2_files = match_out2_files(out1_files, resolve_field_files(case_dir, OUT2_FIELD))

    validate_variables(
        out1_files[-1].path,
        [RHO_VARIABLE, PRESSURE_VARIABLE] + list(SPECIES_TO_VARIABLE.values()),
    )
    validate_variables(out2_files[-1].path, [TEMPERATURE_VARIABLE])

    kinetics_options = KineticsOptions.from_yaml(str(config_path))
    kinetics = Kinetics(kinetics_options)
    species_names, species_index, reaction_index = kintera_indices(kinetics_options)
    evaporation_parameters = evaporation_options_by_reaction(kinetics_options)
    molar_masses = kinetic_molar_masses(config_path)
    if molar_masses.size < len(species_names):
        raise ValueError(
            f"Kintera config has {len(species_names)} kinetic species, "
            f"but only {molar_masses.size} kinetic molar masses were available."
        )

    pressure_profiles = []
    kintera_profiles = {name: [] for name in SPECIES}
    python_profiles = {name: [] for name in SPECIES}
    rate_stats = {
        f"{name}_max_abs_rate_difference": [] for name in SPECIES
    } | {
        f"{name}_max_relative_rate_difference": [] for name in SPECIES
    }
    for out1_file, out2_file in zip(out1_files, out2_files):
        pressure, kintera, python, stats = read_snapshot_coefficients(
            out1_file.path,
            out2_file.path,
            kinetics,
            species_names,
            species_index,
            reaction_index,
            evaporation_parameters,
            molar_masses,
        )
        pressure_profiles.append(pressure)
        for name, profile in kintera.items():
            kintera_profiles[name].append(profile)
        for name, profile in python.items():
            python_profiles[name].append(profile)
        for name, value in stats.items():
            rate_stats[name].append(value)

    pressure_bar = quiet_nanmean(np.stack(pressure_profiles, axis=0), axis=0) / PRESSURE_REFERENCE_PA
    output = {
        "case_name": np.array(case_dir.name),
        "config": np.array(str(config_path)),
        "snapshots": np.asarray([item.snapshot for item in out1_files]),
        "pressure_bar": pressure_bar,
    }
    for name, profiles in kintera_profiles.items():
        stacked = np.stack(profiles, axis=0)
        output[f"{name}_kintera_mean"] = quiet_nanmean(stacked, axis=0)
        output[f"{name}_kintera_std"] = quiet_nanstd(stacked, axis=0)
    for name, profiles in python_profiles.items():
        stacked = np.stack(profiles, axis=0)
        output[f"{name}_python_mean"] = quiet_nanmean(stacked, axis=0)
        output[f"{name}_python_std"] = quiet_nanstd(stacked, axis=0)
    for name, values in rate_stats.items():
        output[name] = np.asarray(values, dtype=np.float64)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)


def positive(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def quiet_nanmean(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def quiet_nanstd(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Degrees of freedom <= 0 for slice.",
            category=RuntimeWarning,
        )
        return np.nanstd(values, axis=axis)


def finite_log_xlim(profiles: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    finite_positive = []
    for mean, std in profiles.values():
        finite_positive.append(positive(mean))
        finite_positive.append(positive(mean + std))
    values = np.concatenate([item[np.isfinite(item)] for item in finite_positive])
    if values.size == 0:
        return LOG_XMIN, 1.0
    xmax = 10.0 ** np.ceil(np.log10(np.nanmax(values)))
    return LOG_XMIN, max(xmax, LOG_XMIN * 10.0)


def plot_cache(path: Path, figure_path: Path, max_pressure: float, draw_std: bool) -> None:
    with np.load(path, allow_pickle=False) as cache:
        pressure_bar = cache["pressure_bar"]
        snapshots = cache["snapshots"].astype(str)
        profiles = {
            name: {
                "kintera": (cache[f"{name}_kintera_mean"], cache[f"{name}_kintera_std"]),
                "python": (cache[f"{name}_python_mean"], cache[f"{name}_python_std"]),
            }
            for name in SPECIES
        }
        max_abs_difference = {
            name: float(np.nanmax(cache[f"{name}_max_abs_rate_difference"]))
            for name in SPECIES
        }
        max_relative_difference = {
            name: float(np.nanmax(cache[f"{name}_max_relative_rate_difference"]))
            for name in SPECIES
        }

    flat_profiles = {
        f"{name}_{method}": profile
        for name, methods in profiles.items()
        for method, profile in methods.items()
    }
    xmin, xmax = finite_log_xlim(flat_profiles)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.0, 5.0),
        sharey=True,
        constrained_layout=True,
    )
    for ax, (name, methods) in zip(axes, profiles.items()):
        spec = SPECIES[name]
        kintera_mean, kintera_std = methods["kintera"]
        python_mean, _ = methods["python"]
        mean_plot = positive(kintera_mean)
        if draw_std:
            lower = positive(kintera_mean - kintera_std)
            upper = positive(kintera_mean + kintera_std)
            ax.fill_betweenx(
                pressure_bar,
                lower,
                upper,
                color=spec.color,
                alpha=0.16,
                linewidth=0.0,
            )
        ax.plot(mean_plot, pressure_bar, color=spec.color, lw=2.0, label="Kintera")
        ax.plot(
            positive(python_mean),
            pressure_bar,
            color=spec.color,
            lw=2.0,
            ls="--",
            label="Pure Python",
        )
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_yscale("log")
        ax.set_ylim(max_pressure, np.nanmin(pressure_bar))
        ax.grid(True, which="both", alpha=0.25)
        for pressure in PRESSURE_MARKERS_BAR:
            ax.axhline(pressure, color="0.55", lw=1.5, alpha=0.55, zorder=0)
        ax.set_title(spec.label)
        ax.set_xlabel("Evaporation rate coefficient [s$^{-1}$]")
        ax.legend(loc="best", frameon=False, fontsize="small")
    axes[0].set_ylabel("Pressure [bar]")
    fig.suptitle(f"Snapshots {snapshots[0]}-{snapshots[-1]}")
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    for name in SPECIES:
        print(
            f"{name} rate verification: max abs diff "
            f"{max_abs_difference[name]:.6e} mol m^-3 s^-1, max relative diff "
            f"{max_relative_difference[name]:.6e}"
        )


def cache_has_python_verification(path: Path) -> bool:
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as cache:
        return all(
            f"{name}_python_mean" in cache
            and f"{name}_kintera_mean" in cache
            and f"{name}_max_abs_rate_difference" in cache
            for name in SPECIES
        )


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
    if args.refresh_cache or not cache_has_python_verification(cache):
        build_cache(case_dir, args.config, cache, args.last)
        print(f"Wrote cache {cache}")
    else:
        print(f"Using cache {cache}")
    plot_cache(cache, figure, args.max_pressure, not args.no_std)
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
