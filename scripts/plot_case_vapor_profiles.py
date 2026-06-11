#!/usr/bin/env python3
"""Plot combined horizontal-mean vapor profiles for CRM cases."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-case-vapor")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_horizontal_mean_profiles import (
    DEFAULT_MAX_PRESSURE_BAR,
    DEFAULT_OUTPUT_DIR,
    PRESSURE_MARKERS_BAR,
    PRESSURE_REFERENCE_PA,
    read_pressure_profile,
    read_variable_stats,
    resolve_field_files,
    select_last_files,
    validate_variables,
)
from initial_profile_cache import read_or_create_initial_profile


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASES = (
    "jup_crm2d_H2O-NH3_F10_nu0.01",
    "jup_crm2d_H2O-NH3_F10_nu0.1",
    "jup_crm2d_H2O-NH3_F10_nu1.0",
    "jup_crm2d_H2O-NH3_F10_nu10.0",
    "jup_crm2d_H2O-NH3_F10_nu100.0",
)
DEFAULT_LAST = 20
DEFAULT_FIELD = "out1"
LOG_XMIN = 1.0e-8


@dataclass(frozen=True)
class VaporSpec:
    variable: str
    cloud_variable: str
    precipitation_variable: str
    output_variable: str
    xlabel: str


VAPOR_SPECS = {
    "H2O": VaporSpec("H2O", "H2O_l_", "H2O_l_p_", "H2O", "H$_2$O mass fraction"),
    "NH3": VaporSpec("NH3", "NH3_s_", "NH3_s_p_", "NH3", "NH$_3$ mass fraction"),
    "H2S": VaporSpec("H2S", "NH4SH_s_", "NH4SH_s_p_", "H2S", "H$_2$S mass fraction"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot horizontal-mean H2O and NH3 vapor, cloud, and precipitation profiles from the last "
            "snapshots of all matching JUPITER CRM cases on one log-pressure axis."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory containing case output folders. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(DEFAULT_CASES),
        help="Case directory names to plot.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=DEFAULT_LAST,
        help=f"Number of latest snapshots to average per case. Default: {DEFAULT_LAST}",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        choices=tuple(VAPOR_SPECS),
        default=["H2O", "NH3"],
        help="Species plots to create. Default: H2O NH3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG and CSV outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-pressure",
        type=float,
        default=DEFAULT_MAX_PRESSURE_BAR,
        help=f"Maximum displayed pressure in bar. Default: {DEFAULT_MAX_PRESSURE_BAR:g}",
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Do not draw shaded mean +/- std bands.",
    )
    return parser.parse_args()


def natural_case_key(path: Path) -> tuple[str, float]:
    match = re.search(r"_nu(?P<nu>[0-9.]+)$", path.name)
    if match is None:
        return path.name, np.inf
    return path.name[: match.start()], float(match.group("nu"))


def case_label(path: Path) -> str:
    match = re.search(r"_nu(?P<nu>[0-9.]+)$", path.name)
    if match is None:
        return path.name
    return f"$\\nu=${match.group('nu')}"


def resolve_case_dirs(root: Path, cases: list[str]) -> list[Path]:
    case_dirs = [root / case for case in cases]
    missing = [path for path in case_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Case directories do not exist: " + ", ".join(str(path) for path in missing)
        )
    return case_dirs


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def experiment_name(case_dirs: list[Path]) -> str:
    names = [re.sub(r"_nu[0-9.]+$", "", path.name) for path in case_dirs]
    if len(set(names)) != 1:
        raise ValueError("All cases must share the same experiment prefix before '_nu'")
    return names[0]


def positive_for_log(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def read_case_profile(
    case_dir: Path,
    last: int,
    spec: VaporSpec,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    field_files = select_last_files(resolve_field_files(case_dir, DEFAULT_FIELD), last)
    variables = [spec.variable, spec.cloud_variable, spec.precipitation_variable]
    validate_variables(field_files[-1].path, variables)

    pressure_x1, pressure_pa = read_pressure_profile(field_files)
    profile_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for variable in variables:
        x1, mean, std = read_variable_stats(
            field_files,
            variable,
            Path("jupiter_crm.yaml"),
        )
        if not np.allclose(pressure_x1, x1):
            raise ValueError(
                f"x1 coordinate mismatch between {variable} and pressure in {case_dir}"
            )
        profile_stats[variable] = (mean, std)

    pressure_bar = pressure_pa / PRESSURE_REFERENCE_PA
    return pressure_bar, profile_stats, [item.snapshot for item in field_files]


def plot_profiles(
    path: Path,
    profiles: dict[str, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    initial_profile: tuple[np.ndarray, np.ndarray],
    spec: VaporSpec,
    max_pressure_bar: float,
    draw_std: bool,
) -> None:
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=0, vmax=len(profiles) - 1)

    fig, ax = plt.subplots(figsize=(7.0, 8.0), constrained_layout=True)

    style_by_variable = {
        spec.variable: ("solid", "vapor"),
        spec.cloud_variable: ("--", "cloud"),
        spec.precipitation_variable: (":", "precipitation"),
    }

    for index, (label, (pressure_bar, profile_stats)) in enumerate(profiles.items()):
        color = cmap(norm(index))
        for variable, (linestyle, kind) in style_by_variable.items():
            mean, std = profile_stats[variable]
            plot_mean = positive_for_log(mean)
            if draw_std and variable == spec.variable:
                lower = positive_for_log(mean - std)
                upper = positive_for_log(mean + std)
                ax.fill_betweenx(
                    pressure_bar,
                    lower,
                    upper,
                    color=color,
                    alpha=0.16,
                    linewidth=0.0,
                )
            ax.plot(
                plot_mean,
                pressure_bar,
                color=color,
                linestyle=linestyle,
                lw=2.0,
                label=label if variable == spec.variable else None,
            )

    initial_pressure_bar, initial_vapor = initial_profile
    ax.plot(
        positive_for_log(initial_vapor),
        initial_pressure_bar,
        color="black",
        linestyle="solid",
        lw=2.0,
        label="_nolegend_",
    )

    ax.set_xscale("log")
    ax.set_xlim(left=LOG_XMIN)
    ax.set_yscale("log")
    ax.invert_yaxis()
    pressure_bottom, pressure_top = ax.get_ylim()
    ax.set_ylim(min(max_pressure_bar, pressure_bottom), pressure_top)
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel("Pressure [bar]")
    ax.grid(True, which="both", alpha=0.25)
    for pressure_marker in PRESSURE_MARKERS_BAR:
        ax.axhline(
            pressure_marker,
            color="0.55",
            lw=1.5,
            alpha=0.55,
            zorder=0,
        )
    case_legend = ax.legend(title="Diffusivity [m$^2$ s$^{-1}$]", loc="best")
    ax.add_artist(case_legend)
    style_handles = [
        Line2D([], [], color="0.15", lw=2.0, linestyle="solid", label="vapor"),
        Line2D([], [], color="0.15", lw=2.0, linestyle="--", label="cloud"),
        Line2D(
            [],
            [],
            color="0.15",
            lw=2.0,
            linestyle=":",
            label="precipitation",
        ),
    ]
    ax.legend(handles=style_handles, loc="lower right")

    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_csv(
    path: Path,
    profiles: dict[str, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    spec: VaporSpec,
) -> None:
    labels = list(profiles)
    header = ["level"]
    for label in labels:
        safe_label = safe_name(label)
        header.extend(
            [
                f"{safe_label}_pressure_bar",
                f"{safe_label}_{spec.output_variable}_mean",
                f"{safe_label}_{spec.output_variable}_std",
                f"{safe_label}_{safe_name(spec.cloud_variable)}_mean",
                f"{safe_label}_{safe_name(spec.cloud_variable)}_std",
                f"{safe_label}_{safe_name(spec.precipitation_variable)}_mean",
                f"{safe_label}_{safe_name(spec.precipitation_variable)}_std",
            ]
        )

    max_levels = max(values[0].size for values in profiles.values())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for level in range(max_levels):
            row: list[float | int | str] = [level]
            for label in labels:
                pressure_bar, profile_stats = profiles[label]
                if level < pressure_bar.size:
                    vapor_mean, vapor_std = profile_stats[spec.variable]
                    cloud_mean, cloud_std = profile_stats[spec.cloud_variable]
                    precipitation_mean, precipitation_std = profile_stats[
                        spec.precipitation_variable
                    ]
                    row.extend(
                        [
                            float(pressure_bar[level]),
                            float(vapor_mean[level]),
                            float(vapor_std[level]),
                            float(cloud_mean[level]),
                            float(cloud_std[level]),
                            float(precipitation_mean[level]),
                            float(precipitation_std[level]),
                        ]
                    )
                else:
                    row.extend(["", "", "", "", "", "", ""])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.last <= 0:
        raise ValueError("--last must be positive")
    if args.max_pressure <= 0.0:
        raise ValueError("--max-pressure must be positive")

    root = args.root.expanduser()
    case_dirs = resolve_case_dirs(root, args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Used {len(case_dirs)} cases from {root}")
    for species in args.species:
        spec = VAPOR_SPECS[species]
        profiles: dict[str, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}
        snapshots_by_case: dict[str, list[str]] = {}
        initial_pressure, initial_vapor, initial_cache = read_or_create_initial_profile(
            case_dirs[0],
            spec.variable,
            args.output_dir,
        )
        initial_profile = (initial_pressure, initial_vapor)
        for case_dir in case_dirs:
            label = case_label(case_dir)
            pressure_bar, profile_stats, snapshots = read_case_profile(
                case_dir,
                args.last,
                spec,
            )
            profiles[label] = (pressure_bar, profile_stats)
            snapshots_by_case[label] = snapshots

        base_name = (
            f"{experiment_name(case_dirs)}_{spec.output_variable}_profiles_last{args.last}"
        )
        plot_path = args.output_dir / f"{base_name}.png"
        csv_path = args.output_dir / f"{base_name}.csv"

        plot_profiles(
            plot_path,
            profiles,
            initial_profile,
            spec,
            args.max_pressure,
            not args.no_std,
        )
        write_csv(csv_path, profiles, spec)

        print(f"{spec.variable}:")
        print(f"  Initial profile cache: {initial_cache}")
        for label, snapshots in snapshots_by_case.items():
            print(
                f"  {label}: snapshots {snapshots[0]}..{snapshots[-1]} "
                f"({len(snapshots)})"
            )
        print(f"  Wrote {plot_path}")
        print(f"  Wrote {csv_path}")


if __name__ == "__main__":
    main()
