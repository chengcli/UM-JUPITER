#!/usr/bin/env python3
"""Plot combined vertical-velocity RMS profiles for CRM cases."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-case-rms-vel1")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np

from case_selection import resolve_case_dirs
from plot_horizontal_mean_profiles import (
    DEFAULT_MAX_PRESSURE_BAR,
    DEFAULT_OUTPUT_DIR,
    PRESSURE_MARKERS_BAR,
    PRESSURE_REFERENCE_PA,
    read_pressure_profile,
    read_profile_samples,
    resolve_field_files,
    select_last_files,
    validate_variables,
)


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASE_REGEX = r"jup_crm2d_H2O-NH3_F10_nu(0\.01|0\.1|1\.0|10\.0|100\.0)"
DEFAULT_LAST = 20
DEFAULT_FIELD = "out1"
VELOCITY_VARIABLE = "vel1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot horizontal root-mean-square vertical-velocity profiles from "
            "the last snapshots of selected JUPITER CRM cases."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory containing case output folders. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--case-regex",
        default=DEFAULT_CASE_REGEX,
        help=f"Full-match regex selecting case directory names. Default: {DEFAULT_CASE_REGEX}",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=DEFAULT_LAST,
        help=f"Number of latest snapshots to include. Default: {DEFAULT_LAST}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG and CSV outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="PNG/CSV basename. Required only to customize mixed-family output names.",
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
        help="Do not draw the temporal standard-deviation shading.",
    )
    return parser.parse_args()


def case_label(path: Path, include_experiment: bool = False) -> str:
    match = re.search(r"_nu(?P<nu>[0-9.]+)$", path.name)
    if match is None:
        return path.name
    if include_experiment:
        experiment = re.match(r"jup_crm(?P<dim>[23])d_.*_F(?P<forcing>[0-9]+)_nu", path.name)
        if experiment is not None:
            return (
                f"{experiment.group('dim')}D F{experiment.group('forcing')}, "
                f"$\\kappa=${match.group('nu')}"
            )
    return f"$\\kappa=${match.group('nu')}"


def experiment_names(case_dirs: list[Path]) -> set[str]:
    names = [re.sub(r"_nu[0-9.]+$", "", path.name) for path in case_dirs]
    return set(names)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def read_case_profile(
    case_dir: Path,
    last: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    files = select_last_files(resolve_field_files(case_dir, DEFAULT_FIELD), last)
    validate_variables(files[-1].path, [VELOCITY_VARIABLE])

    x1_ref: np.ndarray | None = None
    squared_samples = []
    snapshot_rms = []
    for file_info in files:
        x1, values = read_profile_samples(file_info.path, VELOCITY_VARIABLE)
        if x1_ref is None:
            x1_ref = x1
        elif not np.allclose(x1_ref, x1):
            raise ValueError(f"x1 coordinate mismatch in {file_info.path}")
        squared_samples.append(values * values)
        snapshot_rms.append(np.sqrt(np.nanmean(values * values, axis=(1, 2))))

    if x1_ref is None:
        raise ValueError(f"No {VELOCITY_VARIABLE} samples read from {case_dir}")

    rms = np.sqrt(np.nanmean(np.stack(squared_samples, axis=0), axis=(0, 2, 3)))
    rms_std = np.nanstd(np.stack(snapshot_rms, axis=0), axis=0)
    pressure_x1, pressure_pa = read_pressure_profile(files)
    if not np.allclose(x1_ref, pressure_x1):
        raise ValueError(f"x1 coordinate mismatch between velocity and pressure in {case_dir}")
    return (
        pressure_pa / PRESSURE_REFERENCE_PA,
        rms,
        rms_std,
        [file_info.snapshot for file_info in files],
    )


def plot_profiles(
    path: Path,
    profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    max_pressure: float,
    draw_std: bool,
    mixed_experiments: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 8.0), constrained_layout=True)

    if mixed_experiments:
        cmap = plt.get_cmap("YlGnBu")
        forcing_values = sorted(
            {int(match.group(1)) for label in profiles if (match := re.search(r" F([0-9]+),", label))}
        )
        forcing_colors = {
            forcing: cmap(position)
            for forcing, position in zip(
                forcing_values,
                np.linspace(0.55, 0.9, max(len(forcing_values), 2)),
            )
        }
        colors = {
            label: forcing_colors[int(re.search(r" F([0-9]+),", label).group(1))]
            for label in profiles
        }
    else:
        cmap = plt.get_cmap("viridis")
        norm = mpl.colors.Normalize(vmin=0, vmax=max(len(profiles) - 1, 1))
        colors = {
            label: cmap(norm(index)) for index, label in enumerate(profiles)
        }

    for index, (label, (pressure_bar, rms, rms_std)) in enumerate(profiles.items()):
        color = colors[label]
        if draw_std:
            ax.fill_betweenx(
                pressure_bar,
                np.maximum(rms - rms_std, 0.0),
                rms + rms_std,
                color=color,
                alpha=0.16,
                linewidth=0.0,
            )
        linestyle = "--" if mixed_experiments and label.startswith("2D") else "-"
        ax.plot(rms, pressure_bar, color=color, linestyle=linestyle, label=label)

    ax.set_yscale("log")
    ax.invert_yaxis()
    bottom, top = ax.get_ylim()
    ax.set_ylim(min(max_pressure, bottom), top)
    ax.set_xlabel(r"RMS vertical velocity [m s$^{-1}$]")
    ax.set_ylabel("Pressure [bar]")
    ax.grid(True, which="both", alpha=0.25)
    for pressure_marker in PRESSURE_MARKERS_BAR:
        ax.axhline(pressure_marker, color="0.55", lw=1.5, alpha=0.55, zorder=0)
    legend_title = "Case" if mixed_experiments else "Diffusivity [m$^2$ s$^{-1}$]"
    ax.legend(title=legend_title, loc="best")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_csv(
    path: Path,
    profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    labels = list(profiles)
    header = ["level"]
    for label in labels:
        name = safe_name(label)
        header.extend(
            [
                f"{name}_pressure_bar",
                f"{name}_rms_vel1",
                f"{name}_rms_vel1_temporal_std",
            ]
        )

    max_levels = max(profile[0].size for profile in profiles.values())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for level in range(max_levels):
            row: list[float | int | str] = [level]
            for pressure, rms, rms_std in profiles.values():
                if level < pressure.size:
                    row.extend(
                        [float(pressure[level]), float(rms[level]), float(rms_std[level])]
                    )
                else:
                    row.extend(["", "", ""])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.last <= 0:
        raise ValueError("--last must be positive")
    if args.max_pressure <= 0.0:
        raise ValueError("--max-pressure must be positive")

    root = args.root.expanduser()
    case_dirs = resolve_case_dirs(root, args.case_regex)
    experiments = experiment_names(case_dirs)
    mixed_experiments = len(experiments) > 1
    profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    snapshots_by_case: dict[str, list[str]] = {}
    for case_dir in case_dirs:
        label = case_label(case_dir, mixed_experiments)
        pressure, rms, rms_std, snapshots = read_case_profile(case_dir, args.last)
        profiles[label] = (pressure, rms, rms_std)
        snapshots_by_case[label] = snapshots

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name is not None:
        base_name = args.output_name
    elif mixed_experiments:
        base_name = f"mixed_cases_rms_vel1_profiles_last{args.last}"
    else:
        base_name = f"{next(iter(experiments))}_rms_vel1_profiles_last{args.last}"
    plot_path = args.output_dir / f"{base_name}.png"
    csv_path = args.output_dir / f"{base_name}.csv"
    plot_profiles(
        plot_path,
        profiles,
        args.max_pressure,
        not args.no_std,
        mixed_experiments,
    )
    write_csv(csv_path, profiles)

    print(f"Used {len(case_dirs)} cases from {root}")
    for label, snapshots in snapshots_by_case.items():
        print(f"{label}: snapshots {snapshots[0]}..{snapshots[-1]} ({len(snapshots)})")
    print(f"Wrote {plot_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
