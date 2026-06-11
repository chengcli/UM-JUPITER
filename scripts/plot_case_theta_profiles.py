#!/usr/bin/env python3
"""Plot combined horizontal-mean potential temperature profiles for CRM cases."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-case-theta")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np

from plot_horizontal_mean_profiles import (
    DEFAULT_MAX_PRESSURE_BAR,
    DEFAULT_OUTPUT_DIR,
    PRESSURE_REFERENCE_PA,
    PRESSURE_MARKERS_BAR,
    read_pressure_profile,
    read_variable_stats,
    resolve_field_files,
    select_last_files,
    validate_variables,
)


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASES = (
    "jup_crm2d_H2O-NH3_F10_nu0.01",
    "jup_crm2d_H2O-NH3_F10_nu0.1",
    "jup_crm2d_H2O-NH3_F10_nu1.0",
    "jup_crm2d_H2O-NH3_F10_nu10.0",
    "jup_crm2d_H2O-NH3_F10_nu100.0",
)
DEFAULT_LAST = 20
DEFAULT_VARIABLE = "theta"
DEFAULT_FIELD = "out2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot horizontal-mean potential temperature profiles from the last "
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


def read_case_profile(
    case_dir: Path,
    last: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    field_files = select_last_files(resolve_field_files(case_dir, DEFAULT_FIELD), last)
    validate_variables(field_files[-1].path, [DEFAULT_VARIABLE])

    pressure_x1, pressure_pa = read_pressure_profile(field_files)
    x1, theta_mean, theta_std = read_variable_stats(
        field_files,
        DEFAULT_VARIABLE,
        Path("jupiter_crm.yaml"),
    )
    if not np.allclose(pressure_x1, x1):
        raise ValueError(f"x1 coordinate mismatch between theta and pressure in {case_dir}")

    pressure_bar = pressure_pa / PRESSURE_REFERENCE_PA
    return pressure_bar, theta_mean, theta_std, [item.snapshot for item in field_files]


def plot_profiles(
    path: Path,
    profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    max_pressure_bar: float,
    draw_std: bool,
) -> None:
    cmap = plt.get_cmap("viridis")          # similar idea to an IDL color table
    norm = mpl.colors.Normalize(vmin=0, vmax=len(profiles) - 1)

    fig, ax = plt.subplots(figsize=(7.0, 8.0), constrained_layout=True)
    #colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (label, (pressure_bar, theta_mean, theta_std)) in enumerate(profiles.items()):
        #color = colors[index % len(colors)]
        color = cmap(norm(index))
        if draw_std:
            ax.fill_betweenx(
                pressure_bar,
                theta_mean - theta_std,
                theta_mean + theta_std,
                color=color,
                alpha=0.16,
                linewidth=0.0,
            )
        ax.plot(theta_mean, pressure_bar, color=color, lw=2.0, label=label)

    ax.set_yscale("log")
    ax.invert_yaxis()
    pressure_bottom, pressure_top = ax.get_ylim()
    ax.set_ylim(min(max_pressure_bar, pressure_bottom), pressure_top)
    ax.set_xlim(158., 172.)
    ax.set_xlabel("Potential temperature [K]")
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
    ax.legend(title="Diffusivity [m$^2$ s$^{-1}$]", loc="best")

    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_csv(
    path: Path,
    profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    labels = list(profiles)
    header = ["level"]
    for label in labels:
        safe_label = safe_name(label)
        header.extend(
            [
                f"{safe_label}_pressure_bar",
                f"{safe_label}_theta_mean",
                f"{safe_label}_theta_std",
            ]
        )

    max_levels = max(values[0].size for values in profiles.values())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for level in range(max_levels):
            row: list[float | int | str] = [level]
            for label in labels:
                pressure_bar, theta_mean, theta_std = profiles[label]
                if level < pressure_bar.size:
                    row.extend(
                        [
                            float(pressure_bar[level]),
                            float(theta_mean[level]),
                            float(theta_std[level]),
                        ]
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
    case_dirs = resolve_case_dirs(root, args.cases)
    profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    snapshots_by_case: dict[str, list[str]] = {}
    for case_dir in case_dirs:
        label = case_label(case_dir)
        pressure_bar, theta_mean, theta_std, snapshots = read_case_profile(
            case_dir,
            args.last,
        )
        profiles[label] = (pressure_bar, theta_mean, theta_std)
        snapshots_by_case[label] = snapshots

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{experiment_name(case_dirs)}_theta_profiles_last{args.last}"
    plot_path = args.output_dir / f"{base_name}.png"
    csv_path = args.output_dir / f"{base_name}.csv"

    print(f"Used {len(case_dirs)} cases from {root}")
    for label, snapshots in snapshots_by_case.items():
        print(f"{label}: snapshots {snapshots[0]}..{snapshots[-1]} ({len(snapshots)})")
    plot_profiles(plot_path, profiles, args.max_pressure, not args.no_std)
    write_csv(csv_path, profiles)
    print(f"Wrote {plot_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
