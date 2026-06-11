#!/usr/bin/env python3
"""Compose cached case profiles into a multi-row H2O/NH3/theta panel grid."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-profile-grid")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np

from initial_profile_cache import initial_profile_cache_path, initial_source_case
from plot_horizontal_mean_profiles import DEFAULT_MAX_PRESSURE_BAR, DEFAULT_OUTPUT_DIR


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
VARIABLES = ("H2O", "NH3", "theta")


@dataclass(frozen=True)
class RowSpec:
    regex: str
    cases: tuple[Path, ...]
    experiment: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a multi-row profile grid. Each repeated --row regex selects "
            "the cases overlaid in one H2O/NH3/theta panel row."
        )
    )
    parser.add_argument(
        "--row",
        action="append",
        required=True,
        help="Extended regular expression matching case folder names for one row.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory containing case output folders. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=20,
        help="Number of latest snapshots represented by profile CSVs. Default: 20",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for profile CSVs and grid output. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-pressure",
        type=float,
        default=DEFAULT_MAX_PRESSURE_BAR,
        help=f"Maximum displayed pressure in bar. Default: {DEFAULT_MAX_PRESSURE_BAR:g}",
    )
    parser.add_argument(
        "--no-initial",
        action="store_true",
        help="Do not draw black snapshot-00000 initial profiles.",
    )
    parser.add_argument(
        "--output-name",
        default="case_profile_grid",
        help="Grid PNG/CSV-list basename. Default: case_profile_grid",
    )
    return parser.parse_args()


def experiment_name(cases: tuple[Path, ...]) -> str:
    experiments = {re.sub(r"_nu[0-9.]+$", "", case.name) for case in cases}
    if len(experiments) != 1:
        raise ValueError(
            "Each --row regex must match exactly one experiment family; matched: "
            + ", ".join(sorted(experiments))
        )
    return experiments.pop()


def case_label(case: Path) -> str:
    match = re.search(r"_nu(?P<nu>[0-9.]+)$", case.name)
    return match.group("nu") if match else case.name


def csv_case_key(case: Path) -> str:
    return f"-nu-{case_label(case)}"


def resolve_rows(root: Path, regexes: list[str]) -> list[RowSpec]:
    names = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name)
    rows = []
    for regex in regexes:
        pattern = re.compile(regex)
        cases = tuple(
            sorted(
                (path for path in names if pattern.fullmatch(path.name)),
                key=lambda path: float(case_label(path)),
            )
        )
        if not cases:
            raise FileNotFoundError(f"No case directories matched --row regex {regex!r}")
        rows.append(RowSpec(regex, cases, experiment_name(cases)))
    return rows


def profile_csv(output_dir: Path, row: RowSpec, variable: str, last: int) -> Path:
    return output_dir / f"{row.experiment}_{variable}_profiles_last{last}.csv"


def required_profile_csvs(
    output_dir: Path,
    row: RowSpec,
    last: int,
) -> dict[str, Path]:
    return {variable: profile_csv(output_dir, row, variable, last) for variable in VARIABLES}


def required_initial_caches(output_dir: Path, row: RowSpec) -> dict[str, Path]:
    source_case = initial_source_case(row.cases[0])
    return {
        variable: initial_profile_cache_path(output_dir, source_case, variable)
        for variable in VARIABLES
    }


def validate_required_caches(
    rows: list[RowSpec],
    row_csvs: list[dict[str, Path]],
    output_dir: Path,
    last: int,
    require_initial: bool,
) -> None:
    missing_by_row: list[tuple[RowSpec, list[Path]]] = []
    for row, csvs in zip(rows, row_csvs):
        required = list(csvs.values())
        if require_initial:
            required.extend(required_initial_caches(output_dir, row).values())
        missing = [path for path in required if not path.exists()]
        if missing:
            missing_by_row.append((row, missing))

    if not missing_by_row:
        return

    lines = ["Required profile caches are missing; no plot was created."]
    for row, missing in missing_by_row:
        lines.append(f"\nRow regex {row.regex!r} is missing:")
        lines.extend(f"  {path}" for path in missing)
        lines.append("Generate this row's caches with:")
        lines.append(
            "  scripts/create_all_plots.sh "
            f"{row.regex!r} --last {last} --output-dir {str(output_dir)!r}"
        )
    raise SystemExit("\n".join(lines))


def read_initial_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as cache:
        return (
            cache["pressure_bar"].astype(np.float64),
            cache["profile"].astype(np.float64),
        )


def read_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if reader.fieldnames is None:
        raise ValueError(f"No CSV header found in {path}")
    return {
        name: np.asarray(
            [float(row[name]) if row[name] else np.nan for row in rows],
            dtype=np.float64,
        )
        for name in reader.fieldnames
    }


def read_case_profile(
    columns: dict[str, np.ndarray],
    case: Path,
    variable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = csv_case_key(case)
    pressure = columns[f"{key}_pressure_bar"]
    mean = columns[f"{key}_{variable}_mean"]
    std = columns[f"{key}_{variable}_std"]
    keep = np.isfinite(pressure) & np.isfinite(mean)
    return pressure[keep], mean[keep], std[keep]


def plot_grid(
    path: Path,
    rows: list[RowSpec],
    row_csvs: list[dict[str, Path]],
    output_dir: Path,
    max_pressure: float,
    draw_initial: bool,
) -> None:
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(12.5, 4.0 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    titles = {"H2O": "H$_2$O", "NH3": "NH$_3$", "theta": "Potential temperature"}
    xlabels = {"H2O": "H$_2$O mass fraction", "NH3": "NH$_3$ mass fraction", "theta": "Potential temperature [K]"}
    species_fields = {
        "H2O": ("H2O", "H2O_l_", "H2O_l_p_"),
        "NH3": ("NH3", "NH3_s_", "NH3_s_p_"),
    }
    species_styles = ("solid", "--", ":")

    for row_index, (row, paths) in enumerate(zip(rows, row_csvs)):
        cmap = plt.get_cmap("viridis")
        norm = mpl.colors.Normalize(vmin=0, vmax=max(len(row.cases) - 1, 1))
        for column_index, variable in enumerate(VARIABLES):
            ax = axes[row_index, column_index]
            columns = read_columns(paths[variable])
            for case_index, case in enumerate(row.cases):
                color = cmap(norm(case_index))
                if variable == "theta":
                    pressure, mean, std = read_case_profile(columns, case, variable)
                    ax.fill_betweenx(
                        pressure,
                        mean - std,
                        mean + std,
                        color=color,
                        alpha=0.14,
                        linewidth=0.0,
                    )
                    ax.plot(mean, pressure, color=color, label=rf"$\kappa={case_label(case)}$")
                else:
                    for field, linestyle in zip(species_fields[variable], species_styles):
                        pressure, mean, std = read_case_profile(columns, case, field)
                        mean = np.where(mean > 0.0, mean, np.nan)
                        if field == variable:
                            lower = np.where(mean - std > 0.0, mean - std, np.nan)
                            upper = np.where(mean + std > 0.0, mean + std, np.nan)
                            ax.fill_betweenx(
                                pressure,
                                lower,
                                upper,
                                color=color,
                                alpha=0.14,
                                linewidth=0.0,
                            )
                        ax.plot(
                            mean,
                            pressure,
                            color=color,
                            linestyle=linestyle,
                            label=rf"$\kappa={case_label(case)}$" if field == variable else None,
                        )

            if draw_initial:
                pressure, profile = read_initial_cache(
                    required_initial_caches(output_dir, row)[variable]
                )
                if variable != "theta":
                    profile = np.where(profile > 0.0, profile, np.nan)
                ax.plot(profile, pressure, color="black")

            if variable != "theta":
                ax.set_xscale("log")
                ax.set_xlim(left=1.0e-8)
            ax.set_yscale("log")
            ax.invert_yaxis()
            bottom, top = ax.get_ylim()
            ax.set_ylim(min(max_pressure, bottom), top)
            ax.grid(True, which="both", alpha=0.25)
            ax.set_xlabel(xlabels[variable])
            if row_index == 0:
                ax.set_title(titles[variable])
            if column_index == 0:
                ax.set_ylabel("Pressure [bar]")
                ax.text(
                    0.03,
                    0.04,
                    row.experiment.removeprefix("jup_"),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=11,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
                )
            else:
                ax.tick_params(labelleft=False)

        axes[row_index, 2].legend(loc="best", fontsize="small")
        axes[row_index, 1].legend(
            handles=[
                mpl.lines.Line2D([], [], color="0.15", linestyle="solid", label="vapor"),
                mpl.lines.Line2D([], [], color="0.15", linestyle="--", label="cloud"),
                mpl.lines.Line2D([], [], color="0.15", linestyle=":", label="precipitation"),
            ],
            loc="best",
            fontsize="small",
        )

    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.last <= 0:
        raise ValueError("--last must be positive")
    if args.max_pressure <= 0.0:
        raise ValueError("--max-pressure must be positive")

    root = args.root.expanduser()
    output_dir = args.output_dir.expanduser()
    rows = resolve_rows(root, args.row)
    row_csvs = [
        required_profile_csvs(output_dir, row, args.last)
        for row in rows
    ]
    validate_required_caches(
        rows,
        row_csvs,
        output_dir,
        args.last,
        not args.no_initial,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / f"{args.output_name}.png"
    plot_grid(
        plot_path,
        rows,
        row_csvs,
        output_dir,
        args.max_pressure,
        not args.no_initial,
    )
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
