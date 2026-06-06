#!/usr/bin/env python3
"""Plot stacked temperature-gradient time series at fixed pressure levels."""

from __future__ import annotations

import argparse
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-dtdz-stacked")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_time_evolution import (
    DEFAULT_OUTPUT_DIR,
    PRESSURE_REFERENCE_PA,
    PRESSURE_VARIABLE,
    interpolate_to_pressures,
    pressure_file_for,
    read_horizontal_mean_profile,
    read_time_days,
    resolve_field_files,
    select_snapshots,
    validate_variables,
)
from plot_horizontal_mean_profiles import read_profile_samples


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASES = (
    "jup_crm2d_H2O-NH3_F100_nu10.0",
    "jup_crm2d_H2O-NH3_F100_nu100.0",
    "jup_crm2d_H2O-NH3_F100_nu1000.0",
)
FIELD = "out2"
TEMPERATURE_VARIABLE = "temp"
PRESSURE_LEVELS_BAR = (1.0, 6.0, 30.0)
CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot stacked time series of the vertical temperature gradient at "
            "1, 6, and 30 bar for selected CRM cases."
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
        "--first",
        type=int,
        default=None,
        help="Use the first N snapshots per case. Cannot be combined with --last.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Use the last N snapshots per case. Cannot be combined with --first.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG and CSV outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for per-case NPZ caches. Default: <output-dir>/dtdz_cache",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-read all NetCDF files and overwrite existing per-case caches.",
    )
    args = parser.parse_args()
    if args.first is not None and args.last is not None:
        parser.error("--first and --last cannot be specified together")
    return args


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


def snapshot_suffix(first: int | None, last: int | None) -> str:
    if first is not None:
        return f"first{first}"
    if last is not None:
        return f"last{last}"
    return "all"


def read_dtdz_time_series(
    case_dir: Path,
    first: int | None,
    last: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    files = select_snapshots(resolve_field_files(case_dir, FIELD), first, last)
    validate_variables(files[-1].path, [TEMPERATURE_VARIABLE])

    times_days = []
    dtdz_by_pressure = []
    for file_info in files:
        pressure_path = pressure_file_for(file_info)
        if not pressure_path.exists():
            raise FileNotFoundError(
                f"Missing pressure file for {file_info.path.name}: {pressure_path}"
            )

        pressure_profile_bar = (
            read_horizontal_mean_profile(pressure_path, PRESSURE_VARIABLE)
            / PRESSURE_REFERENCE_PA
        )
        x1_m, temperature_samples = read_profile_samples(
            file_info.path,
            TEMPERATURE_VARIABLE,
        )
        temperature_profile = np.nanmean(temperature_samples, axis=(1, 2))
        dtdz_k_per_km = np.gradient(temperature_profile, x1_m) * 1000.0
        dtdz_by_pressure.append(
            interpolate_to_pressures(
                pressure_profile_bar,
                dtdz_k_per_km,
                PRESSURE_LEVELS_BAR,
            )
        )
        times_days.append(read_time_days(file_info.path))

    times = np.asarray(times_days, dtype=np.float64)
    values = np.asarray(dtdz_by_pressure, dtype=np.float64)
    if not np.isfinite(times).all():
        times = np.arange(len(files), dtype=np.float64)
    order = np.argsort(times)
    snapshots = [files[index].snapshot for index in order]
    return times[order], values[order], snapshots


def cache_is_current(cache_path: Path, snapshots: list[str]) -> bool:
    if not cache_path.exists():
        return False
    try:
        with np.load(cache_path) as cache:
            return (
                int(cache["cache_version"].item()) == CACHE_VERSION
                and str(cache["first_snapshot"].item()) == snapshots[0]
                and str(cache["last_snapshot"].item()) == snapshots[-1]
                and int(cache["snapshot_count"].item()) == len(snapshots)
                and np.array_equal(cache["pressure_levels_bar"], PRESSURE_LEVELS_BAR)
            )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def prepare_case_cache(
    case_dir: Path,
    first: int | None,
    last: int | None,
    cache_dir: Path,
    refresh_cache: bool,
) -> tuple[str, Path, str]:
    selected_files = select_snapshots(resolve_field_files(case_dir, FIELD), first, last)
    snapshots = [file_info.snapshot for file_info in selected_files]
    cache_path = cache_dir / (
        f"{case_dir.name}_dtdz_p1_p6_p30_{snapshot_suffix(first, last)}.npz"
    )
    if not refresh_cache and cache_is_current(cache_path, snapshots):
        return case_dir.name, cache_path, "reused"

    times_days, dtdz_by_pressure, sorted_snapshots = read_dtdz_time_series(
        case_dir,
        first,
        last,
    )
    np.savez_compressed(
        cache_path,
        cache_version=CACHE_VERSION,
        times_days=times_days,
        dtdz_by_pressure=dtdz_by_pressure,
        pressure_levels_bar=np.asarray(PRESSURE_LEVELS_BAR),
        snapshots=np.asarray(sorted_snapshots),
        first_snapshot=sorted_snapshots[0],
        last_snapshot=sorted_snapshots[-1],
        snapshot_count=len(sorted_snapshots),
    )
    return case_dir.name, cache_path, "written"


def read_case_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(cache_path) as cache:
        times_days = cache["times_days"].astype(np.float64)
        dtdz_by_pressure = cache["dtdz_by_pressure"].astype(np.float64)
        snapshots = [str(value) for value in cache["snapshots"]]
    return times_days, dtdz_by_pressure, snapshots


def write_csv(
    path: Path,
    series_by_case: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["case", "time_days"]
            + [f"dtdz_{pressure:g}bar_K_per_km" for pressure in PRESSURE_LEVELS_BAR]
        )
        for label, (times_days, values) in series_by_case.items():
            for time_day, dtdz_values in zip(times_days, values):
                writer.writerow(
                    [label, float(time_day), *[float(value) for value in dtdz_values]]
                )


def plot_stacked_time_series(
    path: Path,
    series_by_case: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    ncases = len(series_by_case)
    fig, axes = plt.subplots(
        ncases,
        1,
        figsize=(12.0, 1.35 * ncases),
        sharex=True,
        sharey=False,
        constrained_layout=False,
        gridspec_kw={"hspace": 0.0},
    )
    if ncases == 1:
        axes = [axes]

    linestyles = ("solid", "--", ":")
    for ax, (label, (times_days, values)) in zip(axes, series_by_case.items()):
        for pressure_index, linestyle in enumerate(linestyles):
            ax.plot(
                times_days,
                values[:, pressure_index],
                color="black",
                linestyle=linestyle,
                lw=2.0,
            )
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError(f"No finite dT/dz values found for {label}")
        ymin = float(np.nanmin(finite_values))
        ymax = float(np.nanmax(finite_values))
        padding = max(0.05 * (ymax - ymin), 0.01)
        ax.set_ylim(ymin - padding, ymax + padding)
        ax.text(0.012, 0.82, label, transform=ax.transAxes, ha="left", va="top")
        ax.axhline(0.0, color="0.5", lw=1.0, zorder=0)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].legend(
        handles=[
            Line2D([], [], color="black", lw=2.0, linestyle=style, label=f"{p:g} bar")
            for p, style in zip(PRESSURE_LEVELS_BAR, linestyles)
        ],
        loc="upper right",
        ncols=3,
    )
    axes[-1].set_xlabel("Time [days]")
    fig.supylabel(r"$dT/dz$ [K km$^{-1}$]")
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.15, top=0.995, hspace=0.0)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser()
    case_dirs = resolve_case_dirs(root, args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        args.cache_dir.expanduser()
        if args.cache_dir is not None
        else args.output_dir / "dtdz_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_paths_by_case: dict[str, Path] = {}
    with ProcessPoolExecutor(max_workers=len(case_dirs)) as executor:
        futures = {
            executor.submit(
                prepare_case_cache,
                case_dir,
                args.first,
                args.last,
                cache_dir,
                args.refresh_cache,
            ): case_dir
            for case_dir in case_dirs
        }
        for future in as_completed(futures):
            case_name, cache_path, action = future.result()
            cache_paths_by_case[case_name] = cache_path
            print(f"{action.capitalize()} cache: {cache_path}")

    series_by_case: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    snapshots_by_case: dict[str, list[str]] = {}
    for case_dir in case_dirs:
        label = case_label(case_dir)
        times_days, values, snapshots = read_case_cache(cache_paths_by_case[case_dir.name])
        series_by_case[label] = (times_days, values)
        snapshots_by_case[label] = snapshots

    selection_suffix = snapshot_suffix(args.first, args.last)
    base_name = f"jup_crm2d_H2O-NH3_F100_dtdz_stacked_time_series_{selection_suffix}"
    plot_path = args.output_dir / f"{base_name}.png"
    csv_path = args.output_dir / f"{base_name}.csv"
    plot_stacked_time_series(plot_path, series_by_case)
    write_csv(csv_path, series_by_case)

    print(f"Used {len(case_dirs)} cases from {root}")
    for label, snapshots in snapshots_by_case.items():
        print(f"{label}: snapshots {snapshots[0]}..{snapshots[-1]} ({len(snapshots)})")
    print(f"Wrote {plot_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
