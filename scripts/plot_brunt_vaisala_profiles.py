#!/usr/bin/env python3
"""Compute, cache, and plot Brunt-Vaisala frequency-squared profiles.

N^2 is computed from recomputed virtual potential temperature:

    N^2 = g / theta_v * d(theta_v) / dz

The existing NetCDF ``theta_v`` field is not used.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-n2")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np
import yaml

from case_selection import resolve_case_dirs
from plot_horizontal_mean_profiles import (
    DEFAULT_MAX_PRESSURE_BAR,
    DEFAULT_OUTPUT_DIR,
    PRESSURE_MARKERS_BAR,
    PRESSURE_REFERENCE_PA,
)
from plot_virtual_potential_temperature_profiles import (
    CACHE_VERSION as THETA_V_CACHE_VERSION,
    DEFAULT_CASE_REGEX,
    DEFAULT_CONFIG,
    DEFAULT_LAST,
    DEFAULT_ROOT,
    dry_gas_constant,
    experiment_name,
    read_snapshot_theta_v,
    selected_file_pairs,
)


CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute virtual potential temperature and plot horizontal-mean "
            "Brunt-Vaisala frequency-squared profiles for matching CRM cases."
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
        help=f"Directory for PNG, CSV, and cache outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory. Default: <output-dir>/brunt_vaisala_profile_cache",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config used for dry gas constant and gravity. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="PNG/CSV basename. Default derives from the shared experiment prefix.",
    )
    parser.add_argument(
        "--max-pressure",
        type=float,
        default=DEFAULT_MAX_PRESSURE_BAR,
        help=f"Maximum displayed pressure in bar. Default: {DEFAULT_MAX_PRESSURE_BAR:g}",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-read NetCDF snapshots and overwrite existing per-case caches.",
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Do not draw mean +/- std shading.",
    )
    return parser.parse_args()


def read_gravity(config_path: Path) -> float:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return abs(float(config["forcing"]["const-gravity"]["grav1"]))


def case_label(path: Path) -> str:
    match = re.search(r"_nu(?P<nu>[0-9.]+)$", path.name)
    if match is None:
        return path.name
    return f"$\\kappa=${match.group('nu')}"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def cache_path(cache_dir: Path, case_name: str, last: int) -> Path:
    return cache_dir / f"{case_name}_n2_profiles_last{last}.npz"


def brunt_vaisala_frequency_squared(
    x1_m: np.ndarray,
    theta_v: np.ndarray,
    gravity: float,
) -> np.ndarray:
    if x1_m.ndim != 1:
        raise ValueError("x1 coordinate must be one-dimensional")
    if theta_v.shape[0] != x1_m.size:
        raise ValueError("theta_v vertical dimension does not match x1")
    edge_order = 2 if x1_m.size >= 3 else 1
    dtheta_v_dz = np.gradient(theta_v, x1_m, axis=0, edge_order=edge_order)
    with np.errstate(divide="ignore", invalid="ignore"):
        n2 = gravity * dtheta_v_dz / theta_v
    n2[~np.isfinite(n2)] = np.nan
    return n2


def build_case_cache(
    case_dir: Path,
    output: Path,
    last: int,
    rd_dry: float,
    gravity: float,
) -> None:
    pairs = selected_file_pairs(case_dir, last)
    x1_ref = None
    pressure_sum = None
    n2_sum = None
    n2_square_sum = None
    sample_count = None

    for out1_info, out2_info in pairs:
        x1, press, theta_v = read_snapshot_theta_v(out1_info.path, out2_info.path, rd_dry)
        if x1_ref is None:
            x1_ref = x1
            pressure_sum = np.zeros_like(np.nanmean(press, axis=(1, 2)))
            n2_sum = np.zeros_like(pressure_sum)
            n2_square_sum = np.zeros_like(pressure_sum)
            sample_count = np.zeros_like(pressure_sum)
        elif not np.allclose(x1_ref, x1):
            raise ValueError(f"x1 coordinate mismatch in {out2_info.path}")

        n2 = brunt_vaisala_frequency_squared(x1, theta_v, gravity)
        pressure_sum += np.nanmean(press, axis=(1, 2))
        valid = np.isfinite(n2)
        n2_sum += np.nansum(n2, axis=(1, 2))
        n2_square_sum += np.nansum(n2 * n2, axis=(1, 2))
        sample_count += np.sum(valid, axis=(1, 2))

    assert x1_ref is not None
    assert pressure_sum is not None
    assert n2_sum is not None
    assert n2_square_sum is not None
    assert sample_count is not None

    pressure_mean = pressure_sum / len(pairs)
    n2_mean = n2_sum / sample_count
    n2_std = np.sqrt(np.maximum(n2_square_sum / sample_count - n2_mean * n2_mean, 0.0))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        cache_version=np.array(CACHE_VERSION),
        theta_v_formula_version=np.array(THETA_V_CACHE_VERSION),
        case_name=np.array(case_dir.name),
        snapshots=np.asarray([out2.snapshot for _, out2 in pairs]),
        x1_m=x1_ref.astype(np.float64),
        pressure_bar=(pressure_mean / PRESSURE_REFERENCE_PA).astype(np.float64),
        n2_mean=n2_mean.astype(np.float64),
        n2_std=n2_std.astype(np.float64),
        rd_dry=np.array(rd_dry, dtype=np.float64),
        gravity=np.array(gravity, dtype=np.float64),
        formula=np.array("N^2 = g / theta_v * d(theta_v) / dz"),
    )


def read_case_cache(path: Path, case_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing cache for {case_name}: {path}")
    with np.load(path) as cache:
        version = int(cache["cache_version"])
        if version != CACHE_VERSION:
            raise ValueError(f"Unsupported cache version {version} in {path}")
        cached_case = str(cache["case_name"])
        if cached_case != case_name:
            raise ValueError(f"Cache {path} contains {cached_case}, expected {case_name}")
        return (
            cache["pressure_bar"].astype(np.float64),
            cache["n2_mean"].astype(np.float64),
            cache["n2_std"].astype(np.float64),
            cache["snapshots"].astype(str).tolist(),
        )


def plot_profiles(
    path: Path,
    profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    max_pressure: float,
    draw_std: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 8.0), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=0, vmax=max(len(profiles) - 1, 1))

    for index, (label, (pressure_bar, mean, std)) in enumerate(profiles.items()):
        color = cmap(norm(index))
        if draw_std:
            ax.fill_betweenx(
                pressure_bar,
                mean - std,
                mean + std,
                color=color,
                alpha=0.16,
                linewidth=0.0,
            )
        ax.plot(mean, pressure_bar, color=color, lw=2.0, label=label)

    ax.axvline(0.0, color="black", lw=1.2, alpha=0.65, zorder=0)
    ax.set_yscale("log")
    ax.invert_yaxis()
    bottom, top = ax.get_ylim()
    ax.set_ylim(min(max_pressure, bottom), top)
    ax.set_xlim(-0.5e-5, 1.5e-5)
    ax.set_xlabel(r"$N^2$ [s$^{-2}$]")
    ax.set_ylabel("Pressure [bar]")
    ax.grid(True, which="both", alpha=0.25)
    for pressure_marker in PRESSURE_MARKERS_BAR:
        ax.axhline(pressure_marker, color="0.55", lw=1.5, alpha=0.55, zorder=0)
    ax.legend(title="Diffusivity [m$^2$ s$^{-1}$]", loc="best")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_csv(path: Path, profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    labels = list(profiles)
    header = ["level"]
    for label in labels:
        name = safe_name(label)
        header.extend(
            [
                f"{name}_pressure_bar",
                f"{name}_n2_mean_s-2",
                f"{name}_n2_std_s-2",
            ]
        )

    max_levels = max(values[0].size for values in profiles.values())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for level in range(max_levels):
            row: list[float | int | str] = [level]
            for pressure, mean, std in profiles.values():
                if level < pressure.size:
                    row.extend([float(pressure[level]), float(mean[level]), float(std[level])])
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
    output_dir = args.output_dir
    cache_dir = args.cache_dir or output_dir / "brunt_vaisala_profile_cache"
    case_dirs = resolve_case_dirs(root, args.case_regex)
    rd_dry = dry_gas_constant(args.config)
    gravity = read_gravity(args.config)

    profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    snapshots_by_case: dict[str, list[str]] = {}
    for case_dir in case_dirs:
        cache = cache_path(cache_dir, case_dir.name, args.last)
        if args.refresh_cache or not cache.exists():
            build_case_cache(case_dir, cache, args.last, rd_dry, gravity)
            print(f"Wrote cache {cache}")
        else:
            print(f"Using cache {cache}")
        pressure, mean, std, snapshots = read_case_cache(cache, case_dir.name)
        label = case_label(case_dir)
        profiles[label] = (pressure, mean, std)
        snapshots_by_case[label] = snapshots

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = args.output_name or f"{experiment_name(case_dirs)}_n2_profiles_last{args.last}"
    plot_path = output_dir / f"{base_name}.png"
    csv_path = output_dir / f"{base_name}.csv"
    plot_profiles(plot_path, profiles, args.max_pressure, not args.no_std)
    write_csv(csv_path, profiles)

    print(f"Used {len(case_dirs)} cases from {root}")
    print(f"Rd_dry = {rd_dry:.10g} J kg^-1 K^-1 from {args.config}")
    print(f"gravity = {gravity:.10g} m s^-2 from {args.config}")
    for label, snapshots in snapshots_by_case.items():
        print(f"{label}: snapshots {snapshots[0]}..{snapshots[-1]} ({len(snapshots)})")
    print(f"Wrote {plot_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
