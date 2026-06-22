#!/usr/bin/env python3
"""Plot vertically averaged horizontal vorticity contours for CRM cases."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-column-vorticity")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np

from cache_horizontal_vorticity import (
    CACHE_VERSION as VORTICITY_CACHE_VERSION,
    build_cache as build_vorticity_cache,
    cache_path as vorticity_cache_path,
)
from case_selection import resolve_case_dirs
from custom_colormaps import diverging_with_white_plateau
from plot_horizontal_mean_profiles import DEFAULT_OUTPUT_DIR
from plot_virtual_potential_temperature_profiles import DEFAULT_ROOT


DEFAULT_CASE_REGEX = r"jup_crm3d_H2O-NH3-H2S_F10_nu(0\.01|0\.1|1\.0)"
DEFAULT_LAST = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 1x3 figure of vertically averaged horizontal vorticity "
            "contours for matching CRM cases. Missing vorticity caches are built."
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
        help=f"Number of latest snapshots to average. Default: {DEFAULT_LAST}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG output and vorticity caches. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="PNG basename. Default derives from selected case family.",
    )
    parser.add_argument(
        "--cmap",
        default="RdBu_r",
        help="Matplotlib colormap. Default: RdBu_r",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=17,
        help="Number of contour levels. Default: 17",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-1.0e-4,
        help="Minimum vorticity colorbar value in s^-1 before scaling. Default: -1e-4",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=1.0e-4,
        help="Maximum vorticity colorbar value in s^-1 before scaling. Default: 1e-4",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0e5,
        help="Scale factor applied before plotting. Default: 1e5",
    )
    parser.add_argument(
        "--white-half-width",
        type=float,
        default=None,
        help=(
            "Half-width of the white colormap plateau in plotted units. "
            "Default: half of one contour interval."
        ),
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute vorticity caches before plotting.",
    )
    return parser.parse_args()


def case_label(path: Path) -> str:
    match = re.search(r"_nu(?P<nu>[0-9.]+)$", path.name)
    if match is None:
        return path.name
    return f"$\\kappa=${match.group('nu')}"


def experiment_name(case_dirs: list[Path]) -> str:
    names = [re.sub(r"_nu[0-9.]+$", "", path.name) for path in case_dirs]
    if len(set(names)) != 1:
        return "mixed_cases"
    return names[0]


def read_vorticity_cache(path: Path, case_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing vorticity cache: {path}")
    with np.load(path) as cache:
        version = int(cache["cache_version"])
        if version != VORTICITY_CACHE_VERSION:
            raise ValueError(f"Unsupported vorticity cache version {version} in {path}")
        cached_case = str(cache["case_name"])
        if cached_case != case_name:
            raise ValueError(f"Cache {path} contains {cached_case}, expected {case_name}")
        return (
            cache["x2_km"].astype(np.float64),
            cache["x3_km"].astype(np.float64),
            cache["column_mean_vorticity_s_inv"].astype(np.float64),
            cache["snapshots"].astype(str).tolist(),
        )


def fixed_levels(vmin: float, vmax: float, count: int, scale: float) -> np.ndarray:
    if not vmin < vmax:
        raise ValueError("--vmin must be less than --vmax")
    return np.linspace(vmin * scale, vmax * scale, count)


def plot_contours(
    path: Path,
    panels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    levels: np.ndarray,
    cmap: str,
    scale: float,
    white_half_width: float | None,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(13.5, 4.2),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    if len(panels) == 1:
        axes = np.asarray([axes])

    contour = None
    if white_half_width is None:
        white_half_width = 0.5 * float(np.min(np.diff(levels)))
    padded_cmap = diverging_with_white_plateau(
        cmap,
        float(levels[0]),
        float(levels[-1]),
        float(white_half_width),
        center=0.0,
        name=f"{cmap}_white_center",
    )
    for ax, (label, (x2_km, x3_km, zeta)) in zip(axes, panels.items()):
        contour = ax.contourf(
            x2_km,
            x3_km,
            zeta * scale,
            levels=levels,
            cmap=padded_cmap,
            extend="both",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label)
        ax.set_xlabel("x2 [km]")
        ax.grid(False)
    axes[0].set_ylabel("x3 [km]")

    assert contour is not None
    colorbar = fig.colorbar(contour, ax=axes.tolist(), orientation="vertical", pad=0.015, fraction=0.035)
    colorbar.set_label(r"Column-mean vorticity, $\langle\zeta\rangle_z$ [$10^{-5}$ s$^{-1}$]")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.last <= 0:
        raise ValueError("--last must be positive")
    if args.levels < 3:
        raise ValueError("--levels must be at least 3")

    root = args.root.expanduser()
    case_dirs = resolve_case_dirs(root, args.case_regex)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    snapshots_by_case: dict[str, list[str]] = {}
    for case_dir in case_dirs:
        cache = vorticity_cache_path(args.output_dir, case_dir.name, args.last)
        if args.refresh_cache or not cache.exists():
            cache = build_vorticity_cache(case_dir, args.output_dir, args.last)
            print(f"Wrote cache {cache}")
        else:
            print(f"Using cache {cache}")
        x2_km, x3_km, zeta, snapshots = read_vorticity_cache(cache, case_dir.name)
        label = case_label(case_dir)
        panels[label] = (x2_km, x3_km, zeta)
        snapshots_by_case[label] = snapshots

    levels = fixed_levels(args.vmin, args.vmax, args.levels, args.scale)
    base_name = args.output_name or f"{experiment_name(case_dirs)}_column_mean_vorticity_last{args.last}"
    plot_path = args.output_dir / f"{base_name}.png"
    plot_contours(plot_path, panels, levels, args.cmap, args.scale, args.white_half_width)

    print(f"Used {len(case_dirs)} cases from {root}")
    for label, snapshots in snapshots_by_case.items():
        print(f"{label}: snapshots {snapshots[0]}..{snapshots[-1]} ({len(snapshots)})")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
