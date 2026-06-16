#!/usr/bin/env python3
"""Plot potential-temperature anomaly on the NH3 path-extrema cross-section."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-theta-nh3-section")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

from cache_cross_section_dynamics import dynamics_cache_path
from case_selection import resolve_case_dirs
from plot_h2o_on_h2o_min_max_path_cross_section import (
    DEFAULT_CASE_REGEX,
    DEFAULT_LAST,
    DEFAULT_ROOT,
)
from plot_horizontal_mean_profiles import DEFAULT_OUTPUT_DIR
from plot_theta_on_h2o_min_max_path_cross_section import (
    build_theta_cache,
    plot_cache,
    theta_cache_path,
)
from species_path_cross_section import section_cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot potential-temperature anomaly on the NH3 min/max path "
            "cross-section with wind and H2O cloud overlays."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-regex", default=DEFAULT_CASE_REGEX)
    parser.add_argument("--last", type=int, default=DEFAULT_LAST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dynamics-cache", type=Path, default=None)
    parser.add_argument("--section-cache", type=Path, default=None)
    parser.add_argument("--theta-cache", type=Path, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--show-wind",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay perpendicular wind contours. Default: enabled.",
    )
    return parser.parse_args()


def output_path(output_dir: Path, case_name: str, last: int) -> Path:
    return (
        output_dir
        / f"{case_name}_theta_on_NH3_min_max_path_cross_section_wind_cloud_last{last}.png"
    )


def nh3_theta_cache_path(output_dir: Path, case_name: str, last: int) -> Path:
    return theta_cache_path(output_dir, case_name, last).with_name(
        f"{case_name}_theta_on_NH3_min_max_path_cross_section_last{last}.npz"
    )


def main() -> None:
    args = parse_args()
    if args.last <= 0:
        raise ValueError("--last must be positive")
    cases = resolve_case_dirs(args.root.expanduser(), args.case_regex)
    if len(cases) != 1:
        raise ValueError("--case-regex must match exactly one case")
    case_dir = cases[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    section = args.section_cache or section_cache_path(args.output_dir, case_dir.name, "NH3", args.last)
    theta = args.theta_cache or nh3_theta_cache_path(args.output_dir, case_dir.name, args.last)
    dynamics = args.dynamics_cache or dynamics_cache_path(
        args.output_dir, case_dir.name, args.last, path_species="NH3"
    )
    figure = output_path(args.output_dir, case_dir.name, args.last)

    if args.refresh_cache or not theta.exists():
        build_theta_cache(case_dir, section, theta, args.last)
        print(f"Wrote cache {theta}")
    else:
        print(f"Using cache {theta}")
    plot_cache(section, theta, dynamics, figure, args.show_wind)
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
