#!/usr/bin/env python3
"""Plot H2O vapor anomaly on the NH3 path-extrema periodic section."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-h2o-on-nh3-section")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

from case_selection import resolve_case_dirs
from plot_h2o_on_h2o_min_max_path_cross_section import DEFAULT_CASE_REGEX, DEFAULT_LAST, DEFAULT_ROOT
from plot_horizontal_mean_profiles import DEFAULT_OUTPUT_DIR
from species_path_cross_section import (
    plot_cache,
    section_cache_path,
    section_dynamics_cache_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot H2O vapor fractional deviation on the NH3 path-extrema section."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-regex", default=DEFAULT_CASE_REGEX)
    parser.add_argument("--last", type=int, default=DEFAULT_LAST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dynamics-cache", type=Path, default=None)
    parser.add_argument(
        "--show-perp-velocity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--show-streamfunction", action="store_true")
    return parser.parse_args()


def output_path(output_dir: Path, case_name: str, last: int) -> Path:
    return (
        output_dir
        / f"{case_name}_H2O_on_NH3_min_max_path_cross_section_last{last}.png"
    )


def main() -> None:
    args = parse_args()
    if args.last <= 0:
        raise ValueError("--last must be positive")
    cases = resolve_case_dirs(args.root.expanduser(), args.case_regex)
    if len(cases) != 1:
        raise ValueError("--case-regex must match exactly one case")
    case_dir = cases[0]
    section = section_cache_path(args.output_dir, case_dir.name, "NH3", args.last)
    if not section.exists():
        raise FileNotFoundError(
            f"Missing NH3 section cache: {section}\nGenerate it with:\n"
            "  python scripts/plot_nh3_on_nh3_min_max_path_cross_section.py "
            f"--case-regex '{args.case_regex}' --last {args.last} "
            "--refresh-cache --cache-only"
        )
    dynamics = args.dynamics_cache or section_dynamics_cache_path(
        args.output_dir, case_dir.name, "NH3", args.last
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = output_path(args.output_dir, case_dir.name, args.last)
    plot_cache(
        section, dynamics, figure, "H2O", False,
        args.show_perp_velocity, args.show_streamfunction,
    )
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
