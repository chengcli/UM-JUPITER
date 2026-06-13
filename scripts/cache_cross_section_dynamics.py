#!/usr/bin/env python3
"""Cache projected velocity and mass streamfunction for a periodic section."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from case_selection import resolve_case_dirs
from cross_section_dynamics import (
    cross_section_basis,
    mass_streamfunction_least_squares,
    project_horizontal_velocity,
)
from periodic_cross_section import sample_periodic_bilinear
from plot_horizontal_mean_profiles import (
    DEFAULT_OUTPUT_DIR,
    resolve_field_files,
    validate_variables,
)


DEFAULT_ROOT = Path("/home/chengcli/data/2026.JupiterCRM")
DEFAULT_CASE_REGEX = r"jup_crm3d_H2O-NH3-H2S_F10_nu0\.01"
DEFAULT_LAST = 20
CACHE_VERSION = 1
VARIABLES = ("vel1", "vel2", "vel3", "rho")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache projected velocities and mass streamfunction on a periodic section."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-regex", default=DEFAULT_CASE_REGEX)
    parser.add_argument("--last", type=int, default=DEFAULT_LAST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--path-species",
        choices=("H2O", "NH3"),
        default="H2O",
        help="Species whose path extrema define the cross-section. Default: H2O.",
    )
    parser.add_argument("--section-cache", type=Path, default=None)
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=None,
        help="Dynamics cache path. Default: selected species-section path under output-dir.",
    )
    return parser.parse_args()


def section_cache_path(
    output_dir: Path, case_name: str, last: int, path_species: str = "H2O"
) -> Path:
    return (
        output_dir
        / f"{path_species.lower()}_min_max_path_cross_section_cache"
        / f"{case_name}_{path_species}_min_max_path_cross_section_last{last}.npz"
    )


def dynamics_cache_path(
    output_dir: Path, case_name: str, last: int, path_species: str = "H2O"
) -> Path:
    return (
        output_dir
        / "cross_section_dynamics_cache"
        / f"{case_name}_{path_species}_min_max_path_dynamics_last{last}.npz"
    )


def read_sampled_fields(
    path: Path,
    section_x2_km: np.ndarray,
    section_x3_km: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with xr.open_dataset(path) as ds:
        x1_m = ds["x1"].values.astype(np.float64)
        x2_km = ds["x2"].values.astype(np.float64) / 1.0e3
        x3_km = ds["x3"].values.astype(np.float64) / 1.0e3
        fields = {}
        for variable in VARIABLES:
            data = ds[variable]
            if "time" in data.dims:
                data = data.isel(time=0)
            values = data.transpose("x1", "x3", "x2").values.astype(np.float64)
            fields[variable] = sample_periodic_bilinear(
                values,
                x2_km,
                x3_km,
                section_x2_km,
                section_x3_km,
            )
    return x1_m, fields


def build_cache(
    case_dir: Path, section_path: Path, output_path: Path, path_species: str = "H2O"
) -> None:
    if not section_path.exists():
        script = (
            "plot_h2o_on_h2o_min_max_path_cross_section.py"
            if path_species == "H2O"
            else "plot_nh3_on_nh3_min_max_path_cross_section.py"
        )
        raise FileNotFoundError(
            f"Missing section cache: {section_path}\nGenerate it with:\n"
            f"  python scripts/{script} "
            f"--case-regex '{case_dir.name}' --refresh-cache"
        )
    with np.load(section_path, allow_pickle=False) as section:
        if str(section["case_name"]) != case_dir.name:
            raise ValueError(f"Section cache case mismatch in {section_path}")
        snapshots = section["snapshots"].astype(str).tolist()
        distance_km = section["section_distance_km"].astype(np.float64)
        section_x2_km = section["section_x2_wrapped_km"].astype(np.float64)
        section_x3_km = section["section_x3_wrapped_km"].astype(np.float64)
        section_x2_unwrapped_km = section["section_x2_unwrapped_km"].astype(np.float64)
        section_x3_unwrapped_km = section["section_x3_unwrapped_km"].astype(np.float64)

    files_by_snapshot = {
        item.snapshot: item for item in resolve_field_files(case_dir, "out1")
    }
    missing = [snapshot for snapshot in snapshots if snapshot not in files_by_snapshot]
    if missing:
        raise FileNotFoundError(f"Missing out1 snapshots: {', '.join(missing)}")
    selected = [files_by_snapshot[snapshot] for snapshot in snapshots]
    validate_variables(selected[-1].path, list(VARIABLES))

    tangent, normal = cross_section_basis(
        section_x2_unwrapped_km[0],
        section_x3_unwrapped_km[0],
        section_x2_unwrapped_km[-1],
        section_x3_unwrapped_km[-1],
    )
    sums: dict[str, np.ndarray] = {}
    x1_m = None
    for file_info in selected:
        current_x1_m, fields = read_sampled_fields(
            file_info.path,
            section_x2_km,
            section_x3_km,
        )
        parallel, perpendicular = project_horizontal_velocity(
            fields["vel2"], fields["vel3"], tangent, normal
        )
        current = {
            "vertical_velocity": fields["vel1"],
            "parallel_velocity": parallel,
            "perpendicular_velocity": perpendicular,
            "density": fields["rho"],
            "parallel_mass_flux": fields["rho"] * parallel,
            "vertical_mass_flux": fields["rho"] * fields["vel1"],
        }
        if x1_m is None:
            x1_m = current_x1_m
            sums = {name: np.zeros_like(values) for name, values in current.items()}
        elif not np.allclose(x1_m, current_x1_m):
            raise ValueError(f"Vertical coordinates changed in {file_info.path}")
        for name, values in current.items():
            sums[name] += values

    assert x1_m is not None
    means = {name: total / len(selected) for name, total in sums.items()}
    streamfunction, diagnostics = mass_streamfunction_least_squares(
        distance_km * 1.0e3,
        x1_m,
        means["parallel_mass_flux"],
        means["vertical_mass_flux"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        cache_version=np.array(CACHE_VERSION),
        case_name=np.array(case_dir.name),
        section_cache=np.array(str(section_path)),
        snapshots=np.asarray(snapshots),
        distance_km=distance_km,
        x1_m=x1_m,
        tangent_x2_x3=tangent,
        normal_x2_x3=normal,
        vertical_velocity=means["vertical_velocity"],
        parallel_velocity=means["parallel_velocity"],
        perpendicular_velocity=means["perpendicular_velocity"],
        density=means["density"],
        parallel_mass_flux=means["parallel_mass_flux"],
        vertical_mass_flux=means["vertical_mass_flux"],
        mass_streamfunction=streamfunction,
        streamfunction_residual_norm=np.array(diagnostics["residual_norm"]),
        streamfunction_normal_equation_residual_norm=np.array(
            diagnostics["normal_equation_residual_norm"]
        ),
        streamfunction_iterations=np.array(diagnostics["iterations"]),
    )


def main() -> None:
    args = parse_args()
    if args.last <= 0:
        raise ValueError("--last must be positive")
    cases = resolve_case_dirs(args.root.expanduser(), args.case_regex)
    if len(cases) != 1:
        raise ValueError("--case-regex must match exactly one case")
    case_dir = cases[0]
    section_path = args.section_cache or section_cache_path(
        args.output_dir, case_dir.name, args.last, args.path_species
    )
    output_path = args.output_cache or dynamics_cache_path(
        args.output_dir, case_dir.name, args.last, args.path_species
    )
    build_cache(case_dir, section_path, output_path, args.path_species)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
