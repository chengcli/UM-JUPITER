"""Extract and reuse horizontal-mean initial profiles from snapshot 00000."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from plot_horizontal_mean_profiles import (
    PRESSURE_REFERENCE_PA,
    find_snapshot_file,
    read_pressure_profile,
    read_variable_stats,
    resolve_field_files,
)


INITIAL_SNAPSHOT = "00000"
def initial_source_case(case_dir: Path) -> Path:
    source_name = re.sub(r"_F10(_nu[0-9.]+)$", r"_F100\1", case_dir.name)
    source_dir = case_dir.with_name(source_name)
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Initial-profile source case does not exist for {case_dir}: {source_dir}"
        )
    return source_dir


def initial_profile_cache_path(
    output_dir: Path,
    source_case: Path,
    variable: str,
) -> Path:
    return output_dir / "initial_profile_cache" / (
        f"{source_case.name}_{variable}_snapshot{INITIAL_SNAPSHOT}.npz"
    )


def read_or_create_initial_profile(
    case_dir: Path,
    variable: str,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, Path]:
    source_case = initial_source_case(case_dir)
    cache_path = initial_profile_cache_path(output_dir, source_case, variable)
    if cache_path.exists():
        with np.load(cache_path) as cache:
            return (
                cache["pressure_bar"].astype(np.float64),
                cache["profile"].astype(np.float64),
                cache_path,
            )

    output_field = "out2" if variable == "theta" else "out1"
    field_files = resolve_field_files(source_case, output_field)
    initial_file = find_snapshot_file(field_files, INITIAL_SNAPSHOT)
    pressure_x1, pressure_pa = read_pressure_profile([initial_file])
    x1, profile, _ = read_variable_stats(
        [initial_file],
        variable,
        Path("jupiter_crm.yaml"),
    )
    if not np.allclose(pressure_x1, x1):
        raise ValueError(
            f"x1 coordinate mismatch between initial {variable} and pressure "
            f"in {source_case}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pressure_bar = pressure_pa / PRESSURE_REFERENCE_PA
    np.savez_compressed(
        cache_path,
        source_case=source_case.name,
        snapshot=INITIAL_SNAPSHOT,
        variable=variable,
        pressure_bar=pressure_bar,
        profile=profile,
    )
    return pressure_bar, profile, cache_path
