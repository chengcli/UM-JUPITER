#!/usr/bin/env python3
"""Plot time evolution of horizontal-mean variables at fixed pressure levels."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-time-evolution")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


DEFAULT_ROOT = Path("/home/chengcli/data00/2026.JUPITER_CRM")
DEFAULT_OUTPUT_DIR = Path("diagnostics")
PRESSURE_VARIABLE = "press"
KINETIC_ENERGY_VARIABLE = "ke"
VELOCITY_VARIABLES = ("vel1", "vel2", "vel3")
PRESSURE_REFERENCE_PA = 1.0e5
PRESSURE_LEVELS_BAR = (0.5, 1.0, 5.0, 10.0, 20.0, 50.0)
SECONDS_PER_DAY = 86400.0
FILE_RE = re.compile(
    r"^(?P<prefix>.+)\.(?P<field>out[0-9]+)\.(?P<snapshot>[0-9]+)\.nc$"
)


@dataclass(frozen=True)
class OutputFile:
    path: Path
    prefix: str
    field: str
    snapshot: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot time evolution of horizontal-mean variables sampled at "
            "0.5, 1, 5, 10, 20, and 50 bar."
        )
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output folder name under --root, or a full path, e.g. output_260513.",
    )
    parser.add_argument(
        "--field",
        required=True,
        help="NetCDF output field to read variables from, e.g. out1 or out2.",
    )
    parser.add_argument(
        "--vars",
        nargs="+",
        required=True,
        help="Variable names to plot.",
    )
    parser.add_argument(
        "--first",
        type=int,
        default=None,
        help="Use the first N snapshots. Cannot be combined with --last.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Use the last N snapshots. Cannot be combined with --first.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Simulation root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG and CSV outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--yscale",
        choices=("linear", "log"),
        default="linear",
        help="Vertical variable-axis scale. Default: linear",
    )
    parser.add_argument(
        "--ylabel",
        default="variable value",
        help="Vertical axis label. Default: 'variable value'",
    )
    args = parser.parse_args()
    if args.first is not None and args.last is not None:
        parser.error("--first and --last cannot be specified together")
    return args


def resolve_output_dir(output: str, root: Path) -> Path:
    output_path = Path(output).expanduser()
    if output_path.is_absolute():
        return output_path
    if root.exists():
        return root / output_path
    raise FileNotFoundError(f"Cannot find output root: {root}")


def parse_output_file(path: Path) -> OutputFile | None:
    match = FILE_RE.match(path.name)
    if match is None:
        return None
    return OutputFile(
        path=path,
        prefix=match.group("prefix"),
        field=match.group("field"),
        snapshot=match.group("snapshot"),
    )


def resolve_field_files(output_dir: Path, field: str) -> list[OutputFile]:
    matches = []
    for filename in glob.glob(str(output_dir / f"*.{field}.*.nc")):
        parsed = parse_output_file(Path(filename))
        if parsed is not None:
            matches.append(parsed)

    matches.sort(key=lambda item: (item.prefix, int(item.snapshot)))
    if not matches:
        raise FileNotFoundError(
            f"No NetCDF files matched {output_dir / f'*.{field}.*.nc'}"
        )

    prefixes = sorted({item.prefix for item in matches})
    if len(prefixes) != 1:
        raise ValueError(
            "Matched multiple file prefixes for one field: "
            f"{', '.join(prefixes)}. Use a directory with one simulation output prefix."
        )
    return matches


def select_snapshots(
    files: list[OutputFile],
    first: int | None,
    last: int | None,
) -> list[OutputFile]:
    if first is not None:
        if first <= 0:
            raise ValueError("--first must be positive")
        if first > len(files):
            raise ValueError(f"--first requested {first} snapshots, only found {len(files)}")
        return files[:first]
    if last is not None:
        if last <= 0:
            raise ValueError("--last must be positive")
        if last > len(files):
            raise ValueError(f"--last requested {last} snapshots, only found {len(files)}")
        return files[-last:]
    return files


def pressure_file_for(file_info: OutputFile) -> Path:
    if file_info.field == "out1":
        return file_info.path
    return file_info.path.with_name(
        f"{file_info.prefix}.out1.{file_info.snapshot}.nc"
    )


def require_time_singleton(data_array: xr.DataArray, path: Path) -> xr.DataArray:
    if "time" not in data_array.dims:
        return data_array
    if data_array.sizes["time"] != 1:
        raise ValueError(
            f"{data_array.name!r} in {path} has time length "
            f"{data_array.sizes['time']}; expected one time per file"
        )
    return data_array.isel(time=0)


def require_profile_dims(data_array: xr.DataArray, path: Path) -> None:
    for dim in ("x1", "x3", "x2"):
        if dim not in data_array.dims:
            raise ValueError(f"{data_array.name!r} in {path} is missing dimension {dim!r}")


def validate_variables(path: Path, variables: list[str]) -> None:
    with xr.open_dataset(path) as ds:
        missing = [
            name
            for name in variables
            if name != KINETIC_ENERGY_VARIABLE and name not in ds
        ]
        if missing:
            available = ", ".join(sorted(ds.data_vars))
            raise ValueError(
                f"Missing variable(s) in {path.name}: {', '.join(missing)}. "
                f"Available variables: {available}"
            )


def validate_kinetic_energy_variables(path: Path) -> None:
    with xr.open_dataset(path) as ds:
        missing = [name for name in VELOCITY_VARIABLES if name not in ds]
        if missing:
            available = ", ".join(sorted(ds.data_vars))
            raise ValueError(
                f"Cannot compute {KINETIC_ENERGY_VARIABLE!r} from {path.name}; "
                f"missing velocity variable(s): {', '.join(missing)}. "
                f"Available variables: {available}"
            )


def read_time_days(path: Path) -> float:
    with xr.open_dataset(path) as ds:
        if "time" not in ds:
            return np.nan
        time_values = np.atleast_1d(ds["time"].values)
        if time_values.size != 1:
            raise ValueError(f"{path} has {time_values.size} time values; expected one")
        return float(time_values[0]) / SECONDS_PER_DAY


def read_horizontal_mean_profile(path: Path, variable: str) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise ValueError(f"{path} does not contain variable {variable!r}")
        data = require_time_singleton(ds[variable], path)
        require_profile_dims(data, path)
        values = data.transpose("x1", "x3", "x2").values.astype(np.float64)
    return np.nanmean(values, axis=(1, 2))


def read_kinetic_energy_profile(path: Path) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        for variable in VELOCITY_VARIABLES:
            if variable not in ds:
                raise ValueError(f"{path} does not contain variable {variable!r}")

        kinetic_energy = None
        for variable in VELOCITY_VARIABLES:
            data = require_time_singleton(ds[variable], path)
            require_profile_dims(data, path)
            values = data.transpose("x1", "x3", "x2").values.astype(np.float64)
            term = values * values
            kinetic_energy = term if kinetic_energy is None else kinetic_energy + term

    if kinetic_energy is None:
        raise ValueError(f"Unable to compute {KINETIC_ENERGY_VARIABLE!r} from {path}")
    return 0.5 * np.nanmean(kinetic_energy, axis=(1, 2))


def interpolate_to_pressures(
    pressure_bar: np.ndarray,
    profile: np.ndarray,
    levels_bar: tuple[float, ...],
) -> np.ndarray:
    order = np.argsort(np.log(pressure_bar))
    log_pressure = np.log(pressure_bar[order])
    profile_sorted = profile[order]
    target_log_pressure = np.log(np.array(levels_bar, dtype=np.float64))
    return np.interp(
        target_log_pressure,
        log_pressure,
        profile_sorted,
        left=np.nan,
        right=np.nan,
    )


def read_time_evolution(
    files: list[OutputFile],
    variables: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    times_days = []
    values = {
        variable: np.full((len(files), len(PRESSURE_LEVELS_BAR)), np.nan, dtype=np.float64)
        for variable in variables
    }

    for i, file_info in enumerate(files):
        pressure_path = pressure_file_for(file_info)
        if not pressure_path.exists():
            raise FileNotFoundError(
                f"Missing pressure file for {file_info.path.name}: {pressure_path}"
            )

        times_days.append(read_time_days(file_info.path))
        pressure_profile_bar = (
            read_horizontal_mean_profile(pressure_path, PRESSURE_VARIABLE)
            / PRESSURE_REFERENCE_PA
        )
        for variable in variables:
            if variable == KINETIC_ENERGY_VARIABLE:
                variable_profile = read_kinetic_energy_profile(pressure_path)
            else:
                variable_profile = read_horizontal_mean_profile(file_info.path, variable)
            values[variable][i, :] = interpolate_to_pressures(
                pressure_profile_bar,
                variable_profile,
                PRESSURE_LEVELS_BAR,
            )

    times = np.array(times_days, dtype=np.float64)
    if not np.isfinite(times).all():
        times = np.arange(len(files), dtype=np.float64)
    order = np.argsort(times)
    return times[order], {variable: array[order, :] for variable, array in values.items()}


def safe_name(values: list[str]) -> str:
    return "_".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", value) for value in values)


def write_csv(
    path: Path,
    times_days: np.ndarray,
    values: dict[str, np.ndarray],
) -> None:
    variables = list(values)
    header = ["time_days"]
    for variable in variables:
        for pressure in PRESSURE_LEVELS_BAR:
            header.append(f"{variable}_p{pressure:g}bar")

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for i, time_day in enumerate(times_days):
            row = [float(time_day)]
            for variable in variables:
                row.extend(float(value) for value in values[variable][i, :])
            writer.writerow(row)


def plot_time_evolution(
    path: Path,
    times_days: np.ndarray,
    values: dict[str, np.ndarray],
    yscale: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]

    for var_index, (variable, array) in enumerate(values.items()):
        color = colors[var_index % len(colors)]
        for pressure_index, pressure in enumerate(PRESSURE_LEVELS_BAR):
            ax.plot(
                times_days,
                array[:, pressure_index],
                color=color,
                linestyle=linestyles[pressure_index % len(linestyles)],
                lw=1.8,
                label=f"{variable}, {pressure:g} bar",
            )

    ax.set_xlabel("time [days]")
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize="small", ncols=2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output, args.root)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    all_files = resolve_field_files(output_dir, args.field)
    selected_files = select_snapshots(all_files, args.first, args.last)
    validate_variables(selected_files[-1].path, args.vars)
    if KINETIC_ENERGY_VARIABLE in args.vars:
        validate_kinetic_energy_variables(pressure_file_for(selected_files[-1]))

    times_days, values = read_time_evolution(selected_files, args.vars)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variable_name = safe_name(args.vars)
    first_id = selected_files[0].snapshot
    last_id = selected_files[-1].snapshot
    base_name = (
        f"{output_dir.name}_{args.field}_{variable_name}_time_evolution_"
        f"{first_id}_{last_id}"
    )
    plot_path = args.output_dir / f"{base_name}.png"
    csv_path = args.output_dir / f"{base_name}.csv"

    plot_time_evolution(plot_path, times_days, values, args.yscale, args.ylabel)
    write_csv(csv_path, times_days, values)

    print(f"Used {len(selected_files)} snapshots from {output_dir}")
    print(f"First snapshot: {selected_files[0].path.name}")
    print(f"Last snapshot: {selected_files[-1].path.name}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
