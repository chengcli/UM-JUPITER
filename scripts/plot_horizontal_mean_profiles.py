#!/usr/bin/env python3
"""Plot horizontal-mean vertical profiles from JUPITER CRM NetCDF outputs."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-profiles")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


DEFAULT_ROOT = Path("/home/chengcli/data00/2026.JUPITER_CRM")
DEFAULT_OUTPUT_DIR = Path("diagnostics")
DEFAULT_MAX_PRESSURE_BAR = 100.0
PRESSURE_VARIABLE = "press"
PRESSURE_REFERENCE_PA = 1.0e5
DEFAULT_REFERENCE_ID = "00000"
PRESSURE_MARKERS_BAR = (0.5, 1.0, 5.0, 10.0, 20.0, 50.0)
ALTITUDE_TICKS_KM = np.array(
    [-300.0, -200.0, -100.0, -50.0, -20.0, -10.0, 0.0, 10.0, 20.0, 50.0, 100.0]
)
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
            "Plot horizontal-mean vertical profiles for one or more variables "
            "from the last N snapshots of a selected NetCDF output field."
        )
    )
    parser.add_argument(
        "output",
        type=str,
        help=(
            "Output folder name under --root, or a full path, e.g. "
            "output_260513."
        ),
    )
    parser.add_argument(
        "--field",
        type=str,
        required=True,
        help="NetCDF output field to read variables from, e.g. out1 or out2.",
    )
    parser.add_argument(
        "--vars",
        nargs="+",
        required=True,
        help="Variable names to plot together on the shared x-axis.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=20,
        help="Number of latest snapshots to include. Default: 20",
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
        "--xscale",
        choices=("linear", "log"),
        default="linear",
        help="Horizontal variable-axis scale. Default: linear",
    )
    parser.add_argument(
        "--reference-id",
        default=DEFAULT_REFERENCE_ID,
        help=(
            "Snapshot id to plot as a dashed reference line. "
            f"Default: {DEFAULT_REFERENCE_ID}"
        ),
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Disable the dashed reference profiles.",
    )
    parser.add_argument(
        "--max-pressure",
        type=float,
        default=DEFAULT_MAX_PRESSURE_BAR,
        help=f"Maximum displayed pressure in bar. Default: {DEFAULT_MAX_PRESSURE_BAR:g}",
    )
    parser.add_argument(
        "--xlabel",
        default="variable value",
        help="Horizontal axis label. Default: 'variable value'",
    )
    return parser.parse_args()


def resolve_output_dir(output: str, root: Path) -> Path:
    output_path = Path(output).expanduser()
    if output_path.is_absolute():
        return output_path

    if root.exists():
        return root / output_path
    else:
        raise FileNotFoundError(f"Cannot find output path")

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
    return matches


def select_last_files(files: list[OutputFile], last: int) -> list[OutputFile]:
    if last <= 0:
        raise ValueError("--last must be positive")
    prefixes = sorted({item.prefix for item in files})
    if len(prefixes) != 1:
        raise ValueError(
            "Matched multiple file prefixes for one field: "
            f"{', '.join(prefixes)}. Use a directory with one simulation output prefix."
        )
    if len(files) < last:
        raise ValueError(f"Only found {len(files)} {files[0].field} files, requested {last}")
    return files[-last:]


def find_snapshot_file(files: list[OutputFile], snapshot: str) -> OutputFile:
    snapshot = snapshot.zfill(len(files[0].snapshot))
    for file_info in files:
        if file_info.snapshot == snapshot:
            return file_info
    raise FileNotFoundError(
        f"No {files[0].field} file found for reference snapshot id {snapshot}"
    )


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


def available_variables(path: Path) -> list[str]:
    with xr.open_dataset(path) as ds:
        return sorted(ds.data_vars)


def validate_variables(path: Path, variables: list[str]) -> None:
    with xr.open_dataset(path) as ds:
        missing = [name for name in variables if name not in ds]
        if missing:
            available = ", ".join(sorted(ds.data_vars))
            raise ValueError(
                f"Missing variable(s) in {path.name}: {', '.join(missing)}. "
                f"Available variables: {available}"
            )


def read_profile_samples(path: Path, variable: str) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise ValueError(f"{path} does not contain variable {variable!r}")
        data = require_time_singleton(ds[variable], path)
        require_profile_dims(data, path)
        x1 = ds["x1"].values.astype(np.float64)
        values = data.transpose("x1", "x3", "x2").values.astype(np.float64)
    return x1, values


def read_variable_stats(
    files: list[OutputFile],
    variable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1_ref: np.ndarray | None = None
    samples = []

    for file_info in files:
        x1, values = read_profile_samples(file_info.path, variable)
        if x1_ref is None:
            x1_ref = x1
        elif not np.allclose(x1_ref, x1):
            raise ValueError(f"x1 coordinate mismatch in {file_info.path}")
        samples.append(values)

    if x1_ref is None:
        raise ValueError(f"No samples read for {variable}")

    stacked = np.stack(samples, axis=0)
    mean = np.nanmean(stacked, axis=(0, 2, 3))
    std = np.nanstd(stacked, axis=(0, 2, 3))
    return x1_ref, mean, std


def read_variable_reference(
    file_info: OutputFile,
    variable: str,
) -> tuple[np.ndarray, np.ndarray]:
    x1, values = read_profile_samples(file_info.path, variable)
    return x1, np.nanmean(values, axis=(1, 2))


def read_pressure_profile(files: list[OutputFile]) -> tuple[np.ndarray, np.ndarray]:
    x1_ref: np.ndarray | None = None
    pressure_samples = []
    for file_info in files:
        path = pressure_file_for(file_info)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing pressure file for {file_info.path.name}: {path}"
            )
        x1, values = read_profile_samples(path, PRESSURE_VARIABLE)
        if x1_ref is None:
            x1_ref = x1
        elif not np.allclose(x1_ref, x1):
            raise ValueError(f"x1 coordinate mismatch in pressure file {path}")
        pressure_samples.append(values)

    if x1_ref is None:
        raise ValueError("No pressure samples were read")

    stacked = np.stack(pressure_samples, axis=0)
    return x1_ref, np.nanmean(stacked, axis=(0, 2, 3))


def altitude_from_pressure(x1_m: np.ndarray, pressure_pa: np.ndarray) -> np.ndarray:
    if np.nanmin(pressure_pa) > PRESSURE_REFERENCE_PA:
        raise ValueError("Pressure profile never reaches 1 bar")
    if np.nanmax(pressure_pa) < PRESSURE_REFERENCE_PA:
        raise ValueError("Pressure profile is always below 1 bar")

    order = np.argsort(np.log(pressure_pa))
    x1_at_1bar = np.interp(
        np.log(PRESSURE_REFERENCE_PA),
        np.log(pressure_pa)[order],
        x1_m[order],
    )
    return (x1_m - x1_at_1bar) / 1000.0


def safe_name(values: list[str]) -> str:
    return "_".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", value) for value in values)


def write_csv(
    path: Path,
    pressure_bar: np.ndarray,
    altitude_km: np.ndarray,
    stats: dict[str, tuple[np.ndarray, np.ndarray]],
    references: dict[str, np.ndarray] | None,
) -> None:
    variables = list(stats)
    header = ["pressure_bar", "altitude_km"]
    for variable in variables:
        header.extend(
            [
                f"{variable}_mean",
                f"{variable}_std",
            ]
        )
        if references is not None:
            header.append(f"{variable}_reference")

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for i in range(pressure_bar.size):
            row = [float(pressure_bar[i]), float(altitude_km[i])]
            for variable in variables:
                mean, std = stats[variable]
                row.extend(
                    [
                        float(mean[i]),
                        float(std[i]),
                    ]
                )
                if references is not None:
                    row.append(float(references[variable][i]))
            writer.writerow(row)


def plot_profiles(
    path: Path,
    pressure_bar: np.ndarray,
    altitude_km: np.ndarray,
    stats: dict[str, tuple[np.ndarray, np.ndarray]],
    references: dict[str, np.ndarray] | None,
    xscale: str,
    max_pressure_bar: float,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 8.0), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (variable, (mean, std)) in enumerate(stats.items()):
        color = colors[index % len(colors)]
        ax.fill_betweenx(
            pressure_bar,
            mean - std,
            mean + std,
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )
        ax.plot(
            mean,
            pressure_bar,
            color=color,
            lw=2.0,
            label=f"{variable} mean +/- std",
        )
        if references is not None:
            ax.plot(
                references[variable],
                pressure_bar,
                color=color,
                linestyle="--",
                lw=1.6,
                label=f"{variable} reference",
            )

    ax.set_xscale(xscale)
    if xscale == "log":
        _, xmax = ax.get_xlim()
        ax.set_xlim(left=1.0e-9, right=xmax)
    ax.set_yscale("log")
    ax.invert_yaxis()
    pressure_bottom, pressure_top = ax.get_ylim()
    ax.set_ylim(min(max_pressure_bar, pressure_bottom), pressure_top)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pressure [bar]")
    ax.grid(True, which="both", alpha=0.25)
    for pressure_marker in PRESSURE_MARKERS_BAR:
        ax.axhline(
            pressure_marker,
            color="0.55",
            lw=2.0,
            alpha=0.65,
            zorder=0,
        )
    ax.legend(loc="best")

    ax_alt = ax.twinx()
    ax_alt.set_yscale("log")
    ax_alt.set_ylim(ax.get_ylim())
    ax_alt.set_ylabel("Altitude [km]")
    tick_pressures, tick_altitudes = altitude_ticks_to_pressure(
        ALTITUDE_TICKS_KM,
        pressure_bar,
        altitude_km,
    )
    ax_alt.set_yticks(tick_pressures)
    ax_alt.set_yticklabels([f"{value:g}" for value in tick_altitudes])

    fig.savefig(path, dpi=220)
    plt.close(fig)


def altitude_ticks_to_pressure(
    altitude_ticks_km: np.ndarray,
    pressure_bar: np.ndarray,
    altitude_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(altitude_km)
    altitude_sorted = altitude_km[order]
    log_pressure_sorted = np.log(pressure_bar[order])
    keep = (
        (altitude_ticks_km >= float(np.nanmin(altitude_sorted)))
        & (altitude_ticks_km <= float(np.nanmax(altitude_sorted)))
    )
    tick_altitudes = altitude_ticks_km[keep]
    tick_pressures = np.exp(
        np.interp(tick_altitudes, altitude_sorted, log_pressure_sorted)
    )
    return tick_pressures, tick_altitudes


def main() -> None:
    args = parse_args()
    if args.max_pressure <= 0.0:
        raise ValueError("--max-pressure must be positive")

    output_dir = resolve_output_dir(args.output, args.root)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    field_files = select_last_files(resolve_field_files(output_dir, args.field), args.last)
    validate_variables(field_files[-1].path, args.vars)
    if args.field == "out1" and PRESSURE_VARIABLE not in available_variables(field_files[-1].path):
        raise ValueError(f"{field_files[-1].path} does not contain {PRESSURE_VARIABLE!r}")

    x1_ref: np.ndarray | None = None
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    references: dict[str, np.ndarray] | None = None
    reference_file: OutputFile | None = None
    all_field_files = resolve_field_files(output_dir, args.field)
    if not args.no_reference:
        reference_file = find_snapshot_file(all_field_files, args.reference_id)
        validate_variables(reference_file.path, args.vars)
        references = {}

    for variable in args.vars:
        x1, mean, std = read_variable_stats(field_files, variable)
        if x1_ref is None:
            x1_ref = x1
        elif not np.allclose(x1_ref, x1):
            raise ValueError(f"x1 coordinate mismatch for {variable}")
        stats[variable] = (mean, std)
        if references is not None and reference_file is not None:
            reference_x1, reference = read_variable_reference(reference_file, variable)
            if not np.allclose(x1_ref, reference_x1):
                raise ValueError(f"x1 coordinate mismatch in reference for {variable}")
            references[variable] = reference

    if x1_ref is None:
        raise ValueError("No variables were read")

    pressure_x1, pressure_pa = read_pressure_profile(field_files)
    if not np.allclose(x1_ref, pressure_x1):
        raise ValueError("x1 coordinate mismatch between variable and pressure files")
    pressure_bar = pressure_pa / PRESSURE_REFERENCE_PA
    altitude_km = altitude_from_pressure(x1_ref, pressure_pa)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variable_name = safe_name(args.vars)
    base_name = f"{output_dir.name}_{args.field}_{variable_name}_profile_last{len(field_files)}"
    plot_path = args.output_dir / f"{base_name}.png"
    csv_path = args.output_dir / f"{base_name}.csv"

    plot_profiles(
        plot_path,
        pressure_bar,
        altitude_km,
        stats,
        references,
        args.xscale,
        args.max_pressure,
        args.xlabel,
    )
    write_csv(csv_path, pressure_bar, altitude_km, stats, references)

    print(f"Used {len(field_files)} snapshots from {output_dir}")
    print(f"First snapshot: {field_files[0].path.name}")
    print(f"Last snapshot: {field_files[-1].path.name}")
    if reference_file is not None:
        print(f"Reference snapshot: {reference_file.path.name}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
