#!/usr/bin/env python3
"""Plot vertical profiles of linear-fit tendencies from selected snapshots."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-tendencies")

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
SECONDS_PER_DAY = 86400.0
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
            "Plot vertical tendency profiles from linear fits to selected "
            "horizontal-mean snapshots."
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
        help="Variable names to fit and plot.",
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
        "--max-pressure",
        type=float,
        default=DEFAULT_MAX_PRESSURE_BAR,
        help=f"Maximum displayed pressure in bar. Default: {DEFAULT_MAX_PRESSURE_BAR:g}",
    )
    parser.add_argument(
        "--xlabel",
        default="tendency [per day]",
        help="Horizontal axis label. Default: 'tendency [per day]'",
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
        if first <= 1:
            raise ValueError("--first must be greater than 1 for a linear fit")
        if first > len(files):
            raise ValueError(f"--first requested {first} snapshots, only found {len(files)}")
        return files[:first]
    if last is not None:
        if last <= 1:
            raise ValueError("--last must be greater than 1 for a linear fit")
        if last > len(files):
            raise ValueError(f"--last requested {last} snapshots, only found {len(files)}")
        return files[-last:]
    if len(files) < 2:
        raise ValueError("Need at least two snapshots for a linear fit")
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
        missing = [name for name in variables if name not in ds]
        if missing:
            available = ", ".join(sorted(ds.data_vars))
            raise ValueError(
                f"Missing variable(s) in {path.name}: {', '.join(missing)}. "
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


def read_horizontal_mean_profile(path: Path, variable: str) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise ValueError(f"{path} does not contain variable {variable!r}")
        data = require_time_singleton(ds[variable], path)
        require_profile_dims(data, path)
        x1 = ds["x1"].values.astype(np.float64)
        values = data.transpose("x1", "x3", "x2").values.astype(np.float64)
    return x1, np.nanmean(values, axis=(1, 2))


def read_profiles(
    files: list[OutputFile],
    variable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1_ref: np.ndarray | None = None
    times = []
    profiles = []
    for file_info in files:
        x1, profile = read_horizontal_mean_profile(file_info.path, variable)
        if x1_ref is None:
            x1_ref = x1
        elif not np.allclose(x1_ref, x1):
            raise ValueError(f"x1 coordinate mismatch in {file_info.path}")
        times.append(read_time_days(file_info.path))
        profiles.append(profile)

    if x1_ref is None:
        raise ValueError(f"No profiles read for {variable}")

    time_days = np.array(times, dtype=np.float64)
    if not np.isfinite(time_days).all():
        time_days = np.arange(len(files), dtype=np.float64)
    return x1_ref, time_days, np.array(profiles, dtype=np.float64)


def fit_tendency(time_days: np.ndarray, profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.unique(time_days).size < 2:
        raise ValueError("Need at least two distinct times for a linear fit")
    centered_time = time_days - float(np.mean(time_days))
    denom = float(np.sum(centered_time * centered_time))
    if denom == 0.0:
        raise ValueError("Selected snapshots have zero time span")
    intercept = np.nanmean(profiles, axis=0)
    centered_profiles = profiles - intercept
    slope = np.nansum(centered_time[:, None] * centered_profiles, axis=0) / denom

    dof = time_days.size - 2
    if dof <= 0:
        return slope, np.full_like(slope, np.nan, dtype=np.float64)

    fitted = intercept[None, :] + centered_time[:, None] * slope[None, :]
    residual = profiles - fitted
    sigma2 = np.nansum(residual * residual, axis=0) / float(dof)
    slope_stderr = np.sqrt(sigma2 / denom)
    return slope, slope_stderr


def read_pressure_profile(files: list[OutputFile]) -> tuple[np.ndarray, np.ndarray]:
    x1_ref: np.ndarray | None = None
    profiles = []
    for file_info in files:
        path = pressure_file_for(file_info)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing pressure file for {file_info.path.name}: {path}"
            )
        x1, profile = read_horizontal_mean_profile(path, PRESSURE_VARIABLE)
        if x1_ref is None:
            x1_ref = x1
        elif not np.allclose(x1_ref, x1):
            raise ValueError(f"x1 coordinate mismatch in pressure file {path}")
        profiles.append(profile)

    if x1_ref is None:
        raise ValueError("No pressure profiles were read")
    return x1_ref, np.nanmean(np.array(profiles, dtype=np.float64), axis=0)


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


def safe_name(values: list[str]) -> str:
    return "_".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", value) for value in values)


def write_csv(
    path: Path,
    pressure_bar: np.ndarray,
    altitude_km: np.ndarray,
    tendencies: dict[str, np.ndarray],
    uncertainties: dict[str, np.ndarray],
) -> None:
    variables = list(tendencies)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["pressure_bar", "altitude_km"]
        for name in variables:
            header.extend([f"{name}_tendency_per_day", f"{name}_stderr_per_day"])
        writer.writerow(header)
        for i in range(pressure_bar.size):
            row = [float(pressure_bar[i]), float(altitude_km[i])]
            for name in variables:
                row.extend([float(tendencies[name][i]), float(uncertainties[name][i])])
            writer.writerow(row)


def plot_tendencies(
    path: Path,
    pressure_bar: np.ndarray,
    altitude_km: np.ndarray,
    tendencies: dict[str, np.ndarray],
    uncertainties: dict[str, np.ndarray],
    max_pressure_bar: float,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 8.0), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (variable, tendency) in enumerate(tendencies.items()):
        color = colors[index % len(colors)]
        stderr = uncertainties[variable]
        ax.fill_betweenx(
            pressure_bar,
            tendency - stderr,
            tendency + stderr,
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )
        ax.plot(
            tendency,
            pressure_bar,
            color=color,
            lw=2.0,
            label=variable,
        )

    ax.axvline(0.0, color="0.25", lw=1.2, alpha=0.8)
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


def main() -> None:
    args = parse_args()
    if args.max_pressure <= 0.0:
        raise ValueError("--max-pressure must be positive")

    output_dir = resolve_output_dir(args.output, args.root)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    selected_files = select_snapshots(
        resolve_field_files(output_dir, args.field),
        args.first,
        args.last,
    )
    validate_variables(selected_files[-1].path, args.vars)

    x1_ref: np.ndarray | None = None
    tendencies: dict[str, np.ndarray] = {}
    uncertainties: dict[str, np.ndarray] = {}
    for variable in args.vars:
        x1, time_days, profiles = read_profiles(selected_files, variable)
        if x1_ref is None:
            x1_ref = x1
        elif not np.allclose(x1_ref, x1):
            raise ValueError(f"x1 coordinate mismatch for {variable}")
        tendency, uncertainty = fit_tendency(time_days, profiles)
        tendencies[variable] = tendency
        uncertainties[variable] = uncertainty

    if x1_ref is None:
        raise ValueError("No variables were read")

    pressure_x1, pressure_pa = read_pressure_profile(selected_files)
    if not np.allclose(x1_ref, pressure_x1):
        raise ValueError("x1 coordinate mismatch between variable and pressure files")
    pressure_bar = pressure_pa / PRESSURE_REFERENCE_PA
    altitude_km = altitude_from_pressure(x1_ref, pressure_pa)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variable_name = safe_name(args.vars)
    base_name = (
        f"{output_dir.name}_{args.field}_{variable_name}_tendency_"
        f"{selected_files[0].snapshot}_{selected_files[-1].snapshot}"
    )
    plot_path = args.output_dir / f"{base_name}.png"
    csv_path = args.output_dir / f"{base_name}.csv"

    plot_tendencies(
        plot_path,
        pressure_bar,
        altitude_km,
        tendencies,
        uncertainties,
        args.max_pressure,
        args.xlabel,
    )
    write_csv(csv_path, pressure_bar, altitude_km, tendencies, uncertainties)

    print(f"Used {len(selected_files)} snapshots from {output_dir}")
    print(f"First snapshot: {selected_files[0].path.name}")
    print(f"Last snapshot: {selected_files[-1].path.name}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
