#!/usr/bin/env python3
"""Plot horizontal filled contours at fixed pressure levels for one snapshot."""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-contours")

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
VELOCITY_POSITIVE_VARIABLE = "vel1p"
VELOCITY_NEGATIVE_VARIABLE = "vel1m"
PRESSURE_REFERENCE_PA = 1.0e5
PRESSURE_LEVELS_BAR = (0.5, 1.0, 5.0, 10.0, 20.0, 50.0)
QUANTILE_CONTOUR_LEVELS = np.arange(0.1, 1.0, 0.1)
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
            "Plot a compact 2x3 filled-contour panel for one variable at "
            "0.5, 1, 5, 10, 20, and 50 bar in one snapshot."
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
        help="NetCDF output field to read the variable from, e.g. out1 or out2.",
    )
    parser.add_argument(
        "--var",
        required=True,
        help=(
            "Variable name to plot. Use 'ke' for specific kinetic energy, "
            "'vel1p' for positive vel1, or 'vel1m' for negative vel1."
        ),
    )
    parser.add_argument(
        "--id",
        required=True,
        help="Snapshot id to plot, e.g. 00000.",
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
        help=f"Directory for PNG output. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Colorbar label. Default: variable name.",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap. Default: viridis",
    )
    return parser.parse_args()


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

    return matches


def find_snapshot_file(files: list[OutputFile], snapshot: str) -> OutputFile:
    snapshot = snapshot.zfill(len(files[0].snapshot))
    matches = [file_info for file_info in files if file_info.snapshot == snapshot]
    if matches:
        return sorted(matches, key=lambda item: (len(item.prefix), item.prefix))[0]
    raise FileNotFoundError(f"No {files[0].field} file found for snapshot id {snapshot}")


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


def read_3d_field(path: Path, variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            available = ", ".join(sorted(ds.data_vars))
            raise ValueError(
                f"{path.name} does not contain variable {variable!r}. "
                f"Available variables: {available}"
            )
        data = require_time_singleton(ds[variable], path)
        require_profile_dims(data, path)
        x2 = ds["x2"].values.astype(np.float64)
        x3 = ds["x3"].values.astype(np.float64)
        values = data.transpose("x1", "x3", "x2").values.astype(np.float64)
    return x2, x3, values


def read_kinetic_energy_field(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        for variable in VELOCITY_VARIABLES:
            if variable not in ds:
                available = ", ".join(sorted(ds.data_vars))
                raise ValueError(
                    f"Cannot compute {KINETIC_ENERGY_VARIABLE!r} from {path.name}; "
                    f"missing {variable!r}. Available variables: {available}"
                )

        x2 = ds["x2"].values.astype(np.float64)
        x3 = ds["x3"].values.astype(np.float64)
        kinetic_energy = None
        for variable in VELOCITY_VARIABLES:
            data = require_time_singleton(ds[variable], path)
            require_profile_dims(data, path)
            values = data.transpose("x1", "x3", "x2").values.astype(np.float64)
            term = values * values
            kinetic_energy = term if kinetic_energy is None else kinetic_energy + term

    if kinetic_energy is None:
        raise ValueError(f"Unable to compute {KINETIC_ENERGY_VARIABLE!r} from {path}")
    return x2, x3, 0.5 * kinetic_energy


def read_signed_velocity_part_field(
    path: Path,
    positive: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x2, x3, vel1 = read_3d_field(path, "vel1")
    if positive:
        return x2, x3, np.maximum(vel1, 0.0)
    return x2, x3, np.minimum(vel1, 0.0)


def interpolate_columns_to_pressure(
    pressure_bar: np.ndarray,
    values: np.ndarray,
    levels_bar: tuple[float, ...],
) -> dict[float, np.ndarray]:
    if pressure_bar.shape != values.shape:
        raise ValueError(
            f"Pressure shape {pressure_bar.shape} does not match variable shape {values.shape}"
        )

    nlevel, nx3, nx2 = values.shape
    pressure_flat = pressure_bar.reshape(nlevel, -1)
    values_flat = values.reshape(nlevel, -1)
    output = {
        pressure: np.full(nx3 * nx2, np.nan, dtype=np.float64)
        for pressure in levels_bar
    }

    for column in range(pressure_flat.shape[1]):
        column_pressure = pressure_flat[:, column]
        column_values = values_flat[:, column]
        keep = np.isfinite(column_pressure) & np.isfinite(column_values) & (column_pressure > 0.0)
        if np.count_nonzero(keep) < 2:
            continue
        order = np.argsort(np.log(column_pressure[keep]))
        log_pressure = np.log(column_pressure[keep])[order]
        profile = column_values[keep][order]
        for pressure in levels_bar:
            output[pressure][column] = np.interp(
                np.log(pressure),
                log_pressure,
                profile,
                left=np.nan,
                right=np.nan,
            )

    return {
        pressure: field.reshape(nx3, nx2)
        for pressure, field in output.items()
    }


def quantile_field(field: np.ndarray) -> np.ndarray:
    quantiles = np.full_like(field, np.nan, dtype=np.float64)
    finite = np.isfinite(field)
    values = field[finite]
    if values.size == 0:
        raise ValueError("No finite values available for contour plot")
    if values.size == 1:
        quantiles[finite] = 0.5
        return quantiles

    unique_values, counts = np.unique(values, return_counts=True)
    if unique_values.size == 1:
        quantiles[finite] = 0.5
        return quantiles

    count_before = np.cumsum(counts) - counts
    midrank = count_before + 0.5 * (counts - 1)
    unique_quantiles = midrank / float(values.size - 1)
    value_indices = np.searchsorted(unique_values, values)
    quantiles[finite] = unique_quantiles[value_indices]
    return quantiles


def plot_panels(
    path: Path,
    x2: np.ndarray,
    x3: np.ndarray,
    fields: dict[float, np.ndarray],
    variable: str,
    label: str,
    cmap: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 4.8), sharex=True, sharey=True)
    x2_km = x2 / 1000.0
    x3_km = x3 / 1000.0
    contour = None

    for ax, pressure in zip(axes.flat, PRESSURE_LEVELS_BAR, strict=True):
        field = quantile_field(fields[pressure])
        contour = ax.contourf(
            x2_km,
            x3_km,
            field,
            levels=QUANTILE_CONTOUR_LEVELS,
            cmap=cmap,
            extend="both",
        )
        ax.set_title(f"{pressure:g} bar", fontsize=10, pad=2)
        ax.set_aspect("equal", adjustable="box")

    for ax in axes.flat:
        ax.tick_params(labelbottom=False, labelleft=False, length=2, pad=1)
    axes[1, 0].tick_params(labelbottom=True, labelleft=True)
    axes[1, 0].set_xlabel("x2 [km]", labelpad=1)
    axes[1, 0].set_ylabel("x3 [km]", labelpad=1)

    if contour is None:
        raise ValueError("No contour panels were created")
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.94, wspace=0.03, hspace=0.10)
    fig.colorbar(contour, ax=axes, label=f"{label} quantile", fraction=0.035, pad=0.02)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output, args.root)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    field_files = resolve_field_files(output_dir, args.field)
    variable_file = find_snapshot_file(field_files, args.id)
    pressure_path = pressure_file_for(variable_file)
    if not pressure_path.exists():
        raise FileNotFoundError(
            f"Missing pressure file for {variable_file.path.name}: {pressure_path}"
        )

    _, _, pressure_pa = read_3d_field(pressure_path, PRESSURE_VARIABLE)
    pressure_bar = pressure_pa / PRESSURE_REFERENCE_PA

    if args.var == KINETIC_ENERGY_VARIABLE:
        x2, x3, values = read_kinetic_energy_field(pressure_path)
    elif args.var == VELOCITY_POSITIVE_VARIABLE:
        x2, x3, values = read_signed_velocity_part_field(pressure_path, positive=True)
    elif args.var == VELOCITY_NEGATIVE_VARIABLE:
        x2, x3, values = read_signed_velocity_part_field(pressure_path, positive=False)
    else:
        x2, x3, values = read_3d_field(variable_file.path, args.var)

    fields = interpolate_columns_to_pressure(pressure_bar, values, PRESSURE_LEVELS_BAR)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = (
        args.output_dir
        / f"{output_dir.name}_{args.field}_{safe_name(args.var)}_contours_{variable_file.snapshot}.png"
    )
    plot_panels(
        plot_path,
        x2,
        x3,
        fields,
        args.var,
        args.label or args.var,
        args.cmap,
    )

    print(f"Used snapshot: {variable_file.path.name}")
    print(f"Pressure file: {pressure_path.name}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
