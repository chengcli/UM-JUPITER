#!/usr/bin/env python3
"""Plot NH3 ideal saturation vapor pressure as a function of temperature."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-um-jupiter-nh3-svp")

import matplotlib as mpl

mpl.use("Agg")
mpl.rc_file(Path(__file__).resolve().parents[1] / "matplotlibrc")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT = Path("diagnostics/nh3_saturation_vapor_pressure.png")
DEFAULT_TMIN = 80.0
DEFAULT_TMAX = 350.0
DEFAULT_NTEMP = 600
DEFAULT_XMIN = 130.0
DEFAULT_XMAX = 160.0
DEFAULT_YMIN = 0.1
DEFAULT_YMAX = 100.0
DEFAULT_PRESSURE_REFERENCES = (24.6, 12.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot NH3 saturation vapor pressure from Kintera's nh3_ideal formula."
    )
    parser.add_argument("--tmin", type=float, default=DEFAULT_TMIN)
    parser.add_argument("--tmax", type=float, default=DEFAULT_TMAX)
    parser.add_argument("--ntemp", type=int, default=DEFAULT_NTEMP)
    parser.add_argument("--xmin", type=float, default=DEFAULT_XMIN)
    parser.add_argument("--xmax", type=float, default=DEFAULT_XMAX)
    parser.add_argument("--ymin", type=float, default=DEFAULT_YMIN)
    parser.add_argument("--ymax", type=float, default=DEFAULT_YMAX)
    parser.add_argument(
        "--pressure-reference",
        type=float,
        nargs="+",
        default=list(DEFAULT_PRESSURE_REFERENCES),
        help="Pressure reference value(s) [Pa]. Each gets a horizontal line and solved T marker.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def logsvp_ideal(temperature_ratio: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    return (1.0 - 1.0 / temperature_ratio) * beta - gamma * np.log(temperature_ratio)


def nh3_ideal_logsvp(temperature: np.ndarray) -> np.ndarray:
    beta_liquid = 20.08
    gamma_liquid = 5.62
    beta_solid = 20.64
    gamma_solid = 1.43
    triple_temperature = 195.4
    triple_pressure = 6060.0
    ratio = temperature / triple_temperature
    return (
        np.where(
            temperature > triple_temperature,
            logsvp_ideal(ratio, beta_liquid, gamma_liquid),
            logsvp_ideal(ratio, beta_solid, gamma_solid),
        )
        + np.log(triple_pressure)
    )


def nh3_ideal_svp(temperature: float | np.ndarray) -> float | np.ndarray:
    return np.exp(nh3_ideal_logsvp(np.asarray(temperature, dtype=np.float64)))


def solve_temperature_for_svp(
    pressure_pa: float,
    tmin: float,
    tmax: float,
    iterations: int = 80,
) -> float:
    low = tmin
    high = tmax
    p_low = float(nh3_ideal_svp(low))
    p_high = float(nh3_ideal_svp(high))
    if not p_low <= pressure_pa <= p_high:
        raise ValueError(
            f"Pressure reference {pressure_pa:g} Pa is outside the SVP range "
            f"{p_low:g}-{p_high:g} Pa over {tmin:g}-{tmax:g} K"
        )

    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if float(nh3_ideal_svp(mid)) < pressure_pa:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def main() -> None:
    args = parse_args()
    if args.tmin <= 0.0:
        raise ValueError("--tmin must be positive")
    if args.tmax <= args.tmin:
        raise ValueError("--tmax must be greater than --tmin")
    if args.ntemp < 2:
        raise ValueError("--ntemp must be at least 2")
    if args.xmax <= args.xmin:
        raise ValueError("--xmax must be greater than --xmin")
    if args.ymin <= 0.0:
        raise ValueError("--ymin must be positive for a log axis")
    if args.ymax <= args.ymin:
        raise ValueError("--ymax must be greater than --ymin")
    if any(pressure <= 0.0 for pressure in args.pressure_reference):
        raise ValueError("--pressure-reference values must be positive")

    temperature = np.linspace(args.tmin, args.tmax, args.ntemp)
    pressure_pa = nh3_ideal_svp(temperature)
    reference_temperatures = [
        solve_temperature_for_svp(pressure, args.tmin, args.tmax)
        for pressure in args.pressure_reference
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    ax.plot(temperature, pressure_pa, color="tab:orange", lw=2.0)
    colors = ("0.20", "0.45", "0.65", "0.80")
    for index, (pressure, temp) in enumerate(
        zip(args.pressure_reference, reference_temperatures)
    ):
        color = colors[index % len(colors)]
        ax.axhline(
            pressure,
            color=color,
            lw=1.5,
            ls=":",
            label=f"{pressure:g} Pa",
        )
        ax.axvline(
            temp,
            color=color,
            lw=1.5,
            ls="--",
            label=f"{temp:.2f} K",
        )
    ax.set_yscale("log")
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("NH$_3$ saturation vapor pressure [Pa]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    for pressure, temp in zip(args.pressure_reference, reference_temperatures):
        print(f"NH3 SVP {pressure:g} Pa at T = {temp:.6f} K")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
