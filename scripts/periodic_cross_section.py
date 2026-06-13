"""Reusable horizontal sampling utilities for doubly periodic cross-sections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeriodicSection:
    distance: np.ndarray
    x2_unwrapped: np.ndarray
    x3_unwrapped: np.ndarray
    x2_wrapped: np.ndarray
    x3_wrapped: np.ndarray
    start_distance: float
    end_distance: float
    x2_lower: float
    x3_lower: float
    x2_period: float
    x3_period: float


def periodic_grid_bounds(coordinates: np.ndarray) -> tuple[float, float, float]:
    """Return lower edge, upper edge, and period for uniform cell centers."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("Periodic coordinates must be a one-dimensional array")
    spacing = np.diff(coordinates)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("Periodic cross-section sampling requires uniform coordinates")
    lower = float(coordinates[0] - 0.5 * spacing[0])
    period = float(spacing[0] * coordinates.size)
    return lower, lower + period, period


def shortest_periodic_displacement(
    start: np.ndarray,
    end: np.ndarray,
    periods: np.ndarray,
) -> np.ndarray:
    """Return the shortest component-wise displacement from start to end."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    periods = np.asarray(periods, dtype=np.float64)
    return (end - start + 0.5 * periods) % periods - 0.5 * periods


def _first_boundary_parameters(
    point: np.ndarray,
    direction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    periods: np.ndarray,
) -> tuple[float, float]:
    forward = []
    backward = []
    for coordinate, delta, low, high, period in zip(
        point, direction, lower, upper, periods
    ):
        if delta > 0.0:
            first_forward = (high - coordinate) / delta
            backward.append((low - coordinate) / delta)
        elif delta < 0.0:
            first_forward = (low - coordinate) / delta
            backward.append((high - coordinate) / delta)
        else:
            continue
        crossing_interval = period / abs(delta)
        if first_forward < 1.0:
            first_forward += np.ceil((1.0 - first_forward) / crossing_interval) * crossing_interval
        forward.append(first_forward)
    if not forward:
        raise ValueError("Cross-section points must not be identical")
    return max(backward), min(forward)


def build_periodic_section(
    x2: np.ndarray,
    x3: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    sample_spacing: float | None = None,
) -> PeriodicSection:
    """Build the first-boundary-to-first-boundary line through two points."""
    x2_lower, x2_upper, x2_period = periodic_grid_bounds(x2)
    x3_lower, x3_upper, x3_period = periodic_grid_bounds(x3)
    lower = np.array([x2_lower, x3_lower])
    upper = np.array([x2_upper, x3_upper])
    periods = np.array([x2_period, x3_period])
    start_point = np.asarray(start, dtype=np.float64)
    direction = shortest_periodic_displacement(start_point, np.asarray(end), periods)
    segment_length = float(np.linalg.norm(direction))
    if segment_length == 0.0:
        raise ValueError("Cross-section points must not be periodic equivalents")

    t_min, t_max = _first_boundary_parameters(
        start_point, direction, lower, upper, periods
    )
    if not (t_min < 0.0 < 1.0 <= t_max):
        raise ValueError("The first-boundary section does not contain both points")

    if sample_spacing is None:
        sample_spacing = min(float(np.diff(x2)[0]), float(np.diff(x3)[0]))
    if sample_spacing <= 0.0:
        raise ValueError("sample_spacing must be positive")
    total_length = (t_max - t_min) * segment_length
    intervals = max(1, int(np.ceil(total_length / sample_spacing)))
    parameters = np.linspace(t_min, t_max, intervals + 1)
    unwrapped = start_point[:, None] + direction[:, None] * parameters
    wrapped = (unwrapped - lower[:, None]) % periods[:, None] + lower[:, None]
    distance = parameters * segment_length
    return PeriodicSection(
        distance=distance,
        x2_unwrapped=unwrapped[0],
        x3_unwrapped=unwrapped[1],
        x2_wrapped=wrapped[0],
        x3_wrapped=wrapped[1],
        start_distance=0.0,
        end_distance=segment_length,
        x2_lower=x2_lower,
        x3_lower=x3_lower,
        x2_period=x2_period,
        x3_period=x3_period,
    )


def sample_periodic_bilinear(
    values: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    sample_x2: np.ndarray,
    sample_x3: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample arrays whose final dimensions are (x3, x2)."""
    values = np.asarray(values)
    x2 = np.asarray(x2, dtype=np.float64)
    x3 = np.asarray(x3, dtype=np.float64)
    sample_x2 = np.asarray(sample_x2, dtype=np.float64)
    sample_x3 = np.asarray(sample_x3, dtype=np.float64)
    if values.shape[-2:] != (x3.size, x2.size):
        raise ValueError(
            f"Expected final dimensions {(x3.size, x2.size)}, got {values.shape[-2:]}"
        )
    if sample_x2.shape != sample_x3.shape:
        raise ValueError("sample_x2 and sample_x3 must have matching shapes")

    dx2 = float(np.diff(x2)[0])
    dx3 = float(np.diff(x3)[0])
    periodic_grid_bounds(x2)
    periodic_grid_bounds(x3)
    index_x2 = ((sample_x2 - x2[0]) / dx2) % x2.size
    index_x3 = ((sample_x3 - x3[0]) / dx3) % x3.size
    i0 = np.floor(index_x2).astype(int)
    j0 = np.floor(index_x3).astype(int)
    i1 = (i0 + 1) % x2.size
    j1 = (j0 + 1) % x3.size
    weight_x2 = index_x2 - i0
    weight_x3 = index_x3 - j0

    v00 = values[..., j0, i0]
    v01 = values[..., j0, i1]
    v10 = values[..., j1, i0]
    v11 = values[..., j1, i1]
    return (
        v00 * (1.0 - weight_x3) * (1.0 - weight_x2)
        + v01 * (1.0 - weight_x3) * weight_x2
        + v10 * weight_x3 * (1.0 - weight_x2)
        + v11 * weight_x3 * weight_x2
    )
