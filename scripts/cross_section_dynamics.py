"""Reusable velocity projection and mass-streamfunction calculations."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr


def cross_section_basis(
    start_x2: float,
    start_x3: float,
    end_x2: float,
    end_x3: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return along-section and left-normal horizontal unit vectors."""
    displacement = np.array([end_x2 - start_x2, end_x3 - start_x3], dtype=np.float64)
    length = float(np.linalg.norm(displacement))
    if length == 0.0:
        raise ValueError("Cross-section endpoints must be distinct")
    tangent = displacement / length
    normal = np.array([-tangent[1], tangent[0]])
    return tangent, normal


def project_horizontal_velocity(
    vel2: np.ndarray,
    vel3: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project x2/x3 velocities into along-section and left-normal components."""
    if vel2.shape != vel3.shape:
        raise ValueError("vel2 and vel3 must have matching shapes")
    tangent = np.asarray(tangent, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    if tangent.shape != (2,) or normal.shape != (2,):
        raise ValueError("tangent and normal must each contain two components")
    parallel = tangent[0] * vel2 + tangent[1] * vel3
    perpendicular = normal[0] * vel2 + normal[1] * vel3
    return parallel, perpendicular


def mass_streamfunction_least_squares(
    distance_m: np.ndarray,
    altitude_m: np.ndarray,
    parallel_mass_flux: np.ndarray,
    vertical_mass_flux: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit psi so dpsi/dz=parallel flux and -dpsi/ds=vertical flux.

    The returned streamfunction has units kg m^-1 s^-1 when the mass fluxes
    have units kg m^-2 s^-1 and coordinates are in meters.
    """
    distance_m = np.asarray(distance_m, dtype=np.float64)
    altitude_m = np.asarray(altitude_m, dtype=np.float64)
    parallel_mass_flux = np.asarray(parallel_mass_flux, dtype=np.float64)
    vertical_mass_flux = np.asarray(vertical_mass_flux, dtype=np.float64)
    shape = (altitude_m.size, distance_m.size)
    if parallel_mass_flux.shape != shape or vertical_mass_flux.shape != shape:
        raise ValueError(f"Mass-flux arrays must have shape {shape}")
    if np.any(np.diff(distance_m) <= 0.0) or np.any(np.diff(altitude_m) <= 0.0):
        raise ValueError("Distance and altitude coordinates must increase")

    rows: list[int] = []
    columns: list[int] = []
    coefficients: list[float] = []
    targets: list[float] = []
    equation = 0
    nz, ns = shape

    def index(j: int, i: int) -> int:
        return j * ns + i

    for j, dz in enumerate(np.diff(altitude_m)):
        face_flux = 0.5 * (parallel_mass_flux[j] + parallel_mass_flux[j + 1])
        for i, target in enumerate(face_flux):
            rows.extend((equation, equation))
            columns.extend((index(j, i), index(j + 1, i)))
            coefficients.extend((-1.0 / dz, 1.0 / dz))
            targets.append(float(target))
            equation += 1

    for i, ds in enumerate(np.diff(distance_m)):
        face_flux = 0.5 * (vertical_mass_flux[:, i] + vertical_mass_flux[:, i + 1])
        for j, target in enumerate(face_flux):
            rows.extend((equation, equation))
            columns.extend((index(j, i), index(j, i + 1)))
            coefficients.extend((1.0 / ds, -1.0 / ds))
            targets.append(float(target))
            equation += 1

    # Fix the arbitrary additive constant without affecting physical gradients.
    gauge_weight = min(
        float(np.median(1.0 / np.diff(distance_m))),
        float(np.median(1.0 / np.diff(altitude_m))),
    )
    rows.append(equation)
    columns.append(index(0, 0))
    coefficients.append(gauge_weight)
    targets.append(0.0)
    equation += 1

    matrix = sparse.coo_matrix(
        (coefficients, (rows, columns)),
        shape=(equation, nz * ns),
    ).tocsr()
    solution = lsqr(matrix, np.asarray(targets), atol=1.0e-11, btol=1.0e-11)
    streamfunction = solution[0].reshape(shape)
    diagnostics = {
        "residual_norm": float(solution[3]),
        "normal_equation_residual_norm": float(solution[4]),
        "iterations": float(solution[2]),
    }
    return streamfunction, diagnostics
