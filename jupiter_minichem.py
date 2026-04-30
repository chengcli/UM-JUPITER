from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import pyminichem

CHEMISTRY_SPECIES_NAMES = (
    "OH",
    "H2",
    "H2O",
    "H",
    "CO",
    "CO2",
    "O",
    "CH4",
    "C2H2",
    "NH3",
    "N2",
    "HCN",
    "He",
)
SPECIES_NAMES = (
    "He",
    "OH",
    "H2",
    "H2O",
    "H",
    "CO",
    "CO2",
    "O",
    "CH4",
    "C2H2",
    "NH3",
    "N2",
    "HCN",
)
ACTIVE_SPECIES_NAMES = CHEMISTRY_SPECIES_NAMES[:-1]
INERT_SPECIES_NAME = SPECIES_NAMES[0]
CHEM_TO_STORAGE = tuple(SPECIES_NAMES.index(name) for name in CHEMISTRY_SPECIES_NAMES)
STORAGE_TO_CHEM = tuple(CHEMISTRY_SPECIES_NAMES.index(name) for name in SPECIES_NAMES)

MOLECULAR_WEIGHTS = torch.tensor(
    [4.0, 17.01, 2.02, 18.02, 1.01, 28.01, 44.01, 16.0, 16.05, 26.04, 17.04, 28.02, 27.03],
    dtype=torch.float64,
)


def molecular_weights_like(ref: torch.Tensor) -> torch.Tensor:
    return MOLECULAR_WEIGHTS.to(device=ref.device, dtype=ref.dtype)


def vmr_to_mmr(vmr: torch.Tensor) -> torch.Tensor:
    mu = molecular_weights_like(vmr)
    raw = vmr * mu.view(-1, *([1] * (vmr.dim() - 1)))
    total = raw.sum(dim=0, keepdim=True)
    return raw / total.clamp_min(torch.finfo(raw.dtype).tiny)


def mmr_to_vmr(mmr: torch.Tensor) -> torch.Tensor:
    mu = molecular_weights_like(mmr)
    raw = mmr / mu.view(-1, *([1] * (mmr.dim() - 1)))
    total = raw.sum(dim=0, keepdim=True)
    return raw / total.clamp_min(torch.finfo(raw.dtype).tiny)


def scalar_u_to_mmr(scalar_s: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    return scalar_s / rho.unsqueeze(0).clamp_min(torch.finfo(scalar_s.dtype).tiny)


def reorder_species(data: torch.Tensor, order: tuple[int, ...]) -> torch.Tensor:
    index = torch.tensor(order, device=data.device, dtype=torch.long)
    return data.index_select(0, index)


def build_pyminichem() -> pyminichem.MiniChem:
    options = pyminichem.MiniChemOptions()
    options.metallicity("1x")
    mc = pyminichem.MiniChem(options)
    mc.initialize()
    species = tuple(mc.species_names())
    if species != ACTIVE_SPECIES_NAMES:
        raise RuntimeError(
            f"PyMiniChem species order mismatch: expected {ACTIVE_SPECIES_NAMES}, got {species}"
        )
    return mc


def _find_bracket(values: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    upper = torch.searchsorted(values, x, right=False)
    upper = upper.clamp(1, values.numel() - 1)
    lower = upper - 1
    x0 = values.index_select(0, lower)
    x1 = values.index_select(0, upper)
    frac = (x - x0) / (x1 - x0).clamp_min(torch.finfo(x.dtype).tiny)
    return lower, upper, frac


@dataclass(frozen=True)
class EquilibriumTable:
    species_names: tuple[str, ...]
    temperatures: torch.Tensor
    log_pressures: torch.Tensor
    vmr_grid: torch.Tensor

    @classmethod
    def from_resource(cls, resource_name: str = "chem_data/IC/mini_chem_IC_FastChem_1x.txt") -> "EquilibriumTable":
        path = Path(pyminichem.find_resource(resource_name))
        with path.open("r", encoding="utf-8") as f:
            header = f.readline().split()
            if len(header) != 4:
                raise RuntimeError(f"Unexpected IC header in {path}")
            ntemp, npres, nrow, nspecies = map(int, header)
            species_names = tuple(f.readline().split())
            temperatures = torch.tensor(
                [float(x) for x in f.readline().split()], dtype=torch.float64
            )
            pressures = torch.tensor(
                [float(x) for x in f.readline().split()], dtype=torch.float64
            )
            rows = [
                [float(x) for x in line.split()]
                for line in f
                if line.strip()
            ]

        if len(rows) != nrow:
            raise RuntimeError(f"Expected {nrow} IC rows in {path}, found {len(rows)}")
        if len(species_names) != len(CHEMISTRY_SPECIES_NAMES):
            raise RuntimeError(
                f"Expected {len(CHEMISTRY_SPECIES_NAMES)} IC species in {path}, found {len(species_names)}"
            )
        if nspecies != len(species_names):
            raise RuntimeError(
                f"Expected {nspecies} species from IC header in {path}, found {len(species_names)}"
            )
        if species_names != CHEMISTRY_SPECIES_NAMES:
            raise RuntimeError(
                f"IC species order mismatch: expected {CHEMISTRY_SPECIES_NAMES}, got {species_names}"
            )

        data = torch.tensor(rows, dtype=torch.float64)
        # First column is mean molecular weight; remaining columns are species VMR.
        vmr_grid = data[:, 1:].reshape(npres, ntemp, len(species_names))
        return cls(
            species_names=species_names,
            temperatures=temperatures,
            log_pressures=pressures.log10(),
            vmr_grid=vmr_grid,
        )

    def interpolate_vmr(self, temp: torch.Tensor, pres: torch.Tensor) -> torch.Tensor:
        shape = temp.shape
        flat_temp = temp.detach().to(dtype=torch.float64).reshape(-1).cpu()
        flat_logp = pres.detach().to(dtype=torch.float64).log10().reshape(-1).cpu()

        tmin = self.temperatures[0]
        tmax = self.temperatures[-1]
        pmin = self.log_pressures[0]
        pmax = self.log_pressures[-1]
        flat_temp = flat_temp.clamp(tmin, tmax)
        flat_logp = flat_logp.clamp(pmin, pmax)

        t0, t1, wt = _find_bracket(self.temperatures, flat_temp)
        p0, p1, wp = _find_bracket(self.log_pressures, flat_logp)

        f00 = self.vmr_grid[p0, t0]
        f01 = self.vmr_grid[p0, t1]
        f10 = self.vmr_grid[p1, t0]
        f11 = self.vmr_grid[p1, t1]

        wt = wt.unsqueeze(-1)
        wp = wp.unsqueeze(-1)
        interp = (
            (1.0 - wp) * (1.0 - wt) * f00
            + (1.0 - wp) * wt * f01
            + wp * (1.0 - wt) * f10
            + wp * wt * f11
        )

        interp = interp.transpose(0, 1).reshape(len(self.species_names), *shape)
        return interp.to(device=temp.device, dtype=temp.dtype)


def initialize_scalar_mmr(table: EquilibriumTable, temp: torch.Tensor, pres: torch.Tensor) -> torch.Tensor:
    vmr = reorder_species(table.interpolate_vmr(temp, pres), CHEM_TO_STORAGE)
    return vmr_to_mmr(vmr)


def advance_minichem(
    mc: pyminichem.MiniChem,
    temp: torch.Tensor,
    pres: torch.Tensor,
    scalar_s: torch.Tensor,
    rho: torch.Tensor,
    dt: float,
) -> None:
    mmr_old = scalar_u_to_mmr(scalar_s, rho)
    vmr_old = mmr_to_vmr(mmr_old)
    vmr_chem_old = reorder_species(vmr_old, STORAGE_TO_CHEM)

    active = vmr_chem_old[: len(ACTIVE_SPECIES_NAMES)].movedim(0, -1).reshape(-1, len(ACTIVE_SPECIES_NAMES))
    inert = vmr_chem_old[len(ACTIVE_SPECIES_NAMES)]

    flat_temp = temp.reshape(-1)
    flat_pres = pres.reshape(-1)
    active_new = active.clone()

    mask = flat_temp > 200.0
    if mask.any():
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        temp_sel = flat_temp.index_select(0, idx)
        pres_sel = flat_pres.index_select(0, idx)
        active_sel = active_new.index_select(0, idx).contiguous()
        updated_sel = mc.forward(temp_sel, pres_sel, active_sel, dt)
        active_new.index_copy_(0, idx, updated_sel)

    vmr_chem_new = torch.cat(
        [
            active_new.reshape(*temp.shape, len(ACTIVE_SPECIES_NAMES)).movedim(-1, 0),
            inert.unsqueeze(0),
        ],
        dim=0,
    )
    vmr_new = reorder_species(vmr_chem_new, CHEM_TO_STORAGE)
    mmr_new = vmr_to_mmr(vmr_new)
    scalar_s.add_(rho.unsqueeze(0) * (mmr_new - mmr_old))
