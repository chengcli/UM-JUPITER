# UM-JUPITER Dry MiniChem Notes

## What Was Implemented

This workspace now includes a Python reimplementation of the Canoe
`examples/2024-XZhang-minichem/dry_mini.cpp` dry-Jupiter chemistry workflow
using:

- Snapy for hydrodynamics and passive-scalar transport
- `pyminichem` for the NCHO chemistry update

Added files:

- [run_jupiter_dry_minichem.py](/home/chengcli/scix/workspace/UM-JUPITER/run_jupiter_dry_minichem.py)
- [jupiter_minichem.py](/home/chengcli/scix/workspace/UM-JUPITER/jupiter_minichem.py)
- [jupiter_crm_dry_minichem.yaml](/home/chengcli/scix/workspace/UM-JUPITER/jupiter_crm_dry_minichem.yaml)

## Chemistry / Scalar Design

The implementation uses 13 passive scalars:

- `He` as the inert species
- 12 active MiniChem species:
  `OH, H2, H2O, H, CO, CO2, O, CH4, C2H2, NH3, N2, HCN`

Passive-scalar storage order is:

`He, OH, H2, H2O, H, CO, CO2, O, CH4, C2H2, NH3, N2, HCN`

Important representation rules:

- Snapy `scalar_r` is mass mixing ratio
- Snapy `scalar_s` is conserved scalar density `rho * scalar_r`
- FastChem IC table values are molar mixing ratios
- `pyminichem` input/output is molar mixing ratio for the 12 active species

All MMR/VMR conversion and species reordering is handled explicitly in
`jupiter_minichem.py`. The IC table and `pyminichem` keep their native
chemistry ordering internally; the runner remaps to the passive-scalar storage
ordering at the boundaries.

## Initialization and Update Path

Hydro initialization:

- Uses the existing dry-Jupiter profile setup through `paddle.setup_profile`
- Applies the same random vertical-velocity perturbation pattern as the local
  Python runners

Scalar initialization:

- Reads `chem_data/IC/mini_chem_IC_FastChem_1x.txt` from packaged
  `pyminichem` resources
- Interpolates equilibrium VMR from the table
- Converts VMR to normalized MMR
- Stores the result in `scalar_r`

Chemistry update:

- After each accepted hydro step, recover MMR from `scalar_s / rho`
- Convert MMR to VMR
- Reorder to MiniChem chemistry order
- Evolve only the 12 active species with `pyminichem`
- Keep `He` inert
- Convert back to storage order and MMR
- Update conserved scalars with `rho * (mmr_new - mmr_old)`

The chemistry update keeps the Canoe-style temperature gate:

- no chemistry evolution for cells with `T <= 200 K`

## Snapy Output Changes

Snapy was updated in:

- [load_scalar_output_data.cpp](/home/chengcli/scix/repos/snapy/src/output/load_scalar_output_data.cpp)

Changes:

- primitive scalar output no longer requires `scalar_s` to be present
- new aggregate selectors were added:
  - `scalar_prim`
  - `scalar_cons`
  - `scalar`

This allows YAML output sections to request all scalar primitive fields without
listing every `r_<name>` entry.

Updated Snapy example YAMLs:

- [straka.yaml](/home/chengcli/scix/repos/snapy/examples/straka.yaml)
- [shallow_xy.yaml](/home/chengcli/scix/repos/snapy/examples/shallow_xy.yaml)
- [shallow_splash.yaml](/home/chengcli/scix/repos/snapy/examples/shallow_splash.yaml)

These now use `scalar_prim`.

## Current Dry-Jupiter Config

`jupiter_crm_dry_minichem.yaml` includes:

- 13 passive scalars with `He` first
- scalar transport through Snapy upwind scalar solver
- a dedicated scalar NetCDF stream using:

`variables: [scalar_prim]`

## Verification Performed

Checked successfully:

- Python syntax for the new runner/helper modules
- MMR/VMR round-trip consistency at machine precision
- `He`-first passive-scalar ordering
- scalar NetCDF output written through `scalar_prim`

Reduced smoke runs created scalar output files such as:

- `/tmp/um_jupiter_smoke_scalarprim/*.out3.00000.nc`

Those files contain the expected scalar primitive fields:

- `r_He, r_OH, r_H2, r_H2O, r_H, r_CO, r_CO2, r_O, r_CH4, r_C2H2, r_NH3, r_N2, r_HCN`

## Caveat

The small reduced smoke case used for validation can still trigger a hydro redo
and terminate abnormally on that tiny setup. The passive-scalar initialization,
species ordering, chemistry remap logic, and scalar-output plumbing were
verified, but this is not yet a statement that the reduced toy case is
numerically robust for production use.
