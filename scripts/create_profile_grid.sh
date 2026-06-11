#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/um-jupiter-pycache}"

cd "$repo_root"

python scripts/plot_case_profile_grid.py \
    --row '^jup_crm2d_H2O-NH3_F10_nu(0\.01|0\.1|1\.0)$' \
    --row '^jup_crm3d_H2O-NH3-H2S_F10_nu(0\.01|0\.1|1\.0)$' \
    --output-name 2d_3d_F10_profile_grid \
    "$@"

python scripts/plot_case_profile_grid.py \
    --row '^jup_crm2d_H2O-NH3_F100_nu(0\.01|0\.1|1\.0)$' \
    --row '^jup_crm3d_H2O-NH3-H2S_F100_nu(0\.01|0\.1|1\.0)$' \
    --output-name 2d_3d_F100_profile_grid \
    "$@"
