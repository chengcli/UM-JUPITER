#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/um-jupiter-pycache}"

cd "$repo_root"

case_regex='jup_crm(2d_H2O-NH3|3d_H2O-NH3-H2S)_F(10|100)_nu0\.01'

echo "Creating RMS vertical-velocity profiles for: $case_regex"
python scripts/plot_case_rms_vel1_profiles.py \
    --case-regex "$case_regex" \
    --output-name jup_crm2d_3d_F10_F100_nu0.01_rms_vel1_profiles \
    "$@"
