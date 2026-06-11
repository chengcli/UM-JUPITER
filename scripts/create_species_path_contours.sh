#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${JUPITER_DATA_ROOT:-/home/chengcli/data/2026.JupiterCRM}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/um-jupiter-pycache}"

case_regex='jup_crm3d_H2O-NH3-H2S_F10_nu0\.01'
case_name='jup_crm3d_H2O-NH3-H2S_F10_nu0.01'
last="${1:-20}"
requested_threshold="${2:-1.0}"
max_locations="${3:-500}"
printf -v threshold_compact '%g' "$requested_threshold"
cache="diagnostics/vel1_column_exceedance_cache/${case_name}_vel1_gt_${threshold_compact//./p}_last${last}.npz"

cd "$repo_root"

echo "Caching adaptive vertical-velocity column exceedances..."
python scripts/cache_vel1_column_exceedance.py \
    --root "$data_root" \
    --case-regex "$case_regex" \
    --last "$last" \
    --threshold "$requested_threshold" \
    --max-locations "$max_locations"

echo "Creating H2O path contours with vertical-velocity X markers..."
python scripts/plot_species_path_contours.py \
    --root "$data_root" \
    --case-regex "$case_regex" \
    --species H2O \
    --last "$last" \
    --vel1-exceedance-cache "$cache"

for species in NH3 H2S; do
    echo "Creating ${species} path contours..."
    python scripts/plot_species_path_contours.py \
        --root "$data_root" \
        --case-regex "$case_regex" \
        --species "$species" \
        --last "$last"
done

echo "Finished species path contours for ${case_name}."
