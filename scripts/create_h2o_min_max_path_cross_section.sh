#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${JUPITER_DATA_ROOT:-/home/chengcli/data/2026.JupiterCRM}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/um-jupiter-pycache}"

case_regex="${1:-jup_crm3d_H2O-NH3-H2S_F10_nu0\\.01}"
last="${2:-20}"
output_dir="${3:-diagnostics}"

cd "$repo_root"

#echo "Rebuilding H2O periodic cross-section cache..."
#python scripts/plot_h2o_min_max_path_cross_section.py \
#    --root "$data_root" \
#    --case-regex "$case_regex" \
#    --last "$last" \
#    --output-dir "$output_dir" \
#    --refresh-cache \
#    --cache-only

#echo "Rebuilding projected-velocity and streamfunction cache..."
#python scripts/cache_cross_section_dynamics.py \
#    --root "$data_root" \
#    --case-regex "$case_regex" \
#    --last "$last" \
#    --output-dir "$output_dir"

echo "Rendering H2O periodic cross-section..."
python scripts/plot_h2o_min_max_path_cross_section.py \
    --root "$data_root" \
    --case-regex "$case_regex" \
    --last "$last" \
    --output-dir "$output_dir"

echo "Finished H2O periodic cross-section."
