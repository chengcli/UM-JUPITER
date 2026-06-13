#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${JUPITER_DATA_ROOT:-/home/chengcli/data/2026.JupiterCRM}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/um-jupiter-pycache}"

case_regex="${1:-jup_crm3d_H2O-NH3-H2S_F10_nu0\\.01}"
last="${2:-20}"
output_dir="${3:-diagnostics}"

cd "$repo_root"

echo "Rebuilding NH3 path-defined section cache..."
python scripts/plot_nh3_on_nh3_min_max_path_cross_section.py \
    --root "$data_root" \
    --case-regex "$case_regex" \
    --last "$last" \
    --output-dir "$output_dir" \
    --refresh-cache \
    --cache-only

echo "Rebuilding NH3-section dynamics cache..."
python scripts/cache_cross_section_dynamics.py \
    --root "$data_root" \
    --case-regex "$case_regex" \
    --last "$last" \
    --output-dir "$output_dir" \
    --path-species NH3

echo "Rendering NH3 primary cross-section..."
python scripts/plot_nh3_on_nh3_min_max_path_cross_section.py \
    --root "$data_root" \
    --case-regex "$case_regex" \
    --last "$last" \
    --output-dir "$output_dir"

echo "Rendering H2O vapor on the NH3-defined cross-section..."
python scripts/plot_h2o_on_nh3_min_max_path_cross_section.py \
    --root "$data_root" \
    --case-regex "$case_regex" \
    --last "$last" \
    --output-dir "$output_dir"

echo "Finished NH3-defined cross-section plots."
