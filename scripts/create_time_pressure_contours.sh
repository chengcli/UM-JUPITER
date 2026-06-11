#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${JUPITER_DATA_ROOT:-/home/chengcli/data/2026.JupiterCRM}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/um-jupiter-pycache}"

max_time="${1:-300}"
case_regexes=(
    'jup_crm2d_H2O-NH3_F100_nu0\.01'
    'jup_crm3d_H2O-NH3-H2S_F100_nu0\.01'
)

cd "$repo_root"

for case_regex in "${case_regexes[@]}"; do
    echo "Creating time-pressure contours for: $case_regex"

    python scripts/plot_rms_vel1_time_pressure_contour.py \
        --root "$data_root" \
        --case-regex "$case_regex" \
        --max-time "$max_time" \
        --vmax 2.5

    python scripts/plot_theta_time_pressure_contour.py \
        --root "$data_root" \
        --case-regex "$case_regex" \
        --max-time "$max_time" \
        --vmax 166 \
        --rms-running-average 3 \
        --rms-contour-min 0.6 \
        --rms-contour-spacing 0.2

    python scripts/plot_rms_vel1_theta_time_pressure_stack.py \
        --root "$data_root" \
        --case-regex "$case_regex" \
        --max-time "$max_time" \
        --rms-vmax 2.5 \
        --theta-vmax 166 \
        --rms-cmap YlOrRd \
        --theta-cmap YlGnBu
done
