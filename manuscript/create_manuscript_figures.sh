#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/um-jupiter-pycache}"

usage() {
  cat <<'EOF'
Usage:
  manuscript/create_manuscript_figures.sh [entry]

Entries:
  evaporation-timescales   Analytical-adiabat precipitation evaporation timescales
  profile-grid             3D/2D F100 H2O-NH3 profile grid
  all                      Generate all manuscript figures

Default entry: all
EOF
}

evaporation_timescales() {
  python3 "${script_dir}/plot_precipitation_evaporation_timescales.py" \
    --output "${script_dir}/precipitation_evaporation_timescales_adiabat.png" \
    --csv "${script_dir}/precipitation_evaporation_timescales_adiabat.csv"
}

profile_grid() {
  local row_3d='^jup_crm3d_H2O-NH3_F100_nu(0\.01|0\.1|1\.0)$'
  local row_2d='^jup_crm2d_H2O-NH3_F100_nu(0\.01|0\.1|1\.0)$'

  "${repo_root}/scripts/create_all_plots.sh" "${row_3d}" \
    --last 20 \
    --output-dir "${script_dir}"

  "${repo_root}/scripts/create_all_plots.sh" "${row_2d}" \
    --last 20 \
    --output-dir "${script_dir}"

  python3 "${script_dir}/plot_case_profile_grid.py" \
    --root "${JUPITER_DATA_ROOT:-/home/chengcli/data/2026.JupiterCRM}" \
    --row 'jup_crm3d_H2O-NH3_F100_nu(0\.01|0\.1|1\.0)' \
    --row 'jup_crm2d_H2O-NH3_F100_nu(0\.01|0\.1|1\.0)' \
    --last 20 \
    --output-dir "${script_dir}" \
    --output-name "jup_crm3d_2d_H2O-NH3_F100_profile_grid"
}

entry="${1:-all}"

cd "${repo_root}"
case "${entry}" in
  evaporation-timescales)
    evaporation_timescales
    ;;
  profile-grid)
    profile_grid
    ;;
  all)
    evaporation_timescales
    profile_grid
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
