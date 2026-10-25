#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${1:-/data01/2026.JUPITER_CRM/jup_beta_H2O-NH3_F100_Rhalf}"
if [ "$#" -gt 0 ]; then shift; fi
export DEVICE="${DEVICE:-cuda}"
export BACKEND="${BACKEND:-ucx}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
exec torchrun --standalone --nproc-per-node=2 "${script_dir}/run_jupiter_beta.py" \
  --config "${script_dir}/jupiter_beta.yaml" --output-dir "${output_dir}" "$@"
