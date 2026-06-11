#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${JUPITER_DATA_ROOT:-/home/chengcli/data/2026.JupiterCRM}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/um-jupiter-pycache}"

if (($# < 1)); then
    echo "Usage: $0 CASE_REGEX [plot options...]" >&2
    echo "Example: $0 '^jup_crm3d_H2O-NH3-H2S_F10_nu[0-9.]+$' --last 20" >&2
    exit 2
fi

case_regex="$1"
shift

mapfile -t cases < <(
    find "$data_root" -maxdepth 1 -type d -printf '%f\n' |
        grep -E "$case_regex" |
        sort -V
)

if ((${#cases[@]} == 0)); then
    echo "No case directories under $data_root matched regex: $case_regex" >&2
    exit 1
fi

experiment="${cases[0]%_nu*}"
for case_name in "${cases[@]}"; do
    if [[ "${case_name%_nu*}" != "$experiment" ]]; then
        echo "Regex matched multiple experiment families; use a narrower regex." >&2
        printf '  %s\n' "${cases[@]}" >&2
        exit 1
    fi
done

species=(H2O NH3)
if [[ "$experiment" == *H2S* ]]; then
    species+=(H2S)
fi

cd "$repo_root"

echo "Matched ${#cases[@]} cases from $data_root:"
printf '  %s\n' "${cases[@]}"

echo "Creating potential-temperature profile plot..."
python scripts/plot_case_theta_profiles.py \
    --root "$data_root" \
    --cases "${cases[@]}" \
    "$@"

echo "Creating ${species[*]} species profile plots..."
python scripts/plot_case_vapor_profiles.py \
    --root "$data_root" \
    --cases "${cases[@]}" \
    --species "${species[@]}" \
    "$@"

echo "Finished creating plots for $experiment in diagnostics/."
