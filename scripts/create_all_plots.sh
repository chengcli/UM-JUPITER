#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/um-jupiter-pycache}"

cd "$repo_root"

echo "Creating F10 potential-temperature profile plot..."
python scripts/plot_case_theta_profiles.py "$@"

echo "Creating F10 H2O and NH3 profile plots..."
python scripts/plot_case_vapor_profiles.py "$@"

echo "Finished creating all F10 plots in diagnostics/."
