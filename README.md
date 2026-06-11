# UM-JUPITER Plotting Tools

Plotting scripts for JUPITER CRM NetCDF output are under `scripts/`. Generated
figures, CSV tables, and compact caches are written to `diagnostics/` by
default.

Run commands from the repository root:

```bash
cd /home/chengcli/scix/workspace/UM-JUPITER
```

The default simulation-data root used by the case-comparison scripts is:

```text
/home/chengcli/data/2026.JupiterCRM
```

## Create All Case Profile Plots

`scripts/create_all_plots.sh` discovers case directories using an extended
regular expression and creates:

- A combined potential-temperature profile plot.
- H2O and NH3 vapor/cloud/precipitation profile plots.
- An H2S profile plot when the experiment name contains `H2S`.
- A CSV table corresponding to each plot.

Usage:

```bash
scripts/create_all_plots.sh CASE_REGEX [plot options...]
```

Examples:

```bash
# All standard 2D F10 cases
scripts/create_all_plots.sh '^jup_crm2d_H2O-NH3_F10_nu[0-9.]+$'

# All standard 3D F100 H2O/NH3/H2S cases
scripts/create_all_plots.sh '^jup_crm3d_H2O-NH3-H2S_F100_nu[0-9.]+$'

# Average the latest 50 snapshots instead of the default 20
scripts/create_all_plots.sh '^jup_crm2d_H2O-NH3_F100_nu[0-9.]+$' --last 50

# Write outputs somewhere else
scripts/create_all_plots.sh '^jup_crm2d_H2O-NH3_F10_nu[0-9.]+$' \
  --output-dir diagnostics/f10
```

The regex must select one experiment family. For example, do not use a regex
that selects both F10 and F100 cases. Specialized case suffixes can be excluded
by anchoring the expression with `$`.

Set a different simulation root with:

```bash
JUPITER_DATA_ROOT=/path/to/cases scripts/create_all_plots.sh CASE_REGEX
```

## Combined Profile Scripts

### Potential Temperature

`scripts/plot_case_theta_profiles.py` plots horizontal-mean potential
temperature profiles for explicitly selected cases. Each profile is averaged
over the latest snapshots and plotted against log pressure.

```bash
python scripts/plot_case_theta_profiles.py \
  --cases case_nu0.01 case_nu0.1 case_nu1.0 \
  --last 20
```

Useful options:

- `--root PATH`: directory containing case folders.
- `--cases CASE ...`: ordered list of cases to compare.
- `--last N`: number of final snapshots to average.
- `--max-pressure BAR`: deepest displayed pressure.
- `--no-std`: disable the mean-plus/minus-standard-deviation shading.

The black curve is the snapshot-`00000` initial theta profile.

### Species, Clouds, and Precipitation

`scripts/plot_case_vapor_profiles.py` creates one plot per selected species:

- Solid colored line: vapor.
- Dashed colored line: cloud condensate.
- Dotted colored line: precipitation.
- Black line: snapshot-`00000` initial vapor profile.
- Shading: vapor mean plus/minus standard deviation.

```bash
python scripts/plot_case_vapor_profiles.py \
  --cases case_nu0.01 case_nu0.1 case_nu1.0 \
  --species H2O NH3 H2S \
  --last 20
```

The horizontal mass-fraction axis is logarithmic with a lower limit of
`1e-8`. Species mappings are:

| Species | Vapor | Cloud | Precipitation |
| --- | --- | --- | --- |
| H2O | `H2O` | `H2O_l_` | `H2O_l_p_` |
| NH3 | `NH3` | `NH3_s_` | `NH3_s_p_` |
| H2S | `H2S` | `NH4SH_s_` | `NH4SH_s_p_` |

For F10 experiments, initial profiles are extracted from the corresponding
F100 case's snapshot `00000`. Compact pressure/profile arrays are saved under
`diagnostics/initial_profile_cache/` and reused by later runs.

## Stacked Time-Series Scripts

### Potential Temperature at Fixed Pressures

`scripts/plot_theta_stacked_time_series.py` plots theta at 1, 6, and 30 bar in
compact stacked case panels:

- Solid: 1 bar.
- Dashed: 6 bar.
- Dotted: 30 bar.

```bash
python scripts/plot_theta_stacked_time_series.py \
  --cases case_nu0.01 case_nu0.1 case_nu1.0
```

Each case is read in a separate process. Compact per-case NPZ files are stored
under `diagnostics/theta_stacked_cache/`, making subsequent plotting runs much
faster.

Use `--refresh-cache` after simulation output changes:

```bash
python scripts/plot_theta_stacked_time_series.py --refresh-cache
```

Use `--first N` or `--last N` to restrict the snapshot selection.

### Temperature Gradient at Fixed Pressures

`scripts/plot_dtdz_pressure_stacked_time_series.py` computes horizontal-mean
temperature profiles, evaluates `dT/dz` in `K km^-1`, and samples the result at
1, 6, and 30 bar.

```bash
python scripts/plot_dtdz_pressure_stacked_time_series.py \
  --cases case_nu10.0 case_nu100.0 case_nu1000.0 \
  --refresh-cache
```

Per-case caches are stored under `diagnostics/dtdz_cache/`.

## Generic Plotting Subroutines

These scripts operate on one output directory at a time. Use an absolute output
path or provide a folder name under `--root`.

### Horizontal-Mean Vertical Profiles

```bash
python scripts/plot_horizontal_mean_profiles.py /path/to/case \
  --field out2 --vars temp theta --last 20 \
  --xlabel 'Temperature [K]'
```

Supports linear/log x scales, optional snapshot-`00000` reference profiles, CSV
output, and the derived `N2` variable.

### Fixed-Pressure Time Evolution

Plots horizontal means at 0.5, 1, 5, 10, 20, and 50 bar:

```bash
python scripts/plot_time_evolution.py /path/to/case \
  --field out2 --vars theta --ylabel 'Potential temperature [K]'
```

### Vertical Tendency Profiles

Fits linear time tendencies from selected snapshots:

```bash
python scripts/plot_vertical_tendency_profiles.py /path/to/case \
  --field out2 --vars theta --last 100
```

### Pressure-Level Contours

Creates a compact 2x3 panel at 0.5, 1, 5, 10, 20, and 50 bar:

```bash
python scripts/plot_pressure_level_contours.py /path/to/case \
  --field out1 --var vel1 --id 00500
```

Special variables include `ke`, `vel1p`, and `vel1m`.

### Pressure-Level Spectra

Creates 2D FFT power spectra at the same six pressure levels:

```bash
python scripts/plot_pressure_level_spectra.py /path/to/case \
  --field out1 --var ke --id 00500
```

## Styling and Outputs

The comparison scripts load the repository-level `matplotlibrc`, which defines
the shared font, tick, axes, and line-width styling.

Most scripts write:

- A PNG figure.
- A CSV table containing the plotted values.

Use each script's built-in help for the complete current interface:

```bash
python scripts/plot_case_theta_profiles.py --help
python scripts/plot_case_vapor_profiles.py --help
python scripts/plot_theta_stacked_time_series.py --help
```
