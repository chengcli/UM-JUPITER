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
- A combined RMS vertical-velocity profile plot.
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
  --case-regex 'jup_crm2d_H2O-NH3_F10_nu(0\.01|0\.1|1\.0)' \
  --last 20
```

Useful options:

- `--root PATH`: directory containing case folders.
- `--case-regex REGEX`: full-match regex selecting case folders to compare.
- `--last N`: number of final snapshots to average.
- `--max-pressure BAR`: deepest displayed pressure.
- `--no-std`: disable the mean-plus/minus-standard-deviation shading.

The black curve is the snapshot-`00000` initial theta profile.

### RMS Vertical Velocity

`scripts/plot_case_rms_vel1_profiles.py` plots the root-mean-square vertical
velocity profile for explicitly selected cases. At each vertical level, it
computes `sqrt(mean(vel1^2))` over horizontal cells and the selected latest
snapshots. Shading shows the standard deviation of the per-snapshot horizontal
RMS profiles.

```bash
python scripts/plot_case_rms_vel1_profiles.py \
  --case-regex 'jup_crm2d_H2O-NH3_F10_nu(0\.01|0\.1|1\.0)' \
  --last 20
```

The script reads `vel1` from `out1` and writes a combined PNG and CSV table.
Use `--no-std` to disable the temporal standard-deviation shading.

### Species, Clouds, and Precipitation

`scripts/plot_case_vapor_profiles.py` creates one plot per selected species:

- Solid colored line: vapor.
- Dashed colored line: cloud condensate.
- Dotted colored line: precipitation.
- Black line: snapshot-`00000` initial vapor profile.
- Shading: vapor mean plus/minus standard deviation.

```bash
python scripts/plot_case_vapor_profiles.py \
  --case-regex 'jup_crm2d_H2O-NH3_F10_nu(0\.01|0\.1|1\.0)' \
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

### Species Path Contours

`scripts/plot_species_path_contours.py` creates one three-panel horizontal map
per matched case using the vapor, cloud, and precipitation path diagnostics
from `out2`. Each map is the time average of the latest selected snapshots.

```bash
python scripts/plot_species_path_contours.py \
  --case-regex 'jup_crm3d_H2O-NH3-H2S_F10_nu(0\.01|0\.1|1\.0)' \
  --species H2O \
  --last 20
```

The default invocation runs this H2O/3D-F10 selection and writes one PNG per
case under `diagnostics/`. Use `--species NH3` or `--species H2S` for the
other species.

To mark strong vertical-motion columns, first create a compact cache from
`vel1` in the latest snapshots:

```bash
python scripts/cache_vel1_column_exceedance.py \
  --case-regex 'jup_crm3d_H2O-NH3-H2S_F10_nu0\.01' \
  --last 20 --threshold 1 --max-locations 500
```

A location qualifies when any snapshot has `vel1` above the effective
threshold anywhere in its vertical column. The threshold starts at the
requested value and increases in `0.5 m s^-1` increments until no more than
`--max-locations` locations qualify. The cache retains per-snapshot masks,
counts, fractions, the union mask, and maximum `vel1`.

Use the cache to add black `X` markers to a species-path plot:

```bash
python scripts/plot_species_path_contours.py \
  --case-regex 'jup_crm3d_H2O-NH3-H2S_F10_nu0\.01' \
  --species H2O --last 20 \
  --vel1-exceedance-cache diagnostics/vel1_column_exceedance_cache/jup_crm3d_H2O-NH3-H2S_F10_nu0.01_vel1_gt_1_last20.npz
```

Run the complete kappa=0.01 workflow, including the adaptive cache, marked
H2O plot, and standard NH3/H2S plots:

```bash
scripts/create_species_path_contours.sh
```

Optional positional arguments set the snapshot count, initial threshold, and
maximum marked locations:

```bash
scripts/create_species_path_contours.sh 20 1.0 500
```

### Periodic H2O Minimum-to-Maximum Cross-Section

`scripts/create_h2o_min_max_path_cross_section.sh` rebuilds the section and
dynamics caches, then renders the final cross-section figure:

```bash
scripts/create_h2o_min_max_path_cross_section.sh
```

Its optional positional arguments are case regex, number of final snapshots,
and output directory:

```bash
scripts/create_h2o_min_max_path_cross_section.sh \
  'jup_crm3d_H2O-NH3-H2S_F10_nu0\.01' 20 diagnostics
```

The horizontal section uses the shortest periodic direction from minimum to
maximum, then extends from the minimum in both directions until each side
first crosses either horizontal domain boundary. It is sampled with periodic
bilinear interpolation at approximately the native horizontal grid spacing.
The vapor panel plots H2O fractional deviation from the section-horizontal
mean at every vertical level, `(H2O - mean(H2O)) / mean(H2O)`. It uses 11
linear `PiYG` levels from `-0.5` through `0.5`. NH3 vapor is retained in the
section cache but is not plotted. Five opaque
filled H2O cloud levels are overlaid using `Blues`, logarithmically spaced
from `1e-5` through `1e-3`, with a thin black contour at `1e-5`. The vertical
axis uses the sampled time-mean pressure on an inverted logarithmic scale and
displays pressures through 50 bar. Compact section coordinates, pressure,
extrema metadata, and vapor/cloud arrays are cached under
`diagnostics/h2o_min_max_path_cross_section_cache/`.
Projected velocities and streamfunction are cached under
`diagnostics/cross_section_dynamics_cache/`. The final PNG is written directly
under `diagnostics/`. Set `JUPITER_DATA_ROOT` to use a data root other than
`/home/chengcli/data/2026.JupiterCRM`.

The reusable helpers in `scripts/periodic_cross_section.py` build a
first-boundary periodic section between any two horizontal points and
bilinearly sample arbitrary arrays whose final dimensions are `(x3, x2)`.
`scripts/custom_colormaps.py` provides `diverging_with_white_plateau()`, which
creates a custom colormap from an existing diverging colormap and maps values
within a user-selected half-width around the center to pure white.
`scripts/cross_section_dynamics.py` provides reusable horizontal velocity
projection and least-squares mass-streamfunction calculations. Generate the
projected velocity and mass-flux cache with:

```bash
python scripts/cache_cross_section_dynamics.py \
  --case-regex 'jup_crm3d_H2O-NH3-H2S_F10_nu0\.01' \
  --last 20
```

The cross-section plot shows perpendicular velocity by default. Positive
values at `[4, 8, 12, 16, 20] m s^-1` use labeled black solid contours;
negative values at `[-4, -8, -12, -16, -20] m s^-1` use labeled black dashed
contours.
Use `--no-show-perp-velocity` to disable them or `--show-perp-velocity` to
explicitly enable them. Streamfunction is cached but hidden by default;
restore its dashed overlay with `--show-streamfunction`.

See [docs/h2o_min_max_cross_section.md](docs/h2o_min_max_cross_section.md) for
the complete generation workflow and a focused guide to tuning contour
levels, colormaps, pressure limits, and colorbar dimensions.

## Multi-Row Profile Grid

`scripts/plot_case_profile_grid.py` consolidates existing profile CSV/cache
values into one figure with three columns per row:

- H2O horizontal-mean profiles.
- NH3 horizontal-mean profiles.
- Potential-temperature horizontal-mean profiles.

Repeat `--row` to add rows. Each row regex must match one experiment family;
all matching cases are overlaid in that row's panels.

```bash
python scripts/plot_case_profile_grid.py \
  --row '^jup_crm3d_H2O-NH3-H2S_F10_nu[0-9.]+$' \
  --row '^jup_crm3d_H2O-NH3-H2S_F100_nu[0-9.]+$' \
  --output-name crm3d_F10_F100_profile_grid
```

The script is read-only with respect to source data: it requires existing
profile CSVs and initial-profile NPZ caches. If any required cache is missing,
it exits before plotting and prints the exact `create_all_plots.sh` command
needed to generate that row's caches. Initial profiles are drawn by default
from `diagnostics/initial_profile_cache/`; disable them with `--no-initial`.

## Stacked Time-Series Scripts

### Potential Temperature at Fixed Pressures

`scripts/plot_theta_stacked_time_series.py` plots theta at 1, 6, and 30 bar in
compact stacked case panels:

- Solid: 1 bar.
- Dashed: 6 bar.
- Dotted: 30 bar.

```bash
python scripts/plot_theta_stacked_time_series.py \
  --case-regex 'jup_crm2d_H2O-NH3_F100_nu(0\.01|0\.1|1\.0)'
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
  --case-regex 'jup_crm2d_H2O-NH3_F100_nu(10\.0|100\.0|1000\.0)' \
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

### RMS Vertical-Velocity Time-Pressure Contour

Plots horizontal RMS `vel1` versus time and log pressure for one case:

```bash
python scripts/plot_rms_vel1_time_pressure_contour.py \
  --case-regex 'jup_crm2d_H2O-NH3_F100_nu0\.01' \
  --max-time 300
```

The first run writes a compact NPZ cache under
`diagnostics/rms_vel1_time_pressure_cache/`. Use `--refresh-cache` after
simulation output changes. `--running-average` is a centered time window in
days. The standard comparison plots use unaveraged values, `YlOrRd`, discrete
`0.25 m s^-1` levels, and a `2.5 m s^-1` upper limit.

### Potential-Temperature Time-Pressure Contour

Plots horizontal-mean potential temperature versus time and log pressure:

```bash
python scripts/plot_theta_time_pressure_contour.py \
  --case-regex 'jup_crm2d_H2O-NH3_F100_nu0\.01' \
  --max-time 300
```

The default shared contour range is `158-166 K` at `0.5 K` spacing. Compact
per-case caches are stored under `diagnostics/theta_time_pressure_cache/`.
Black labeled contours show RMS-`vel1` from the matching RMS cache; their
default range starts at `0.6 m s^-1` with `0.2 m s^-1` spacing, after a
centered 3-day running average.

### Stacked RMS-Velocity and Theta Contours

Composes existing RMS-`vel1` and theta caches into a two-row figure:

```bash
python scripts/plot_rms_vel1_theta_time_pressure_stack.py \
  --case-regex 'jup_crm2d_H2O-NH3_F100_nu0\.01'
```

The top panel shows RMS-`vel1`; the bottom panel shows potential temperature.
The script exits with cache-generation commands if either required cache is
missing. The composed figure uses `YlOrRd` for RMS-`vel1`, `YlGnBu` for theta,
and a 2:1 horizontal-to-vertical aspect ratio.

### Create Standard Time-Pressure Contours

`scripts/create_time_pressure_contours.sh` creates all three time-pressure
products for the 2D and 3D F100, kappa=0.01 cases:

- Unaveraged RMS-`vel1` contours through day 300.
- Unaveraged theta contours with black labeled RMS-`vel1` overlays. Only the
  overlay is smoothed with a centered 3-day running average.
- Two-row RMS-`vel1`/theta composed plots.

```bash
scripts/create_time_pressure_contours.sh
```

Pass a different maximum time in days as the first argument:

```bash
scripts/create_time_pressure_contours.sh 500
```

The script reads simulation folders from
`/home/chengcli/data/2026.JupiterCRM`. Override this with
`JUPITER_DATA_ROOT=/path/to/cases`. Existing NPZ caches are reused; run the
individual source scripts with `--refresh-cache` after simulation results
change.

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
python scripts/plot_case_rms_vel1_profiles.py --help
python scripts/plot_case_vapor_profiles.py --help
python scripts/plot_species_path_contours.py --help
python scripts/cache_vel1_column_exceedance.py --help
python scripts/plot_h2o_min_max_path_cross_section.py --help
python scripts/cache_cross_section_dynamics.py --help
python scripts/plot_theta_stacked_time_series.py --help
python scripts/plot_rms_vel1_time_pressure_contour.py --help
python scripts/plot_theta_time_pressure_contour.py --help
python scripts/plot_rms_vel1_theta_time_pressure_stack.py --help
```
