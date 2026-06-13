# Species Minimum-to-Maximum Periodic Cross-Sections

This workflow creates periodic vertical cross-sections defined by the minimum
and maximum time-mean vapor-path locations of either H2O or NH3.

For each path-defining species, two figures can be rendered:

| Path-defined section | Primary plot | Companion plot |
| --- | --- | --- |
| H2O | H2O vapor fractional deviation with H2O cloud | NH3 vapor fractional deviation with NH3 cloud |
| NH3 | NH3 vapor fractional deviation with NH3 cloud | H2O vapor fractional deviation |

All figures can include perpendicular velocity contours projected relative to
their own path-defined section.

## Generate Complete Workflows

Run from the repository root:

```bash
scripts/create_h2o_min_max_path_cross_sections.sh \
  'jup_crm3d_H2O-NH3-H2S_F10_nu0\.01' 20 diagnostics

scripts/create_nh3_min_max_path_cross_sections.sh \
  'jup_crm3d_H2O-NH3-H2S_F10_nu0\.01' 20 diagnostics
```

The optional positional arguments are case regex, number of final snapshots,
and output directory. Set `JUPITER_DATA_ROOT` to override the default data
root:

```bash
JUPITER_DATA_ROOT=/path/to/data \
  scripts/create_nh3_min_max_path_cross_sections.sh
```

The H2O wrapper currently renders both H2O-defined figures from existing
section and dynamics caches. Its cache-rebuild commands are retained as
commented blocks in the script. The NH3 wrapper rebuilds its section and
dynamics caches before rendering both NH3-defined figures.

## Symmetric Plot Scripts

The script name has the form:

```text
plot_<displayed-species>_on_<path-defining-species>_min_max_path_cross_section.py
```

The four scripts are:

```text
scripts/plot_h2o_on_h2o_min_max_path_cross_section.py
scripts/plot_nh3_on_h2o_min_max_path_cross_section.py
scripts/plot_h2o_on_nh3_min_max_path_cross_section.py
scripts/plot_nh3_on_nh3_min_max_path_cross_section.py
```

## Build Caches Manually

Build an H2O-defined section cache:

```bash
python scripts/plot_h2o_on_h2o_min_max_path_cross_section.py \
  --case-regex 'jup_crm3d_H2O-NH3-H2S_F10_nu0\.01' \
  --last 20 --refresh-cache --cache-only
```

Build an NH3-defined section cache:

```bash
python scripts/plot_nh3_on_nh3_min_max_path_cross_section.py \
  --case-regex 'jup_crm3d_H2O-NH3-H2S_F10_nu0\.01' \
  --last 20 --refresh-cache --cache-only
```

Build the matching projected-velocity and streamfunction cache:

```bash
python scripts/cache_cross_section_dynamics.py \
  --case-regex 'jup_crm3d_H2O-NH3-H2S_F10_nu0\.01' \
  --last 20 \
  --path-species H2O

python scripts/cache_cross_section_dynamics.py \
  --case-regex 'jup_crm3d_H2O-NH3-H2S_F10_nu0\.01' \
  --last 20 \
  --path-species NH3
```

## Outputs

Section caches:

```text
diagnostics/h2o_min_max_path_cross_section_cache/
diagnostics/nh3_min_max_path_cross_section_cache/
```

Dynamics caches:

```text
diagnostics/cross_section_dynamics_cache/*_H2O_min_max_path_dynamics_last20.npz
diagnostics/cross_section_dynamics_cache/*_NH3_min_max_path_dynamics_last20.npz
```

Figures:

```text
diagnostics/*_H2O_on_H2O_min_max_path_cross_section_last20.png
diagnostics/*_NH3_on_H2O_min_max_path_cross_section_last20.png
diagnostics/*_H2O_on_NH3_min_max_path_cross_section_last20.png
diagnostics/*_NH3_on_NH3_min_max_path_cross_section_last20.png
```

Each species-section cache stores H2O and NH3 vapor/cloud cross-sections so
companion figures can be rendered without rereading simulation snapshots.

## Cross-Section Definition

For a selected path species, the cache builder:

1. Averages `path_<species>` over the final selected `out2` snapshots.
2. Finds the horizontal minimum and maximum of that mean vapor path.
3. Uses the shortest displacement between them in the doubly periodic domain.
4. Extends the line from the minimum in both directions until each side first
   crosses a horizontal domain boundary.
5. Samples H2O, NH3, their cloud fields, and pressure using periodic bilinear
   interpolation.
6. Averages the sampled cross-sections over the selected snapshots.

Reusable implementations:

```text
scripts/periodic_cross_section.py
scripts/species_path_cross_section.py
```

## Direction Definitions

The model coordinates and velocities are:

- `x1`, `vel1`: vertical coordinate and vertical velocity. Positive `vel1`
  points upward.
- `x2`, `vel2`: first horizontal coordinate and velocity component.
- `x3`, `vel3`: second horizontal coordinate and velocity component.

The positive parallel direction points from the minimum-path location toward
the maximum-path location of the species defining the section. For unwrapped
horizontal displacement `delta = (delta_x2, delta_x3)`:

```text
length = sqrt(delta_x2^2 + delta_x3^2)
tangent = (delta_x2 / length, delta_x3 / length)
normal = (-tangent_x3, tangent_x2)
```

The positive perpendicular direction is the left-normal direction when facing
along the positive parallel direction:

```text
u_vertical      = vel1
u_parallel      = tangent_x2 * vel2 + tangent_x3 * vel3
u_perpendicular = normal_x2  * vel2 + normal_x3  * vel3
```

The figure plots inverted logarithmic pressure rather than geometric height,
so upward motion corresponds visually to decreasing pressure.

## Velocity and Streamfunction Overlays

Perpendicular velocity is enabled by default:

- Positive contours: black solid at `[4, 8, 12, 16, 20] m s^-1`.
- Negative contours: black dashed at `[-20, -16, -12, -8, -4] m s^-1`.

Use `--no-show-perp-velocity` to disable it. Streamfunction is cached but
hidden by default; enable it with `--show-streamfunction`.

Projection is performed for each snapshot before averaging. The cache stores
`mean(rho * u_parallel)` and `mean(rho * vel1)`. The in-section mass
streamfunction is the least-squares solution of:

```text
d(psi)/dz = mean(rho * u_parallel)
-d(psi)/ds = mean(rho * vel1)
```

Reusable dynamics implementations:

```text
scripts/cross_section_dynamics.py
scripts/cache_cross_section_dynamics.py
```

## Vapor Fractional Deviation

At every vertical level, vapor is plotted as fractional deviation from its
horizontal section mean:

```text
(vapor - horizontal_mean(vapor)) / horizontal_mean(vapor)
```

Current shared settings:

```python
MAX_PRESSURE_BAR = 50.0
VAPOR_FRACTIONAL_DEVIATION_LIMIT = 0.5
VAPOR_FRACTIONAL_WHITE_HALF_WIDTH = 0.05
```

The vapor field uses 11 linear levels from `-0.5` through `0.5` and the
`PiYG` colormap. Values from `-0.05` through `0.05` use a flat white plateau.
The reusable colormap helper is:

```text
scripts/custom_colormaps.py
```

## Cloud Contours

Primary species plots overlay cloud mass fraction using five logarithmically
spaced levels:

```python
CLOUD_LEVELS = np.geomspace(1.0e-5, 1.0e-3, 5)
```

Values below `1e-5` are masked. Filled contours use `Blues`, and a thin blue
contour marks the `1e-5` threshold.

## Layout

Plots use logarithmic pressure decreasing upward and stop at `50 bar`.
Primary vapor/cloud plots use two vertically stacked right-side colorbars:
vapor on top and cloud below. Vapor-only companion plots use one right-side
colorbar.

Useful options:

```bash
# Render from existing caches
python scripts/plot_h2o_on_h2o_min_max_path_cross_section.py

# Select another case
python scripts/plot_nh3_on_nh3_min_max_path_cross_section.py \
  --case-regex 'jup_crm3d_H2O-NH3-H2S_F10_nu0\.1'

# Change snapshot count or output directory
python scripts/plot_h2o_on_nh3_min_max_path_cross_section.py \
  --last 40 \
  --output-dir diagnostics/experiment_a
```

The case regex must match exactly one case directory.

## Path-Extrema Profile Figures

Create one three-panel profile figure at H2O or NH3 path extrema:

```bash
python scripts/plot_path_min_max_profiles.py \
  --case-regex 'jup_crm3d_H2O-NH3-H2S_F10_nu0\.01' \
  --last 20 \
  --path-species H2O

python scripts/plot_path_min_max_profiles.py --path-species NH3
```

Panels show:

1. H2O vapor, cloud, and precipitation.
2. NH3 vapor, cloud, and precipitation.
3. Potential temperature.

Species use the existing profile styles: solid vapor, dashed cloud, and
dotted precipitation. Species axes are logarithmic. Path-minimum profiles use
`PiYG(0.0)`, the magenta limit; path-maximum profiles use `PiYG(1.0)`, the
green limit. Black solid lines show the cached snapshot-00000 initial H2O,
NH3, and potential-temperature profiles.

The script reads min/max locations and selected snapshots from the matching
H2O- or NH3-defined section cache. It caches local pressure, species, and theta
profiles under:

```text
diagnostics/path_extrema_profile_cache/
```

Rebuild that compact cache after simulation data changes:

```bash
python scripts/plot_path_min_max_profiles.py --path-species NH3 --refresh-cache
```
