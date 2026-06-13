# H2O Minimum-to-Maximum Periodic Cross-Section

This plot shows the time-mean vertical cross-section passing through the
minimum and maximum H2O vapor-path locations.

The base field is H2O vapor fractional deviation from the horizontal mean.
Filled H2O cloud contours are overlaid on the same panel. NH3 vapor is cached
for later use but is not plotted.

## Generate the Plot

Run from the repository root:

```bash
scripts/create_h2o_min_max_path_cross_section.sh
```

This rebuilds the section cache, rebuilds the projected-velocity and
streamfunction cache, and then renders the figure. Optional positional
arguments select the case regex, number of final snapshots, and output
directory:

```bash
scripts/create_h2o_min_max_path_cross_section.sh \
  'jup_crm3d_H2O-NH3-H2S_F10_nu0\.01' 20 diagnostics
```

Set `JUPITER_DATA_ROOT` to override the default data root:

```bash
JUPITER_DATA_ROOT=/path/to/data \
  scripts/create_h2o_min_max_path_cross_section.sh
```

The wrapper reads the NetCDF snapshots and writes:

```text
diagnostics/h2o_min_max_path_cross_section_cache/
diagnostics/cross_section_dynamics_cache/
diagnostics/jup_crm3d_H2O-NH3-H2S_F10_nu0.01_H2O_min_max_path_cross_section_last20.png
```

To render directly from existing caches without rebuilding them:

```bash
python scripts/plot_h2o_min_max_path_cross_section.py \
  --case-regex 'jup_crm3d_H2O-NH3-H2S_F10_nu0\.01' \
  --last 20
```

The standard species workflow also generates this plot:

```bash
scripts/create_species_path_contours.sh
```

## Cross-Section Definition

The script:

1. Averages `path_H2O` from the final selected `out2` snapshots.
2. Finds the horizontal minimum and maximum of the mean H2O vapor path.
3. Uses the shortest direction between them in the doubly periodic domain.
4. Extends the line from the minimum in both directions until each side first
   crosses a horizontal domain boundary.
5. Samples `H2O`, `NH3`, `H2O_l_`, and `press` using periodic bilinear
   interpolation.
6. Averages the sampled cross-sections over the selected snapshots.

Reusable geometry and sampling functions are in:

```text
scripts/periodic_cross_section.py
```

## Velocity and Streamfunction Overlay

Build the dynamics cache after the section cache exists:

```bash
python scripts/cache_cross_section_dynamics.py \
  --case-regex 'jup_crm3d_H2O-NH3-H2S_F10_nu0\.01' \
  --last 20
```

The dynamics cache is written under:

```text
diagnostics/cross_section_dynamics_cache/
```

### Vertical, Parallel, and Perpendicular Directions

The model coordinates and velocities are:

- `x1`, `vel1`: vertical coordinate and vertical velocity. Positive `vel1`
  points toward increasing `x1`, which is upward in the model.
- `x2`, `vel2`: first horizontal coordinate and its velocity component.
- `x3`, `vel3`: second horizontal coordinate and its velocity component.

The plot's horizontal coordinate is distance along the periodic cross-section.
The positive parallel direction points along the shortest periodic displacement
from the minimum H2O vapor-path location toward the maximum H2O vapor-path
location. Let this unwrapped horizontal displacement be:

```text
delta = (delta_x2, delta_x3)
length = sqrt(delta_x2^2 + delta_x3^2)
```

The parallel unit vector is:

```text
tangent = (tangent_x2, tangent_x3)
        = (delta_x2 / length, delta_x3 / length)
```

The positive perpendicular direction is the left-normal direction when looking
along the positive parallel direction:

```text
normal = (normal_x2, normal_x3)
       = (-tangent_x3, tangent_x2)
```

The velocity projections are:

```text
u_vertical      = vel1
u_parallel      = tangent_x2 * vel2 + tangent_x3 * vel3
u_perpendicular = normal_x2  * vel2 + normal_x3  * vel3
```

Therefore:

- Positive `u_parallel` flows from the minimum-path location toward the
  maximum-path location.
- Negative `u_parallel` flows in the opposite direction.
- Positive `u_perpendicular` flows toward the left side of the section when
  facing from minimum path toward maximum path.
- Negative `u_perpendicular` flows toward the right side.
- Positive `u_vertical` flows upward.

The figure does not plot geometric height on its vertical axis. It plots
pressure on an inverted logarithmic axis, so upward motion corresponds
visually to decreasing pressure. The black velocity contours in the current
figure show `u_perpendicular`: positive values are solid and negative values
are dashed.

Projection is performed for each snapshot before averaging. The cache also
stores `mean(rho * u_parallel)` and `mean(rho * vel1)`, preserving
density-velocity covariance.

The in-section mass streamfunction is the least-squares solution of:

```text
d(psi)/dz = mean(rho * u_parallel)
-d(psi)/ds = mean(rho * vel1)
```

This finds the nondivergent mass circulation that best fits both in-section
mass-flux components. Streamfunction units are `kg m^-1 s^-1`; its arbitrary
constant is removed before contouring.

The plot uses:

- Black solid contours: positive perpendicular velocity at
  `[4, 8, 12, 16, 20] m s^-1`.
- Black dashed contours: negative perpendicular velocity at
  `[-20, -16, -12, -8, -4] m s^-1`.
- Optional black dashed contours: mass streamfunction.
- Thin blue contour: H2O cloud threshold.

Every perpendicular-velocity contour is labeled. Its overlay is enabled
by default and can be explicitly enabled or disabled with
`--show-perp-velocity` or `--no-show-perp-velocity`.

Streamfunction uses seven symmetric levels based on the 98th percentile of
the displayed absolute values. Streamfunction calculation and caching remain
enabled, but its overlay is disabled by default. Restore the dashed
streamfunction overlay with:

```bash
python scripts/plot_h2o_min_max_path_cross_section.py --show-streamfunction
```

## Current Plot Settings

The main settings are near the top of:

```text
scripts/plot_h2o_min_max_path_cross_section.py
```

Current constants:

```python
MAX_PRESSURE_BAR = 50.0
VAPOR_FRACTIONAL_DEVIATION_LIMIT = 0.5
VAPOR_FRACTIONAL_WHITE_HALF_WIDTH = 0.05
```

The vertical axis is logarithmic pressure, decreases upward, and stops at
`50 bar`.

## Vapor Anomaly Colormap

At every vertical level, the script calculates fractional deviation from the
horizontal mean along the periodic section:

```python
vapor_horizontal_mean = np.nanmean(fields["vapor"], axis=1, keepdims=True)
fields["vapor"] = (fields["vapor"] - vapor_horizontal_mean) / vapor_horizontal_mean
```

The current vapor levels are 11 linearly spaced values from `-0.5` to `0.5`:

```python
vapor_levels = np.linspace(
    -VAPOR_FRACTIONAL_DEVIATION_LIMIT,
    VAPOR_FRACTIONAL_DEVIATION_LIMIT,
    11,
)
```

The plotted vapor field is:

```text
(H2O - horizontal_mean(H2O)) / horizontal_mean(H2O)
```

The cache also stores the time-mean sampled `NH3` vapor cross-section as
`nh3_vapor_cross_section`, but the plot does not render it.

The base diverging colormap is `PiYG`. Values between `-0.05` and `0.05`
use a flat white plateau:

```python
vapor_cmap = diverging_with_white_plateau(
    "PiYG",
    -VAPOR_FRACTIONAL_DEVIATION_LIMIT,
    VAPOR_FRACTIONAL_DEVIATION_LIMIT,
    VAPOR_FRACTIONAL_WHITE_HALF_WIDTH,
)
```

To change the fractional-deviation range, edit
`VAPOR_FRACTIONAL_DEVIATION_LIMIT`. To widen or narrow the white region, edit
`VAPOR_FRACTIONAL_WHITE_HALF_WIDTH`.

The reusable white-plateau colormap helper is:

```text
scripts/custom_colormaps.py
```

It can be used with another Matplotlib diverging colormap:

```python
custom = diverging_with_white_plateau(
    "RdBu_r",
    vmin=-0.01,
    vmax=0.01,
    white_half_width=0.002,
)
```

## Cloud Filled Contours

The cloud overlay currently uses five logarithmically spaced levels from
`1e-5` through `1e-3`:

```python
cloud_levels = np.geomspace(1.0e-5, 1.0e-3, 5)
```

The filled contours use `Blues`:

```python
cloud_contour = ax.contourf(
    ...,
    levels=cloud_levels,
    cmap="Blues",
    extend="max",
)
```

Values below `1e-5` are masked so they do not cover the vapor fractional
deviation:

```python
np.ma.masked_less(fields["cloud"], cloud_levels[0])
```

A thin black contour marks the lower cloud threshold:

```python
ax.contour(
    ...,
    levels=[1.0e-5],
    colors="black",
    linewidths=0.8,
)
```

When changing the cloud lower limit, update both `cloud_levels` and the black
contour level.

## Colorbars and Layout

The two vertical colorbars share one right-side column, with vapor on top and
cloud below. Their axes are created with the figure grid:

```python
grid = fig.add_gridspec(
    2,
    2,
    width_ratios=(1.0, 0.035),
    height_ratios=(1.0, 1.0),
    hspace=0.04,
)
vapor_colorbar_ax = fig.add_subplot(grid[0, 1])
cloud_colorbar_ax = fig.add_subplot(grid[1, 1])

vapor_colorbar = fig.colorbar(
    vapor_contour,
    cax=vapor_colorbar_ax,
    orientation="vertical",
)
```

- Change the second `width_ratios` value to adjust colorbar thickness.
- Change `hspace` to adjust vertical spacing between colorbars.

The lower cloud colorbar uses its dedicated axis:

```python
cloud_colorbar = fig.colorbar(
    cloud_contour,
    cax=cloud_colorbar_ax,
    orientation="vertical",
)
```

## Other Useful Options

Plot a different number of snapshots:

```bash
python scripts/plot_h2o_min_max_path_cross_section.py --last 40
```

Write output and cache files elsewhere:

```bash
python scripts/plot_h2o_min_max_path_cross_section.py \
  --output-dir diagnostics/experiment_a
```

Select a different case:

```bash
python scripts/plot_h2o_min_max_path_cross_section.py \
  --case-regex 'jup_crm3d_H2O-NH3-H2S_F10_nu0\.1'
```

The case regex must match exactly one case directory.
