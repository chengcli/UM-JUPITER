Moist equatorial Jupiter beta plane
===================================

``run_jupiter_beta.py`` adapts the Uranus radiation driver to a Cartesian
beta plane. ``jupiter_beta.yaml`` is the production configuration;
``jupiter_beta_smoke.yaml`` is a short, coarse CPU test. Existing Jupiter
cases and drivers are unchanged.

Domain and rotation
-------------------

The effective planetary radius is 35,746 km, half Jupiter's nominal
equatorial radius of 71,492 km (`NASA fact sheet
<https://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html>`_).
Rotation retains the existing Jupiter case's 1.7585e-4 rad/s.
With x1 vertical, x2 east, and x3 north, the scripted Coriolis forcing uses

.. math::

   f(y)=\beta y,\qquad \beta=\frac{2\Omega}{R}
       =9.83886309\times10^{-12}\ \mathrm{m^{-1}\,s^{-1}}.

The equator is x3=0. The meridional interval is -R*pi/6 to R*pi/6,
representing 30 S to 30 N in the beta-plane coordinate mapping. This is a
linear Coriolis approximation, not spherical geometry.

* Vertical: 300 km, 100 cells, dz=3 km.
* North-south: 37,433.124 km, 200 cells.
* East-west: 74,866.247 km, 400 cells.
* dx=dy=187.166 km; dx/dz=62.389. Total: 8 million active cells.

The final 400-cell zonal choice makes the zonal extent twice the meridional
extent while preserving equal horizontal spacing. Zonal boundaries are
periodic. Meridional and vertical boundaries are reflecting. The default
slab layout has two zonal blocks, one per process, with no vertical split.
CRM integration uses ``cfl: 0.9`` and ``implicit-scheme: 1``.

Moist physics and radiation
---------------------------

Species, reactions, sedimentation, bottom composition replenishment, and
initial thermodynamic parameters come from ``jup_gcm_H2O-NH3_F100.yaml``:
H2O/NH3 vapor, cloud, and precipitation; Ps=100 bar, Ts=610 K, Tmin=110 K;
xH2O=0.003 and xNH3=0.00032; gravity=22 m/s^2. The obsolete ``tracer-floor``
key is omitted because the current Snapy parser does not accept it. The
thermodynamic iteration limit is increased from 20 to 100 to converge the
moist initial profile; the existing tolerance is unchanged.

Internal heating supplies 100 W/m^2 over the lowest two cells, matching the
F100 experiment's forcing scale rather than an observational estimate.
The old bottom-temperature thermostat is removed so it does not add an
uncontrolled thermal source. Bottom composition relaxation remains.
Prescribed top cooling is replaced by shortwave/longwave radiative transfer.
The native constant Coriolis forcing is replaced by ``BetaPlaneCoriolis``.

``RTStateCartesian`` uses the Jupiter J2000 orbit at longitude=latitude=0
(radians) and broadcasts one incidence angle over all columns. The actual
beam changes with orbital distance and day/night rotation; spatial
latitude-dependent sunlight is not modeled in this configuration. Each
column retains its own opacity and temperature. Radiation updates every
600 s, with heating held fixed between solves. Time is seconds since J2000.

The gas and cloud grey opacity coefficients are inherited from the Uranus
example, with the species changed to H2O/NH3 and their molecular weights
read from Kintera. They are illustrative, not calibrated Jupiter spectra.
The launcher exports rank-specific TorchScript opacity, orbital-heating,
and Coriolis files under the output directory's ``models/rankN`` folders.

Output and execution
---------------------

Daily hydro slices at 75, 150, and 225 km appear in ``out1``, ``out3``, and
``out4``. ``out2`` contains temperature, total non-dry mass fraction,
volumetric radiative heating, CCF ``net_flux`` in W/m^2, and separate species
paths in kg/m^2. Restarts are written every ten days and at termination.
The configured target is one Earth year (31,557,600 s).

From UM-JUPITER, a short CPU check is::

   DEVICE=cpu OMP_NUM_THREADS=1 python run_jupiter_beta.py \
       --config jupiter_beta_smoke.yaml --output-dir /tmp/jupiter-beta-smoke

A short full-resolution, two-GPU check is::

   bash run_jupiter_beta.sh /path/to/check --max-cycles 2

For a production run, omit ``--max-cycles``. The default production output
directory is ``/data01/2026.JUPITER_CRM/jup_beta_H2O-NH3_F100_Rhalf``. To continue a saved case::

   bash run_jupiter_beta.sh /path/to/continued \
       --restart /path/to/check/jupiter_beta.final.restart

The full grid is memory intensive; use the short check to measure the
available hardware before a long integration. The smoke configuration
preserves the physical bounds and dx=dy but intentionally coarsens the
vertical and horizontal grids by different factors.

Validation::

   OMP_NUM_THREADS=1 python -m pytest test_jupiter_beta.py -q -p no:cacheprovider

Tests cover the agreed geometry, retained moist parameters, nonzero solar
illumination, beta, conservative radiative heating, species paths, slice
values, and restart continuation. The case requires Paddle's
``RTStateCartesian`` and ``BetaPlaneCoriolis`` and Snapy's ``_flux`` suffix
metadata rules for CCF staggering and W/m^2 units.
