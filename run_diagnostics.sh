#! /bin/bash

./scripts/plot_horizontal_mean_profiles.py output_260513 --field out1 --vars H2O NH3 H2S --last 20 --xscale log --xlabel "Mole fraction"
./scripts/plot_time_evolution.py output_260513 --field out1 --vars H2O NH3 H2S --last 200 --yscale log --ylabel "Mole fraction"
./scripts/plot_pressure_level_contours.py output_260513 --field out1 --var H2O --id 01000
./scripts/plot_pressure_level_contours.py output_260513 --field out1 --var theta --id 01000
./scripts/plot_pressure_level_contours.py output_260513 --field out1 --var vel1p --id 01000
./scripts/plot_pressure_level_contours.py output_260513 --field out1 --var vel1m --id 01000
./scripts/plot_pressure_level_spectra.py output_260513 --field out1 --var ke --id 01000
./scripts/plot_vertical_tendency_profiles.py output_260513_dry --field out1 --vars temp theta --last 20 --xlabel "Tendency [K/day]"
