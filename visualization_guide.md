# Visualization Guide

This document lists the visualization scripts included in the repository and how to execute them. All MATLAB scripts should be run from the repository root with MATLAB on your `PATH`. Python scripts require the development conda environment.

## Navigation Results

### `plot_results.m`
Generates a set of figures from a navigation results file.

Run inside MATLAB:
```matlab
plot_results               % analyzes results/nav_results_0000.mat
plot_results('results/FILE.mat')
```
The script produces trajectory, performance and summary PNG files in the same directory as the input file.

### `run_plot_results.sh`
Shell wrapper that calls `plot_results.m` non‑interactively.
```bash
./run_plot_results.sh results/nav_results_0000.mat
```
Four PNG figures are written to the `results/` directory.

### `analyze_results.m`
Loads `results/nav_results_0000.mat`, creates multiple figures and writes a text summary.
```matlab
run('analyze_results.m')
```
Figures open in MATLAB and a `_summary.txt` file is saved next to the results.

### `view_results.py`
Python version of the analysis tool. Requires numpy, scipy and matplotlib.
```bash
python view_results.py results/nav_results_0000.mat
```
Saves a `<file>_plots.png` image and prints basic metrics to the console.

## Plume Visualizations

### `plot_both_plumes.m`
Displays Crimaldi and Smoke plumes side by side with model initialization zones.
```matlab
run('plot_both_plumes.m')
```
Creates `both_plumes_comparison.png` in the project root.

### `plot_init_with_plumes.m`
Overlays the chosen initialization region on each plume using current configuration files.
```matlab
run('plot_init_with_plumes.m')
```
Outputs `results/init_zones_with_plumes.png`.

### `visualize_smoke_plume.m`
Generated by `setup_smoke_plume_config.sh` when configuring a smoke plume. It plots sample frames from the configured dataset.
```matlab
run('visualize_smoke_plume.m')
```

### `quick_dimension_check.m`
Reports the dimensions of plume HDF5 files and verifies orientation.
```matlab
run('quick_dimension_check.m')
```
Useful for sanity‑checking raw plume data.

## Diagnostics

### `plot_odor_distance_diagnostic.m`
Produces odor intensity and distance‑to‑source plots for test data created by `test_both_plumes_complete.m`.
```matlab
plot_odor_distance_diagnostic   % default: 5 agents
plot_odor_distance_diagnostic(10)
```
Creates several PNG files in the `results/` folder summarizing odor approach behavior.

### `clean_quick_test.m` and `quick_test_both_plumes.m`
Both scripts perform short validation runs on the two plumes and display the resulting trajectories.
Run either script in MATLAB to visualize quick test results.

