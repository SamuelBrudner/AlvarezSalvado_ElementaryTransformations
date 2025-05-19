# AlvarezSalvado_ElementaryTransformations

This repository provides the MATLAB implementation of the navigation model described in Álvarez-Salvado et al., "Elementary sensory-motor transformations underlying olfactory navigation in walking fruit flies" (eLife 2018, http://dx.doi.org/10.7554/eLife.37815).

## Overview

The code simulates walking fruit flies navigating in different odor plumes. Several built-in plume environments are provided, including:
- **Crimaldi** – a real plume from the Crimaldi laboratory dataset.
- **gaussian** – a static 2‑D Gaussian concentration profile without wind.
- **openlooppulse** – a single square odor pulse with constant wind.
- **openlooppulsewb** – the same pulse without wind.
- **openlooppulse15** and **openlooppulsewb15** – low-frequency versions of the above.

Model parameters are defined in `Code/navigation_model_vec.m`. Data import functions for analyzing experimental trials are located in `Code/import functions feb2017`.

## Requirements

- MATLAB (tested on R2017b or later).
- For the Crimaldi plume environment, download the file `10302017_10cms_bounded.hdf5` and place it in the repository root.

## Running Simulations

Add the `Code` directory to your MATLAB path and call `navigation_model_vec`:

```matlab
triallength = 3500;       % number of timesteps
environment = 'Crimaldi'; % choose plume type
plotting = 1;             % 1 enables plotting
ntrials = 10;             % number of simulated flies

result = navigation_model_vec(triallength, environment, plotting, ntrials);
```

For parameter sweeps, use `runmodel.m` to systematically vary model parameters across trials.

## Repository Layout

```
Code/                            MATLAB scripts for simulations and analysis
   navigation_model_vec.m        Main navigation model
   runmodel.m                    Run batches of simulations
   import functions feb2017/     Utilities to load raw experimental data
Arena 4f/                        LabVIEW files for behavioral assays
FlyTracker 3.6.vi                Data acquisition software
```

## Citation

If you use this code in your research, please cite:

Álvarez-Salvado et al. "Elementary sensory-motor transformations underlying olfactory navigation in walking fruit flies." eLife, 2018. http://dx.doi.org/10.7554/eLife.37815

