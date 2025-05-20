# AlvarezSalvado_ElementaryTransformations

This repository provides the MATLAB implementation of the navigation model described in Álvarez-Salvado et al., "Elementary sensory-motor transformations underlying olfactory navigation in walking fruit flies" (eLife 2018, http://dx.doi.org/10.7554/eLife.37815).

## Overview

The code simulates walking fruit flies navigating in different odor plumes. Several built-in plume environments are provided, including:
- **Crimaldi** – a real plume from the Crimaldi laboratory dataset.
- **gaussian** – a static 2‑D Gaussian concentration profile without wind.
- **openlooppulse** – a single square odor pulse with constant wind.
- **openlooppulsewb** – the same pulse without wind.
- **openlooppulse15** and **openlooppulsewb15** – low-frequency versions of the above.
- **video** – load a custom plume movie using `load_plume_video.m`.

Model parameters are defined in `Code/navigation_model_vec.m`. Data import functions for analyzing experimental trials are located in `Code/import functions feb2017`.

## Requirements

- MATLAB (tested on R2017b or later).
- For the Crimaldi plume environment, download the file `10302017_10cms_bounded.hdf5` and place it in the `data/` directory.

## Data Directory

All plume data files should be stored inside the `data/` directory located at the
repository root. Custom plume movies or processed MATLAB files can also be kept
in this folder so that configuration files can reference them with a relative
path.

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

### Loading parameters from a configuration file

You can also store simulation parameters in a YAML file and load them in MATLAB
using the `load_config` helper:

```matlab
cfg = load_config(fullfile('tests', 'sample_config.yaml'));
result = run_navigation_cfg(cfg);
```

### Using custom plume videos

To run the model with your own plume movies, convert the `.avi` file to a
MATLAB structure using `load_plume_video.m`:

```matlab
plume = load_plume_video('my_plume.avi', 20, 40); % 20 px/mm, 40 Hz
triallength = size(plume.data, 3);
result = navigation_model_vec(triallength, 'video', 1, 1, plume);
```

The spatial scale (pixels per millimeter) and frame rate are supplied when
loading the movie so that the simulation can handle different resolutions and
durations.

### Loading simulation parameters from YAML

Common simulation options can be stored in a YAML configuration file and loaded
with `load_config.m` or passed directly to `run_navigation_cfg`:

```matlab
cfg = load_config('tests/sample_config.yaml');
result = run_navigation_cfg(cfg);
```

### Running the bilateral model

To include bilateral odor sensing, set the `bilateral` flag in your
configuration and invoke `run_navigation_cfg`:

```matlab
cfg = load_config('tests/sample_config_bilateral.json');
result = run_navigation_cfg(cfg);
```



## Repository Layout

```
Code/                            MATLAB scripts for simulations and analysis
   navigation_model_vec.m        Main navigation model
   run_navigation_cfg.m          Wrapper for running simulations from configs
   runmodel.m                    Run batches of simulations
   load_plume_video.m            Convert .avi movies for custom plumes

   import functions feb2017/     Utilities to load raw experimental data
Arena 4f/                        LabVIEW files for behavioral assays
FlyTracker 3.6.vi                Data acquisition software
```

## Citation

If you use this code in your research, please cite:

Álvarez-Salvado et al. "Elementary sensory-motor transformations underlying olfactory navigation in walking fruit flies." eLife, 2018. http://dx.doi.org/10.7554/eLife.37815

