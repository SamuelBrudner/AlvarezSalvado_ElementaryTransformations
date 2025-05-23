# AlvarezSalvado_ElementaryTransformations

This repository provides the MATLAB implementation of the navigation model described in Álvarez-Salvado et al., "Elementary sensory-motor transformations underlying olfactory navigation in walking fruit flies" (eLife 2018, http://dx.doi.org/10.7554/eLife.37815).

## Quick Start

Run a default simulation and export the results in just a few commands:

```matlab
% Load the default configuration
cfg = load_config('tests/sample_config.yaml');

% Run the simulation
result = run_navigation_cfg(cfg);

% Export results to CSV and JSON
export_results('data/raw/gaussian_bilateral/1_1/result.mat', 'data/processed');
```

This will:
1. Run a simulation with default parameters
2. Save the raw results in the standard directory structure
3. Export processed data to `data/processed/` in open formats

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

- MATLAB (tested on **R2021a or later**).
- For the Crimaldi plume environment, download the file `10302017_10cms_bounded.hdf5` and place it in the `data/` directory.

### Development environment

If you plan to run the Python tests or scripts, create a local conda environment:

```bash
source setup_env.sh --dev
conda activate .env
```

This installs the required packages and activates an environment named `.env`.

## Data Organization

### Directory Structure

- `data/raw/` - Store raw plume files here
- `data/processed/` - Store processed outputs here
- `logs/` - Log files for each batch job
- `data/raw/<plume>_<sensing>/<agentIndex>_<seed>/` - Simulation results (auto-created)
  - `config_used.yaml` - Exact configuration used for the run
  - `result.mat` - Simulation output (see below)

### Environment Variables

The batch script `run_batch_job.sh` uses two environment variables for file
management:

- `PLUME_CONFIG` – path to the YAML configuration describing the plume and
  simulation defaults (defaults to `configs/my_complex_plume_config.yaml`).
- `OUTPUT_BASE` – base directory for raw simulation outputs (defaults to
  `data/raw`).

Override these when submitting jobs to customize where configuration is loaded
from and where results are written.

### Output Data

#### In-Memory Structure
Every model call returns a structure with these fields:

| Field | Size | Description |
|-------|------|-------------|
| `x`, `y` | T × N | Arena coordinates (cm) for each timestep T and agent N |
| `theta` | T × N | Heading angle (°; 0 = down-wind, +90 = up-wind) |
| `odor` | T × N | Raw odor concentration (0–1) at agent's centroid |
| `ON`, `OFF` | T × N | Temporal filters of the odor signal |
| `turn` | T × N | Boolean: 1 = stochastic turn executed that frame |
| `start` | N × 2 | Initial x,y positions for each agent |
| `successrate` | 1×1 | Fraction of agents that reached the 2 cm goal (Crimaldi/Gaussian only) |
| `latency` | 1 × N | Time to goal for successful agents (s) |
| `params` | struct | Snapshot of all model parameters |

**Tip:** Save results with `save('result.mat','out','-v7.3')` for portable HDF5 format.

### Naming Conventions

- `<plume>`: Name of the plume configuration (from `PLUME_CONFIG`)
- `<sensing>`: `bilateral` or `unilateral` (from `cfg.bilateral`)
- `<agentIndex>`: 1-based index for agent grouping
- `<seed>`: PRNG seed for reproducible simulations

### Example Usage

```matlab
% Load a single agent result
r = load('data/raw/gaussian_bilateral/17_17/result.mat');
x = r.out.x;          % trajectories
pars = r.out.params;  % model parameters

% Aggregate success rates across multiple runs
files = dir('data/raw/*/*/result.mat');
succ = arrayfun(@(f) load(fullfile(f.folder,f.name),'out'), files);
success_rate = mean(arrayfun(@(s) s.out.successrate, succ));
```

## Data Dictionary

### Trajectory Data (trajectories.csv)

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| t | int | step | Time step (0-indexed) |
| trial | int | - | Trial/agent number (0-indexed) |
| x | float | cm | X-coordinate in arena (0 = left edge) |
| y | float | cm | Y-coordinate in arena (0 = bottom edge) |
| theta | float | ° | Heading angle (0 = upwind, 90° = to the right) |
| odor | float | 0-1 | Normalized odor concentration at agent's position |
| ON | float | - | ON channel temporal filter output |
| OFF | float | - | OFF channel temporal filter output |
| turn | bool | - | Whether a turn was executed this timestep |

### Parameters (params.json)

Contains all model parameters used in the simulation, including:
- `bilateral`: Whether bilateral sensing is enabled
- `tau_ON`, `tau_OFF`: Time constants for ON/OFF filters (s)
- `v0`: Base walking speed (cm/s)
- `R0`: Base turning rate (rad/s)
- And all other model parameters

### Summary (summary.json)

- `successrate`: Fraction of agents that reached the goal (0-1)
- `latency`: Array of goal-reaching times for successful agents (s)
- `n_trials`: Total number of agents/trials
- `timesteps`: Duration of simulation (time steps)

## Exporting Results to Open Formats

The `export_results.m` function converts MATLAB `.mat` output files into more accessible formats (CSV/JSON) for analysis in other tools or for long-term storage.

### Features

- Converts binary MATLAB files to standard CSV and JSON formats
- Preserves all simulation metadata in human-readable JSON
- Handles both single and multi-trial simulations
- Provides flexible output format options
- Maintains data relationships and structure

### Usage

```matlab
% Basic usage (exports both CSV and JSON)
export_results('result.mat', 'output_dir');

% Export only CSV
export_results('result.mat', 'output_dir', 'Format', 'csv');

% Export only JSON
export_results('result.mat', 'output_dir', 'Format', 'json');
```

### Output Files

- `trajectories.csv` - Time series data with columns: t, trial, x, y, theta, odor, ON, OFF, turn
- `params.json` - Complete model parameters used in the simulation
- `summary.json` - High-level statistics (success rate, latency, etc.)

### Example: Loading the Data

```matlab
% Load trajectory data
trajectories = readtable('output_dir/trajectories.csv');

% Load parameters
fid = fopen('output_dir/params.json');
raw = fread(fid, inf, 'uint8=>char')';
fclose(fid);
params = jsondecode(raw);

% Display summary
fprintf('Success rate: %.2f\n', params.successrate);
fprintf('Number of trials: %d\n', max(trajectories.trial) + 1);
```

### Notes

- The function automatically creates the output directory if it doesn't exist
- Timesteps are 0-indexed in the output files
- Boolean values are converted to 0/1 in CSV output

## Running Simulations

When MATLAB starts in this repository it automatically executes `startup.m`,
which adds the `Code` directory to your path. You can then call
`navigation_model_vec` directly:

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
triallength = 3500; % can exceed movie length to loop the plume
result = navigation_model_vec(triallength, 'video', 1, 1, plume);
```

When `triallength` exceeds the number of frames in the plume movie, the odor
frames automatically repeat so that the simulation continues for the full
duration. By default a wind speed of one unit is used for video plumes so that
upwind and downwind responses match the Crimaldi environment. Set the optional
`ws` parameter in the configuration to override this value.

The spatial scale (pixels per millimeter) and frame rate are supplied when
loading the movie so that the simulation can handle different resolutions and
durations.

Alternatively, if you have a metadata YAML file describing the processed
video, you can load the plume directly using `load_custom_plume` and run the
simulation via `run_navigation_cfg`:

```matlab
plume = load_custom_plume('my_plume_metadata.yaml');
cfg.plume_metadata = 'my_plume_metadata.yaml';
cfg.plotting = 0;
cfg.ntrials = 1;
result = run_navigation_cfg(cfg);
```

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
The same options are available in YAML format:

```matlab
cfg = load_config('tests/sample_config_bilateral.yaml');
result = run_navigation_cfg(cfg);
```

The batch script `run_batch_job.sh` also accepts this `bilateral` flag when
creating configuration structures for large simulation runs.


> **Important**: `run_batch_job.sh` contains a `--mail-user` directive for
> SLURM job notifications. Replace the placeholder email address with your own,
> or comment out the line to disable email alerts before submitting jobs.

### `run_batch_job_4000.sh`

`run_batch_job_4000.sh` drops environment variable expansion from its `SBATCH`
headers. Specify job parameters when launching the job:

```bash
sbatch --job-name=${EXPERIMENT_NAME}_sim \
       --array=0-$((TOTAL_JOBS-1))%${SLURM_ARRAY_CONCURRENT} \
       run_batch_job_4000.sh
```

## Analysis Configuration

All analysis scripts use a shared YAML file located at
`configs/analysis_config.yaml`. Load it in Python with:

```python
from Code.load_analysis_config import load_analysis_config
cfg = load_analysis_config('configs/analysis_config.yaml')
```

The configuration file includes a `data_loading_options` section that controls
which files are loaded for each discovered run. For example:

```yaml
data_loading_options:
  load_summary_json: true
  load_trajectories_csv: false
  load_params_json: false
  load_config_used_yaml: true
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

