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

- **MATLAB** (tested on **R2021a or later**).
- **Conda** (Miniconda or Anaconda) for Python environment management.
- For the Crimaldi plume environment, download the file `10302017_10cms_bounded.hdf5` and place it in the `data/` directory.

### Python Dependencies

Python dependencies are managed using Conda and are split into two files:

1. `environment.yml` - Core dependencies required for running the project
2. `dev-environment.yml` - Additional development tools and dependencies (only needed for development)

## Development Environment Setup

1. **Set up the base environment**:
   ```bash
   # Create and activate the base environment
   conda env create -f environment.yml
   conda activate elementary-transformations
   ```

2. **For development, set up the development environment**:
   ```bash
   # Install development tools and dependencies
   conda env update -f dev-environment.yml
   ```

3. **Alternative: Use the setup script**:
  ```bash
  # This will create a local environment in ./dev-env
  ./setup_env.sh --dev
  # Compatible with both new and older Conda versions
  # Skip lock generation with --skip-conda-lock if conda-lock is unavailable
  ```
   This prepares the development environment. Run scripts located in the
   `Code` package with the module syntax so the repository root is added to
   `sys.path`:
   ```bash
   python -m Code.some_script
   ```

4. **Activate the development environment**:
   ```bash
   conda activate ./dev-env
   ```

5. **Install pre-commit hooks (optional but recommended)**:
   If you ran the setup script with the `--dev` flag, hooks were installed
   automatically. Otherwise, you can install them manually:
   ```bash
   pre-commit install
   ```

### Running Tests

To run the test suite:

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_example.py

# Run tests with more verbose output
pytest -v
```

### Development Workflow

1. Make your changes to the code
2. Run tests to ensure nothing is broken
3. The pre-commit hooks will automatically format and check your code when you commit
4. Create a pull request with your changes

## Configuration

### Paths Configuration

The project uses a centralized paths configuration file (`configs/paths.yaml`) that is automatically generated during setup. This file contains paths to important resources like data files and output directories.

Key paths:
- `crimaldi_hdf5`: Path to the Crimaldi HDF5 data file
- `output.raw`: Directory for raw simulation outputs
- `output.processed`: Directory for processed data
- `output.figures`: Directory for generated figures

To customize paths:
1. Edit `configs/paths.yaml` after running the setup script
2. Use environment variables in the template (e.g., `${PROJECT_DIR}/data/file.h5`)

### Development Environment

#### Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to enforce code quality standards. The configuration is automatically generated during environment setup to work with the local conda environment.

Key features:
- **Automatic Setup**: Hooks are configured automatically when you run `./setup_env.sh --dev`
  (compatible with both new and old Conda versions)
- **Portable**: Configuration works across different machines and platforms
- **Consistent**: Uses the project's conda environment for all tools
- **Local Configuration**: The `.pre-commit-config.yaml` file is generated from
  `.pre-commit-config.yaml.template` during environment setup and is ignored by
  Git

Available hooks include:
- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Linting and code style
- **Mypy**: Static type checking (runs manually)
- **Pytest**: Test runner (runs manually)
- **Interrogate**: Docstring coverage (runs manually)

To run all pre-commit checks manually:
```bash
conda run -p ./dev-env pre-commit run --all-files
```

#### Setting up the environment

Create and set up the local Conda environment:

```bash
# Run the setup script
./setup_env.sh --dev
# Compatible with new and old Conda releases

# For interactive shell usage
conda activate ./dev-env

# For scripts and non-interactive usage (recommended for batch jobs/CI)
conda run -p ./dev-env your_script.py
```

#### Running tests

After setup, you can run tests using:

```bash
# Run all tests
conda run -p ./dev-env pytest tests/

# Run a specific test file
conda run -p ./dev-env pytest tests/test_module.py
```

#### Running batch jobs

For batch processing, use the `conda run -p` approach in your scripts:

```bash
#!/bin/bash
conda run -p ./dev-env python process_data.py --input data/input --output results/
```

The script creates a local environment in `./dev-env`. Rather than activating
it globally, prefix all Python commands with `conda run --prefix ./dev-env` to
ensure the correct interpreter is used. For example:

```bash
conda run --prefix ./dev-env pytest -q
```

`setup_env.sh` installs the required packages and sets up pre-commit so
formatting and tests run automatically before each commit.
It detects whether your Conda version supports `--force` and works on
both modern and older releases.

### Developer Workflow

Typical steps when contributing:

```bash
# Run formatters and linters
conda run --prefix ./dev-env pre-commit run --all-files

# Run the Python test suite
conda run --prefix ./dev-env pytest -q
```

Use MATLAB's Code Analyzer and unit testing framework in parallel for MATLAB
changes. Ensure all tests pass before committing.

### Maintenance

- Update pre-commit hooks periodically with:

  ```bash
  conda run --prefix ./dev-env pre-commit autoupdate
  ```

- Regenerate `conda-lock.yml` whenever `environment.yml` changes to keep
  dependencies pinned.


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

`load_analysis_config` uses `yaml.safe_load` under the hood, so the
configuration must be valid YAML rather than JSON.

The configuration file includes a `data_loading_options` section that controls
which files are loaded for each discovered run. For example:

```yaml
data_loading_options:
  load_summary_json: true
  load_trajectories_csv: false
  load_params_json: false
  load_config_used_yaml: true
```

### Running the Python analysis pipeline

Once processed simulation results are available you can execute the full
analysis workflow with:

```bash
python Code/main_analysis.py configs/analysis_config.yaml
```

This script generates any requested tables and plots, then performs the
statistical tests defined in the configuration. All tables and the
resulting p-values are written to the directory specified by
`output_paths.tables`.

To compare the built‑in Crimaldi plume with a custom plume, include a
`statistical_analysis` block in your YAML:

```yaml
statistical_analysis:
  - test_type: t_test_ind
    metric_name: success_rate
    grouping_variable: plume_type
    groups_to_compare:
      - crimaldi
      - custom_video
```

## Plume Intensity Utilities

Two Python helper scripts simplify working with plume intensity data. See
[docs/intensity_comparison.md](docs/intensity_comparison.md) for a concise
overview of the workflow and example commands.

### Characterize plume intensities

`Code/characterize_plume_intensities.py` computes basic statistics for a plume
and stores them in a JSON file. Use it for Crimaldi HDF5 files or for custom
video plumes processed via MATLAB.

```bash
# Crimaldi plume example
python -m Code.characterize_plume_intensities \
    --plume_type crimaldi \
    --file_path data/10302017_10cms_bounded.hdf5 \
    --plume_id crimaldi \
    --output_json plume_stats.json

# Video plume example

First create a small MATLAB script to load your movie and write out the
intensity vector. For ``data/smoke_1a_bgsub_raw.avi`` the script might look like
``video_script.m``:

```matlab
plume = load_plume_video('data/smoke_1a_bgsub_raw.avi', 20, 40);
all_intensities = plume.data(:);
save('temp_intensities.mat', 'all_intensities');
fprintf('TEMP_MAT_FILE_SUCCESS:%s\n', which('temp_intensities.mat'));
```

Run the utility with the path to this script:

```bash
python -m Code.characterize_plume_intensities \
    --plume_type video \
    --file_path path/to/video_script.m \
    --plume_id my_video \
    --px_per_mm 20 \
    --frame_rate 40 \
    --matlab_exec /path/to/matlab \
    --output_json plume_stats.json
```

``px_per_mm`` and ``frame_rate`` are inserted into the temporary MATLAB script
before execution so that your MATLAB code can access these values as workspace
variables.
```

### Compare intensity statistics

`Code/compare_intensity_stats.py` reads intensity vectors from one or more HDF5 files and prints a table of summary statistics or writes them to CSV. It also accepts MATLAB scripts that output a MAT-file of intensities, enabling comparisons between video plumes and the Crimaldi dataset.

```bash
# Display results in the terminal
python -m Code.compare_intensity_stats A data/crimaldi.hdf5 B data/custom.hdf5
    --matlab_exec /path/to/matlab

# Write to CSV
python -m Code.compare_intensity_stats A data/crimaldi.hdf5 B data/custom.hdf5 \
    --csv intensity_comparison.csv
    --matlab_exec /path/to/matlab
```


To compare a custom video plume against Crimaldi, first create the development environment with `./setup_env.sh --dev` (compatible with old and new Conda) and then run:

```bash
conda run --prefix ./dev-env python -m Code.compare_intensity_stats VID video path/to/video_script.m CRIM crimaldi data/10302017_10cms_bounded.hdf5 --matlab_exec /path/to/matlab
```

``path/to/video_script.m`` should point to the MATLAB file shown above that
loads your ``.avi`` movie and writes ``temp_intensities.mat``.

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

If you use this code in your research, please cite the accompanying paper and
refer to the metadata files in the repository root. `CITATION.cff` provides a
machine-readable citation for the software, and `codemeta.json` contains rich
metadata for automated tools.

Álvarez-Salvado et al. "Elementary sensory-motor transformations underlying
olfactory navigation in walking fruit flies." eLife, 2018.
<http://dx.doi.org/10.7554/eLife.37815>

## Roadmap

Future work includes improving the Python analysis tools and expanding the test
coverage for MATLAB functions. Contributions are welcome.

