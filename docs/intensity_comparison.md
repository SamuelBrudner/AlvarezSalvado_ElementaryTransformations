# Intensity Dataset Comparison

This page describes how to characterise the intensity of individual odour plumes and how to compare multiple intensity datasets.

## Table of Contents

- [Step-by-Step](#step-by-step)
- [Initial Setup](#initial-setup)
- [Path Management](#path-management)
- [Characterising a Single Plume](#characterising-a-single-plume)
- [Comparing Multiple Datasets](#comparing-multiple-datasets)
- [Configuration](#configuration)
- [Notes](#notes)

## Step-by-Step

1. Run `./setup_env.sh --dev` to create `./dev_env`.
2. Source `./paths.sh` to generate `configs/project_paths.yaml` and detect MATLAB. The script uses this file and falls back to default paths when `yq` is missing.
3. Execute `conda run --prefix ./dev_env python -m Code.compare_intensity_stats`.

## Initial Setup

1. **Set up the development environment**:
   ```bash
   ./setup_env.sh --dev
   ```

2. **Update your paths**:
   The project uses a centralized path configuration. After setup, source the paths:
   ```bash
   source ./paths.sh
   ```
   This sets up all necessary environment variables and paths. If `matlab` isn't on your `PATH`,
   `paths.sh` attempts to load a module named `MATLAB/$MATLAB_VERSION` (or the
   value in `MATLAB_MODULE`). Set either variable before sourcing when working on
   HPC clusters. You can run this anytime to update paths without rebuilding the
   conda environment.

   When MATLAB still cannot be found, the script runs `module load
   MATLAB/$MATLAB_VERSION` automatically. The `MATLAB_VERSION` variable defaults
   to `2023b` but can be set to any available module version before sourcing
   `paths.sh`.

3. **Running scripts**:
   Use the module form when executing Python scripts to ensure the repository root is on `sys.path`:
   ```bash
   python -m Code.<script>
   ```

## Path Management

The project uses `project_paths.yaml` for managing file paths. Key features:

- **Automatic Path Resolution**: All paths are relative to the project root
- **Environment Variables**: Uses `PROJECT_DIR` and `TMPDIR`
- **Generated Files**: `project_paths.yaml` is automatically generated from the template

To regenerate paths configuration:
```bash
rm configs/project_paths.yaml && ./paths.sh
```

Key paths are available as environment variables after sourcing `paths.sh`:
- `$PROJECT_DIR`: Root directory of the project
- `$PYTHONPATH`: Includes the `Code` directory
- `$TMPDIR`: System temporary directory

## Characterising a Single Plume

To obtain intensity statistics for a single plume, use the `analyze_crimaldi_data.py` script. The command prints summary statistics such as the minimum, maximum and percentile values.

```bash
conda run --prefix ./dev_env python -m Code.analyze_crimaldi_data data/raw/plume1.hdf5
```

Expected output:

```
Min: 0.05
Max: 3.2
Mean: 1.4
Std: 0.8
1th percentile: 0.10
5th percentile: 0.20
95th percentile: 2.9
99th percentile: 3.1
```

## Comparing Multiple Datasets

Use the `compare_intensity_stats.py` script with multiple input files. The script computes cross‑dataset statistics and produces a plot showing the distribution of intensities.

```bash
conda run --prefix ./dev_env python -m Code.compare_intensity_stats data/raw/plume1.hdf5 data/raw/plume2.hdf5
```

To see the mean and median differences when exactly two datasets are provided, add the `--diff` option:

```bash
conda run --prefix ./dev_env python -m Code.compare_intensity_stats A data/raw/plume1.hdf5 B data/raw/plume2.hdf5 --diff
```

To save the computed statistics, provide a path via `--csv` or `--json`:

```bash
conda run --prefix ./dev_env python -m Code.compare_intensity_stats A data/raw/plume1.hdf5 B data/raw/plume2.hdf5 --json results/stats.json
```
The JSON file contains a list of objects with ``identifier`` and ``statistics`` keys for each plume.

Sample output:

```
identifier	mean	median	p95	p99	min	max	count
A	1.200	1.100	...
B	1.500	1.300	...
DIFF	-0.300	-0.200				
```

Typical output:

```
Plume1 peak: 3.2
Plume2 peak: 2.8
Difference (mean): 0.4
Figure saved to figures/intensity_comparison.png
```

### Quick Start Example

Follow these steps to run a simple comparison using the bundled datasets.

1. **Create the development environment**

   ```bash
   ./setup_env.sh --dev
   ```

2. **Generate local paths and detect MATLAB**

   ```bash
   source ./paths.sh
   ```

   This command writes `configs/project_paths.yaml` if it does not yet exist and sets the `$MATLAB_EXEC` variable.

3. **Run the comparison**

   ```bash
   conda run --prefix ./dev_env python -m Code.compare_intensity_stats \
       CRIM data/10302017_10cms_bounded_2.h5 \
       SMOKE process_smoke_video.m
   ```

   The smoke video is located at `data/smoke_1a_bgsub_raw.avi` and the Crimaldi HDF5 file at `data/10302017_10cms_bounded_2.h5`.


### Comparing a Video Plume to Crimaldi

To compare a video plume with Crimaldi data, you'll need to extract intensity values using MATLAB. The repository includes a pre-configured script `process_smoke_video.m` that handles this process with robust error checking and path handling.

#### Using the Pre-configured Script

The `process_smoke_video.m` script is designed to work with the smoke video data and includes these features:
- Automatic path resolution using `orig_script_dir`
- File existence checks for all inputs
- Proper temporary file handling with MATLAB's `tempdir()`
- Clear error messages for troubleshooting

#### Prerequisites

1. MATLAB must be installed and available in your system PATH
2. The following MATLAB toolboxes are required:
   - Image Processing Toolbox
   - Statistics and Machine Learning Toolbox
3. The repository must contain:
   - `configs/my_complex_plume_config.yaml` with `px_per_mm` and `frame_rate`
   - `data/smoke_1a_bgsub_raw.avi` (or update the path in the script)

#### Running the Comparison

Run the comparison using the development environment:

```bash
conda run --prefix ./dev_env python -m Code.compare_intensity_stats \
    CRIM data/10302017_10cms_bounded_2.h5 \
    SMOKE process_smoke_video.m
```

The MATLAB executable is auto-detected when you source `paths.sh`. Pass
`--matlab_exec` only if you need to override the detected path. When datasets
have different lengths the tool logs a warning but still computes statistics.

#### Frame Rate and Spatial Scale

`px_per_mm` sets how many pixels correspond to one millimetre and `frame_rate` defines the video sampling rate. Velocities are now scaled using these values so movement speeds remain in millimetres per second regardless of the original frame rate or resolution.

Example excerpt from `configs/my_complex_plume_config.yaml`:

```yaml
# Pixels per millimeter conversion factor
px_per_mm: 6.536
# Frame rate of the video in Hz
frame_rate: 60
```


### Pure Python Workflow

Video files can be processed without MATLAB using the `--pure-python` flag. The
helper function reads frames via `imageio` and stores the resulting intensities
in a NumPy ``.npy`` file. ``imageio`` along with the ``imageio-ffmpeg`` plugin
is included in the default environment so this workflow works out of the box
for ``.avi`` files.

Example:

```bash
conda run --prefix ./dev_env python -m Code.compare_intensity_stats \
    SMOKE data/smoke_1a_bgsub_raw.avi \
    CRIM data/10302017_10cms_bounded_2.h5 \
    --pure-python
```

#### When MATLAB cannot be found

If `compare_intensity_stats` cannot locate MATLAB it prints an error:

```text
ERROR: MATLAB executable not found. Set $MATLAB_EXEC or use --matlab_exec
```

For example, running the command without a detected MATLAB might look like:

```bash
conda run --prefix ./dev_env python -m Code.compare_intensity_stats \
    CRIM data/10302017_10cms_bounded_2.h5 \
    SMOKE process_smoke_video.m
```

To fix this, either export the executable path:

```bash
export MATLAB_EXEC=/path/to/matlab
```

or pass it explicitly:

```bash
conda run --prefix ./dev_env python -m Code.compare_intensity_stats \
    CRIM data/10302017_10cms_bounded_2.h5 \
    SMOKE process_smoke_video.m \
    --matlab_exec /path/to/matlab
```

For convenience, you can also use `scripts/run_intensity_batch.py` which wraps
`Code.compare_intensity_stats` for the common case of comparing the default
Crimaldi dataset with a smoke plume script. The helper reads the MATLAB path
from `configs/project_paths.yaml` if available.

```bash
conda run --prefix ./dev_env python scripts/run_intensity_batch.py \
    data/10302017_10cms_bounded_2.h5 process_smoke_video.m
```


### MATLAB Script Execution

The project includes `video_script.m` for processing smoke video data. This script is designed to work with the project's path management system.

When `video_script.m` is launched via the Python wrapper, it receives an `orig_script_dir` variable pointing to the project root. This lets the script find the `Code/` directory and any data files even though MATLAB runs in a temporary directory.

#### video_script.m

This script processes smoke video data and extracts intensity values. It's designed to be called from Python but can also be run directly in MATLAB.

```matlab
% video_script.m
% Processes smoke video data and extracts intensity values
plume = load_plume_video('data/smoke_1a_bgsub_raw.avi', 6.536, 60);
all_intensities = plume.data(:);
save('temp_intensities.mat', 'all_intensities');
fprintf('TEMP_MAT_FILE_SUCCESS:%s\n', which('temp_intensities.mat'));
```

**Key Features**:
- Automatic path handling: when invoked from Python, `orig_script_dir` is set so the script can locate `Code/` and data files even when run from a temporary directory
- Improved error handling via a `try`/`catch` block that prints a stack trace on failure
- Loads and processes smoke video data
- Saves intensity values to a temporary MAT file
- Outputs the path to the generated file for Python integration

#### Running from Python

Use the `compare_intensity_stats.py` script to process video data. The MATLAB configuration is automatically handled by the `paths.sh` script using settings from `configs/project_paths.yaml`.

```bash
# First source the paths if you haven't already
source ./paths.sh

# Then run the comparison
conda run --prefix ./dev_env python -m Code.compare_intensity_stats \
    CRIM data/10302017_10cms_bounded_2.h5 \
    SMOKE video_script.m \
    ${MATLAB_EXEC:+--matlab_exec "$MATLAB_EXEC"}
```

If MATLAB is not auto-detected or you want to use a specific version, supply the
path explicitly:

```bash
conda run --prefix ./dev_env python -m Code.compare_intensity_stats \
    CRIM data/10302017_10cms_bounded_2.h5 \
    SMOKE video_script.m \
    --matlab_exec /path/to/matlab
```

#### MATLAB Configuration

The MATLAB configuration is stored in `configs/project_paths.yaml` under the `matlab` section. You can edit this file directly or let the system auto-detect MATLAB.

```yaml
matlab:
  # Path to MATLAB executable (auto-detected if not specified)
  # executable: "/Applications/MATLAB_R2023a.app/bin/matlab"
  
  # Additional MATLAB paths to add (relative to project root)
  paths:
    - "${PROJECT_DIR}/Code"
    - "${PROJECT_DIR}"
  
  # MATLAB toolboxes required (for documentation purposes)
  required_toolboxes:
    - "Image Processing Toolbox"
    - "Statistics and Machine Learning Toolbox"
```

#### Custom MATLAB Path

`paths.sh` normally detects MATLAB automatically and exposes the path via
`$MATLAB_EXEC`. If detection fails, you can set `MATLAB_EXEC` before sourcing the
script or edit `configs/project_paths.yaml`:

```bash
MATLAB_EXEC=/usr/local/MATLAB/bin/matlab source ./paths.sh
```

#### Key Paths in `project_paths.yaml`

```yaml
scripts:
  matlab: "${PROJECT_DIR}"  # Project root for MATLAB scripts (orig_script_dir)
  python: "${PROJECT_DIR}/Code"     # Python modules
  temp: "${TMPDIR:-/tmp}/matlab_scripts"   # Temporary execution directory
```

#### Custom MATLAB Scripts

If you need to create a custom MATLAB script, follow this template:

```matlab
% Process smoke video for intensity comparison
% This script is designed to be called by compare_intensity_stats.py

% Get the path to the original script directory
if ~exist('orig_script_dir', 'var')
    orig_script_dir = pwd;
end

% Load configuration
cfgPath = fullfile(orig_script_dir, 'configs', 'my_complex_plume_config.yaml');
if ~exist(cfgPath, 'file')
    error('Config file not found: %s', cfgPath);
end
cfg = load_config(cfgPath);

% Process the smoke video
videoPath = fullfile(orig_script_dir, 'data', 'smoke_1a_bgsub_raw.avi');
if ~exist(videoPath, 'file')
    error('Video file not found: %s', videoPath);
end

plume = load_plume_video(videoPath, cfg.px_per_mm, cfg.frame_rate);
all_intensities = plume.data(:);

% Save to temporary file
outputFile = fullfile(tempdir, 'temp_intensities.mat');
save(outputFile, 'all_intensities');
fprintf('TEMP_MAT_FILE_SUCCESS:%s\n', outputFile);
```

#### Notes on Execution

- The script runs in a temporary directory, so always use absolute paths or `orig_script_dir`
- MATLAB must be properly licensed and installed on the system
- The script handles errors gracefully with descriptive messages
- Temporary files are automatically cleaned up after execution

## Configuration

### Project Structure


```
project_root/
├── Code/                  # Python modules and utilities
├── configs/               # Configuration files
│   └── project_paths.yaml         # Local paths configuration (generated from template)
├── data/                  # Data files (HDF5, videos, etc.)
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── scripts/               # Helper utilities (MATLAB & Python)
├── figures/               # Output figures
└── *.m                    # MATLAB scripts live at the project root
```

MATLAB scripts like `video_script.m` and `process_smoke_video.m` reside directly
in the project root. There is no dedicated `scripts/matlab` folder.

### Local Paths Configuration

Each user should have their own local `configs/project_paths.yaml` file (gitignored for security). A template is provided at `configs/project_paths.yaml.template`.

Key paths configured in `project_paths.yaml`:
- `scripts.matlab`: Points to the project root where MATLAB scripts are located
- `scripts.python`: Points to the `Code` directory with Python modules
- `data.*`: Paths to various data files and directories

### MATLAB Path Resolution

When running MATLAB scripts through the Python wrapper:
1. The `orig_script_dir` variable is set to the project root directory
2. The MATLAB path is automatically configured to include:
   - The project root directory (where the MATLAB scripts live)
   - All subdirectories under `Code/`
   - The `scripts/` directory for helper utilities

For direct MATLAB execution, ensure your MATLAB path includes these directories.

To set up your local configuration:

1. Copy the template to create your local config:
   ```bash
   cp configs/project_paths.yaml.template configs/project_paths.yaml
   ```

2. Edit `configs/project_paths.yaml` to set the correct paths for your system, particularly:
   - `crimaldi_hdf5`: Path to your Crimaldi HDF5 file
   - `output` directories: Where to store processed files and figures

### Example Configuration

For a fully annotated sample configuration, see
`configs/examples/smoke_plume.yaml`. This file mirrors the default settings
but includes comments describing each option.

### MATLAB Configuration

For MATLAB integration, ensure you have these MATLAB toolboxes installed:
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

`paths.sh` adds MATLAB's `bin` directory to your `PATH` when it locates the
executable, so no manual export is required.

## Notes

- All commands assume the development environment created via `./setup_env.sh --dev`
- Output paths are controlled by `configs/project_paths.yaml`
- The MATLAB script `process_smoke_video.m` is pre-configured to work with the default paths
- To preview example frames from both datasets, run `notebooks/orient_plumes.ipynb`:
  ```bash
  conda run --prefix ./dev_env jupyter notebook notebooks/orient_plumes.ipynb
  ```
