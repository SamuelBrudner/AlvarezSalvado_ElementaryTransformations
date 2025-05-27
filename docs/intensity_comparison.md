# Intensity Dataset Comparison

This page describes how to characterise the intensity of individual odour plumes and how to compare multiple intensity datasets.

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
   This sets up all necessary environment variables and paths. You can run this anytime to update paths without rebuilding the conda environment.

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
conda run --prefix ./dev-env python -m Code.analyze_crimaldi_data data/raw/plume1.hdf5
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
conda run --prefix ./dev-env python -m Code.compare_intensity_stats data/raw/plume1.hdf5 data/raw/plume2.hdf5
```

To see the mean and median differences when exactly two datasets are provided, add the `--diff` option:

```bash
conda run --prefix ./dev-env python -m Code.compare_intensity_stats A data/raw/plume1.hdf5 B data/raw/plume2.hdf5 --diff
```

To save the computed statistics, provide a path via `--csv` or `--json`:

```bash
conda run --prefix ./dev-env python -m Code.compare_intensity_stats A data/raw/plume1.hdf5 B data/raw/plume2.hdf5 --json results/stats.json
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
conda run --prefix ./dev-env python -m Code.compare_intensity_stats \
    CRIM data/10302017_10cms_bounded_2.h5 \
    SMOKE process_smoke_video.m \
    --matlab_exec /path/to/matlab/executable
```

### MATLAB Script Execution

The project includes `video_script.m` for processing smoke video data. This script is designed to work with the project's path management system.

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
- Loads and processes smoke video data
- Saves intensity values to a temporary MAT file
- Outputs the path to the generated file for Python integration

#### Running from Python

Use the `compare_intensity_stats.py` script to process video data. The MATLAB configuration is automatically handled by the `paths.sh` script using settings from `configs/project_paths.yaml`.

```bash
# First source the paths if you haven't already
source ./paths.sh

# Then run the comparison
conda run --prefix ./dev-env python -m Code.compare_intensity_stats \
    CRIM data/10302017_10cms_bounded_2.h5 \
    SMOKE video_script.m \
    ${MATLAB_EXEC:+--matlab_exec "$MATLAB_EXEC"}
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

If you need to specify a custom MATLAB path, you can either:

1. Edit `configs/project_paths.yaml` and set the `matlab.executable` path
2. Or set it via environment variable before sourcing `paths.sh`:
   ```bash
   MATLAB_EXEC="/path/to/matlab" source ./paths.sh
   ```

#### What the Script Does

1. Checks for MATLAB configuration in `project_paths.yaml`
2. If not found, auto-detects MATLAB installation
3. Updates the configuration with the detected path (if yq is installed)
4. Sets up MATLAB paths as specified in the configuration
5. Makes the MATLAB executable available via `$MATLAB_EXEC`
6. Processes the video and generates comparison plots

#### MATLAB Path Management

When running MATLAB scripts, ensure:
1. MATLAB is in your system PATH
2. Required toolboxes are installed:
   - Image Processing Toolbox
   - Statistics and Machine Learning Toolbox
3. The repository contains the required data files in the expected locations

For development, you can test the MATLAB script directly:

```matlab
% In MATLAB
cd /path/to/project
video_script  % Run the script
load('temp_intensities.mat');  % Load the results
whos all_intensities  % Verify the output
```

For production use, the Python wrapper handles all path resolution and file management automatically.

When processing video data, the system uses a MATLAB script (`process_smoke_video.m`) to extract intensity values. The script execution follows this workflow:

1. **Path Configuration**: The system loads paths from `configs/paths.yaml`
2. **Script Preparation**: The MATLAB script is copied to a temporary directory for execution
3. **Path Variables**: The following variables are automatically set:
   - `orig_script_dir`: Points to the MATLAB scripts directory from `paths.yaml`
   - `scriptDir`: Set to the temporary execution directory

This ensures that while the script executes in a temporary directory, it can still locate all necessary dependencies through the configured paths.

#### Key Paths in `paths.yaml`

```yaml
scripts:
  matlab: "${PROJECT_DIR}/scripts"  # Original script directory (orig_script_dir)
  python: "${PROJECT_DIR}/Code"     # Python modules
  temp: "${TMPDIR}/matlab_scripts"   # Temporary execution directory
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
│   └── paths.yaml         # Local paths configuration (generated from template)
├── data/                  # Data files (HDF5, videos, etc.)
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── scripts/               # MATLAB scripts and functions
└── figures/               # Output figures
```

### Local Paths Configuration

Each user should have their own local `configs/paths.yaml` file (gitignored for security). A template is provided at `configs/paths.yaml.template`.

Key paths configured in `paths.yaml`:
- `scripts.matlab`: Points to the project root where MATLAB scripts are located
- `scripts.python`: Points to the `Code` directory with Python modules
- `data.*`: Paths to various data files and directories

### MATLAB Path Resolution

When running MATLAB scripts through the Python wrapper:
1. The `orig_script_dir` variable is set to the project root directory
2. The MATLAB path is automatically configured to include:
   - The project root directory
   - All subdirectories under `Code/`
   - The `scripts/` directory

For direct MATLAB execution, ensure your MATLAB path includes these directories.

To set up your local configuration:

1. Copy the template to create your local config:
   ```bash
   cp configs/paths.yaml.template configs/paths.yaml
   ```

2. Edit `configs/paths.yaml` to set the correct paths for your system, particularly:
   - `crimaldi_hdf5`: Path to your Crimaldi HDF5 file
   - `output` directories: Where to store processed files and figures

### MATLAB Configuration

For MATLAB integration, ensure you have these MATLAB toolboxes installed:
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

Set the path to your MATLAB executable in your shell configuration (e.g., `~/.bashrc` or `~/.zshrc`):

```bash
export PATH="/path/to/matlab/bin:$PATH"
```

## Notes

- All commands assume the development environment created via `./setup_env.sh --dev`
- Output paths are controlled by `configs/paths.yaml`
- The MATLAB script `process_smoke_video.m` is pre-configured to work with the default paths
