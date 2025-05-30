# AlvarezSalvado Elementary Transformations

This repository hosts MATLAB and Python code for the navigation model from Álvarez-Salvado *et al.* (2018) and related analysis tools.

For detailed instructions on environment setup and intensity comparison workflows, see [docs/intensity_comparison.md](docs/intensity_comparison.md).

- [Quick Start](#quick-start)
- [Step-by-Step](#step-by-step)
- [Overview](#overview)
- [Requirements](#requirements)
- [Development Environment Setup](#development-environment-setup)
- [Configuration](#configuration)
- [Data Organization](#data-organization)
- [Data Dictionary](#data-dictionary)
- [Exporting Results to Open Formats](#exporting-results-to-open-formats)
- [Running Simulations](#running-simulations)
- [Analysis Configuration](#analysis-configuration)
- [Plume Intensity Utilities](#plume-intensity-utilities)
- [Repository Layout](#repository-layout)
- [Citation](#citation)
- [Roadmap](#roadmap)

## Quick Start

Check if the conda environment exists, if not, create it.

 - Check: `./dev_env` exists
 - If not, run: `./setup_env.sh --dev`
   - Use `--clean-install` for a fresh reinstall if needed

Check if pre commit hooks are installed, if not, install them.

 - Check: `./dev_env/.git/hooks/pre-commit` exists
 - If not, run: `conda run --prefix dev_env pre-commit install`

Sourcing ``paths.sh`` creates ``configs/project_paths.yaml`` from the template
if it does not already exist.

`paths.sh` attempts to locate MATLAB automatically and sets `MATLAB_EXEC`.
Set this variable yourself or use the `--matlab_exec` option to override the
auto-detected path.
If MATLAB is not found, `paths.sh` tries to load a module named
`MATLAB/$MATLAB_VERSION` (or `MATLAB_MODULE` if set). Set these variables
before sourcing to override the default.
Python utilities such as `Code.video_intensity` also honour `MATLAB_EXEC` when
it points to a valid executable. `Code.compare_intensity_stats` now processes
datasets of different lengths automatically and logs a warning when lengths
vary.

`imageio` together with its `imageio-ffmpeg` backend is included in the default
environment so the `--pure-python` option works without any extra
installation when reading ``.avi`` files.

`opencv-python` is installed via pip from PyPI rather than conda, ensuring the
latest precompiled wheels are used.

See [docs/intensity_comparison.md](docs/intensity_comparison.md#initial-setup)
for a detailed explanation of the path setup process.

With the environment active you can run MATLAB and Python scripts from the `Code` directory using the module syntax:

```bash
conda run --prefix ./dev_env python -m Code.some_script
```

### Video Processing Example

The `Code.rotate_video` module provides utilities to rotate videos and
convert them to HDF5.  Paths below are given relative to the project
root:

```python
from Code.rotate_video import rotate_video_clockwise, video_to_hdf5

rotate_video_clockwise(
    "data/raw/example.avi",
    "data/processed/example_rotated.avi",
)
video_to_hdf5(
    "data/processed/example_rotated.avi",
    "data/processed/example.h5",
)
```

The convenience function :func:`Code.plume_pipeline.video_to_scaled_rotated_h5`
performs conversion, scaling to the Crimaldi range, and rotation in one step.
See `docs/plume_pipeline.md` or run the script directly:

```bash
conda run --prefix ./dev_env python -m scripts.run_plume_pipeline \
    data/raw/example.avi data/processed/example_raw.h5 \
    data/processed/example_scaled.h5 data/processed/example_rotated.h5
```

## Step-by-Step

1 and 2 are only *as required* -- check if there's evidence they've already run

1. Run `./setup_env.sh --dev` to create `./dev_env`.
   - Add `--clean-install` to force removal of the existing environment.
2. Source `./paths.sh` to generate `configs/project_paths.yaml` and detect MATLAB. `paths.sh` uses this file and falls back to default paths when `yq` is missing.
3. Execute `conda run --prefix ./dev_env python -m Code.compare_intensity_stats`.
4. Run `pre-commit` and verify all checks succeed before committing changes.

## Directory Overview

- `Code/` – MATLAB and Python modules
- `configs/` – configuration templates
- `data/` – raw and processed datasets
  - Processed files can live outside this repository as long as you provide the
    correct paths in your configuration files or on the command line.
- `docs/` – project documentation

For citations and metadata, see `CITATION.cff` and `codemeta.json`.

## Frame Rate and Spatial Scale

Video-based simulations rely on two parameters defined in your plume configuration: `px_per_mm` sets the conversion from pixels to millimetres and `frame_rate` specifies the number of frames per second. The navigation model automatically scales velocities using these values so that fly speeds remain in millimetres per second regardless of the video's native frame rate or scale.

Example snippet from `configs/my_complex_plume_config.yaml`:

```yaml
# Pixels per millimeter conversion factor
px_per_mm: 6.536
# Frame rate of the video in Hz
frame_rate: 60
# Override the movie length with a specific number of frames
# triallength: 7200
```

When `triallength` exceeds the movie length, the video automatically loops.
Setting `triallength` lets you cut or extend each trial to an explicit number of
frames.


## Running Tests

Invoke `pytest` from the repository root. Slow tests that set up a full
environment are skipped unless the `--runslow` flag is supplied:

```bash
conda run --prefix ./dev_env pytest --runslow
```

## Plume Intensity Utilities

The file `configs/plume_intensity_stats.yaml` stores summary statistics for the
standard smoke (SMOKE) and Crimaldi (CRIM) plumes. It lists mean, median,
percentile, and range values that are referenced when rescaling plume
intensities.

Both MATLAB and Python utilities automatically locate this file if no path is
provided:

- `plume_intensity_stats.m` determines the repository root at runtime and loads
  the YAML via `load_yaml`.
- `Code.plume_utils.get_intensity_stats()` resolves the path relative to the
  `Code` directory and parses the YAML using PyYAML (or a minimal fallback
  parser when PyYAML is absent).
- `scale_custom_plume.m` rescales a custom plume movie once so its values match
  the CRIM range specified in `configs/plume_intensity_stats.yaml`.
- The optional `scaled_to_crim` flag prevents `load_custom_plume` from applying
  this scaling twice when the plume has already been normalised.
- [Plume Transformation Utilities](docs/intensity_comparison.md#plume-transformation-utilities) provide scaling and rotation helpers.
- Intensity ranges for processed plumes are stored in
  `configs/plume_registry.yaml`; each entry is keyed by filename only.
  Transformation scripts update this file automatically. See
  [Plume Registry](docs/intensity_comparison.md#plume-registry).

Example workflow:

```matlab
meta = 'my_plume_meta.yaml';
scale_custom_plume(meta);       % writes <output_filename>_scaled.avi
plume = load_custom_plume(meta); % scaled_to_crim skips another scaling step
```

### Converting AVI to HDF5

Use `video_to_hdf5` when you need an HDF5 version of the smoke video for
Python-only pipelines:

```bash
conda run --prefix ./dev_env python - <<'PY'
from Code.rotate_video import video_to_hdf5
video_to_hdf5(
    "data/smoke_1a_orig_backgroundsubtracted.avi",
    "data/smoke_1a_orig_backgroundsubtracted.h5",
)
PY
```

Add the resulting path under `data.video_h5` in `configs/project_paths.yaml` so
all scripts can locate it automatically.
