# Plume Processing Pipeline

`video_to_scaled_rotated_h5` converts an AVI file to HDF5, rescales
intensities to the Crimaldi range, rotates each frame 90° clockwise and
records the resulting files in `configs/plume_registry.yaml`.

## Usage

Ensure the development environment is available and activate the paths:

```bash
./setup_env.sh --dev
source ./paths.sh
```

Run the pipeline using the command line helper:

```bash
conda run --prefix ./dev_env python -m scripts.run_plume_pipeline \
    data/raw/input.avi data/processed/input_raw.h5 \
    data/processed/input_scaled.h5 data/processed/input_rotated.h5
```

The script creates the three HDF5 files and updates
`configs/plume_registry.yaml` with their intensity ranges.

## Python API

You can invoke the pipeline directly from Python:

```python
from Code.plume_pipeline import video_to_scaled_rotated_h5

video_to_scaled_rotated_h5(
    "data/raw/input.avi",
    "data/processed/input_raw.h5",
    "data/processed/input_scaled.h5",
    "data/processed/input_rotated.h5",
)
```
