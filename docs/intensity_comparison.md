# Intensity Dataset Comparison

This page describes how to characterise the intensity of individual odour plumes and how to compare multiple intensity datasets.
Before running any commands, create the development environment using `./setup_env.sh --dev`.
Use the module form `python -m Code.<script>` when executing scripts so the repository root is on `sys.path`.

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

If you have a custom plume movie, extract the intensity values in MATLAB and compare
them to the Crimaldi data. Below is a minimal script `video_script.m`. Replace
`'my_plume.avi'` with the path to your movie. The example below loads
`configs/my_complex_plume_config.yaml` so the pixel conversion and frame rate
are pulled from that file. The YAML file defines `px_per_mm` and
`frame_rate` used by `load_plume_video`.
`get_intensities_from_video_via_matlab` now also exposes ``orig_script_path`` and
``orig_script_dir`` variables for convenience:

```matlab
% When run via `compare_intensity_stats.py` this script is copied to a
% temporary location. The helper function `get_intensities_from_video_via_matlab`
% automatically inserts ``cd(work_dir)`` so `pwd` points to the original
% directory. It now also exposes ``orig_script_path`` and ``orig_script_dir`` to
% help resolve resources relative to the original script folder. Use
% ``orig_script_dir`` in preference to `pwd` when constructing paths.
cfgPath = fullfile(orig_script_dir, 'configs', 'my_complex_plume_config.yaml');
cfg = load_config(cfgPath);
plume = load_plume_video('data/smoke_1a_bgsub_raw.avi', cfg.px_per_mm, cfg.frame_rate);
all_intensities = plume.data(:);
save('temp_intensities.mat', 'all_intensities');
fprintf('TEMP_MAT_FILE_SUCCESS:%s\n', which('temp_intensities.mat'));
```

`compare_intensity_stats.py` executes a temporary copy of the script so
``mfilename('fullpath')`` refers to that temporary location. Because
``get_intensities_from_video_via_matlab`` inserts ``cd(work_dir)`` and now
defines ``orig_script_path`` and ``orig_script_dir`` variables, the
current directory is the original script folder. Prefer ``orig_script_dir``
or an absolute path when locating configuration files.

Save the script and pass **its full path** to the Python utility. The
`TEMP_MAT_FILE_SUCCESS` line is used by `compare_intensity_stats.py` to locate the
generated MAT-file.

Run the comparison using the development environment created with `./setup_env.sh --dev`:

```bash
conda run --prefix ./dev-env python -m Code.compare_intensity_stats VID video path/to/video_script.m CRIM crimaldi data/10302017_10cms_bounded.hdf5 --matlab_exec /path/to/matlab
```

## Notes

- All commands assume the development environment created via `./setup_env.sh --dev`.
- Output paths are controlled by `configs/paths.yaml`.
