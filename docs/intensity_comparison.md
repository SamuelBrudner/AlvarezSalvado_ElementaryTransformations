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
`frame_rate` used by `load_plume_video`:

```matlab
cfg = load_config('configs/my_complex_plume_config.yaml');
plume = load_plume_video('data/smoke_1a_bgsub_raw.avi', cfg.px_per_mm, cfg.frame_rate);
all_intensities = plume.data(:);
save('temp_intensities.mat', 'all_intensities');
fprintf('TEMP_MAT_FILE_SUCCESS:%s\n', which('temp_intensities.mat'));
```

Save the script as `docs/video_script_example.m` (or your own version) and pass **its full path** to the Python utility. The
`TEMP_MAT_FILE_SUCCESS` line is used by `compare_intensity_stats.py` to locate the
generated MAT-file.

Run the comparison using the development environment created with `./setup_env.sh --dev`:

```bash
conda run --prefix ./dev-env python -m Code.compare_intensity_stats VID video docs/video_script_example.m CRIM crimaldi data/10302017_10cms_bounded.hdf5 --matlab_exec /path/to/matlab
```

In this example, `--file_path` points to `docs/video_script_example.m`. You can also provide the path to your own MATLAB script that outputs `temp_intensities.mat`.

## Notes

- All commands assume the development environment created via `./setup_env.sh --dev`.
- Output paths are controlled by `configs/paths.yaml`.
