# Plume Processing Pipeline

This repository includes helper functions to convert plume videos to HDF5 and apply common transformations. The main entry point is `video_to_scaled_rotated_h5` which performs the following steps:

1. **Convert** the input AVI file to HDF5 using `video_to_hdf5`. The raw intensities and frame dimensions are stored under the dataset `dataset1`.
2. **Scale** the intensities to the Crimaldi range with `scale_hdf5_to_crim_range`.
3. **Rotate** the scaled frames 90° clockwise via `rotate_hdf5_clockwise`.
4. **Record** the intensity ranges of the intermediate files in `configs/plume_registry.yaml`.

The resulting HDF5 files preserve the original video shape as dataset attributes `height`, `width` and `frames`.
