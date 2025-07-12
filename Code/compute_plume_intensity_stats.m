function stats = compute_plume_intensity_stats(config_path, n_samples)
%COMPUTE_PLUME_INTENSITY_STATS  Basic intensity statistics of a plume dataset.
%
%   stats = COMPUTE_PLUME_INTENSITY_STATS(config_path) reads the plume
%   configuration JSON located at CONFIG_PATH, loads the referenced HDF5
%   dataset, samples up to 100 frames (uniformly spaced) and returns the
%   min, max and mean intensity values across those frames.
%
%   stats = COMPUTE_PLUME_INTENSITY_STATS(config_path, n_samples) samples
%   up to N_SAMPLES frames instead of the default 100.
%
%   The function is lightweight and suitable for execution on HPC login or
%   compute nodes – it never attempts to load the entire 3-D plume into
%   memory.
%
%   Returned structure fields
%   -------------------------
%   min             Global minimum across sampled frames
%   max             Global maximum across sampled frames
%   mean            Mean intensity across sampled frames
%   sampled_frames  Number of frames actually sampled
%   total_frames    Total number of frames in the dataset
%   plume_file      Absolute path to the HDF5 file
%   dataset_name    Dataset path within the HDF5 file
%
%   Example
%   -------
%   cfg = 'configs/plumes/crimaldi_10cms_bounded.json';
%   stats = compute_plume_intensity_stats(cfg);
%   disp(stats);
%
%   Samuel Brudner Lab – July 2025

arguments
    config_path (1,1) string
    n_samples   (1,1) double {mustBeInteger, mustBePositive} = 100
end

% -------------------------------------------------------------------------
% Read plume configuration -------------------------------------------------
cfg = jsondecode(fileread(config_path));

plume_file   = string(cfg.data_path.path);
dataset_name = string(cfg.data_path.dataset_name);
if dataset_name(1) ~= "/"
    dataset_name = "/" + dataset_name; %#ok<*STRNU>
end

if ~isfile(plume_file)
    error("Plume file not found: %s", plume_file);
end

% -------------------------------------------------------------------------
% Determine dataset dimensions --------------------------------------------
info      = h5info(plume_file, dataset_name);
dims      = info.Dataspace.Size;   % [X  Y  T]
if numel(dims) ~= 3
    error("Expected 3-D dataset [X Y T], got dims = [%s]", num2str(dims));
end
n_frames  = dims(3);

% Frame indices to sample (uniformly spaced) --------------------------------
step      = max(floor(n_frames / n_samples), 1);
indices   = 1:step:n_frames;
indices   = indices(1:min(numel(indices), n_samples));

% -------------------------------------------------------------------------
% Streaming pass to accumulate statistics ----------------------------------
min_val   =  inf;
max_val   = -inf;
sum_val   = 0.0;
count_px  = 0;

for idx = indices
    frame = h5read(plume_file, dataset_name, [1 1 idx], [dims(1) dims(2) 1]);
    frame = double(frame);

    min_val  = min(min_val, min(frame(:))); %#ok<MINMAX>
    max_val  = max(max_val, max(frame(:)));
    sum_val  = sum_val + sum(frame(:));
    count_px = count_px + numel(frame);
end

stats = struct( ...
    "min",            min_val, ...
    "max",            max_val, ...
    "mean",           sum_val / count_px, ...
    "sampled_frames", numel(indices), ...
    "total_frames",   n_frames, ...
    "plume_file",     plume_file, ...
    "dataset_name",   dataset_name);
end
