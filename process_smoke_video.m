%% Process smoke video for intensity comparison
% This script is designed to be called by compare_intensity_stats.py
% It processes a smoke video and extracts intensity values for comparison with Crimaldi data

% Get the path to the original script directory for loading config files
% Note: orig_script_dir is set by the Python wrapper
if ~exist('orig_script_dir', 'var')
    % If not set by Python, use the current directory
    orig_script_dir = pwd;
end

% Construct path to config file
cfgPath = fullfile(orig_script_dir, 'configs', 'my_complex_plume_config.yaml');
if ~exist(cfgPath, 'file')
    error('Config file not found: %s', cfgPath);
end

% Load configuration
cfg = load_config(cfgPath);

% Process the smoke video
videoPath = fullfile(orig_script_dir, 'data', 'smoke_1a_bgsub_raw.avi');
if ~exist(videoPath, 'file')
    error('Video file not found: %s', videoPath);
end

plume = load_plume_video(videoPath, cfg.px_per_mm, cfg.frame_rate);

% Flatten the data to a 1D array of intensities
all_intensities = plume.data(:);

% Save the intensities to a temporary file
outputFile = fullfile(tempdir, 'temp_intensities.mat');
save(outputFile, 'all_intensities');

% Print the path to the temporary file for the Python script to find
fprintf('TEMP_MAT_FILE_SUCCESS:%s\n', outputFile);
