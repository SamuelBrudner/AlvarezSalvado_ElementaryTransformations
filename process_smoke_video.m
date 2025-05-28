%% Process smoke video data to extract intensity measurements
% 
% This script processes smoke video data to extract intensity measurements
% and save the results to a MAT file for further analysis.
% 
% Paths are resolved via ``load_paths_config``, which reads the project's
% ``configs/project_paths.yaml`` (falling back to ``configs/paths.yaml``).
% When invoked from Python, the wrapper sets
% ``orig_script_dir`` to the MATLAB scripts directory specified in that
% configuration. No ``video_path`` or ``output_path`` variables are required.
%
% Example (Python)
%   >>> from Code.video_intensity import get_intensities_from_video_via_matlab
%   >>> arr = get_intensities_from_video_via_matlab('process_smoke_video.m', 'matlab')

% Add project directories to MATLAB path
if exist('orig_script_dir', 'var')
    % When called from Python wrapper, use the provided script directory
    projectRoot = orig_script_dir;
else
    % Fallback: use the directory containing this script
    projectRoot = fileparts(mfilename('fullpath'));
end

% Add necessary directories to MATLAB path
addpath(genpath(fullfile(projectRoot, 'Code')));  % Add all subdirectories in Code
addpath(fullfile(projectRoot, 'scripts'));        % Add scripts directory

% Display path information for debugging
fprintf('Project root: %s\n', projectRoot);
fprintf('MATLAB path contains %d directories\n', length(strsplit(path, pathsep)));

% Load paths configuration
try
    paths = load_paths_config();
catch ME
    error('Failed to load paths configuration: %s', ME.message);
end

% Load plume configuration
if ~exist(paths.configs.plume, 'file')
    error('Plume config file not found: %s', paths.configs.plume);
end
cfg = load_config(paths.configs.plume);

% Process the smoke video
if ~exist(paths.data.video, 'file')
    error('Video file not found: %s', paths.data.video);
end

fprintf('Processing video: %s\n', paths.data.video);
plume = load_plume_video(paths.data.video, cfg.px_per_mm, cfg.frame_rate);

% Flatten the data to a 1D array of intensities
all_intensities = plume.data(:);
fprintf('Extracted %d intensity values\n', numel(all_intensities));

% Save the intensities to a temporary file
outputFile = fullfile(paths.output.matlab_temp, 'temp_intensities.mat');
save(outputFile, 'all_intensities', '-v7.3');

% Print the path to the temporary file for the Python script to find
fprintf('TEMP_MAT_FILE_SUCCESS:%s\n', outputFile);
