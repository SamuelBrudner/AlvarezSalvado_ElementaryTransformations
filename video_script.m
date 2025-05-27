% Get the directory where this script is located
scriptDir = fileparts(mfilename('fullpath'));
fprintf('Script directory: %s\n', scriptDir);

% Add necessary directories to MATLAB path
codeDir = fullfile(scriptDir, 'Code');
if ~exist(codeDir, 'dir')
    error('Code directory not found: %s', codeDir);
end
addpath(genpath(codeDir));  % Add Code directory and all subdirectories
fprintf('Added to path: %s\n', codeDir);

% Check if we have a video file path passed as a parameter
if exist('video_file', 'var') && ~isempty(video_file)
    videoPath = video_file;
    fprintf('Using provided video file: %s\n', videoPath);
else
    % Fall back to default path if no parameter provided
    videoRelativePath = 'data/smoke_1a_bgsub_raw.avi';
    videoPath = fullfile(scriptDir, videoRelativePath);
    fprintf('No video file provided, using default: %s\n', videoPath);
end

% Check if video file exists
if ~exist(videoPath, 'file')
    error('Video file not found: %s', videoPath);
end
fprintf('Found video file: %s\n', videoPath);

% List all functions in the path that match 'load_plume_video'
fprintf('Searching for load_plume_video in path...\n');
which('load_plume_video', '-all')

try
    
    % Display MATLAB version and toolboxes
    ver
    
    % Display current path
    fprintf('Current working directory: %s\n', pwd);
    fprintf('MATLAB path:\n');
    path
    
    % Test if we can read the video file
    fprintf('Attempting to read video file...\n');
    try
        videoInfo = mmfileinfo(videoPath);
        fprintf('Video info: %dx%d, %d frames, duration: %.2f sec\n', ...
            videoInfo.Video.Width, videoInfo.Video.Height, ...
            videoInfo.Video.NumFrames, videoInfo.Duration);
    catch videoErr
        fprintf('Warning: Could not read video file: %s\n', videoErr.message);
    end
    
    % Load plume video with parameters
    fprintf('Calling load_plume_video(%s, 6.536, 60)...\n', videoPath);
    plume = load_plume_video(videoPath, 6.536, 60);
    
    % Check the plume structure
    fprintf('Loaded plume data. Fields: %s\n', strjoin(fieldnames(plume), ', '));
    if ~isfield(plume, 'data')
        error('Plume structure does not contain ''data'' field');
    end
    
    % Extract and save intensities
    all_intensities = plume.data(:);
    fprintf('Extracted %d intensity values\n', numel(all_intensities));
    
    outputPath = fullfile(scriptDir, 'temp_intensities.mat');
    fprintf('Saving to: %s\n', outputPath);
    save(outputPath, 'all_intensities', '-v7.3');
    
    % Verify the file was saved
    if ~exist(outputPath, 'file')
        error('Failed to save output file: %s', outputPath);
    end
    
    % Print success message with full path
    fprintf('TEMP_MAT_FILE_SUCCESS:%s\n', outputPath);
    
catch ME
    % Print detailed error information
    fprintf(2, 'MATLAB Error: %s\n', ME.message);
    for i = 1:min(length(ME.stack), 10)  % Limit stack trace to 10 frames
        frame = ME.stack(i);
        fprintf(2, '  In %s (line %d)\n', frame.name, frame.line);
    end
    
    % Try to get more detailed error information
    if exist('getReport', 'file')
        fprintf(2, '\nFull error report:\n%s\n', getReport(ME, 'extended'));
    end
    
    % Exit with error code
    exit(1);
end