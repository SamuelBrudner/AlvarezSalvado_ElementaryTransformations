% Add the Code directory to MATLAB path
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, 'Code'));

try
    % Load the video data
    videoPath = fullfile(scriptDir, 'data', 'smoke_1a_bgsub_raw.avi');
    if ~exist(videoPath, 'file')
        error('Video file not found: %s', videoPath);
    end
    
    % Load plume video with parameters
    plume = load_plume_video(videoPath, 6.536, 60);
    
    % Extract and save intensities
    all_intensities = plume.data(:);
    outputPath = fullfile(scriptDir, 'temp_intensities.mat');
    save(outputPath, 'all_intensities');
    
    % Print success message with full path
    fprintf('TEMP_MAT_FILE_SUCCESS:%s\n', outputPath);
catch ME
    % Print detailed error information
    fprintf(2, 'MATLAB Error: %s\n', ME.message);
    for i = 1:length(ME.stack)
        fprintf(2, '  In %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    rethrow(ME);
end