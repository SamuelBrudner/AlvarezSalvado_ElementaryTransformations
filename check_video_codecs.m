% Check available video codecs
fprintf('MATLAB version: %s\n', version);
fprintf('Computer type: %s\n', computer);

% Check if we can read video info without VideoReader
video_path = 'data/smoke_1a_orig_backgroundsubtracted.avi';
try
    info = audiovideoinfo(video_path);
    fprintf('\naudiovideoinfo succeeded:\n');
    fprintf('  Format: %s\n', info.FileFormat);
    fprintf('  Duration: %.2f seconds\n', info.Duration);
catch ME
    fprintf('\naudiovideoinfo failed: %s\n', ME.message);
end

% Check available codecs
fprintf('\nChecking mmreader (legacy):\n');
try
    mmreader.getFileFormats()
catch
    fprintf('mmreader not available (expected in newer MATLAB)\n');
end

% Try aviread (very legacy)
fprintf('\nTrying aviread:\n');
try
    info = aviinfo(video_path);
    fprintf('aviread might work! Video is %dx%d\n', info.Width, info.Height);
catch ME
    fprintf('aviread failed: %s\n', ME.message);
end

exit;
