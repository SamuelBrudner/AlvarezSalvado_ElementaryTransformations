function plume = load_plume_video(filename, px_per_mm, frame_rate)
%LOAD_PLUME_VIDEO Load an odor plume movie from an AVI file.
%   PLUME = LOAD_PLUME_VIDEO(FILENAME, PX_PER_MM, FRAME_RATE) reads the
%   specified AVI movie and returns a structure compatible with
%   NAVIGATION_MODEL_VEC. PX_PER_MM defines the spatial scale of the
%   movie in pixels per millimeter and FRAME_RATE specifies the movie
%   frame rate in Hz.
%
%   The returned structure contains:
%       data       - double array (height x width x frames) of odor values
%       px_per_mm  - pixels per millimeter
%       frame_rate - frames per second
%
%   Example:
%       plume = load_plume_video('plume.avi', 20, 50);
%
%   The movie is converted to grayscale and normalised in the range [0,1].
%   An error with identifier 'load_plume_video:FrameSizeMismatch' is thrown if
%   a frame differs in size from the first frame.

v = VideoReader(filename);

% Early exit if estimated array would exceed ~8 GB
estFrames = ceil(v.FrameRate * v.Duration);
bytesNeeded = double(v.Height) * double(v.Width) * double(estFrames) * 8;
if bytesNeeded > 8 * 1024^3
    warning('load_plume_video:MemoryExceeded', ...
        'Estimated movie size %.1f GB exceeds 8 GB limit; aborting load.', ...
        bytesNeeded / 1024^3);
    plume.data = [];
    plume.px_per_mm = px_per_mm;
    plume.frame_rate = frame_rate;
    return;
end

% Determine frame count and check frame sizes
frameCount = 0;
height = [];
width = [];

while hasFrame(v)
    frame = readFrame(v);
    if size(frame,3) == 3
        frame = rgb2gray(frame);
    end
    [h, w] = size(frame);
    if frameCount == 0
        height = h;
        width = w;
    else
        if h ~= height || w ~= width
            error('load_plume_video:FrameSizeMismatch', ...
                'Frame %d is %dx%d, expected %dx%d', ...
                frameCount + 1, h, w, height, width);
        end
    end
    frameCount = frameCount + 1;
end

fprintf('%d x %d, %d frames\n', height, width, frameCount);

% Reset the video reader to the beginning
v = VideoReader(filename);

% Pre-allocate using the first frame's dimensions
frames = zeros(height, width, frameCount, 'double');

% Read frames
for idx = 1:frameCount
    frame = readFrame(v);
    if size(frame,3) == 3
        frame = rgb2gray(frame);
    end
    frames(:,:,idx) = im2double(frame);
end

plume.data = frames;
plume.px_per_mm = px_per_mm;
plume.frame_rate = frame_rate;
