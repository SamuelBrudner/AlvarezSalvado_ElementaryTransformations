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

v = VideoReader(filename);
numFrames = floor(v.Duration * v.FrameRate);
frames = zeros(v.Height, v.Width, numFrames);
count = 1;
while hasFrame(v)
    frame = readFrame(v);
    if size(frame,3) == 3
        frame = rgb2gray(frame);
    end
    frames(:,:,count) = im2double(frame);
    count = count + 1;
end
plume.data = frames;
plume.px_per_mm = px_per_mm;
plume.frame_rate = frame_rate;
