function out = navigation_model_vec_stream(triallength, environment, plotting, ntrials, vr, params)
%NAVIGATION_MODEL_VEC_STREAM Run navigation model using streaming video input.
%   OUT = NAVIGATION_MODEL_VEC_STREAM(TRIALLENGTH, ENVIRONMENT, PLOTTING,
%   NTRIALS, VR, PARAMS) reads frames from the VideoReader object VR and passes
%   them to NAVIGATION_MODEL_VEC. Only the 'video' environment is supported.
%   PARAMS must include 'px_per_mm' and 'frame_rate'.
%
%   This is a convenience wrapper so that existing code can operate on a
%   VideoReader without loading the entire movie into memory first.

if nargin < 6
    params = struct();
end
if nargin < 4
    ntrials = 1;
end

assert(strcmpi(environment,'video'), 'Environment must be ''video''.');
assert(all(isfield(params,{ 'px_per_mm','frame_rate'})), ...
    'px_per_mm and frame_rate are required parameters');

% Read all frames from the VideoReader
frames = zeros(vr.Height, vr.Width, floor(vr.Duration * vr.FrameRate), 'single');
idx = 1;
while hasFrame(vr)
    f = readFrame(vr);
    if size(f,3) == 3
        f = rgb2gray(f);
    end
    frames(:,:,idx) = im2single(f);
    idx = idx + 1;
end
plume.data = frames;
plume.px_per_mm = params.px_per_mm;
plume.frame_rate = params.frame_rate;

out = navigation_model_vec(triallength, 'video', plotting, ntrials, plume, params);
end
