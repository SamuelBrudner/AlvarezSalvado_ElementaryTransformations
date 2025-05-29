function data = rescale_plume_range(data, target_min, target_max)
%RESCALE_PLUME_RANGE Linearly rescale plume data to a target range.
%   DATA = RESCALE_PLUME_RANGE(DATA, TARGET_MIN, TARGET_MAX) rescales the
%   numeric array DATA so that its minimum value becomes TARGET_MIN and its
%   maximum value becomes TARGET_MAX. Empty inputs are returned unchanged.

if isempty(data)
    return;
end

src_min = min(data(:));
src_max = max(data(:));
if src_max == src_min
    data(:) = target_min;
    return;
end
scale = (target_max - target_min) / (src_max - src_min);
data = (data - src_min) * scale + target_min;
end
