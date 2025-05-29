function plume = load_plume_hdf5(filename, px_per_mm, frame_rate)
%LOAD_PLUME_HDF5 Load plume data from an HDF5 file.
%   PLUME = LOAD_PLUME_HDF5(FILENAME, PX_PER_MM, FRAME_RATE) reads the
%   dataset named 'dataset1' from the specified HDF5 file. The returned
%   structure contains fields:
%       data       - numeric array (height x width x frames)
%       px_per_mm  - pixels per millimeter
%       frame_rate - frames per second
%
%   Example:
%       plume = load_plume_hdf5('plume.h5', 20, 50);

arguments
    filename (1,:) char
    px_per_mm (1,1) double
    frame_rate (1,1) double
end

data = h5read(filename, '/dataset1');
plume.data = double(data);
plume.px_per_mm = px_per_mm;
plume.frame_rate = frame_rate;
end
