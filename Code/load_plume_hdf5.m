function plume = load_plume_hdf5(filename, px_per_mm, frame_rate)
%LOAD_PLUME_HDF5 Load plume data from an HDF5 file.
%   PLUME = LOAD_PLUME_HDF5(FILENAME, PX_PER_MM, FRAME_RATE) reads the
%   dataset from the specified HDF5 file. It will look for dataset2 first,
%   then dataset1. The returned structure contains fields:
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

% Try to find the dataset
info = h5info(filename);
dataset_name = '';

% First check for dataset2 (Crimaldi), then dataset1 (smoke)
if any(strcmp({info.Datasets.Name}, 'dataset2'))
    dataset_name = '/dataset2';
elseif any(strcmp({info.Datasets.Name}, 'dataset1'))
    dataset_name = '/dataset2';
else
    error('No dataset1 or dataset2 found in %s', filename);
end

fprintf('Loading dataset: %s\n', dataset_name);
data = h5read(filename, dataset_name);

% Handle 1D data (needs reshaping based on attributes)
if isvector(data)
    if isfield(info.Datasets(1).Attributes, 'height') && ...
       isfield(info.Datasets(1).Attributes, 'width') && ...
       isfield(info.Datasets(1).Attributes, 'frames')
        % Use attributes if available
        height = info.Datasets(1).Attributes.height;
        width = info.Datasets(1).Attributes.width;
        frames = info.Datasets(1).Attributes.frames;
    else
        % For smoke data, we need to figure out dimensions
        % The smoke HDF5 has 6768230400 elements
        % Typical smoke video: 3120x3120x688 would be 6,703,872,000 (close)
        % Let's check if it's stored with attributes
        fprintf('Warning: 1D data without shape attributes, cannot reshape\n');
        height = 0; width = 0; frames = 0;
    end
    
    if height > 0 && width > 0 && frames > 0
        data = reshape(data, [height, width, frames]);
    end
end

plume.data = double(data);
plume.px_per_mm = px_per_mm;
plume.frame_rate = frame_rate;
end
