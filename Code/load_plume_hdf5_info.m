function plume = load_plume_hdf5_info(filename, px_per_mm, frame_rate)
%LOAD_PLUME_HDF5_INFO Load plume metadata without loading data into memory.
%   Returns a structure with HDF5 file info for direct access during simulation.

arguments
    filename (1,:) char
    px_per_mm (1,1) double
    frame_rate (1,1) double
end

% Get file info
info = h5info(filename);

% Find dataset name
dataset_name = '';
if any(strcmp({info.Datasets.Name}, 'dataset2'))
    dataset_name = '/dataset2';
elseif any(strcmp({info.Datasets.Name}, 'dataset1'))
    dataset_name = '/dataset1';
else
    error('No dataset1 or dataset2 found in %s', filename);
end

% Get dimensions
dataset_info = h5info(filename, dataset_name);
dims = dataset_info.Dataspace.Size;

fprintf('HDF5 plume info: %s %s [%d x %d x %d]\n', filename, dataset_name, dims);

% Return structure with file info instead of data
plume.filename = filename;
plume.dataset = dataset_name;
plume.dims = dims;  % [width height frames] for our files
plume.px_per_mm = px_per_mm;
plume.frame_rate = frame_rate;
end
