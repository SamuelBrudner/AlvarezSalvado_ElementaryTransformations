% Convert smoke HDF5 from 1D to proper 3D format
fprintf('Converting smoke HDF5 to 3D format...\n');

% Paths
input_path = '/home/snb6/palmer_scratch/plume/smoke_1a_orig_backgroundsubtracted_rotated.h5';
output_path = '/home/snb6/palmer_scratch/plume/smoke_1a_rotated_3d.h5';

% Read the 1D data and attributes
fprintf('Reading 1D data...\n');
data_1d = h5read(input_path, '/dataset1');
info = h5info(input_path, '/dataset1');

% Get dimensions from attributes
height = [];
width = [];
frames = [];
for i = 1:length(info.Attributes)
    switch info.Attributes(i).Name
        case 'height'
            height = double(info.Attributes(i).Value);
        case 'width'
            width = double(info.Attributes(i).Value);
        case 'frames'
            frames = double(info.Attributes(i).Value);
    end
end

fprintf('Dimensions: %d x %d x %d\n', height, width, frames);

% Reshape to 3D
fprintf('Reshaping to 3D...\n');
% The data is stored as [frames, height, width] flattened
data_3d = reshape(data_1d, [frames, height, width]);
data_3d = permute(data_3d, [2, 3, 1]); % Reorder to [height, width, frames]

% Verify the reshape worked
fprintf('Reshaped dimensions: %s\n', mat2str(size(data_3d)));
fprintf('Data range: [%.6f, %.6f]\n', min(data_3d(:)), max(data_3d(:)));

% Create new HDF5 file with 3D data
fprintf('Writing 3D HDF5 file...\n');
if exist(output_path, 'file')
    delete(output_path);
end

% Write the 3D data directly
h5create(output_path, '/dataset2', size(data_3d), 'Datatype', 'single');
h5write(output_path, '/dataset2', single(data_3d));

% Add metadata as attributes
h5writeatt(output_path, '/', 'height', height);
h5writeatt(output_path, '/', 'width', width);
h5writeatt(output_path, '/', 'frames', frames);
h5writeatt(output_path, '/', 'px_per_mm', 1/0.15299877600979192);
h5writeatt(output_path, '/', 'frame_rate', 60);
h5writeatt(output_path, '/', 'scaled_to_crimaldi', true);

fprintf('Created: %s\n', output_path);

% Create symlink in project directory
system('ln -sf /home/snb6/palmer_scratch/plume/smoke_1a_rotated_3d.h5 data/smoke_1a_rotated_3d.h5');
fprintf('Created symlink: data/smoke_1a_rotated_3d.h5\n');
