% Convert smoke HDF5 from 1D to proper 3D format with correct reshape order
fprintf('Converting smoke HDF5 to 3D format (corrected)...\n');

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

% Reshape to 3D using the correct order
fprintf('Reshaping to 3D using [frames, height, width] order...\n');
% The data is stored in [frames, height, width] order when flattened
data_3d_temp = reshape(data_1d, [frames, height, width]);

% Now rearrange to MATLAB's preferred [height, width, frames] order
data_3d = permute(data_3d_temp, [2, 3, 1]);

% Verify the reshape worked
fprintf('Final dimensions: %s\n', mat2str(size(data_3d)));
fprintf('Data range: [%.6f, %.6f]\n', min(data_3d(:)), max(data_3d(:)));

% Quick visual check - display middle frame
middle_frame = data_3d(:, :, round(frames/2));
fprintf('Middle frame size: %s\n', mat2str(size(middle_frame)));
fprintf('Middle frame intensity range: [%.6f, %.6f]\n', ...
        min(middle_frame(:)), max(middle_frame(:)));

% Create new HDF5 file with 3D data
fprintf('Writing 3D HDF5 file...\n');
if exist(output_path, 'file')
    delete(output_path);
end

% Write the 3D data in MATLAB order [height, width, frames]
h5create(output_path, '/dataset2', size(data_3d), 'Datatype', 'single');
h5write(output_path, '/dataset2', single(data_3d));

% Add metadata as attributes
h5writeatt(output_path, '/', 'height', uint32(height));
h5writeatt(output_path, '/', 'width', uint32(width));
h5writeatt(output_path, '/', 'frames', uint32(frames));
h5writeatt(output_path, '/', 'px_per_mm', 1/0.15299877600979192);
h5writeatt(output_path, '/', 'frame_rate', uint32(60));
h5writeatt(output_path, '/', 'scaled_to_crimaldi', uint8(1));

fprintf('Created: %s\n', output_path);

% Create symlink in project directory
[status, msg] = system('ln -sf /home/snb6/palmer_scratch/plume/smoke_1a_rotated_3d.h5 data/smoke_1a_rotated_3d.h5');
if status == 0
    fprintf('Created symlink: data/smoke_1a_rotated_3d.h5\n');
else
    fprintf('Warning: Could not create symlink: %s\n', msg);
end

% Verify the file by reading it back
fprintf('\nVerifying the created file...\n');
test_data = h5read(output_path, '/dataset2');
fprintf('Read back dimensions: %s\n', mat2str(size(test_data)));
fprintf('Read back range: [%.6f, %.6f]\n', min(test_data(:)), max(test_data(:)));
fprintf('Success! The smoke plume has been converted to 3D format.\n');
