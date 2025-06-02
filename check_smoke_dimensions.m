% Check smoke HDF5 dimensions
h5_path = 'data/smoke_1a_rotated.h5';

% Check if file exists
if ~exist(h5_path, 'file')
    error('File not found: %s', h5_path);
end

% Get info about the dataset
info = h5info(h5_path, '/dataset1');
total_elements = info.Dataspace.Size;
fprintf('Total elements: %d\n', total_elements);
fprintf('Total elements in scientific notation: %.2e\n', total_elements);

% Common video dimensions to try
% 6,768,230,400 elements - let's find possible dimensions
fprintf('\nSearching for possible dimensions...\n');
possible_dims = [];

% Try common aspect ratios and frame counts
% Standard aspect ratios: 4:3, 16:9, etc.
for frames = [3600, 4800, 6000, 7200, 9000, 10800]  % Common frame counts
    remainder = total_elements / frames;
    if mod(total_elements, frames) == 0
        % Now find height/width that multiply to remainder
        for h = 100:50:1000
            if mod(remainder, h) == 0
                w = remainder / h;
                if w > 100 && w < 2000  % Reasonable width
                    possible_dims = [possible_dims; h, w, frames];
                end
            end
        end
    end
end

% Also try factorizing differently
fprintf('\nPossible dimensions (height, width, frames):\n');
fprintf('Height\tWidth\tFrames\tAspect Ratio\n');
fprintf('------\t-----\t------\t------------\n');
for i = 1:size(possible_dims, 1)
    h = possible_dims(i, 1);
    w = possible_dims(i, 2);
    f = possible_dims(i, 3);
    aspect = w/h;
    fprintf('%d\t%d\t%d\t%.3f\n', h, w, f, aspect);
end

% Try to read a small sample to verify
fprintf('\nAttempting to read first element...\n');
try
    first_val = h5read(h5_path, '/dataset1', 1, 1);
    fprintf('First value: %f\n', first_val);
catch ME
    fprintf('Error reading data: %s\n', ME.message);
end
