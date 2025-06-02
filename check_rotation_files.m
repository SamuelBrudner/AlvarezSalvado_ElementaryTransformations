% Check dimensions from scaled and rotated files
base_path = '/home/snb6/palmer_scratch/plume/';
scaled_file = fullfile(base_path, 'smoke_1a_orig_backgroundsubtracted_scaled.h5');
rotated_file = fullfile(base_path, 'smoke_1a_orig_backgroundsubtracted_rotated.h5');

fprintf('Checking HDF5 files before and after rotation...\n\n');

% Check scaled file
fprintf('=== SCALED FILE ===\n');
fprintf('File: %s\n', scaled_file);
try
    info_scaled = h5info(scaled_file, '/dataset1');
    fprintf('Dataset size: %s\n', mat2str(info_scaled.Dataspace.Size));
    
    % Check for attributes
    if ~isempty(info_scaled.Attributes)
        fprintf('Attributes found:\n');
        for i = 1:length(info_scaled.Attributes)
            attr = info_scaled.Attributes(i);
            fprintf('  %s: ', attr.Name);
            if isnumeric(attr.Value)
                fprintf('%d\n', attr.Value);
            else
                disp(attr.Value);
            end
        end
    else
        fprintf('No attributes in dataset\n');
    end
    
    % Check root attributes
    info_root = h5info(scaled_file);
    if ~isempty(info_root.Attributes)
        fprintf('Root attributes:\n');
        for i = 1:length(info_root.Attributes)
            attr = info_root.Attributes(i);
            fprintf('  %s: ', attr.Name);
            disp(attr.Value);
        end
    end
catch ME
    fprintf('Error reading scaled file: %s\n', ME.message);
end

fprintf('\n=== ROTATED FILE ===\n');
fprintf('File: %s\n', rotated_file);
try
    info_rotated = h5info(rotated_file, '/dataset1');
    fprintf('Dataset size: %s\n', mat2str(info_rotated.Dataspace.Size));
    
    % Check for attributes
    if ~isempty(info_rotated.Attributes)
        fprintf('Attributes found:\n');
        for i = 1:length(info_rotated.Attributes)
            attr = info_rotated.Attributes(i);
            fprintf('  %s: ', attr.Name);
            if isnumeric(attr.Value)
                fprintf('%d\n', attr.Value);
            else
                disp(attr.Value);
            end
        end
    else
        fprintf('No attributes in dataset\n');
    end
catch ME
    fprintf('Error reading rotated file: %s\n', ME.message);
end

% Also check the metadata YAML
fprintf('\n=== METADATA FILE ===\n');
meta_file = fullfile(base_path, 'smoke_1a_orig_backgroundsubtracted_meta.yaml');
if exist(meta_file, 'file')
    fprintf('Reading: %s\n', meta_file);
    try
        fid = fopen(meta_file, 'r');
        content = fread(fid, '*char')';
        fclose(fid);
        fprintf('%s\n', content);
    catch
        fprintf('Could not read metadata file\n');
    end
end

% Calculate dimensions from total elements
total_elements = 6768230400;
fprintf('\n=== DIMENSION CALCULATION ===\n');
fprintf('Total elements: %d\n', total_elements);

% If we can get the frame count from somewhere, we can determine h×w
% Let's check a few specific possibilities based on the rotation
test_dims = [
    % frames, height_scaled, width_scaled, height_rotated, width_rotated
    3600, 1880, 1002, 1002, 1880;  % Just a guess
    7200, 940, 1002, 1002, 940;
    10800, 627, 1002, 1002, 627;
    14400, 470, 1002, 1002, 470;
    3600, 1002, 1880, 1880, 1002;  % Opposite orientation
    7200, 1002, 940, 940, 1002;
];

fprintf('\nChecking if any standard configurations match:\n');
for i = 1:size(test_dims, 1)
    f = test_dims(i, 1);
    h = test_dims(i, 2);
    w = test_dims(i, 3);
    if f * h * w == total_elements
        fprintf('MATCH: %d frames, %d×%d (scaled) → %d×%d (rotated)\n', ...
                f, h, w, test_dims(i, 4), test_dims(i, 5));
    end
end
