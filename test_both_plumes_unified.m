addpath('Code');

fprintf('=== Testing Unified HDF5 Access for Both Plumes ===\n\n');

% Test Crimaldi plume
fprintf('1. CRIMALDI PLUME:\n');
crimaldi = load_plume_hdf5_info('data/10302017_10cms_bounded.hdf5', 1/0.74, 15);
fprintf('   File: %s\n', crimaldi.filename);
fprintf('   Dataset: %s\n', crimaldi.dataset);
fprintf('   Dimensions: [%d x %d x %d] = [width x height x frames]\n', crimaldi.dims);
fprintf('   px_per_mm: %.3f\n', crimaldi.px_per_mm);
fprintf('   frame_rate: %d Hz\n', crimaldi.frame_rate);

% Test a few points
fprintf('\n   Testing access points:\n');
test_points = [1 1 1; 100 200 1; 216 406 1; 216 406 3600];
for i = 1:size(test_points, 1)
    try
        val = h5read(crimaldi.filename, crimaldi.dataset, test_points(i,:), [1 1 1]);
        fprintf('   [%3d %3d %4d] = %.6f ✓\n', test_points(i,:), val);
    catch ME
        fprintf('   [%3d %3d %4d] = ERROR: %s\n', test_points(i,:), ME.message);
    end
end

% Test smoke plume  
fprintf('\n2. SMOKE PLUME:\n');
% First check if smoke HDF5 exists
smoke_h5 = 'data/smoke_1a_rotated_3d.h5';
if ~exist(smoke_h5, 'file')
    fprintf('   Smoke HDF5 not found at %s\n', smoke_h5);
    fprintf('   Checking for alternative paths...\n');
    
    % Check alternative locations
    alt_paths = {
        'data/smoke_1a_crimaldi_format.h5',
        'data/smoke_1a_orig_backgroundsubtracted.h5',
        '/home/snb6/palmer_scratch/plume/smoke_1a_crimaldi_format.h5'
    };
    
    for j = 1:length(alt_paths)
        if exist(alt_paths{j}, 'file')
            smoke_h5 = alt_paths{j};
            fprintf('   Found at: %s\n', smoke_h5);
            break;
        end
    end
end

if exist(smoke_h5, 'file')
    try
        smoke = load_plume_hdf5_info(smoke_h5, 6.536, 60);
        fprintf('   File: %s\n', smoke.filename);
        fprintf('   Dataset: %s\n', smoke.dataset);
        fprintf('   Dimensions: [%d x %d x %d] = [width x height x frames]\n', smoke.dims);
        fprintf('   px_per_mm: %.3f\n', smoke.px_per_mm);
        fprintf('   frame_rate: %d Hz\n', smoke.frame_rate);
        
        % Test access
        fprintf('\n   Testing access points:\n');
        test_val = h5read(smoke.filename, smoke.dataset, [1 1 1], [1 1 1]);
        fprintf('   [1 1 1] = %.6f ✓\n', test_val);
        
        % Test middle and end points
        mid_x = round(smoke.dims(1)/2);
        mid_y = round(smoke.dims(2)/2);
        test_val2 = h5read(smoke.filename, smoke.dataset, [mid_x mid_y 1], [1 1 1]);
        fprintf('   [%d %d 1] = %.6f ✓\n', mid_x, mid_y, test_val2);
        
    catch ME
        fprintf('   ERROR loading smoke plume: %s\n', ME.message);
    end
else
    fprintf('   No smoke HDF5 file found\n');
end

fprintf('\n=== Dimension Comparison ===\n');
fprintf('Crimaldi: %d x %d x %d (%.1f x %.1f mm, %d sec @ %d Hz)\n', ...
    crimaldi.dims, crimaldi.dims(1)/crimaldi.px_per_mm/10, ...
    crimaldi.dims(2)/crimaldi.px_per_mm/10, ...
    crimaldi.dims(3)/crimaldi.frame_rate, crimaldi.frame_rate);

if exist('smoke', 'var')
    fprintf('Smoke:    %d x %d x %d (%.1f x %.1f mm, %d sec @ %d Hz)\n', ...
        smoke.dims, smoke.dims(1)/smoke.px_per_mm/10, ...
        smoke.dims(2)/smoke.px_per_mm/10, ...
        smoke.dims(3)/smoke.frame_rate, smoke.frame_rate);
end

exit
