addpath('Code');

% Load the plume
fprintf('Loading HDF5 plume...\n');
plume = load_custom_plume('/home/snb6/palmer_scratch/plume/smoke_1a_orig_backgroundsubtracted_meta.yaml');

% Check plume statistics
fprintf('\nPlume data statistics:\n');
fprintf('  Size: %d x %d x %d\n', size(plume.data));
fprintf('  Min: %.4f, Max: %.4f\n', min(plume.data(:)), max(plume.data(:)));
fprintf('  Mean: %.4f, Std: %.4f\n', mean(plume.data(:)), std(plume.data(:)));

% Check non-zero values
nonzero = plume.data(plume.data > 0);
if ~isempty(nonzero)
    fprintf('\nNon-zero values:\n');
    fprintf('  Count: %d (%.2f%%)\n', length(nonzero), 100*length(nonzero)/numel(plume.data));
    fprintf('  Min: %.4f, Max: %.4f\n', min(nonzero), max(nonzero));
    fprintf('  Mean: %.4f\n', mean(nonzero));
else
    fprintf('\nNo positive values in plume!\n');
end

% Check a few frames
fprintf('\nChecking frames:\n');
for f = [1, 100, 500, 1000]
    frame = plume.data(:,:,f);
    fprintf('  Frame %d: min=%.4f, max=%.4f, mean=%.4f\n', ...
        f, min(frame(:)), max(frame(:)), mean(frame(:)));
end

% Check fly starting positions
fprintf('\nDefault fly starting position for video environment: (0, 0)\n');

exit;
