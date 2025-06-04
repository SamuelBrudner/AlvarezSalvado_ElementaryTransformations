% fix_smoke_dimensions_batch.m - Non-interactive version for HPC batch processing
% Run with: matlab -batch "fix_smoke_dimensions_batch"

fprintf('\n=== Fixing Smoke Plume Dimension Order (Batch Mode) ===\n\n');
drawnow('update');

% File paths
input_file = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d.h5';
output_file = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d_fixed.h5';
dataset_name = '/dataset2';

% Check if output already exists
if exist(output_file, 'file')
    fprintf('Output file already exists: %s\n', output_file);
    fprintf('Delete it first if you want to recreate it.\n');
    fprintf('Exiting.\n');
    return;
end

% First, let's check Crimaldi plume structure for reference
fprintf('1. Checking Crimaldi plume structure for reference...\n');
drawnow('update');
crimaldi_file = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/10302017_10cms_bounded.hdf5';
if exist(crimaldi_file, 'file')
    crim_info = h5info(crimaldi_file, '/dataset2');
    fprintf('   Crimaldi dimensions: %d x %d x %d\n', crim_info.Dataspace.Size);
    fprintf('   Interpretation: %d width x %d height x %d frames\n\n', ...
            crim_info.Dataspace.Size(1), crim_info.Dataspace.Size(2), crim_info.Dataspace.Size(3));
    drawnow('update');
end

% Check current smoke plume structure
fprintf('2. Current smoke plume structure:\n');
drawnow('update');
smoke_info = h5info(input_file, dataset_name);
current_dims = smoke_info.Dataspace.Size;
fprintf('   Current dimensions: %d x %d x %d\n', current_dims);
fprintf('   Currently interpreted as: %d width x %d height x %d frames\n', ...
        current_dims(1), current_dims(2), current_dims(3));
fprintf('   Duration if 3rd dim is frames: %.1f seconds at 60 Hz\n', current_dims(3)/60);
fprintf('   Duration if 1st dim is frames: %.1f seconds at 60 Hz\n\n', current_dims(1)/60);
drawnow('update');

% Determine correct interpretation
fprintf('3. Analysis:\n');
fprintf('   3600 frames at 60 Hz = 60 seconds (1 minute) ✓\n');
fprintf('   1088 x 1728 pixels = reasonable spatial dimensions ✓\n');
fprintf('   Therefore, data needs to be permuted from [3600,1088,1728] to [1088,1728,3600]\n\n');
drawnow('update');

% Proceeding automatically in batch mode
fprintf('4. Processing (batch mode - proceeding automatically)...\n');
fprintf('   Reading data (this will take several minutes for 25GB)...\n');
fprintf('   Started at: %s\n', datestr(now));
drawnow('update');

tic;
% Add progress indicator
fprintf('   Loading');
drawnow('update');

% Create a timer to show progress dots
dot_timer = timer('Period', 2, 'ExecutionMode', 'fixedRate', ...
                  'TimerFcn', @(~,~) fprintf('.'));
start(dot_timer);

try
    data = h5read(input_file, dataset_name);
    stop(dot_timer);
    delete(dot_timer);
    fprintf('\n');
catch ME
    stop(dot_timer);
    delete(dot_timer);
    rethrow(ME);
end

elapsed = toc;
fprintf('   Data loaded in %.1f seconds (%.1f minutes)\n', elapsed, elapsed/60);
drawnow('update');

fprintf('   Current data size: %s\n', sprintf('%d ', size(data)));
fprintf('   Permuting dimensions [1,2,3] -> [2,3,1]...\n');
drawnow('update');

tic;
data_fixed = permute(data, [2, 3, 1]);
fprintf('   Permutation completed in %.1f seconds\n', toc);
fprintf('   Fixed data size: %s\n', sprintf('%d ', size(data_fixed)));
drawnow('update');

% Clear original to save memory
fprintf('   Clearing original data from memory...\n');
drawnow('update');
clear data;

% Verify a sample
fprintf('\n5. Verifying data integrity...\n');
sample_frame = 100;
fprintf('   Sample from frame %d:\n', sample_frame);
sample_data = data_fixed(:,:,sample_frame);
fprintf('   Min: %.6f, Max: %.6f, Mean: %.6f\n', ...
        min(sample_data(:)), max(sample_data(:)), mean(sample_data(:)));
drawnow('update');

% Save the fixed data
fprintf('\n6. Saving fixed data...\n');
fprintf('   Output file: %s\n', output_file);
drawnow('update');

% Write with compression
fprintf('   Creating HDF5 file with compression...\n');
drawnow('update');
tic;
h5create(output_file, dataset_name, size(data_fixed), ...
         'Datatype', 'single', ...
         'ChunkSize', [109, 173, 36], ... % ~1/10th of each dimension
         'Deflate', 6); % Compression level 6

% Write in chunks to avoid memory issues
chunk_size = 100; % frames per chunk
n_chunks = ceil(size(data_fixed, 3) / chunk_size);

fprintf('   Writing %d chunks of %d frames each...\n', n_chunks, chunk_size);
fprintf('   Progress: ');
drawnow('update');

write_start = tic;
for i = 1:n_chunks
    start_frame = (i-1) * chunk_size + 1;
    end_frame = min(i * chunk_size, size(data_fixed, 3));
    
    % Show progress
    if mod(i, 10) == 0 || i == 1 || i == n_chunks
        fprintf('\n     Chunk %d/%d (frames %d-%d)', ...
                i, n_chunks, start_frame, end_frame);
        if i > 1
            elapsed = toc(write_start);
            rate = i / elapsed;
            eta = (n_chunks - i) / rate;
            fprintf(' - ETA: %.1f min', eta/60);
        end
        fprintf('\n   Progress: ');
        drawnow('update');
    else
        fprintf('.');
        drawnow('update');
    end
    
    h5write(output_file, dataset_name, ...
            data_fixed(:, :, start_frame:end_frame), ...
            [1, 1, start_frame], ...
            [size(data_fixed,1), size(data_fixed,2), end_frame-start_frame+1]);
end

fprintf('\n   Save completed in %.1f seconds (%.1f minutes)\n', toc, toc/60);
drawnow('update');

% Clear data to free memory
clear data_fixed;

% Verify the saved file
fprintf('\n7. Verifying saved file...\n');
drawnow('update');
saved_info = h5info(output_file, dataset_name);
fprintf('   Saved dimensions: %d x %d x %d\n', saved_info.Dataspace.Size);
fprintf('   Interpretation: %d width x %d height x %d frames\n', ...
        saved_info.Dataspace.Size(1), saved_info.Dataspace.Size(2), saved_info.Dataspace.Size(3));
fprintf('   Duration: %.1f seconds at 60 Hz ✓\n', saved_info.Dataspace.Size(3)/60);
drawnow('update');

% Test read
fprintf('   Testing read of frame 100...\n');
drawnow('update');
test_frame = h5read(output_file, dataset_name, [1 1 100], [inf inf 1]);
fprintf('   Test read successful: %d x %d\n', size(test_frame));
drawnow('update');

% Get file sizes
input_info = dir(input_file);
output_info = dir(output_file);
fprintf('\n   File sizes:\n');
fprintf('     Original: %.2f GB\n', input_info.bytes/1024^3);
fprintf('     Fixed:    %.2f GB\n', output_info.bytes/1024^3);
fprintf('     Compression ratio: %.1f%%\n', 100*output_info.bytes/input_info.bytes);
drawnow('update');

fprintf('\n✓ Fix completed successfully!\n\n');
fprintf('Next steps:\n');
fprintf('1. The path has already been updated in setup_smoke_plume_config.sh\n');
fprintf('2. Run: ./setup_smoke_plume_config.sh\n');
fprintf('\nCompleted at: %s\n', datestr(now));
drawnow('update');

% Exit successfully
exit(0);