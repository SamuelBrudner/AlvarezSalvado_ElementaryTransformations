% Smoke plume analysis script
fprintf('\n=== MATLAB STARTED SUCCESSFULLY ===\n');
fprintf('Time: %s\n', datestr(now));
fprintf('MATLAB version: %s\n', version);
fprintf('Working directory: %s\n\n', pwd);

try
    % Change to project root
    cd('/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations');
    fprintf('Changed to project root: %s\n', pwd);
    
    % Check Code directory
    if ~exist('Code', 'dir')
        error('Code directory not found in %s', pwd);
    end
    
    % Add to path
    addpath(genpath('Code'));
    fprintf('Added Code directory to path\n');
    
    % Parameters from shell script
    mm_per_pixel = 0.15299877600979192;
    fps = 60.0;
    n_sample_frames = 100;
    
    % File info
    plume_file = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d_fixed.h5';
    dataset_name = '/dataset2';
    
    fprintf('\nConfiguration:\n');
    fprintf('  mm_per_pixel: %.6f\n', mm_per_pixel);
    fprintf('  fps: %.1f Hz\n', fps);
    fprintf('  HDF5 file: %s\n', plume_file);
    fprintf('  Dataset: %s\n', dataset_name);
    fprintf('  Sampling: %d frames\n', n_sample_frames);
    
    % Check file exists
    if ~exist(plume_file, 'file')
        error('HDF5 file not found: %s', plume_file);
    end
    
    % Get file info
    fprintf('\nReading HDF5 file info (may take time for large files)...\n');
    tic;
    info = h5info(plume_file);
    fprintf('  File info retrieved in %.1f seconds\n', toc);
    
    % Get dataset info
    ds_info = h5info(plume_file, dataset_name);
    width = ds_info.Dataspace.Size(1);
    height = ds_info.Dataspace.Size(2);
    n_frames = ds_info.Dataspace.Size(3);
    
    fprintf('  Dimensions: %d x %d x %d\n', width, height, n_frames);
    
    % Sample frames
    fprintf('\nSampling %d frames...\n', n_sample_frames);
    rng(42);
    sample_indices = sort(randperm(n_frames, min(n_sample_frames, n_frames)));
    
    all_values = [];
    mean_map = zeros(width, height);
    
    fprintf('  Progress: ');
    for i = 1:length(sample_indices)
        if mod(i, max(1, round(n_sample_frames/10))) == 0
            fprintf('%d%% ', round(i/length(sample_indices)*100));
        end
        
        frame = h5read(plume_file, dataset_name, [1 1 sample_indices(i)], [inf inf 1]);
        all_values = [all_values; frame(:)];
        
        if i <= 5  % Use first 5 frames for mean map
            mean_map = mean_map + double(frame);
        end
    end
    fprintf('Done!\n');
    
    mean_map = mean_map / min(5, length(sample_indices));
    
    % Calculate statistics
    data_min = min(all_values);
    data_max = max(all_values);
    data_mean = mean(all_values);
    data_std = std(all_values);
    
    % Find source position
    [~, max_idx] = max(mean_map(:));
    [max_x, max_y] = ind2sub(size(mean_map), max_idx);
    
    center_x_px = width / 2;
    center_y_px = height / 2;
    source_x_cm = (max_x - center_x_px) * mm_per_pixel / 10;
    source_y_cm = -(max_y - center_y_px) * mm_per_pixel / 10;
    
    % Calculate arena bounds
    arena_width_cm = width * mm_per_pixel / 10;
    arena_height_cm = height * mm_per_pixel / 10;
    
    % Calculate scaling
    temporal_scale = fps / 15.0;  % Relative to Crimaldi at 15 Hz
    spatial_scale = mm_per_pixel / 0.74;  % Relative to Crimaldi at 0.74 mm/px
    
    % Determine beta
    if data_max <= 1.0 && data_min >= 0
        beta_suggestion = 0.01;
    else
        beta_suggestion = data_mean * 0.1;
    end
    
    % Save results
    results_file = fullfile('/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/temp_matlab_339442', 'analysis_results.txt');
    fid = fopen(results_file, 'w');
    fprintf(fid, 'width=%d\n', width);
    fprintf(fid, 'height=%d\n', height);
    fprintf(fid, 'frames=%d\n', n_frames);
    fprintf(fid, 'dataset=%s\n', dataset_name);
    fprintf(fid, 'data_min=%.6f\n', data_min);
    fprintf(fid, 'data_max=%.6f\n', data_max);
    fprintf(fid, 'data_mean=%.6f\n', data_mean);
    fprintf(fid, 'data_std=%.6f\n', data_std);
    fprintf(fid, 'source_x_cm=%.3f\n', source_x_cm);
    fprintf(fid, 'source_y_cm=%.3f\n', source_y_cm);
    fprintf(fid, 'arena_width_cm=%.3f\n', arena_width_cm);
    fprintf(fid, 'arena_height_cm=%.3f\n', arena_height_cm);
    fprintf(fid, 'temporal_scale=%.3f\n', temporal_scale);
    fprintf(fid, 'spatial_scale=%.3f\n', spatial_scale);
    fprintf(fid, 'beta_suggestion=%.6f\n', beta_suggestion);
    fprintf(fid, 'normalized=%d\n', data_max <= 1.0 && data_min >= 0);
    fclose(fid);
    
    fprintf('\n✓ Analysis complete\n');
    exit(0);
    
catch ME
    fprintf('\nERROR: %s\n', ME.message);
    exit(1);
end
