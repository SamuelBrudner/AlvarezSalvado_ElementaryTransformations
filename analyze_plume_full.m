% Comprehensive analysis script
mm_per_pixel = 0.15299877600979192;
fps = 60.0;
plume_file = 'data/plumes/smoke_1a_orig_backgroundsubtracted_rotated.h5';
config_file = 'configs/plumes/smoke_1a_backgroundsubtracted.json';

fprintf('Loading file info...\n');

% Get file info
info = h5info(plume_file);

% Find dataset
dataset_name = '/dataset2';
if ~any(strcmp({info.Datasets.Name}, 'dataset2'))
    if ~isempty(info.Datasets)
        dataset_name = ['/' info.Datasets(1).Name];
    end
end

% Get dimensions
ds_info = h5info(plume_file, dataset_name);
width = ds_info.Dataspace.Size(1);
height = ds_info.Dataspace.Size(2);
n_frames = ds_info.Dataspace.Size(3);

fprintf('  Dimensions: %d x %d x %d\n', width, height, n_frames);

% Sample 100 random frames
fprintf('Sampling 100 random frames...\n');
rng(42);
sample_indices = sort(randperm(n_frames, min(100, n_frames)));

all_values = [];
mean_map = zeros(width, height);

for i = 1:length(sample_indices)
    if mod(i, 20) == 0
        fprintf('  %d/%d frames\n', i, length(sample_indices));
    end
    
    frame = h5read(plume_file, dataset_name, ...
                   [1 1 sample_indices(i)], [inf inf 1]);
    
    all_values = [all_values; frame(:)];
    
    if i <= 20
        mean_map = mean_map + double(frame);
    end
end

mean_map = mean_map / min(20, length(sample_indices));

% Calculate statistics
data_min = min(all_values);
data_max = max(all_values);
data_mean = mean(all_values);
data_std = std(all_values);
pct_zeros = sum(all_values == 0) / length(all_values) * 100;

fprintf('\nIntensity statistics:\n');
fprintf('  Range: [%.6f, %.6f]\n', data_min, data_max);
fprintf('  Mean: %.6f (±%.6f)\n', data_mean, data_std);
fprintf('  Zeros: %.1f%%\n', pct_zeros);

% Find source position
[max_val, max_idx] = max(mean_map(:));
[max_x, max_y] = ind2sub(size(mean_map), max_idx);

center_x_px = width / 2;
center_y_px = height / 2;
source_x_cm = (max_x - center_x_px) * mm_per_pixel / 10;
source_y_cm = -(max_y - center_y_px) * mm_per_pixel / 10;

fprintf('\nSource position:\n');
fprintf('  Pixels: (%d, %d)\n', max_x, max_y);
fprintf('  Centered cm: (%.2f, %.2f)\n', source_x_cm, source_y_cm);

% Calculate arena bounds
arena_width_cm = width * mm_per_pixel / 10;
arena_height_cm = height * mm_per_pixel / 10;

% Calculate scaling
temporal_scale = fps / 15.0;
spatial_scale = mm_per_pixel / 0.74;

% Determine beta
if data_max <= 1.0 && data_min >= 0
    beta_suggestion = 0.01;
else
    beta_suggestion = data_mean * 0.1;
end

% Save results to file
fid = fopen('analysis_results.txt', 'w');
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
