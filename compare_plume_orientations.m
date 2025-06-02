% Compare Crimaldi and smoke plume orientations
fprintf('Loading and comparing plume orientations...\n\n');

% Add paths
addpath('Code');

% Frame to compare (middle of the dataset)
frame_idx = 1800;  % 30 seconds in

% Load Crimaldi plume
fprintf('Loading Crimaldi plume...\n');
crim_path = 'data/10302017_10cms_bounded.hdf5';
crim_data = h5read(crim_path, '/dataset2');  % Note: Crimaldi uses dataset2
fprintf('Crimaldi dimensions: %s\n', mat2str(size(crim_data)));

% Load smoke plume (3D version)
fprintf('Loading smoke plume...\n');
smoke_path = 'data/smoke_1a_rotated_3d.h5';
if exist(smoke_path, 'file')
    smoke_data = h5read(smoke_path, '/dataset2');
    fprintf('Smoke dimensions: %s\n', mat2str(size(smoke_data)));
else
    % Fallback to loading 1D and reshaping
    fprintf('3D file not found, using 1D version...\n');
    smoke_path = 'data/smoke_1a_rotated.h5';
    data_1d = h5read(smoke_path, '/dataset1');
    % Reshape using known dimensions
    smoke_data = reshape(data_1d, [3600, 1728, 1088]);
    smoke_data = permute(smoke_data, [2, 3, 1]);
    fprintf('Smoke dimensions after reshape: %s\n', mat2str(size(smoke_data)));
end

% Extract frames
crim_frame = crim_data(:, :, frame_idx);
smoke_frame = smoke_data(:, :, frame_idx);

% Create figure
figure('Position', [100, 100, 1400, 600]);

% Plot Crimaldi frame
subplot(1, 3, 1);
imagesc(crim_frame);
colorbar;
title(sprintf('Crimaldi Frame %d', frame_idx));
xlabel('Width (pixels)');
ylabel('Height (pixels)');
axis image;
text(10, 30, sprintf('Size: %dx%d', size(crim_frame, 2), size(crim_frame, 1)), ...
     'Color', 'white', 'FontSize', 12, 'BackgroundColor', 'black');

% Plot smoke frame
subplot(1, 3, 2);
imagesc(smoke_frame);
colorbar;
title(sprintf('Smoke Frame %d', frame_idx));
xlabel('Width (pixels)');
ylabel('Height (pixels)');
axis image;
text(10, 50, sprintf('Size: %dx%d', size(smoke_frame, 2), size(smoke_frame, 1)), ...
     'Color', 'white', 'FontSize', 12, 'BackgroundColor', 'black');

% Plot difference in sizes
subplot(1, 3, 3);
% Resize smoke to match Crimaldi dimensions for comparison
smoke_resized = imresize(smoke_frame, size(crim_frame));
diff_img = smoke_resized - crim_frame;
imagesc(diff_img);
colorbar;
title('Difference (smoke resized - crimaldi)');
xlabel('Width (pixels)');
ylabel('Height (pixels)');
axis image;

% Add overall title
sgtitle('Plume Orientation Comparison', 'FontSize', 16);

% Print statistics
fprintf('\n=== Frame Statistics ===\n');
fprintf('Crimaldi frame %d:\n', frame_idx);
fprintf('  Size: %d x %d\n', size(crim_frame, 1), size(crim_frame, 2));
fprintf('  Range: [%.4f, %.4f]\n', min(crim_frame(:)), max(crim_frame(:)));
fprintf('  Mean: %.4f, Std: %.4f\n', mean(crim_frame(:)), std(crim_frame(:)));

fprintf('\nSmoke frame %d:\n', frame_idx);
fprintf('  Size: %d x %d\n', size(smoke_frame, 1), size(smoke_frame, 2));
fprintf('  Range: [%.4f, %.4f]\n', min(smoke_frame(:)), max(smoke_frame(:)));
fprintf('  Mean: %.4f, Std: %.4f\n', mean(smoke_frame(:)), std(smoke_frame(:)));

% Check orientation by looking at intensity profiles
fprintf('\n=== Checking Orientation ===\n');
% Take horizontal and vertical slices through the center
crim_h_slice = crim_frame(round(size(crim_frame,1)/2), :);
crim_v_slice = crim_frame(:, round(size(crim_frame,2)/2));
smoke_h_slice = smoke_frame(round(size(smoke_frame,1)/2), :);
smoke_v_slice = smoke_frame(:, round(size(smoke_frame,2)/2));

fprintf('Crimaldi - horizontal variance: %.4f, vertical variance: %.4f\n', ...
        var(crim_h_slice), var(crim_v_slice));
fprintf('Smoke - horizontal variance: %.4f, vertical variance: %.4f\n', ...
        var(smoke_h_slice), var(smoke_v_slice));

% Save the figure
print('plume_orientation_comparison.png', '-dpng', '-r150');
fprintf('\nSaved comparison plot to plume_orientation_comparison.png\n');

% Also create a simple animation preview
figure('Position', [100, 100, 800, 800]);
for t = 1:10:100  % First 100 frames, every 10th
    clf;
    subplot(2,1,1);
    imagesc(crim_data(:,:,t));
    title(sprintf('Crimaldi - Frame %d', t));
    axis image;
    colorbar;
    
    subplot(2,1,2);
    imagesc(smoke_data(:,:,t));
    title(sprintf('Smoke - Frame %d', t));
    axis image;
    colorbar;
    
    drawnow;
    pause(0.1);
end