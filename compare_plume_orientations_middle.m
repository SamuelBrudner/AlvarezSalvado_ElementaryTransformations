% Compare Crimaldi and smoke plume orientations using middle frames
fprintf('Loading and comparing plume orientations (middle frames)...\n\n');

% Add paths
addpath('Code');

% Load Crimaldi plume
fprintf('Loading Crimaldi plume...\n');
crim_path = 'data/10302017_10cms_bounded.hdf5';
crim_data = h5read(crim_path, '/dataset2');
crim_dims = size(crim_data);
fprintf('Crimaldi dimensions: %d x %d x %d\n', crim_dims(1), crim_dims(2), crim_dims(3));

% Load smoke plume (3D version)
fprintf('Loading smoke plume...\n');
smoke_path = 'data/smoke_1a_rotated_3d.h5';
if exist(smoke_path, 'file')
    smoke_data = h5read(smoke_path, '/dataset2');
    smoke_dims = size(smoke_data);
    fprintf('Smoke dimensions: %d x %d x %d\n', smoke_dims(1), smoke_dims(2), smoke_dims(3));
else
    error('3D smoke file not found. Please run convert_smoke_to_3d_fixed.m first.');
end

% Calculate middle frames
crim_middle = round(crim_dims(3) / 2);
smoke_middle = round(smoke_dims(3) / 2);

fprintf('\nUsing middle frames:\n');
fprintf('  Crimaldi: frame %d of %d (%.1f seconds at 15 Hz)\n', ...
        crim_middle, crim_dims(3), crim_middle/15);
fprintf('  Smoke: frame %d of %d (%.1f seconds at 60 Hz)\n', ...
        smoke_middle, smoke_dims(3), smoke_middle/60);

% Extract middle frames
crim_frame = crim_data(:, :, crim_middle);
smoke_frame = smoke_data(:, :, smoke_middle);

% Create figure with better layout
figure('Position', [50, 50, 1600, 900]);

% Plot Crimaldi frame
subplot(2, 3, 1);
imagesc(crim_frame);
colorbar;
caxis([min(crim_frame(:)), prctile(crim_frame(:), 99)]); % Clip at 99th percentile for better contrast
title(sprintf('Crimaldi - Middle Frame (%d)', crim_middle));
xlabel('Width (pixels)');
ylabel('Height (pixels)');
axis image;
colormap(gca, 'hot');
text(10, 30, sprintf('%dx%d', size(crim_frame, 2), size(crim_frame, 1)), ...
     'Color', 'white', 'FontSize', 12, 'BackgroundColor', 'black');

% Plot smoke frame
subplot(2, 3, 2);
imagesc(smoke_frame);
colorbar;
caxis([min(smoke_frame(:)), prctile(smoke_frame(:), 99)]);
title(sprintf('Smoke - Middle Frame (%d)', smoke_middle));
xlabel('Width (pixels)');
ylabel('Height (pixels)');
axis image;
colormap(gca, 'hot');
text(10, 50, sprintf('%dx%d', size(smoke_frame, 2), size(smoke_frame, 1)), ...
     'Color', 'white', 'FontSize', 12, 'BackgroundColor', 'black');

% Plot smoke resized to Crimaldi dimensions
subplot(2, 3, 3);
smoke_resized = imresize(smoke_frame, size(crim_frame));
imagesc(smoke_resized);
colorbar;
caxis([min(smoke_resized(:)), prctile(smoke_resized(:), 99)]);
title('Smoke Resized to Crimaldi Dims');
xlabel('Width (pixels)');
ylabel('Height (pixels)');
axis image;
colormap(gca, 'hot');

% Plot intensity profiles
subplot(2, 3, 4);
% Horizontal profile through center
crim_h_profile = crim_frame(round(size(crim_frame,1)/2), :);
smoke_h_profile = smoke_frame(round(size(smoke_frame,1)/2), :);
plot(1:length(crim_h_profile), crim_h_profile, 'b-', 'LineWidth', 2);
hold on;
% Resample smoke profile to match crimaldi width for comparison
smoke_h_resampled = interp1(1:length(smoke_h_profile), smoke_h_profile, ...
                            linspace(1, length(smoke_h_profile), length(crim_h_profile)));
plot(1:length(crim_h_profile), smoke_h_resampled, 'r-', 'LineWidth', 2);
xlabel('Width (pixels)');
ylabel('Intensity');
title('Horizontal Profile (Center)');
legend('Crimaldi', 'Smoke (resampled)', 'Location', 'best');
grid on;

% Vertical profile through center
subplot(2, 3, 5);
crim_v_profile = crim_frame(:, round(size(crim_frame,2)/2));
smoke_v_profile = smoke_frame(:, round(size(smoke_frame,2)/2));
plot(1:length(crim_v_profile), crim_v_profile, 'b-', 'LineWidth', 2);
hold on;
% Resample smoke profile to match crimaldi height for comparison
smoke_v_resampled = interp1(1:length(smoke_v_profile), smoke_v_profile, ...
                            linspace(1, length(smoke_v_profile), length(crim_v_profile)));
plot(1:length(crim_v_profile), smoke_v_resampled, 'r-', 'LineWidth', 2);
xlabel('Height (pixels)');
ylabel('Intensity');
title('Vertical Profile (Center)');
legend('Crimaldi', 'Smoke (resampled)', 'Location', 'best');
grid on;

% Statistics comparison
subplot(2, 3, 6);
stats_text = sprintf(['Frame Statistics:\n\n' ...
                     'Crimaldi (frame %d):\n' ...
                     '  Size: %d x %d\n' ...
                     '  Range: [%.3f, %.3f]\n' ...
                     '  Mean: %.3f\n' ...
                     '  Std: %.3f\n' ...
                     '  99th pct: %.3f\n\n' ...
                     'Smoke (frame %d):\n' ...
                     '  Size: %d x %d\n' ...
                     '  Range: [%.3f, %.3f]\n' ...
                     '  Mean: %.3f\n' ...
                     '  Std: %.3f\n' ...
                     '  99th pct: %.3f\n\n' ...
                     'Aspect Ratios:\n' ...
                     '  Crimaldi: %.3f\n' ...
                     '  Smoke: %.3f'], ...
                     crim_middle, ...
                     size(crim_frame, 1), size(crim_frame, 2), ...
                     min(crim_frame(:)), max(crim_frame(:)), ...
                     mean(crim_frame(:)), std(crim_frame(:)), ...
                     prctile(crim_frame(:), 99), ...
                     smoke_middle, ...
                     size(smoke_frame, 1), size(smoke_frame, 2), ...
                     min(smoke_frame(:)), max(smoke_frame(:)), ...
                     mean(smoke_frame(:)), std(smoke_frame(:)), ...
                     prctile(smoke_frame(:), 99), ...
                     size(crim_frame,2)/size(crim_frame,1), ...
                     size(smoke_frame,2)/size(smoke_frame,1));

text(0.1, 0.5, stats_text, 'FontSize', 10, 'FontName', 'FixedWidth', ...
     'VerticalAlignment', 'middle');
axis off;

% Add overall title
sgtitle('Plume Orientation Comparison - Middle Frames', 'FontSize', 16);

% Save the figure
print('plume_middle_frame_comparison.png', '-dpng', '-r150');
fprintf('\nSaved comparison plot to plume_middle_frame_comparison.png\n');

% Also create a zoomed-in view of interesting regions
figure('Position', [50, 50, 1200, 600]);

% Find regions with high intensity
[~, crim_max_idx] = max(crim_frame(:));
[crim_max_y, crim_max_x] = ind2sub(size(crim_frame), crim_max_idx);
[~, smoke_max_idx] = max(smoke_frame(:));
[smoke_max_y, smoke_max_x] = ind2sub(size(smoke_frame), smoke_max_idx);

% Define zoom windows (100x100 around max intensity)
crim_zoom_y = max(1, crim_max_y-50):min(size(crim_frame,1), crim_max_y+50);
crim_zoom_x = max(1, crim_max_x-50):min(size(crim_frame,2), crim_max_x+50);
smoke_zoom_y = max(1, smoke_max_y-50):min(size(smoke_frame,1), smoke_max_y+50);
smoke_zoom_x = max(1, smoke_max_x-50):min(size(smoke_frame,2), smoke_max_x+50);

% Plot zoomed regions
subplot(1, 2, 1);
imagesc(crim_frame(crim_zoom_y, crim_zoom_x));
colorbar;
title(sprintf('Crimaldi - Zoomed (around %d,%d)', crim_max_y, crim_max_x));
xlabel('Width (pixels)');
ylabel('Height (pixels)');
axis image;
colormap(gca, 'hot');

subplot(1, 2, 2);
imagesc(smoke_frame(smoke_zoom_y, smoke_zoom_x));
colorbar;
title(sprintf('Smoke - Zoomed (around %d,%d)', smoke_max_y, smoke_max_x));
xlabel('Width (pixels)');
ylabel('Height (pixels)');
axis image;
colormap(gca, 'hot');

sgtitle('High Intensity Regions - Zoomed View', 'FontSize', 14);
print('plume_zoomed_comparison.png', '-dpng', '-r150');
fprintf('Saved zoomed comparison to plume_zoomed_comparison.png\n');

% Check orientation consistency
fprintf('\n=== Orientation Check ===\n');
if (size(crim_frame,1) < size(crim_frame,2)) && (size(smoke_frame,1) > size(smoke_frame,2))
    fprintf('✓ Orientations match: Both are landscape format\n');
    fprintf('  (Crimaldi is naturally landscape, Smoke is landscape after rotation)\n');
elseif (size(crim_frame,1) > size(crim_frame,2)) && (size(smoke_frame,1) < size(smoke_frame,2))
    fprintf('✗ Orientations opposite: One is portrait, one is landscape\n');
else
    fprintf('~ Both have same orientation type\n');
end
