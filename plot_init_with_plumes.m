% plot_init_with_plumes.m - Plot initialization zones with actual plume data

fprintf('=== Plotting Initialization Zones with Plume Data ===\n\n');

%% Load configs
crim_cfg = jsondecode(fileread('configs/plumes/crimaldi_10cms_bounded.json'));
smoke_cfg = jsondecode(fileread('configs/plumes/smoke_1a_backgroundsubtracted.json'));

% Get initialization parameters
init_x = crim_cfg.simulation.agent_initialization.x_range_cm;
init_y = crim_cfg.simulation.agent_initialization.y_range_cm;

%% Load plume data (middle frame)
fprintf('Loading plume data...\n');

% Crimaldi
crim_info = h5info(crim_cfg.data_path.path, crim_cfg.data_path.dataset_name);
middle_frame = round(crim_info.Dataspace.Size(3) / 2);
crim_data = h5read(crim_cfg.data_path.path, crim_cfg.data_path.dataset_name, ...
                   [1 1 middle_frame], [crim_info.Dataspace.Size(1:2), 1]);
crim_data = squeeze(crim_data)';  % Transpose for correct orientation

% Smoke
smoke_info = h5info(smoke_cfg.data_path.path, smoke_cfg.data_path.dataset_name);
smoke_data = h5read(smoke_cfg.data_path.path, smoke_cfg.data_path.dataset_name, ...
                    [1 1 middle_frame], [smoke_info.Dataspace.Size(1:2), 1]);
smoke_data = squeeze(smoke_data)';

%% Calculate coordinate arrays
% Crimaldi
crim_x = linspace(crim_cfg.spatial.arena_bounds.x_min, ...
                  crim_cfg.spatial.arena_bounds.x_max, ...
                  crim_cfg.spatial.resolution.width);
crim_y = linspace(crim_cfg.spatial.arena_bounds.y_max, ...
                  crim_cfg.spatial.arena_bounds.y_min, ...
                  crim_cfg.spatial.resolution.height);

% Smoke
smoke_x = linspace(smoke_cfg.spatial.arena_bounds.x_min, ...
                   smoke_cfg.spatial.arena_bounds.x_max, ...
                   smoke_cfg.spatial.resolution.width);
smoke_y = linspace(smoke_cfg.spatial.arena_bounds.y_max, ...
                   smoke_cfg.spatial.arena_bounds.y_min, ...
                   smoke_cfg.spatial.resolution.height);

%% Create figure
figure('Position', [100 100 1200 550]);

%% Plot Crimaldi
subplot(1,2,1);

% Plot plume data
imagesc(crim_x, crim_y, crim_data);
set(gca, 'YDir', 'normal');
colormap(hot);
caxis([0, prctile(crim_data(:), 95)]);  % Adjust color scale for visibility
hold on;

% Add semi-transparent overlay to make annotations visible
h = patch([-10 10 10 -10], [-32 -32 2 2], 'w');
set(h, 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Arena boundary
rectangle('Position', [crim_cfg.spatial.arena_bounds.x_min, ...
                      crim_cfg.spatial.arena_bounds.y_min, ...
                      crim_cfg.spatial.arena_bounds.x_max - crim_cfg.spatial.arena_bounds.x_min, ...
                      crim_cfg.spatial.arena_bounds.y_max - crim_cfg.spatial.arena_bounds.y_min], ...
          'EdgeColor', 'blue', 'LineWidth', 2.5);

% Original init zone (for reference)
rectangle('Position', [-8, -30, 16, 5], ...
          'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 1.5, 'LineStyle', '--');
text(0, -27.5, 'Original Init', 'Color', [0.3 0.3 0.3], ...
     'HorizontalAlignment', 'center', 'FontSize', 10, 'BackgroundColor', [1 1 1 0.7]);

% New init zone
rectangle('Position', [init_x(1), init_y(1), diff(init_x), diff(init_y)], ...
          'EdgeColor', 'yellow', 'LineWidth', 3);
text(0, mean(init_y), 'New Init Zone', 'Color', [0.8 0.8 0], ...
     'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
     'BackgroundColor', [0 0 0 0.7]);

% Source
theta = linspace(0, 2*pi, 100);
plot(2*cos(theta), 2*sin(theta), 'Color', [0 1 0], 'LineWidth', 3);
plot(0, 0, 'g*', 'MarkerSize', 20, 'LineWidth', 2);
text(0, 1.2, 'Source', 'Color', [0 1 0], 'HorizontalAlignment', 'center', ...
     'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);

% Y=0 line
plot([-8 8], [0 0], 'k--', 'LineWidth', 1.5);
text(7, 0.5, 'Y=0', 'HorizontalAlignment', 'right', 'FontSize', 10);

% Distance annotation
quiver(-9, init_y(1), 0, diff(init_y), 0, 'k', 'LineWidth', 2, ...
       'MaxHeadSize', 0.3, 'ShowArrowHead', 'off');
quiver(-9, init_y(2), 0, -diff(init_y), 0, 'k', 'LineWidth', 2, ...
       'MaxHeadSize', 0.3, 'ShowArrowHead', 'off');
text(-9.5, mean(init_y), sprintf('%.0f cm', abs(diff(init_y))), ...
     'Rotation', 90, 'HorizontalAlignment', 'center', 'FontSize', 10, ...
     'BackgroundColor', 'white');

title('Crimaldi Arena', 'FontSize', 14);
xlabel('X (cm)'); ylabel('Y (cm)');
axis equal;
xlim([-10, 10]); ylim([-32, 2]);
grid on;
set(gca, 'GridAlpha', 0.3);

%% Plot Smoke
subplot(1,2,2);

% Plot plume data
imagesc(smoke_x, smoke_y, smoke_data);
set(gca, 'YDir', 'normal');
colormap(hot);
caxis([0, prctile(smoke_data(:), 95)]);  % Adjust color scale for visibility
hold on;

% Add semi-transparent overlay
h = patch([-10 10 10 -10], [-28 -28 2 2], 'w');
set(h, 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Arena boundary
rectangle('Position', [smoke_cfg.spatial.arena_bounds.x_min, ...
                      smoke_cfg.spatial.arena_bounds.y_min, ...
                      smoke_cfg.spatial.arena_bounds.x_max - smoke_cfg.spatial.arena_bounds.x_min, ...
                      smoke_cfg.spatial.arena_bounds.y_max - smoke_cfg.spatial.arena_bounds.y_min], ...
          'EdgeColor', 'blue', 'LineWidth', 2.5);

% New init zone (exactly at bottom)
rectangle('Position', [init_x(1), init_y(1), diff(init_x), diff(init_y)], ...
          'EdgeColor', 'yellow', 'LineWidth', 3);
text(0, mean(init_y), 'New Init Zone', 'Color', [0.8 0.8 0], ...
     'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
     'BackgroundColor', [0 0 0 0.7]);

% Source (middle of arena)
smoke_source_y = smoke_cfg.simulation.source_position.y_cm;
plot(2*cos(theta), smoke_source_y + 2*sin(theta), 'Color', [0 1 0], 'LineWidth', 3);
plot(0, smoke_source_y, 'g*', 'MarkerSize', 20, 'LineWidth', 2);
text(0, smoke_source_y + 1.2, 'Source', 'Color', [0 1 0], ...
     'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', ...
     'BackgroundColor', [0 0 0 0.7]);

% Y=0 line
plot([-8.3 8.3], [0 0], 'k--', 'LineWidth', 1.5);
text(7, 0.5, 'Y=0', 'HorizontalAlignment', 'right', 'FontSize', 10);

title('Smoke Arena', 'FontSize', 14);
xlabel('X (cm)'); ylabel('Y (cm)');
axis equal;
xlim([-10, 10]); ylim([-28, 2]);
grid on;
set(gca, 'GridAlpha', 0.3);

%% Add shared colorbar
h = colorbar('Position', [0.93 0.3 0.02 0.4]);
ylabel(h, 'Odor Concentration', 'FontSize', 11);

%% Overall title
sgtitle('Updated Initialization Zones with Plume Data', 'FontSize', 16);

%% Add info box
info_text = sprintf(['Initialization: X ∈ [%.0f, %.0f], Y ∈ [%.1f, %.1f] cm\n' ...
                     'Crimaldi: %.0f Hz, %.3f mm/px, Source at (0, 0)\n' ...
                     'Smoke: %.0f Hz, %.3f mm/px, Source at (0, %.1f)'], ...
                    init_x(1), init_x(2), init_y(1), init_y(2), ...
                    crim_cfg.temporal.frame_rate, crim_cfg.spatial.mm_per_pixel, ...
                    smoke_cfg.temporal.frame_rate, smoke_cfg.spatial.mm_per_pixel, ...
                    smoke_source_y);
annotation('textbox', [0.35 0.02 0.3 0.08], 'String', info_text, ...
           'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
           'BackgroundColor', 'white', 'EdgeColor', 'black', 'FontSize', 9);

%% Save figure
saveas(gcf, 'results/init_zones_with_plumes.pdf');
fprintf('\n✓ Saved figure to: results/init_zones_with_plumes.png\n');

%% Print plume statistics
fprintf('\nPlume statistics:\n');
fprintf('Crimaldi: min=%.3f, max=%.3f, mean=%.3f\n', ...
        min(crim_data(:)), max(crim_data(:)), mean(crim_data(:)));
fprintf('Smoke: min=%.3f, max=%.3f, mean=%.3f\n', ...
        min(smoke_data(:)), max(smoke_data(:)), mean(smoke_data(:)));