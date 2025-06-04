% plot_both_plumes.m - Visualize both plumes with initialization zones

fprintf('=== Plotting Both Plumes ===\n\n');

%% Load configs
crim_cfg = jsondecode(fileread('configs/plumes/crimaldi_10cms_bounded.json'));
smoke_cfg = jsondecode(fileread('configs/plumes/smoke_1a_backgroundsubtracted.json'));

%% Load plume data (middle frame)
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
figure('Position', [100 100 1400 600]);

%% Plot Crimaldi
subplot(1,2,1);
imagesc(crim_x, crim_y, crim_data);
set(gca, 'YDir', 'normal');
colormap(hot);
cbar = colorbar();
ylabel(cbar, 'Odor Concentration');
hold on;

% Arena boundary
rectangle('Position', [crim_cfg.spatial.arena_bounds.x_min, ...
                      crim_cfg.spatial.arena_bounds.y_min, ...
                      crim_cfg.spatial.arena_bounds.x_max - crim_cfg.spatial.arena_bounds.x_min, ...
                      crim_cfg.spatial.arena_bounds.y_max - crim_cfg.spatial.arena_bounds.y_min], ...
          'EdgeColor', 'blue', 'LineWidth', 2, 'LineStyle', '--');

% Model's hardcoded initialization zone
init_x = [-8, 8];
init_y = [-30, -25];
rectangle('Position', [init_x(1), init_y(1), init_x(2)-init_x(1), init_y(2)-init_y(1)], ...
          'EdgeColor', 'yellow', 'LineWidth', 3);
text(mean(init_x), mean(init_y), 'Init Zone', ...
     'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'center');

% Mark Y=0 (top)
plot([-8 8], [0 0], 'g--', 'LineWidth', 2);
text(0, 1, 'Y=0 (top)', 'Color', 'green', 'FontSize', 10, 'HorizontalAlignment', 'center');

% Success zone (2cm radius at origin)
theta = linspace(0, 2*pi, 100);
plot(2*cos(theta), 2*sin(theta), 'g-', 'LineWidth', 2);
plot(0, 0, 'g*', 'MarkerSize', 15);

% Labels
xlabel('X (cm)');
ylabel('Y (cm)');
title(sprintf('Crimaldi Plume (%d Hz, %.3f mm/px)', ...
              crim_cfg.temporal.frame_rate, crim_cfg.spatial.mm_per_pixel));
grid on;
axis equal;
xlim([-10, 10]);
ylim([-32, 2]);

% Data info
text(0.02, 0.02, sprintf('Size: %d×%d px\nArena: %.1f×%.1f cm', ...
                         crim_cfg.spatial.resolution.width, ...
                         crim_cfg.spatial.resolution.height, ...
                         crim_cfg.spatial.arena_bounds.x_max - crim_cfg.spatial.arena_bounds.x_min, ...
                         crim_cfg.spatial.arena_bounds.y_max - crim_cfg.spatial.arena_bounds.y_min), ...
     'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 9);

%% Plot Smoke
subplot(1,2,2);
imagesc(smoke_x, smoke_y, smoke_data);
set(gca, 'YDir', 'normal');
colormap(hot);
cbar = colorbar();
ylabel(cbar, 'Odor Concentration');
hold on;

% Arena boundary
rectangle('Position', [smoke_cfg.spatial.arena_bounds.x_min, ...
                      smoke_cfg.spatial.arena_bounds.y_min, ...
                      smoke_cfg.spatial.arena_bounds.x_max - smoke_cfg.spatial.arena_bounds.x_min, ...
                      smoke_cfg.spatial.arena_bounds.y_max - smoke_cfg.spatial.arena_bounds.y_min], ...
          'EdgeColor', 'blue', 'LineWidth', 2, 'LineStyle', '--');

% Model's hardcoded initialization would be here
rectangle('Position', [init_x(1), init_y(1), init_x(2)-init_x(1), init_y(2)-init_y(1)], ...
          'EdgeColor', 'red', 'LineWidth', 3, 'LineStyle', ':');
text(mean(init_x), mean(init_y), 'Model Init\n(out of bounds!)', ...
     'Color', 'red', 'FontSize', 10, 'HorizontalAlignment', 'center');

% Adjusted init zone that would fit
adj_init_y = [-25, -20];  % Near bottom but within bounds
rectangle('Position', [init_x(1), adj_init_y(1), init_x(2)-init_x(1), adj_init_y(2)-adj_init_y(1)], ...
          'EdgeColor', 'yellow', 'LineWidth', 3);
text(mean(init_x), mean(adj_init_y), 'Adjusted Init', ...
     'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'center');

% Mark Y=0 (top)
plot([-8 8], [0 0], 'g--', 'LineWidth', 2);
text(0, 1, 'Y=0 (top)', 'Color', 'green', 'FontSize', 10, 'HorizontalAlignment', 'center');

% Likely source location (middle of arena)
source_y = -smoke_cfg.spatial.arena_bounds.y_min / 2;  % -13.2
plot(2*cos(theta), source_y + 2*sin(theta), 'g-', 'LineWidth', 2);
plot(0, source_y, 'g*', 'MarkerSize', 15);
text(0, source_y+1, 'Likely source', 'Color', 'green', 'FontSize', 10, 'HorizontalAlignment', 'center');

% Labels
xlabel('X (cm)');
ylabel('Y (cm)');
title(sprintf('Smoke Plume (%d Hz, %.3f mm/px)', ...
              smoke_cfg.temporal.frame_rate, smoke_cfg.spatial.mm_per_pixel));
grid on;
axis equal;
xlim([-10, 10]);
ylim([-28, 2]);

% Data info
text(0.02, 0.02, sprintf('Size: %d×%d px\nArena: %.1f×%.1f cm', ...
                         smoke_cfg.spatial.resolution.width, ...
                         smoke_cfg.spatial.resolution.height, ...
                         smoke_cfg.spatial.arena_bounds.x_max - smoke_cfg.spatial.arena_bounds.x_min, ...
                         smoke_cfg.spatial.arena_bounds.y_max - smoke_cfg.spatial.arena_bounds.y_min), ...
     'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 9);

%% Overall title
sgtitle('Plume Comparison with Model Initialization Zones', 'FontSize', 16);

% Save
saveas(gcf, 'both_plumes_comparison.png');
fprintf('\n✓ Saved plot to: both_plumes_comparison.png\n');

%% Summary
fprintf('\nCoordinate Check:\n');
fprintf('Both plumes use Y=0 at top (pixel 1)\n');
fprintf('Crimaldi: Y ∈ [%.1f, %.1f] cm\n', crim_cfg.spatial.arena_bounds.y_min, crim_cfg.spatial.arena_bounds.y_max);
fprintf('Smoke:    Y ∈ [%.1f, %.1f] cm\n', smoke_cfg.spatial.arena_bounds.y_min, smoke_cfg.spatial.arena_bounds.y_max);
fprintf('\nModel init at Y ∈ [-30, -25] cm:\n');
fprintf('  ✓ Valid for Crimaldi (arena goes to -30)\n');
fprintf('  ✗ Invalid for Smoke (arena only goes to -26.4)\n');
fprintf('  → Smoke needs shifted init or model modification\n');