% generate_clean_configs.m - Generate complete configs with plume backgrounds

fprintf('=== Generating Complete Clean Configs ===\n\n');

%% Define shared parameters
% Initialization zone that works for both plumes
init_x_range = [-8, 8];        % cm
init_y_range = [-26.4, -21.4]; % cm (at bottom of smoke arena)
n_agents_per_job = 10;
success_radius = 2.0;          % cm
source_position = [0, 0];      % Both sources at origin

%% CRIMALDI CONFIG
fprintf('1. Generating Crimaldi config...\n');

% Known experimental parameters
CRIM_FRAME_RATE = 15;      % Hz
CRIM_MM_PER_PIXEL = 0.74;  % mm/px

% Get actual HDF5 dimensions
crim_file = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/10302017_10cms_bounded.hdf5';
crim_info = h5info(crim_file, '/dataset2');
crim_dims = crim_info.Dataspace.Size;

fprintf('   HDF5: %d x %d x %d\n', crim_dims);

% Build complete config
crim_cfg = struct();

% Essential identifiers
crim_cfg.plume_id = 'crimaldi_10cms_bounded';

% Data source
crim_cfg.data_path = struct();
crim_cfg.data_path.path = crim_file;
crim_cfg.data_path.dataset_name = '/dataset2';

% Spatial parameters
crim_cfg.spatial = struct();
crim_cfg.spatial.resolution.width = crim_dims(1);
crim_cfg.spatial.resolution.height = crim_dims(2);
crim_cfg.spatial.mm_per_pixel = CRIM_MM_PER_PIXEL;

% Calculate arena bounds (Y=0 at top)
crim_width_cm = crim_dims(1) * CRIM_MM_PER_PIXEL / 10;
crim_height_cm = crim_dims(2) * CRIM_MM_PER_PIXEL / 10;
crim_cfg.spatial.arena_bounds.x_min = -crim_width_cm/2;
crim_cfg.spatial.arena_bounds.x_max = crim_width_cm/2;
crim_cfg.spatial.arena_bounds.y_min = -crim_height_cm;
crim_cfg.spatial.arena_bounds.y_max = 0;

% Temporal parameters
crim_cfg.temporal = struct();
crim_cfg.temporal.frame_rate = CRIM_FRAME_RATE;
crim_cfg.temporal.total_frames = crim_dims(3);

% Model scaling parameters
crim_cfg.model_params = struct();
crim_cfg.model_params.tscale = CRIM_FRAME_RATE / 50.0;  % Scaling for 50Hz base parameters
crim_cfg.model_params.pxscale = CRIM_MM_PER_PIXEL;     % Same as spatial.mm_per_pixel

% Simulation parameters
crim_cfg.simulation = struct();
crim_cfg.simulation.success_radius_cm = success_radius;
crim_cfg.simulation.duration_seconds = 5.0;  % QUICK TEST  % 4 minutes

% Agent initialization
crim_cfg.simulation.agent_initialization = struct();
crim_cfg.simulation.agent_initialization.x_range_cm = init_x_range;
crim_cfg.simulation.agent_initialization.y_range_cm = init_y_range;
crim_cfg.simulation.agent_initialization.n_agents_per_job = n_agents_per_job;

% Source position
crim_cfg.simulation.source_position = struct();
crim_cfg.simulation.source_position.x_cm = source_position(1);
crim_cfg.simulation.source_position.y_cm = source_position(2);

% Save
% Ensure the output directory exists
cfg_dir = 'configs/plumes';
if ~exist(cfg_dir, 'dir')
    fprintf('Creating directory: %s\n', cfg_dir);
    mkdir(cfg_dir);
end

% Open file with error handling
config_file = [cfg_dir, '/crimaldi_10cms_bounded.json'];
fprintf('Saving config to: %s\n', config_file);
fid = fopen(config_file, 'w');
if fid == -1
    error('Failed to open file for writing: %s', config_file);
end

% Write and close
fprintf(fid, '%s', jsonencode(crim_cfg));
fclose(fid);

fprintf('   ✓ Saved: Arena %.1f x %.1f cm\n', crim_width_cm, crim_height_cm);
fprintf('   ✓ Init zone: Y ∈ [%.1f, %.1f] (%.1f cm from bottom)\n', ...
        init_y_range(1), init_y_range(2), ...
        init_y_range(1) - crim_cfg.spatial.arena_bounds.y_min);
fprintf('   ✓ Source at (%.1f, %.1f)\n', source_position(1), source_position(2));

%% SMOKE CONFIG
fprintf('\n2. Generating Smoke config...\n');

% Known experimental parameters
SMOKE_FRAME_RATE = 60;       % Hz
SMOKE_MM_PER_PIXEL = 0.153;  % mm/px

% Get actual HDF5 dimensions
smoke_file = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d_fixed.h5';
smoke_info = h5info(smoke_file, '/dataset2');
smoke_dims = smoke_info.Dataspace.Size;

fprintf('   HDF5: %d x %d x %d\n', smoke_dims);

% Build complete config
smoke_cfg = struct();

% Essential identifiers
smoke_cfg.plume_id = 'smoke_1a_backgroundsubtracted';

% Data source
smoke_cfg.data_path = struct();
smoke_cfg.data_path.path = smoke_file;
smoke_cfg.data_path.dataset_name = '/dataset2';

% Spatial parameters
smoke_cfg.spatial = struct();
smoke_cfg.spatial.resolution.width = smoke_dims(1);
smoke_cfg.spatial.resolution.height = smoke_dims(2);
smoke_cfg.spatial.mm_per_pixel = SMOKE_MM_PER_PIXEL;

% Calculate arena bounds (Y=0 at top)
smoke_width_cm = smoke_dims(1) * SMOKE_MM_PER_PIXEL / 10;
smoke_height_cm = smoke_dims(2) * SMOKE_MM_PER_PIXEL / 10;
smoke_cfg.spatial.arena_bounds.x_min = -smoke_width_cm/2;
smoke_cfg.spatial.arena_bounds.x_max = smoke_width_cm/2;
smoke_cfg.spatial.arena_bounds.y_min = -smoke_height_cm;
smoke_cfg.spatial.arena_bounds.y_max = 0;

% Temporal parameters
smoke_cfg.temporal = struct();
smoke_cfg.temporal.frame_rate = SMOKE_FRAME_RATE;
smoke_cfg.temporal.total_frames = smoke_dims(3);

% Model scaling parameters
smoke_cfg.model_params = struct();
smoke_cfg.model_params.tscale = SMOKE_FRAME_RATE / 50.0;  % Scaling for 50Hz base parameters
smoke_cfg.model_params.pxscale = SMOKE_MM_PER_PIXEL;      % Same as spatial.mm_per_pixel

% Simulation parameters
smoke_cfg.simulation = struct();
smoke_cfg.simulation.success_radius_cm = success_radius;
smoke_cfg.simulation.duration_seconds = 5.0;  % QUICK TEST  % 1 minute (60Hz data)

% Agent initialization
smoke_cfg.simulation.agent_initialization = struct();
smoke_cfg.simulation.agent_initialization.x_range_cm = init_x_range;
smoke_cfg.simulation.agent_initialization.y_range_cm = init_y_range;
smoke_cfg.simulation.agent_initialization.n_agents_per_job = n_agents_per_job;

% Source position
smoke_cfg.simulation.source_position = struct();
smoke_cfg.simulation.source_position.x_cm = source_position(1);
smoke_cfg.simulation.source_position.y_cm = source_position(2);

% Save
% Ensure the output directory exists (already checked above but being safe)
cfg_dir = 'configs/plumes';
if ~exist(cfg_dir, 'dir')
    fprintf('Creating directory: %s\n', cfg_dir);
    mkdir(cfg_dir);
end

% Open file with error handling
config_file = [cfg_dir, '/smoke_1a_backgroundsubtracted.json'];
fprintf('Saving config to: %s\n', config_file);
fid = fopen(config_file, 'w');
if fid == -1
    error('Failed to open file for writing: %s', config_file);
end

% Write and close
fprintf(fid, '%s', jsonencode(smoke_cfg));
fclose(fid);

fprintf('   ✓ Saved: Arena %.1f x %.1f cm\n', smoke_width_cm, smoke_height_cm);
fprintf('   ✓ Init zone: Y ∈ [%.1f, %.1f] (exactly at bottom)\n', ...
        init_y_range(1), init_y_range(2));
fprintf('   ✓ Source at (%.1f, %.1f)\n', source_position(1), source_position(2));

%% SUMMARY
fprintf('\n=== Summary ===\n');
fprintf('Both configs generated with:\n');
fprintf('  - Initialization: X ∈ [%.1f, %.1f], Y ∈ [%.1f, %.1f] cm\n', ...
        init_x_range(1), init_x_range(2), init_y_range(1), init_y_range(2));
fprintf('  - Source position: (%.1f, %.1f) for both plumes\n', ...
        source_position(1), source_position(2));
fprintf('  - Success radius: %.1f cm\n', success_radius);
fprintf('  - Agents per job: %d\n', n_agents_per_job);
fprintf('\nArena details:\n');
fprintf('  - Crimaldi: %d×%d px = %.1f×%.1f cm, Y∈[%.1f,%.1f]\n', ...
        crim_dims(1), crim_dims(2), crim_width_cm, crim_height_cm, ...
        crim_cfg.spatial.arena_bounds.y_min, crim_cfg.spatial.arena_bounds.y_max);
fprintf('  - Smoke:    %d×%d px = %.1f×%.1f cm, Y∈[%.1f,%.1f]\n', ...
        smoke_dims(1), smoke_dims(2), smoke_width_cm, smoke_height_cm, ...
        smoke_cfg.spatial.arena_bounds.y_min, smoke_cfg.spatial.arena_bounds.y_max);
fprintf('\nModel scaling parameters:\n');
fprintf('  - Crimaldi: tscale=%.3f (for %dHz), pxscale=%.3f mm/px\n', ...
        crim_cfg.model_params.tscale, CRIM_FRAME_RATE, crim_cfg.model_params.pxscale);
fprintf('  - Smoke:    tscale=%.3f (for %dHz), pxscale=%.3f mm/px\n', ...
        smoke_cfg.model_params.tscale, SMOKE_FRAME_RATE, smoke_cfg.model_params.pxscale);

%% Load plume data for visualization
fprintf('\nLoading plume data for visualization...\n');

% Load Crimaldi plume data
middle_frame_crim = round(crim_dims(3) / 2);
crim_data = h5read(crim_file, '/dataset2', ...
                   [1 1 middle_frame_crim], [crim_dims(1:2), 1]);
crim_data = squeeze(crim_data)';  % Transpose for correct orientation

% Load Smoke plume data
middle_frame_smoke = round(smoke_dims(3) / 2);
smoke_data = h5read(smoke_file, '/dataset2', ...
                    [1 1 middle_frame_smoke], [smoke_dims(1:2), 1]);
smoke_data = squeeze(smoke_data)';  % Transpose for correct orientation

% Calculate coordinate arrays
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

%% Create visualization with plume data - FIXED SCALE
fprintf('Creating visualization with plume data (same physical scale)...\n');

figure('Position', [100 100 1200 500]);

% Define common axis limits for both panels
common_xlim = [-10, 10];
common_ylim = [-32, 2];  % Use the larger range to show everything

% Crimaldi setup
subplot(1,2,1);
hold on;

% Display plume data first
imagesc(crim_x, crim_y, crim_data);
set(gca, 'YDir', 'normal');
colormap(flipud(gray));
caxis([0, prctile(crim_data(:), 95)]);  % Adjust color scale for visibility

% Add semi-transparent overlay to make annotations visible
h = patch([common_xlim(1) common_xlim(2) common_xlim(2) common_xlim(1)], ...
          [common_ylim(1) common_ylim(1) common_ylim(2) common_ylim(2)], 'w');
set(h, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
% Unified arena decorations
plot_arena_elements(crim_cfg.spatial.arena_bounds, init_x_range, init_y_range, success_radius, common_xlim, common_ylim, 5);

% Init zone

% Annotations
plot([-8 8], [0 0], 'k--', 'LineWidth', 1);
text(7, 0.5, 'Y=0', 'HorizontalAlignment', 'right');

title(sprintf('Crimaldi Arena (%.0f Hz)', CRIM_FRAME_RATE));
xlabel('X (cm)'); ylabel('Y (cm)');
axis equal;
xlim(common_xlim);
ylim(common_ylim);
grid on;
set(gca, 'GridAlpha', 0.3);

% Data info
text(0.02, 0.98, sprintf('Frame %d/%d\nSize: %d×%d px\nArena: %.1f×%.1f cm', ...
                         middle_frame_crim, crim_dims(3), ...
                         crim_cfg.spatial.resolution.width, ...
                         crim_cfg.spatial.resolution.height, ...
                         crim_cfg.spatial.arena_bounds.x_max - crim_cfg.spatial.arena_bounds.x_min, ...
                         crim_cfg.spatial.arena_bounds.y_max - crim_cfg.spatial.arena_bounds.y_min), ...
     'Units', 'normalized', 'BackgroundColor', [1 1 1 0.8], 'FontSize', 9, ...
     'VerticalAlignment', 'top');

% Smoke setup
subplot(1,2,2);
hold on;

% Display plume data first
imagesc(smoke_x, smoke_y, smoke_data);
set(gca, 'YDir', 'normal');
colormap(flipud(gray));
caxis([0, prctile(smoke_data(:), 95)]);  % Adjust color scale for visibility

% Add semi-transparent overlay
h = patch([common_xlim(1) common_xlim(2) common_xlim(2) common_xlim(1)], ...
          [common_ylim(1) common_ylim(1) common_ylim(2) common_ylim(2)], 'w');
set(h, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
% Unified arena decorations
plot_arena_elements(crim_cfg.spatial.arena_bounds, init_x_range, init_y_range, success_radius, common_xlim, common_ylim, 5);



% Init zone



% Source




% Annotations
plot([-8.3 8.3], [0 0], 'k--', 'LineWidth', 1);
text(7, 0.5, 'Y=0', 'HorizontalAlignment', 'right');

title(sprintf('Smoke Arena (%.0f Hz)', SMOKE_FRAME_RATE));
xlabel('X (cm)'); ylabel('Y (cm)');
axis equal;
xlim(common_xlim);
ylim(common_ylim);
grid on;
set(gca, 'GridAlpha', 0.3);

% Data info
text(0.02, 0.98, sprintf('Frame %d/%d\nSize: %d×%d px\nArena: %.1f×%.1f cm', ...
                         middle_frame_smoke, smoke_dims(3), ...
                         smoke_cfg.spatial.resolution.width, ...
                         smoke_cfg.spatial.resolution.height, ...
                         smoke_cfg.spatial.arena_bounds.x_max - smoke_cfg.spatial.arena_bounds.x_min, ...
                         smoke_cfg.spatial.arena_bounds.y_max - smoke_cfg.spatial.arena_bounds.y_min), ...
     'Units', 'normalized', 'BackgroundColor', [1 1 1 0.8], 'FontSize', 9, ...
     'VerticalAlignment', 'top');

% Add shared colorbar
h = colorbar('Position', [0.93 0.3 0.02 0.4]);
ylabel(h, 'Odor Concentration', 'FontSize', 11);



sgtitle('Complete Configuration Setup with Plume Data (Same Physical Scale)', 'FontSize', 16);

% Save figure
if ~exist('results', 'dir')
    mkdir('results');
end
saveas(gcf, 'results/complete_config_setup_with_plumes.pdf');
fprintf('✓ Saved visualization to results/complete_config_setup_with_plumes.pdf\n');

fprintf('\n✓ Complete clean configs generated with plume visualization (same scale)!\n');