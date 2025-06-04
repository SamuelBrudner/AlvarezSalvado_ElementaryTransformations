% generate_clean_configs.m - Generate complete configs with all correct settings

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
crim_cfg.simulation.duration_seconds = 240.0;  % 4 minutes

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
fid = fopen('configs/plumes/crimaldi_10cms_bounded.json', 'w');
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
smoke_cfg.simulation.duration_seconds = 60.0;  % 1 minute (60Hz data)

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
fid = fopen('configs/plumes/smoke_1a_backgroundsubtracted.json', 'w');
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

%% Create visualization
fprintf('\nCreating visualization...\n');

figure('Position', [100 100 1200 500]);

% Crimaldi setup
subplot(1,2,1);
hold on;

% Arena
rectangle('Position', [crim_cfg.spatial.arena_bounds.x_min, ...
                      crim_cfg.spatial.arena_bounds.y_min, ...
                      crim_width_cm, crim_height_cm], ...
          'EdgeColor', 'blue', 'LineWidth', 2);

% Init zone
rectangle('Position', [init_x_range(1), init_y_range(1), ...
                      diff(init_x_range), diff(init_y_range)], ...
          'EdgeColor', 'yellow', 'LineWidth', 3);
text(0, mean(init_y_range), 'Init Zone', 'Color', 'yellow', ...
     'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Source
theta = linspace(0, 2*pi, 100);
plot(success_radius*cos(theta), success_radius*sin(theta), 'g-', 'LineWidth', 2);
plot(0, 0, 'g*', 'MarkerSize', 15);
text(0, 1, 'Source (0,0)', 'Color', 'green', 'HorizontalAlignment', 'center');

% Annotations
plot([-8 8], [0 0], 'k--', 'LineWidth', 1);
text(7, 0.5, 'Y=0', 'HorizontalAlignment', 'right');

title(sprintf('Crimaldi Arena (%.0f Hz)', CRIM_FRAME_RATE));
xlabel('X (cm)'); ylabel('Y (cm)');
axis equal;
xlim([-10, 10]); ylim([-32, 2]);
grid on;

% Smoke setup
subplot(1,2,2);
hold on;

% Arena
rectangle('Position', [smoke_cfg.spatial.arena_bounds.x_min, ...
                      smoke_cfg.spatial.arena_bounds.y_min, ...
                      smoke_width_cm, smoke_height_cm], ...
          'EdgeColor', 'blue', 'LineWidth', 2);

% Init zone
rectangle('Position', [init_x_range(1), init_y_range(1), ...
                      diff(init_x_range), diff(init_y_range)], ...
          'EdgeColor', 'yellow', 'LineWidth', 3);
text(0, mean(init_y_range), 'Init Zone', 'Color', 'yellow', ...
     'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Source
plot(success_radius*cos(theta), success_radius*sin(theta), 'g-', 'LineWidth', 2);
plot(0, 0, 'g*', 'MarkerSize', 15);
text(0, 1, 'Source (0,0)', 'Color', 'green', 'HorizontalAlignment', 'center');

% Annotations
plot([-8.3 8.3], [0 0], 'k--', 'LineWidth', 1);
text(7, 0.5, 'Y=0', 'HorizontalAlignment', 'right');

title(sprintf('Smoke Arena (%.0f Hz)', SMOKE_FRAME_RATE));
xlabel('X (cm)'); ylabel('Y (cm)');
axis equal;
xlim([-10, 10]); ylim([-28, 2]);
grid on;

sgtitle('Complete Configuration Setup', 'FontSize', 16);

% Save figure
if ~exist('results', 'dir')
    mkdir('results');
end
saveas(gcf, 'results/complete_config_setup.png');
fprintf('✓ Saved visualization to results/complete_config_setup.png\n');

fprintf('\n✓ Complete clean configs generated!\n');