% update_configs_shifted_init.m - Update configs with shifted initialization zones

fprintf('=== Updating Configs with Shifted Initialization ===\n\n');

% New initialization zone that works for both plumes
new_init_y = [-26.4, -21.4];  % 5cm band at bottom of smoke arena
new_init_x = [-8, 8];         % Keep X range the same

%% 1. Update Crimaldi config
fprintf('1. Updating Crimaldi config...\n');
crim_file = 'configs/plumes/crimaldi_10cms_bounded.json';
crim_cfg = jsondecode(fileread(crim_file));

% Add simulation parameters if not present
if ~isfield(crim_cfg, 'simulation')
    crim_cfg.simulation = struct();
end

% Set success parameters
crim_cfg.simulation.success_radius_cm = 2.0;
crim_cfg.simulation.duration_seconds = 240.0;  % 4 minutes

% Set agent initialization
crim_cfg.simulation.agent_initialization = struct();
crim_cfg.simulation.agent_initialization.x_range_cm = new_init_x;
crim_cfg.simulation.agent_initialization.y_range_cm = new_init_y;
crim_cfg.simulation.agent_initialization.n_agents_per_job = 10;

% Set source position (at origin for Crimaldi)
crim_cfg.simulation.source_position = struct();
crim_cfg.simulation.source_position.x_cm = 0;
crim_cfg.simulation.source_position.y_cm = 0;

% Save updated config
fid = fopen(crim_file, 'w');
fprintf(fid, '%s', jsonencode(crim_cfg));
fclose(fid);

fprintf('   ✓ Updated initialization to Y ∈ [%.1f, %.1f]\n', new_init_y(1), new_init_y(2));
fprintf('   ✓ This is %.1f cm from arena bottom (was %.1f cm)\n', ...
        new_init_y(1) - crim_cfg.spatial.arena_bounds.y_min, ...
        -30 - (-30));  % Original was at bottom edge

%% 2. Update Smoke config
fprintf('\n2. Updating Smoke config...\n');
smoke_file = 'configs/plumes/smoke_1a_backgroundsubtracted.json';
smoke_cfg = jsondecode(fileread(smoke_file));

% Add simulation parameters if not present
if ~isfield(smoke_cfg, 'simulation')
    smoke_cfg.simulation = struct();
end

% Set success parameters
smoke_cfg.simulation.success_radius_cm = 2.0;
smoke_cfg.simulation.duration_seconds = 60.0;  % 1 minute (60Hz data)

% Set agent initialization
smoke_cfg.simulation.agent_initialization = struct();
smoke_cfg.simulation.agent_initialization.x_range_cm = new_init_x;
smoke_cfg.simulation.agent_initialization.y_range_cm = new_init_y;
smoke_cfg.simulation.agent_initialization.n_agents_per_job = 10;

% Set source position (middle of arena for smoke)
smoke_source_y = smoke_cfg.spatial.arena_bounds.y_min / 2;  % -13.2 cm
smoke_cfg.simulation.source_position = struct();
smoke_cfg.simulation.source_position.x_cm = 0;
smoke_cfg.simulation.source_position.y_cm = smoke_source_y;

% Save updated config
fid = fopen(smoke_file, 'w');
fprintf(fid, '%s', jsonencode(smoke_cfg));
fclose(fid);

fprintf('   ✓ Updated initialization to Y ∈ [%.1f, %.1f]\n', new_init_y(1), new_init_y(2));
fprintf('   ✓ This is at arena bottom (perfect fit!)\n');
fprintf('   ✓ Set source position at Y = %.1f cm (middle of arena)\n', smoke_source_y);

%% 3. Create visualization of new setup
fprintf('\n3. Creating visualization of updated configs...\n');

figure('Position', [100 100 1200 500]);

% Crimaldi setup
subplot(1,2,1);
hold on;

% Arena
rectangle('Position', [-8, -30, 16, 30], ...
          'EdgeColor', 'blue', 'LineWidth', 2);

% Original init zone (for reference)
rectangle('Position', [-8, -30, 16, 5], ...
          'EdgeColor', [0.7 0.7 0.7], 'LineWidth', 1, 'LineStyle', '--');
text(0, -27.5, 'Original Init', 'Color', [0.5 0.5 0.5], ...
     'HorizontalAlignment', 'center');

% New init zone
rectangle('Position', [new_init_x(1), new_init_y(1), ...
                      diff(new_init_x), diff(new_init_y)], ...
          'EdgeColor', 'yellow', 'LineWidth', 3);
text(0, mean(new_init_y), 'New Init Zone', 'Color', 'yellow', ...
     'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Source
theta = linspace(0, 2*pi, 100);
plot(2*cos(theta), 2*sin(theta), 'g-', 'LineWidth', 2);
plot(0, 0, 'g*', 'MarkerSize', 15);
text(0, 1, 'Source', 'Color', 'green', 'HorizontalAlignment', 'center');

% Annotations
plot([-8 8], [0 0], 'k--', 'LineWidth', 1);
text(7, 0.5, 'Y=0', 'HorizontalAlignment', 'right');

title('Crimaldi Arena');
xlabel('X (cm)'); ylabel('Y (cm)');
axis equal;
xlim([-10, 10]); ylim([-32, 2]);
grid on;

% Distance annotation
annotation('doublearrow', [0.18 0.18], [0.23 0.42]);
text(-9.5, mean(new_init_y), sprintf('%.1f cm', abs(diff(new_init_y))), ...
     'Rotation', 90, 'HorizontalAlignment', 'center');

% Smoke setup
subplot(1,2,2);
hold on;

% Arena
rectangle('Position', [-8.3, -26.4, 16.6, 26.4], ...
          'EdgeColor', 'blue', 'LineWidth', 2);

% New init zone (exactly at bottom)
rectangle('Position', [new_init_x(1), new_init_y(1), ...
                      diff(new_init_x), diff(new_init_y)], ...
          'EdgeColor', 'yellow', 'LineWidth', 3);
text(0, mean(new_init_y), 'New Init Zone', 'Color', 'yellow', ...
     'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Source (middle of arena)
plot(2*cos(theta), smoke_source_y + 2*sin(theta), 'g-', 'LineWidth', 2);
plot(0, smoke_source_y, 'g*', 'MarkerSize', 15);
text(0, smoke_source_y + 1, 'Source', 'Color', 'green', 'HorizontalAlignment', 'center');

% Annotations
plot([-8.3 8.3], [0 0], 'k--', 'LineWidth', 1);
text(7, 0.5, 'Y=0', 'HorizontalAlignment', 'right');

title('Smoke Arena');
xlabel('X (cm)'); ylabel('Y (cm)');
axis equal;
xlim([-10, 10]); ylim([-28, 2]);
grid on;

% Overall title
sgtitle('Updated Initialization Zones', 'FontSize', 16);

% Save figure
saveas(gcf, 'results/shifted_init_zones.png');
fprintf('   ✓ Saved visualization to results/shifted_init_zones.png\n');

%% 4. Summary
fprintf('\n=== Summary ===\n');
fprintf('✓ Both configs updated with Y ∈ [%.1f, %.1f] cm initialization\n', ...
        new_init_y(1), new_init_y(2));
fprintf('✓ This zone works for BOTH plumes:\n');
fprintf('  - Crimaldi: %.1f cm from bottom (arena extends to -30)\n', ...
        new_init_y(1) - (-30));
fprintf('  - Smoke: Exactly at bottom edge (arena extends to -26.4)\n');
fprintf('\nSource positions:\n');
fprintf('  - Crimaldi: (0, 0) - at arena top\n');
fprintf('  - Smoke: (0, %.1f) - at arena middle\n', smoke_source_y);
fprintf('\n✓ Model can now use the same initialization for both plumes!\n');

%% 5. Test that configs are valid JSON
fprintf('\nValidating JSON configs...\n');
try
    test_crim = jsondecode(fileread(crim_file));
    fprintf('  ✓ Crimaldi config is valid JSON\n');
    test_smoke = jsondecode(fileread(smoke_file));
    fprintf('  ✓ Smoke config is valid JSON\n');
catch ME
    error('JSON validation failed: %s', ME.message);
end

fprintf('\n✓ All updates complete!\n');