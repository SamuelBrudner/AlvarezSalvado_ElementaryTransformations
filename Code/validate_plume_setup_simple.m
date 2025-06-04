function validate_plume_setup_simple(output_file)
%VALIDATE_PLUME_SETUP_SIMPLE Create validation figure (simplified version)

if nargin < 1
    output_file = '';
end

fprintf('Creating plume validation figure (simplified)...\n');

% Load configuration
try
    paths = load_paths();
    [plume_file, plume_config] = get_plume_file();
catch ME
    error('Failed to load configuration: %s', ME.message);
end

% Set default parameters
success_radius = 2.0;
init_x_range = [-8, 8];
init_y_range = [-30, -25];
n_agents = 10;
source_x = 0;
source_y = 0;

% Try to load from config
try
    config_path = paths.plume_config;
    cfg = jsondecode(fileread(config_path));
    if isfield(cfg, 'simulation')
        sim = cfg.simulation;
        if isfield(sim, 'success_radius_cm')
            success_radius = sim.success_radius_cm;
        end
        if isfield(sim, 'agent_initialization')
            init_x_range = sim.agent_initialization.x_range_cm;
            init_y_range = sim.agent_initialization.y_range_cm;
            n_agents = sim.agent_initialization.n_agents_per_job;
        end
    end
    fprintf('Loaded parameters from config\n');
catch
    fprintf('Using default parameters\n');
end

% Load plume data
fprintf('Loading plume data...\n');
dataset_name = plume_config.dataset_name;
if ~startsWith(dataset_name, '/')
    dataset_name = ['/', dataset_name];
end

% Get plume info
info = h5info(plume_file, dataset_name);
plume_size = info.Dataspace.Size;
middle_frame = round(plume_size(3) / 2);

% Read middle frame
plume_data = h5read(plume_file, dataset_name, [1 1 middle_frame], [plume_size(1) plume_size(2) 1]);
plume_data = squeeze(plume_data)';

% Convert coordinates
pxscale = plume_config.mm_per_pixel;
x_pixels = 1:plume_size(1);
y_pixels = 1:plume_size(2);
x_mm = (x_pixels - 108) * pxscale / 10;
y_mm = -(y_pixels - 1) * pxscale / 10;

% Create figure
figure('Position', [100 100 800 900]);

% Plot plume
imagesc(x_mm, y_mm, plume_data);
set(gca, 'YDir', 'normal');
colormap(hot);

% Add simple colorbar (no label to avoid issues)
colorbar();

hold on;

% Plume boundary
plume_x_bounds = [min(x_mm), max(x_mm)];
plume_y_bounds = [min(y_mm), max(y_mm)];
plot([plume_x_bounds(1) plume_x_bounds(2) plume_x_bounds(2) plume_x_bounds(1) plume_x_bounds(1)], ...
     [plume_y_bounds(1) plume_y_bounds(1) plume_y_bounds(2) plume_y_bounds(2) plume_y_bounds(1)], ...
     'b--', 'LineWidth', 2);

% Success zone (circle drawn manually)
theta = linspace(0, 2*pi, 100);
plot(source_x + success_radius*cos(theta), source_y + success_radius*sin(theta), ...
     'g-', 'LineWidth', 3);
plot(source_x, source_y, 'g*', 'MarkerSize', 15, 'LineWidth', 2);
text(source_x + 0.5, source_y + 0.5, 'Source', 'Color', 'green', ...
     'FontSize', 12, 'FontWeight', 'bold');

% Agent init zone
plot([init_x_range(1) init_x_range(2) init_x_range(2) init_x_range(1) init_x_range(1)], ...
     [init_y_range(1) init_y_range(1) init_y_range(2) init_y_range(2) init_y_range(1)], ...
     'y-', 'LineWidth', 3);
text(mean(init_x_range), mean(init_y_range), 'Start Zone', ...
     'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'center');

% Sample positions
rng(42);
sample_x = init_x_range(1) + diff(init_x_range) * rand(n_agents, 1);
sample_y = init_y_range(1) + diff(init_y_range) * rand(n_agents, 1);
plot(sample_x, sample_y, 'y.', 'MarkerSize', 10);

% Scale bar
scale_length = 10;
scale_x = plume_x_bounds(2) - scale_length - 2;
scale_y = plume_y_bounds(1) + 3;
plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 'k-', 'LineWidth', 4);
text(scale_x + scale_length/2, scale_y - 1, '10 cm', ...
     'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');

% Labels
xlabel('X Position (cm)');
ylabel('Y Position (cm)');
title(sprintf('Plume Validation - Frame %d/%d', middle_frame, plume_size(3)));

% Info text
info_text = sprintf(['Success radius: %.1f cm\n' ...
                     'Init zone: X=[%.1f,%.1f], Y=[%.1f,%.1f]\n' ...
                     'Agents/job: %d\n' ...
                     'Duration: %.0f s'], ...
                    success_radius, init_x_range(1), init_x_range(2), ...
                    init_y_range(1), init_y_range(2), n_agents, ...
                    plume_config.simulation.duration_seconds);
text(0.02, 0.98, info_text, 'Units', 'normalized', ...
     'VerticalAlignment', 'top', 'BackgroundColor', 'white', ...
     'FontSize', 10);

% Set limits
xlim([-15, 15]);
ylim([-35, 5]);
grid on;
axis equal;

% Save if requested
if ~isempty(output_file)
    fprintf('Saving to: %s\n', output_file);
    print(gcf, output_file, '-dpng', '-r150');
    fprintf('✓ Figure saved\n');
end

end