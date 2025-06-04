function validate_plume_setup(output_file)
%VALIDATE_PLUME_SETUP Create a validation figure showing plume setup
%   validate_plume_setup() - Display figure
%   validate_plume_setup('filename.png') - Save to file

if nargin < 1
    output_file = '';
end

fprintf('Creating plume validation figure...\n');

% Load configuration
try
    paths = load_paths();
    [plume_file, plume_config] = get_plume_file();
catch ME
    error('Failed to load configuration: %s', ME.message);
end

% Load simulation parameters from config
try
    config_path = paths.plume_config;
    cfg = jsondecode(fileread(config_path));
    
    % Get simulation parameters with defaults
    if isfield(cfg, 'simulation')
        sim_params = cfg.simulation;
        
        % Success radius
        if isfield(sim_params, 'success_radius_cm')
            success_radius = sim_params.success_radius_cm;
        else
            success_radius = 2.0;  % Default
            warning('Using default success radius: %.1f cm', success_radius);
        end
        
        % Agent initialization
        if isfield(sim_params, 'agent_initialization')
            agent_init = sim_params.agent_initialization;
            init_x_range = agent_init.x_range_cm;
            init_y_range = agent_init.y_range_cm;
            n_agents = agent_init.n_agents_per_job;
        else
            init_x_range = [-8, 8];
            init_y_range = [-30, -25];
            n_agents = 10;
            warning('Using default agent initialization parameters');
        end
        
        % Source position
        if isfield(sim_params, 'source_position')
            source_x = sim_params.source_position.x_cm;
            source_y = sim_params.source_position.y_cm;
        else
            source_x = 0;
            source_y = 0;
        end
    else
        % All defaults
        success_radius = 2.0;
        init_x_range = [-8, 8];
        init_y_range = [-30, -25];
        n_agents = 10;
        source_x = 0;
        source_y = 0;
        warning('No simulation parameters in config, using all defaults');
    end
catch
    % Fallback to hardcoded defaults
    success_radius = 2.0;
    init_x_range = [-8, 8];
    init_y_range = [-30, -25];
    n_agents = 10;
    source_x = 0;
    source_y = 0;
    warning('Could not read config, using hardcoded defaults');
end

% Load plume data
fprintf('Loading plume data from: %s\n', plume_file);
if ~exist(plume_file, 'file')
    error('Plume file not found: %s', plume_file);
end

% Get dataset name
dataset_name = plume_config.dataset_name;
if ~startsWith(dataset_name, '/')
    dataset_name = ['/', dataset_name];
end

% Read plume info
info = h5info(plume_file, dataset_name);
plume_size = info.Dataspace.Size; % [width, height, time]
fprintf('Plume dimensions: %d x %d x %d\n', plume_size);

% Get middle frame
middle_frame = round(plume_size(3) / 2);
fprintf('Using frame %d (middle of %d frames)\n', middle_frame, plume_size(3));

% Read the middle frame
plume_data = h5read(plume_file, dataset_name, [1 1 middle_frame], [plume_size(1) plume_size(2) 1]);
plume_data = squeeze(plume_data)';  % Transpose for correct orientation

% Scale factors
pxscale = plume_config.mm_per_pixel;  % mm per pixel
frame_rate = plume_config.frame_rate;

% Convert to physical coordinates
x_pixels = 1:plume_size(1);
y_pixels = 1:plume_size(2);

% Convert to mm coordinates (matching the model's coordinate system)
x_mm = (x_pixels - 108) * pxscale / 10;  % Convert to cm
y_mm = -(y_pixels - 1) * pxscale / 10;   % Convert to cm, flip y

% Create figure
figure('Position', [100 100 900, 1000]);

% Plot plume data
imagesc(x_mm, y_mm, plume_data);
set(gca, 'YDir', 'normal');  % Correct Y direction
colormap(hot);

% Add colorbar with label (compatible syntax)
cbar = colorbar();
ylabel(cbar, 'Odor Concentration');

hold on;

% Add plume boundary box
plume_x_bounds = [min(x_mm), max(x_mm)];
plume_y_bounds = [min(y_mm), max(y_mm)];
rectangle('Position', [plume_x_bounds(1), plume_y_bounds(1), ...
                      plume_x_bounds(2)-plume_x_bounds(1), ...
                      plume_y_bounds(2)-plume_y_bounds(1)], ...
          'EdgeColor', 'blue', 'LineWidth', 2, 'LineStyle', '--');

% Add odor source
viscircles([source_x, source_y], success_radius, 'Color', 'green', 'LineWidth', 3);
plot(source_x, source_y, 'g*', 'MarkerSize', 15, 'LineWidth', 2);
text(source_x + 0.5, source_y + 0.5, 'Source', 'Color', 'green', ...
     'FontSize', 12, 'FontWeight', 'bold');

% Add success zone annotation
viscircles([source_x, source_y], success_radius, 'Color', 'green', ...
           'LineStyle', '--', 'LineWidth', 1);
text(source_x + success_radius + 0.5, source_y, ...
     sprintf('%.0f cm', success_radius), ...
     'Color', 'green', 'FontSize', 10);

% Add agent initialization box
rectangle('Position', [init_x_range(1), init_y_range(1), ...
                      init_x_range(2)-init_x_range(1), ...
                      init_y_range(2)-init_y_range(1)], ...
          'EdgeColor', 'yellow', 'LineWidth', 3);
          
% Add text annotation for init region
text(mean(init_x_range), mean(init_y_range), ...
     sprintf('Agent Start Zone\n%d agents/job', n_agents), ...
     'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'center');

% Add sample starting positions
rng(42);  % For reproducibility
sample_x = init_x_range(1) + diff(init_x_range) * rand(n_agents, 1);
sample_y = init_y_range(1) + diff(init_y_range) * rand(n_agents, 1);
plot(sample_x, sample_y, 'y.', 'MarkerSize', 8);

% Add scale bar
scale_length = 10;  % 10 cm
scale_x = plume_x_bounds(2) - scale_length - 2;
scale_y = plume_y_bounds(1) + 3;
plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 'k-', 'LineWidth', 4);
text(scale_x + scale_length/2, scale_y - 1, sprintf('%d cm', scale_length), ...
     'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');

% Add north arrow (wind direction)
arrow_x = plume_x_bounds(1) + 3;
arrow_y = plume_y_bounds(2) - 5;
arrow_length = 3;
quiver(arrow_x, arrow_y - arrow_length/2, 0, arrow_length, 0, ...
       'k', 'LineWidth', 2, 'MaxHeadSize', 0.5);
text(arrow_x, arrow_y + arrow_length/2 + 1, 'Wind', ...
     'HorizontalAlignment', 'center', 'FontSize', 10);

% Add grid
grid on;
set(gca, 'GridAlpha', 0.3);

% Labels and title
xlabel('X Position (cm)');
ylabel('Y Position (cm)');
title(sprintf('Plume Validation - Frame %d/%d at t=%.1f s', ...
              middle_frame, plume_size(3), middle_frame/frame_rate), ...
      'FontSize', 14);

% Add information text box
info_text = {
    sprintf('Environment: Crimaldi plume data');
    sprintf('Data file: %s', plume_file);
    sprintf('Frame rate: %.1f Hz', frame_rate);
    sprintf('Pixel scale: %.3f mm/pixel', pxscale);
    sprintf('Plume size: %.1f x %.1f cm', ...
            (plume_x_bounds(2)-plume_x_bounds(1)), ...
            (plume_y_bounds(2)-plume_y_bounds(1)));
    '';
    sprintf('=== Simulation Parameters ===');
    sprintf('Source position: (%.1f, %.1f) cm', source_x, source_y);
    sprintf('Success radius: %.1f cm', success_radius);
    sprintf('Agent init X: [%.1f, %.1f] cm', init_x_range(1), init_x_range(2));
    sprintf('Agent init Y: [%.1f, %.1f] cm', init_y_range(1), init_y_range(2));
    sprintf('Agents per job: %d', n_agents);
    sprintf('Simulation duration: %.1f seconds', plume_config.simulation.duration_seconds);
};

text(0.02, 0.98, info_text, 'Units', 'normalized', ...
     'VerticalAlignment', 'top', 'BackgroundColor', 'white', ...
     'EdgeColor', 'black', 'FontSize', 9, 'FontName', 'FixedWidth');

% Add odor concentration info
odor_info = sprintf('Odor Concentration\nMax: %.3f\nMin: %.3f\nMean: %.3f', ...
                    max(plume_data(:)), min(plume_data(:)), mean(plume_data(:)));
text(0.98, 0.02, odor_info, ...
     'Units', 'normalized', 'HorizontalAlignment', 'right', ...
     'VerticalAlignment', 'bottom', 'BackgroundColor', 'white', ...
     'EdgeColor', 'black', 'FontSize', 9);

% Add legend
plot(NaN, NaN, 'b--', 'LineWidth', 2);
plot(NaN, NaN, 'g-', 'LineWidth', 3);
plot(NaN, NaN, 'y-', 'LineWidth', 3);
plot(NaN, NaN, 'y.', 'MarkerSize', 8);
legend({'Plume boundary', ...
        sprintf('Odor source (r=%.0fcm success)', success_radius), ...
        'Agent init zone', ...
        'Sample start positions'}, ...
       'Location', 'southeast');

% Set axis limits to show relevant area
xlim([-15, 15]);
ylim([-35, 5]);
axis equal;

% Add timestamp
timestamp_text = sprintf('Generated: %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
text(0.5, 0.01, timestamp_text, 'Units', 'normalized', ...
     'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', [0.5 0.5 0.5]);

% Save figure if requested
if ~isempty(output_file)
    fprintf('Saving validation figure to: %s\n', output_file);
    print(gcf, output_file, '-dpng', '-r150');  % Higher resolution
    fprintf('✓ Validation figure saved\n');
    
    % Also save parameters used
    [filepath, name, ~] = fileparts(output_file);
    params_file = fullfile(filepath, [name '_params.mat']);
    validation_params = struct();
    validation_params.success_radius = success_radius;
    validation_params.init_x_range = init_x_range;
    validation_params.init_y_range = init_y_range;
    validation_params.n_agents = n_agents;
    validation_params.source_position = [source_x, source_y];
    validation_params.middle_frame = middle_frame;
    validation_params.timestamp = datestr(now);
    save(params_file, 'validation_params');
    fprintf('✓ Parameters saved to: %s\n', params_file);
else
    fprintf('✓ Validation figure displayed\n');
end

end
