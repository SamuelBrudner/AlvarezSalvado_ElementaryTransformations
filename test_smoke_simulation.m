% test_smoke_simulation.m - Test smoke plume simulation

% Store current directory
original_dir = pwd;
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
cd(script_dir);

% Add Code directory
if exist('Code', 'dir')
    addpath(genpath('Code'));
end

fprintf('\n=== Testing Smoke Plume Simulation ===\n\n');

try
    [plume_file, plume_config] = get_plume_file();
    fprintf('Active plume: %s\n', plume_config.plume_id);
    fprintf('Frame rate: %.1f Hz\n', plume_config.temporal.frame_rate);
    
    % 10 second test
    test_frames = round(10 * plume_config.temporal.frame_rate);
    
    fprintf('\nRunning 10-second test (%d frames)...\n', test_frames);
    
    out = navigation_model_vec(test_frames, 'Crimaldi', 0, 2);
    fprintf('✓ Success! Generated %d time points\n', size(out.x, 1));
    
    % Visualize
    figure;
    plot(out.x(:,1), out.y(:,1), 'b-', 'LineWidth', 2);
    hold on;
    if size(out.x, 2) > 1
        plot(out.x(:,2), out.y(:,2), 'r-', 'LineWidth', 2);
    end
    
    % Source
    plot(plume_config.simulation.source_position.x_cm, ...
         plume_config.simulation.source_position.y_cm, ...
         'r*', 'MarkerSize', 15, 'LineWidth', 2);
    
    xlabel('X (cm)'); ylabel('Y (cm)');
    title(sprintf('Test - %s', plume_config.plume_id));
    axis equal; grid on;
    
catch ME
    fprintf('Error: %s\n', ME.message);
end

cd(original_dir);
