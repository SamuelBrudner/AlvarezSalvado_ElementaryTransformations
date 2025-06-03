% Add paths
script_dir = getenv('SCRIPT_DIR');
test_dir = getenv('TEST_DIR');
environment = getenv('TEST_ENVIRONMENT');
num_agents = str2double(getenv('TEST_NUM_AGENTS'));
trial_length = str2double(getenv('TEST_TRIAL_LENGTH'));

addpath(genpath(fullfile(script_dir, 'Code')));

% Set up diary to capture output
diary(fullfile(test_dir, 'simulation_output.txt'));

fprintf('\n=== Test Run Started at %s ===\n', datestr(now));
fprintf('Environment: %s\n', environment);
fprintf('Number of agents: %d\n', num_agents);
fprintf('Trial length: %d\n\n', trial_length);

try
    % Run the simulation
    tic;
    out = navigation_model_vec(trial_length, environment, 2, num_agents);
    elapsed = toc;
    
    fprintf('\nSimulation completed successfully in %.2f seconds\n', elapsed);
    
    % Display summary statistics
    fprintf('\n--- Summary Statistics ---\n');
    fprintf('Number of trajectories: %d\n', size(out.x, 2));
    fprintf('X position range: [%.2f, %.2f] cm\n', min(out.x(:)), max(out.x(:)));
    fprintf('Y position range: [%.2f, %.2f] cm\n', min(out.y(:)), max(out.y(:)));
    
    if isfield(out, 'successrate')
        fprintf('Success rate: %.2f%%\n', out.successrate * 100);
    end
    
    if isfield(out, 'latency')
        valid_latencies = out.latency(~isnan(out.latency));
        if ~isempty(valid_latencies)
            fprintf('Mean latency: %.2f s\n', mean(valid_latencies));
        end
    end
    
    % Save results
    save(fullfile(test_dir, 'test_results.mat'), 'out');
    fprintf('\nResults saved to test_results.mat\n');
    
    % Save a sample trajectory plot
    if size(out.x, 2) >= 1
        figure('Visible', 'off');
        plot(out.x(:,1), out.y(:,1), 'k-', 'LineWidth', 1.5);
        hold on;
        plot(out.x(1,1), out.y(1,1), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
        plot(out.x(end,1), out.y(end,1), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        xlabel('X position (cm)');
        ylabel('Y position (cm)');
        title(sprintf('Sample trajectory - %s environment', environment));
        axis equal;
        grid on;
        saveas(gcf, fullfile(test_dir, 'sample_trajectory.png'));
        fprintf('Sample trajectory saved to sample_trajectory.png\n');
    end
    
catch ME
    fprintf('\nERROR: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  In %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    diary off;
    exit(1);
end

diary off;
exit(0);
