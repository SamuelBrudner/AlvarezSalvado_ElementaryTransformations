% analyze_results.m - Analyze and visualize navigation model results
%
% Usage: Run this script in MATLAB from the project root directory
%        Analyzes the first result file (nav_results_0000.mat)
%        Creates multiple figures and saves a summary
%
% Outputs: - Multiple figure windows with trajectory and performance plots
%          - Text summary file with statistics

clear; close all;

%% Load Results
result_file = 'results/nav_results_0000.mat';
if ~exist(result_file, 'file')
    error('Results file not found. Run simulation first.');
end

load(result_file);
fprintf('Loaded results from: %s\n\n', result_file);

%% Display Basic Information
fprintf('=== Simulation Overview ===\n');
fprintf('Environment: %s\n', out.environment);
fprintf('Number of agents: %d\n', size(out.x, 2));
fprintf('Trajectory length: %d samples\n', size(out.x, 1));
fprintf('Time duration: %.1f seconds (at 15 Hz)\n', size(out.x, 1) / 15);

% Success metrics (if available)
if isfield(out, 'successrate')
    fprintf('\n=== Performance Metrics ===\n');
    fprintf('Success rate: %.1f%% (%d/%d agents)\n', ...
        out.successrate * 100, ...
        sum(~isnan(out.latency)), ...
        size(out.x, 2));
end

if isfield(out, 'latency')
    valid_latencies = out.latency(~isnan(out.latency));
    if ~isempty(valid_latencies)
        fprintf('Mean latency to target: %.1f seconds\n', mean(valid_latencies));
        fprintf('Median latency: %.1f seconds\n', median(valid_latencies));
        fprintf('Range: %.1f - %.1f seconds\n', min(valid_latencies), max(valid_latencies));
    end
end

%% Analyze Trajectories
fprintf('\n=== Trajectory Analysis ===\n');

% Starting positions
start_x = out.x(1, :);
start_y = out.y(1, :);
fprintf('Starting positions:\n');
fprintf('  X range: [%.1f, %.1f] cm\n', min(start_x), max(start_x));
fprintf('  Y range: [%.1f, %.1f] cm\n', min(start_y), max(start_y));

% Ending positions
end_x = out.x(end, :);
end_y = out.y(end, :);
fprintf('Ending positions:\n');
fprintf('  X range: [%.1f, %.1f] cm\n', min(end_x), max(end_x));
fprintf('  Y range: [%.1f, %.1f] cm\n', min(end_y), max(end_y));

% Distance traveled
distances = zeros(1, size(out.x, 2));
for i = 1:size(out.x, 2)
    dx = diff(out.x(:, i));
    dy = diff(out.y(:, i));
    distances(i) = sum(sqrt(dx.^2 + dy.^2));
end
fprintf('\nDistance traveled:\n');
fprintf('  Mean: %.1f cm\n', mean(distances));
fprintf('  Range: [%.1f, %.1f] cm\n', min(distances), max(distances));

%% Visualizations

% 1. All trajectories
figure('Name', 'All Trajectories', 'Position', [100 100 600 800]);
hold on;
for i = 1:size(out.x, 2)
    plot(out.x(:, i), out.y(:, i), 'Color', [0.5 0.5 0.5 0.3], 'LineWidth', 0.5);
end
plot(0, 0, 'r*', 'MarkerSize', 15, 'LineWidth', 2); % Target
xlabel('X position (cm)');
ylabel('Y position (cm)');
title(sprintf('All %d Agent Trajectories', size(out.x, 2)));
axis equal;
grid on;

% Add success zone
theta = linspace(0, 2*pi, 100);
plot(2*cos(theta), 2*sin(theta), 'r--', 'LineWidth', 2);
text(3, 3, 'Success zone (2cm)', 'Color', 'red');

% 2. Example trajectory with odor encounters
figure('Name', 'Example Trajectory', 'Position', [750 100 600 800]);
agent_idx = 1; % First agent
subplot(3, 1, 1);
plot(out.x(:, agent_idx), out.y(:, agent_idx), 'k-', 'LineWidth', 1.5);
hold on;
plot(out.x(1, agent_idx), out.y(1, agent_idx), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
plot(out.x(end, agent_idx), out.y(end, agent_idx), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
xlabel('X position (cm)');
ylabel('Y position (cm)');
title(sprintf('Agent %d Trajectory', agent_idx));
axis equal;
grid on;

% Plot odor encounters
if isfield(out, 'odor')
    odor_thresh = 0.01; % Threshold for odor detection
    odor_idx = find(out.odor(:, agent_idx) > odor_thresh);
    if ~isempty(odor_idx)
        plot(out.x(odor_idx, agent_idx), out.y(odor_idx, agent_idx), 'm.', 'MarkerSize', 8);
        legend('Trajectory', 'Start', 'End', 'Odor encounters', 'Location', 'best');
    end
end

% 3. Odor and behavioral responses
if isfield(out, 'odor') && isfield(out, 'ON') && isfield(out, 'OFF')
    time_vec = (1:size(out.odor, 1)) / 15; % Convert to seconds (15 Hz)
    
    subplot(3, 1, 2);
    plot(time_vec, out.odor(:, agent_idx), 'k-');
    hold on;
    plot(time_vec, out.ON(:, agent_idx), 'm-', 'LineWidth', 1.5);
    plot(time_vec, out.OFF(:, agent_idx), 'c-', 'LineWidth', 1.5);
    xlabel('Time (seconds)');
    ylabel('Response');
    title('Sensory Responses');
    legend('Odor', 'ON response', 'OFF response', 'Location', 'best');
    xlim([0 max(time_vec)]);
    
    % Heading over time
    subplot(3, 1, 3);
    plot(time_vec(1:end-1), out.theta(:, agent_idx), 'b-');
    xlabel('Time (seconds)');
    ylabel('Heading (degrees)');
    title('Heading Direction');
    xlim([0 max(time_vec)]);
    ylim([-180 180]);
end

% 4. Success analysis
if isfield(out, 'latency')
    figure('Name', 'Success Analysis', 'Position', [100 550 800 400]);
    
    subplot(1, 2, 1);
    successful = ~isnan(out.latency);
    bar([sum(successful), sum(~successful)]);
    set(gca, 'XTickLabel', {'Successful', 'Failed'});
    ylabel('Number of agents');
    title('Success vs Failure');
    
    subplot(1, 2, 2);
    valid_latencies = out.latency(~isnan(out.latency));
    if ~isempty(valid_latencies)
        histogram(valid_latencies, 10);
        xlabel('Time to reach target (seconds)');
        ylabel('Count');
        title('Latency Distribution');
    end
end

% 5. Starting position vs success
figure('Name', 'Position vs Success', 'Position', [950 550 600 400]);
if isfield(out, 'latency')
    successful = ~isnan(out.latency);
    scatter(start_x(successful), start_y(successful), 50, 'g', 'filled');
    hold on;
    scatter(start_x(~successful), start_y(~successful), 50, 'r', 'filled');
    xlabel('Starting X position (cm)');
    ylabel('Starting Y position (cm)');
    title('Starting Position vs Success');
    legend('Successful', 'Failed', 'Location', 'best');
    grid on;
else
    scatter(start_x, start_y, 50, 'b', 'filled');
    xlabel('Starting X position (cm)');
    ylabel('Starting Y position (cm)');
    title('Starting Positions');
    grid on;
end

%% Save Summary
summary_file = strrep(result_file, '.mat', '_summary.txt');
fid = fopen(summary_file, 'w');
fprintf(fid, 'Navigation Model Results Summary\n');
fprintf(fid, '================================\n\n');
fprintf(fid, 'File: %s\n', result_file);
fprintf(fid, 'Date analyzed: %s\n\n', datestr(now));
fprintf(fid, 'Environment: %s\n', out.environment);
fprintf(fid, 'Number of agents: %d\n', size(out.x, 2));
fprintf(fid, 'Trajectory length: %d samples (%.1f seconds)\n', size(out.x, 1), size(out.x, 1)/15);

if isfield(out, 'successrate')
    fprintf(fid, '\nPerformance:\n');
    fprintf(fid, 'Success rate: %.1f%%\n', out.successrate * 100);
    if ~isempty(valid_latencies)
        fprintf(fid, 'Mean latency: %.1f seconds\n', mean(valid_latencies));
    end
end

fprintf(fid, '\nTrajectory stats:\n');
fprintf(fid, 'Mean distance traveled: %.1f cm\n', mean(distances));
fprintf(fid, 'Starting Y range: [%.1f, %.1f] cm\n', min(start_y), max(start_y));
fprintf(fid, 'Ending Y range: [%.1f, %.1f] cm\n', min(end_y), max(end_y));
fclose(fid);

fprintf('\n✓ Summary saved to: %s\n', summary_file);
fprintf('✓ Figures created for visualization\n');