% plot_results.m - Create figures from navigation model results
%
% Usage: In MATLAB, run:
%        plot_results  (analyzes nav_results_0000.mat)
%        plot_results('results/nav_results_0001.mat')  (specific file)
%
% Creates multiple figures showing trajectories, success analysis, etc.

function plot_results(result_file)
    
    % Default to first result file
    if nargin < 1
        result_file = 'results/nav_results_0000.mat';
    end
    
    % Load results
    if ~exist(result_file, 'file')
        error('File not found: %s', result_file);
    end
    
    fprintf('Loading %s...\n', result_file);
    load(result_file);
    
    % Get dimensions
    [n_samples, n_agents] = size(out.x);
    
    %% Figure 1: All Trajectories
    figure('Name', 'All Agent Trajectories', 'Position', [100 100 800 600]);
    clf;
    
    % Determine colors based on success
    successful = ~isnan(out.latency);
    
    % Plot trajectories
    hold on;
    for i = 1:n_agents
        if successful(i)
            plot(out.x(:,i), out.y(:,i), 'g-', 'LineWidth', 1, 'Alpha', 0.5);
        else
            plot(out.x(:,i), out.y(:,i), 'r-', 'LineWidth', 0.5, 'Alpha', 0.3);
        end
    end
    
    % Mark start positions
    plot(out.x(1,:), out.y(1,:), 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k');
    
    % Mark target and success zone
    plot(0, 0, 'r*', 'MarkerSize', 20, 'LineWidth', 3);
    theta = linspace(0, 2*pi, 100);
    plot(2*cos(theta), 2*sin(theta), 'r--', 'LineWidth', 2);
    
    % Labels and formatting
    xlabel('X position (cm)');
    ylabel('Y position (cm)');
    title(sprintf('%d Agents - %.1f%% Success Rate', n_agents, out.successrate*100));
    grid on;
    axis equal;
    
    % Add legend
    h1 = plot(NaN, NaN, 'g-', 'LineWidth', 2);
    h2 = plot(NaN, NaN, 'r-', 'LineWidth', 2);
    legend([h1 h2], 'Successful', 'Failed', 'Location', 'best');
    
    %% Figure 2: Example Trajectory with Details
    figure('Name', 'Example Agent Trajectory', 'Position', [950 100 800 800]);
    clf;
    
    % Pick an interesting agent (first successful one, or just first)
    idx = find(successful, 1);
    if isempty(idx), idx = 1; end
    
    % Main trajectory plot
    subplot(3,1,1);
    plot(out.x(:,idx), out.y(:,idx), 'k-', 'LineWidth', 2);
    hold on;
    
    % Color code by odor intensity if available
    if isfield(out, 'odor')
        odor_thresh = 0.01;
        high_odor = out.odor(:,idx) > odor_thresh;
        scatter(out.x(high_odor,idx), out.y(high_odor,idx), 20, out.odor(high_odor,idx), 'filled');
        colorbar;
        colormap('hot');
    end
    
    % Mark start and end
    plot(out.x(1,idx), out.y(1,idx), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'LineWidth', 2);
    plot(out.x(end,idx), out.y(end,idx), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'LineWidth', 2);
    
    % Target
    plot(0, 0, 'k*', 'MarkerSize', 15, 'LineWidth', 2);
    plot(2*cos(theta), 2*sin(theta), 'k--', 'LineWidth', 1);
    
    xlabel('X position (cm)');
    ylabel('Y position (cm)');
    title(sprintf('Agent %d Trajectory', idx));
    grid on;
    axis equal;
    
    % Time series plots
    time_vec = (1:n_samples) / 15; % Convert to seconds
    
    % Heading over time
    subplot(3,1,2);
    plot(time_vec(1:end-1), out.theta(:,idx), 'b-', 'LineWidth', 1);
    ylabel('Heading (degrees)');
    xlabel('Time (seconds)');
    title('Heading Direction');
    grid on;
    ylim([-180 180]);
    
    % Sensory responses if available
    subplot(3,1,3);
    if isfield(out, 'odor') && isfield(out, 'ON') && isfield(out, 'OFF')
        plot(time_vec, out.odor(:,idx), 'k-', 'LineWidth', 1);
        hold on;
        plot(time_vec, out.ON(:,idx), 'm-', 'LineWidth', 2);
        plot(time_vec, out.OFF(:,idx), 'c-', 'LineWidth', 2);
        ylabel('Response');
        xlabel('Time (seconds)');
        title('Sensory Responses');
        legend('Odor', 'ON', 'OFF', 'Location', 'best');
        grid on;
    else
        % Just plot position over time
        yyaxis left;
        plot(time_vec, out.x(:,idx), 'b-');
        ylabel('X position (cm)');
        yyaxis right;
        plot(time_vec, out.y(:,idx), 'r-');
        ylabel('Y position (cm)');
        xlabel('Time (seconds)');
        title('Position over Time');
        grid on;
    end
    
    %% Figure 3: Performance Analysis
    figure('Name', 'Performance Analysis', 'Position', [100 600 1200 400]);
    clf;
    
    % Success pie chart
    subplot(1,3,1);
    n_success = sum(successful);
    n_fail = n_agents - n_success;
    pie([n_success, n_fail]);
    title('Success vs Failure');
    legend({'Successful', 'Failed'}, 'Location', 'best');
    
    % Latency histogram
    subplot(1,3,2);
    if any(successful)
        histogram(out.latency(successful), 10, 'FaceColor', 'b');
        xlabel('Time to reach target (s)');
        ylabel('Number of agents');
        title('Success Time Distribution');
        grid on;
        
        % Add mean line
        hold on;
        mean_lat = mean(out.latency(successful));
        yl = ylim;
        plot([mean_lat mean_lat], yl, 'r--', 'LineWidth', 2);
        text(mean_lat+2, yl(2)*0.8, sprintf('Mean: %.1fs', mean_lat), 'Color', 'r');
    else
        text(0.5, 0.5, 'No successful agents', 'HorizontalAlignment', 'center');
        axis off;
    end
    
    % Starting position vs success
    subplot(1,3,3);
    scatter(out.x(1,successful), out.y(1,successful), 100, 'g', 'filled', 'MarkerEdgeColor', 'k');
    hold on;
    scatter(out.x(1,~successful), out.y(1,~successful), 100, 'r', 'filled', 'MarkerEdgeColor', 'k');
    xlabel('Starting X (cm)');
    ylabel('Starting Y (cm)');
    title('Starting Position vs Success');
    legend('Successful', 'Failed', 'Location', 'best');
    grid on;
    axis equal;
    
    %% Figure 4: Summary Statistics
    figure('Name', 'Summary Statistics', 'Position', [950 500 600 500]);
    clf;
    
    % Calculate distances
    distances = zeros(n_agents, 1);
    for i = 1:n_agents
        dx = diff(out.x(:,i));
        dy = diff(out.y(:,i));
        distances(i) = sum(sqrt(dx.^2 + dy.^2));
    end
    
    % Distance histogram
    subplot(2,1,1);
    histogram(distances, 15, 'FaceColor', [0.3 0.3 0.3]);
    xlabel('Total distance traveled (cm)');
    ylabel('Number of agents');
    title('Distance Traveled Distribution');
    grid on;
    
    % Add mean line
    hold on;
    mean_dist = mean(distances);
    yl = ylim;
    plot([mean_dist mean_dist], yl, 'r--', 'LineWidth', 2);
    text(mean_dist+10, yl(2)*0.8, sprintf('Mean: %.1f cm', mean_dist), 'Color', 'r');
    
    % Text summary
    subplot(2,1,2);
    axis off;
    text_str = {
        sprintf('Environment: %s', out.environment);
        sprintf('Number of agents: %d', n_agents);
        sprintf('Simulation duration: %.1f seconds', n_samples/15);
        sprintf('Success rate: %.1f%%', out.successrate*100);
        sprintf('Successful agents: %d/%d', n_success, n_agents);
        '';
        sprintf('Mean distance traveled: %.1f cm', mean(distances));
        sprintf('Distance range: %.1f - %.1f cm', min(distances), max(distances));
        '';
        'Latency Statistics:';
    };
    
    if any(successful)
        text_str{end+1} = sprintf('  Mean: %.1f seconds', mean(out.latency(successful)));
        text_str{end+1} = sprintf('  Range: %.1f - %.1f seconds', ...
            min(out.latency(successful)), max(out.latency(successful)));
    else
        text_str{end+1} = '  No successful agents';
    end
    
    text(0.1, 0.9, text_str, 'VerticalAlignment', 'top', 'FontSize', 12, 'FontName', 'Courier');
    title('Summary Statistics', 'FontSize', 14);
    
    %% Save figures
    [pathstr, name, ~] = fileparts(result_file);
    
    % Save as PNG
    figure(1); saveas(gcf, fullfile(pathstr, [name '_trajectories.pdf']));
    figure(2); saveas(gcf, fullfile(pathstr, [name '_example.pdf']));
    figure(3); saveas(gcf, fullfile(pathstr, [name '_performance.pdf']));
    figure(4); saveas(gcf, fullfile(pathstr, [name '_summary.pdf']));
    
    fprintf('\nFigures saved to %s directory\n', pathstr);
    fprintf('Files created:\n');
    fprintf('  %s_trajectories.png\n', name);
    fprintf('  %s_example.png\n', name);
    fprintf('  %s_performance.png\n', name);
    fprintf('  %s_summary.png\n', name);
end