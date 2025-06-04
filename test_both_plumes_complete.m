% test_both_plumes_complete.m - Comprehensive test of navigation in both plumes

% Helper function for output flushing (compatible with older MATLAB)
flush_output = @() evalc('diary off; diary on');

fprintf('=== Testing Navigation in Both Plumes ===\n\n');
flush_output();  % Force flush header

%% Load configurations
crim_cfg = jsondecode(fileread('configs/plumes/crimaldi_10cms_bounded.json'));
smoke_cfg = jsondecode(fileread('configs/plumes/smoke_1a_backgroundsubtracted.json'));

% Get shared parameters
init_x = crim_cfg.simulation.agent_initialization.x_range_cm;
init_y = crim_cfg.simulation.agent_initialization.y_range_cm;
n_agents = 50;  % Run a decent number for statistics
success_radius = crim_cfg.simulation.success_radius_cm;

fprintf('Test parameters:\n');
fprintf('  Agents: %d\n', n_agents);
fprintf('  Init zone: X ∈ [%.1f, %.1f], Y ∈ [%.1f, %.1f] cm\n', ...
        init_x(1), init_x(2), init_y(1), init_y(2));
fprintf('  Success radius: %.1f cm\n', success_radius);
flush_output();  % Force flush parameters

%% Test 1: Crimaldi Plume
fprintf('\n1. Testing Crimaldi plume...\n');
fprintf('   Duration: %.0f seconds (%.0f frames at %d Hz)\n', ...
        crim_cfg.simulation.duration_seconds, ...
        crim_cfg.simulation.duration_seconds * crim_cfg.temporal.frame_rate, ...
        crim_cfg.temporal.frame_rate);
flush_output();  % Force flush before simulation

% Clear any environment override
setenv('MATLAB_PLUME_FILE', '');

% Run simulation
tic;
out_crim = Elifenavmodel_bilateral(crim_cfg.simulation.duration_seconds * crim_cfg.temporal.frame_rate, ...
                                   'Crimaldi', 0, n_agents);
crim_time = toc;

% Extract results
crim_success_rate = out_crim.successrate;
crim_latencies = out_crim.latency;
crim_avg_latency = nanmean(crim_latencies);

fprintf('   ✓ Completed in %.1f seconds\n', crim_time);
fprintf('   Success rate: %.1f%% (%d/%d agents)\n', ...
        crim_success_rate * 100, sum(~isnan(crim_latencies)), n_agents);
fprintf('   Average latency: %.1f seconds\n', crim_avg_latency);
flush_output();  % Force flush Crimaldi results

%% Test 2: Smoke Plume
fprintf('\n2. Testing Smoke plume...\n');
fprintf('   Duration: %.0f seconds (%.0f frames at %d Hz)\n', ...
        smoke_cfg.simulation.duration_seconds, ...
        smoke_cfg.simulation.duration_seconds * smoke_cfg.temporal.frame_rate, ...
        smoke_cfg.temporal.frame_rate);
flush_output();  % Force flush before smoke simulation

% Set smoke plume file
setenv('MATLAB_PLUME_FILE', smoke_cfg.data_path.path);

% Run simulation
% Note: We use 'Crimaldi' environment type because it handles HDF5 plume data
% The actual frame rate is determined by the config
tic;
out_smoke = Elifenavmodel_bilateral(smoke_cfg.simulation.duration_seconds * smoke_cfg.temporal.frame_rate, ...
                                    'Crimaldi', 0, n_agents);
smoke_time = toc;

% Extract results
smoke_success_rate = out_smoke.successrate;
smoke_latencies = out_smoke.latency;
smoke_avg_latency = nanmean(smoke_latencies);

fprintf('   ✓ Completed in %.1f seconds\n', smoke_time);
fprintf('   Success rate: %.1f%% (%d/%d agents)\n', ...
        smoke_success_rate * 100, sum(~isnan(smoke_latencies)), n_agents);
fprintf('   Average latency: %.1f seconds\n', smoke_avg_latency);
flush_output();  % Force flush smoke results

%% Save results
fprintf('\n3. Saving results...\n');
flush_output();  % Force flush before saving

if ~exist('results', 'dir')
    mkdir('results');
end

% Save mat files
save('results/crimaldi_test_results.mat', 'out_crim', 'crim_cfg', '-v7.3');
save('results/smoke_test_results.mat', 'out_smoke', 'smoke_cfg', '-v7.3');
fprintf('   ✓ Saved .mat files\n');
flush_output();  % Force flush after saving

%% Create visualization
fprintf('\n4. Creating visualizations...\n');
flush_output();  % Force flush before visualization

% Figure 1: Trajectories comparison
figure('Position', [100 100 1400 600]);

% Crimaldi trajectories
subplot(1,2,1);
plot(out_crim.x, out_crim.y, '-', 'LineWidth', 0.5, 'Color', [0 0 1 0.3]);
hold on;
plot(out_crim.start(:,1), out_crim.start(:,2), 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
viscircles([0, 0], success_radius, 'Color', 'g', 'LineWidth', 2);
plot(0, 0, 'g*', 'MarkerSize', 15);
rectangle('Position', [crim_cfg.spatial.arena_bounds.x_min, ...
                      crim_cfg.spatial.arena_bounds.y_min, ...
                      crim_cfg.spatial.arena_bounds.x_max - crim_cfg.spatial.arena_bounds.x_min, ...
                      crim_cfg.spatial.arena_bounds.y_max - crim_cfg.spatial.arena_bounds.y_min], ...
          'EdgeColor', 'k', 'LineWidth', 2);
title(sprintf('Crimaldi: %.1f%% success', crim_success_rate*100));
xlabel('X (cm)'); ylabel('Y (cm)');
axis equal; xlim([-10, 10]); ylim([-32, 2]);
grid on;

% Smoke trajectories
subplot(1,2,2);
plot(out_smoke.x, out_smoke.y, '-', 'LineWidth', 0.5, 'Color', [1 0 0 0.3]);
hold on;
plot(out_smoke.start(:,1), out_smoke.start(:,2), 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
viscircles([0, 0], success_radius, 'Color', 'g', 'LineWidth', 2);
plot(0, 0, 'g*', 'MarkerSize', 15);
rectangle('Position', [smoke_cfg.spatial.arena_bounds.x_min, ...
                      smoke_cfg.spatial.arena_bounds.y_min, ...
                      smoke_cfg.spatial.arena_bounds.x_max - smoke_cfg.spatial.arena_bounds.x_min, ...
                      smoke_cfg.spatial.arena_bounds.y_max - smoke_cfg.spatial.arena_bounds.y_min], ...
          'EdgeColor', 'k', 'LineWidth', 2);
title(sprintf('Smoke: %.1f%% success', smoke_success_rate*100));
xlabel('X (cm)'); ylabel('Y (cm)');
axis equal; xlim([-10, 10]); ylim([-28, 2]);
grid on;

sgtitle(sprintf('Navigation Test Results (%d agents each)', n_agents), 'FontSize', 16);
saveas(gcf, 'results/trajectory_comparison.png');

% Figure 2: Statistics comparison
figure('Position', [100 100 800 600]);

% Success rates
subplot(2,2,1);
bar([crim_success_rate, smoke_success_rate] * 100);
set(gca, 'XTickLabel', {'Crimaldi', 'Smoke'});
ylabel('Success Rate (%)');
title('Success Rates');
ylim([0 100]);
grid on;

% Latency distributions
subplot(2,2,2);
edges = 0:5:max([crim_latencies(:); smoke_latencies(:)]);
histogram(crim_latencies(~isnan(crim_latencies)), edges, 'FaceColor', 'b', 'FaceAlpha', 0.5);
hold on;
histogram(smoke_latencies(~isnan(smoke_latencies)), edges, 'FaceColor', 'r', 'FaceAlpha', 0.5);
xlabel('Latency (seconds)');
ylabel('Count');
title('Success Latencies');
legend('Crimaldi', 'Smoke');
grid on;

% Starting positions
subplot(2,2,3);
plot(out_crim.start(:,1), out_crim.start(:,2), 'bo', 'MarkerSize', 6);
hold on;
plot(out_smoke.start(:,1), out_smoke.start(:,2), 'r+', 'MarkerSize', 8);
rectangle('Position', [init_x(1), init_y(1), diff(init_x), diff(init_y)], ...
          'EdgeColor', 'k', 'LineWidth', 2, 'LineStyle', '--');
xlabel('X (cm)'); ylabel('Y (cm)');
title('Starting Positions');
legend('Crimaldi', 'Smoke');
axis equal;
xlim([-10, 10]); ylim([-28, -20]);
grid on;

% Summary statistics
subplot(2,2,4);
axis off;
text(0.1, 0.9, 'Summary Statistics', 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.7, sprintf('Crimaldi:'), 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.6, sprintf('  Success: %.1f%% (%d/%d)', ...
     crim_success_rate*100, sum(~isnan(crim_latencies)), n_agents));
text(0.1, 0.5, sprintf('  Avg latency: %.1f s', crim_avg_latency));
text(0.1, 0.4, sprintf('  Runtime: %.1f s', crim_time));
text(0.1, 0.2, sprintf('Smoke:'), 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.1, sprintf('  Success: %.1f%% (%d/%d)', ...
     smoke_success_rate*100, sum(~isnan(smoke_latencies)), n_agents));
text(0.1, 0.0, sprintf('  Avg latency: %.1f s', smoke_avg_latency));
text(0.1, -0.1, sprintf('  Runtime: %.1f s', smoke_time));

sgtitle('Navigation Performance Comparison', 'FontSize', 16);
saveas(gcf, 'results/statistics_comparison.png');

% Figure 3: Time series for a few example agents
figure('Position', [100 100 1200 800]);
n_examples = min(5, n_agents);

for i = 1:n_examples
    % Crimaldi examples
    subplot(2, n_examples, i);
    plot(out_crim.x(:,i), out_crim.y(:,i), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(out_crim.x(1,i), out_crim.y(1,i), 'ro', 'MarkerSize', 8);
    plot(out_crim.x(end,i), out_crim.y(end,i), 'ks', 'MarkerSize', 8);
    viscircles([0, 0], success_radius, 'Color', 'g', 'LineWidth', 1);
    title(sprintf('Agent %d', i));
    if i == 1
        ylabel('Crimaldi', 'FontSize', 12, 'FontWeight', 'bold');
    end
    axis equal;
    xlim([-10, 10]); ylim([-32, 2]);
    grid on;
    
    % Smoke examples
    subplot(2, n_examples, n_examples + i);
    plot(out_smoke.x(:,i), out_smoke.y(:,i), 'r-', 'LineWidth', 1.5);
    hold on;
    plot(out_smoke.x(1,i), out_smoke.y(1,i), 'ro', 'MarkerSize', 8);
    plot(out_smoke.x(end,i), out_smoke.y(end,i), 'ks', 'MarkerSize', 8);
    viscircles([0, 0], success_radius, 'Color', 'g', 'LineWidth', 1);
    if i == 1
        ylabel('Smoke', 'FontSize', 12, 'FontWeight', 'bold');
    end
    xlabel('X (cm)');
    axis equal;
    xlim([-10, 10]); ylim([-28, 2]);
    grid on;
end

sgtitle('Example Agent Trajectories', 'FontSize', 16);
saveas(gcf, 'results/example_trajectories.png');

fprintf('   ✓ Saved visualizations\n');
flush_output();  % Force flush after all visualizations

%% Final summary
fprintf('\n=== Test Summary ===\n');
fprintf('Both plumes tested with %d agents\n', n_agents);
fprintf('Initialization zone: Y ∈ [%.1f, %.1f] cm\n', init_y(1), init_y(2));
fprintf('\nResults:\n');
fprintf('  Crimaldi: %.1f%% success, %.1f s avg latency\n', ...
        crim_success_rate*100, crim_avg_latency);
fprintf('  Smoke:    %.1f%% success, %.1f s avg latency\n', ...
        smoke_success_rate*100, smoke_avg_latency);
fprintf('\nSaved outputs:\n');
fprintf('  - results/crimaldi_test_results.mat\n');
fprintf('  - results/smoke_test_results.mat\n');
fprintf('  - results/trajectory_comparison.png\n');
fprintf('  - results/statistics_comparison.png\n');
fprintf('  - results/example_trajectories.png\n');
flush_output();  % Force flush final summary

% Clear environment variable
setenv('MATLAB_PLUME_FILE', '');
fprintf('\n✓ Test complete!\n');
flush_output();  % Force flush completion message