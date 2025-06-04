% plot_odor_distance_diagnostic.m - Plot odor intensity and distance to source over time

function plot_odor_distance_diagnostic(n_agents_to_show)
% PLOT_ODOR_DISTANCE_DIAGNOSTIC Create diagnostic plots showing odor and distance
%   plot_odor_distance_diagnostic() - Shows 5 agents by default
%   plot_odor_distance_diagnostic(N) - Shows N agents

if nargin < 1
    n_agents_to_show = 5;  % Default to showing 5 agents
end

if ~exist('results','dir')
    fprintf('Creating results directory...\n');
    mkdir('results');
end

fprintf('Creating odor/distance diagnostic plots...\n');

%% Load data
try
    crim_data = load('results/crimaldi_test_results.mat');
    smoke_data = load('results/smoke_test_results.mat');
catch ME
    error('Could not load results. Run test_both_plumes_complete.m first!');
end
fprintf('Loaded diagnostic result files.\n');

crim_out = crim_data.out_crim;
smoke_out = smoke_data.out_smoke;

% Get number of agents to plot
n_agents_crim = min(n_agents_to_show, size(crim_out.x, 2));
n_agents_smoke = min(n_agents_to_show, size(smoke_out.x, 2));

% Get time vectors
t_crim = (0:size(crim_out.x,1)-1) / crim_data.crim_cfg.temporal.frame_rate;
t_crim = t_crim(:); % ensure column vector for fill()
t_smoke = (0:size(smoke_out.x,1)-1) / smoke_data.smoke_cfg.temporal.frame_rate;
t_smoke = t_smoke(:); % ensure column vector

% Calculate distances to source
dist_crim = sqrt(crim_out.x.^2 + crim_out.y.^2);
dist_smoke = sqrt(smoke_out.x.^2 + smoke_out.y.^2);

success_radius = crim_data.crim_cfg.simulation.success_radius_cm;

%% Figure 1: Individual agent plots for Crimaldi
fprintf('Generating Crimaldi individual plots...\n');
figure('Position', [50 50 1400 900]);
sgtitle('Crimaldi Plume - Odor Intensity & Distance to Source', 'FontSize', 16);

for i = 1:n_agents_crim
    subplot(n_agents_crim, 1, i);
    
    % Create two y-axes
    yyaxis left
    plot(t_crim, crim_out.odor(:,i), 'b-', 'LineWidth', 1.5);
    ylabel('Odor Intensity', 'Color', 'b');
    max_odor = max(max(crim_out.odor(:,1:n_agents_crim)));
    if max_odor > 0
        ylim([0 max_odor*1.1]);
    else
        ylim([0 1]);
    end
    set(gca, 'YColor', 'b');
    
    yyaxis right
    plot(t_crim, dist_crim(:,i), 'r-', 'LineWidth', 1.5);
    hold on;
    plot([0 max(t_crim)], [success_radius success_radius], 'g--', 'LineWidth', 1);
    ylabel('Distance to Source (cm)', 'Color', 'r');
    ylim([0 max(max(dist_crim(:,1:n_agents_crim)))*1.1]);
    set(gca, 'YColor', 'r');
    
    % Mark if successful
    if ~isnan(crim_out.latency(i))
        success_time = crim_out.latency(i);
        plot(success_time, success_radius, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        title(sprintf('Agent %d - SUCCESS at %.1f s', i, success_time));
    else
        title(sprintf('Agent %d - No success', i));
    end
    
    xlabel('Time (s)');
    grid on;
    xlim([0 max(t_crim)]);
end

fprintf('Saving Crimaldi figure...\n');
saveas(gcf, 'results/odor_distance_crimaldi.png');

%% Figure 2: Individual agent plots for Smoke
fprintf('Generating Smoke individual plots...\n');
figure('Position', [100 100 1400 900]);
sgtitle('Smoke Plume - Odor Intensity & Distance to Source', 'FontSize', 16);

for i = 1:n_agents_smoke
    subplot(n_agents_smoke, 1, i);
    
    % Create two y-axes
    yyaxis left
    plot(t_smoke, smoke_out.odor(:,i), 'b-', 'LineWidth', 1.5);
    ylabel('Odor Intensity', 'Color', 'b');
    max_odor = max(max(smoke_out.odor(:,1:n_agents_smoke)));
    if max_odor > 0
        ylim([0 max_odor*1.1]);
    else
        ylim([0 1]);
    end
    set(gca, 'YColor', 'b');
    
    yyaxis right
    plot(t_smoke, dist_smoke(:,i), 'r-', 'LineWidth', 1.5);
    hold on;
    plot([0 max(t_smoke)], [success_radius success_radius], 'g--', 'LineWidth', 1);
    ylabel('Distance to Source (cm)', 'Color', 'r');
    ylim([0 max(max(dist_smoke(:,1:n_agents_smoke)))*1.1]);
    set(gca, 'YColor', 'r');
    
    % Mark if successful
    if ~isnan(smoke_out.latency(i))
        success_time = smoke_out.latency(i);
        plot(success_time, success_radius, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        title(sprintf('Agent %d - SUCCESS at %.1f s', i, success_time));
    else
        title(sprintf('Agent %d - No success', i));
    end
    
    xlabel('Time (s)');
    grid on;
    xlim([0 max(t_smoke)]);
end

fprintf('Saving Smoke figure...\n');
saveas(gcf, 'results/odor_distance_smoke.png');

%% Figure 3: Overlay plot showing all agents together
fprintf('Generating overlay comparison...\n');
figure('Position', [150 150 1400 700]);

% Crimaldi overlay
subplot(1,2,1);
hold on;

% Plot distance for all agents (thin lines)
for i = 1:size(crim_out.x, 2)
    if ~isnan(crim_out.latency(i))
        plot(t_crim, dist_crim(:,i), '-', 'Color', [0.7 0.2 0.2 0.3], 'LineWidth', 0.5);
    else
        plot(t_crim, dist_crim(:,i), '-', 'Color', [0.7 0.7 0.7 0.3], 'LineWidth', 0.5);
    end
end

% Highlight a few agents with odor
colors = lines(n_agents_crim);
for i = 1:n_agents_crim
    % Scale odor to distance range for visualization
    max_odor = max(crim_out.odor(:));
    if max_odor > 0
        odor_scaled = crim_out.odor(:,i) * 30 / max_odor;  % Scale to ~30 cm
    else
        odor_scaled = crim_out.odor(:,i);  % Keep as is if no odor
    end
    plot(t_crim, odor_scaled, '-', 'Color', [colors(i,:) 0.7], 'LineWidth', 2);
end

plot([0 max(t_crim)], [success_radius success_radius], 'g--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Distance to Source (cm) / Scaled Odor');
title(sprintf('Crimaldi - All %d Agents', size(crim_out.x, 2)));
ylim([0 40]);
grid on;
legend({'Failed trajectories', 'Successful trajectories', 'Odor (scaled)', 'Success radius'}, ...
       'Location', 'best');

% Smoke overlay
subplot(1,2,2);
hold on;

colors_smoke = lines(n_agents_smoke);

% Plot distance for all agents (thin lines)
for i = 1:size(smoke_out.x, 2)
    if ~isnan(smoke_out.latency(i))
        plot(t_smoke, dist_smoke(:,i), '-', 'Color', [0.7 0.2 0.2 0.3], 'LineWidth', 0.5);
    else
        plot(t_smoke, dist_smoke(:,i), '-', 'Color', [0.7 0.7 0.7 0.3], 'LineWidth', 0.5);
    end
end

% Highlight a few agents with odor
for i = 1:n_agents_smoke
    % Scale odor to distance range for visualization
    max_odor = max(smoke_out.odor(:));
    if max_odor > 0
        odor_scaled = smoke_out.odor(:,i) * 30 / max_odor;  % Scale to ~30 cm
    else
        odor_scaled = smoke_out.odor(:,i);  % Keep as is if no odor
    end
    plot(t_smoke, odor_scaled, '-', 'Color', [colors_smoke(i,:) 0.7], 'LineWidth', 2);
end

plot([0 max(t_smoke)], [success_radius success_radius], 'g--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Distance to Source (cm) / Scaled Odor');
title(sprintf('Smoke - All %d Agents', size(smoke_out.x, 2)));
ylim([0 40]);
grid on;

sgtitle('Distance & Odor Trajectories - All Agents', 'FontSize', 16);
fprintf('Saving overlay figure...\n');
saveas(gcf, 'results/odor_distance_overlay.png');

%% Figure 4: Statistical summary plot
fprintf('Generating statistical summary...\n');
figure('Position', [200 200 1200 800]);

% Crimaldi statistics
subplot(2,2,1);
hold on;
% Plot mean distance with confidence intervals
mean_dist_crim = mean(dist_crim, 2);
std_dist_crim = std(dist_crim, 0, 2);
fill([t_crim; flipud(t_crim)], ...
     [mean_dist_crim - std_dist_crim; flipud(mean_dist_crim + std_dist_crim)], ...
     'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(t_crim, mean_dist_crim, 'r-', 'LineWidth', 2);

% Plot mean odor
mean_odor_crim = mean(crim_out.odor, 2);
if max(mean_odor_crim) > 0
    plot(t_crim, mean_odor_crim * 30 / max(mean_odor_crim), 'b-', 'LineWidth', 2);
else
    plot(t_crim, mean_odor_crim, 'b-', 'LineWidth', 2);
end

plot([0 max(t_crim)], [success_radius success_radius], 'g--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Distance (cm) / Scaled Odor');
title('Crimaldi - Mean ± STD');
legend('Distance ± STD', 'Mean Distance', 'Mean Odor (scaled)', 'Success radius');
grid on;
ylim([0 40]);

% Smoke statistics
subplot(2,2,2);
hold on;
% Plot mean distance with confidence intervals
mean_dist_smoke = mean(dist_smoke, 2);
std_dist_smoke = std(dist_smoke, 0, 2);
fill([t_smoke; flipud(t_smoke)], ...
     [mean_dist_smoke - std_dist_smoke; flipud(mean_dist_smoke + std_dist_smoke)], ...
     'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(t_smoke, mean_dist_smoke, 'r-', 'LineWidth', 2);

% Plot mean odor
mean_odor_smoke = mean(smoke_out.odor, 2);
if max(mean_odor_smoke) > 0
    plot(t_smoke, mean_odor_smoke * 30 / max(mean_odor_smoke), 'b-', 'LineWidth', 2);
else
    plot(t_smoke, mean_odor_smoke, 'b-', 'LineWidth', 2);
end

plot([0 max(t_smoke)], [success_radius success_radius], 'g--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Distance (cm) / Scaled Odor');
title('Smoke - Mean ± STD');
grid on;
ylim([0 40]);

% Correlation analysis - Crimaldi
subplot(2,2,3);
% Calculate correlation between odor and distance change
correlation_window = 15;  % frames
correlations_crim = zeros(n_agents_crim, 1);
for i = 1:n_agents_crim
    odor_smooth = smooth(crim_out.odor(:,i), correlation_window);
    dist_change = -diff(smooth(dist_crim(:,i), correlation_window));  % Negative because we want approach
    valid_idx = odor_smooth(1:end-1) > 0.01;  % Only when in odor
    if sum(valid_idx) > 10
        correlations_crim(i) = corr(odor_smooth(valid_idx), dist_change(valid_idx));
    else
        correlations_crim(i) = NaN;
    end
end
bar(1:n_agents_crim, correlations_crim);
xlabel('Agent #');
ylabel('Odor-Approach Correlation');
title('Crimaldi - Odor vs Distance Reduction');
ylim([-1 1]);
grid on;

% Correlation analysis - Smoke
subplot(2,2,4);
correlations_smoke = zeros(n_agents_smoke, 1);
for i = 1:n_agents_smoke
    odor_smooth = smooth(smoke_out.odor(:,i), correlation_window);
    dist_change = -diff(smooth(dist_smoke(:,i), correlation_window));
    valid_idx = odor_smooth(1:end-1) > 0.01;
    if sum(valid_idx) > 10
        correlations_smoke(i) = corr(odor_smooth(valid_idx), dist_change(valid_idx));
    else
        correlations_smoke(i) = NaN;
    end
end
bar(1:n_agents_smoke, correlations_smoke);
xlabel('Agent #');
ylabel('Odor-Approach Correlation');
title('Smoke - Odor vs Distance Reduction');
ylim([-1 1]);
grid on;

sgtitle('Statistical Analysis of Odor Response', 'FontSize', 16);
fprintf('Saving statistics figure...\n');
saveas(gcf, 'results/odor_distance_statistics.png');

%% Summary output
fprintf('\nDiagnostic plots created:\n');
fprintf('  - odor_distance_crimaldi.png: Individual agents (Crimaldi)\n');
fprintf('  - odor_distance_smoke.png: Individual agents (Smoke)\n');
fprintf('  - odor_distance_overlay.png: All agents overlaid\n');
fprintf('  - odor_distance_statistics.png: Statistical analysis\n');

fprintf('\nKey observations:\n');
fprintf('  Crimaldi: %.1f%% agents show positive odor-approach correlation\n', ...
        sum(correlations_crim > 0.1) / n_agents_crim * 100);
fprintf('  Smoke: %.1f%% agents show positive odor-approach correlation\n', ...
        sum(correlations_smoke > 0.1) / n_agents_smoke * 100);

end