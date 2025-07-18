function summarize_success_rates()
%SUMMARIZE_SUCCESS_RATES Aggregate success-rate across all result files.
%
%   Called with no arguments, this script scans the project `results/`
%   directory for any files matching the pattern `*_nav_results_*.mat`.
%   For each environment (e.g. Crimaldi, Smoke) it computes the mean
%   success-rate across every result file and prints a concise table like:
%
%       === Success-rate summary ===
%       Crimaldi : mean 72.5 %  (N=400 files)
%       Smoke    : mean 68.0 %  (N=400 files)
%
%   A copy of the summary is saved to
%       logs/pipeline/success_summary_<timestamp>.txt
%   so that it never clutters the repo root.
%
%   Usage (from project root):
%       matlab -nodisplay -nosplash -r "addpath(genpath('Code')); summarize_success_rates; exit"
%
%   This function adheres to the lab’s logging convention: all log-type
%   artefacts live under `logs/`, and simulation outputs remain in
%   `results/`.
%
%   Author: Automated refactor via Cascade, 2025-07-11

%% Resolve project structure
script_dir  = fileparts(mfilename('fullpath'));        % Code/
project_dir = fileparts(script_dir);                   % repo root
results_dir = fullfile(project_dir, 'results');
log_dir     = fullfile(project_dir, 'logs', 'pipeline');
if ~exist(log_dir, 'dir'); mkdir(log_dir); end

%% QC: ensure each results file has a unique RNG seed
verify_rng_seeds();

%% Gather result files
files = dir(fullfile(results_dir, '*_nav_results_*.mat'));
if isempty(files)
    fprintf('No result files found in %s\n', results_dir);
    return
end

%% Aggregate by environment
env_keys      = strings(0);
success_sums  = [];
agent_counts  = [];

for f = files'
    try
        data = load(fullfile(f.folder, f.name), 'out');
    catch ME
        warning('Could not load %s: %s', f.name, ME.message);
        continue
    end

    if ~isfield(data, 'out') || ~isfield(data.out, 'environment') || ~isfield(data.out, 'successrate')
        warning('File %s missing required fields; skipping', f.name);
        continue
    end

    env  = string(data.out.environment);
    if ~isfield(data.out,'success') || isempty(data.out.success)
        % Halt immediately – success vector is mandatory for reproducible QC
        error('Result file %s lacks mandatory field out.success. Investigate simulation output before summarizing.', f.name);
    end
    successes = sum(data.out.success(:) == 1, 'omitnan');
    agents    = numel(data.out.success);

    idx = find(env_keys == env, 1);
    if isempty(idx)
        env_keys(end+1)     = env;          %#ok<*AGROW>
        success_sums(end+1) = successes;
        agent_counts(end+1) = agents;
    else
        success_sums(idx)  = success_sums(idx) + successes;
        agent_counts(idx)  = agent_counts(idx) + agents;
    end
end

%% Print summary table (per navigator)
failure_counts = agent_counts - success_sums;
fprintf('\n=== Success summary (per navigator) ===\n');
for i = 1:numel(env_keys)
    mean_rate = (success_sums(i) / agent_counts(i)) * 100;
    fprintf('%-10s :  %d successes / %d failures  (%.1f %% success)\n', env_keys(i), success_sums(i), failure_counts(i), mean_rate);
end

%% Visualization: success proportions with standard errors
% Reorder so that the complex plume category (e.g. "smoke") is shown first on the x-axis
is_complex = false(size(env_keys));
for i = 1:numel(env_keys)
    env_l = lower(env_keys(i));
    is_complex(i) = contains(env_l, 'smoke') || contains(env_l, 'complex');
end
plot_order   = [find(is_complex) find(~is_complex)];  % complex first, then others

% Apply the new ordering to all relevant arrays
env_keys     = env_keys(plot_order);
success_sums = success_sums(plot_order);
agent_counts = agent_counts(plot_order);

% Compute proportions and standard errors after reordering
prop = success_sums ./ agent_counts;
se   = sqrt(prop .* (1 - prop) ./ agent_counts);

% Create bar plot with error bars, following lab style guide
fig = figure('Name', 'Success proportions', 'Color', 'w', ...
             'Units', 'inches', 'Position', [0 0 3.25 1.9], ...
             'Renderer', 'painters');
ax = axes(fig);

% Map each environment to its plume category colour
colors = zeros(numel(env_keys), 3);
for i = 1:numel(env_keys)
    env_l = lower(env_keys(i));
    if contains(env_l, 'crim') || contains(env_l, 'smooth')
        colors(i, :) = [0.580, 0.404, 0.741];   % smooth (purple)
    elseif contains(env_l, 'smoke') || contains(env_l, 'complex')
        colors(i, :) = [0.737, 0.741, 0.133];   % complex (yellow-green)
    else
        colors(i, :) = [0.7, 0.7, 0.7];         % fallback grey
    end
end

b = bar(ax, prop, 'FaceColor', 'flat', 'LineWidth', 1);
b.CData = colors;

hold(ax, 'on');
errorbar(ax, 1:numel(prop), prop, se, 'k.', 'LineWidth', 1, 'CapSize', 8);

% Generate display labels ("smooth" / "complex")
display_labels = strings(size(env_keys));
for i = 1:numel(env_keys)
    env_l = lower(env_keys(i));
    if contains(env_l, 'crim') || contains(env_l, 'smooth')
        display_labels(i) = "smooth";
    elseif contains(env_l, 'smoke') || contains(env_l, 'complex')
        display_labels(i) = "complex";
    else
        display_labels(i) = env_keys(i);
    end
end

% Apply axis labelling with custom display labels
set(ax, 'XTick', 1:numel(prop), 'XTickLabel', cellstr(display_labels), ...
        'FontName', 'Arial', 'FontSize', 10, ...
        'TickDir', 'in', 'LineWidth', 1.5, ...
        'TickLength', [0.03 0]);

ylabel(ax, 'Proportion of successes', 'FontName', 'Arial', 'FontSize', 10);

title(ax, 'Success rate ± SE', 'FontName', 'Arial', 'FontSize', 12);

ylim(ax, [0 1]);
box(ax, 'off');


% Save figure to logs for provenance (PDF)
fig_file = fullfile(log_dir, ['success_proportions_' datestr(now, 'yyyymmdd_HHMMSS') '.pdf']);
saveas(fig, fig_file);
fprintf('✓ Saved success proportion plot to %s\n', fig_file);

%% Additional QC: odor intensity diagnostics
try
    qc_odor_traces();
catch ME
    warning('qc_odor_traces failed: %s', ME.message);
end

%% Save to log file
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
log_file  = fullfile(log_dir, ['success_summary_' timestamp '.txt']);
fid = fopen(log_file, 'w');
if fid ~= -1
    fprintf(fid, 'Success-rate summary generated %s\n', timestamp);
    for i = 1:numel(env_keys)
        mean_rate = (success_sums(i) / agent_counts(i)) * 100;
        fprintf(fid, '%s %d successes / %d failures (%.1f %% success)\n', env_keys(i), success_sums(i), failure_counts(i), mean_rate);
    end
    fclose(fid);
    fprintf('\n✓ Summary saved to %s\n', log_file);
else
    warning('Could not write summary log to %s', log_file);
end
end
