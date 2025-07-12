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
        error('Result file %s lacks "success" vector – aborting summary. Fix simulation output.', f.name);
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
fprintf('\n=== Success-rate summary (per navigator) ===\n');
for i = 1:numel(env_keys)
    mean_rate = (success_sums(i) / agent_counts(i)) * 100;
    fprintf('%-10s :  mean %.1f %%  (N=%d navigators)\n', env_keys(i), mean_rate, agent_counts(i));
end

%% Save to log file
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
log_file  = fullfile(log_dir, ['success_summary_' timestamp '.txt']);
fid = fopen(log_file, 'w');
if fid ~= -1
    fprintf(fid, 'Success-rate summary generated %s\n', timestamp);
    for i = 1:numel(env_keys)
        mean_rate = (success_sums(i) / agent_counts(i)) * 100;
        fprintf(fid, '%s %.1f %% (N=%d navigators)\n', env_keys(i), mean_rate, agent_counts(i));
    end
    fclose(fid);
    fprintf('\n✓ Summary saved to %s\n', log_file);
else
    warning('Could not write summary log to %s', log_file);
end
end
