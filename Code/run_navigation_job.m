function run_navigation_job(cfg_path)
%RUN_NAVIGATION_JOB  Wrapper to run navigation_model_vec for a given plume.
%
%   Syntax
%   -----
%   run_navigation_job(cfg_path)
%
%   Input
%   -----
%   cfg_path : char | string
%       Absolute path to a plume JSON configuration file (e.g.
%       `configs/plumes/crimaldi_10cms_bounded.json`).  The script relies on
%       `get_plume_file.m` to resolve the corresponding HDF5 plume file and
%       to derive simulation parameters.
%
%   Description
%   -----------
%   1. Sets the environment variable `PLUME_CONFIG` to `cfg_path`, ensuring
%      downstream helpers (e.g. `get_plume_file`) pick up the correct
%      configuration.
%   2. Resolves the HDF5 plume file via `get_plume_file`, then exports its
%      path through `MATLAB_PLUME_FILE` for any legacy code that still
%      expects the variable.
%   3. Derives key simulation parameters (frame-rate, duration, number of
%      agents) directly from the JSON to avoid hard-coding.
%   4. Calls `navigation_model_vec` and persists its output in
%      `results/<env>_nav_results_<taskId>.mat`, where `<env>` is inferred
%      from the file name and `<taskId>` comes from the SLURM array ID (or
%      defaults to 0 when none is present).
%
%   This wrapper allows SLURM job scripts to shrink to a few lines:
%
%       export PLUME_CONFIG=/path/to/smoke.json
%       matlab -batch "run_navigation_job('$PLUME_CONFIG')"
%
%   thereby eliminating the need for large, error-prone heredocs duplicated
%   across multiple job files.

arguments
    cfg_path (1,1) string {mustBeFile}
end

%% Environment bootstrap
project_dir = fileparts(fileparts(mfilename('fullpath'))); % repo root
addpath(genpath(fullfile(project_dir, 'Code')));

% Expose configuration so that legacy helpers work unchanged
setenv('PLUME_CONFIG', cfg_path);

%% Resolve plume data & config struct
[plume_file, cfg] = get_plume_file();
setenv('MATLAB_PLUME_FILE', plume_file);

%% Determine environment label from cfg filename
cfg_lower = lower(cfg_path);
if contains(cfg_lower, "crimaldi")
    env = "Crimaldi";
elseif contains(cfg_lower, "smoke")
    env = "Smoke";
else
    warning('Unknown plume type – defaulting environment label to "Crimaldi"');
    env = "Crimaldi";
end

%% Extract simulation parameters
duration_seconds = cfg.simulation.duration_seconds;
frame_rate       = cfg.temporal.frame_rate;
n_agents         = cfg.simulation.agent_initialization.n_agents_per_job;

n_frames = round(duration_seconds * frame_rate);

fprintf('[MATLAB] %s: %d agents, %.1f s, %d frames @ %.1f Hz\n', ...
        env, n_agents, duration_seconds, n_frames, frame_rate);

%% Run simulation
out = navigation_model_vec(n_frames, env, 0, n_agents);

%% Persist results
array_id = getenv('SLURM_ARRAY_TASK_ID');
if isempty(array_id)
    task_id = 0;
else
    task_id = str2double(array_id);
    if isnan(task_id); task_id = 0; end
end

result_name = sprintf('results/%s_nav_results_%04d.mat', lower(env), task_id);
if ~isfolder('results'); mkdir('results'); end
save(result_name, 'out', '-v7.3');

fprintf('[MATLAB] Results saved ➜ %s\n', result_name);
end
