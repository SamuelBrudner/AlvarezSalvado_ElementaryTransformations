function run_batch_job(config_file, job_id, start_agent, end_agent, use_parallel)
% RUN_BATCH_JOB Run a batch of agent simulations
%   RUN_BATCH_JOB(CONFIG_FILE, JOB_ID, START_AGENT, END_AGENT, USE_PARALLEL) runs
%   simulations for agents from START_AGENT to END_AGENT as part of the
%   batch job identified by JOB_ID, using the configuration from CONFIG_FILE.
%   If USE_PARALLEL is true, simulations are executed with PARFOR. The
%   parameter defaults to false if omitted.
%
% Example:
%   run_batch_job('configs/batch_job.yaml', 0, 1, 50, true);

% Add Code directory to path if not already there
if isempty(which('run_agent_simulation'))
    addpath(fullfile(pwd, 'Code'));
end

% Ensure we have the required functions
required_functions = {'load_experiment_config', 'run_agent_simulation'};
for i = 1:length(required_functions)
    if ~exist(required_functions{i}, 'file')
        error('Required function not found: %s', required_functions{i});
    end
end

% Load experiment configuration
try
    cfg = load_experiment_config(config_file);
    fprintf('=== Experiment: %s ===\n', cfg.experiment.name);
    fprintf('Job ID: %d\n', job_id);
    fprintf('Agents: %d to %d\n', start_agent, end_agent);
    fprintf('Output directory: %s\n', cfg.experiment.output_base);
    fprintf('========================\n\n');
catch ME
    error('Failed to load experiment configuration: %s', ME.message);
end

% Run simulations for each agent in this batch
if nargin < 5
    use_parallel = false;
end

if use_parallel
    if isempty(gcp('nocreate'))
        parpool;
    end
    parfor agent_id = start_agent:end_agent
        try
            run_agent_simulation(job_id, agent_id, config_file);
        catch ME
            warning('Error running agent %d: %s', agent_id, getReport(ME));
        end
    end
else
    for agent_id = start_agent:end_agent
        try
            run_agent_simulation(job_id, agent_id, config_file);
        catch ME
            warning('Error running agent %d: %s', agent_id, getReport(ME));
            % Continue with next agent even if one fails
        end
    end
end

% Collect error logs
error_logs = {};
for agent_id = start_agent:end_agent
    plume_idx = mod((job_id - 1) ./ cfg.experiment.jobs_per_condition, cfg.experiment.num_plumes) + 1;
    sensing_idx = mod(floor((job_id - 1) ./ cfg.experiment.jobs_per_condition ./ cfg.experiment.num_plumes), cfg.experiment.num_sensing) + 1;
    plume_name = cfg.experiment.plume_types{plume_idx};
    sensing_name = cfg.experiment.sensing_modes{sensing_idx};
    seed = sum(double(plume_name)) + sum(double(sensing_name)) + agent_id;
    agent_dir = cfg.get_output_dir(plume_name, sensing_name, agent_id, seed);
    log_file = fullfile(agent_dir, 'error.log');
    if exist(log_file, 'file')
        error_logs{end+1} = log_file; %#ok<AGROW>
    end
end

fprintf('Batch job %d completed (agents %d-%d)\n', job_id, start_agent, end_agent);
if ~isempty(error_logs)
    fprintf('Errors occurred for %d agents:\n', numel(error_logs));
    fprintf('  %s\n', error_logs{:});
else
    fprintf('No errors detected for this batch.\n');
end
end
