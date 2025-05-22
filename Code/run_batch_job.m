function run_batch_job(config_file, job_id, start_agent, end_agent, use_parallel)
% RUN_BATCH_JOB Run a batch of agent simulations
%   RUN_BATCH_JOB(CONFIG_FILE, JOB_ID, START_AGENT, END_AGENT, USE_PARALLEL) runs
%   simulations for agents from START_AGENT to END_AGENT as part of the
%   batch job identified by JOB_ID, using the configuration from CONFIG_FILE.
%   If USE_PARALLEL is true, simulations are executed with PARFOR. The
%   parameter defaults to false if omitted.

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

fprintf('Batch job %d completed (agents %d-%d)\n', job_id, start_agent, end_agent);
end
