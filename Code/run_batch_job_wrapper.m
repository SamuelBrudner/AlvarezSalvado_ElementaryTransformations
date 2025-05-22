function run_batch_job_wrapper(job_id, config_file)
%RUN_BATCH_JOB_WRAPPER Thin wrapper to call RUN_BATCH_JOB
%   RUN_BATCH_JOB_WRAPPER(JOB_ID, CONFIG_FILE) calculates the agent range for the
%   given JOB_ID using the configuration and then calls RUN_BATCH_JOB with the
%   expanded arguments.

arguments
    job_id (1,1) {mustBeInteger, mustBeNonnegative}
    config_file (1,:) char
end

% Ensure Code directory is on the path
if isempty(which('calculateJobParams'))
    addpath(fullfile(pwd, 'Code'));
end

cfg = load_experiment_config(config_file);
params = calculateJobParams(job_id + 1, ...
    cfg.experiment.num_conditions, ...
    cfg.experiment.agents_per_condition, ...
    cfg.experiment.agents_per_job);

run_batch_job(config_file, job_id + 1, params.startAgent, params.endAgent);
end
