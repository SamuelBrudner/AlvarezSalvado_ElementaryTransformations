function run_agent_simulation(job_id, agent_id, config_file)
% RUN_AGENT_SIMULATION Run a single agent's simulation
%   RUN_AGENT_SIMULATION(JOB_ID, AGENT_ID, CONFIG_FILE) runs the simulation
%   for a single agent with the specified JOB_ID and AGENT_ID, using the
%   configuration from CONFIG_FILE.

% Set up error handling
original_warning_state = warning('off', 'all');
cleanupObj = onCleanup(@() warning(original_warning_state));

% Add Code directory to path if not already there
if isempty(which('run_navigation_cfg'))
    addpath(fullfile(pwd, 'Code'));
end

% Ensure we have the required functions
required_functions = {'load_experiment_config', 'load_config', 'run_navigation_cfg'};
for i = 1:length(required_functions)
    if ~exist(required_functions{i}, 'file')
        error('Required function not found: %s', required_functions{i});
    end
end

% Load experiment configuration
try
    cfg = load_experiment_config(config_file);
catch ME
    error('Failed to load experiment configuration: %s', ME.message);
end

% Calculate agent parameters
plume_idx = mod((job_id - 1) ./ cfg.experiment.jobs_per_condition, cfg.experiment.num_plumes) + 1;
sensing_idx = mod(floor((job_id - 1) ./ cfg.experiment.jobs_per_condition ./ cfg.experiment.num_plumes), ...
                  cfg.experiment.num_sensing) + 1;

plume_name = cfg.experiment.plume_types{plume_idx};
sensing_name = cfg.experiment.sensing_modes{sensing_idx};

% Seed includes plume and sensing names for reproducibility across conditions
seed = sum(double(plume_name)) + sum(double(sensing_name)) + agent_id;

% Create output directory
output_dir = cfg.get_output_dir(plume_name, sensing_name, agent_id, seed);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('Starting simulation for job %d, agent %d (%s, %s)\n', ...
        job_id, agent_id, plume_name, sensing_name);

try
    % Load the base configuration
    sim_cfg = load_config(cfg.plume_config);
    
    % Override with simulation-specific parameters
    sim_cfg.bilateral = strcmp(sensing_name, 'bilateral');
    sim_cfg.randomSeed = seed;
    sim_cfg.ntrials = 1;
    sim_cfg.plotting = 0;
    sim_cfg.outputDir = output_dir;
    
    % Run the simulation
    result = run_navigation_cfg(sim_cfg);
    
    % Save the results under a top-level ''out'' field for downstream tools
    out = result;
    save(fullfile(output_dir, 'result.mat'), 'out', '-v7');
    fprintf('Successfully completed simulation for agent %d (seed %d)\n', agent_id, seed);
    
    % Clear large variables to save memory
    clear result sim_cfg;
    
catch ME
    error('Error in simulation (agent %d, seed %d): %s', ...
          agent_id, seed, getReport(ME));
end

% Add a small pause to prevent file system overload
pause(0.1);
end
