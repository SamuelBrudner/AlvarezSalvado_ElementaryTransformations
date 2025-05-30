function cfg = load_experiment_config(config_file, varargin)
% LOAD_EXPERIMENT_CONFIG Load and validate experiment configuration
%   CFG = LOAD_EXPERIMENT_CONFIG(CONFIG_FILE) loads the YAML configuration
%   from CONFIG_FILE and returns a struct with experiment parameters.
%   
%   CFG = LOAD_EXPERIMENT_CONFIG(..., 'param', value, ...) overrides any
%   configuration parameters with the specified name-value pairs.

% Default configuration file if none provided
if nargin < 1 || isempty(config_file)
    config_file = fullfile('configs', 'batch_job_config.yaml');
end

% Load base configuration
if ~exist('load_yaml', 'file')
    error('YAML parser not found. Please ensure the YAML Toolbox is installed.');
end

% Load configuration
if ~exist(config_file, 'file')
    error('Configuration file not found: %s', config_file);
end


% Load YAML configuration
try
    yaml_cfg = load_yaml(config_file);
catch ME
    error('Failed to parse YAML configuration: %s', ME.message);
end

% Convert YAML structure to MATLAB struct
cfg = struct(yaml_cfg);

% Ensure experiment struct exists before mapping fields
if ~isfield(cfg, 'experiment') || ~isstruct(cfg.experiment)
    cfg.experiment = struct();
end

% Map legacy plume and sensing fields onto experiment settings
if isfield(cfg, 'plumes')
    cfg.experiment.plume_types = cfg.plumes;
end
if isfield(cfg, 'sensing_modes')
    cfg.experiment.sensing_modes = cfg.sensing_modes;
end

% Apply any overrides from varargin
for i = 1:2:length(varargin)
    field = varargin{i};
    if isfield(cfg, field) || isprop(cfg, field)
        cfg.(field) = varargin{i+1};
    else
        warning('Unknown configuration field: %s', field);
    end
end

% Set default experiment values
defaults = {
    'name', 'default_experiment', ...
    'agents_per_condition', 1000, ...
    'agents_per_job', 100, ...
    'output_base', 'data/raw', ...
    'plume_types', {{'crimaldi', 'custom'}}, ...
    'sensing_modes', {{'bilateral', 'unilateral'}} ...
};

for i = 1:2:length(defaults)
    if ~isfield(cfg.experiment, defaults{i})
        cfg.experiment.(defaults{i}) = defaults{i+1};
    end
end

% Ensure paths are absolute
if ~startsWith(cfg.experiment.output_base, '/')
    cfg.experiment.output_base = fullfile(pwd, cfg.experiment.output_base);
end

% Calculate derived parameters
cfg.experiment.num_plumes = numel(cfg.experiment.plume_types);
cfg.experiment.num_sensing = numel(cfg.experiment.sensing_modes);
cfg.experiment.num_conditions = cfg.experiment.num_plumes * cfg.experiment.num_sensing;
cfg.experiment.jobs_per_condition = ceil(cfg.experiment.agents_per_condition / cfg.experiment.agents_per_job);
cfg.experiment.total_jobs = cfg.experiment.num_conditions * cfg.experiment.jobs_per_condition;

% Set default MATLAB options
if ~isfield(cfg, 'matlab') || ~isstruct(cfg.matlab)
    cfg.matlab = struct();
end

% Set default MATLAB values
matlab_defaults = {
    'version', 'R2021a', ...
    'module', 'matlab/R2021a', ...
    'options', '-nodisplay -nosplash', ...
    'path', {{'Code'}} ...
};

for i = 1:2:length(matlab_defaults)
    if ~isfield(cfg.matlab, matlab_defaults{i})
        cfg.matlab.(matlab_defaults{i}) = matlab_defaults{i+1};
    end
end

% Set default SLURM options
if ~isfield(cfg, 'slurm') || ~isstruct(cfg.slurm)
    cfg.slurm = struct();
end

% Set default SLURM values
slurm_defaults = {
    'partition', 'day', ...
    'time', '6:00:00', ...
    'mem', '16G', ...
    'cpus_per_task', 1, ...
    'array_concurrent', 100, ...
    'mail_user', [getenv('USER') '@yale.edu'], ...
    'mail_type', 'ALL' ...
};

for i = 1:2:length(slurm_defaults)
    if ~isfield(cfg.slurm, slurm_defaults{i})
        cfg.slurm.(slurm_defaults{i}) = slurm_defaults{i+1};
    end
end

% Validate configuration
validate_config(cfg);

% Add helper functions
cfg.get_output_dir = @(plume, sensing, agent_id, seed) ...
    fullfile(cfg.experiment.output_base, ...
             sprintf('%s_%s', plume, sensing), ...
             sprintf('%d_%d', agent_id, seed));

end

function validate_config(cfg)
% VALIDATE_CONFIG Validate the configuration structure
%   Validates that all required fields are present and have valid values.

% Check required fields
required_fields = {'experiment', 'matlab', 'slurm'};
for i = 1:length(required_fields)
    if ~isfield(cfg, required_fields{i})
        error('Missing required configuration section: %s', required_fields{i});
    end
end

% Validate experiment parameters
if ~isnumeric(cfg.experiment.agents_per_condition) || cfg.experiment.agents_per_condition <= 0
    error('agents_per_condition must be a positive number');
end

validateAgentsPerJob(cfg.experiment.agents_per_job);

% Ensure output directory exists
if ~exist(cfg.experiment.output_base, 'dir')
    try
        mkdir(cfg.experiment.output_base);
    catch ME
        error('Failed to create output directory %s: %s', ...
              cfg.experiment.output_base, ME.message);
    end
end
end
