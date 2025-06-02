function export_results(input_file, output_dir, varargin)
%EXPORT_RESULTS Export simulation results to open formats (CSV/JSON)
%   EXPORT_RESULTS(INPUT_FILE, OUTPUT_DIR) exports the simulation results from
%   INPUT_FILE to CSV and JSON files in OUTPUT_DIR.
%   
%   EXPORT_RESULTS(..., 'Format', 'csv') exports only CSV files.
%   EXPORT_RESULTS(..., 'Format', 'json') exports only JSON files.
%   
%   Example:
%     % Export to both formats (default)
%     export_results('result.mat', 'output')
%     
%     % Export only CSV
%     export_results('result.mat', 'output', 'Format', 'csv')
% 
%   Output files:
%     - trajectories.csv: Time series data (t, trial, x, y, theta, odor, ON, OFF, turn)
%     - params.json: Model parameters
%     - summary.json: Simulation summary (success rate, latency, etc.)

% Parse input arguments
p = inputParser;
addRequired(p, 'input_file', @ischar);
addRequired(p, 'output_dir', @ischar);
addParameter(p, 'Format', 'both', @(x) any(validatestring(lower(x), {'csv', 'json', 'both'})));
parse(p, input_file, output_dir, varargin{:});

% Create output directory if it doesn't exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Load the result file
fprintf('Loading %s...\n', input_file);
data = load(input_file);
% Handle nested result structure
if isfield(data, 'result')
    result = data.result;
elseif isfield(data, 'out')
    result = data.out;
else
    result = data;
end
if isfield(result, 'out')
    result = result.out;
end
if ~isfield(result, 'x')
    error('export_results:NoTrajectories', ...
        'result.mat does not contain trajectories; was it saved with -struct?');
end

% Prepare output data
data = struct();

% Extract trajectories
[T, N] = size(result.x);
t = (1:T)' - 1;  % 0-based time indices

% Determine which optional columns to include
includeON = isfield(result, 'ON') && ~isempty(result.ON);
includeOFF = isfield(result, 'OFF') && ~isempty(result.OFF);

% Create a table for trajectory data
trajectories = table();
for i = 1:N
    % Each trial shares the same time vector, so repmat is unnecessary
    varNames = {'t', 'trial', 'x', 'y', 'theta', 'odor'};
    vars = {t, ...                     % Time
            repmat(i-1, T, 1), ...    % 0-based trial index
            result.x(:,i), ...        % X position
            result.y(:,i), ...        % Y position
            result.theta(:,i), ...    % Heading angle
            result.odor(:,i)};        % Odor concentration
    if includeON
        vars{end+1} = getField(result,'ON',T,i); %#ok<AGROW>
        varNames{end+1} = 'ON';
    end
    if includeOFF
        vars{end+1} = getField(result,'OFF',T,i); %#ok<AGROW>
        varNames{end+1} = 'OFF';
    end
    vars{end+1} = logical(result.turn(:,i));
    varNames{end+1} = 'turn';

    trial_data = table(vars{:}, 'VariableNames', varNames);
    trajectories = [trajectories; trial_data];
end

% Create parameter structure
params = struct();
if isfield(result, 'params')
    params = result.params;
end

% Create summary structure
summary = struct();
summary.successrate = 0;
summary.latency = [];
summary.n_trials = N;
summary.timesteps = T;

if isfield(result, 'successrate')
    summary.successrate = result.successrate;
end
if isfield(result, 'latency')
    summary.latency = result.latency(:)';
end

% Export data based on requested format
export_csv = strcmpi(p.Results.Format, 'both') || strcmpi(p.Results.Format, 'csv');
export_json = strcmpi(p.Results.Format, 'both') || strcmpi(p.Results.Format, 'json');

if export_csv
    % Write trajectories to CSV
    traj_file = fullfile(output_dir, 'trajectories.csv');
    writetable(trajectories, traj_file);
    fprintf('Wrote trajectories to %s\n', traj_file);
end

if export_json
    % Write parameters to JSON
    param_file = fullfile(output_dir, 'params.json');
    fid = fopen(param_file, 'w');
    fprintf(fid, '%s', jsonencode(params, 'PrettyPrint', true));
    fclose(fid);
    fprintf('Wrote parameters to %s\n', param_file);
    
    % Write summary to JSON
    summary_file = fullfile(output_dir, 'summary.json');
    fid = fopen(summary_file, 'w');
    fprintf(fid, '%s', jsonencode(summary, 'PrettyPrint', true));
    fclose(fid);
    fprintf('Wrote summary to %s\n', summary_file);
end

fprintf('Export completed successfully.\n');

end

function val = getField(s, name, T, idx)
%GETFIELD Helper to safely extract ON/OFF columns
    if isfield(s, name) && ~isempty(s.(name))
        data = s.(name);
        if size(data,1) >= T
            val = data(1:T, idx);
        else
            val = zeros(T,1);
        end
    else
        val = zeros(T,1);
    end
end
