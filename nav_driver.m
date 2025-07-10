% nav_driver.m
% MATLAB entry point for parameterised SLURM job submission.
% Reads configuration from environment variables set by nav_job_base.slurm.
%
% Required env vars (validated in the SLURM wrapper):
%   PLUME_JSON     – Full path to plume JSON config.
%
% Optional env vars:
%   FRAME_RATE     – Override frame rate (Hz) when calculating n_frames.
%   RESULTS_PREFIX – Prefix for output .mat filename (default "nav_results").
%   PROJECT_DIR    – Repository root; defaults to pwd.
%
% The script updates configs/paths.json to point to PLUME_JSON, determines
% the number of frames for the simulation, runs navigation_model_vec and
% saves the output to results/<prefix>_<task>.mat.

try
    fprintf('==== nav_driver.m started at %s ===\n', datestr(now));

    % Read env vars ------------------------------------------------------
    project_dir    = getenv('PROJECT_DIR');
    if isempty(project_dir)
        project_dir = pwd;
    end
    plume_json     = getenv('PLUME_JSON');
    frame_rate_env = getenv('FRAME_RATE');
    results_prefix = getenv('RESULTS_PREFIX');
    if isempty(results_prefix); results_prefix = 'nav_results'; end
    task_id        = str2double(getenv('SLURM_ARRAY_TASK_ID'));
    if isnan(task_id); task_id = 0; end

    fprintf('Project dir      : %s\n', project_dir);
    fprintf('Plume config JSON: %s\n', plume_json);
    fprintf('Results prefix   : %s\n', results_prefix);
    fprintf('Task ID          : %d\n', task_id);

    % Ensure we are inside project directory ----------------------------
    cd(project_dir);
    addpath(genpath('Code'));

    % Update paths.json --------------------------------------------------
    paths_file = fullfile('configs', 'paths.json');
    
    % Validate existing JSON first
    try
        paths_data = jsondecode(fileread(paths_file));
        fprintf('Successfully loaded paths.json\n');
    catch jsonEx
        error('Error loading paths.json: %s\nCheck JSON format and fix it before continuing.', jsonEx.message);
    end
    
    % Create backup before modifying
    backup_file = [paths_file, '.bak'];
    copyfile(paths_file, backup_file);
    fprintf('Created backup at %s\n', backup_file);
    
    % Only update the plume_config field, preserving everything else
    paths_data.plume_config = plume_json;
    
    % Validate the modified structure before writing
    try
        % Test that the modified structure can be encoded properly
        json_str = jsonencode(paths_data);
        % Parse it back to verify integrity
        jsondecode(json_str);
        
        % Write with pretty formatting to maintain readability
        fid = fopen(paths_file, 'w');
        if fid == -1
            error('Could not open paths.json for writing');
        end
        
        % Use jsonencode with proper spacing and indentation
        json_str = jsonencode(paths_data, 'PrettyPrint', true);
        fprintf(fid, '%s', json_str);
        fclose(fid);
        fprintf('configs/paths.json updated with validation.\n');
    catch jsonEx
        % Restore from backup if something went wrong
        fprintf(2, 'Error updating paths.json: %s\n', jsonEx.message);
        fprintf(2, 'Restoring from backup...\n');
        copyfile(backup_file, paths_file);
        error('Failed to update paths.json. Restored from backup. Error: %s', jsonEx.message);
    end

    % Load plume config --------------------------------------------------
    [plume_file, plume_cfg] = get_plume_file();
    fprintf('Active plume file : %s\n', plume_file);
    fprintf('Config frame rate : %.1f Hz\n', plume_cfg.temporal.frame_rate);

    % Determine frame rate & n_frames -----------------------------------
    if isempty(frame_rate_env)
        frame_rate = plume_cfg.temporal.frame_rate;
    else
        frame_rate = str2double(frame_rate_env);
        if isnan(frame_rate) || frame_rate <= 0
            error('Invalid FRAME_RATE env var: %s', frame_rate_env);
        end
        fprintf('Overriding frame rate to %.1f Hz\n', frame_rate);
    end
    duration_seconds = plume_cfg.simulation.duration_seconds;
    n_frames = round(duration_seconds * frame_rate);
    fprintf('Simulation duration: %.1f s → %d frames at %.1f Hz\n', ...
            duration_seconds, n_frames, frame_rate);

    % Run simulation -----------------------------------------------------
    fprintf('\nStarting navigation_model_vec...\n');
    tStart = tic;
    out = navigation_model_vec(n_frames, 'Crimaldi', 0, 10);  % env string not used
    fprintf('Simulation finished in %.1f s\n', toc(tStart));

    % Save results -------------------------------------------------------
    results_dir = 'results';
    if ~exist(results_dir, 'dir'); mkdir(results_dir); end
    filename = fullfile(results_dir, sprintf('%s_%04d.mat', results_prefix, task_id));
    save(filename, 'out', '-v7.3');
    fprintf('Results saved to %s\n', filename);

    fprintf('==== nav_driver.m completed OK ===\n');
catch ME
    fprintf(2, 'ERROR in nav_driver: %s\n', ME.message);
    for k = 1:numel(ME.stack)
        fprintf(2, '  at %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
    end
    exit(1);
end

exit(0);
