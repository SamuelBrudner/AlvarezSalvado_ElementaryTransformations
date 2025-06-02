% fix_and_rerun_batch.m
% Fix triallength issue and rerun all failed simulations

addpath('Code');

base_dir = 'data/raw/plume_comparison_20250530_095842';
conditions = dir(fullfile(base_dir, '*_*'));

total_failed = 0;
total_success = 0;
total_fixed = 0;
failed_dirs = {};

fprintf('=== FIXING AND RERUNNING FAILED SIMULATIONS ===\n\n');

% First, count failures
for c = 1:length(conditions)
    if ~conditions(c).isdir || strcmp(conditions(c).name, '.') || strcmp(conditions(c).name, '..')
        continue;
    end
    
    condition_path = fullfile(base_dir, conditions(c).name);
    agent_dirs = dir(fullfile(condition_path, '*_*'));
    
    fprintf('Checking %s...\n', conditions(c).name);
    
    for a = 1:length(agent_dirs)
        if ~agent_dirs(a).isdir || strcmp(agent_dirs(a).name, '.') || strcmp(agent_dirs(a).name, '..')
            continue;
        end
        
        agent_path = fullfile(condition_path, agent_dirs(a).name);
        result_file = fullfile(agent_path, 'result.mat');
        
        if exist(result_file, 'file')
            total_success = total_success + 1;
        else
            total_failed = total_failed + 1;
            failed_dirs{end+1} = agent_path;
        end
    end
end

fprintf('\nInitial Status:\n');
fprintf('  Successful: %d\n', total_success);
fprintf('  Failed: %d\n', total_failed);

if total_failed == 0
    fprintf('\nNo failed simulations to fix!\n');
    return;
end

% Ask for confirmation
fprintf('\nPress Enter to fix and rerun %d failed simulations (or Ctrl+C to cancel): ', total_failed);
pause;

% Process failed simulations
for i = 1:length(failed_dirs)
    agent_path = failed_dirs{i};
    [~, condition_name] = fileparts(fileparts(agent_path));
    [~, agent_name] = fileparts(agent_path);
    
    fprintf('\n[%d/%d] Processing %s/%s...\n', i, length(failed_dirs), condition_name, agent_name);
    
    % Load config
    config_file = fullfile(agent_path, 'config_used.yaml');
    if ~exist(config_file, 'file')
        fprintf('  ERROR: No config file found\n');
        continue;
    end
    
    try
        % Read JSON config
        fid = fopen(config_file, 'r');
        json_str = fread(fid, '*char')';
        fclose(fid);
        cfg = jsondecode(json_str);
        
        % Fix triallength if it's a string
        if ischar(cfg.triallength) || isstring(cfg.triallength)
            cfg.triallength = str2double(regexp(cfg.triallength, '^\d+', 'match', 'once'));
            fprintf('  Fixed triallength: %d\n', cfg.triallength);
        end
        
        % Add missing fields
        if ~isfield(cfg, 'ws'), cfg.ws = 1; end
        
        % Run simulation
        fprintf('  Running simulation...\n');
        result = run_navigation_cfg(cfg);
        
        % Save result
        result_file = fullfile(cfg.outputDir, 'result.mat');
        save(result_file, 'result');
        fprintf('  SUCCESS! Saved result.mat\n');
        
        % Export to CSV/JSON
        try
            export_results(result_file, cfg.outputDir, 'Format', 'both');
            fprintf('  Exported to CSV/JSON\n');
        catch
            fprintf('  WARNING: Export failed, but MAT file saved\n');
        end
        
        total_fixed = total_fixed + 1;
        
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        % Save error log
        error_file = fullfile(agent_path, 'rerun_error.log');
        fid = fopen(error_file, 'w');
        fprintf(fid, '%s\n', getReport(ME));
        fclose(fid);
    end
    
    % Progress update every 10 simulations
    if mod(i, 10) == 0
        fprintf('\nProgress: %d/%d completed, %d fixed successfully\n', i, length(failed_dirs), total_fixed);
    end
end

fprintf('\n=== FINAL SUMMARY ===\n');
fprintf('Total failed: %d\n', total_failed);
fprintf('Successfully fixed: %d\n', total_fixed);
fprintf('Still failed: %d\n', total_failed - total_fixed);

% Save summary
summary_file = fullfile(base_dir, 'rerun_summary.txt');
fid = fopen(summary_file, 'w');
fprintf(fid, 'Rerun Summary - %s\n', datestr(now));
fprintf(fid, 'Total failed: %d\n', total_failed);
fprintf(fid, 'Successfully fixed: %d\n', total_fixed);
fprintf(fid, 'Still failed: %d\n', total_failed - total_fixed);
fclose(fid);

fprintf('\nSummary saved to: %s\n', summary_file);