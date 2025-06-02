#!/bin/bash
# Export test .mat files to CSV/JSON format

set -euo pipefail

echo "=== EXPORTING TEST DATA TO CSV/JSON ==="
echo

# Create base output directory
mkdir -p data/processed/test_exports

# Create a temporary MATLAB script
MATLAB_SCRIPT=$(mktemp /tmp/export_test_data_XXXX.m)

cat > "$MATLAB_SCRIPT" << 'EOF'
% Export test data script
addpath('Code');

% Find all test .mat files
mat_files = dir(fullfile('test_output', '**', '*.mat'));
fprintf('Found %d MAT files to export\n', length(mat_files));

success_count = 0;
failed_files = {};

for i = 1:length(mat_files)
    mat_file = fullfile(mat_files(i).folder, mat_files(i).name);
    
    % Create output directory structure
    rel_path = strrep(mat_files(i).folder, 'test_output/', '');
    output_dir = fullfile('data/processed/test_exports', rel_path);
    
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    fprintf('\n[%d/%d] Processing: %s\n', i, length(mat_files), mat_file);
    
    try
        % Try using export_results if it exists
        if exist('export_results', 'file')
            export_results(mat_file, output_dir, 'Format', 'both');
            fprintf('  ✓ Exported using export_results\n');
            success_count = success_count + 1;
        else
            error('export_results function not found');
        end
    catch ME
        fprintf('  ! export_results failed: %s\n', ME.message);
        failed_files{end+1} = mat_file;
        
        % Try manual export
        try
            fprintf('  Attempting manual export...\n');
            
            % Load the mat file
            data = load(mat_file);
            
            % Try to find the main result structure
            result = [];
            if isfield(data, 'out')
                result = data.out;
            elseif isfield(data, 'result')
                result = data.result;
            elseif isfield(data, 'R')
                result = data.R;
            elseif isfield(data, 'x') && isfield(data, 'y')
                result = data;
            else
                % Look for the main struct
                fields = fieldnames(data);
                for f = 1:length(fields)
                    if isstruct(data.(fields{f})) && isfield(data.(fields{f}), 'x')
                        result = data.(fields{f});
                        break;
                    end
                end
            end
            
            if ~isempty(result) && isfield(result, 'x') && isfield(result, 'y')
                % Export trajectories
                export_trajectories_manual(result, output_dir);
                fprintf('  ✓ Manual export successful\n');
                success_count = success_count + 1;
            else
                fprintf('  ✗ No trajectory data found\n');
            end
            
        catch ME2
            fprintf('  ✗ Manual export also failed: %s\n', ME2.message);
        end
    end
end

fprintf('\n=== EXPORT SUMMARY ===\n');
fprintf('Successfully exported: %d/%d files\n', success_count, length(mat_files));
if ~isempty(failed_files)
    fprintf('\nFailed files:\n');
    for i = 1:length(failed_files)
        fprintf('  - %s\n', failed_files{i});
    end
end

% Exit MATLAB
exit;

% Helper function for manual export
function export_trajectories_manual(result, output_dir)
    % Get dimensions
    if size(result.x, 2) > 1
        [T, N] = size(result.x);
    else
        T = length(result.x);
        N = 1;
    end
    
    % Create trajectories table
    trajectories = [];
    for trial = 1:N
        trial_data = table();
        trial_data.t = (0:T-1)';
        trial_data.trial = repmat(trial-1, T, 1);
        
        if size(result.x, 2) > 1
            trial_data.x = result.x(:, trial);
            trial_data.y = result.y(:, trial);
        else
            trial_data.x = result.x(:);
            trial_data.y = result.y(:);
        end
        
        % Add optional fields
        if isfield(result, 'theta')
            if size(result.theta, 2) > 1
                trial_data.theta = result.theta(:, trial);
            else
                trial_data.theta = result.theta(:);
            end
        end
        
        if isfield(result, 'odor')
            if size(result.odor, 2) > 1
                trial_data.odor = result.odor(:, trial);
            else
                trial_data.odor = result.odor(:);
            end
        end
        
        if isfield(result, 'turn')
            if size(result.turn, 2) > 1
                trial_data.turn = logical(result.turn(:, trial));
            else
                trial_data.turn = logical(result.turn(:));
            end
        end
        
        if isempty(trajectories)
            trajectories = trial_data;
        else
            trajectories = [trajectories; trial_data];
        end
    end
    
    % Save trajectories
    writetable(trajectories, fullfile(output_dir, 'trajectories.csv'));
    
    % Create and save summary
    summary = struct();
    summary.n_trials = N;
    summary.timesteps = T;
    
    if isfield(result, 'successrate')
        summary.successrate = result.successrate;
    else
        summary.successrate = 0;
    end
    
    if isfield(result, 'latency')
        summary.latency = result.latency(:)';
    else
        summary.latency = [];
    end
    
    % Save summary as JSON
    fid = fopen(fullfile(output_dir, 'summary.json'), 'w');
    fprintf(fid, '%s', jsonencode(summary, 'PrettyPrint', true));
    fclose(fid);
end
EOF

# Run MATLAB with the script
echo "Running MATLAB export script..."
matlab -nodisplay -nosplash -r "run('$MATLAB_SCRIPT')" 2>&1 | grep -v "Warning: No MATLAB command specified" || true

# Clean up
rm -f "$MATLAB_SCRIPT"

# Check results
echo
echo "=== CHECKING EXPORT RESULTS ==="
CSV_COUNT=$(find data/processed/test_exports -name "trajectories.csv" 2>/dev/null | wc -l || echo 0)
JSON_COUNT=$(find data/processed/test_exports -name "summary.json" 2>/dev/null | wc -l || echo 0)

echo "Exported files:"
echo "  - Trajectory CSV files: $CSV_COUNT"
echo "  - Summary JSON files: $JSON_COUNT"
echo

if [ "$CSV_COUNT" -gt 0 ]; then
    echo "Exported data locations:"
    find data/processed/test_exports -name "trajectories.csv" | sort
else
    echo "No CSV files were exported. Check the MATLAB output above for errors."
fi

echo
echo "Export process complete."