% verify_simulation_results.m - HPC-compatible verification of simulation results
%
% Usage: Run this script in MATLAB on the HPC after pipeline completion
%        Checks for result files and basic success criteria
%        No visualization, only text output for HPC environment
%
% Author: Samuel Brudner
% Date: 2025-07-10

% Helper function for output flushing (compatible with older MATLAB)
flush_output = @() evalc('diary off; diary on');

%% Setup

fprintf('=== Verifying Simulation Results ===\n\n');
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
log_file = sprintf('results/verification_%s.log', timestamp);

diary(log_file); % Start logging to file

% Check for basic requirements
if ~exist('results', 'dir')
    error('Results directory not found. Pipeline may not have run successfully.');
end

% Define expected result files
crimaldi_file = 'results/nav_results_0000.mat';
smoke_file = 'results/smoke_nav_results_1000.mat';
plume_viz = 'results/complete_config_setup_with_plumes.pdf';

%% Check Crimaldi results
fprintf('1. Checking Crimaldi plume results...\n');

if ~exist(crimaldi_file, 'file')
    fprintf('   ❌ ERROR: Crimaldi results file not found: %s\n', crimaldi_file);
    success_crimaldi = false;
else
    fprintf('   ✓ Crimaldi results file found: %s\n', crimaldi_file);
    
    % Load and verify content
    try
        data = load(crimaldi_file);
        if ~isfield(data, 'out')
            fprintf('   ❌ ERROR: Invalid Crimaldi results file (missing "out" structure)\n');
            success_crimaldi = false;
        else
            out = data.out;
            
            % Check essential fields
            required_fields = {'x', 'y', 'environment'};
            missing_fields = required_fields(~isfield(out, required_fields));
            
            if ~isempty(missing_fields)
                fprintf('   ❌ WARNING: Missing fields in Crimaldi results: %s\n', ...
                    strjoin(missing_fields, ', '));
                success_crimaldi = false;
            else
                % Check data dimensions and content
                [frames, n_agents] = size(out.x);
                fprintf('   - Trajectory data: %d frames x %d agents\n', frames, n_agents);
                
                if frames < 10 || n_agents < 1
                    fprintf('   ❌ WARNING: Suspicious data dimensions in Crimaldi results\n');
                    success_crimaldi = false;
                else
                    success_crimaldi = true;
                    
                    % Show performance metrics if available
                    if isfield(out, 'successrate')
                        fprintf('   - Success rate: %.1f%%\n', out.successrate * 100);
                    end
                    
                    if isfield(out, 'latency')
                        valid_latencies = out.latency(~isnan(out.latency));
                        if ~isempty(valid_latencies)
                            fprintf('   - Mean latency: %.1f seconds\n', mean(valid_latencies));
                        end
                    end
                end
            end
        end
    catch err
        fprintf('   ❌ ERROR: Could not load Crimaldi results: %s\n', err.message);
        success_crimaldi = false;
    end
end

flush_output(); % Force flush output

%% Check Smoke results
fprintf('\n2. Checking Smoke plume results...\n');

if ~exist(smoke_file, 'file')
    fprintf('   ❌ ERROR: Smoke results file not found: %s\n', smoke_file);
    success_smoke = false;
else
    fprintf('   ✓ Smoke results file found: %s\n', smoke_file);
    
    % Load and verify content
    try
        data = load(smoke_file);
        if ~isfield(data, 'out')
            fprintf('   ❌ ERROR: Invalid Smoke results file (missing "out" structure)\n');
            success_smoke = false;
        else
            out = data.out;
            
            % Check essential fields
            required_fields = {'x', 'y', 'environment'};
            missing_fields = required_fields(~isfield(out, required_fields));
            
            if ~isempty(missing_fields)
                fprintf('   ❌ WARNING: Missing fields in Smoke results: %s\n', ...
                    strjoin(missing_fields, ', '));
                success_smoke = false;
            else
                % Check data dimensions and content
                [frames, n_agents] = size(out.x);
                fprintf('   - Trajectory data: %d frames x %d agents\n', frames, n_agents);
                
                if frames < 10 || n_agents < 1
                    fprintf('   ❌ WARNING: Suspicious data dimensions in Smoke results\n');
                    success_smoke = false;
                else
                    success_smoke = true;
                    
                    % Show performance metrics if available
                    if isfield(out, 'successrate')
                        fprintf('   - Success rate: %.1f%%\n', out.successrate * 100);
                    end
                    
                    if isfield(out, 'latency')
                        valid_latencies = out.latency(~isnan(out.latency));
                        if ~isempty(valid_latencies)
                            fprintf('   - Mean latency: %.1f seconds\n', mean(valid_latencies));
                        end
                    end
                end
            end
        end
    catch err
        fprintf('   ❌ ERROR: Could not load Smoke results: %s\n', err.message);
        success_smoke = false;
    end
end

flush_output(); % Force flush output

%% Check visualization
fprintf('\n3. Checking plume visualization...\n');

if ~exist(plume_viz, 'file')
    fprintf('   ❌ WARNING: Plume visualization file not found: %s\n', plume_viz);
    success_viz = false;
else
    fprintf('   ✓ Plume visualization file found: %s\n', plume_viz);
    
    % Check file size
    file_info = dir(plume_viz);
    if file_info.bytes < 1000
        fprintf('   ❌ WARNING: Plume visualization file suspiciously small: %d bytes\n', ...
            file_info.bytes);
        success_viz = false;
    else
        fprintf('   ✓ Plume visualization file size: %.1f KB\n', file_info.bytes / 1024);
        success_viz = true;
    end
end

flush_output(); % Force flush output

%% Check other output files
fprintf('\n4. Checking other output files...\n');
flush_output();

result_files = dir('results/*.mat');
fprintf('   - Found %d .mat files in results directory\n', length(result_files));

config_files = dir('results/*.json');
fprintf('   - Found %d .json files in results directory\n', length(config_files));

vis_files = dir('results/*.{pdf,png}');
fprintf('   - Found %d visualization files in results directory\n', length(vis_files));

flush_output(); % Force flush output

%% Final summary
fprintf('\n=== Verification Summary ===\n');

overall_success = success_crimaldi && success_smoke;

if overall_success
    fprintf('✅ PASS: All required simulation outputs verified successfully!\n');
else
    fprintf('❌ FAIL: Some simulation outputs are missing or invalid.\n');
end

fprintf('\nIndividual checks:\n');
if success_crimaldi
    fprintf('✅ PASS: Crimaldi simulation results\n');
else
    fprintf('❌ FAIL: Crimaldi simulation results\n');
end

if success_smoke
    fprintf('✅ PASS: Smoke simulation results\n');
else
    fprintf('❌ FAIL: Smoke simulation results\n');
end

if success_viz
    fprintf('✅ PASS: Plume visualization\n');
else
    fprintf('⚠️ WARNING: Plume visualization issues\n');
end

fprintf('\nVerification log saved to: %s\n', log_file);
fprintf('\n✓ Verification complete!\n');

% End logging
diary off;
