#!/bin/bash
#SBATCH --job-name=nav_test
#SBATCH --partition=day
#SBATCH --time=00:30:00
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
#SBATCH --output=test_output_%j.out
#SBATCH --error=test_output_%j.err

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run MATLAB with test parameters
matlab -nodisplay -r "
    addpath(genpath('$SCRIPT_DIR/Code'));
    
    % Test parameters - small scale for quick validation
    trial_length = 3600;
    environment = 'Crimaldi';
    num_agents = 10;
    
    fprintf('Test run started at %s\n', datestr(now));
    fprintf('Memory allocated: 82GB\n');
    fprintf('Environment: %s\n', environment);
    fprintf('Number of agents: %d\n', num_agents);
    fprintf('Trial length: %d samples\n\n', trial_length);
    
    try
        tic;
        out = navigation_model_vec(trial_length, environment, 0, num_agents);
        elapsed = toc;
        
        fprintf('\n✓ Simulation completed successfully in %.2f seconds\n', elapsed);
        fprintf('  Trajectories generated: %d\n', size(out.x, 2));
        
        if isfield(out, 'successrate')
            fprintf('  Success rate: %.2f%%\n', out.successrate * 100);
        end
        
        if isfield(out, 'latency')
            valid_latencies = out.latency(~isnan(out.latency));
            if ~isempty(valid_latencies)
                fprintf('  Mean latency: %.2f s\n', mean(valid_latencies));
            end
        end
        
        % Save results
        save(sprintf('test_results_%s.mat', getenv('SLURM_JOB_ID')), 'out', '-v7.3');
        fprintf('\nResults saved successfully\n');
        
    catch ME
        fprintf('\n✗ ERROR: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('  In %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
        exit(1);
    end
    
    exit(0);
"
