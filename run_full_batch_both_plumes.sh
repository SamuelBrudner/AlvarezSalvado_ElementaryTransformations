#!/bin/bash
#SBATCH --job-name=full_both_plumes
#SBATCH --output=logs/full_batch_%A_%a.out
#SBATCH --error=logs/full_batch_%A_%a.err
#SBATCH --array=0-39%20  # 40 jobs: 2 plumes x 20 jobs per plume, max 20 concurrent
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --partition=day

set -euo pipefail

# Create directories
mkdir -p logs data/raw/full_batch data/processed/full_batch

# Load MATLAB module
module load MATLAB/2023b

# Full run parameters
AGENTS_PER_CONDITION=1000
AGENTS_PER_JOB=50
NUM_PLUMES=2
JOBS_PER_PLUME=$((AGENTS_PER_CONDITION / AGENTS_PER_JOB))

# Calculate which plume and agents for this array task
PLUME_IDX=$((SLURM_ARRAY_TASK_ID / JOBS_PER_PLUME))
JOB_IN_PLUME=$((SLURM_ARRAY_TASK_ID % JOBS_PER_PLUME))
START_AGENT=$((JOB_IN_PLUME * AGENTS_PER_JOB + 1))
END_AGENT=$(((JOB_IN_PLUME + 1) * AGENTS_PER_JOB))

# Select plume config
if [ $PLUME_IDX -eq 0 ]; then
    PLUME_NAME="crimaldi"
    CONFIG_FILE="configs/batch_crimaldi.yaml"
else
    PLUME_NAME="smoke_hdf5"
    CONFIG_FILE="configs/batch_smoke_hdf5.yaml"
fi

echo "=== Full Batch Job ===" 
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Plume: $PLUME_NAME"
echo "Config: $CONFIG_FILE"
echo "Agents: $START_AGENT to $END_AGENT"
echo "===================="

# Create temporary MATLAB script
MATLAB_SCRIPT=$(mktemp full_batch_XXXX.m)
cat > "$MATLAB_SCRIPT" << MATLAB
addpath('Code');

% Load configuration
cfg = load_config('$CONFIG_FILE');

% Set for single trials per agent
cfg.ntrials = 1;
cfg.plotting = 0;

% Set output directory
output_base = 'data/raw/full_batch/${PLUME_NAME}';
if ~exist(output_base, 'dir')
    mkdir(output_base);
end

% Run simulations for this batch
successful = 0;
failed = 0;

for agent = $START_AGENT:$END_AGENT
    fprintf('[%s] Running agent %d/%d...\n', datestr(now), agent, $END_AGENT);
    
    % Set random seed based on agent number and plume
    cfg.randomSeed = agent + 1000 * $PLUME_IDX;
    
    % Set output for this agent
    agent_dir = fullfile(output_base, sprintf('agent_%04d', agent));
    cfg.outputDir = agent_dir;
    
    try
        % Run simulation
        tic;
        result = run_navigation_cfg(cfg);
        elapsed = toc;
        
        % Export to CSV/JSON
        export_results(fullfile(agent_dir, 'result.mat'), ...
                      fullfile('data/processed/full_batch/${PLUME_NAME}/agent_${sprintf('%04d', agent)}'));
        
        successful = successful + 1;
        fprintf('  Agent %d completed in %.2f seconds\n', agent, elapsed);
        
    catch ME
        failed = failed + 1;
        fprintf('  ERROR with agent %d: %s\n', agent, ME.message);
        % Save error info
        error_info = struct('agent', agent, 'error', ME.message, 'stack', ME.stack);
        save(fullfile(output_base, sprintf('error_agent_%04d.mat', agent)), 'error_info');
    end
end

fprintf('\nBatch completed: %d successful, %d failed\n', successful, failed);
exit;
MATLAB

# Run MATLAB
matlab -nodisplay -nosplash -r "run('$MATLAB_SCRIPT')" || {
    echo "MATLAB execution failed"
    exit 1
}

# Clean up
rm -f "$MATLAB_SCRIPT"

echo "Full batch job completed"
