#!/bin/bash
#SBATCH --job-name=test_both_plumes_v2
#SBATCH --output=logs/test_batch_v2_%A_%a.out
#SBATCH --error=logs/test_batch_v2_%A_%a.err
#SBATCH --array=0-3  # 4 jobs total: 2 plumes x 2 test runs each
#SBATCH --time=00:30:00
#SBATCH --mem=80G  # Increased from 16G to 80G
#SBATCH --partition=day

set -euo pipefail

# Create directories
mkdir -p logs data/raw/test_batch_v2 data/processed/test_batch_v2

# Load MATLAB module
module load MATLAB/2023b

# Test parameters
AGENTS_PER_CONDITION=10  # Small number for testing
AGENTS_PER_JOB=5
NUM_PLUMES=2

# Calculate which plume and agents for this array task
PLUME_IDX=$((SLURM_ARRAY_TASK_ID / 2))
JOB_IN_PLUME=$((SLURM_ARRAY_TASK_ID % 2))
START_AGENT=$((JOB_IN_PLUME * AGENTS_PER_JOB + 1))
END_AGENT=$(((JOB_IN_PLUME + 1) * AGENTS_PER_JOB))
END_AGENT=$((END_AGENT > AGENTS_PER_CONDITION ? AGENTS_PER_CONDITION : END_AGENT))

# Select plume config
if [ $PLUME_IDX -eq 0 ]; then
    PLUME_NAME="crimaldi"
    CONFIG_FILE="configs/batch_crimaldi.yaml"
else
    PLUME_NAME="smoke_hdf5"
    CONFIG_FILE="configs/batch_smoke_hdf5.yaml"
fi

echo "=== Test Batch Job V2 ===" 
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Plume: $PLUME_NAME"
echo "Config: $CONFIG_FILE"
echo "Agents: $START_AGENT to $END_AGENT"
echo "Memory: 80GB"
echo "===================="

# Create temporary MATLAB script
MATLAB_SCRIPT=$(mktemp test_batch_v2_XXXX.m)
cat > "$MATLAB_SCRIPT" << MATLAB
addpath('Code');

% Load configuration
cfg = load_config('$CONFIG_FILE');

% Override for test runs - run one agent at a time to reduce memory
cfg.ntrials = 1;
cfg.plotting = 0;

% Set output directory
output_base = 'data/raw/test_batch_v2/${PLUME_NAME}';
if ~exist(output_base, 'dir')
    mkdir(output_base);
end

% Run simulations for this batch
for agent = $START_AGENT:$END_AGENT
    fprintf('Running agent %d for %s plume...\n', agent, '$PLUME_NAME');
    
    % Set random seed based on agent number and plume
    cfg.randomSeed = agent + 1000 * $PLUME_IDX;
    
    % Set output for this agent
    agent_dir = fullfile(output_base, sprintf('agent_%04d', agent));
    cfg.outputDir = agent_dir;
    
    try
        % Run simulation
        result = run_navigation_cfg(cfg);
        
        % Save just the summary to conserve memory
        summary = struct();
        summary.plume = '$PLUME_NAME';
        summary.agent = agent;
        summary.success_rate = result.successrate;
        summary.completed = true;
        
        save(fullfile(agent_dir, 'summary.mat'), 'summary', '-v7');
        
        % Also save minimal result file
        result_minimal = struct();
        result_minimal.successrate = result.successrate;
        result_minimal.environment = result.environment;
        save(fullfile(agent_dir, 'result_minimal.mat'), 'result_minimal', '-v7');
        
        fprintf('  Agent %d completed successfully\n', agent);
        
        % Clear variables to free memory
        clear result;
        
    catch ME
        fprintf('  Error with agent %d: %s\n', agent, ME.message);
        % Save error info
        error_info = struct('agent', agent, 'error', ME.message);
        save(fullfile(output_base, sprintf('error_agent_%04d.mat', agent)), 'error_info', '-v7');
    end
end

fprintf('Batch job completed for %s plume, agents %d-%d\n', ...
    '$PLUME_NAME', $START_AGENT, $END_AGENT);
exit;
MATLAB

# Run MATLAB
matlab -nodisplay -nosplash -r "run('$MATLAB_SCRIPT')" || {
    echo "MATLAB execution failed"
    exit 1
}

# Clean up
rm -f "$MATLAB_SCRIPT"

echo "Test batch job V2 completed successfully"
