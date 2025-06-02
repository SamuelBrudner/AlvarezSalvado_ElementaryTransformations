#!/bin/bash
#SBATCH --job-name=test_batch_v3
#SBATCH --output=logs/test_batch_v3_%A_%a.out
#SBATCH --error=logs/test_batch_v3_%A_%a.err
#SBATCH --array=0-9%5  # 10 jobs: 2 plumes x 5 jobs per plume, max 5 concurrent
#SBATCH --time=00:30:00
#SBATCH --mem=80G
#SBATCH --partition=day

set -euo pipefail

# Create directories
mkdir -p logs data/raw/test_batch_v3 data/processed/test_batch_v3 configs

# Check and create config files if they don't exist
if [ ! -f "configs/batch_crimaldi.yaml" ]; then
    echo "Creating configs/batch_crimaldi.yaml..."
    cat > configs/batch_crimaldi.yaml << 'EOF'
# Configuration for Crimaldi plume batch runs
environment: Crimaldi
triallength: 3600  # Full length for 15Hz data
plotting: 0
ntrials: 1
EOF
fi

if [ ! -f "configs/batch_smoke_hdf5.yaml" ]; then
    echo "Creating configs/batch_smoke_hdf5.yaml..."
    cat > configs/batch_smoke_hdf5.yaml << 'EOF'
# Configuration for smoke HDF5 plume batch runs
environment: video
plume_metadata: data/smoke_hdf5_meta.yaml  # This should point to your HDF5 metadata
plotting: 0
ntrials: 1
triallength: 3600  # Adjust based on your video length
EOF
fi

# Check if the smoke HDF5 metadata exists, if not create a template
if [ ! -f "data/smoke_hdf5_meta.yaml" ]; then
    echo "WARNING: data/smoke_hdf5_meta.yaml not found!"
    echo "Creating template - you'll need to update the paths..."
    cat > data/smoke_hdf5_meta.yaml << 'EOF'
# Metadata for smoke HDF5 plume
output_directory: data
output_filename: smoke_1a_orig_backgroundsubtracted.h5
output_h5: smoke_1a_orig_backgroundsubtracted.h5
vid_mm_per_px: 0.1530  # 1/6.536 - adjust if different
fps: 60
scaled_to_crim: true
EOF
fi

# Load MATLAB module
module load MATLAB/2023b

# Test run parameters (smaller than full run)
AGENTS_PER_CONDITION=10  # 10 agents per plume (vs 1000 in full)
AGENTS_PER_JOB=2         # 2 agents per job (vs 50 in full)
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

echo "=== Test Batch Job V3 ==="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Plume: $PLUME_NAME"
echo "Config: $CONFIG_FILE"
echo "Agents: $START_AGENT to $END_AGENT"
echo "Memory: 80GB"
echo "===================="

# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file $CONFIG_FILE not found!"
    exit 1
fi

# Create temporary MATLAB script
MATLAB_SCRIPT=$(mktemp test_batch_v3_XXXX.m)
cat > "$MATLAB_SCRIPT" << 'MATLAB'
addpath('Code');

% Load configuration
cfg = load_config('$CONFIG_FILE');

% Set for single trials per agent
cfg.ntrials = 1;
cfg.plotting = 0;

% Set output directory
output_base = 'data/raw/test_batch_v3/${PLUME_NAME}';
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
        
        % Save minimal result to conserve memory
        save(fullfile(agent_dir, 'result.mat'), 'result', '-v7');
        
        % Export to CSV/JSON
        export_dir = fullfile('data/processed/test_batch_v3', '${PLUME_NAME}', sprintf('agent_%04d', agent));
        if ~exist(export_dir, 'dir')
            mkdir(export_dir);
        end
        
        % Check if export_results function exists
        if exist('export_results', 'file')
            export_results(fullfile(agent_dir, 'result.mat'), export_dir);
        else
            fprintf('  Warning: export_results function not found, skipping CSV/JSON export\n');
        end
        
        successful = successful + 1;
        fprintf('  Agent %d completed in %.2f seconds\n', agent, elapsed);
        
        % Clear result to free memory
        clear result;
        
    catch ME
        failed = failed + 1;
        fprintf('  ERROR with agent %d: %s\n', agent, ME.message);
        
        % Save error info
        error_info = struct('agent', agent, 'error', ME.message, 'stack', ME.stack);
        save(fullfile(output_base, sprintf('error_agent_%04d.mat', agent)), 'error_info', '-v7');
    end
end

fprintf('\nBatch completed: %d successful, %d failed\n', successful, failed);
exit;
MATLAB

# Variable substitution in MATLAB script
sed -i "s|\$CONFIG_FILE|$CONFIG_FILE|g" "$MATLAB_SCRIPT"
sed -i "s|\${PLUME_NAME}|$PLUME_NAME|g" "$MATLAB_SCRIPT"
sed -i "s|\$START_AGENT|$START_AGENT|g" "$MATLAB_SCRIPT"
sed -i "s|\$END_AGENT|$END_AGENT|g" "$MATLAB_SCRIPT"
sed -i "s|\$PLUME_IDX|$PLUME_IDX|g" "$MATLAB_SCRIPT"

# Run MATLAB
matlab -nodisplay -nosplash -r "run('$MATLAB_SCRIPT')" || {
    echo "MATLAB execution failed"
    exit 1
}

# Clean up
rm -f "$MATLAB_SCRIPT"

echo "Test batch job V3 completed successfully"