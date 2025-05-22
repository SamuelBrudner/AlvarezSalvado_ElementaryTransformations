#!/bin/bash

#SBATCH --begin=now
#SBATCH --job-name=matlab_nav_sim
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=day
# ============================================
# SLURM Configuration - DO NOT MODIFY
# The array size is calculated automatically below
# ============================================
# Array will be set after calculating required jobs
#SBATCH --open-mode=append
#SBATCH --output=slurm_out/%A_%a.out
#SBATCH --error=slurm_err/%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --mail-user=samuel.brudner@yale.edu
#SBATCH --mail-type=ALL

# Create output directories if they don't exist
mkdir -p slurm_out slurm_err data/raw data/processed

# Disable GUI and plotting for batch jobs
export DISPLAY=
unset X11

# Load MATLAB module
module load matlab/R2021a

# Calculate condition parameters based on job ID
# ============================================
# Configuration - Adjust these values as needed
# ============================================

# Number of conditions (e.g., with/without bilateral sensing)
NUM_CONDITIONS=2

# Number of agents per condition (total agents = NUM_CONDITIONS * AGENTS_PER_CONDITION)
AGENTS_PER_CONDITION=50  # e.g., 50 agents per condition = 100 total agents

# Agents to run per job (for parallelization)
# Each agent will get a unique random seed
AGENTS_PER_JOB=1  # Set higher to run multiple agents per job

# Calculate derived values
JOBS_PER_CONDITION=$(( (AGENTS_PER_CONDITION + AGENTS_PER_JOB - 1) / AGENTS_PER_JOB ))
TOTAL_JOBS=$((NUM_CONDITIONS * JOBS_PER_CONDITION))

# ============================================
# SLURM Configuration - Array size is set here
# ============================================
#SBATCH --array=0-$((TOTAL_JOBS - 1))  # Auto-calculated number of jobs

# ============================================
# Job Parameter Calculation
# ============================================
# Calculate current condition and job index within condition
CONDITION=$((SLURM_ARRAY_TASK_ID % NUM_CONDITIONS))
JOB_INDEX_IN_CONDITION=$((SLURM_ARRAY_TASK_ID / NUM_CONDITIONS))

# Calculate which agents this job should handle
START_AGENT=$((JOB_INDEX_IN_CONDITION * AGENTS_PER_JOB + 1))
END_AGENT=$(( (JOB_INDEX_IN_CONDITION + 1) * AGENTS_PER_JOB ))
END_AGENT=$((END_AGENT < AGENTS_PER_CONDITION ? END_AGENT : AGENTS_PER_CONDITION))

# Set parameters based on condition
if [ $CONDITION -eq 0 ]; then
    # Condition 0: With bilateral sensing
    BILATERAL="true"
    CONDITION_NAME="bilateral"
else
    # Condition 1: Without bilateral sensing
    BILATERAL="false"
    CONDITION_NAME="unilateral"
fi

# Set random seed based on replicate for reproducibility
RANDOM_SEED=$((REPLICATE + 1))  # +1 to avoid seed 0

# ============================================
# Run Simulation
# ============================================

echo "=== Simulation Parameters ==="
echo "Job ID: $SLURM_ARRAY_TASK_ID"
echo "Condition: $CONDITION_NAME ($CONDITION)"
echo "Agents: $START_AGENT to $END_AGENT"
echo "Random seeds: ${RANDOM_SEEDS[*]}"
echo "Total agents per condition: $AGENTS_PER_CONDITION"
echo "Agents per job: $AGENTS_PER_JOB"
echo "============================"

# Create a temporary MATLAB script to run all agents for this job
MATLAB_SCRIPT=$(mktemp /tmp/batch_job_XXXX.m)

for ((i=0; i<${#RANDOM_SEEDS[@]}; i++)); do
    AGENT_INDEX=$((START_AGENT + i))
    SEED=${RANDOM_SEEDS[$i]}
    AGENT_DIR="data/raw/${CONDITION_NAME}_${AGENT_INDEX}"

    cat >> "$MATLAB_SCRIPT" <<EOF
config = struct();
config.bilateralSensing = $BILATERAL;
config.randomSeed = $SEED;
config.outputDir = '$AGENT_DIR';
config.condition = $CONDITION;
config.agentIndex = $AGENT_INDEX;
fprintf('Running agent %d with seed %d\n', $AGENT_INDEX, $SEED);
try
    run_navigation_cfg(config);
catch e
    disp(getReport(e));
    exit(1);
end
EOF
done

echo "exit(0);" >> "$MATLAB_SCRIPT"

# Run MATLAB with the generated script
matlab -nodisplay -nosplash -r "run('$MATLAB_SCRIPT');"
rm "$MATLAB_SCRIPT"

# Optional: Post-processing steps could be added here
# For example, to process the raw data after simulation completes

# Exit with success code
exit 0
