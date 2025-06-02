#!/bin/bash
#SBATCH --job-name=${EXP_NAME}_sim
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM_PER_TASK}
#SBATCH --cpus-per-task=1
#SBATCH --array=0-${ARRAY_UPPER}%${MAX_CONCURRENT}
#SBATCH --output=${OUTPUT_DIR}/${EXP_NAME}_logs/slurm-%A_%a.out
#SBATCH --error=${OUTPUT_DIR}/${EXP_NAME}_logs/slurm-%A_%a.err

# Change to project directory
cd ${PROJECT_DIR}

# Run MATLAB with proper path setup
matlab -nodisplay -r "addpath(genpath('Code')); navigation_model_vec(${TRIAL_LENGTH}, '${ENVIRONMENT}', 0, ${AGENTS_PER_JOB}); exit"