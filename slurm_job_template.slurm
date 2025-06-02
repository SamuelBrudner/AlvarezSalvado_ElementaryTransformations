#!/bin/bash
#SBATCH --job-name=${EXP_NAME}_sim
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM_PER_TASK}
#SBATCH --cpus-per-task=1
#SBATCH --array=0-${ARRAY_UPPER}%${MAX_CONCURRENT}
#SBATCH --output=${OUTPUT_DIR}/slurm-%A_%a.out

matlab -nodisplay -r "navigation_model_vec(${TRIAL_LENGTH}, '${ENVIRONMENT}', 0, ${AGENTS_PER_JOB}); exit"
