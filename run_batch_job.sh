#!/bin/bash
#SBATCH --partition=<partition>
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 00:30:00

# plotting should be disabled for batch jobs
export plotting=0

module load matlab  # or load octave
conda activate .env
matlab -batch "run_my_simulation"
