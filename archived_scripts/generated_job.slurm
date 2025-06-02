#!/bin/bash
#SBATCH --job-name=crimaldi_sim
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-399%100
#SBATCH --output=/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/crimaldi_logs/slurm-%A_%a.out
#SBATCH --error=/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/crimaldi_logs/slurm-%A_%a.err

# Change to project directory
cd /home/snb6/Documents/AlvarezSalvado_ElementaryTransformations

# Run MATLAB with proper path setup
matlab -nodisplay -r "addpath(genpath('Code')); navigation_model_vec(3600, 'Crimaldi', 0, 10); exit"