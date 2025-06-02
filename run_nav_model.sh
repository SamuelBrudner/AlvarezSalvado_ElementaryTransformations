#!/bin/bash
# Simple script to run navigation model

if [ "$1" == "test" ]; then
    echo "Submitting single test job..."
    sbatch --array=0-0 nav_job_final.slurm
elif [ "$1" == "full" ]; then
    echo "Submitting full array job (400 tasks)..."
    sbatch nav_job_final.slurm
else
    echo "Usage: ./run_nav_model.sh [test|full]"
    echo "  test - Run single test job"
    echo "  full - Run full 400-task array"
fi
