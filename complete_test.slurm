#!/bin/bash
# Complete test workflow
#SBATCH --job-name=complete_test
#SBATCH --partition=day
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/complete_test/complete_test_%j.out
#SBATCH --error=logs/complete_test/complete_test_%j.err

set -euo pipefail

module load MATLAB/2023b

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs/complete_test results

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Starting complete test at $(date)" | tee "results/diagnostics_${TIMESTAMP}.txt"

echo "=== Generating configs ===" | tee -a "results/diagnostics_${TIMESTAMP}.txt"
matlab -batch "try, run('generate_clean_configs.m'); catch ME, disp(getReport(ME)); exit(1); end; exit(0);" 2>&1 | tee -a "results/diagnostics_${TIMESTAMP}.txt"

echo "=== Running complete test ===" | tee -a "results/diagnostics_${TIMESTAMP}.txt"
matlab -batch "try, run('test_both_plumes_complete.m'); catch ME, disp(getReport(ME)); exit(1); end; exit(0);" 2>&1 | tee -a "results/diagnostics_${TIMESTAMP}.txt"

echo "=== Diagnostics ===" | tee -a "results/diagnostics_${TIMESTAMP}.txt"
ls -lh results | tee -a "results/diagnostics_${TIMESTAMP}.txt"

if compgen -G "results/*.png" > /dev/null; then
    echo "Plot files:" | tee -a "results/diagnostics_${TIMESTAMP}.txt"
    ls -1 results/*.png | tee -a "results/diagnostics_${TIMESTAMP}.txt"
else
    echo "No plot files found" | tee -a "results/diagnostics_${TIMESTAMP}.txt"
fi

echo "Finished at $(date)" | tee -a "results/diagnostics_${TIMESTAMP}.txt"

