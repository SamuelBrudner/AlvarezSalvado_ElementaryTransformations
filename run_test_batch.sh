#!/bin/bash
set -euo pipefail

# ───────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────
TEST_NAME="plume_test_$(date +%Y%m%d_%H%M%S)"
TEST_JOBS=3
TEST_AGENTS=2
TEST_AGENTS_PER_JOB=1
TEST_TIME="00:10:00"
TEST_MEM="4G"
TEST_PARTITION="day"
TEST_CONFIG="configs/default_config.yaml"

# ───────────────────────────────────────────────────────────
# Setup
# ───────────────────────────────────────────────────────────
echo "🚀 Starting test batch: $TEST_NAME"

# Create output directory
TEST_OUTPUT="test_output/$TEST_NAME"
mkdir -p "$TEST_OUTPUT"

# Check for required files
for file in "run_batch_job_4000.sh" "$TEST_CONFIG"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Error: Required file not found: $file"
        exit 1
    fi
done

# Check SLURM
if ! command -v sbatch &> /dev/null; then
    echo "❌ Error: SLURM's sbatch command not found"
    exit 1
fi

# ───────────────────────────────────────────────────────────
# Submit Test Jobs
# ───────────────────────────────────────────────────────────
echo -e "\n🔧 Test Configuration:"
echo "- Jobs to run: $TEST_JOBS"
echo "- Agents per condition: $TEST_AGENTS"
echo "- Agents per job: $TEST_AGENTS_PER_JOB"
echo "- Time limit: $TEST_TIME"
echo "- Memory: $TEST_MEM"
echo "- Output directory: $TEST_OUTPUT"
echo -e "\n📤 Submitting test jobs..."

# Submit the job array
JOB_ID=$(sbatch \
    --job-name="${TEST_NAME}" \
    --output="${TEST_OUTPUT}/slurm_%A_%a.out" \
    --error="${TEST_OUTPUT}/slurm_%A_%a.err" \
    --time="$TEST_TIME" \
    --mem="$TEST_MEM" \
    --cpus-per-task=1 \
    --array=0-$((TEST_JOBS-1)) \
    --export=ALL,\
           EXPERIMENT_NAME="$TEST_NAME",\
           AGENTS_PER_CONDITION=$TEST_AGENTS,\
           AGENTS_PER_JOB=$TEST_AGENTS_PER_JOB,\
           OUTPUT_BASE="$TEST_OUTPUT",\
           SLURM_PARTITION="$TEST_PARTITION",\
           SLURM_TIME="$TEST_TIME",\
           SLURM_MEM="$TEST_MEM",\
           SLURM_CPUS_PER_TASK=1,\
           SLURM_ARRAY_CONCURRENT=$TEST_JOBS,\
           MATLAB_OPTIONS="-nodisplay -nosplash" \
    run_batch_job_4000.sh | awk '{print $4}')

if [[ -z "$JOB_ID" ]]; then
    echo "❌ Failed to submit job"
    exit 1
fi

# ───────────────────────────────────────────────────────────
# Show Monitoring Commands
# ───────────────────────────────────────────────────────────
echo -e "\n✅ Test jobs submitted with ID: $JOB_ID"
echo -e "\n📋 Monitoring commands:"
echo "  # Tail output of all jobs:"
echo "  tail -f $TEST_OUTPUT/slurm_${JOB_ID}_*.out"
echo -e "\n  # Check for errors:"
echo "  grep -i error $TEST_OUTPUT/slurm_${JOB_ID}_*.err"
echo -e "\n  # Check job status:"
echo "  squeue -j $JOB_ID"
echo -e "\n  # Cancel all test jobs:"
echo "  scancel $JOB_ID"
echo -e "\n📁 Output directory:"
echo "  $TEST_OUTPUT/"

echo -e "\n🏁 Test batch started successfully! Use the commands above to monitor progress.\n"