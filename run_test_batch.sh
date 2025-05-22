#!/bin/bash
set -euo pipefail

echo "🚀 Setting up test run..."

# Create test output directory with timestamp
TEST_OUTPUT="test_output_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_OUTPUT"

# Test parameters
TEST_JOBS=3               # Run 3 jobs total
TEST_AGENTS=2             # Only 2 agents per condition
TEST_AGENTS_PER_JOB=1     # 1 agent per job
TEST_TIME="00:10:00"      # 10 minute time limit
TEST_MEM="4G"            # 4GB memory per job

echo "🔧 Test Configuration:"
echo "- Jobs to run: $TEST_JOBS"
echo "- Agents per condition: $TEST_AGENTS"
echo "- Agents per job: $TEST_AGENTS_PER_JOB"
echo "- Time limit: $TEST_TIME"
echo "- Memory: $TEST_MEM"
echo "- Output directory: $TEST_OUTPUT"

# Run the test
echo -e "\n🚀 Submitting test jobs..."
sbatch \
    --job-name=plume_test \
    --output="${TEST_OUTPUT}/slurm_%A_%a.out" \
    --error="${TEST_OUTPUT}/slurm_%A_%a.err" \
    --time=$TEST_TIME \
    --mem=$TEST_MEM \
    --cpus-per-task=1 \
    --array=0-$((TEST_JOBS-1)) \
    --export=ALL,\
           EXPERIMENT_NAME="plume_test_run",\
           AGENTS_PER_CONDITION=$TEST_AGENTS,\
           AGENTS_PER_JOB=$TEST_AGENTS_PER_JOB,\
           OUTPUT_BASE="$TEST_OUTPUT",\
           SLURM_PARTITION="day",\
           SLURM_TIME="00:15:00",\
           SLURM_MEM="4G",\
           SLURM_CPUS_PER_TASK=1,\
           SLURM_ARRAY_CONCURRENT=3,\
           MATLAB_OPTIONS="-nodisplay -nosplash" \
    run_batch_job_4000.sh

echo -e "\n✅ Test jobs submitted!"
echo -e "\n📋 To monitor progress:"
echo "  tail -f $TEST_OUTPUT/slurm_*.out"
echo -e "\n🔍 To check for errors:"
echo "  grep -i error $TEST_OUTPUT/slurm_*.err"
echo -e "\n📁 Output will be in: $TEST_OUTPUT/"
echo -e "\n💡 Quick status check:"
echo "  squeue -u $USER -n plume_test"
