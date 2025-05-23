#!/bin/bash
# run_test_batch.sh — quick SLURM smoke-test for the plume simulator
# ---------------------------------------------------------------
set -euo pipefail

# ───────────────────────────────────────────────────────────
# 1.  User-tweakable parameters for the smoke test
# ───────────────────────────────────────────────────────────
TEST_NAME="plume_test_$(date +%Y%m%d_%H%M%S)"

TEST_JOBS=3               # number of SLURM array tasks
TEST_AGENTS=2             # agents per condition
TEST_AGENTS_PER_JOB=1     # agents simulated by each array task

TEST_TIME="00:10:00"      # wall-clock limit
TEST_MEM="4G"             # memory per task
TEST_PARTITION="day"      # partition/queue

TEST_CONFIG="configs/default_config.yaml"

# ───────────────────────────────────────────────────────────
# 2.  Output directories and sanity checks
# ───────────────────────────────────────────────────────────
TEST_OUTPUT="test_output/$TEST_NAME"
mkdir -p "$TEST_OUTPUT"

echo "🚀 Starting test batch: $TEST_NAME"

# Required support files
for f in run_batch_job_4000.sh "$TEST_CONFIG"; do
    [[ -f $f ]] || { echo "❌ Required file missing: $f" >&2; exit 1; }
done

command -v sbatch >/dev/null || { echo "❌ SLURM’s sbatch not found" >&2; exit 1; }

# ───────────────────────────────────────────────────────────
# 3.  Display configuration summary
# ───────────────────────────────────────────────────────────
cat <<EOF

🔧 Test Configuration
--------------------
Jobs to run          : $TEST_JOBS
Agents/condition     : $TEST_AGENTS
Agents/job           : $TEST_AGENTS_PER_JOB
Time limit           : $TEST_TIME
Memory               : $TEST_MEM
Output directory     : $TEST_OUTPUT
EOF

# ───────────────────────────────────────────────────────────
# 4.  Assemble environment variables for sbatch --export
#     (one long string = no stray back-slashes or commas)
# ───────────────────────────────────────────────────────────
EXPORT_VARS=$(cat <<EOV
ALL,\
EXPERIMENT_NAME=$TEST_NAME,\
AGENTS_PER_CONDITION=$TEST_AGENTS,\
AGENTS_PER_JOB=$TEST_AGENTS_PER_JOB,\
OUTPUT_BASE=$TEST_OUTPUT,\
SLURM_PARTITION=$TEST_PARTITION,\
SLURM_TIME=$TEST_TIME,\
SLURM_MEM=$TEST_MEM,\
SLURM_CPUS_PER_TASK=1,\
SLURM_ARRAY_CONCURRENT=$TEST_JOBS,\
MATLAB_OPTIONS="-nodisplay -nosplash"
EOV
)

# ───────────────────────────────────────────────────────────
# 5.  Submit the SLURM job array
# ───────────────────────────────────────────────────────────
JOB_ID=$(sbatch \
  --job-name="$TEST_NAME" \
  --output="$TEST_OUTPUT/slurm_%A_%a.out" \
  --error="$TEST_OUTPUT/slurm_%A_%a.err" \
  --time="$TEST_TIME" \
  --mem="$TEST_MEM" \
  --cpus-per-task=1 \
  --array=0-$((TEST_JOBS-1)) \
  --export="$EXPORT_VARS" \
  run_batch_job_4000.sh | awk '{print $4}')

if [[ -z $JOB_ID ]]; then
    echo "❌ Failed to submit job" >&2
    exit 1
fi

# ───────────────────────────────────────────────────────────
# 6.  Handy monitoring commands
# ───────────────────────────────────────────────────────────
cat <<EOF

✅ Test jobs submitted (SLURM job ID: $JOB_ID)

📋 Monitoring commands
----------------------
# Tail output of all tasks
tail -f $TEST_OUTPUT/slurm_${JOB_ID}_*.out

# Check for errors
grep -i error $TEST_OUTPUT/slurm_${JOB_ID}_*.err

# Job status
squeue -j $JOB_ID

# Cancel all test jobs
scancel $JOB_ID

📁 Output directory
$TEST_OUTPUT/

🏁 Smoke-test launched — happy simulating!
EOF