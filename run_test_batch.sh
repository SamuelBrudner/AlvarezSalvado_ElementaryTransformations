#!/bin/bash
# quick smoke-test launcher for the plume simulator
set -euo pipefail

TEST_NAME="plume_test_$(date +%Y%m%d_%H%M%S)"
TEST_JOBS=3
TEST_AGENTS=2
TEST_AGENTS_PER_JOB=1
TEST_TIME="00:10:00"
TEST_MEM="64G"
TEST_PARTITION="day"

# absolute paths to the YAML and the movie on the *login* node
PLUME_CONFIG="$(pwd)/configs/my_complex_plume_config.yaml"
PLUME_VIDEO="$(pwd)/data/smoke_1a_orig_backgroundsubtracted.avi"

TEST_OUTPUT="test_output/$TEST_NAME"; mkdir -p "$TEST_OUTPUT"

if [[ -n "${PLUME_METADATA:-}" ]]; then
    DESC="Metadata"
    PLUME_PATH="$PLUME_METADATA"
else
    DESC="Movie"
    PLUME_PATH="$PLUME_VIDEO"
fi

echo "🚀 Starting test batch: $TEST_NAME"
cat <<EOF

Jobs               : $TEST_JOBS
Agents/condition   : $TEST_AGENTS
$DESC              : $PLUME_PATH
Output             : $TEST_OUTPUT
EOF

# build one long --export= list
EXPORT_VARS=$(cat <<EOV
ALL,\
EXPERIMENT_NAME=$TEST_NAME,\
AGENTS_PER_CONDITION=$TEST_AGENTS,\
AGENTS_PER_JOB=$TEST_AGENTS_PER_JOB,\
OUTPUT_BASE=$TEST_OUTPUT,\
PLUME_CONFIG=$PLUME_CONFIG,\
PLUME_VIDEO=$PLUME_VIDEO,\
PLUME_METADATA=$PLUME_METADATA,\
SLURM_PARTITION=$TEST_PARTITION,\
SLURM_TIME=$TEST_TIME,\
SLURM_MEM=$TEST_MEM,\
SLURM_CPUS_PER_TASK=1,\
SLURM_ARRAY_CONCURRENT=$TEST_JOBS,\
MATLAB_OPTIONS="-nodisplay -nosplash"
EOV
)

# expand any environment variables in EXPORT_VARS (e.g., from the user's
# shell) before passing to sbatch
EXPORT_VARS=$(envsubst <<<"$EXPORT_VARS")

JOB_ID=$(sbatch \
  --job-name="$TEST_NAME" \
  --output="$TEST_OUTPUT/slurm_%A_%a.out" \
  --error="$TEST_OUTPUT/slurm_%A_%a.err" \
  --time="$TEST_TIME" --mem="$TEST_MEM" --cpus-per-task=1 \
  --array=0-$((TEST_JOBS-1)) \
  --export="$EXPORT_VARS" \
  run_batch_job_4000.sh | awk '{print $4}')

echo -e "\n✅ Submitted as array ID $JOB_ID"
echo "  tail -f $TEST_OUTPUT/slurm_${JOB_ID}_*.out"
