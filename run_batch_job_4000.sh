#!/bin/bash
set -euo pipefail  # Exit on error, undefined variable, and pipe failure

#SBATCH --begin=now
#SBATCH --job-name=matlab_nav_sim
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1                 # 1 MATLAB per task → 1 CPU core
#SBATCH --partition=day
# ───────────────────────────────────────────────────────────
# Submit with:  sbatch --array=0-$((TOTAL_JOBS-1))%100 run_batch_job.sh
# ⇧ the "%100" part guarantees ≤ 100 concurrent tasks (≤ 100 cores)
# ───────────────────────────────────────────────────────────
#SBATCH --open-mode=append
#SBATCH --output=slurm_out/%A_%a.out
#SBATCH --error=slurm_err/%A_%a.err
#SBATCH --time=6:00:00
# Uncomment / edit to receive e-mail notifications
# #SBATCH --mail-user=your_email@example.com
# #SBATCH --mail-type=ALL

# Cleanup function to run on exit or error
cleanup() {
    local exit_code=$?
    # Clean up temporary MATLAB script if it exists
    [[ -n "${MATLAB_SCRIPT:-}" && -f "$MATLAB_SCRIPT" ]] && rm -f "$MATLAB_SCRIPT"
    exit $exit_code
}

trap cleanup EXIT SIGTERM SIGINT

# ───────────────────────────────────────────────────────────
# 1.  Directory plumbing and environment setup
# ───────────────────────────────────────────────────────────
# Create necessary directories with error checking
for dir in slurm_out slurm_err data/raw data/processed; do
    if ! mkdir -p "$dir"; then
        echo "ERROR: Failed to create directory: $dir" >&2
        exit 1
    fi
done

# Disable GUI / X11 during batch jobs
export DISPLAY=
unset X11

# Check if MATLAB module is available
if ! module is-avail matlab/R2021a 2>/dev/null; then
    echo "ERROR: MATLAB module 'matlab/R2021a' is not available" >&2
    exit 1
fi

# Load MATLAB module
if ! module load matlab/R2021a; then
    echo "ERROR: Failed to load MATLAB module" >&2
    exit 1
fi

# ───────────────────────────────────────────────────────────
# 2.  High-level experiment description
#     – 2 plumes × 2 sensing modes × 1 000 agents  = 4 000 agents
# ───────────────────────────────────────────────────────────
PLUMES=(crimaldi custom)                   # 0 → crimaldi   1 → custom-video
SENSING=(bilateral unilateral)            # 0 → bilateral  1 → unilateral
NUM_PLUMES=${#PLUMES[@]}
NUM_SENSING=${#SENSING[@]}
NUM_CONDITIONS=$((NUM_PLUMES * NUM_SENSING))

AGENTS_PER_CONDITION=1000                 # 1 000 agents for each (plume,sensing) pair
AGENTS_PER_JOB=20                         # *** tweak this to fit <100 cores ***

# Derived counts
JOBS_PER_CONDITION=$(( (AGENTS_PER_CONDITION + AGENTS_PER_JOB - 1) / AGENTS_PER_JOB ))
TOTAL_JOBS=$(( NUM_CONDITIONS * JOBS_PER_CONDITION ))

# (SLURM sets $SLURM_ARRAY_TASK_ID from 0 .. TOTAL_JOBS-1)
if [[ -n "$SLURM_ARRAY_TASK_ID" && $SLURM_ARRAY_TASK_ID -ge $TOTAL_JOBS ]]; then
  echo "Task $SLURM_ARRAY_TASK_ID exceeds TOTAL_JOBS=$TOTAL_JOBS — exiting."
  exit 0
fi

# ───────────────────────────────────────────────────────────
# 3.  Map array index → (plume,sensing) + agent-slice
# ───────────────────────────────────────────────────────────
CONDITION=$(( SLURM_ARRAY_TASK_ID % NUM_CONDITIONS ))
JOB_INDEX_IN_CONDITION=$(( SLURM_ARRAY_TASK_ID / NUM_CONDITIONS ))

PLUME_INDEX=$(( CONDITION / NUM_SENSING ))            # 0 or 1
SENSING_INDEX=$(( CONDITION % NUM_SENSING ))          # 0 or 1

PLUME_NAME=${PLUMES[$PLUME_INDEX]}
SENSING_NAME=${SENSING[$SENSING_INDEX]}

# Slice of agents handled by this job
START_AGENT=$(( JOB_INDEX_IN_CONDITION * AGENTS_PER_JOB + 1 ))
END_AGENT=$(( (JOB_INDEX_IN_CONDITION + 1) * AGENTS_PER_JOB ))
(( END_AGENT > AGENTS_PER_CONDITION )) && END_AGENT=$AGENTS_PER_CONDITION

# ───────────────────────────────────────────────────────────
# 4.  Echo run parameters
# ───────────────────────────────────────────────────────────
echo "──────── Job context ────────"
echo " SLURM_ARRAY_TASK_ID : $SLURM_ARRAY_TASK_ID"
echo " Plume               : $PLUME_NAME ($PLUME_INDEX)"
echo " Sensing             : $SENSING_NAME ($SENSING_INDEX)"
echo " Agents              : $START_AGENT – $END_AGENT  of  $AGENTS_PER_CONDITION"
echo " Agents / job        : $AGENTS_PER_JOB"
echo " Total jobs          : $TOTAL_JOBS  (limit 100 running concurrently)"
echo "──────────────────────────────"

# ───────────────────────────────────────────────────────────
# 5.  Build a temporary MATLAB script that runs every agent slice serially
#     (1 CPU per MATLAB, so we keep the bash loop outside MATLAB)
# ───────────────────────────────────────────────────────────
# Create temporary file in a more robust way
TMP_DIR="${TMPDIR:-/tmp}"
MATLAB_SCRIPT=$(mktemp -p "$TMP_DIR" batch_job_XXXXXX.m)

if [[ ! -f "$MATLAB_SCRIPT" ]]; then
    echo "ERROR: Failed to create temporary MATLAB script" >&2
    exit 1
fi

# Ensure the script is deleted on exit
trap 'rm -f "$MATLAB_SCRIPT"' EXIT

for AGENT_ID in $(seq $START_AGENT $END_AGENT); do
  SEED=$AGENT_ID                       # reproducible 1-to-1 seed
  OUT_DIR="data/raw/${PLUME_NAME}_${SENSING_NAME}/${AGENT_ID}_${SEED}"
  mkdir -p "$OUT_DIR"

  # ------- craft per-agent cfg structure in MATLAB syntax -------
  if [[ $PLUME_NAME == "crimaldi" ]]; then
    ENV_LINE="cfg.environment = 'Crimaldi';"
  else
    # custom video plume: point to metadata YAML
    local plume_config="configs/my_complex_plume_config.yaml"
    if [[ ! -f "$plume_config" ]]; then
        echo "ERROR: Plume config file not found: $plume_config" >&2
        exit 1
    fi
    ENV_LINE="cfg.plume_metadata = '$plume_config'; cfg.environment = 'video';"
  fi

  cat >> "$MATLAB_SCRIPT" <<EOF
cfg = struct();
$ENV_LINE
cfg.bilateral  = $( [[ $SENSING_NAME == "bilateral" ]] && echo true || echo false );
cfg.randomSeed = $SEED;
cfg.ntrials    = 1;
cfg.plotting   = 0;
cfg.outputDir  = '$OUT_DIR';
try
    run_navigation_cfg(cfg);
catch ME
    disp(getReport(ME,'extended'));
end

EOF
done

echo "exit;" >> "$MATLAB_SCRIPT"

# ───────────────────────────────────────────────────────────
# 6.  Launch MATLAB (single-threaded) to run the generated file
# ───────────────────────────────────────────────────────────
echo "Running MATLAB with script: $MATLAB_SCRIPT"
if ! matlab -nodisplay -nosplash -r "try, run('$MATLAB_SCRIPT'); catch ME, disp(getReport(ME, 'extended')); exit(1); end"; then
    echo "ERROR: MATLAB execution failed" >&2
    exit 1
fi

echo "Job completed successfully"
exit 0
