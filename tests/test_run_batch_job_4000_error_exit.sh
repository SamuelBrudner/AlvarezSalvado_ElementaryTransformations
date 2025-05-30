#!/bin/bash
set -e
TMP=$(mktemp -d)
cp run_batch_job_4000.sh "$TMP/"
cd "$TMP"

# create stub module command
cat > module <<'EOS'
#!/bin/sh
if [ "$1" = "-t" ] && [ "$2" = "avail" ]; then
  echo "MATLAB/2023b"
elif [ "$1" = "load" ]; then
  exit 0
else
  exit 0
fi
EOS
chmod +x module

# create matlab script that reports failure but exits 0
cat > matlab <<'EOS'
#!/bin/sh
echo "Simulated MATLAB failure" >&2
# no result file created and exit code 0
exit 0
EOS
chmod +x matlab

export PATH="$TMP:$PATH"
export SLURM_SUBMIT_DIR="$TMP"
export OUTPUT_BASE="$TMP/out"
mkdir -p "$OUTPUT_BASE"
export PLUME_CONFIG="$TMP/config.yaml"
: > "$PLUME_CONFIG"
export PLUME_VIDEO=""
export PLUME_METADATA=""
export AGENTS_PER_CONDITION=1
export AGENTS_PER_JOB=1
export PLUME_TYPES="p"
export SENSING_MODES="s"
export MATLAB_OPTIONS=""
export BYTES_PER_AGENT=1

if ./run_batch_job_4000.sh >run.log 2>&1; then
  echo "Script succeeded unexpectedly"
  exit 1
else
  echo "Wrapper exited with non-zero as expected"
  exit 0
fi
