#!/bin/bash
# robust SLURM wrapper – absolute YAML & movie paths + auto MATLAB module
# Uses conda run -p for Python environment activation
set -euo pipefail

# Load conda if not already loaded
if [ -z "${CONDA_DEFAULT_ENV:-}" ] && [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

#SBATCH --begin=now
#SBATCH --open-mode=append
#SBATCH --output=slurm_out/%A_%a.out
#SBATCH --error=slurm_err/%A_%a.err

########################  graceful cleanup  ########################
cleanup(){ rc=$?; [[ -f ${MATLAB_SCRIPT:-}  ]] && rm -f "$MATLAB_SCRIPT"
                   [[ -f ${EXPORT_SCRIPT:-} ]] && rm -f "$EXPORT_SCRIPT"; exit $rc; }
trap cleanup EXIT SIGINT SIGTERM

########################  directories  #############################
for d in slurm_out slurm_err data/processed; do mkdir -p "$d"; done
mkdir -p "$OUTPUT_BASE"
mkdir -p logs
JOB_LOG="logs/${SLURM_ARRAY_TASK_ID:-0}.log"
echo "Starting job ${SLURM_ARRAY_TASK_ID:-0}" > "$JOB_LOG"

########################  defaults  ################################
: ${EXPERIMENT_NAME:=default_experiment}
: ${PLUME_TYPES:="crimaldi custom"}
: ${SENSING_MODES:="bilateral unilateral"}
: ${AGENTS_PER_CONDITION:=1000}
: ${AGENTS_PER_JOB:=100}
: ${PLUME_CONFIG:=configs/my_complex_plume_config.yaml}
: ${PLUME_VIDEO:=data/smoke_1a_orig_backgroundsubtracted.avi}
: ${PLUME_METADATA:=}
: ${OUTPUT_BASE:=data/raw}
: ${MATLAB_VERSION:=2023b}
: ${MATLAB_MODULE:=MATLAB/${MATLAB_VERSION}}
: ${SLURM_ARRAY_CONCURRENT:=100}
: ${MATLAB_OPTIONS:="-nodisplay -nosplash"}

########## strip stray quotes then absolutise YAML & movie #########
for var in PLUME_CONFIG PLUME_VIDEO PLUME_METADATA; do
    val=${!var:-}
    val=${val#\"}; val=${val%\"}
    if [[ -n "$val" && "$val" != /* ]]; then
        val="$SLURM_SUBMIT_DIR/$val"
    fi
    declare "$var=$val"
done

########################  counts  ##################################
IFS=' ' read -ra PLUMES   <<< "$PLUME_TYPES"
IFS=' ' read -ra SENSING  <<< "$SENSING_MODES"
NUM_CONDITIONS=$(( ${#PLUMES[@]} * ${#SENSING[@]} ))
JOBS_PER_COND=$(( (AGENTS_PER_CONDITION + AGENTS_PER_JOB -1)/AGENTS_PER_JOB ))
TOTAL_JOBS=$(( NUM_CONDITIONS * JOBS_PER_COND ))

########################  disk space check  ########################
: ${BYTES_PER_AGENT:=50000000} # BYTES_PER_AGENT=50000000

REQUIRED=$(( AGENTS_PER_CONDITION * NUM_CONDITIONS * BYTES_PER_AGENT * 12 / 10 / 1024 ))
FREE=$(df -k --output=avail "$OUTPUT_BASE" | tail -1)
(( FREE >= REQUIRED )) || { echo "ERR not enough space"; exit 1; }

########################  MATLAB module loader  ####################
load_matlab(){
  all=$(module -t avail 2>&1 | awk '{gsub(/^[[:space:]]+/,"")} /^[mM][aA][tT][lL][aA][bB]\//')
  for cand in "$1" "${1/matlab/MATLAB}" "${1/MATLAB/matlab}"; do
      printf '%s\n' "$all" | grep -qx "$cand" && { echo "$cand"; return; }
  done
  printf '%s\n' "$all" | sort -V | tail -1
}
MATLAB_MODULE=$(load_matlab "$MATLAB_MODULE") || { echo "No MATLAB module"; exit 1; }
module load "$MATLAB_MODULE"; echo "Loaded $MATLAB_MODULE"
export DISPLAY=

########################  array mapping  ###########################
TASK=${SLURM_ARRAY_TASK_ID:-0}
(( TASK<TOTAL_JOBS )) || exit 0
PICK=$(( TASK % NUM_CONDITIONS ))
BLOCK=$(( TASK / NUM_CONDITIONS ))
PLUME=${PLUMES[$((PICK/${#SENSING[@]}))]}
SENSE=${SENSING[$((PICK%${#SENSING[@]}))]}
PLUME_NAME="$PLUME"
SENSING_NAME="$SENSE"
START=$(( BLOCK*AGENTS_PER_JOB + 1 ))
END=$(( (BLOCK+1)*AGENTS_PER_JOB )); (( END>AGENTS_PER_CONDITION )) && END=$AGENTS_PER_CONDITION

########################  build MATLAB script  #####################
TMPDIR="${TMPDIR:-/tmp}"
MATLAB_SCRIPT=$(mktemp -p "$TMPDIR" batch_job_XXXX.m)

# Initialize MATLAB script with proper header
cat >"$MATLAB_SCRIPT"<<'MAT'
if isempty(which('run_navigation_cfg')), addpath(fullfile(pwd,'Code')); end
ws=warning('off','all'); cleanupObj=onCleanup(@()warning(ws));

% Process all agents
MAT

# Ensure triallength is set for fair comparison
echo "  cfg.triallength = 3600;" >>"$MATLAB_SCRIPT"

# Handle plume-specific configuration
if [[ "$PLUME_NAME" == "custom" ]]; then
  echo "  cfg.plume_metadata = '$PLUME_METADATA';" >>"$MATLAB_SCRIPT"
  echo "  cfg.environment = 'video';" >>"$MATLAB_SCRIPT"
elif [[ "$PLUME_NAME" == "crimaldi" ]]; then
  echo "  cfg.environment = 'crimaldi';" >>"$MATLAB_SCRIPT"
fi

# Generate code for each agent
for AG in $(seq $START $END); do
  AGENT_ID=$AG
  SEED=$AG
  OUT_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}/${PLUME_NAME}_${SENSING_NAME}/${AGENT_ID}_${SEED}"
  
  cat >>"$MATLAB_SCRIPT"<<MAT
% Agent $AG
try
  cfg = load_config('$PLUME_CONFIG');
MAT
  
  if [ -n "$PLUME_METADATA" ]; then
    echo "  cfg.plume_metadata = '$PLUME_METADATA';" >>"$MATLAB_SCRIPT"
  else
    echo "  cfg.plume_video = '$PLUME_VIDEO';" >>"$MATLAB_SCRIPT"
  fi
  
  cat >>"$MATLAB_SCRIPT"<<MAT
  cfg.bilateral = $( [[ $SENSE == bilateral ]] && echo true || echo false );
  cfg.randomSeed = $SEED;
  cfg.ntrials = 1; 
  cfg.plotting = 0;
  cfg.outputDir = '$OUT_DIR';
  fprintf('Saving results to %s\\n', cfg.outputDir);
  if ~exist(cfg.outputDir, 'dir')
    mkdir(cfg.outputDir);
  end
  R = run_navigation_cfg(cfg);
  save(fullfile(cfg.outputDir,'result.mat'), '-struct', 'R', '-v7');
  fprintf('Successfully saved result for agent $AG\\n');
catch ME
  fprintf(2,'ERROR: Agent $AG: %s\\n', getReport(ME));
end
pause(0.05);

MAT
  
  PROGRESS=$(( AG * 100 / AGENTS_PER_CONDITION ))
  echo "Progress: ${PROGRESS}%" >> "$JOB_LOG"
done

# Properly close the MATLAB script
cat >>"$MATLAB_SCRIPT"<<'MAT'
clear cleanupObj;
exit;
MAT

########################  run MATLAB  ##############################
echo "Running MATLAB with script: $MATLAB_SCRIPT" >> "$JOB_LOG"
matlab $MATLAB_OPTIONS -r "run('$MATLAB_SCRIPT');" || { echo "MATLAB failed"; exit 1; }

# Check if any result files were produced
if ! find "$OUTPUT_BASE" -name result.mat | grep -q .; then
  echo "ERROR: no result.mat produced" >&2
  exit 1
fi

########################  export CSV/JSON  #########################
# Use conda run -p for any Python commands
# Example: conda run -p ./dev_env python script.py

# MATLAB-based export (kept as is since it's MATLAB-specific)
EXPORT_SCRIPT=$(mktemp -p "$TMPDIR" export_job_XXXX.m)
echo "if isempty(which('export_results')), addpath(fullfile(pwd,'Code')); end" > "$EXPORT_SCRIPT"

find "$OUTPUT_BASE" -name result.mat | while read -r f; do
  out=${f/$OUTPUT_BASE/data\/processed}
  out=${out%/result.mat}
#!/bin/bash
# robust SLURM wrapper – absolute YAML & movie paths + auto MATLAB module
# Uses conda run -p for Python environment activation
set -euo pipefail

# Load conda if not already loaded
if [ -z "${CONDA_DEFAULT_ENV:-}" ] && [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

#SBATCH --begin=now
#SBATCH --open-mode=append
#SBATCH --output=slurm_out/%A_%a.out
#SBATCH --error=slurm_err/%A_%a.err

########################  graceful cleanup  ########################
cleanup(){ rc=$?; [[ -f ${MATLAB_SCRIPT:-}  ]] && rm -f "$MATLAB_SCRIPT"
                   [[ -f ${EXPORT_SCRIPT:-} ]] && rm -f "$EXPORT_SCRIPT"; exit $rc; }
trap cleanup EXIT SIGINT SIGTERM

########################  directories  #############################
for d in slurm_out slurm_err data/processed; do mkdir -p "$d"; done
mkdir -p logs
JOB_LOG="logs/${SLURM_ARRAY_TASK_ID:-0}.log"
echo "Starting job ${SLURM_ARRAY_TASK_ID:-0}" > "$JOB_LOG"

########################  defaults  ################################
: ${EXPERIMENT_NAME:=default_experiment}
: ${PLUME_TYPES:="crimaldi custom"}
: ${SENSING_MODES:="bilateral unilateral"}
: ${AGENTS_PER_CONDITION:=1000}
: ${AGENTS_PER_JOB:=100}
: ${PLUME_CONFIG:=configs/my_complex_plume_config.yaml}
: ${PLUME_VIDEO:=data/smoke_1a_orig_backgroundsubtracted.avi}
: ${PLUME_METADATA:=}
: ${OUTPUT_BASE:=data/raw}
: ${MATLAB_VERSION:=2023b}
: ${MATLAB_MODULE:=MATLAB/${MATLAB_VERSION}}
: ${SLURM_ARRAY_CONCURRENT:=100}

########## strip stray quotes then absolutise YAML & movie #########
for var in PLUME_CONFIG PLUME_VIDEO PLUME_METADATA; do
    val=${!var:-}
    val=${val#\"}; val=${val%\"}
    if [[ -n "$val" && "$val" != /* ]]; then
        val="$SLURM_SUBMIT_DIR/$val"
    fi
    declare "$var=$val"
done

########################  counts  ##################################
IFS=' ' read -ra PLUMES   <<< "$PLUME_TYPES"
IFS=' ' read -ra SENSING  <<< "$SENSING_MODES"
NUM_CONDITIONS=$(( ${#PLUMES[@]} * ${#SENSING[@]} ))
JOBS_PER_COND=$(( (AGENTS_PER_CONDITION + AGENTS_PER_JOB -1)/AGENTS_PER_JOB ))
TOTAL_JOBS=$(( NUM_CONDITIONS * JOBS_PER_COND ))

########################  disk space check  ########################
: ${BYTES_PER_AGENT:=50000000} # BYTES_PER_AGENT=50000000

REQUIRED=$(( AGENTS_PER_CONDITION * NUM_CONDITIONS * BYTES_PER_AGENT * 12 / 10 / 1024 ))
FREE=$(df -k --output=avail "$OUTPUT_BASE" | tail -1)
(( FREE >= REQUIRED )) || { echo "ERR not enough space"; exit 1; }

########################  MATLAB module loader  ####################
load_matlab(){
  all=$(module -t avail 2>&1 | awk '{gsub(/^[[:space:]]+/,"")} /^[mM][aA][tT][lL][aA][bB]\//')
  for cand in "$1" "${1/matlab/MATLAB}" "${1/MATLAB/matlab}"; do
      printf '%s\n' "$all" | grep -qx "$cand" && { echo "$cand"; return; }
  done
  printf '%s\n' "$all" | sort -V | tail -1
}
MATLAB_MODULE=$(load_matlab "$MATLAB_MODULE") || { echo "No MATLAB module"; exit 1; }
module load "$MATLAB_MODULE"; echo "Loaded $MATLAB_MODULE"
export DISPLAY=

########################  array mapping  ###########################
TASK=${SLURM_ARRAY_TASK_ID:-0}
(( TASK<TOTAL_JOBS )) || exit 0
PICK=$(( TASK % NUM_CONDITIONS ))
BLOCK=$(( TASK / NUM_CONDITIONS ))
PLUME=${PLUMES[$((PICK/${#SENSING[@]}))]}
SENSE=${SENSING[$((PICK%${#SENSING[@]}))]}
PLUME_NAME="$PLUME"
SENSING_NAME="$SENSE"
START=$(( BLOCK*AGENTS_PER_JOB + 1 ))
END=$(( (BLOCK+1)*AGENTS_PER_JOB )); (( END>AGENTS_PER_CONDITION )) && END=$AGENTS_PER_CONDITION

echo "Task $TASK: ${PLUME_NAME}_${SENSING_NAME}, agents ${START}-${END}" >> "$JOB_LOG"

########################  build MATLAB script  #####################
TMPDIR="${TMPDIR:-/tmp}"
MATLAB_SCRIPT=$(mktemp -p "$TMPDIR" batch_job_XXXX.m)

# Initialize MATLAB script with proper header
cat >"$MATLAB_SCRIPT"<<'MAT'
fprintf('Current directory: %s\n', pwd);
if isempty(which('run_navigation_cfg')), addpath(fullfile(pwd,'Code')); end
ws=warning('off','all'); cleanupObj=onCleanup(@()warning(ws));

% Process all agents
MAT

# Create output directories and generate code for each agent
for AG in $(seq $START $END); do
  AGENT_ID=$AG
  SEED=$AG
  
  # Fix: Check if OUTPUT_BASE already contains EXPERIMENT_NAME
  if [[ "$OUTPUT_BASE" == *"$EXPERIMENT_NAME"* ]]; then
    # OUTPUT_BASE already includes experiment name (from run_test_batch.sh)
    OUT_DIR="${OUTPUT_BASE}/${PLUME_NAME}_${SENSING_NAME}/${AGENT_ID}_${SEED}"
  else
    # OUTPUT_BASE doesn't include experiment name (normal case)
    OUT_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}/${PLUME_NAME}_${SENSING_NAME}/${AGENT_ID}_${SEED}"
  fi
  
  # Create the directory from the shell to ensure it exists
  mkdir -p "$OUT_DIR"
  echo "Created directory: $OUT_DIR" >> "$JOB_LOG"
  
  # Get absolute path
  OUT_DIR_ABS=$(cd "$OUT_DIR" && pwd)
  
  cat >>"$MATLAB_SCRIPT"<<MAT
% Agent $AG
fprintf('\\n=== Processing Agent $AG ===\\n');
try
  cfg = load_config('$PLUME_CONFIG');
MAT
  
  if [ -n "$PLUME_METADATA" ]; then
    echo "  cfg.plume_metadata = '$PLUME_METADATA';" >>"$MATLAB_SCRIPT"
  else
    echo "  cfg.plume_video = '$PLUME_VIDEO';" >>"$MATLAB_SCRIPT"
  fi
  
  cat >>"$MATLAB_SCRIPT"<<MAT
  cfg.bilateral = $( [[ $SENSE == bilateral ]] && echo true || echo false );
  cfg.randomSeed = $SEED;
  cfg.ntrials = 1; 
  cfg.plotting = 0;
  cfg.outputDir = '$OUT_DIR_ABS';
  
  fprintf('Configuration:\\n');
  fprintf('  Output dir: %s\\n', cfg.outputDir);
  fprintf('  Plume: ${PLUME_NAME}\\n');
  fprintf('  Sensing: ${SENSING_NAME}\\n');
  fprintf('  Bilateral: %d\\n', cfg.bilateral);
  
  % Double-check directory exists
  if ~exist(cfg.outputDir, 'dir')
    fprintf('Creating directory from MATLAB: %s\\n', cfg.outputDir);
    mkdir(cfg.outputDir);
  end
  
  fprintf('Running simulation...\\n');
  R = run_navigation_cfg(cfg);
  
  % Save with explicit path
  resultPath = fullfile(cfg.outputDir, 'result.mat');
  fprintf('Saving to: %s\\n', resultPath);
  save(resultPath, '-struct', 'R', '-v7');
  
  % Verify save worked
  if exist(resultPath, 'file')
    fprintf('SUCCESS: Result saved for agent $AG\\n');
    d = dir(resultPath);
    fprintf('  File size: %d bytes\\n', d.bytes);
  else
    error('Failed to save result.mat');
  end
  
catch ME
  fprintf(2,'ERROR: Agent $AG failed\\n');
  fprintf(2,'  Message: %s\\n', ME.message);
  fprintf(2,'  Stack:\\n');
  for k = 1:length(ME.stack)
    fprintf(2,'    %s (line %d)\\n', ME.stack(k).name, ME.stack(k).line);
  end
end
pause(0.05);

MAT
  
  PROGRESS=$(( AG * 100 / AGENTS_PER_CONDITION ))
  echo "Progress: ${PROGRESS}%" >> "$JOB_LOG"
done

# Properly close the MATLAB script
cat >>"$MATLAB_SCRIPT"<<'MAT'
fprintf('\nAll agents completed.\n');
clear cleanupObj;
exit;
MAT

########################  run MATLAB  ##############################
echo "Running MATLAB with script: $MATLAB_SCRIPT" >> "$JOB_LOG"
echo "Working directory: $(pwd)" >> "$JOB_LOG"
matlab -nodisplay -nosplash -r "run('$MATLAB_SCRIPT');" || { echo "MATLAB failed"; exit 1; }

########################  check results  ##############################
# List all directories created
echo "Checking output directories:" >> "$JOB_LOG"
find "$OUTPUT_BASE" -type d | sort >> "$JOB_LOG"

echo "Looking for result.mat files:" >> "$JOB_LOG"
find "$OUTPUT_BASE" -name "*.mat" -type f -ls >> "$JOB_LOG" || echo "No .mat files found" >> "$JOB_LOG"

# Check if any result files were produced
RESULT_COUNT=$(find "$OUTPUT_BASE" -name result.mat -type f 2>/dev/null | wc -l)
echo "Found $RESULT_COUNT result.mat files" >> "$JOB_LOG"

if [ "$RESULT_COUNT" -eq 0 ]; then
  echo "ERROR: no result.mat produced" >&2
  echo "Directory contents:" >&2
  ls -la "$OUTPUT_BASE"/ >&2 || true
  find "$OUTPUT_BASE" -type f -name "*" | head -20 >&2 || true
  exit 1
fi

########################  export CSV/JSON  #########################
# MATLAB-based export
EXPORT_SCRIPT=$(mktemp -p "$TMPDIR" export_job_XXXX.m)
echo "if isempty(which('export_results')), addpath(fullfile(pwd,'Code')); end" > "$EXPORT_SCRIPT"

find "$OUTPUT_BASE" -name result.mat | while read -r f; do
  out=${f/$OUTPUT_BASE/data\/processed}
  out=${out%/result.mat}
  
  echo "Exporting: $f -> $out" >> "$JOB_LOG"
  mkdir -p "$out"
  
  cat >> "$EXPORT_SCRIPT" <<EOF
try
  export_results('$f','$out','Format','both');
  fprintf('Exported: $f\\n');
catch ME
  fprintf(2,'Export failed for $f: %s\\n', ME.message);
end
EOF
done

echo "exit;" >> "$EXPORT_SCRIPT"

if [[ -s "$EXPORT_SCRIPT" ]]; then
  echo "Running export script..." >> "$JOB_LOG"
  matlab -nodisplay -nosplash -r "run('$EXPORT_SCRIPT');" || true
fi

echo "Job finished successfully." | tee -a "$JOB_LOG"
echo "Summary: processed agents ${START}-${END} of ${AGENTS_PER_CONDITION} for ${SENSING_NAME} ${PLUME_NAME}" >> "$JOB_LOG"