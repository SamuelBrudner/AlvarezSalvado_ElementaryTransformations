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
RAW_DIR="$OUTPUT_BASE"; mkdir -p "$RAW_DIR"
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

cat >"$MATLAB_SCRIPT"<<MAT
if isempty(which('run_navigation_cfg')), addpath(fullfile(pwd,'Code')); end
ws=warning('off','all'); cleanupObj=onCleanup(@()warning(ws));
MAT
for AG in $(seq $START $END); do
  AGENT_ID=$AG
  SEED=$AG
  OUT_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}/${PLUME_NAME}_${SENSING_NAME}/${AGENT_ID}_${SEED}"
  cat >>"$MATLAB_SCRIPT"<<MAT
try
  cfg = load_config('$PLUME_CONFIG');
MAT
  if [ -n "$PLUME_METADATA" ]; then
    echo "  cfg.plume_metadata = '$PLUME_METADATA';" >>"$MATLAB_SCRIPT"
  else
    echo "  cfg.plume_video = '$PLUME_VIDEO';" >>"$MATLAB_SCRIPT"
  fi
cat >>"$MATLAB_SCRIPT"<<MAT
  cfg.bilateral   = $( [[ $SENSE == bilateral ]] && echo true || echo false );
  cfg.randomSeed  = $SEED;
  cfg.ntrials=1; cfg.plotting=0;
  cfg.outputDir   = '$OUT_DIR';
  mkdir(cfg.outputDir);
  R = run_navigation_cfg(cfg);
  save(fullfile(cfg.outputDir,'result.mat'),'R','-v7');
catch ME, fprintf(2,'Seed %d: %s\\n',$AG,getReport(ME)); end
pause(0.05);
MAT
  PROGRESS=$(( AG * 100 / AGENTS_PER_CONDITION ))
  echo "Progress: ${PROGRESS}%" >> "$JOB_LOG"
done
echo "exit" >>"$MATLAB_SCRIPT"

########################  run MATLAB  ##############################
matlab $MATLAB_OPTIONS -r "run('$MATLAB_SCRIPT');" || { echo "MATLAB failed"; exit 1; }

########################  export CSV/JSON  #########################
# Use conda run -p for any Python commands
# Example: conda run -p ./dev_env python script.py

# MATLAB-based export (kept as is since it's MATLAB-specific)
EXPORT_SCRIPT=$(mktemp -p "$TMPDIR" export_job_XXXX.m)
find "$OUTPUT_BASE" -name result.mat | while read -r f; do
  out=${f/$OUTPUT_BASE/data\/processed}; out=${out%/result.mat}
  mkdir -p "$out"; echo "try,export_results('$f','$out','Format','both');catch,end" >>"$EXPORT_SCRIPT"
done
echo "clear cleanupObj;" >>"$EXPORT_SCRIPT"
echo "exit" >>"$EXPORT_SCRIPT"
[[ -s "$EXPORT_SCRIPT" ]] && matlab -nodisplay -nosplash -r "run('$EXPORT_SCRIPT');" || true

echo "Job finished successfully."
echo "Summary: processed agents ${START}-${END} of ${AGENTS_PER_CONDITION} for ${SENSE} ${PLUME}" >> "$JOB_LOG"

# Example of how to run Python scripts with the conda environment:
# if [ -d "./dev_env" ]; then
#     conda run -p ./dev_env python your_script.py --input input_file --output output_dir
# fi
