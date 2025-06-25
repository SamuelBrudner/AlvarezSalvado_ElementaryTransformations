#!/bin/bash

# ---------------------------------------------------------------------------
# Setup a development conda environment for the AlvarezSalvado project.
# ---------------------------------------------------------------------------
set -euo pipefail

# Verbose logging flag
VERBOSE=0

# Logging function for verbose output
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
        local message="[$timestamp] setup_env.sh: $*"
        echo "$message" >&2
        
        # Also log to file if logs directory exists or can be created
        local log_dir="../logs"
        if [[ -d "$log_dir" ]] || mkdir -p "$log_dir" 2>/dev/null; then
            echo "$message" >> "$log_dir/setup_env.log"
        fi
    fi
}

usage() {
    cat <<'USAGE'
Usage: setup_env.sh [--dev] [--prefix PATH] [--print] [-v|--verbose] [-h|--help]

Options:
  --dev            Create the development environment
  --prefix PATH    Environment prefix (default: ./dev_env)
  --print          Print the conda command instead of executing it
  -v, --verbose    Enable verbose logging output
  -h, --help       Show this help message
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="dev_env"
# Updated path to account for script being in scripts/ subdirectory
ENV_FILE="$SCRIPT_DIR/../configs/environment.yaml"
PRINT_ONLY=false
DEV=false

log_verbose "Starting environment setup script"
log_verbose "Script directory: $SCRIPT_DIR"
log_verbose "Environment file path: $ENV_FILE"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV=true
            log_verbose "Development mode enabled"
            shift
            ;;
        --prefix)
            PREFIX="$2"
            log_verbose "Custom prefix set: $PREFIX"
            shift 2
            ;;
        --print)
            PRINT_ONLY=true
            log_verbose "Print-only mode enabled"
            shift
            ;;
        -v|--verbose)
            VERBOSE=1
            log_verbose "Verbose logging enabled"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

log_verbose "Argument parsing completed"

if ! $DEV; then
    log_verbose "ERROR: --dev flag not provided"
    usage >&2
    exit 1
fi

log_verbose "Checking if environment file exists: $ENV_FILE"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Environment file not found: $ENV_FILE" >&2
    log_verbose "ERROR: Environment file not found at $ENV_FILE"
    exit 1
fi

log_verbose "Environment file found successfully"

CMD=(conda env create --file "$ENV_FILE" --prefix "$PREFIX")

log_verbose "Conda command prepared: ${CMD[*]}"

printf 'Creating conda environment at %s\n' "$PREFIX" >&2
printf 'Using environment file %s\n' "$ENV_FILE" >&2
printf 'Command: %s\n' "${CMD[*]}" >&2

if $PRINT_ONLY; then
    log_verbose "Print-only mode: displaying command without execution"
    printf '%s ' "${CMD[@]}"
    echo
else
    log_verbose "Checking if conda is available in PATH"
    if command -v conda >/dev/null 2>&1; then
        log_verbose "Conda found, executing environment creation command"
        "${CMD[@]}"
        log_verbose "Conda environment creation completed successfully"
    else
        echo "conda not found in PATH" >&2
        log_verbose "ERROR: conda not found in PATH"
        exit 1
    fi
fi

log_verbose "Environment setup script completed successfully"