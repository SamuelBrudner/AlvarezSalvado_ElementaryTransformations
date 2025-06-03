#!/bin/bash

# ---------------------------------------------------------------------------
# Setup a development conda environment for the AlvarezSalvado project.
# ---------------------------------------------------------------------------
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: setup_env.sh [--dev] [--prefix PATH] [--print] [-h|--help]

Options:
  --dev            Create the development environment
  --prefix PATH    Environment prefix (default: ./dev_env)
  --print          Print the conda command instead of executing it
  -h, --help       Show this help message
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="dev_env"
ENV_FILE="$SCRIPT_DIR/configs/environment.yaml"
PRINT_ONLY=false
DEV=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV=true
            shift
            ;;
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --print)
            PRINT_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

if ! $DEV; then
    usage >&2
    exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Environment file not found: $ENV_FILE" >&2
    exit 1
fi

CMD=(conda env create --file "$ENV_FILE" --prefix "$PREFIX")

printf 'Creating conda environment at %s\n' "$PREFIX" >&2
printf 'Using environment file %s\n' "$ENV_FILE" >&2
printf 'Command: %s\n' "${CMD[*]}" >&2

if $PRINT_ONLY; then
    printf '%s ' "${CMD[@]}"
    echo
else
    if command -v conda >/dev/null 2>&1; then
        "${CMD[@]}"
    else
        echo "conda not found in PATH" >&2
        exit 1
    fi
fi

