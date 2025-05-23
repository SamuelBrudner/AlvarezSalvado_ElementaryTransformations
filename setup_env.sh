#!/usr/bin/env bash

# setup_env.sh - Configure development environment
# Usage: source setup_env.sh [--dev]

# Prevent re-running if already sourced
if [[ -n "${SETUP_ENV_INITIALIZED}" ]]; then
    return 0
fi
export SETUP_ENV_INITIALIZED=1

usage() {
    echo "Usage: source setup_env.sh [--dev]"
}

DEV=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV=1
            shift
            ;;
        -h|--help)
            usage
            return 0
            ;;
        *)
            usage
            return 1
            ;;
    esac
done

if [[ "$DEV" -eq 1 ]]; then
    if command -v conda >/dev/null 2>&1; then
        if conda env list | grep -q '^\.env'; then
            echo "Conda environment .env already exists"
        else
            echo "Creating conda environment .env"
            conda create -y -n .env python=3.10 numpy pandas pytest h5py pyyaml pre-commit >/dev/null
        fi
        echo "Installing pre-commit hooks"
        conda run -n .env pre-commit install >/dev/null
    else
        echo "conda not found; skipping environment creation"
    fi
fi

return 0
