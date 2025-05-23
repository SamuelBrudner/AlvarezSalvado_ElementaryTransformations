#!/usr/bin/env bash
# Setup local conda environment.
# Usage: source setup_env.sh [--dev]

ENV_NAME=".env"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but not installed" >&2
    return 1 2>/dev/null || exit 1
fi

if ! conda env list | grep -q "^$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.11
fi

conda activate "$ENV_NAME"

if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

if [ "$1" = "--dev" ] && [ -f requirements-dev.txt ]; then
    pip install -r requirements-dev.txt
fi

pip install pre-commit
pre-commit install
