#!/bin/bash
# Simple environment setup script
usage() {
    echo "Usage: $0 --dev"
    exit 1
}

if [ "$1" != "--dev" ]; then
    usage
fi

set -e

ENV_NAME="nav_model_dev"

echo "Creating conda environment $ENV_NAME" >&2
conda create -y -n "$ENV_NAME" python=3.12 >/dev/null

echo "Installing requirements" >&2
conda run -n "$ENV_NAME" pip install -r requirements.txt >/dev/null

echo "Environment $ENV_NAME ready" >&2
