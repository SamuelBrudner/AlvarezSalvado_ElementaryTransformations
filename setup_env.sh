#!/bin/bash
# Set up local conda prefix environment for the project
set -euo pipefail

# Load utility functions
source "$(dirname "$0")/setup_utils.sh"

usage() {
  echo "Usage: $0 [--dev] [--no-tests] [--help]"
}

# --- Configuration ---
# LOCAL_ENV_DIR: Directory name for the local Conda prefix environment.
# This will be created in the current project directory (e.g., "./dev-env/").
LOCAL_ENV_DIR="dev-env"
BASE_ENV_FILE="environment.yml"
# DEV_REQUIREMENTS_FILE: Optional file for pip-based development-specific dependencies.
DEV_REQUIREMENTS_FILE="requirements-dev.txt"
# ---

INSTALL_DEV_EXTRAS=0
RUN_TESTS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      INSTALL_DEV_EXTRAS=1
      ;;
    --no-tests)
      RUN_TESTS=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

if [ "$INSTALL_DEV_EXTRAS" -eq 1 ]; then
  log INFO "Development mode enabled: Extras like dev-specific packages and pre-commit hooks will be set up."
fi

# Check if conda is installed
if ! command -v conda >/dev/null 2>&1; then
  if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    log WARNING "conda not found, installing Miniconda"
    if ! command -v wget >/dev/null 2>&1; then
      run_command_verbose apt-get update
      run_command_verbose apt-get install -y wget bzip2
    fi
    run_command_verbose wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    run_command_verbose bash /tmp/miniconda.sh -b -p "$HOME/miniconda"
    export PATH="$HOME/miniconda/bin:$PATH"
  else
    error "conda is required but not found in PATH. Please install Miniconda or Anaconda."
  fi
fi

# Initialize conda for this shell
CONDA_BASE_DIR="$(conda info --base)"
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"

# Check if base environment file exists

if [ ! -f "$BASE_ENV_FILE" ]; then
  error "Base environment file '$BASE_ENV_FILE' not found in the current directory."
fi

# Ensure conda-lock is installed and generate lock file
if ! command -v conda-lock >/dev/null 2>&1; then
  log INFO "Installing conda-lock"
  run_command_verbose conda install -y -n base -c conda-forge conda-lock
fi

PLATFORM="$(conda info --json | python -c 'import sys,json;print(json.load(sys.stdin)["platform"])')"
log INFO "Generating conda-lock.yml for $PLATFORM"
run_command_verbose conda-lock lock -f "$BASE_ENV_FILE" -p "$PLATFORM" --lockfile conda-lock.yml

# Create/update the local prefix environment
# The environment will be located at "./$LOCAL_ENV_DIR"
section "Creating/updating local Conda environment in './$LOCAL_ENV_DIR'"
if [ -f conda-lock.yml ]; then
  run_command_verbose conda env create --prefix "./$LOCAL_ENV_DIR" --file conda-lock.yml --force
else
  run_command_verbose conda env update --prefix "./$LOCAL_ENV_DIR" -f "$BASE_ENV_FILE" --prune --yes
fi
log SUCCESS "Base environment './$LOCAL_ENV_DIR' created/updated successfully."

# Create/update the local prefix environment
# The environment will be located at "./$LOCAL_ENV_DIR"

# Install development dependencies and set up pre-commit if --dev flag is present
if [ "$INSTALL_DEV_EXTRAS" -eq 1 ]; then
  section "Processing development extras"

  # Install development-specific pip packages
  if [ -f "$DEV_REQUIREMENTS_FILE" ]; then
    section "Installing development dependencies from '$DEV_REQUIREMENTS_FILE'"
  run_command_verbose conda run --prefix "./$LOCAL_ENV_DIR" pip install -r "$DEV_REQUIREMENTS_FILE"
  if [ $? -ne 0 ]; then
      error "Failed to install development dependencies from '$DEV_REQUIREMENTS_FILE'."
  fi
  log SUCCESS "Development dependencies from '$DEV_REQUIREMENTS_FILE' installed successfully."
  else
    log WARNING "Development mode enabled, but '$DEV_REQUIREMENTS_FILE' not found. Skipping additional dev-specific pip packages."
  fi

  # Install pre-commit and set up hooks
  # pre-commit should be listed as a dependency in your environment.yml or requirements-dev.txt
  # Alternatively, install it directly here:
  section "Installing pre-commit"
  run_command_verbose conda run --prefix "./$LOCAL_ENV_DIR" pip install pre-commit
  if [ $? -ne 0 ]; then
      error "Failed to install pre-commit. Add it to '$BASE_ENV_FILE' or '$DEV_REQUIREMENTS_FILE'."
  fi
  log INFO "Setting up pre-commit hooks..."
  if run_command_verbose conda run --prefix "./$LOCAL_ENV_DIR" pre-commit install --install-hooks; then
      log SUCCESS "Pre-commit hooks set up successfully."
  else
      log WARNING "Failed to set up pre-commit hooks, but pre-commit command was found."
  fi
fi

if [ "$RUN_TESTS" -eq 1 ]; then
  section "Running tests"
  if run_command_verbose conda run --prefix "./$LOCAL_ENV_DIR" pytest -q; then
    log SUCCESS "Tests passed"
  else
    log WARNING "Some tests failed"
  fi
fi
log SUCCESS "Setup complete."
log INFO "To activate the Conda environment, run: conda activate \"./$LOCAL_ENV_DIR\""
