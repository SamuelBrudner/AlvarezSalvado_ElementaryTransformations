#!/usr/bin/env bash
# Set up local conda prefix environment for the project
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_utils.sh"

# --- Configuration ---
# LOCAL_ENV_DIR: Directory name for the local Conda prefix environment.
# This will be created in the current project directory (e.g., "./dev-env/").
LOCAL_ENV_DIR="dev-env"
BASE_ENV_FILE="environment.yml"
LOCK_FILE="conda-lock.yml"
# DEV_REQUIREMENTS_FILE: Optional file for pip-based development-specific dependencies.
DEV_REQUIREMENTS_FILE="requirements-dev.txt"
# ---

INSTALL_DEV_EXTRAS=0
if [[ "${1:-}" == "--dev" ]]; then
  INSTALL_DEV_EXTRAS=1
  log INFO "Development mode enabled: Extras like dev-specific packages and pre-commit hooks will be set up."
fi

# Check if conda is installed

if ! command -v conda >/dev/null 2>&1; then
  error "conda is required but not found in PATH. Install Miniconda and re-run."
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

generate_conda_lock() {
  if command -v conda-lock >/dev/null 2>&1; then
    if [ ! -f "$LOCK_FILE" ] || [ "$BASE_ENV_FILE" -nt "$LOCK_FILE" ]; then
      section "Generating $LOCK_FILE"
      conda-lock lock -f "$BASE_ENV_FILE" -p linux-64 --lockfile "$LOCK_FILE"
    fi
  else
    log WARNING "conda-lock not installed; skipping lock file generation"
  fi
}

generate_conda_lock

# Check if base environment file exists
if [ ! -f "$BASE_ENV_FILE" ]; then
  error "Base environment file '$BASE_ENV_FILE' not found in the current directory."
  exit 1
fi

# Create/update the local prefix environment
# The environment will be located at "./$LOCAL_ENV_DIR"
section "Creating or updating Conda environment"
ENV_FILE="$BASE_ENV_FILE"
if [ -f "$LOCK_FILE" ]; then
  ENV_FILE="$LOCK_FILE"
fi
conda env update --prefix "./$LOCAL_ENV_DIR" -f "$ENV_FILE" --prune --yes
if [ $? -ne 0 ]; then
  error "Failed to create/update Conda environment "./$LOCAL_ENV_DIR"."
  exit 1
fi
log SUCCESS "Base environment './$LOCAL_ENV_DIR' created/updated successfully."

# Install development dependencies and set up pre-commit if --dev flag is present
if [ "$INSTALL_DEV_EXTRAS" -eq 1 ]; then
  section "Processing development extras"

  # Install development-specific pip packages
  if [ -f "$DEV_REQUIREMENTS_FILE" ]; then
    log INFO "Installing development dependencies from '$DEV_REQUIREMENTS_FILE' into './$LOCAL_ENV_DIR'..."
    conda run --prefix "./$LOCAL_ENV_DIR" pip install -r "$DEV_REQUIREMENTS_FILE"
    if [ $? -ne 0 ]; then
      error "Failed to install development dependencies from '$DEV_REQUIREMENTS_FILE'." >&2
      # Consider if this should be a fatal error (exit 1)
    else
      log SUCCESS "Development dependencies from '$DEV_REQUIREMENTS_FILE' installed successfully."
    fi
  else
    log WARNING "Note: Development mode enabled, but '$DEV_REQUIREMENTS_FILE' not found. Skipping additional dev-specific pip packages."
  fi

  # Install pre-commit and set up hooks
  # pre-commit should be listed as a dependency in your environment.yml or requirements-dev.txt
  # Alternatively, install it directly here:
  run_command_verbose "Installing pre-commit into './$LOCAL_ENV_DIR'..."
  conda run --prefix "./$LOCAL_ENV_DIR" pip install pre-commit
  if [ $? -ne 0 ]; then
      error "Failed to install pre-commit. Add it to '$BASE_ENV_FILE' or '$DEV_REQUIREMENTS_FILE'." 
      # Consider if this should be a fatal error (exit 1)
  else
    log INFO "Setting up pre-commit hooks..."
    conda run --prefix "./$LOCAL_ENV_DIR" pre-commit install --install-hooks
    if [ $? -ne 0 ]; then
        log WARNING "Warning: Failed to set up pre-commit hooks, but pre-commit command was found." 
    else
        log SUCCESS "Pre-commit hooks set up successfully."
    fi
  fi
fi

log SUCCESS "Setup complete."
log INFO "To activate the Conda environment, run: conda activate \"./$LOCAL_ENV_DIR\""
