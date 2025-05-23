#!/bin/bash
# Set up local conda prefix environment for the project
set -euo pipefail

# Load utility functions
source "$(dirname "$0")/setup_utils.sh"

# --- Configuration ---
# LOCAL_ENV_DIR: Directory name for the local Conda prefix environment.
# This will be created in the current project directory (e.g., "./dev-env/").
LOCAL_ENV_DIR="dev-env"
BASE_ENV_FILE="environment.yml"
# DEV_REQUIREMENTS_FILE: Optional file for pip-based development-specific dependencies.
DEV_REQUIREMENTS_FILE="requirements-dev.txt"
# ---

INSTALL_DEV_EXTRAS=0
if [[ "${1:-}" == "--dev" ]]; then
  INSTALL_DEV_EXTRAS=1
  echo "Development mode enabled: Extras like dev-specific packages and pre-commit hooks will be set up."
fi

# Check if conda is installed
if ! command -v conda >/dev/null 2>&1; then
  error "conda is required but not found in PATH."
fi

# Check if base environment file exists
if [ ! -f "$BASE_ENV_FILE" ]; then
  error "Base environment file '$BASE_ENV_FILE' not found in the current directory."
fi

# Create/update the local prefix environment
# The environment will be located at "./$LOCAL_ENV_DIR"
section "Creating/updating local Conda environment in './$LOCAL_ENV_DIR' using '$BASE_ENV_FILE'"
run_command_verbose conda env update --prefix "./$LOCAL_ENV_DIR" -f "$BASE_ENV_FILE" --prune --yes
if [ $? -ne 0 ]; then
  error "Failed to create/update Conda environment './$LOCAL_ENV_DIR'."
fi
echo "Base environment './$LOCAL_ENV_DIR' created/updated successfully."

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
    echo "Development dependencies from '$DEV_REQUIREMENTS_FILE' installed successfully."
  else
    echo "Note: Development mode enabled, but '$DEV_REQUIREMENTS_FILE' not found. Skipping additional dev-specific pip packages."
  fi

  # Install pre-commit and set up hooks
  # pre-commit should be listed as a dependency in your environment.yml or requirements-dev.txt
  # Alternatively, install it directly here:
  section "Installing pre-commit"
  run_command_verbose conda run --prefix "./$LOCAL_ENV_DIR" pip install pre-commit
  if [ $? -ne 0 ]; then
      error "Failed to install pre-commit. Add it to '$BASE_ENV_FILE' or '$DEV_REQUIREMENTS_FILE'."
  fi
  echo "Setting up pre-commit hooks..."
  if run_command_verbose conda run --prefix "./$LOCAL_ENV_DIR" pre-commit install --install-hooks; then
      echo "Pre-commit hooks set up successfully."
  else
      echo "Warning: Failed to set up pre-commit hooks, but pre-commit command was found." >&2
  fi
fi

echo ""
echo "Setup complete."
echo "To activate the Conda environment, run: conda activate \"./$LOCAL_ENV_DIR\""