#!/bin/bash
# Set up local conda prefix environment for the project
set -euo pipefail

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
  echo "Error: conda is required but not found in PATH." >&2
  exit 1
fi

# Check if base environment file exists
if [ ! -f "$BASE_ENV_FILE" ]; then
  echo "Error: Base environment file '$BASE_ENV_FILE' not found in the current directory." >&2
  exit 1
fi

# Create/update the local prefix environment
# The environment will be located at "./$LOCAL_ENV_DIR"
echo "Creating/updating local Conda environment in './$LOCAL_ENV_DIR' using '$BASE_ENV_FILE'..."
conda env update --prefix "./$LOCAL_ENV_DIR" -f "$BASE_ENV_FILE" --prune --yes
if [ $? -ne 0 ]; then
  echo "Error: Failed to create/update Conda environment './$LOCAL_ENV_DIR'." >&2
  exit 1
fi
echo "Base environment './$LOCAL_ENV_DIR' created/updated successfully."

# Install development dependencies and set up pre-commit if --dev flag is present
if [ "$INSTALL_DEV_EXTRAS" -eq 1 ]; then
  echo "Processing development extras..."

  # Install development-specific pip packages
  if [ -f "$DEV_REQUIREMENTS_FILE" ]; then
    echo "Installing development dependencies from '$DEV_REQUIREMENTS_FILE' into './$LOCAL_ENV_DIR'..."
    conda run --prefix "./$LOCAL_ENV_DIR" pip install -r "$DEV_REQUIREMENTS_FILE"
    if [ $? -ne 0 ]; then
      echo "Error: Failed to install development dependencies from '$DEV_REQUIREMENTS_FILE'." >&2
      # Consider if this should be a fatal error (exit 1)
    else
      echo "Development dependencies from '$DEV_REQUIREMENTS_FILE' installed successfully."
    fi
  else
    echo "Note: Development mode enabled, but '$DEV_REQUIREMENTS_FILE' not found. Skipping additional dev-specific pip packages."
  fi

  # Install pre-commit and set up hooks
  # pre-commit should be listed as a dependency in your environment.yml or requirements-dev.txt
  # Alternatively, install it directly here:
  echo "Installing pre-commit into './$LOCAL_ENV_DIR'..."
  conda run --prefix "./$LOCAL_ENV_DIR" pip install pre-commit
  if [ $? -ne 0 ]; then
      echo "Error: Failed to install pre-commit. Add it to '$BASE_ENV_FILE' or '$DEV_REQUIREMENTS_FILE'." >&2
      # Consider if this should be a fatal error (exit 1)
  else
    echo "Setting up pre-commit hooks..."
    conda run --prefix "./$LOCAL_ENV_DIR" pre-commit install --install-hooks
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to set up pre-commit hooks, but pre-commit command was found." >&2
    else
        echo "Pre-commit hooks set up successfully."
    fi
  fi
fi

echo ""
echo "Setup complete."
echo "To activate the Conda environment, run: conda activate \"./$LOCAL_ENV_DIR\""