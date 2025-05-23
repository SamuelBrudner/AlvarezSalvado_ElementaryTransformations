#!/bin/bash
# Set up local conda prefix environment for the project
set -euo pipefail

# --- Logging utility ---
log() {
  local level="$1"; shift
  local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

  local color_reset="\e[0m"
  local color_info="\e[34m"
  local color_success="\e[32m"
  local color_warning="\e[33m"
  local color_error="\e[31m"

  local color="$color_info"
  case "$level" in
    INFO) color="$color_info" ;;
    SUCCESS) color="$color_success" ;;
    WARNING) color="$color_warning" ;;
    ERROR) color="$color_error" ;;
  esac

  echo -e "${color}[$timestamp] [$level] $*${color_reset}"
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
if [[ "${1:-}" == "--dev" ]]; then
  INSTALL_DEV_EXTRAS=1
  log INFO "Development mode enabled: Extras like dev-specific packages and pre-commit hooks will be set up."
fi

# Check if conda is installed
if ! command -v conda >/dev/null 2>&1; then
  log ERROR "conda is required but not found in PATH." >&2
  exit 1
fi

# Check if base environment file exists
if [ ! -f "$BASE_ENV_FILE" ]; then
  log ERROR "Base environment file '$BASE_ENV_FILE' not found in the current directory." >&2
  exit 1
fi

# Create/update the local prefix environment
# The environment will be located at "./$LOCAL_ENV_DIR"
log INFO "Creating/updating local Conda environment in './$LOCAL_ENV_DIR' using '$BASE_ENV_FILE'..."
conda env update --prefix "./$LOCAL_ENV_DIR" -f "$BASE_ENV_FILE" --prune --yes
if [ $? -ne 0 ]; then
  log ERROR "Failed to create/update Conda environment './$LOCAL_ENV_DIR'." >&2
  exit 1
fi
log SUCCESS "Base environment './$LOCAL_ENV_DIR' created/updated successfully."

# Install development dependencies and set up pre-commit if --dev flag is present
if [ "$INSTALL_DEV_EXTRAS" -eq 1 ]; then
  log INFO "Processing development extras..."

  # Install development-specific pip packages
  if [ -f "$DEV_REQUIREMENTS_FILE" ]; then
    log INFO "Installing development dependencies from '$DEV_REQUIREMENTS_FILE' into './$LOCAL_ENV_DIR'..."
    conda run --prefix "./$LOCAL_ENV_DIR" pip install -r "$DEV_REQUIREMENTS_FILE"
    if [ $? -ne 0 ]; then
      log ERROR "Failed to install development dependencies from '$DEV_REQUIREMENTS_FILE'." >&2
      # Consider if this should be a fatal error (exit 1)
    else
      log SUCCESS "Development dependencies from '$DEV_REQUIREMENTS_FILE' installed successfully."
    fi
  else
    log WARNING "Development mode enabled, but '$DEV_REQUIREMENTS_FILE' not found. Skipping additional dev-specific pip packages."
  fi

  # Install pre-commit and set up hooks
  # pre-commit should be listed as a dependency in your environment.yml or requirements-dev.txt
  # Alternatively, install it directly here:
  log INFO "Installing pre-commit into './$LOCAL_ENV_DIR'..."
  conda run --prefix "./$LOCAL_ENV_DIR" pip install pre-commit
  if [ $? -ne 0 ]; then
      log ERROR "Failed to install pre-commit. Add it to '$BASE_ENV_FILE' or '$DEV_REQUIREMENTS_FILE'." >&2
      # Consider if this should be a fatal error (exit 1)
  else
    log INFO "Setting up pre-commit hooks..."
    conda run --prefix "./$LOCAL_ENV_DIR" pre-commit install --install-hooks
    if [ $? -ne 0 ]; then
        log WARNING "Failed to set up pre-commit hooks, but pre-commit command was found." >&2
    else
        log SUCCESS "Pre-commit hooks set up successfully."
    fi
  fi
fi

echo ""
log SUCCESS "Setup complete."
log INFO "To activate the Conda environment, run: conda activate \"./$LOCAL_ENV_DIR\""

