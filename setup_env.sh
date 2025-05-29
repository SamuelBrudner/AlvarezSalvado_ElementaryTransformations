#!/bin/bash
# Set up local conda prefix environment for the project

# Handle both sourced and direct execution
(return 0 2>/dev/null) && SOURCED=1 || SOURCED=0

# Only set these options when not sourced to prevent exiting the parent shell
if [ "$SOURCED" -eq 0 ]; then
    set -euo pipefail
    set -o pipefail || true
else
    set -u  # Only fail on undefined variables when sourced
fi

# Function to safely exit or return
safe_exit() {
    local exit_code=$1
    shift
    [ "$SOURCED" -eq 1 ] && return $exit_code || exit $exit_code
}

# Debug function
debug() {
    if [ "${DEBUG:-0}" -eq 1 ]; then
        echo "[DEBUG] $*" >&2
    fi
}

# Load utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-}")" && pwd)"

# Source setup_utils.sh safely
if [ ! -f "${SCRIPT_DIR}/setup_utils.sh" ]; then
    echo "Error: setup_utils.sh not found in ${SCRIPT_DIR}/" >&2
    safe_exit 1
fi

# shellcheck source=./setup_utils.sh
if ! source "${SCRIPT_DIR}/setup_utils.sh"; then
    echo "Error: Failed to source setup_utils.sh" >&2
    safe_exit 1
fi

debug "Script directory: $SCRIPT_DIR"
debug "SOURCED: $SOURCED"

# Only proceed with setup if not sourced or if explicitly run
if [ "$SOURCED" -eq 1 ] && [ "${1:-}" != "--run-setup" ]; then
    debug "Script sourced, not running setup. Use 'source $0 --run-setup' to force setup."
    return 0 2>/dev/null || safe_exit 0
fi

# --- Configuration ---
# Readonly constants for better maintainability
readonly LOCAL_ENV_DIR="dev_env"
readonly BASE_ENV_FILE="environment.yml"
readonly DEV_ENV_FILE="dev-environment.yml"
readonly PRE_COMMIT_TEMPLATE=".pre-commit-config.yaml.template"
readonly PRE_COMMIT_CONFIG=".pre-commit-config.yaml"
readonly PATHS_SCRIPT="$(dirname "$0")/paths.sh"
# ---

# Setup usage function
usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --dev                Install development dependencies and set up pre-commit hooks
  --no-tests           Skip running tests after setup
  --skip-conda-lock    Skip conda-lock installation and lock file generation
  --help               Show this help message

Set DEBUG=1 to enable verbose logging.

When sourced, the script will set up the environment but not run any installation steps.
When executed directly, it will perform the full setup process.

Examples:
  # Source the script to set up environment variables
  source $0

  # Run the full setup
  $0 --dev

  # Run with debug output
  DEBUG=1 $0 --dev
EOF
  safe_exit 0
}

# Parse command line arguments
INSTALL_DEV_EXTRAS=0
RUN_TESTS=1
SKIP_CONDA_LOCK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      INSTALL_DEV_EXTRAS=1
      log INFO "Development mode enabled: Extras like dev-specific packages and pre-commit hooks will be set up."
      ;;
    --no-tests)
      RUN_TESTS=0
      log INFO "Tests will be skipped after setup."
      ;;
    --skip-conda-lock)
      SKIP_CONDA_LOCK=1
      log INFO "Skipping conda-lock installation and lock file generation."
      ;;
    -h|--help)
      usage
      ;;
    *)
      log ERROR "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

# Detect whether 'conda env create --force' is supported
conda_supports_force() {
  conda env create --help 2>&1 | grep -q -- '--force'
}

# Attempt to load Conda via the environment modules system
try_load_conda_module() {
  if type module >/dev/null 2>&1; then
    for m in miniconda anaconda conda; do
      if module avail "$m" 2>&1 | grep -qi "$m"; then
        log INFO "Loading $m module for Conda"
        if module load "$m"; then
          return 0
        fi
      fi
    done

  fi
  return 1
}

# Ensure conda-lock command exists and functions
ensure_conda_lock() {
    if ! command -v conda-lock >/dev/null 2>&1 || ! conda-lock --version >/dev/null 2>&1; then
        if [ -x "./${LOCAL_ENV_DIR}/bin/conda-lock" ]; then
            log INFO "Using conda-lock from ${LOCAL_ENV_DIR}"
            export PATH="./${LOCAL_ENV_DIR}/bin:${PATH}"
            hash -r
            return 0
        fi
        if [ -d "./${LOCAL_ENV_DIR}" ]; then
            log INFO "Installing conda-lock into ${LOCAL_ENV_DIR}"
            if ! run_command_verbose conda run --prefix "./${LOCAL_ENV_DIR}" conda install -y -c conda-forge conda-lock; then
                log WARNING "conda install failed, attempting pip fallback"
                if ! run_command_verbose conda run --prefix "./${LOCAL_ENV_DIR}" python -m pip install conda-lock; then
                    log WARNING "Failed to install conda-lock in prefix, falling back to user"
                    if ! run_command_verbose python -m pip install --user conda-lock; then
                        error "Failed to install conda-lock"
                    fi
                fi
            fi
        else
            log INFO "Installing conda-lock via pip"
            if ! run_command_verbose python -m pip install --user conda-lock; then
                error "Failed to install conda-lock with pip"
            fi
        fi

        # Refresh the shell to update PATH
        if [ -f "${CONDA_BASE_DIR}/etc/profile.d/conda.sh" ]; then
            source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh"
        fi

        # Add conda to PATH
        export PATH="${CONDA_BASE_DIR}/bin:${PATH}"

        # Always include user's pip bin directory
        USER_BIN="$(python -m site --user-base)/bin"
        export PATH="${USER_BIN}:${PATH}"
        [ -x "${USER_BIN}/conda-lock" ] && "${USER_BIN}/conda-lock" --version >/dev/null 2>&1 || true
        hash -r

        # Verify installation
        if ! command -v conda-lock >/dev/null 2>&1; then
            if [ -x "${USER_BIN}/conda-lock" ]; then
                "${USER_BIN}/conda-lock" --version >/dev/null 2>&1 || true
            else
                error "conda-lock installed but not on PATH. Add ${USER_BIN}/conda-lock to PATH."
            fi
        fi
    fi
}
cleanup_nfs_temp_files() {
    find "./${LOCAL_ENV_DIR}" -name '.nfs*' -type f -exec rm -f {} +
}

conda_env_exists() {
    if [ -d "./${LOCAL_ENV_DIR}/conda-meta" ]; then
        return 0
    fi
    return 1
}


check_not_in_active_env() {
    if [ -d "./${LOCAL_ENV_DIR}" ]; then
        local env_path
        env_path="$(cd "./${LOCAL_ENV_DIR}" && pwd)"
        if [ "${CONDA_PREFIX:-}" = "$env_path" ]; then
            if ! command -v conda-lock >/dev/null 2>&1; then
                log ERROR "dev_env is currently active. Please 'conda deactivate' before running setup_env.sh"
                return 1
            fi
        fi
    fi
}

# --- Main setup function ---
setup_environment() {
  section "Starting environment setup"

  # Check if conda is installed, attempt to load via modules if missing
  if ! command -v conda >/dev/null 2>&1; then
    if ! check_not_in_active_env; then
      return 1
    fi
    if try_load_conda_module && command -v conda >/dev/null 2>&1; then
      log INFO "Loaded Conda via module system"
    elif [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
      log WARNING "conda not found, installing Miniconda"
      if ! command -v wget >/dev/null 2>&1; then
        run_command_verbose apt-get update
        run_command_verbose apt-get install -y wget bzip2
      fi
      run_command_verbose wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
      run_command_verbose bash /tmp/miniconda.sh -b -p "$HOME/miniconda"
      export PATH="$HOME/miniconda/bin:$PATH"
    else
      error "conda is required but not found in PATH. Please load the appropriate module or install Miniconda."
    fi
  fi

  # Initialize conda for this shell
  CONDA_BASE_DIR="$(conda info --base 2>/dev/null || echo "")"
  if [ -z "$CONDA_BASE_DIR" ] || [ ! -d "$CONDA_BASE_DIR" ]; then
    error "Failed to determine conda base directory"
  fi
  
  # Source conda.sh safely
  if [ -f "${CONDA_BASE_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh"
  else
    log WARNING "conda.sh not found in expected location, conda commands might not work correctly"
  fi

  # Check if base environment file exists
  if [ ! -f "$BASE_ENV_FILE" ]; then
    error "Base environment file '$BASE_ENV_FILE' not found in the current directory."
  fi

  if [ "$SKIP_CONDA_LOCK" -ne 1 ]; then
    # Ensure conda-lock is installed and functional
    ensure_conda_lock

    # Get platform information safely
    if ! PLATFORM="$(conda info --json 2>/dev/null | python -c 'import sys,json;print(json.load(sys.stdin).get("platform", ""))' 2>/dev/null)" || [ -z "$PLATFORM" ]; then
      # Fallback for older conda versions or if json parsing fails
      PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"
      log WARNING "Could not detect conda platform, using fallback: $PLATFORM"
    fi

    # Always regenerate lock file to ensure consistency
    section "Generating conda environment lock file"
    log INFO "Generating conda-lock.yml for $PLATFORM"

    if [ -f "conda-lock.yml" ]; then
      log INFO "Removing existing conda-lock.yml"
      rm -f "conda-lock.yml"
    fi

    # Get conda-lock version to handle different versions
    CONDA_LOCK_VERSION=$(conda-lock --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "")

    log INFO "Creating conda-lock file from $BASE_ENV_FILE"
    if [ -z "$CONDA_LOCK_VERSION" ]; then
      # If we can't determine version, try without --overwrite
      run_command_verbose conda-lock lock -f "${BASE_ENV_FILE}" -p "${PLATFORM}" --lockfile conda-lock.yml
    else
      # For versions that support --overwrite
      run_command_verbose conda-lock lock -f "${BASE_ENV_FILE}" -p "${PLATFORM}" --lockfile conda-lock.yml --overwrite 2>/dev/null || \
        run_command_verbose conda-lock lock -f "${BASE_ENV_FILE}" -p "${PLATFORM}" --lockfile conda-lock.yml
    fi
  fi

  # Abort if dev_env is currently active and conda-lock is unavailable
  if ! check_not_in_active_env; then
    return 1
  fi

  # Create/update the local prefix environment
  section "Setting up Conda environment"
  log INFO "Creating/updating local Conda environment in './$LOCAL_ENV_DIR'"
  
  if [ -f "conda-lock.yml" ]; then
    if conda_env_exists; then
      log INFO "Updating existing environment from lock file"
      run_command_verbose conda env update --prefix "./${LOCAL_ENV_DIR}" --file conda-lock.yml
    elif conda_supports_force; then
      run_command_verbose conda env create --prefix "./${LOCAL_ENV_DIR}" --file conda-lock.yml --force
    else
      log INFO "Old conda detected - removing existing environment"
      run_command_verbose conda env remove --prefix "./${LOCAL_ENV_DIR}" -y || true
      cleanup_nfs_temp_files || true
      run_command_verbose conda env create --prefix "./${LOCAL_ENV_DIR}" --file conda-lock.yml
    fi
  else
    log WARNING "conda-lock.yml not found, falling back to direct environment creation"
    if conda_env_exists; then
      log INFO "Updating existing environment using $BASE_ENV_FILE"
      run_command_verbose conda env update --prefix "./${LOCAL_ENV_DIR}" -f "${BASE_ENV_FILE}"
    elif conda_supports_force; then
      run_command_verbose conda env create --prefix "./${LOCAL_ENV_DIR}" -f "${BASE_ENV_FILE}" --force
    else
      log INFO "Old conda detected - removing existing environment"
      run_command_verbose conda env remove --prefix "./${LOCAL_ENV_DIR}" -y || true
      cleanup_nfs_temp_files || true
      run_command_verbose conda env create --prefix "./${LOCAL_ENV_DIR}" -f "${BASE_ENV_FILE}"
    fi
  fi
  
  log SUCCESS "Base environment './${LOCAL_ENV_DIR}' created/updated successfully."

  # Verify NumPy is available in the environment
  if ! conda run --prefix "./${LOCAL_ENV_DIR}" python -c "import numpy" >/dev/null 2>&1; then
    log INFO "NumPy not found, installing into ${LOCAL_ENV_DIR}..."
    if ! conda run --prefix "./${LOCAL_ENV_DIR}" conda install -y numpy; then
      log WARNING "conda install numpy failed, attempting pip fallback"
      if ! conda run --prefix "./${LOCAL_ENV_DIR}" python -m pip install numpy; then
        log ERROR "Failed to install NumPy via conda or pip"
        return 1
      fi
    fi
    log SUCCESS "NumPy installed successfully"
  fi

  # Install development dependencies and set up pre-commit if --dev flag is present
  if [ "$INSTALL_DEV_EXTRAS" -eq 1 ]; then
    setup_development_environment
  fi
  
  # Run tests if requested
  if [ "$RUN_TESTS" -eq 1 ]; then
    run_tests
  fi
  
  log SUCCESS "Environment setup completed successfully!"
  log INFO "To activate the Conda environment, run: conda activate \"$PWD/${LOCAL_ENV_DIR}\""
}

# --- Development environment setup ---
setup_development_environment() {
  section "Setting up development environment"
  
  # Install development environment if the file exists
  if [ -f "$DEV_ENV_FILE" ]; then
    log INFO "Installing development environment from '$DEV_ENV_FILE'"
    run_command_verbose conda env update --prefix "./$LOCAL_ENV_DIR" -f "$DEV_ENV_FILE" --prune
    log SUCCESS "Development environment from '$DEV_ENV_FILE' updated successfully."
  else
    log WARNING "Development mode enabled, but '$DEV_ENV_FILE' not found. Using base environment."
  fi

  # Setup pre-commit hooks
  setup_pre_commit
}

# --- Pre-commit setup ---
setup_pre_commit() {
  # Check if pre-commit is installed in the target environment
  if ! conda run --prefix "./${LOCAL_ENV_DIR}" pre-commit --version >/dev/null 2>&1; then
    log INFO "Installing pre-commit into ${LOCAL_ENV_DIR}..."
    if ! conda run --prefix "./${LOCAL_ENV_DIR}" conda install -y -c conda-forge pre-commit; then
      log WARNING "conda install failed, attempting pip fallback"
      if ! conda run --prefix "./${LOCAL_ENV_DIR}" pip install pre-commit; then
        log WARNING "Failed to install pre-commit with both conda and pip"
        return 1
      fi
    fi
  fi

  # Generate pre-commit config from template if template exists
  if [ -f "${PRE_COMMIT_TEMPLATE}" ]; then
    generate_pre_commit_config
  fi
  
  # Set up the hooks if config exists
  if [ -f "$PRE_COMMIT_CONFIG" ]; then
    log INFO "Setting up pre-commit hooks..."
    if run_command_verbose conda run --prefix "./$LOCAL_ENV_DIR" pre-commit install --install-hooks; then
      log SUCCESS "Pre-commit hooks set up successfully."
    else
      log WARNING "Failed to set up pre-commit hooks, but pre-commit command was found."
      return 1
    fi
  else
    log WARNING "No pre-commit configuration found. Skipping pre-commit setup."
  fi
}

# --- Generate pre-commit config from template ---
generate_pre_commit_config() {
  local env_path="${PWD}/${LOCAL_ENV_DIR}"
  local conda_prefix
  
  # Safely get conda prefix
  if ! conda_prefix="$(conda info --base 2>/dev/null)"; then
    conda_prefix="${CONDA_PREFIX:-/opt/conda}"  # Fallback default
    log WARNING "Could not determine conda base directory, using fallback: ${conda_prefix}"
  fi
  
  log INFO "Generating pre-commit configuration from template..."
  
  # Create backup if file exists
  if [ -f "${PRE_COMMIT_CONFIG}" ]; then
    mv "${PRE_COMMIT_CONFIG}" "${PRE_COMMIT_CONFIG}.bak"
    log INFO "Backed up existing ${PRE_COMMIT_CONFIG} to ${PRE_COMMIT_CONFIG}.bak"
  fi
  
  # Process template
  if ! sed -e "s|{{ENV_PATH}}|${env_path}|g" \
           -e "s|{{CONDA_PREFIX}}|${conda_prefix}|g" \
           "${PRE_COMMIT_TEMPLATE}" > "${PRE_COMMIT_CONFIG}"; then
    # Restore backup if processing failed
    if [ -f "${PRE_COMMIT_CONFIG}.bak" ]; then
      mv "${PRE_COMMIT_CONFIG}.bak" "${PRE_COMMIT_CONFIG}"
    fi
    log ERROR "Failed to generate ${PRE_COMMIT_CONFIG} from template"
    return 1
  fi
  
  # Remove backup if successful
  if [ -f "${PRE_COMMIT_CONFIG}.bak" ]; then
    rm -f "${PRE_COMMIT_CONFIG}.bak"
  fi
  
  log SUCCESS "Generated ${PRE_COMMIT_CONFIG} from template"
}

# --- Run tests ---
run_tests() {
  section "Running tests"
  
  # Get the absolute path to the project root
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  
  # Check if pytest is available
  if conda run --prefix "./${LOCAL_ENV_DIR}" python -m pytest --version >/dev/null 2>&1; then
    log INFO "Running tests with pytest..."
    if conda run --prefix "./${LOCAL_ENV_DIR}" python -m pytest -v tests/; then
      log SUCCESS "All tests passed!"
    else
      log WARNING "Some tests failed. Continuing with setup..."
    fi
  else
    log WARNING "pytest not found in the environment. Skipping tests."
  fi
}

# --- Main execution ---
main() {
  # Create a trap to handle early exits
  trap 'log ERROR "Setup was interrupted. Cleaning up..."' INT TERM
  
  # Run the main setup
  setup_environment
  
  # Clear the trap on successful completion
  trap - INT TERM
  
  exit 0
}

# --- Main execution ---
if [ "$SOURCED" -eq 1 ] && [ "${1:-}" == "--run-setup" ]; then
    # Remove --run-setup from arguments
    shift
    debug "Running setup in sourced mode with args: $*"
    main "$@"
elif [ "$SOURCED" -eq 0 ]; then
    # Run normally when executed directly
    debug "Running in direct execution mode with args: $*"
    main "$@"
fi

# Source the paths script if it exists
setup_paths() {
    if [ -f "$PATHS_SCRIPT" ]; then
        log INFO "Setting up paths using $PATHS_SCRIPT..."
        # Source the script to set up paths in the current shell
        if ! source "$PATHS_SCRIPT"; then
            log ERROR "Failed to set up paths using $PATHS_SCRIPT"
            return 1
        fi
        # Run the setup_paths function if it exists
        if command -v setup_paths >/dev/null 2>&1; then
            setup_paths
        fi
    else
        log WARNING "Paths script not found: $PATHS_SCRIPT"
        return 1
    fi
}

# Set up paths at the end of the environment setup
if [ "$SOURCED" -eq 1 ]; then
    # Only set up paths if not in direct execution mode (already handled in main)
    if [ "${1:-}" != "--run-setup" ]; then
        if command -v setup_paths >/dev/null 2>&1; then
            setup_paths
        else
            debug "setup_paths function not found"
        fi
    fi
else
    setup_paths
fi

# Print success message and usage instructions
if [ "$SOURCED" -eq 1 ]; then
    log INFO "Environment sourced successfully"
    log INFO "To run the full setup, use: source $0 --run-setup [options]"
fi
log SUCCESS "Environment setup complete!"
echo
log INFO "To use this environment:"
echo
echo "For interactive shell usage:"
echo "  conda activate $PWD/$LOCAL_ENV_DIR"
echo
echo "For scripts and non-interactive usage (recommended for batch jobs/CI):"
echo "  conda run -p $PWD/$LOCAL_ENV_DIR your_script.py"
echo
echo "To run tests:"
echo "  conda run -p $PWD/$LOCAL_ENV_DIR pytest tests/"
echo
log INFO "See README.md for more details on using the environment."
