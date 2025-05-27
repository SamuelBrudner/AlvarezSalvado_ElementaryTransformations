#!/bin/bash
# Set up local conda prefix environment for the project
set -euo pipefail

# Exit immediately if a command fails, but allow for pipefail to work properly
set -o pipefail || true

# Load utility functions
source "$(dirname "$0")/setup_utils.sh"

# --- Configuration ---
# Readonly constants for better maintainability
readonly LOCAL_ENV_DIR="dev-env"
readonly BASE_ENV_FILE="environment.yml"
readonly DEV_ENV_FILE="dev-environment.yml"
readonly PRE_COMMIT_TEMPLATE=".pre-commit-config.yaml.template"
readonly PRE_COMMIT_CONFIG=".pre-commit-config.yaml"
readonly PATHS_TEMPLATE="configs/paths.yaml.template"
readonly PATHS_CONFIG="configs/paths.yaml"
# ---

# Setup usage function
usage() {
  log INFO "Usage: $0 [--dev] [--no-tests] [--help]"
  log INFO "  --dev        Install development dependencies and set up pre-commit hooks"
  log INFO "  --no-tests   Skip running tests after setup"
  log INFO "  --help       Show this help message"
  exit 0
}

# Parse command line arguments
INSTALL_DEV_EXTRAS=0
RUN_TESTS=1

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
    local USER_BIN="$(python -m site --user-base)/bin"
    if ! command -v conda-lock >/dev/null 2>&1 || ! conda-lock --version >/dev/null 2>&1; then
        log INFO "Installing conda-lock"
        if ! run_command_verbose conda install -y -n base -c conda-forge conda-lock; then
            log WARNING "conda install failed, attempting pip fallback"
            if ! run_command_verbose python -m pip install --user conda-lock; then
                error "Failed to install conda-lock with conda or pip"
            fi
        fi

        # Refresh the shell to update PATH
        if [ -f "${CONDA_BASE_DIR}/etc/profile.d/conda.sh" ]; then
            source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh" 
        fi

        # Add conda to PATH
        export PATH="${CONDA_BASE_DIR}/bin:${PATH}"

        # Always include user's pip bin directory
        append_path_if_missing "${USER_BIN}"
        if [ -x "${USER_BIN}/conda-lock" ]; then
            "${USER_BIN}/conda-lock" --version >/dev/null 2>&1 || true
        fi
        hash -r

        # Verify installation
        if ! command -v conda-lock >/dev/null 2>&1 || ! conda-lock --version >/dev/null 2>&1; then
            error "conda-lock installed but not on PATH. Add ${USER_BIN}/conda-lock to PATH."
        fi
    fi
}

# --- Main setup function ---
setup_environment() {
  section "Starting environment setup"
  
  # Check if conda is installed, attempt to load via modules if missing
  if ! command -v conda >/dev/null 2>&1; then
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

  # Create/update the local prefix environment
  section "Setting up Conda environment"
  log INFO "Creating/updating local Conda environment in './$LOCAL_ENV_DIR'"
  
  if [ -f "conda-lock.yml" ]; then
    if conda_supports_force; then
      run_command_verbose conda env create --prefix "./${LOCAL_ENV_DIR}" --file conda-lock.yml --force
    else
      log INFO "Old conda detected - removing existing environment"
      run_command_verbose conda env remove --prefix "./${LOCAL_ENV_DIR}" -y || true
      run_command_verbose conda env create --prefix "./${LOCAL_ENV_DIR}" --file conda-lock.yml
    fi
  else
    log WARNING "conda-lock.yml not found, falling back to direct environment creation"
    if conda_supports_force; then
      run_command_verbose conda env create --prefix "./${LOCAL_ENV_DIR}" -f "${BASE_ENV_FILE}" --force
    else
      log INFO "Old conda detected - removing existing environment"
      run_command_verbose conda env remove --prefix "./${LOCAL_ENV_DIR}" -y || true
      run_command_verbose conda env create --prefix "./${LOCAL_ENV_DIR}" -f "${BASE_ENV_FILE}"
    fi
  fi
  
  log SUCCESS "Base environment './${LOCAL_ENV_DIR}' created/updated successfully."

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
      log ERROR "Failed to install pre-commit. Add it to '${BASE_ENV_FILE}' or '${DEV_ENV_FILE}'."
      return 1
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

# Run the main function
main "$@"

# Simple variable substitution function
substitute_vars() {
    local content
    content=$(<"$1")
    
    # Replace ${VAR} or $VAR with their values
    while [[ $content =~ (\$\{([a-zA-Z_][a-zA-Z0-9_]*)(:[-=][^}]*)?\}|\$([a-zA-Z_][a-zA-Z0-9_]*)) ]]; do
        var_full=${BASH_REMATCH[0]}
        var_name=${BASH_REMATCH[2]:-${BASH_REMATCH[4]}}
        
        # Handle default values
        if [[ ${BASH_REMATCH[3]} =~ ^:- ]]; then
            default_value=${BASH_REMATCH[3]:2}
            var_value=${!var_name:-$default_value}
        else
            var_value=${!var_name}
        fi
        
        content=${content//"$var_full"/"$var_value"}
    done
    
    echo "$content"
}

# Setup paths configuration
setup_paths_config() {
    if [ -f "$PATHS_TEMPLATE" ] && [ ! -f "$PATHS_CONFIG" ]; then
        log INFO "Setting up paths configuration..."
        export PROJECT_DIR="$PWD"
        substitute_vars "$PATHS_TEMPLATE" > "$PATHS_CONFIG"
        log SUCCESS "Created paths configuration at $PATHS_CONFIG"
        log INFO "Please review and edit $PATHS_CONFIG if needed"
    fi
}
setup_paths_config

# Print success message and usage instructions
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
