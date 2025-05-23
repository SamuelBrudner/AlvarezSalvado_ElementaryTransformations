#!/bin/bash
# Set up local conda prefix environment for the project
set -euo pipefail

# Exit immediately if a command fails, but allow for pipefail to work properly
set -o pipefail || true

# Load utility functions
source "$(dirname "$0")/setup_utils.sh"

usage() {
  echo "Usage: $0 [--dev] [--no-tests] [--help]"
}

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
source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh"

# Check if base environment file exists

if [ ! -f "$BASE_ENV_FILE" ]; then
  error "Base environment file '$BASE_ENV_FILE' not found in the current directory."
fi

# Ensure conda-lock is installed and generate lock file
if ! command -v conda-lock >/dev/null 2>&1; then
  log INFO "Installing conda-lock"
  # Install conda-lock and ensure it's in the PATH
  run_command_verbose conda install -y -n base -c conda-forge conda-lock
  # Refresh the shell to update PATH
  if [ -f "${CONDA_BASE_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh"
  fi
  # Add conda to PATH if not already there
  export PATH="${CONDA_BASE_DIR}/bin:${PATH}"
  # Verify installation
  if ! command -v conda-lock >/dev/null 2>&1; then
    error "Failed to install conda-lock or make it available in PATH"
  fi
fi

# Get platform information safely
if ! PLATFORM="$(conda info --json 2>/dev/null | python -c 'import sys,json;print(json.load(sys.stdin).get("platform", ""))' 2>/dev/null)" || [ -z "$PLATFORM" ]; then
  # Fallback for older conda versions or if json parsing fails
  PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"
  log WARNING "Could not detect conda platform, using fallback: $PLATFORM"
fi

# Always regenerate lock file to ensure consistency
log INFO "Generating conda-lock.yml for $PLATFORM"
if [ -f "conda-lock.yml" ]; then
  rm -f "conda-lock.yml"
fi

# Get conda-lock version to handle different versions
CONDA_LOCK_VERSION=$(conda-lock --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "")

if [ -z "$CONDA_LOCK_VERSION" ]; then
  # If we can't determine version, try without --overwrite
  run_command_verbose conda-lock lock -f "${BASE_ENV_FILE}" -p "${PLATFORM}" --lockfile conda-lock.yml
else
  # For versions that support --overwrite
  run_command_verbose conda-lock lock -f "${BASE_ENV_FILE}" -p "${PLATFORM}" --lockfile conda-lock.yml --overwrite 2>/dev/null || \
    run_command_verbose conda-lock lock -f "${BASE_ENV_FILE}" -p "${PLATFORM}" --lockfile conda-lock.yml
fi

# Create/update the local prefix environment
# The environment will be located at "./$LOCAL_ENV_DIR"
section "Creating/updating local Conda environment in './$LOCAL_ENV_DIR'"
if [ -f "conda-lock.yml" ]; then
  run_command_verbose conda env create --prefix "./${LOCAL_ENV_DIR}" --file conda-lock.yml --force
else
  run_command_verbose conda env create --prefix "./${LOCAL_ENV_DIR}" -f "${BASE_ENV_FILE}" --force
fi
log SUCCESS "Base environment './${LOCAL_ENV_DIR}' created/updated successfully."

# Create/update the local prefix environment
# The environment will be located at "./$LOCAL_ENV_DIR"

# Install development dependencies and set up pre-commit if --dev flag is present
if [ "$INSTALL_DEV_EXTRAS" -eq 1 ]; then
  section "Processing development extras"

  # Install development environment if the file exists
  if [ -f "$DEV_ENV_FILE" ]; then
    section "Installing development environment from '$DEV_ENV_FILE'"
    run_command_verbose conda env update --prefix "./$LOCAL_ENV_DIR" -f "$DEV_ENV_FILE" --prune
    log SUCCESS "Development environment from '$DEV_ENV_FILE' updated successfully."
  else
    log WARNING "Development mode enabled, but '$DEV_ENV_FILE' not found. Using base environment."
  fi

  # All development dependencies are handled by dev-environment.yml

  # Install pre-commit and set up hooks
  # pre-commit should be listed as a dependency in your environment.yml or requirements-dev.txt
  # Check if pre-commit is installed in the target environment
  if ! conda run --prefix "./${LOCAL_ENV_DIR}" pre-commit --version >/dev/null 2>&1; then
    log INFO "Installing pre-commit into ${LOCAL_ENV_DIR}..."
    if ! conda run --prefix "./${LOCAL_ENV_DIR}" conda install -y -c conda-forge pre-commit; then
      error "Failed to install pre-commit. Add it to '${BASE_ENV_FILE}' or '${DEV_REQUIREMENTS_FILE}'."
    fi
  fi

  # Generate pre-commit config from template if template exists
  setup_pre_commit_config() {
    local env_path="${PWD}/${LOCAL_ENV_DIR}"
    local conda_prefix
    
    # Safely get conda prefix
    if ! conda_prefix="$(conda info --base 2>/dev/null)"; then
      conda_prefix="${CONDA_PREFIX:-/opt/conda}"  # Fallback default
      log WARNING "Could not determine conda base directory, using fallback: ${conda_prefix}"
    fi
    
    if [ -f "${PRE_COMMIT_TEMPLATE}" ]; then
      log INFO "Generating pre-commit configuration..."
      if ! sed -e "s|{{ENV_PATH}}|${env_path}|g" \
               -e "s|{{CONDA_PREFIX}}|${conda_prefix}|g" \
               "${PRE_COMMIT_TEMPLATE}" > "${PRE_COMMIT_CONFIG}"; then
        error "Failed to generate ${PRE_COMMIT_CONFIG} from template"
      fi
      log SUCCESS "Generated ${PRE_COMMIT_CONFIG} from template"
    fi
  }

  # Generate the config from template if it exists
  setup_pre_commit_config
  
  # Set up the hooks if config exists
  if [ -f "$PRE_COMMIT_CONFIG" ]; then
    log INFO "Setting up pre-commit hooks..."
    if run_command_verbose conda run --prefix "./$LOCAL_ENV_DIR" pre-commit install --install-hooks; then
      log SUCCESS "Pre-commit hooks set up successfully."
    else
      log WARNING "Failed to set up pre-commit hooks, but pre-commit command was found."
    fi
  fi
fi

if [ "$RUN_TESTS" -eq 1 ]; then
  section "Running tests"
  # Get the absolute path to the project root
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  
  # Activate the environment
  log INFO "Activating the environment..."
  eval "$(conda shell.bash hook)"
  conda activate "$PROJECT_ROOT/$LOCAL_ENV_DIR"
  
  # Run tests with the project root in PYTHONPATH
  log INFO "Running tests..."
  PYTHONPATH="$PROJECT_ROOT" pytest -v tests/ || \
    log WARNING "Some tests failed"
  
  # Deactivate the environment
  conda deactivate
fi

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
