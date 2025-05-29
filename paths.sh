#!/bin/bash
# Set up environment paths for the project
# Can be sourced directly or called as a script

# Handle both sourced and direct execution
(return 0 2>/dev/null) && SOURCED=1 || SOURCED=0

# Determine if the script is sourced in a non-interactive context
NONINTERACTIVE=0
if [ "$SOURCED" -eq 1 ]; then
    if [ -n "${PS4:-}" ] || [ -z "${PS1:-}" ]; then
        NONINTERACTIVE=1
    fi
fi

# Only set these options when not sourced to prevent exiting the parent shell
if [ "$SOURCED" -eq 0 ]; then
    set -euo pipefail
else
    set -u  # Only fail on undefined variables when sourced
fi
MATLAB_VERSION=${MATLAB_VERSION:-2023b}
MATLAB_MODULE=${MATLAB_MODULE:-MATLAB/$MATLAB_VERSION}


# Function to safely exit or return
safe_exit() {
    local exit_code=$1
    shift
    [ "$SOURCED" -eq 1 ] && return $exit_code || exit $exit_code
}

# Function to find MATLAB executable
find_matlab() {
    # First check if MATLAB is already in PATH
    if command -v matlab >/dev/null 2>&1; then
        command -v matlab
        return 0
    fi
    
    # Try module system if available before searching common paths
    if command -v module >/dev/null 2>&1; then
        module load "$MATLAB_MODULE" >/dev/null 2>&1 || module load MATLAB >/dev/null 2>&1 || true
        if command -v matlab >/dev/null 2>&1; then
            command -v matlab
            return 0
        fi
    fi

    # Try common MATLAB executable locations
    local matlab_paths=(
        "/usr/local/MATLAB/*/bin/matlab"
        "/opt/matlab/*/bin/matlab"
        "/Applications/MATLAB_*/bin/matlab"
        "/usr/local/bin/matlab"
        "/usr/bin/matlab"
    )
    
    for path_pattern in "${matlab_paths[@]}"; do
        # Expand the glob pattern
        for path in $path_pattern; do
            if [ -x "$path" ]; then
                echo "$path"
                return 0
            fi
        done
    done

    return 1
}

# Function to setup MATLAB environment
setup_matlab() {
    # If MATLAB_EXEC is already set, use it
    if [ -n "${MATLAB_EXEC:-}" ] && [ -x "$MATLAB_EXEC" ]; then
        log INFO "Using MATLAB from MATLAB_EXEC: $MATLAB_EXEC"
        return 0
    fi
    
    # Try to find MATLAB
    local matlab_path
    if matlab_path=$(find_matlab); then
        export MATLAB_EXEC="$matlab_path"
        # Add MATLAB's bin directory to PATH if not already there
        local matlab_bin="$(dirname "$matlab_path")"
        if [[ ":$PATH:" != *":$matlab_bin:"* ]]; then
            export PATH="$matlab_bin:$PATH"
            log INFO "Added MATLAB to PATH: $matlab_bin"
        fi
        log INFO "Found MATLAB at: $matlab_path"
        return 0
    else
        log WARNING "MATLAB not found. Some functionality may be limited."
        log WARNING "You can set MATLAB_EXEC environment variable to the MATLAB executable path."
        return 1
    fi
}

# Function to make paths relative to project root when possible
make_relative_path() {
    local target_path="$1"
    local project_root="$2"
    
    # If the path is already relative, return as is
    if [[ "$target_path" != /* ]]; then
        echo "$target_path"
        return 0
    fi
    
    # Try to make path relative to project root
    if [[ "$target_path" == "$project_root"/* ]]; then
        echo "${target_path#$project_root/}"
    elif [[ "$target_path" == "$project_root" ]]; then
        echo "."
    else
        echo "$target_path"
    fi
}

# Debug info
if [ "${DEBUG:-0}" -eq 1 ]; then
    echo "[DEBUG] Starting paths.sh"
    echo "[DEBUG] Script directory: $(pwd)"
    echo "[DEBUG] BASH_SOURCE[0]: ${BASH_SOURCE[0]:-not set}"
    echo "[DEBUG] SOURCED: $SOURCED"
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-}")" && pwd)"

# Source utility functions - required
if [ ! -f "${SCRIPT_DIR}/setup_utils.sh" ]; then
    echo "Error: setup_utils.sh not found in ${SCRIPT_DIR}/" >&2
    echo "Please ensure you're running this from the project root directory" >&2
    safe_exit 1
fi

# shellcheck source=./setup_utils.sh
if ! source "${SCRIPT_DIR}/setup_utils.sh"; then
    echo "Error: Failed to source setup_utils.sh" >&2
    safe_exit 1
fi

# Silence log output if sourced non-interactively unless DEBUG is enabled
if [ "$NONINTERACTIVE" -eq 1 ] && [ "${DEBUG:-0}" -ne 1 ]; then
    log() { :; }
fi

# Configuration
PATHS_TEMPLATE="${SCRIPT_DIR}/configs/project_paths.yaml.template"
PATHS_CONFIG="${SCRIPT_DIR}/configs/project_paths.yaml"

generate_paths_config() {
    log INFO "Setting up paths configuration..."
    
    # Create configs directory if it doesn't exist
    mkdir -p "$(dirname "$PATHS_CONFIG")" || {
        log ERROR "Failed to create configs directory"
        return 1
    }
    
    # Export PROJECT_DIR for the template
    export PROJECT_DIR="$SCRIPT_DIR"
    export TMPDIR="${TMPDIR:-/tmp}"
    
    # Generate project_paths.yaml from template
    if [ ! -f "$PATHS_TEMPLATE" ]; then
        log ERROR "Template file not found: $PATHS_TEMPLATE"
        return 1
    fi
    
    # Always generate the config file to ensure it's up to date
    log INFO "Generating $PATHS_CONFIG from template..."
    
    # Create a temporary file for the processed template
    local temp_template
    temp_template="${TMPDIR}/paths_template_${RANDOM}.yaml"
    
    # Process the template with environment variables
    if command -v envsubst >/dev/null 2>&1; then
        log INFO "Using envsubst for variable substitution"
        if ! envsubst < "$PATHS_TEMPLATE" > "$temp_template"; then
            log ERROR "Failed to process template with envsubst"
            rm -f "$temp_template" 2>/dev/null || true
            return 1
        fi
    elif command -v python3 >/dev/null 2>&1; then
        log INFO "Using Python for variable substitution"
        if ! python3 -c "
import os
import sys

def substitute_vars(template_path, output_path):
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Simple variable substitution for ${VAR} and $VAR
    for var_name in os.environ:
        placeholder = '$' + var_name
        content = content.replace(placeholder, os.environ[var_name])
        placeholder = '${' + var_name + '}'
        content = content.replace(placeholder, os.environ[var_name])
    
    with open(output_path, 'w') as f:
        f.write(content)

substitute_vars('$PATHS_TEMPLATE', '$temp_template')
        "; then
            log ERROR "Failed to process template with Python"
            rm -f "$temp_template" 2>/dev/null || true
            return 1
        fi
    else
        log ERROR "Neither envsubst nor Python is available for variable substitution"
        return 1
    fi
    
    # Process the template to make paths relative where possible using helper script
    if command -v python3 >/dev/null 2>&1; then
        log INFO "Processing paths to make them relative where possible..."
        if ! python3 "$SCRIPT_DIR/scripts/make_paths_relative.py" "$temp_template" "$PROJECT_DIR"; then
            log WARNING "Failed to process paths to be relative, using absolute paths"
        fi
    fi
    
    # Move the temporary file to the final location
    if ! mv "$temp_template" "$PATHS_CONFIG"; then
        log ERROR "Failed to create $PATHS_CONFIG"
        rm -f "$temp_template" 2>/dev/null || true
        return 1
    fi
    
    log SUCCESS "Created/updated paths configuration at $PATHS_CONFIG"
    
    # Set up additional environment paths
    export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}${SCRIPT_DIR}/Code"

    # Load MATLAB configuration from project_paths.yaml if it exists
    if [ -f "$PATHS_CONFIG" ] && command -v yq >/dev/null 2>&1; then
        # Check if MATLAB path is explicitly set in config
        if CONFIG_MATLAB=$(yq eval '.matlab.executable // ""' "$PATHS_CONFIG" 2>/dev/null) && 
           [ -n "$CONFIG_MATLAB" ] && [ -x "$CONFIG_MATLAB" ]; then
            export MATLAB_EXEC="$CONFIG_MATLAB"
            log INFO "Using MATLAB from project_paths.yaml: $MATLAB_EXEC"
        else
            # Auto-detect MATLAB if not in config
            if MATLAB_PATH=$(find_matlab); then
                export MATLAB_EXEC="$MATLAB_PATH"
                log INFO "Auto-detected MATLAB at: $MATLAB_EXEC"
                
                # Update the config with the found MATLAB path if yq is available
                if command -v yq >/dev/null 2>&1; then
                    yq eval ".matlab.executable = \"$MATLAB_EXEC\"" -i "$PATHS_CONFIG" 2>/dev/null && \
                    log INFO "Updated MATLAB path in $PATHS_CONFIG"
                fi
            else
                log WARNING "MATLAB not found. Some functionality may be limited."
                export MATLAB_EXEC=""
            fi
        fi
        
        # Export MATLAB paths from config
        if [ -f "$PATHS_CONFIG" ] && command -v yq >/dev/null 2>&1; then
            IFS=$'\n' read -r -d '' -a matlab_paths < <(
                yq eval '.matlab.paths[]' "$PATHS_CONFIG" 2>/dev/null | 
                sed 's/^/\n    /' | 
                tr -d '\n' | 
                sed 's/^\n//'
            )
            if [ ${#matlab_paths[@]} -gt 0 ]; then
                export MATLABPATH="${matlab_paths[*]}"
                log INFO "MATLAB paths set from config"
            fi
        fi
    else
        log WARNING "project_paths.yaml not found or yq not installed. Using default MATLAB configuration."
    fi

    log SUCCESS "Environment paths have been set up"
    log INFO "Project root: $PROJECT_DIR"
    return 0
}

setup_paths() {
    # First set up the basic paths configuration
    if ! generate_paths_config; then
        log ERROR "Failed to set up paths configuration"
        return 1
    fi
    
    # Set up MATLAB if available
    if ! setup_matlab; then
        log WARNING "MATLAB setup failed - some functionality may be limited"
        # Don't fail the entire setup if MATLAB isn't available
    fi
    
    return 0
}

# Main execution
if [ "$SOURCED" -eq 1 ]; then
    # When sourced, run setup_paths and capture the return code
    if ! setup_paths; then
        echo "Warning: Failed to set up some paths. Some functionality may be limited." >&2
    fi
else
    # When executed directly, run setup_paths and exit with appropriate status
    if setup_paths; then
        echo "Successfully set up project paths"
        exit 0
    else
        echo "Failed to set up project paths" >&2
        exit 1
    fi
fi
