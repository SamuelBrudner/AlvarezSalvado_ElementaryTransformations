#!/bin/bash
# Set up environment paths for the project
# Can be sourced directly or called as a script

# Exit on error and undefined variables
set -euo pipefail

# Function to find MATLAB executable
find_matlab() {
    # Try common MATLAB executable locations
    local matlab_paths=(
        "$(command -v matlab 2>/dev/null)"
        "/Applications/MATLAB_*/bin/matlab"
        "/usr/local/MATLAB/*/bin/matlab"
        "/opt/matlab/*/bin/matlab"
    )
    
    for path in "${matlab_paths[@]}"; do
        if [ -n "$path" ] && [ -x "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    
    # If not found, try to find it using which
    if command -v which >/dev/null 2>&1; then
        local which_matlab
        which_matlab=$(which matlab 2>/dev/null)
        if [ -n "$which_matlab" ] && [ -x "$which_matlab" ]; then
            echo "$which_matlab"
            return 0
        fi
    fi
    
    return 1
}

# Debug info
echo "Starting paths.sh"
echo "Script directory: $(pwd)"
echo "BASH_SOURCE[0]: ${BASH_SOURCE[0]:-not set}"

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-}")" && pwd)"

# Source utility functions - required
if [ ! -f "${SCRIPT_DIR}/setup_utils.sh" ]; then
    echo "Error: setup_utils.sh not found in ${SCRIPT_DIR}/" >&2
    echo "Please ensure you're running this from the project root directory" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck source=./setup_utils.sh
if ! source "${SCRIPT_DIR}/setup_utils.sh"; then
    echo "Error: Failed to source setup_utils.sh" >&2
    return 1 2>/dev/null || exit 1
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
    
    # Generate paths.yaml from template if it doesn't exist
    if [ ! -f "$PATHS_TEMPLATE" ]; then
        log ERROR "Template file not found: $PATHS_TEMPLATE"
        return 1
    fi
    
    if [ ! -f "$PATHS_CONFIG" ]; then
        log INFO "Generating $PATHS_CONFIG from template..."
        
        # Try using substitute_vars if available, otherwise use envsubst
        if command -v envsubst >/dev/null 2>&1; then
            log INFO "Using envsubst for variable substitution"
            if ! envsubst < "$PATHS_TEMPLATE" > "$PATHS_CONFIG"; then
                log ERROR "Failed to generate $PATHS_CONFIG using envsubst"
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

substitute_vars('$PATHS_TEMPLATE', '$PATHS_CONFIG')
            "; then
                log ERROR "Failed to generate $PATHS_CONFIG using Python"
                return 1
            fi
        else
            log ERROR "Neither envsubst nor Python is available for variable substitution"
            return 1
        fi
        log SUCCESS "Created paths configuration at $PATHS_CONFIG"
    else
        log INFO "Paths configuration already exists at $PATHS_CONFIG"
    fi
    
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
    if ! generate_paths_config; then
        log ERROR "Failed to set up paths configuration"
        return 1
    fi
}

# If script is executed directly, run setup_paths
if [[ "${BASH_SOURCE[0]:-}" == "${0}" ]]; then
    setup_paths
fi
