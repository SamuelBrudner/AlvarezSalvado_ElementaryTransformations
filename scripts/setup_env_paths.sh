#!/bin/bash
# setup_env_paths.sh - Capture and store correct paths for HPC environment
#
# Run this ONCE after cloning/setting up the repository to store the correct
# symlink paths that all other scripts will reference.
#
# Enhanced with verbose logging support for debugging and monitoring.
#
# Usage: setup_env_paths.sh [-v|--verbose]
#   -v, --verbose    Enable detailed trace output for debugging

set -euo pipefail

# Initialize verbose logging flag
VERBOSE=0
SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}")
LOG_DIR=""

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            -h|--help)
                echo "Usage: $SCRIPT_NAME [-v|--verbose]"
                echo ""
                echo "Environment path configuration script for setting up project-specific"
                echo "paths and variables. Captures and stores correct paths for HPC environment."
                echo ""
                echo "Options:"
                echo "  -v, --verbose    Enable detailed trace output for debugging"
                echo "  -h, --help       Show this help message"
                echo ""
                echo "This script should be run ONCE after cloning/setting up the repository"
                echo "to store the correct symlink paths that all other scripts will reference."
                exit 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                echo "Use -h or --help for usage information" >&2
                exit 1
                ;;
        esac
    done
}

# Verbose logging function
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        local message="[$timestamp] $SCRIPT_NAME: $*"
        echo "$message"
        
        # Also log to file if logs directory exists
        if [[ -n "$LOG_DIR" && -d "$LOG_DIR" ]]; then
            echo "$message" >> "$LOG_DIR/setup_env_paths.log"
        fi
    fi
}

# Standard output function (always displayed)
log_info() {
    echo "$*"
}

# Parse arguments
parse_args "$@"

log_verbose "Starting environment path setup with verbose logging enabled"
log_verbose "Script location: ${BASH_SOURCE[0]}"

log_info "=== AlvarezSalvado Environment Path Setup ==="
log_info ""

# Get the script directory (preserving symlinks)
# Note: This script is now in scripts/ subdirectory, so we need to go up one level for project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -L )"
log_verbose "Script directory: $SCRIPT_DIR"

# Calculate project root (one level up from scripts/)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
log_verbose "Calculated project root: $PROJECT_ROOT"

# Set up logging directory early
LOG_DIR="${PROJECT_ROOT}/logs"
if [[ ! -d "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
    log_verbose "Created logs directory: $LOG_DIR"
fi

# Capture key paths using pwd -L to preserve symlinks
CODE_DIR="${PROJECT_ROOT}/Code"
DATA_DIR="${PROJECT_ROOT}/data"
CONFIG_DIR="${PROJECT_ROOT}/configs"
PLUME_FILE="${DATA_DIR}/plumes/10302017_10cms_bounded.hdf5"
PLUME_CONFIG="${CONFIG_DIR}/plumes/crimaldi_10cms_bounded.json"

log_verbose "Calculating environment paths..."
log_verbose "  CODE_DIR: $CODE_DIR"
log_verbose "  DATA_DIR: $DATA_DIR"
log_verbose "  CONFIG_DIR: $CONFIG_DIR"
log_verbose "  PLUME_FILE: $PLUME_FILE"
log_verbose "  PLUME_CONFIG: $PLUME_CONFIG"

log_info "Detected paths:"
log_info "  Project root: $PROJECT_ROOT"
log_info "  Code directory: $CODE_DIR"
log_info "  Data directory: $DATA_DIR"
log_info "  Config directory: $CONFIG_DIR"
log_info ""

# Create paths config file
PATHS_CONFIG="${CONFIG_DIR}/paths.json"

log_verbose "Preparing to create paths configuration file"
log_info "Creating paths configuration at: $PATHS_CONFIG"

# Ensure configs directory exists
if [[ ! -d "$CONFIG_DIR" ]]; then
    log_verbose "Creating configs directory: $CONFIG_DIR"
    mkdir -p "$CONFIG_DIR"
fi

log_verbose "Writing paths configuration JSON..."
cat > "$PATHS_CONFIG" << EOF
{
  "project_root": "$PROJECT_ROOT",
  "code_dir": "$CODE_DIR",
  "data_dir": "$DATA_DIR",
  "config_dir": "$CONFIG_DIR",
  "plume_file": "$PLUME_FILE",
  "plume_config": "$PLUME_CONFIG",
  "paths_generated": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "generated_by": "$USER@$(hostname)",
  "comment": "Auto-generated paths preserving symlinks - DO NOT EDIT MANUALLY"
}
EOF

log_verbose "Paths configuration file created successfully"
log_info "✓ Paths configuration created"

# Create MATLAB path loader function
log_info ""
log_info "Creating MATLAB path loader..."
log_verbose "Preparing to create load_paths.m function"

# Ensure Code directory exists
if [[ ! -d "$CODE_DIR" ]]; then
    log_verbose "Creating Code directory: $CODE_DIR"
    mkdir -p "$CODE_DIR"
fi

log_verbose "Writing load_paths.m MATLAB function..."
cat > "${CODE_DIR}/load_paths.m" << 'EOF'
function paths = load_paths()
%LOAD_PATHS Load environment paths from configuration
%   This function loads the paths configuration created by setup_env_paths.sh
%   ensuring consistent path usage across all MATLAB scripts.

% Get the directory where this function lives
this_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(this_dir);

% Load paths config
paths_config_file = fullfile(project_root, 'configs', 'paths.json');

if ~exist(paths_config_file, 'file')
    error('Paths configuration not found. Run setup_env_paths.sh first!');
end

% Read and parse the JSON
try
    json_text = fileread(paths_config_file);
    paths = jsondecode(json_text);
catch ME
    error('Failed to load paths config: %s', ME.message);
end

% Set environment variables for compatibility
setenv('MATLAB_PROJECT_ROOT', paths.project_root);
setenv('MATLAB_PLUME_FILE', paths.plume_file);
setenv('PLUME_CONFIG', paths.plume_config);

% Add Code directory to path if not already there
if ~contains(path, paths.code_dir)
    addpath(genpath(paths.code_dir));
end

% Display loaded paths (optional)
if nargout == 0
    fprintf('Loaded paths from: %s\n', paths_config_file);
    fprintf('  Project root: %s\n', paths.project_root);
    fprintf('  Plume file: %s\n', paths.plume_file);
end

end
EOF

log_verbose "load_paths.m function created successfully"
log_info "✓ load_paths.m created"

# Create updated get_plume_file.m that uses the paths config
log_info ""
log_info "Creating paths-aware get_plume_file.m..."
log_verbose "Preparing to create enhanced get_plume_file.m function"

log_verbose "Writing get_plume_file_paths.m temporary file..."
cat > "${CODE_DIR}/get_plume_file_paths.m" << 'EOF'
function [plume_file, plume_config] = get_plume_file()
%GET_PLUME_FILE Return plume HDF5 filename from stored paths configuration

% Load stored paths
try
    paths = load_paths();
    plume_file = paths.plume_file;
    config_path = paths.plume_config;
catch
    % Fallback to environment variables
    plume_file = getenv('MATLAB_PLUME_FILE');
    config_path = getenv('PLUME_CONFIG');
    
    if isempty(plume_file) || isempty(config_path)
        error('No paths configuration found. Run setup_env_paths.sh first!');
    end
end

fprintf('Using plume file: %s\n', plume_file);

% Initialize default config
plume_config = struct();
plume_config.mm_per_pixel = 0.74;
plume_config.pixel_scale = 0.74;
plume_config.frame_rate = 15;
plume_config.time_scale_50hz = 15/50;
plume_config.time_scale_15hz = 1.0;
plume_config.plume_xlims = [1, 216];
plume_config.plume_ylims = [1, 406];
plume_config.dataset_name = '/dataset2';

% Load config file
if exist(config_path, 'file')
    try
        cfg = jsondecode(fileread(config_path));
        
        % Update config from file
        if isfield(cfg, 'spatial')
            if isfield(cfg.spatial, 'mm_per_pixel')
                plume_config.mm_per_pixel = cfg.spatial.mm_per_pixel;
                plume_config.pixel_scale = cfg.spatial.mm_per_pixel;
            end
            if isfield(cfg.spatial, 'resolution')
                plume_config.plume_xlims = [1, cfg.spatial.resolution.width];
                plume_config.plume_ylims = [1, cfg.spatial.resolution.height];
            end
        end
        
        if isfield(cfg, 'temporal')
            if isfield(cfg.temporal, 'frame_rate')
                plume_config.frame_rate = cfg.temporal.frame_rate;
                plume_config.time_scale_50hz = cfg.temporal.frame_rate / 50.0;
                plume_config.time_scale_15hz = cfg.temporal.frame_rate / 15.0;
            end
        end
        
        if isfield(cfg, 'data_path') && isfield(cfg.data_path, 'dataset_name')
            plume_config.dataset_name = cfg.data_path.dataset_name;
        end
        
        if isfield(cfg, 'simulation')
            if isfield(cfg.simulation, 'duration_seconds')
                plume_config.simulation.duration_seconds = cfg.simulation.duration_seconds;
            end
        end
        
    catch err
        warning('Could not parse config file: %s', err.message);
    end
else
    warning('Config file not found: %s', config_path);
end

if nargout < 2
    clear plume_config;
end

end
EOF

# Backup and replace get_plume_file.m
log_verbose "Backing up existing get_plume_file.m if it exists"
if [ -f "${CODE_DIR}/get_plume_file.m" ]; then
    backup_file="${CODE_DIR}/get_plume_file.m.backup_$(date +%Y%m%d_%H%M%S)"
    cp "${CODE_DIR}/get_plume_file.m" "$backup_file"
    log_verbose "Existing get_plume_file.m backed up to: $backup_file"
fi

log_verbose "Replacing get_plume_file.m with paths-aware version"
cp "${CODE_DIR}/get_plume_file_paths.m" "${CODE_DIR}/get_plume_file.m"
log_verbose "get_plume_file.m successfully updated"

log_info "✓ get_plume_file.m updated to use paths config"

# Create startup.m that loads paths automatically
log_info ""
log_info "Creating startup.m..."
log_verbose "Preparing to create MATLAB startup.m file"

log_verbose "Writing startup.m to project root..."
cat > "${PROJECT_ROOT}/startup.m" << 'EOF'
% startup.m - Auto-load paths configuration
fprintf('Loading AlvarezSalvado environment paths...\n');

try
    % Load the stored paths
    paths = load_paths();
    
    % Change to project root
    cd(paths.project_root);
    
    fprintf('Environment ready:\n');
    fprintf('  Working directory: %s\n', pwd);
    fprintf('  Plume file: %s\n', paths.plume_file);
catch ME
    warning('Failed to load paths: %s', ME.message);
    warning('Run setup_env_paths.sh to configure paths');
end
EOF

log_verbose "startup.m created successfully"
log_info "✓ startup.m created"

# Create test script
log_info ""
log_info "Creating test script..."
log_verbose "Preparing to create test_paths_config.m test script"

log_verbose "Writing test_paths_config.m to project root..."
cat > "${PROJECT_ROOT}/test_paths_config.m" << 'EOF'
% test_paths_config.m - Test the paths configuration

fprintf('\n=== Testing Paths Configuration ===\n\n');

% Test 1: Load paths
fprintf('1. Loading paths config:\n');
try
    paths = load_paths();
    fprintf('   ✓ Paths loaded successfully\n');
    fprintf('   Project root: %s\n', paths.project_root);
    fprintf('   Has /vast/palmer: %s\n\n', iif(contains(paths.project_root, '/vast/palmer'), 'NO (bad)', 'YES (good)'));
catch ME
    fprintf('   ✗ Failed: %s\n\n', ME.message);
    return;
end

% Test 2: Check files exist
fprintf('2. Checking files:\n');
fprintf('   Plume file exists: %s\n', iif(exist(paths.plume_file, 'file'), 'YES', 'NO'));
fprintf('   Config exists: %s\n\n', iif(exist(paths.plume_config, 'file'), 'YES', 'NO'));

% Test 3: Get plume file
fprintf('3. Testing get_plume_file:\n');
try
    [pf, pc] = get_plume_file();
    fprintf('   ✓ Function works\n');
    fprintf('   Path: %s\n', pf);
    fprintf('   Path correct: %s\n\n', iif(strcmp(pf, paths.plume_file), 'YES', 'NO'));
catch ME
    fprintf('   ✗ Failed: %s\n\n', ME.message);
end

% Test 4: Quick simulation
fprintf('4. Testing simulation:\n');
try
    out = navigation_model_vec(100, 'Crimaldi', 0, 1);
    fprintf('   ✓ Simulation successful! %d samples\n', size(out.x, 1));
catch ME
    fprintf('   ✗ Failed: %s\n', ME.message);
end

fprintf('\n=== Test Complete ===\n');

function r = iif(c,t,f)
    if c, r=t; else, r=f; end
end
EOF

log_verbose "test_paths_config.m created successfully"
log_info "✓ test_paths_config.m created"

# Create SLURM template that uses paths config
log_info ""
log_info "Creating SLURM template..."
log_verbose "Preparing to create SLURM job template with paths configuration"

# Updated to reference slurm/ directory as per the Summary of Changes
SLURM_TEMPLATE="${PROJECT_ROOT}/slurm/nav_job_paths.slurm"
log_verbose "SLURM template will be created at: $SLURM_TEMPLATE"

# Ensure slurm directory exists
if [[ ! -d "${PROJECT_ROOT}/slurm" ]]; then
    log_verbose "Creating slurm directory: ${PROJECT_ROOT}/slurm"
    mkdir -p "${PROJECT_ROOT}/slurm"
fi

log_verbose "Writing SLURM template..."
cat > "$SLURM_TEMPLATE" << EOF
#!/bin/bash
#SBATCH --job-name=nav_model
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-99%20
#SBATCH --output=slurm_logs/nav_model/nav-%A_%a.out
#SBATCH --error=slurm_logs/nav_model/nav-%A_%a.err

# Load the project paths
source "${PROJECT_ROOT}/configs/paths.json"

# Change to project directory
cd "$PROJECT_ROOT"

# Create directories
mkdir -p slurm_logs/nav_model results

# Load MATLAB
module load MATLAB/2023b

# Run simulation
matlab -nodisplay -nosplash -r "
% Load paths configuration
paths = load_paths();
fprintf('Using paths from: %s\\n', paths.project_root);

% Get task ID
task_id = str2double(getenv('SLURM_ARRAY_TASK_ID'));
if isnan(task_id), task_id = 0; end

fprintf('Task %d: Starting simulation\\n', task_id);

try
    % Run simulation
    out = navigation_model_vec('config', 'Crimaldi', 0, 10);
    
    % Save results
    filename = sprintf('results/nav_results_%04d.mat', task_id);
    save(filename, 'out', '-v7.3');
    fprintf('Task %d: Success! Saved to %s\\n', task_id, filename);
catch ME
    fprintf('Task %d ERROR: %s\\n', task_id, ME.message);
    exit(1);
end
exit(0);
"
EOF

log_verbose "SLURM template created successfully"
log_info "✓ nav_job_paths.slurm created"

# Final summary
log_info ""
log_info "=== Setup Complete! ==="
log_info ""
log_info "Paths have been captured and stored in:"
log_info "  $PATHS_CONFIG"
log_info ""
log_info "All MATLAB scripts will now use these stored paths instead of"
log_info "resolving symlinks."
log_info ""
log_info "To test the setup:"
log_info "  matlab -nodisplay -r \"run('test_paths_config.m'); exit\""
log_info ""
log_info "To run simulations:"
log_info "  sbatch slurm/nav_job_paths.slurm"
log_info ""
log_info "IMPORTANT: This setup is machine-specific. If you move to a different"
log_info "node or the paths change, run this script again."

# Final verbose logging summary
if [[ $VERBOSE -eq 1 ]]; then
    log_verbose "=== Verbose Logging Summary ==="
    log_verbose "Script completed successfully with all components created:"
    log_verbose "  - Paths configuration: $PATHS_CONFIG"
    log_verbose "  - MATLAB loader: ${CODE_DIR}/load_paths.m"
    log_verbose "  - Enhanced get_plume_file: ${CODE_DIR}/get_plume_file.m"
    log_verbose "  - Startup script: ${PROJECT_ROOT}/startup.m"
    log_verbose "  - Test script: ${PROJECT_ROOT}/test_paths_config.m"
    log_verbose "  - SLURM template: $SLURM_TEMPLATE"
    log_verbose "  - Log output: $LOG_DIR/setup_env_paths.log"
    log_verbose "Total execution time: $(( $(date +%s) - ${start_time:-$(date +%s)} )) seconds"
fi