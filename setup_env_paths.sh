#!/bin/bash
# setup_env_paths.sh - Capture and store correct paths for HPC environment
#
# Run this ONCE after cloning/setting up the repository to store the correct
# symlink paths that all other scripts will reference.

set -euo pipefail

echo "=== AlvarezSalvado Environment Path Setup ==="
echo ""

# Get the script directory (preserving symlinks)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -L )"

# Capture key paths using pwd -L to preserve symlinks
PROJECT_ROOT="$SCRIPT_DIR"
CODE_DIR="${PROJECT_ROOT}/Code"
DATA_DIR="${PROJECT_ROOT}/data"
CONFIG_DIR="${PROJECT_ROOT}/configs"
PLUME_FILE="${DATA_DIR}/plumes/10302017_10cms_bounded.hdf5"
PLUME_CONFIG="${CONFIG_DIR}/plumes/crimaldi_10cms_bounded.json"

echo "Detected paths:"
echo "  Project root: $PROJECT_ROOT"
echo "  Code directory: $CODE_DIR"
echo "  Data directory: $DATA_DIR"
echo "  Config directory: $CONFIG_DIR"
echo ""

# Create paths config file
PATHS_CONFIG="${CONFIG_DIR}/paths.json"

echo "Creating paths configuration at: $PATHS_CONFIG"

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

echo "✓ Paths configuration created"

# Create MATLAB path loader function
echo ""
echo "Creating MATLAB path loader..."

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

echo "✓ load_paths.m created"

# Create updated get_plume_file.m that uses the paths config
echo ""
echo "Creating paths-aware get_plume_file.m..."

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
if [ -f "${CODE_DIR}/get_plume_file.m" ]; then
    cp "${CODE_DIR}/get_plume_file.m" "${CODE_DIR}/get_plume_file.m.backup_$(date +%Y%m%d_%H%M%S)"
fi
cp "${CODE_DIR}/get_plume_file_paths.m" "${CODE_DIR}/get_plume_file.m"

echo "✓ get_plume_file.m updated to use paths config"

# Create startup.m that loads paths automatically
echo ""
echo "Creating startup.m..."

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

echo "✓ startup.m created"

# Create test script
echo ""
echo "Creating test script..."

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

echo "✓ test_paths_config.m created"

# Create SLURM template that uses paths config
echo ""
echo "Creating SLURM template..."

cat > "${PROJECT_ROOT}/nav_job_paths.slurm" << EOF
#!/bin/bash
#SBATCH --job-name=nav_model
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-99%20
#SBATCH --output=logs/nav-%A_%a.out
#SBATCH --error=logs/nav-%A_%a.err

# Load the project paths using proper JSON parsing
if command -v jq >/dev/null 2>&1; then
    # Use jq if available (preferred method)
    if [[ -f "${PROJECT_ROOT}/configs/paths.json" ]]; then
        log_message "INFO" "Loading paths from ${PROJECT_ROOT}/configs/paths.json using jq"
        export PATHS_PROJECT_ROOT=$(jq -r '.project_root' "${PROJECT_ROOT}/configs/paths.json" 2>/dev/null)
        export PATHS_CODE_DIR=$(jq -r '.code_dir' "${PROJECT_ROOT}/configs/paths.json" 2>/dev/null)
        export PATHS_DATA_DIR=$(jq -r '.data_dir' "${PROJECT_ROOT}/configs/paths.json" 2>/dev/null)
        export PATHS_CONFIG_DIR=$(jq -r '.config_dir' "${PROJECT_ROOT}/configs/paths.json" 2>/dev/null)
        export PATHS_PLUME_FILE=$(jq -r '.plume_file' "${PROJECT_ROOT}/configs/paths.json" 2>/dev/null)
        export PATHS_PLUME_CONFIG=$(jq -r '.plume_config' "${PROJECT_ROOT}/configs/paths.json" 2>/dev/null)
    else
        log_message "ERROR" "Could not find paths config: ${PROJECT_ROOT}/configs/paths.json"
    fi
else
    # Fallback if jq is not available - using grep and sed
    log_message "WARNING" "jq not found, using fallback JSON parsing method"
    if [[ -f "${PROJECT_ROOT}/configs/paths.json" ]]; then
        log_message "INFO" "Loading paths from ${PROJECT_ROOT}/configs/paths.json using grep/sed"
        export PATHS_PROJECT_ROOT=$(grep -o '"project_root":"[^"]*"' "${PROJECT_ROOT}/configs/paths.json" 2>/dev/null | sed 's/"project_root":"\(.*\)"/\1/')
        export PATHS_CODE_DIR=$(grep -o '"code_dir":"[^"]*"' "${PROJECT_ROOT}/configs/paths.json" 2>/dev/null | sed 's/"code_dir":"\(.*\)"/\1/')
        export PATHS_DATA_DIR=$(grep -o '"data_dir":"[^"]*"' "${PROJECT_ROOT}/configs/paths.json" 2>/dev/null | sed 's/"data_dir":"\(.*\)"/\1/')
        export PATHS_CONFIG_DIR=$(grep -o '"config_dir":"[^"]*"' "${PROJECT_ROOT}/configs/paths.json" 2>/dev/null | sed 's/"config_dir":"\(.*\)"/\1/')
        export PATHS_PLUME_FILE=$(grep -o '"plume_file":"[^"]*"' "${PROJECT_ROOT}/configs/paths.json" 2>/dev/null | sed 's/"plume_file":"\(.*\)"/\1/')
        export PATHS_PLUME_CONFIG=$(grep -o '"plume_config":"[^"]*"' "${PROJECT_ROOT}/configs/paths.json" 2>/dev/null | sed 's/"plume_config":"\(.*\)"/\1/')
    else
        log_message "ERROR" "Could not find paths config: ${PROJECT_ROOT}/configs/paths.json"
    fi
fi

# Log the parsed values for debugging
log_message "DEBUG" "PATHS_PROJECT_ROOT=${PATHS_PROJECT_ROOT}"
log_message "DEBUG" "PATHS_PLUME_CONFIG=${PATHS_PLUME_CONFIG}"

# Change to project directory
cd "$PROJECT_ROOT"

# Create directories
mkdir -p logs results

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

echo "✓ nav_job_paths.slurm created"

# Final summary
echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Paths have been captured and stored in:"
echo "  $PATHS_CONFIG"
echo ""
echo "All MATLAB scripts will now use these stored paths instead of"
echo "resolving symlinks."
echo ""
echo "To test the setup:"
echo "  matlab -nodisplay -r \"run('test_paths_config.m'); exit\""
echo ""
echo "To run simulations:"
echo "  sbatch nav_job_paths.slurm"
echo ""
echo "IMPORTANT: This setup is machine-specific. If you move to a different"
echo "node or the paths change, run this script again."