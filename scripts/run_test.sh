#!/bin/bash
# run_test.sh - Simple test of config duration
# Relocated from repository root to scripts/ directory with enhanced verbose logging support

# Initialize verbose flag
VERBOSE=0

# Parse command line arguments for verbose flag
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        *)
            # Unknown option, keep it for potential future use
            shift
            ;;
    esac
done

# Verbose logging function
log_verbose() {
    [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_test.sh: $1"
}

# Create logs directory if it doesn't exist
if [[ $VERBOSE -eq 1 ]]; then
    if [[ ! -d "logs" ]]; then
        mkdir -p logs 2>/dev/null || true
    fi
fi

log_verbose "Starting config-based duration test"
echo "Testing config-based duration..."

# Determine project directory relative to scripts/ location
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
log_verbose "Project directory resolved to: $PROJECT_DIR"

log_verbose "Creating temporary MATLAB test file"
# Quick MATLAB test
TEMP=$(mktemp /tmp/test_XXXXXX.m)
log_verbose "Temporary test file created: $TEMP"

cat > "$TEMP" << EOF
% CRITICAL: Change to project directory first!
cd('$PROJECT_DIR');
addpath(genpath('Code'));

fprintf('\n1. Config check: ');
try
    [~,pc] = get_plume_file();
    if isfield(pc,'simulation') && isfield(pc.simulation, 'duration_seconds')
        fprintf('%.0f seconds\n', pc.simulation.duration_seconds);
    else
        fprintf('not set\n');
    end
catch ME
    fprintf('Error: %s\n', ME.message);
end

fprintf('\n2. Test run: ');
try
    out = navigation_model_vec('config', 'gaussian', 0, 2);
    fprintf('SUCCESS! %d samples\n', size(out.x,1));
catch
    try
        out = navigation_model_vec(0, 'gaussian', 0, 2);
        fprintf('SUCCESS with 0! %d samples\n', size(out.x,1));
    catch ME
        fprintf('FAILED: %s\n', ME.message);
    end
end
exit;
EOF

log_verbose "Executing MATLAB test in headless mode"
matlab -nodisplay -nosplash -nojvm -r "run('$TEMP')" 2>&1 | grep -v ">>"

log_verbose "Cleaning up temporary test file"
rm "$TEMP"
log_verbose "Test execution completed successfully"