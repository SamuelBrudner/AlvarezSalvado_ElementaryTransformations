#!/bin/bash

# ---------------------------------------------------------------------------
# Test script for running a small number of navigation simulations locally
# 
# Usage:
#   ./test_run.sh [environment] [num_agents] [trial_length]
#
# Examples:
#   ./test_run.sh                    # Run with defaults
#   ./test_run.sh Crimaldi 5         # Run 5 agents in Crimaldi environment
#   ./test_run.sh gaussian 10 3000   # Run 10 agents in gaussian for 3000ms
# ---------------------------------------------------------------------------

set -euo pipefail

# Parse command line arguments with defaults
ENVIRONMENT="${1:-Crimaldi}"
NUM_AGENTS="${2:-5}"
TRIAL_LENGTH="${3:-5000}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Navigation Model Test Run ===${NC}"
echo "Environment: $ENVIRONMENT"
echo "Number of agents: $NUM_AGENTS"
echo "Trial length: $TRIAL_LENGTH"
echo "Working directory: $SCRIPT_DIR"
echo ""

# Check if MATLAB is available
if ! command -v matlab &> /dev/null; then
    echo -e "${RED}Error: MATLAB not found in PATH${NC}"
    exit 1
fi

# Create test output directory
TEST_DIR="$SCRIPT_DIR/test_output_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"
echo -e "${YELLOW}Output directory: $TEST_DIR${NC}"

# Create MATLAB script to run
cat > "$TEST_DIR/test_script.m" << 'EOF'
% Add paths
script_dir = getenv('SCRIPT_DIR');
test_dir = getenv('TEST_DIR');
environment = getenv('TEST_ENVIRONMENT');
num_agents = str2double(getenv('TEST_NUM_AGENTS'));
trial_length = str2double(getenv('TEST_TRIAL_LENGTH'));

addpath(genpath(fullfile(script_dir, 'Code')));

% Set up diary to capture output
diary(fullfile(test_dir, 'simulation_output.txt'));

fprintf('\n=== Test Run Started at %s ===\n', datestr(now));
fprintf('Environment: %s\n', environment);
fprintf('Number of agents: %d\n', num_agents);
fprintf('Trial length: %d\n\n', trial_length);

try
    % Run the simulation
    tic;
    out = navigation_model_vec(trial_length, environment, 2, num_agents);
    elapsed = toc;
    
    fprintf('\nSimulation completed successfully in %.2f seconds\n', elapsed);
    
    % Display summary statistics
    fprintf('\n--- Summary Statistics ---\n');
    fprintf('Number of trajectories: %d\n', size(out.x, 2));
    fprintf('X position range: [%.2f, %.2f] cm\n', min(out.x(:)), max(out.x(:)));
    fprintf('Y position range: [%.2f, %.2f] cm\n', min(out.y(:)), max(out.y(:)));
    
    if isfield(out, 'successrate')
        fprintf('Success rate: %.2f%%\n', out.successrate * 100);
    end
    
    if isfield(out, 'latency')
        valid_latencies = out.latency(~isnan(out.latency));
        if ~isempty(valid_latencies)
            fprintf('Mean latency: %.2f s\n', mean(valid_latencies));
        end
    end
    
    % Save results
    save(fullfile(test_dir, 'test_results.mat'), 'out');
    fprintf('\nResults saved to test_results.mat\n');
    
    % Save a sample trajectory plot
    if size(out.x, 2) >= 1
        figure('Visible', 'off');
        plot(out.x(:,1), out.y(:,1), 'k-', 'LineWidth', 1.5);
        hold on;
        plot(out.x(1,1), out.y(1,1), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
        plot(out.x(end,1), out.y(end,1), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        xlabel('X position (cm)');
        ylabel('Y position (cm)');
        title(sprintf('Sample trajectory - %s environment', environment));
        axis equal;
        grid on;
        saveas(gcf, fullfile(test_dir, 'sample_trajectory.png'));
        fprintf('Sample trajectory saved to sample_trajectory.png\n');
    end
    
catch ME
    fprintf('\nERROR: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  In %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    diary off;
    exit(1);
end

diary off;
exit(0);
EOF

# Run MATLAB simulation
echo -e "\n${GREEN}Starting MATLAB simulation...${NC}\n"

# Export environment variables for MATLAB
export SCRIPT_DIR="$SCRIPT_DIR"
export TEST_DIR="$TEST_DIR"
export TEST_ENVIRONMENT="$ENVIRONMENT"
export TEST_NUM_AGENTS="$NUM_AGENTS"
export TEST_TRIAL_LENGTH="$TRIAL_LENGTH"

matlab -nodisplay -logfile "$TEST_DIR/matlab_log.txt" -r "run('$TEST_DIR/test_script.m')"

# Check if MATLAB exited successfully
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Test run completed successfully!${NC}"
    echo -e "${YELLOW}Results saved in: $TEST_DIR${NC}"
    echo ""
    echo "Files created:"
    ls -la "$TEST_DIR"
    
    # Display last few lines of output
    echo -e "\n${GREEN}--- End of simulation output ---${NC}"
    tail -n 20 "$TEST_DIR/simulation_output.txt"
else
    echo -e "\n${RED}✗ Test run failed!${NC}"
    echo -e "${YELLOW}Check logs in: $TEST_DIR${NC}"
    exit 1
fi