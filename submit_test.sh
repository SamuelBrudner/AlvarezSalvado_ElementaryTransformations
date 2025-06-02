#!/bin/bash

# submit_test.sh - Submit a test job to SLURM with proper memory allocation

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Submitting Navigation Model Test Job ===${NC}"
echo "This will submit a small test job to SLURM with 82GB memory allocation"
echo ""

# Check if the test script exists
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test_nav_model.slurm"

# Create the test SLURM script
cat > "$TEST_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH --job-name=nav_test
#SBATCH --partition=day
#SBATCH --time=00:30:00
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
#SBATCH --output=test_output_%j.out
#SBATCH --error=test_output_%j.err

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run MATLAB with test parameters
matlab -nodisplay -r "
    addpath(genpath('$SCRIPT_DIR/Code'));
    
    % Test parameters - small scale for quick validation
    trial_length = 3600;
    environment = 'Crimaldi';
    num_agents = 10;
    
    fprintf('Test run started at %s\n', datestr(now));
    fprintf('Memory allocated: 82GB\n');
    fprintf('Environment: %s\n', environment);
    fprintf('Number of agents: %d\n', num_agents);
    fprintf('Trial length: %d samples\n\n', trial_length);
    
    try
        tic;
        out = navigation_model_vec(trial_length, environment, 0, num_agents);
        elapsed = toc;
        
        fprintf('\n✓ Simulation completed successfully in %.2f seconds\n', elapsed);
        fprintf('  Trajectories generated: %d\n', size(out.x, 2));
        
        if isfield(out, 'successrate')
            fprintf('  Success rate: %.2f%%\n', out.successrate * 100);
        end
        
        if isfield(out, 'latency')
            valid_latencies = out.latency(~isnan(out.latency));
            if ~isempty(valid_latencies)
                fprintf('  Mean latency: %.2f s\n', mean(valid_latencies));
            end
        end
        
        % Save results
        save(sprintf('test_results_%s.mat', getenv('SLURM_JOB_ID')), 'out', '-v7.3');
        fprintf('\nResults saved successfully\n');
        
    catch ME
        fprintf('\n✗ ERROR: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('  In %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
        exit(1);
    end
    
    exit(0);
"
EOF

# Submit the job
echo -e "${YELLOW}Submitting test job to SLURM...${NC}"
JOB_ID=$(sbatch "$TEST_SCRIPT" | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo -e "${RED}Failed to submit job${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Job submitted successfully with ID: $JOB_ID${NC}"
echo ""
echo "To check job status:"
echo "  squeue -j $JOB_ID"
echo ""
echo "To view output when complete:"
echo "  cat test_output_${JOB_ID}.out"
echo ""
echo "To cancel if needed:"
echo "  scancel $JOB_ID"

# Optional: Wait and show status
echo ""
echo -e "${YELLOW}Waiting for job to start...${NC}"
sleep 2

# Check job status
while true; do
    STATUS=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null || echo "COMPLETED")
    
    case $STATUS in
        PENDING)
            echo -ne "\rStatus: PENDING (waiting for resources)..."
            sleep 5
            ;;
        RUNNING)
            echo -e "\r${GREEN}Status: RUNNING${NC}                    "
            echo "Job is now running. Check output file for progress:"
            echo "  tail -f test_output_${JOB_ID}.out"
            break
            ;;
        COMPLETED|"")
            echo -e "\r${GREEN}Status: COMPLETED${NC}                 "
            echo "Job finished. Check output:"
            echo "  cat test_output_${JOB_ID}.out"
            break
            ;;
        FAILED)
            echo -e "\r${RED}Status: FAILED${NC}                    "
            echo "Check error file:"
            echo "  cat test_output_${JOB_ID}.err"
            exit 1
            ;;
        *)
            echo -e "\rStatus: $STATUS"
            sleep 5
            ;;
    esac
done