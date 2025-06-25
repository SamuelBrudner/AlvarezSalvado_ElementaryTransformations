#!/bin/bash
# run_plume_sim.sh - Simple launcher for common plume simulation scenarios
#
# Usage: ./scripts/run_plume_sim.sh [SCENARIO] [NUM_AGENTS] [-v|--verbose]
#
# Scenarios:
#   test     - Quick test (10 agents each plume)
#   smoke    - Smoke plume only
#   crimaldi - Crimaldi plume only  
#   both     - Both plumes for comparison
#   large    - Large comparative study (5000 agents each)
#
# Options:
#   -v, --verbose    Enable verbose logging with detailed trace output

# Initialize verbose flag
VERBOSE=0

# Parse arguments for verbose flag
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Verbose logging enabled"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Set positional parameters from filtered arguments
set -- "${ARGS[@]}"

SCENARIO="${1:-test}"
NUM_AGENTS="${2:-}"

[[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Scenario='$SCENARIO', Agents='$NUM_AGENTS'"

# Ensure we're in the right directory
[[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Checking for SLURM templates in slurm/ directory"
if [ ! -f "slurm/nav_job_smoke.slurm" ]; then
    [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: SLURM templates not found, attempting to change to project directory"
    cd /vast/palmer/home.grace/snb6/Documents/AlvarezSalvado_ElementaryTransformations 2>/dev/null || {
        echo "Error: Cannot find project directory"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Failed to locate project directory - exiting"
        exit 1
    }
    [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Successfully changed to project directory"
fi

[[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Processing scenario '$SCENARIO'"

case $SCENARIO in
    test)
        echo "=== Quick Test Run ==="
        echo "Running 10 agents on each plume..."
        [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Executing hpc_batch_submit.sh for test scenario"
        if [[ $VERBOSE -eq 1 ]]; then
            ./scripts/hpc_batch_submit.sh both 10 --verbose
        else
            ./scripts/hpc_batch_submit.sh both 10
        fi
        echo ""
        echo "Results will appear in:"
        echo "  results/nav_results_0000.mat     (Crimaldi)"
        echo "  results/smoke_nav_results_1000.mat (Smoke)"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Test scenario submission completed"
        ;;
        
    smoke)
        AGENTS="${NUM_AGENTS:-1000}"
        echo "=== Smoke Plume Simulation ==="
        echo "Agents: $AGENTS"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Executing smoke plume simulation with $AGENTS agents"
        if [[ $VERBOSE -eq 1 ]]; then
            ./scripts/hpc_batch_submit.sh smoke $AGENTS --verbose
        else
            ./scripts/hpc_batch_submit.sh smoke $AGENTS
        fi
        [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Smoke simulation submission completed"
        ;;
        
    crimaldi|crim)
        AGENTS="${NUM_AGENTS:-1000}"
        echo "=== Crimaldi Plume Simulation ==="
        echo "Agents: $AGENTS"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Executing Crimaldi plume simulation with $AGENTS agents"
        if [[ $VERBOSE -eq 1 ]]; then
            ./scripts/hpc_batch_submit.sh crimaldi $AGENTS --verbose
        else
            ./scripts/hpc_batch_submit.sh crimaldi $AGENTS
        fi
        [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Crimaldi simulation submission completed"
        ;;
        
    both|compare)
        AGENTS="${NUM_AGENTS:-1000}"
        echo "=== Comparative Study ==="
        echo "Agents per plume: $AGENTS"
        echo "Total agents: $((AGENTS * 2))"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Executing comparative study with $AGENTS agents per plume"
        if [[ $VERBOSE -eq 1 ]]; then
            ./scripts/hpc_batch_submit.sh both $AGENTS --verbose
        else
            ./scripts/hpc_batch_submit.sh both $AGENTS
        fi
        echo ""
        echo "Monitor with: ./scripts/hpc_monitor_results.sh watch"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Comparative study submission completed"
        ;;
        
    large)
        echo "=== Large Comparative Study ==="
        echo "This will run 5000 agents on each plume (10,000 total)"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Prompting user confirmation for large study"
        read -p "Continue? (yes/no): " CONFIRM
        if [ "$CONFIRM" = "yes" ]; then
            [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: User confirmed - executing large study with 8-hour time limit"
            if [[ $VERBOSE -eq 1 ]]; then
                ./scripts/hpc_batch_submit.sh both 5000 --time=8:00:00 --verbose
            else
                ./scripts/hpc_batch_submit.sh both 5000 --time=8:00:00
            fi
        else
            echo "Cancelled"
            [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: User cancelled large study"
        fi
        ;;
        
    interactive)
        echo "=== Interactive Submission ==="
        [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Launching interactive validation and submission interface"
        if [[ $VERBOSE -eq 1 ]]; then
            ./scripts/validate_and_submit_plume.sh --verbose
        else
            ./scripts/validate_and_submit_plume.sh
        fi
        [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Interactive submission completed"
        ;;
        
    *)
        echo "Simple plume simulation launcher"
        echo ""
        echo "Usage: $0 [SCENARIO] [NUM_AGENTS] [-v|--verbose]"
        echo ""
        echo "Quick scenarios:"
        echo "  $0 test              # Quick test (10 agents each)"
        echo "  $0 smoke 100         # 100 agents on smoke plume"
        echo "  $0 crimaldi 100      # 100 agents on Crimaldi plume"
        echo "  $0 both 500          # 500 agents on each plume"
        echo "  $0 large             # Large study (5000 each)"
        echo "  $0 interactive       # Full validation interface"
        echo ""
        echo "Options:"
        echo "  -v, --verbose        # Enable detailed trace output"
        echo ""
        echo "Examples:"
        echo "  $0 test              # Quick functionality test"
        echo "  $0 both 1000         # Standard comparison"
        echo "  $0 smoke 2000        # Large smoke-only run"
        echo "  $0 test --verbose    # Test with detailed logging"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Help information displayed"
        ;;
esac

[[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_plume_sim: Script execution completed"