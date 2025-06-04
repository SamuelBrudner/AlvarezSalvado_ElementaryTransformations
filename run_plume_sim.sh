#!/bin/bash
# run_plume_sim.sh - Simple launcher for common plume simulation scenarios
#
# Usage: ./run_plume_sim.sh [SCENARIO] [NUM_AGENTS]
#
# Scenarios:
#   test     - Quick test (10 agents each plume)
#   smoke    - Smoke plume only
#   crimaldi - Crimaldi plume only  
#   both     - Both plumes for comparison
#   large    - Large comparative study (5000 agents each)

SCENARIO="${1:-test}"
NUM_AGENTS="${2:-}"

# Ensure we're in the right directory
if [ ! -f "nav_job_smoke.slurm" ]; then
    cd /vast/palmer/home.grace/snb6/Documents/AlvarezSalvado_ElementaryTransformations 2>/dev/null || {
        echo "Error: Cannot find project directory"
        exit 1
    }
fi

case $SCENARIO in
    test)
        echo "=== Quick Test Run ==="
        echo "Running 10 agents on each plume..."
        ./hpc_batch_submit.sh both 10
        echo ""
        echo "Results will appear in:"
        echo "  results/nav_results_0000.mat     (Crimaldi)"
        echo "  results/smoke_nav_results_1000.mat (Smoke)"
        ;;
        
    smoke)
        AGENTS="${NUM_AGENTS:-1000}"
        echo "=== Smoke Plume Simulation ==="
        echo "Agents: $AGENTS"
        ./hpc_batch_submit.sh smoke $AGENTS
        ;;
        
    crimaldi|crim)
        AGENTS="${NUM_AGENTS:-1000}"
        echo "=== Crimaldi Plume Simulation ==="
        echo "Agents: $AGENTS"
        ./hpc_batch_submit.sh crimaldi $AGENTS
        ;;
        
    both|compare)
        AGENTS="${NUM_AGENTS:-1000}"
        echo "=== Comparative Study ==="
        echo "Agents per plume: $AGENTS"
        echo "Total agents: $((AGENTS * 2))"
        ./hpc_batch_submit.sh both $AGENTS
        echo ""
        echo "Monitor with: ./hpc_monitor_results.sh watch"
        ;;
        
    large)
        echo "=== Large Comparative Study ==="
        echo "This will run 5000 agents on each plume (10,000 total)"
        read -p "Continue? (yes/no): " CONFIRM
        if [ "$CONFIRM" = "yes" ]; then
            ./hpc_batch_submit.sh both 5000 --time=8:00:00
        else
            echo "Cancelled"
        fi
        ;;
        
    interactive)
        echo "=== Interactive Submission ==="
        ./validate_and_submit_plume.sh
        ;;
        
    *)
        echo "Simple plume simulation launcher"
        echo ""
        echo "Usage: $0 [SCENARIO] [NUM_AGENTS]"
        echo ""
        echo "Quick scenarios:"
        echo "  $0 test              # Quick test (10 agents each)"
        echo "  $0 smoke 100         # 100 agents on smoke plume"
        echo "  $0 crimaldi 100      # 100 agents on Crimaldi plume"
        echo "  $0 both 500          # 500 agents on each plume"
        echo "  $0 large             # Large study (5000 each)"
        echo "  $0 interactive       # Full validation interface"
        echo ""
        echo "Examples:"
        echo "  $0 test              # Quick functionality test"
        echo "  $0 both 1000         # Standard comparison"
        echo "  $0 smoke 2000        # Large smoke-only run"
        ;;
esac