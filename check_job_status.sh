#!/bin/bash
# check_job_status.sh - Check the status of navigation model jobs

echo "=== Navigation Model Job Status ==="
echo ""

# Check current queue
echo "1. Currently running/pending jobs:"
squeue -u $USER -n nav_model --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" | head -20

echo ""
echo "2. Completed results:"
TOTAL_RESULTS=$(ls -1 results/nav_results_*.mat 2>/dev/null | wc -l)
echo "   Found $TOTAL_RESULTS result files"

if [ $TOTAL_RESULTS -gt 0 ]; then
    echo ""
    echo "   Latest results:"
    ls -lt results/nav_results_*.mat 2>/dev/null | head -5
fi

echo ""
echo "3. Recent successful completions in logs:"
grep -l "Task.*Complete\|COMPLETE" logs/nav-*.out 2>/dev/null | tail -10 | while read logfile; do
    TASK_NUM=$(basename "$logfile" | sed 's/nav-[0-9]*_\([0-9]*\)\.out/\1/')
    SUCCESS_RATE=$(grep -o "Success rate: [0-9.]*%" "$logfile" 2>/dev/null | tail -1)
    echo "   Task $TASK_NUM: $SUCCESS_RATE"
done

echo ""
echo "4. Recent errors:"
ERROR_COUNT=$(grep -l "FAILED\|Error:" logs/nav-*.err 2>/dev/null | wc -l)
echo "   Found $ERROR_COUNT error logs"

if [ $ERROR_COUNT -gt 0 ]; then
    echo "   Recent errors:"
    grep -h "Error:" logs/nav-*.err 2>/dev/null | sort | uniq -c | head -5
fi

echo ""
echo "5. Job summary:"
echo "   Target: 100 tasks (0-99)"
echo "   Completed: $TOTAL_RESULTS"
echo "   Remaining: $((100 - TOTAL_RESULTS))"

if [ $TOTAL_RESULTS -lt 100 ]; then
    echo ""
    echo "To submit remaining jobs:"
    echo "  sbatch --array=$TOTAL_RESULTS-99%20 nav_job_paths.slurm"
fi

echo ""
echo "To check a specific result:"
echo "  ./matlab_results_check.sh results/nav_results_0000.mat"
echo ""
echo "To monitor live:"
echo "  watch -n 5 'ls -1 results/*.mat | wc -l'"