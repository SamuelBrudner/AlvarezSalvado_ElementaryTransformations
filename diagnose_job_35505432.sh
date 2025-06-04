#!/bin/bash
# diagnose_job_35505432.sh - Diagnose what went wrong with job 35505432

JOB_ID="35505432"

echo "=== Diagnosing Job $JOB_ID Failure ==="
echo ""

echo "1. Error Analysis:"
echo "   The job failed because nav_job_paths.slurm tried to 'source' a JSON file"
echo "   This line was the problem:"
echo "   source \"/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/configs/paths.json\""
echo ""
echo "   JSON files contain data, not shell commands!"
echo ""

echo "2. What happened:"
echo "   - Bash tried to execute the JSON file as a shell script"
echo "   - Each line like '\"project_root\": \"/home/...\"' was interpreted as a command"
echo "   - Bash couldn't find commands named 'project_root:', 'code_dir:', etc."
echo "   - This filled the error log with 'command not found' messages"
echo ""

echo "3. MATLAB behavior:"
echo "   - MATLAB actually started successfully"
echo "   - The startup.m script ran and loaded paths correctly"
echo "   - But then MATLAB exited because the -r flag had no actual command"
echo "   - The MATLAB command string was incomplete/empty"
echo ""

echo "4. Checking how many tasks failed:"
if [ -d "logs" ]; then
    TOTAL_LOGS=$(ls -1 logs/nav-${JOB_ID}_*.out 2>/dev/null | wc -l)
    EMPTY_LOGS=$(find logs -name "nav-${JOB_ID}_*.out" -size 0 2>/dev/null | wc -l)
    echo "   Total log files: $TOTAL_LOGS"
    echo "   Empty log files: $EMPTY_LOGS"
    
    # Check for actual completions
    COMPLETED=$(grep -l "Task.*COMPLETE" logs/nav-${JOB_ID}_*.out 2>/dev/null | wc -l)
    echo "   Completed tasks: $COMPLETED"
    
    # Sample a few logs
    echo ""
    echo "5. Sample of log contents:"
    for i in 0 1 2; do
        LOG="logs/nav-${JOB_ID}_${i}.out"
        if [ -f "$LOG" ]; then
            echo ""
            echo "   Task $i log size: $(wc -c < $LOG) bytes"
            if [ $(wc -c < $LOG) -gt 100 ]; then
                echo "   Last few lines:"
                tail -5 "$LOG" | sed 's/^/     /'
            fi
        fi
    done
fi

echo ""
echo "6. The fix:"
echo "   - Remove the 'source' command for JSON files"
echo "   - Use proper MATLAB code in the -r flag"
echo "   - Let MATLAB's load_paths() function handle the JSON"
echo ""
echo "   Run: ./apply_slurm_fix.sh"
echo "   Then resubmit with: sbatch nav_job_paths.slurm"