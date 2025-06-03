#!/bin/bash
# create_results_report.sh - Generate a summary report of all results
#
# Usage: ./create_results_report.sh
# 
# Analyzes all nav_results_*.mat files in the results/ directory
# Creates a summary report with timestamp
#
# Requires: Python with scipy installed (use conda environment)

RESULTS_DIR="results"
REPORT_FILE="results_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "=== Navigation Model Results Summary ===" > $REPORT_FILE
echo "Generated: $(date)" >> $REPORT_FILE
echo "" >> $REPORT_FILE

# Count result files
N_FILES=$(ls -1 $RESULTS_DIR/nav_results_*.mat 2>/dev/null | wc -l)
echo "Total result files: $N_FILES" >> $REPORT_FILE
echo "" >> $REPORT_FILE

if [ $N_FILES -eq 0 ]; then
    echo "No result files found in $RESULTS_DIR/" >> $REPORT_FILE
    echo "Report saved to: $REPORT_FILE"
    exit 0
fi

# Analyze each file
echo "Individual Results:" >> $REPORT_FILE
echo "==================" >> $REPORT_FILE

for file in $RESULTS_DIR/nav_results_*.mat; do
    echo "" >> $REPORT_FILE
    echo "File: $(basename $file)" >> $REPORT_FILE
    
    # Quick Python analysis
    conda run --prefix ./dev_env python << EOF >> $REPORT_FILE 2>&1
import scipy.io
import numpy as np

try:
    data = scipy.io.loadmat('$file', struct_as_record=False, squeeze_me=True)
    out = data['out']
    
    # Get dimensions
    if hasattr(out.x, 'shape') and len(out.x.shape) > 1:
        n_agents = out.x.shape[1]
    else:
        n_agents = 1
    
    print(f"  Agents: {n_agents}")
    
    if hasattr(out, 'successrate'):
        print(f"  Success rate: {out.successrate*100:.1f}%")
    
    if hasattr(out, 'latency'):
        if np.isscalar(out.latency):
            if not np.isnan(out.latency):
                print(f"  Time to target: {out.latency:.1f}s")
        else:
            valid = out.latency[~np.isnan(out.latency)]
            if len(valid) > 0:
                print(f"  Mean latency: {np.mean(valid):.1f}s (n={len(valid)})")
                
except Exception as e:
    print(f"  Error: {e}")
EOF
done

# Overall statistics
echo "" >> $REPORT_FILE
echo "Overall Statistics:" >> $REPORT_FILE
echo "==================" >> $REPORT_FILE

conda run --prefix ./dev_env python3 << 'EOF' >> $REPORT_FILE 2>&1
import scipy.io
import numpy as np
import glob

all_success_rates = []
all_latencies = []
total_agents = 0

for file in glob.glob('results/nav_results_*.mat'):
    try:
        data = scipy.io.loadmat(file, struct_as_record=False, squeeze_me=True)
        out = data['out']
        
        if hasattr(out.x, 'shape') and len(out.x.shape) > 1:
            total_agents += out.x.shape[1]
        else:
            total_agents += 1
        
        if hasattr(out, 'successrate'):
            all_success_rates.append(out.successrate)
        
        if hasattr(out, 'latency'):
            if np.isscalar(out.latency):
                if not np.isnan(out.latency):
                    all_latencies.append(out.latency)
            else:
                all_latencies.extend(out.latency[~np.isnan(out.latency)])
                
    except:
        pass

print(f"Total agents simulated: {total_agents}")
if all_success_rates:
    print(f"Mean success rate: {np.mean(all_success_rates)*100:.1f}%")
if all_latencies:
    print(f"Overall mean latency: {np.mean(all_latencies):.1f} seconds")
    print(f"Fastest: {np.min(all_latencies):.1f}s, Slowest: {np.max(all_latencies):.1f}s")
EOF

echo "" >> $REPORT_FILE
echo "Report saved to: $REPORT_FILE"
cat $REPORT_FILE