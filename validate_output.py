#!/usr/bin/env python3
"""
validate_output.py - Validate navigation model outputs

This script checks SLURM output files for common errors and provides
a summary of job status.

Usage:
    python validate_output.py [logs_directory]
"""

import sys
import os
import re
import glob
from collections import defaultdict
from pathlib import Path


def parse_slurm_output(filepath):
    """Parse a SLURM output file and extract key information."""
    info = {
        'job_id': None,
        'array_id': None,
        'status': 'unknown',
        'error_message': None,
        'matlab_errors': [],
        'completed': False,
        'runtime': None
    }
    
    # Extract job and array ID from filename
    filename = os.path.basename(filepath)
    match = re.match(r'slurm-(\d+)_(\d+)\.(out|err)', filename)
    if match:
        info['job_id'] = match.group(1)
        info['array_id'] = match.group(2)
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Check for successful completion
        if 'Simulation completed successfully' in content:
            info['status'] = 'success'
            info['completed'] = True
            
            # Extract runtime
            runtime_match = re.search(r'completed successfully in ([\d.]+) seconds', content)
            if runtime_match:
                info['runtime'] = float(runtime_match.group(1))
        
        # Check for MATLAB errors
        if 'Unrecognized function or variable' in content:
            info['status'] = 'error'
            info['error_message'] = 'Function not found - check MATLAB path'
            
        if 'Error using' in content or 'Error in' in content:
            info['status'] = 'error'
            error_lines = [line.strip() for line in content.split('\n') 
                          if 'Error' in line and line.strip()]
            info['matlab_errors'] = error_lines[:5]  # First 5 error lines
            
        # Check for out of memory
        if 'Out of memory' in content:
            info['status'] = 'oom'
            info['error_message'] = 'Out of memory'
            
        # Check for timeout
        if 'DUE TO TIME LIMIT' in content:
            info['status'] = 'timeout'
            info['error_message'] = 'Job exceeded time limit'
            
    except Exception as e:
        info['status'] = 'read_error'
        info['error_message'] = str(e)
        
    return info


def print_summary(results):
    """Print a summary of all job results."""
    total = len(results)
    if total == 0:
        print("No output files found!")
        return
        
    status_counts = defaultdict(int)
    total_runtime = 0
    completed_count = 0
    
    for info in results.values():
        status_counts[info['status']] += 1
        if info['runtime']:
            total_runtime += info['runtime']
            completed_count += 1
    
    print(f"\n{'='*60}")
    print(f"SLURM Job Summary")
    print(f"{'='*60}")
    print(f"Total jobs: {total}")
    print(f"\nStatus breakdown:")
    
    for status, count in sorted(status_counts.items()):
        percentage = (count / total) * 100
        symbol = {
            'success': '✓',
            'error': '✗',
            'oom': '⚠',
            'timeout': '⏱',
            'unknown': '?',
            'read_error': '!'
        }.get(status, '-')
        
        print(f"  {symbol} {status:12s}: {count:4d} ({percentage:5.1f}%)")
    
    if completed_count > 0:
        avg_runtime = total_runtime / completed_count
        print(f"\nAverage runtime for successful jobs: {avg_runtime:.1f} seconds")
    
    # Show sample errors
    error_samples = defaultdict(list)
    for filepath, info in results.items():
        if info['status'] == 'error' and info['matlab_errors']:
            key = info['matlab_errors'][0] if info['matlab_errors'] else 'Unknown error'
            error_samples[key].append(info['array_id'])
    
    if error_samples:
        print(f"\n{'='*60}")
        print("Common errors:")
        print(f"{'='*60}")
        for error, array_ids in list(error_samples.items())[:5]:
            print(f"\n{error}")
            print(f"  Affected array IDs: {', '.join(array_ids[:10])}")
            if len(array_ids) > 10:
                print(f"  ... and {len(array_ids) - 10} more")


def main():
    if len(sys.argv) > 1:
        logs_dir = sys.argv[1]
    else:
        # Try to find the most recent logs directory
        pattern = '*_logs'
        dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if dirs:
            logs_dir = dirs[0]
            print(f"Using most recent logs directory: {logs_dir}")
        else:
            print("No logs directory specified and none found.")
            print(f"Usage: {sys.argv[0]} [logs_directory]")
            sys.exit(1)
    
    if not os.path.isdir(logs_dir):
        print(f"Error: {logs_dir} is not a directory")
        sys.exit(1)
    
    # Find all output files
    out_files = glob.glob(os.path.join(logs_dir, 'slurm-*.out'))
    
    if not out_files:
        print(f"No SLURM output files found in {logs_dir}")
        sys.exit(1)
    
    print(f"Found {len(out_files)} output files in {logs_dir}")
    print("Analyzing...")
    
    # Parse all files
    results = {}
    for filepath in out_files:
        info = parse_slurm_output(filepath)
        results[filepath] = info
    
    # Print summary
    print_summary(results)
    
    # List failed jobs for easy re-running
    failed_arrays = [info['array_id'] for info in results.values() 
                    if info['status'] in ['error', 'oom', 'timeout']]
    
    if failed_arrays:
        print(f"\n{'='*60}")
        print("Failed array IDs (for re-running):")
        print(f"{'='*60}")
        print(','.join(failed_arrays))
        print(f"\nTo re-run failed jobs, use:")
        print(f"#SBATCH --array={','.join(failed_arrays[:10])}")
        if len(failed_arrays) > 10:
            print(f"# ... and {len(failed_arrays) - 10} more")


if __name__ == '__main__':
    main()