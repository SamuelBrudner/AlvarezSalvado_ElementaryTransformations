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
from datetime import datetime

# Enhanced logging support with loguru
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    # Fallback logger for environments without loguru
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


def setup_logging():
    """Configure loguru logging to output to logs/ directory with structured format."""
    if not LOGURU_AVAILABLE:
        logger.warning("loguru not available, falling back to standard logging")
        return
    
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Remove default handler to avoid duplicate console output
    logger.remove()
    
    # Add structured file logging with timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"validate_output_{timestamp}.log"
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="gz"
    )
    
    # Add console handler only for errors and warnings to avoid cluttering user output
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        level="WARNING",
        colorize=True
    )
    
    logger.info("Logging initialized", log_file=str(log_file))


def parse_slurm_output(filepath):
    """Parse a SLURM output file and extract key information."""
    logger.debug("Parsing SLURM output file", filepath=filepath)
    
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
        logger.debug("Extracted job identifiers", 
                    job_id=info['job_id'], 
                    array_id=info['array_id'],
                    filename=filename)
    else:
        logger.warning("Could not extract job ID from filename", filename=filename)
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        logger.debug("Read file content", filepath=filepath, content_size=len(content))
            
        # Check for successful completion
        if 'Simulation completed successfully' in content:
            info['status'] = 'success'
            info['completed'] = True
            logger.debug("Job completed successfully", job_id=info['job_id'], array_id=info['array_id'])
            
            # Extract runtime
            runtime_match = re.search(r'completed successfully in ([\d.]+) seconds', content)
            if runtime_match:
                info['runtime'] = float(runtime_match.group(1))
                logger.debug("Extracted runtime", 
                           job_id=info['job_id'], 
                           array_id=info['array_id'],
                           runtime=info['runtime'])
        
        # Check for MATLAB errors
        if 'Unrecognized function or variable' in content:
            info['status'] = 'error'
            info['error_message'] = 'Function not found - check MATLAB path'
            logger.warning("MATLAB function not found error detected", 
                         job_id=info['job_id'], 
                         array_id=info['array_id'])
            
        if 'Error using' in content or 'Error in' in content:
            info['status'] = 'error'
            error_lines = [line.strip() for line in content.split('\n') 
                          if 'Error' in line and line.strip()]
            info['matlab_errors'] = error_lines[:5]  # First 5 error lines
            logger.error("MATLAB execution errors detected", 
                        job_id=info['job_id'], 
                        array_id=info['array_id'],
                        error_count=len(error_lines),
                        first_error=error_lines[0] if error_lines else None)
            
        # Check for out of memory
        if 'Out of memory' in content:
            info['status'] = 'oom'
            info['error_message'] = 'Out of memory'
            logger.warning("Out of memory error detected", 
                         job_id=info['job_id'], 
                         array_id=info['array_id'])
            
        # Check for timeout
        if 'DUE TO TIME LIMIT' in content:
            info['status'] = 'timeout'
            info['error_message'] = 'Job exceeded time limit'
            logger.warning("Job timeout detected", 
                         job_id=info['job_id'], 
                         array_id=info['array_id'])
            
    except Exception as e:
        info['status'] = 'read_error'
        info['error_message'] = str(e)
        logger.error("Error reading SLURM output file", 
                    filepath=filepath, 
                    error=str(e))
        
    logger.debug("Parsed SLURM output", 
                filepath=filepath,
                status=info['status'],
                completed=info['completed'])
    
    return info


def print_summary(results):
    """Print a summary of all job results."""
    logger.info("Generating job summary", total_jobs=len(results))
    
    total = len(results)
    if total == 0:
        print("No output files found!")
        logger.warning("No output files found for analysis")
        return
        
    status_counts = defaultdict(int)
    total_runtime = 0
    completed_count = 0
    
    for info in results.values():
        status_counts[info['status']] += 1
        if info['runtime']:
            total_runtime += info['runtime']
            completed_count += 1
    
    logger.info("Job statistics calculated", 
               status_counts=dict(status_counts),
               completed_jobs=completed_count,
               total_runtime=total_runtime)
    
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
        logger.info("Status breakdown", 
                   status=status, 
                   count=count, 
                   percentage=percentage)
    
    if completed_count > 0:
        avg_runtime = total_runtime / completed_count
        print(f"\nAverage runtime for successful jobs: {avg_runtime:.1f} seconds")
        logger.info("Runtime statistics", 
                   avg_runtime=avg_runtime,
                   total_runtime=total_runtime,
                   completed_count=completed_count)
    
    # Show sample errors
    error_samples = defaultdict(list)
    for filepath, info in results.items():
        if info['status'] == 'error' and info['matlab_errors']:
            key = info['matlab_errors'][0] if info['matlab_errors'] else 'Unknown error'
            error_samples[key].append(info['array_id'])
    
    if error_samples:
        logger.info("Error analysis", error_types=len(error_samples))
        print(f"\n{'='*60}")
        print("Common errors:")
        print(f"{'='*60}")
        for error, array_ids in list(error_samples.items())[:5]:
            print(f"\n{error}")
            print(f"  Affected array IDs: {', '.join(array_ids[:10])}")
            if len(array_ids) > 10:
                print(f"  ... and {len(array_ids) - 10} more")
            
            logger.warning("Common error pattern detected",
                         error_message=error,
                         affected_jobs=len(array_ids),
                         sample_array_ids=array_ids[:5])


def main():
    """Main entry point for output validation script."""
    # Initialize structured logging
    setup_logging()
    
    logger.info("Starting output validation", 
               command_args=sys.argv,
               working_directory=os.getcwd())
    
    if len(sys.argv) > 1:
        logs_dir = sys.argv[1]
        logger.info("Using specified logs directory", logs_dir=logs_dir)
    else:
        # Try to find the most recent logs directory
        pattern = '*_logs'
        dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if dirs:
            logs_dir = dirs[0]
            print(f"Using most recent logs directory: {logs_dir}")
            logger.info("Auto-detected logs directory", 
                       logs_dir=logs_dir,
                       available_dirs=dirs[:3])
        else:
            error_msg = "No logs directory specified and none found."
            print(error_msg)
            print(f"Usage: {sys.argv[0]} [logs_directory]")
            logger.error(error_msg, usage_pattern=f"{sys.argv[0]} [logs_directory]")
            sys.exit(1)
    
    if not os.path.isdir(logs_dir):
        error_msg = f"Error: {logs_dir} is not a directory"
        print(error_msg)
        logger.error("Invalid logs directory", logs_dir=logs_dir)
        sys.exit(1)
    
    # Find all output files
    out_files = glob.glob(os.path.join(logs_dir, 'slurm-*.out'))
    
    if not out_files:
        error_msg = f"No SLURM output files found in {logs_dir}"
        print(error_msg)
        logger.error("No SLURM output files found", 
                    logs_dir=logs_dir,
                    search_pattern="slurm-*.out")
        sys.exit(1)
    
    print(f"Found {len(out_files)} output files in {logs_dir}")
    print("Analyzing...")
    
    logger.info("Starting analysis", 
               output_files_count=len(out_files),
               logs_directory=logs_dir)
    
    # Parse all files
    results = {}
    for i, filepath in enumerate(out_files, 1):
        info = parse_slurm_output(filepath)
        results[filepath] = info
        
        # Log progress for large batches
        if i % 100 == 0 or i == len(out_files):
            logger.info("Analysis progress", 
                       processed=i, 
                       total=len(out_files),
                       percentage=round((i/len(out_files))*100, 1))
    
    logger.info("Analysis completed", 
               total_files_processed=len(results),
               logs_directory=logs_dir)
    
    # Print summary
    print_summary(results)
    
    # List failed jobs for easy re-running
    failed_arrays = [info['array_id'] for info in results.values() 
                    if info['status'] in ['error', 'oom', 'timeout'] and info['array_id']]
    
    if failed_arrays:
        print(f"\n{'='*60}")
        print("Failed array IDs (for re-running):")
        print(f"{'='*60}")
        print(','.join(failed_arrays))
        print(f"\nTo re-run failed jobs, use:")
        print(f"#SBATCH --array={','.join(failed_arrays[:10])}")
        if len(failed_arrays) > 10:
            print(f"# ... and {len(failed_arrays) - 10} more")
        
        logger.info("Failed jobs summary", 
                   failed_count=len(failed_arrays),
                   failed_array_ids=failed_arrays,
                   rerun_command=f"#SBATCH --array={','.join(failed_arrays[:10])}")
    
    logger.info("Output validation completed successfully")


if __name__ == '__main__':
    main()