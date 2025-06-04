# HPC Plume Simulation Guide

Quick reference for running navigation model simulations on HPC with Crimaldi and smoke plumes.

## Quick Start

```bash
# Make scripts executable
chmod +x *.sh

# Quick test (10 agents on each plume)
./run_plume_sim.sh test

# Standard runs
./run_plume_sim.sh smoke 1000      # 1000 agents on smoke
./run_plume_sim.sh crimaldi 1000   # 1000 agents on Crimaldi
./run_plume_sim.sh both 1000        # 1000 agents on EACH plume

# Monitor progress
./hpc_monitor_results.sh watch
```

## Available Scripts

### 1. **run_plume_sim.sh** - Simple Launcher
The easiest way to start simulations:
```bash
./run_plume_sim.sh test              # Quick test
./run_plume_sim.sh both 1000         # Comparative study
./run_plume_sim.sh large             # 5000 agents each
```

### 2. **hpc_batch_submit.sh** - Direct Batch Submission
For more control:
```bash
./hpc_batch_submit.sh smoke 1000
./hpc_batch_submit.sh crimaldi 2000
./hpc_batch_submit.sh both 500 --partition gpu
```

### 3. **validate_and_submit_plume.sh** - Full Validation
Interactive submission with validation figures:
```bash
./validate_and_submit_plume.sh              # Interactive menu
./validate_and_submit_plume.sh 100 0 both   # 100 tasks on both plumes
```

### 4. **hpc_comparative_study.sh** - Detailed Comparison
Creates organized comparative studies:
```bash
./hpc_comparative_study.sh 1000 wind_test
# Creates comparative_studies/wind_test/ with:
# - Custom SLURM script
# - Analysis scripts
# - Organized results
```

### 5. **hpc_monitor_results.sh** - Monitor & Analyze
```bash
./hpc_monitor_results.sh status   # Job status
./hpc_monitor_results.sh results  # Recent results
./hpc_monitor_results.sh compare  # Compare matched pairs
./hpc_monitor_results.sh watch    # Live monitoring
```

## Running Both Plumes Simultaneously

### Method 1: Simple (Recommended)
```bash
./run_plume_sim.sh both 1000
```

### Method 2: Batch Submit
```bash
./hpc_batch_submit.sh both 1000
```

### Method 3: Full Validation
```bash
./validate_and_submit_plume.sh 100 0 both
```

## How "Both" Mode Works

When you select "both" plumes:
1. **Crimaldi tasks**: 0-99 → `results/nav_results_XXXX.mat`
2. **Smoke tasks**: 1000-1099 → `results/smoke_nav_results_1XXX.mat`

This offset (1000) keeps results organized and allows easy pairing for comparison.

## Memory Requirements

| Plume | File Size | Memory Needed | Time (100 agents) |
|-------|-----------|---------------|-------------------|
| Crimaldi | 9 GB | 32 GB | ~30 min |
| Smoke | 0.58 GB | 16 GB | ~20 min |

## Quick Analysis

### Check single result:
```bash
matlab -batch "
    load('results/nav_results_0000.mat');
    fprintf('Success: %.1f%%\n', out.successrate*100);
"
```

### Compare plumes:
```bash
matlab -batch "
    c = load('results/nav_results_0000.mat');
    s = load('results/smoke_nav_results_1000.mat');
    fprintf('Crimaldi: %.1f%%\n', c.out.successrate*100);
    fprintf('Smoke: %.1f%%\n', s.out.successrate*100);
"
```

### Full comparison:
```bash
./hpc_monitor_results.sh compare
```

## Common Workflows

### 1. Quick Test
```bash
# Test both plumes work
./run_plume_sim.sh test
squeue -u $USER
# Wait for completion
./hpc_monitor_results.sh results
```

### 2. Production Run
```bash
# Run 5000 agents on each plume
./hpc_batch_submit.sh both 5000
./hpc_monitor_results.sh watch  # Monitor progress
```

### 3. Parameter Study
```bash
# Create named study
./hpc_comparative_study.sh 2000 param_test_v1
sbatch comparative_studies/param_test_v1/compare_param_test_v1.slurm
```

## File Organization

```
results/
  nav_results_0000.mat       # Crimaldi task 0
  nav_results_0001.mat       # Crimaldi task 1
  ...
  smoke_nav_results_1000.mat # Smoke task 0 (array ID 1000)
  smoke_nav_results_1001.mat # Smoke task 1 (array ID 1001)
  ...

comparative_studies/
  wind_test/
    compare_wind_test.slurm  # SLURM script
    analyze_results.m        # Analysis script
    results/                 # Study-specific results
    summary.mat             # Summary statistics
    comparison_plot.png     # Visualization
```

## Tips

1. **Check active plume**: 
   ```bash
   ./check_plume_status.sh
   ```

2. **Switch plumes manually**:
   ```bash
   ./setup_crimaldi_plume.sh
   ./setup_smoke_plume_config.sh
   ```

3. **Clean up old results**:
   ```bash
   mkdir -p results/archive_$(date +%Y%m%d)
   mv results/*nav_results_*.mat results/archive_$(date +%Y%m%d)/
   ```

4. **Resource optimization**:
   - Smoke plume needs only 16GB (compressed file)
   - Can run more concurrent smoke tasks
   - Crimaldi needs 32GB minimum

5. **Debugging failures**:
   ```bash
   # Check specific task log
   tail logs/smoke-JOBID_1000.err
   tail logs/crim-JOBID_0.err
   ```

## Example: Complete Comparative Study

```bash
# 1. Run simulations
./run_plume_sim.sh both 1000

# 2. Monitor progress
./hpc_monitor_results.sh watch

# 3. Once complete, analyze
./hpc_monitor_results.sh compare

# 4. Create detailed report
matlab -r "
    % Load all results
    crim_success = [];
    smoke_success = [];
    for i = 0:99
        if exist(sprintf('results/nav_results_%04d.mat', i), 'file')
            c = load(sprintf('results/nav_results_%04d.mat', i));
            crim_success(end+1) = c.out.successrate;
        end
        if exist(sprintf('results/smoke_nav_results_%04d.mat', i+1000), 'file')
            s = load(sprintf('results/smoke_nav_results_%04d.mat', i+1000));
            smoke_success(end+1) = s.out.successrate;
        end
    end
    
    fprintf('\n=== Final Results ===\n');
    fprintf('Crimaldi: %.1f%% ± %.1f%% (n=%d)\n', ...
            mean(crim_success)*100, std(crim_success)*100, length(crim_success));
    fprintf('Smoke: %.1f%% ± %.1f%% (n=%d)\n', ...
            mean(smoke_success)*100, std(smoke_success)*100, length(smoke_success));
    
    [h,p] = ttest2(crim_success, smoke_success);
    fprintf('t-test p-value: %.4f\n', p);
    exit;
"
```