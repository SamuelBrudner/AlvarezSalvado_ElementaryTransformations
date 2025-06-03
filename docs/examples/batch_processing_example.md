# Batch Processing Example Guide

## Table of Contents

- [Introduction to Batch Processing](#introduction-to-batch-processing)
- [Basic Batch Processing Example](#basic-batch-processing-example)
- [Advanced Batch Processing with BatchExecutor](#advanced-batch-processing-with-batchexecutor)
- [Progress Monitoring and Visualization](#progress-monitoring-and-visualization)
- [Algorithm Comparison Workflows](#algorithm-comparison-workflows)
- [Error Handling and Recovery](#error-handling-and-recovery)
- [Performance Optimization Strategies](#performance-optimization-strategies)
- [Scientific Computing Best Practices](#scientific-computing-best-practices)
- [Complete Workflow Examples](#complete-workflow-examples)
- [Troubleshooting and Optimization](#troubleshooting-and-optimization)

## Overview

This comprehensive example guide demonstrates how to execute large-scale batch processing of plume navigation algorithm simulations using the plume simulation system. The guide covers complete workflows from basic batch execution to advanced optimization strategies, providing practical examples for achieving scientific computing excellence with >95% correlation accuracy and <7.2 seconds average simulation time.

## Introduction to Batch Processing

### Batch Processing Overview

Batch processing capabilities enable the execution of 4000+ simulations within an 8-hour target timeframe through coordinated parallel processing, comprehensive resource management, and scientific reproducibility standards. This approach is essential for large-scale algorithm evaluation studies that require systematic comparison across multiple navigation strategies and environmental conditions.

**Key Capabilities:**
- **Automated Parallel Execution**: Coordinate multiple simulation workers for optimal resource utilization
- **Progress Monitoring**: Real-time tracking of batch progress with detailed performance metrics
- **Resource Management**: Intelligent allocation and monitoring of CPU, memory, and I/O resources
- **Scientific Reproducibility**: Consistent methodology ensuring reproducible results across computational environments
- **Quality Validation**: Comprehensive validation against correlation thresholds and performance targets

**Architecture Overview:**
The batch processing system employs a modular architecture separating data preprocessing, simulation execution, and analysis phases. This enables parallel processing of independent simulation tasks while maintaining serial dependency management for data-dependent operations.

### Key Benefits and Use Cases

**Primary Benefits:**
- **Automated Execution**: Eliminate manual intervention for large-scale simulation studies
- **Parallel Processing Optimization**: Achieve optimal throughput through intelligent worker allocation
- **Comprehensive Progress Monitoring**: Real-time visibility into batch execution status and performance
- **Statistical Validation**: Built-in correlation analysis and reproducibility assessment
- **Scientific Reproducibility**: Consistent methodology ensuring reliable research outcomes

**Primary Use Cases:**

1. **Algorithm Comparison Studies**: Compare navigation algorithms (infotaxis, casting, gradient following, plume tracking, hybrid strategies) across standardized datasets
2. **Performance Benchmarking**: Establish baseline performance metrics for algorithm evaluation
3. **Large-Scale Research Validation**: Execute comprehensive studies with statistical significance
4. **Parameter Sensitivity Analysis**: Systematic evaluation of algorithm parameters across environmental conditions
5. **Cross-Platform Validation**: Verify algorithm performance across different plume data formats

### Prerequisites and System Requirements

**Hardware Requirements:**
- **CPU**: 8+ cores recommended for optimal parallel processing
- **Memory**: 8GB+ RAM minimum, 16GB+ recommended for large-scale batches
- **Storage**: SSD recommended for video data processing, 100GB+ available space
- **Network**: Stable connection if accessing remote data sources

**Software Dependencies:**
- Python 3.9+ with scientific computing libraries
- NumPy 2.1.3+ for numerical operations
- SciPy 1.15.3+ for statistical analysis
- OpenCV 4.11.0+ for video processing
- Pandas 2.2.0+ for data management
- Joblib 1.6.0+ for parallel processing

**Data Prerequisites:**
- **Normalized Plume Data**: Properly processed video files in supported formats
- **Calibration Parameters**: Arena size, pixel resolution, temporal sampling configuration
- **Algorithm Configuration**: Valid parameter sets for selected navigation algorithms
- **Reference Data**: Benchmark datasets for correlation validation

**Environment Setup Validation:**
```bash
# Verify system capabilities
python -c "import multiprocessing; print(f'CPU cores: {multiprocessing.cpu_count()}')"
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total/1024/1024/1024:.1f}GB')"

# Validate required libraries
python -c "import numpy, scipy, cv2, pandas, joblib; print('Dependencies verified')"
```

## Basic Batch Processing Example

### Simple Batch Execution Setup

This section demonstrates basic batch processing using the `SimpleBatchSimulationExample` class for executing small to medium-scale simulation batches with comprehensive progress monitoring and result validation.

**Step 1: Configuration Setup**

Create a configuration file for your batch simulation:

```json
{
  "simulation_config": {
    "default_algorithm_parameters": {
      "infotaxis": {
        "lambda": 0.1,
        "max_steps": 1000,
        "step_size": 1.0
      },
      "casting": {
        "cast_width": 10.0,
        "surge_distance": 20.0,
        "max_steps": 1000
      },
      "gradient_following": {
        "step_size": 1.0,
        "gradient_threshold": 0.01,
        "max_steps": 1000
      }
    }
  },
  "data_processing": {
    "normalization_enabled": true,
    "validation_checks": true,
    "output_format": "comprehensive"
  },
  "performance_monitoring": {
    "enable_progress_bars": true,
    "show_real_time_metrics": true,
    "log_detailed_statistics": true
  }
}
```

**Step 2: Directory Structure Preparation**

```bash
# Create organized directory structure
mkdir -p results/basic_batch_example/{logs,data,analysis}
mkdir -p data/normalized
mkdir -p config

# Verify data availability
ls -la data/normalized/
```

**Step 3: Validation Procedures**

Before executing batch processing, validate your environment and data:

```python
#!/usr/bin/env python3
"""
Environment and Data Validation for Batch Processing

Validates system capabilities, data availability, and configuration
parameters before executing batch simulations.
"""

import json
import os
from pathlib import Path
import sys

def validate_system_requirements():
    """Validate system meets minimum requirements for batch processing"""
    import multiprocessing
    import psutil
    
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"System Validation:")
    print(f"  CPU Cores: {cpu_count}")
    print(f"  Available RAM: {memory_gb:.1f}GB")
    
    if cpu_count < 4:
        print("  ⚠ Warning: Less than 4 CPU cores available")
    if memory_gb < 8:
        print("  ⚠ Warning: Less than 8GB RAM available")
    
    return cpu_count >= 4 and memory_gb >= 8

def validate_data_availability(data_directory):
    """Validate required data files are available"""
    data_path = Path(data_directory)
    
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_directory}")
        return False
    
    video_files = list(data_path.glob("*.avi"))
    print(f"Data Validation:")
    print(f"  Directory: {data_directory}")
    print(f"  Video files found: {len(video_files)}")
    
    if len(video_files) == 0:
        print("  ❌ No video files found")
        return False
    
    # Validate file accessibility
    accessible_files = []
    for video_file in video_files:
        if video_file.is_file() and os.access(video_file, os.R_OK):
            accessible_files.append(video_file)
    
    print(f"  Accessible files: {len(accessible_files)}")
    return len(accessible_files) > 0

def validate_configuration(config_path):
    """Validate configuration file structure and parameters"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Configuration Validation:")
        print(f"  File: {config_path}")
        
        # Check required sections
        required_sections = ['simulation_config', 'data_processing', 'performance_monitoring']
        for section in required_sections:
            if section in config:
                print(f"  ✓ {section} section found")
            else:
                print(f"  ❌ {section} section missing")
                return False
        
        return True
    except Exception as e:
        print(f"  ❌ Configuration validation error: {e}")
        return False

if __name__ == "__main__":
    print("Batch Processing Environment Validation\n")
    
    system_ok = validate_system_requirements()
    data_ok = validate_data_availability("data/normalized")
    config_ok = validate_configuration("config/example_config.json")
    
    print(f"\nValidation Summary:")
    print(f"  System Requirements: {'✓' if system_ok else '❌'}")
    print(f"  Data Availability: {'✓' if data_ok else '❌'}")
    print(f"  Configuration: {'✓' if config_ok else '❌'}")
    
    if all([system_ok, data_ok, config_ok]):
        print("\n✅ Environment ready for batch processing")
        sys.exit(0)
    else:
        print("\n❌ Environment validation failed")
        sys.exit(1)
```

### Running Your First Batch

Complete example demonstrating execution of 100 simulations using `simple_batch_simulation.py` with infotaxis, casting, and gradient_following algorithms:

```python
#!/usr/bin/env python3
"""
Basic Batch Processing Example

Demonstrates simple batch execution with progress monitoring
and performance validation for plume navigation algorithms.
"""

from pathlib import Path
from src.backend.examples.simple_batch_simulation import SimpleBatchSimulationExample
from src.backend.utils.progress_display import create_progress_bar, TERMINAL_COLORS
import time
import json

def main():
    """Execute basic batch processing workflow"""
    
    # Configuration parameters
    config_path = "config/example_config.json"
    output_dir = "results/basic_batch_example"
    
    print(f"{TERMINAL_COLORS['BLUE']}Starting Basic Batch Processing Example{TERMINAL_COLORS['RESET']}")
    print(f"Configuration: {config_path}")
    print(f"Output Directory: {output_dir}\n")
    
    # Initialize batch simulation example
    batch_example = SimpleBatchSimulationExample(
        config_path=config_path,
        output_directory=output_dir,
        verbose_output=True,
        enable_detailed_logging=True
    )
    
    # Define input data sources
    input_videos = [
        "data/normalized/crimaldi_sample_01.avi",
        "data/normalized/crimaldi_sample_02.avi",
        "data/normalized/custom_sample_01.avi",
        "data/normalized/custom_sample_02.avi",
        "data/normalized/custom_sample_03.avi"
    ]
    
    # Algorithm configuration
    algorithms = ["infotaxis", "casting", "gradient_following"]
    
    print(f"{TERMINAL_COLORS['GREEN']}Batch Configuration:{TERMINAL_COLORS['RESET']}")
    print(f"  Input Videos: {len(input_videos)}")
    print(f"  Algorithms: {len(algorithms)}")
    print(f"  Total Simulations: {len(input_videos) * len(algorithms)}")
    print(f"  Expected Duration: ~{len(input_videos) * len(algorithms) * 7.2:.1f} seconds\n")
    
    # Validate input data before execution
    print(f"{TERMINAL_COLORS['YELLOW']}Validating input data...{TERMINAL_COLORS['RESET']}")
    
    validation_results = batch_example.validate_input_data(
        video_paths=input_videos,
        algorithms=algorithms
    )
    
    if not validation_results["all_valid"]:
        print(f"{TERMINAL_COLORS['RED']}❌ Input validation failed:{TERMINAL_COLORS['RESET']}")
        for issue in validation_results["issues"]:
            print(f"  - {issue}")
        return 1
    
    print(f"{TERMINAL_COLORS['GREEN']}✓ Input validation successful{TERMINAL_COLORS['RESET']}\n")
    
    # Execute complete batch workflow
    print(f"{TERMINAL_COLORS['GREEN']}Executing batch simulation...{TERMINAL_COLORS['RESET']}")
    
    start_time = time.time()
    
    try:
        results = batch_example.run_complete_example(
            input_video_paths=input_videos,
            algorithms_to_test=algorithms,
            enable_progress_monitoring=True,
            save_intermediate_results=True
        )
        
        execution_time = time.time() - start_time
        
        # Analyze execution results
        execution_summary = batch_example.get_execution_summary()
        
        print(f"\n{TERMINAL_COLORS['BOLD']}Batch Execution Summary:{TERMINAL_COLORS['RESET']}")
        print(f"{'='*50}")
        print(f"Total Simulations: {execution_summary['total_simulations']}")
        print(f"Successful Simulations: {execution_summary['successful_simulations']}")
        print(f"Failed Simulations: {execution_summary['failed_simulations']}")
        print(f"Success Rate: {execution_summary['success_rate']:.1%}")
        print(f"Total Execution Time: {execution_time:.2f} seconds")
        print(f"Average Time per Simulation: {execution_summary['average_time']:.2f} seconds")
        print(f"Correlation Accuracy: {execution_summary['correlation_accuracy']:.1%}")
        print(f"Reproducibility Coefficient: {execution_summary['reproducibility_coefficient']:.3f}")
        
        # Performance target validation
        print(f"\n{TERMINAL_COLORS['BOLD']}Performance Target Validation:{TERMINAL_COLORS['RESET']}")
        
        if execution_summary['correlation_accuracy'] >= 0.95:
            print(f"{TERMINAL_COLORS['GREEN']}✓ Correlation target achieved (>95%){TERMINAL_COLORS['RESET']}")
        else:
            print(f"{TERMINAL_COLORS['YELLOW']}⚠ Correlation below target (<95%){TERMINAL_COLORS['RESET']}")
        
        if execution_summary['average_time'] <= 7.2:
            print(f"{TERMINAL_COLORS['GREEN']}✓ Average time target achieved (<7.2s){TERMINAL_COLORS['RESET']}")
        else:
            print(f"{TERMINAL_COLORS['YELLOW']}⚠ Average time above target (>7.2s){TERMINAL_COLORS['RESET']}")
        
        if execution_summary['reproducibility_coefficient'] >= 0.99:
            print(f"{TERMINAL_COLORS['GREEN']}✓ Reproducibility target achieved (>0.99){TERMINAL_COLORS['RESET']}")
        else:
            print(f"{TERMINAL_COLORS['YELLOW']}⚠ Reproducibility below target (<0.99){TERMINAL_COLORS['RESET']}")
        
        # Generate detailed results report
        report_path = batch_example.generate_execution_report(
            include_visualizations=True,
            include_statistical_analysis=True
        )
        
        print(f"\n{TERMINAL_COLORS['CYAN']}Detailed report generated: {report_path}{TERMINAL_COLORS['RESET']}")
        
        return 0
        
    except Exception as e:
        print(f"\n{TERMINAL_COLORS['RED']}❌ Batch execution failed: {e}{TERMINAL_COLORS['RESET']}")
        
        # Generate error report for debugging
        error_report = batch_example.generate_error_report()
        print(f"{TERMINAL_COLORS['CYAN']}Error report generated: {error_report}{TERMINAL_COLORS['RESET']}")
        
        return 1

if __name__ == "__main__":
    exit(main())
```

**Command-Line Execution:**

```bash
# Execute basic batch processing
python examples/basic_batch_processing.py

# Monitor execution with detailed output
python examples/basic_batch_processing.py --verbose --show-progress

# Execute with custom configuration
python examples/basic_batch_processing.py --config config/custom_config.json
```

**Expected Output:**
```
Starting Basic Batch Processing Example
Configuration: config/example_config.json
Output Directory: results/basic_batch_example

Batch Configuration:
  Input Videos: 5
  Algorithms: 3
  Total Simulations: 15
  Expected Duration: ~108.0 seconds

Validating input data...
✓ Input validation successful

Executing batch simulation...
[████████████████████████████████████████████████████] 100% | 15/15 simulations complete

Batch Execution Summary:
==================================================
Total Simulations: 15
Successful Simulations: 15
Failed Simulations: 0
Success Rate: 100.0%
Total Execution Time: 98.45 seconds
Average Time per Simulation: 6.56 seconds
Correlation Accuracy: 96.8%
Reproducibility Coefficient: 0.994

Performance Target Validation:
✓ Correlation target achieved (>95%)
✓ Average time target achieved (<7.2s)
✓ Reproducibility target achieved (>0.99)

Detailed report generated: results/basic_batch_example/execution_report_20240615_143052.html
```

### Understanding Batch Results

Interpretation of batch execution results including performance metrics, success rates, execution statistics, and quality validation with scientific context:

**Result File Structure:**
```
results/basic_batch_example/
├── execution_report_20240615_143052.html    # Comprehensive HTML report
├── summary_statistics.json                  # Statistical summary data
├── performance_metrics.csv                  # Detailed performance data
├── correlation_analysis.json               # Correlation validation results
├── logs/
│   ├── execution.log                       # Detailed execution log
│   ├── performance.log                     # Performance monitoring log
│   └── errors.log                         # Error and warning log
├── data/
│   ├── simulation_results_infotaxis.json  # Algorithm-specific results
│   ├── simulation_results_casting.json
│   └── simulation_results_gradient_following.json
└── analysis/
    ├── performance_comparison.png          # Performance visualization
    ├── correlation_matrix.png             # Correlation analysis
    └── trajectory_analysis.png            # Path analysis visualization
```

**Performance Metrics Interpretation:**

1. **Success Rate Analysis:**
   - Values approaching 100% indicate robust algorithm implementation
   - Rates below 95% may indicate parameter tuning requirements
   - Consistent rates across algorithms suggest proper data normalization

2. **Execution Time Analysis:**
   - Average times below 7.2 seconds meet performance targets
   - High variance may indicate optimization opportunities
   - Consistent timing across runs validates reproducibility

3. **Correlation Accuracy:**
   - Values above 95% confirm algorithm implementation correctness
   - Lower values require investigation of algorithm parameters
   - Consistent correlation validates scientific reproducibility

4. **Quality Validation Metrics:**
   - Reproducibility coefficients above 0.99 ensure reliable results
   - Statistical significance testing validates scientific relevance
   - Cross-platform consistency confirms implementation robustness

**Result Analysis Example:**

```python
#!/usr/bin/env python3
"""
Batch Results Analysis Example

Demonstrates comprehensive analysis of batch processing results
with statistical validation and performance assessment.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_batch_results(results_directory):
    """Analyze batch processing results with statistical validation"""
    
    results_path = Path(results_directory)
    
    # Load summary statistics
    with open(results_path / "summary_statistics.json", 'r') as f:
        summary_stats = json.load(f)
    
    # Load detailed performance metrics
    performance_df = pd.read_csv(results_path / "performance_metrics.csv")
    
    # Load correlation analysis
    with open(results_path / "correlation_analysis.json", 'r') as f:
        correlation_data = json.load(f)
    
    print("Batch Results Analysis")
    print("=" * 40)
    
    # Overall performance summary
    print(f"\nOverall Performance:")
    print(f"  Total Simulations: {summary_stats['total_simulations']}")
    print(f"  Success Rate: {summary_stats['success_rate']:.1%}")
    print(f"  Average Time: {summary_stats['average_execution_time']:.2f} seconds")
    print(f"  Standard Deviation: {summary_stats['time_std_deviation']:.2f} seconds")
    
    # Algorithm-specific analysis
    print(f"\nAlgorithm Performance Comparison:")
    algorithm_stats = performance_df.groupby('algorithm').agg({
        'execution_time': ['mean', 'std', 'min', 'max'],
        'success': 'mean',
        'correlation_score': 'mean'
    }).round(3)
    
    print(algorithm_stats)
    
    # Quality validation results
    print(f"\nQuality Validation:")
    print(f"  Overall Correlation: {correlation_data['overall_correlation']:.3f}")
    print(f"  Reproducibility Coefficient: {correlation_data['reproducibility_coefficient']:.3f}")
    print(f"  Statistical Significance: p = {correlation_data['statistical_significance']:.4f}")
    
    # Performance target assessment
    print(f"\nPerformance Target Assessment:")
    
    target_correlation = 0.95
    target_avg_time = 7.2
    target_reproducibility = 0.99
    
    correlation_met = correlation_data['overall_correlation'] >= target_correlation
    time_met = summary_stats['average_execution_time'] <= target_avg_time
    reproducibility_met = correlation_data['reproducibility_coefficient'] >= target_reproducibility
    
    print(f"  Correlation Target (≥{target_correlation}): {'✓' if correlation_met else '❌'}")
    print(f"  Time Target (≤{target_avg_time}s): {'✓' if time_met else '❌'}")
    print(f"  Reproducibility Target (≥{target_reproducibility}): {'✓' if reproducibility_met else '❌'}")
    
    # Generate recommendations
    print(f"\nRecommendations:")
    
    if not correlation_met:
        print("  - Review algorithm implementation and parameter settings")
        print("  - Validate input data quality and normalization")
    
    if not time_met:
        print("  - Consider parallel processing optimization")
        print("  - Review algorithm parameter efficiency")
    
    if not reproducibility_met:
        print("  - Verify consistent random seed usage")
        print("  - Check for non-deterministic algorithm components")
    
    if all([correlation_met, time_met, reproducibility_met]):
        print("  - All targets achieved - configuration suitable for large-scale studies")
        print("  - Consider scaling to full 4000+ simulation batches")
    
    return {
        'targets_met': all([correlation_met, time_met, reproducibility_met]),
        'summary_stats': summary_stats,
        'correlation_data': correlation_data,
        'recommendations': []
    }

if __name__ == "__main__":
    results = analyze_batch_results("results/basic_batch_example")
```

## Advanced Batch Processing with BatchExecutor

### BatchExecutor Configuration

Comprehensive configuration of the `BatchExecutor` class for large-scale batch processing with parallel execution, resource optimization, and performance monitoring:

**Advanced Configuration Template:**

```json
{
  "batch_executor_config": {
    "execution_parameters": {
      "batch_size": 4000,
      "target_completion_hours": 8.0,
      "max_retries": 3,
      "checkpoint_interval": 100,
      "enable_resume": true
    },
    "parallel_processing": {
      "enabled": true,
      "max_workers": 8,
      "worker_allocation_strategy": "dynamic",
      "load_balancing": "least_loaded",
      "worker_timeout_seconds": 300,
      "memory_per_worker_gb": 1.0
    },
    "resource_management": {
      "memory_limit_gb": 8.0,
      "cpu_utilization_target": 0.85,
      "io_optimization": true,
      "enable_memory_mapping": true,
      "garbage_collection_frequency": 50
    },
    "performance_optimization": {
      "enable_caching": true,
      "cache_size_mb": 512,
      "preload_algorithms": true,
      "optimize_data_loading": true,
      "enable_jit_compilation": true
    },
    "monitoring_and_logging": {
      "enable_real_time_monitoring": true,
      "performance_sampling_interval": 10,
      "detailed_logging": true,
      "log_level": "INFO",
      "enable_profiling": false
    }
  },
  "performance_targets": {
    "average_simulation_time_seconds": 7.2,
    "correlation_threshold": 0.95,
    "success_rate_target": 0.99,
    "reproducibility_threshold": 0.99,
    "memory_efficiency_target": 0.8
  },
  "quality_assurance": {
    "enable_validation": true,
    "validation_sample_rate": 0.1,
    "statistical_validation": true,
    "cross_reference_validation": true,
    "fail_fast_on_errors": false
  }
}
```

**BatchExecutor Initialization:**

```python
#!/usr/bin/env python3
"""
Advanced BatchExecutor Configuration Example

Demonstrates comprehensive configuration and initialization of
BatchExecutor for large-scale simulation processing.
"""

from src.backend.core.simulation.batch_executor import (
    create_batch_executor, BatchExecutionContext, BatchExecutorConfig
)
from src.backend.utils.resource_management import ResourceManager
from src.backend.utils.performance_monitoring import PerformanceMonitor
import json
import logging

def create_optimized_batch_executor(config_path, executor_id="advanced_batch"):
    """Create and configure optimized BatchExecutor for large-scale processing"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    batch_config = config["batch_executor_config"]
    performance_targets = config["performance_targets"]
    
    # Setup logging for batch execution
    logging.basicConfig(
        level=getattr(logging, batch_config["monitoring_and_logging"]["log_level"]),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/{executor_id}_execution.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(f"BatchExecutor.{executor_id}")
    
    # Initialize resource manager
    resource_manager = ResourceManager(
        memory_limit_gb=batch_config["resource_management"]["memory_limit_gb"],
        cpu_utilization_target=batch_config["resource_management"]["cpu_utilization_target"],
        enable_monitoring=True
    )
    
    # Initialize performance monitor
    performance_monitor = PerformanceMonitor(
        sampling_interval=batch_config["monitoring_and_logging"]["performance_sampling_interval"],
        enable_profiling=batch_config["monitoring_and_logging"]["enable_profiling"],
        performance_targets=performance_targets
    )
    
    # Create executor configuration
    executor_config = BatchExecutorConfig(
        executor_id=executor_id,
        parallel_processing=batch_config["parallel_processing"],
        resource_management=batch_config["resource_management"],
        performance_optimization=batch_config["performance_optimization"],
        quality_assurance=config["quality_assurance"]
    )
    
    # Create batch executor with optimization
    executor = create_batch_executor(
        executor_id=executor_id,
        executor_config=executor_config,
        resource_manager=resource_manager,
        performance_monitor=performance_monitor,
        enable_parallel_execution=True,
        enable_resource_optimization=True,
        enable_performance_monitoring=True
    )
    
    logger.info(f"BatchExecutor '{executor_id}' created with configuration:")
    logger.info(f"  Max Workers: {batch_config['parallel_processing']['max_workers']}")
    logger.info(f"  Memory Limit: {batch_config['resource_management']['memory_limit_gb']}GB")
    logger.info(f"  Target Batch Size: {batch_config['execution_parameters']['batch_size']}")
    logger.info(f"  Target Completion: {batch_config['execution_parameters']['target_completion_hours']} hours")
    
    return executor, resource_manager, performance_monitor

def configure_executor_optimization(executor, optimization_config):
    """Configure advanced optimization settings for BatchExecutor"""
    
    # Configure parallel processing optimization
    executor.configure_parallel_processing(
        worker_allocation_strategy=optimization_config["worker_allocation_strategy"],
        load_balancing_algorithm=optimization_config["load_balancing"],
        dynamic_scaling=True,
        worker_health_monitoring=True
    )
    
    # Configure memory optimization
    executor.configure_memory_optimization(
        enable_memory_mapping=optimization_config["enable_memory_mapping"],
        cache_optimization=optimization_config["enable_caching"],
        garbage_collection_strategy="adaptive"
    )
    
    # Configure I/O optimization
    executor.configure_io_optimization(
        async_file_operations=True,
        read_ahead_buffering=True,
        write_batching=True
    )
    
    return executor

if __name__ == "__main__":
    # Example usage
    executor, resource_manager, performance_monitor = create_optimized_batch_executor(
        config_path="config/advanced_batch_config.json",
        executor_id="large_scale_batch"
    )
    
    print(f"BatchExecutor configured and ready for large-scale processing")
    print(f"Resource monitoring: {resource_manager.is_monitoring_enabled()}")
    print(f"Performance tracking: {performance_monitor.is_enabled()}")
```

### Large-Scale Batch Execution

Example demonstrating execution of 4000+ simulations using BatchExecutor with parallel processing, checkpoint management, and performance monitoring:

```python
#!/usr/bin/env python3
"""
Large-Scale Batch Execution Example

Demonstrates execution of 4000+ simulations using BatchExecutor with
parallel processing, checkpoint management, progress tracking, and
comprehensive performance monitoring.
"""

from src.backend.core.simulation.batch_executor import (
    create_batch_executor, BatchExecutionContext
)
from src.backend.utils.progress_display import (
    create_progress_bar, display_batch_summary, TERMINAL_COLORS
)
from src.backend.algorithms.algorithm_registry import list_algorithms
from src.backend.utils.checkpoint_manager import CheckpointManager
import time
import json
import logging
from pathlib import Path

def prepare_large_scale_simulation_tasks(batch_size=4000):
    """Prepare simulation tasks for large-scale batch execution"""
    
    # Algorithm distribution for comprehensive testing
    algorithms = ["infotaxis", "casting", "gradient_following", "plume_tracking", "hybrid_strategies"]
    simulations_per_algorithm = batch_size // len(algorithms)
    
    # Data sources for batch processing
    data_sources = [
        "data/normalized/crimaldi_dataset_01.avi",
        "data/normalized/crimaldi_dataset_02.avi",
        "data/normalized/crimaldi_dataset_03.avi",
        "data/normalized/custom_dataset_01.avi",
        "data/normalized/custom_dataset_02.avi",
        "data/normalized/custom_dataset_03.avi",
        "data/normalized/custom_dataset_04.avi",
        "data/normalized/custom_dataset_05.avi"
    ]
    
    simulation_tasks = []
    task_id = 0
    
    for algorithm in algorithms:
        for i in range(simulations_per_algorithm):
            data_source = data_sources[i % len(data_sources)]
            
            simulation_tasks.append({
                "task_id": f"{algorithm}_{task_id:05d}",
                "algorithm": algorithm,
                "input_data": data_source,
                "algorithm_parameters": {
                    "random_seed": task_id,  # Ensure reproducibility
                    "max_steps": 1000,
                    "algorithm_specific": get_algorithm_parameters(algorithm)
                },
                "simulation_metadata": {
                    "batch_id": "large_scale_001",
                    "task_index": task_id,
                    "priority": "normal"
                }
            })
            task_id += 1
    
    # Add remaining tasks to reach exact batch size
    remaining_tasks = batch_size - len(simulation_tasks)
    for i in range(remaining_tasks):
        algorithm = algorithms[i % len(algorithms)]
        data_source = data_sources[i % len(data_sources)]
        
        simulation_tasks.append({
            "task_id": f"{algorithm}_{task_id:05d}",
            "algorithm": algorithm,
            "input_data": data_source,
            "algorithm_parameters": {
                "random_seed": task_id,
                "max_steps": 1000,
                "algorithm_specific": get_algorithm_parameters(algorithm)
            },
            "simulation_metadata": {
                "batch_id": "large_scale_001",
                "task_index": task_id,
                "priority": "normal"
            }
        })
        task_id += 1
    
    return simulation_tasks

def get_algorithm_parameters(algorithm_name):
    """Get algorithm-specific parameters for optimal performance"""
    
    algorithm_params = {
        "infotaxis": {
            "lambda": 0.1,
            "step_size": 1.0,
            "exploration_noise": 0.01
        },
        "casting": {
            "cast_width": 10.0,
            "surge_distance": 20.0,
            "turn_angle": 45.0
        },
        "gradient_following": {
            "step_size": 1.0,
            "gradient_threshold": 0.01,
            "smoothing_factor": 0.1
        },
        "plume_tracking": {
            "tracking_sensitivity": 0.05,
            "memory_decay": 0.9,
            "adaptation_rate": 0.1
        },
        "hybrid_strategies": {
            "strategy_weights": [0.3, 0.3, 0.4],
            "switching_threshold": 0.1,
            "adaptation_enabled": True
        }
    }
    
    return algorithm_params.get(algorithm_name, {})

def main():
    """Execute large-scale batch processing with comprehensive monitoring"""
    
    # Batch configuration for 4000+ simulations
    batch_config = {
        "batch_size": 4000,
        "target_completion_hours": 8.0,
        "parallel_processing": {
            "enabled": True,
            "max_workers": 8,
            "load_balancing": "dynamic",
            "worker_timeout": 300
        },
        "performance_targets": {
            "average_simulation_time_seconds": 7.2,
            "correlation_threshold": 0.95,
            "success_rate_target": 0.99,
            "memory_efficiency_target": 0.8
        },
        "resource_optimization": {
            "memory_limit_gb": 8.0,
            "enable_checkpoints": True,
            "checkpoint_interval": 100,
            "enable_resume": True
        },
        "quality_assurance": {
            "enable_validation": True,
            "validation_frequency": 500,
            "fail_fast_threshold": 0.05
        }
    }
    
    print(f"{TERMINAL_COLORS['BLUE']}Large-Scale Batch Processing Example{TERMINAL_COLORS['RESET']}")
    print(f"Target: {batch_config['batch_size']} simulations in {batch_config['target_completion_hours']} hours")
    print(f"Performance target: <{batch_config['performance_targets']['average_simulation_time_seconds']}s average per simulation\n")
    
    # Create optimized batch executor
    executor = create_batch_executor(
        executor_id="large_scale_batch_001",
        executor_config=batch_config,
        enable_parallel_execution=True,
        enable_resource_optimization=True,
        enable_checkpoint_management=True
    )
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_directory="checkpoints/large_scale_batch_001",
        checkpoint_interval=batch_config["resource_optimization"]["checkpoint_interval"],
        enable_compression=True
    )
    
    # Prepare simulation tasks
    print(f"{TERMINAL_COLORS['YELLOW']}Preparing simulation tasks...{TERMINAL_COLORS['RESET']}")
    simulation_tasks = prepare_large_scale_simulation_tasks(batch_config["batch_size"])
    
    print(f"Simulation tasks prepared:")
    print(f"  Total tasks: {len(simulation_tasks)}")
    print(f"  Algorithms: {len(set(task['algorithm'] for task in simulation_tasks))}")
    print(f"  Data sources: {len(set(task['input_data'] for task in simulation_tasks))}\n")
    
    # Execute batch with comprehensive context management
    with BatchExecutionContext(
        context_name="large_scale_processing",
        batch_executor=executor,
        context_config=batch_config,
        checkpoint_manager=checkpoint_manager
    ) as batch_context:
        
        # Create comprehensive progress monitoring
        progress_bar = create_progress_bar(
            bar_id="large_scale_batch_progress",
            total_items=len(simulation_tasks),
            description="Large-Scale Batch Processing",
            show_eta=True,
            show_rate=True,
            show_performance_metrics=True,
            update_interval=10
        )
        
        # Setup performance monitoring
        performance_metrics = {
            "start_time": time.time(),
            "completed_simulations": 0,
            "failed_simulations": 0,
            "total_execution_time": 0.0,
            "memory_usage_samples": [],
            "cpu_utilization_samples": []
        }
        
        print(f"{TERMINAL_COLORS['GREEN']}Starting large-scale batch execution...{TERMINAL_COLORS['RESET']}")
        
        try:
            # Execute batch simulation with monitoring
            batch_result = batch_context.execute_batch(
                batch_id="large_scale_batch_001",
                simulation_tasks=simulation_tasks,
                batch_config=batch_config,
                progress_callback=lambda completed, total, metrics: update_progress_monitoring(
                    progress_bar, completed, total, metrics, performance_metrics
                ),
                checkpoint_callback=lambda checkpoint_data: save_checkpoint_data(
                    checkpoint_manager, checkpoint_data
                )
            )
            
            total_execution_time = time.time() - performance_metrics["start_time"]
            
            # Comprehensive result analysis
            print(f"\n{TERMINAL_COLORS['BOLD']}Large-Scale Batch Execution Results:{TERMINAL_COLORS['RESET']}")
            print(f"{'='*60}")
            
            # Basic execution statistics
            print(f"Execution Statistics:")
            print(f"  Total Simulations: {batch_result.total_simulations}")
            print(f"  Successful Simulations: {batch_result.successful_simulations}")
            print(f"  Failed Simulations: {batch_result.failed_simulations}")
            print(f"  Success Rate: {batch_result.success_rate:.1%}")
            print(f"  Total Execution Time: {total_execution_time/3600:.2f} hours")
            print(f"  Average Time per Simulation: {batch_result.average_execution_time:.2f} seconds")
            
            # Performance target validation
            target_hours = batch_config["target_completion_hours"]
            target_avg_time = batch_config["performance_targets"]["average_simulation_time_seconds"]
            target_success_rate = batch_config["performance_targets"]["success_rate_target"]
            
            print(f"\nPerformance Target Assessment:")
            
            time_target_met = total_execution_time <= target_hours * 3600
            avg_time_target_met = batch_result.average_execution_time <= target_avg_time
            success_rate_met = batch_result.success_rate >= target_success_rate
            
            print(f"  Completion Time Target (≤{target_hours}h): {'✓' if time_target_met else '❌'} ({total_execution_time/3600:.2f}h)")
            print(f"  Average Time Target (≤{target_avg_time}s): {'✓' if avg_time_target_met else '❌'} ({batch_result.average_execution_time:.2f}s)")
            print(f"  Success Rate Target (≥{target_success_rate:.0%}): {'✓' if success_rate_met else '❌'} ({batch_result.success_rate:.1%})")
            
            # Quality validation
            quality_validation = batch_result.validate_quality_metrics(
                correlation_threshold=batch_config["performance_targets"]["correlation_threshold"]
            )
            
            print(f"\nQuality Validation:")
            print(f"  Overall Correlation: {quality_validation['overall_correlation']:.3f}")
            print(f"  Reproducibility Coefficient: {quality_validation['reproducibility_coefficient']:.3f}")
            print(f"  Statistical Significance: p = {quality_validation['statistical_significance']:.4f}")
            
            correlation_met = quality_validation['overall_correlation'] >= batch_config["performance_targets"]["correlation_threshold"]
            print(f"  Correlation Target: {'✓' if correlation_met else '❌'}")
            
            # Resource utilization summary
            resource_summary = batch_result.get_resource_utilization_summary()
            
            print(f"\nResource Utilization:")
            print(f"  Peak Memory Usage: {resource_summary['peak_memory_gb']:.2f}GB")
            print(f"  Average CPU Utilization: {resource_summary['avg_cpu_utilization']:.1%}")
            print(f"  I/O Efficiency: {resource_summary['io_efficiency']:.1%}")
            
            # Generate comprehensive summary
            display_batch_summary(
                batch_id="large_scale_batch_001",
                total_simulations=batch_result.total_simulations,
                completed_simulations=batch_result.successful_simulations,
                failed_simulations=batch_result.failed_simulations,
                elapsed_time_hours=total_execution_time / 3600,
                performance_summary={
                    "average_time": batch_result.average_execution_time,
                    "success_rate": batch_result.success_rate,
                    "correlation_accuracy": quality_validation['overall_correlation'],
                    "resource_efficiency": resource_summary['overall_efficiency']
                },
                include_recommendations=True
            )
            
            # Overall success assessment
            all_targets_met = all([time_target_met, avg_time_target_met, success_rate_met, correlation_met])
            
            if all_targets_met:
                print(f"\n{TERMINAL_COLORS['GREEN']}✅ All performance targets achieved - batch processing successful{TERMINAL_COLORS['RESET']}")
                return 0
            else:
                print(f"\n{TERMINAL_COLORS['YELLOW']}⚠ Some performance targets not met - review optimization strategies{TERMINAL_COLORS['RESET']}")
                return 1
                
        except Exception as e:
            print(f"\n{TERMINAL_COLORS['RED']}❌ Batch execution failed: {e}{TERMINAL_COLORS['RESET']}")
            
            # Attempt to save partial results
            partial_results = batch_context.get_partial_results()
            if partial_results:
                print(f"{TERMINAL_COLORS['YELLOW']}Saving partial results for analysis...{TERMINAL_COLORS['RESET']}")
                partial_results.save_to_file("results/large_scale_batch_001_partial.json")
            
            return 1

def update_progress_monitoring(progress_bar, completed, total, metrics, performance_metrics):
    """Update progress monitoring with real-time metrics"""
    
    performance_metrics["completed_simulations"] = completed
    
    # Update progress bar with current metrics
    progress_bar.update(
        current_items=completed,
        status_message=f"Processing large-scale batch ({completed}/{total})",
        performance_metrics={
            "avg_time": metrics.get("average_execution_time", 0.0),
            "success_rate": metrics.get("success_rate", 0.0),
            "memory_usage": metrics.get("memory_usage_gb", 0.0),
            "cpu_utilization": metrics.get("cpu_utilization", 0.0)
        }
    )
    
    # Sample performance metrics
    performance_metrics["memory_usage_samples"].append(metrics.get("memory_usage_gb", 0.0))
    performance_metrics["cpu_utilization_samples"].append(metrics.get("cpu_utilization", 0.0))

def save_checkpoint_data(checkpoint_manager, checkpoint_data):
    """Save checkpoint data for batch resumption"""
    
    checkpoint_manager.save_checkpoint(
        checkpoint_id=checkpoint_data["checkpoint_id"],
        checkpoint_data=checkpoint_data,
        include_metadata=True
    )
    
    print(f"Checkpoint saved: {checkpoint_data['checkpoint_id']} ({checkpoint_data['completed_tasks']} tasks completed)")

if __name__ == "__main__":
    exit(main())
```

### Batch Performance Optimization

Advanced optimization techniques for achieving target performance metrics through parallel worker configuration, memory management, and load balancing:

**Performance Optimization Configuration:**

```python
#!/usr/bin/env python3
"""
Batch Performance Optimization Example

Demonstrates advanced optimization techniques including parallel worker
configuration, memory management, load balancing strategies, and
performance tuning to achieve <7.2 seconds average simulation time.
"""

from src.backend.core.simulation.batch_executor import BatchExecutor
from src.backend.utils.performance_optimization import (
    ParallelOptimizer, MemoryOptimizer, LoadBalancer, PerformanceTuner
)
from src.backend.utils.resource_monitoring import ResourceMonitor
import psutil
import multiprocessing
import time

class BatchPerformanceOptimizer:
    """Comprehensive performance optimization for batch processing"""
    
    def __init__(self, target_avg_time=7.2, target_completion_hours=8.0):
        self.target_avg_time = target_avg_time
        self.target_completion_hours = target_completion_hours
        self.resource_monitor = ResourceMonitor()
        
        # System capability assessment
        self.system_capabilities = self._assess_system_capabilities()
        
        # Initialize optimization components
        self.parallel_optimizer = ParallelOptimizer(self.system_capabilities)
        self.memory_optimizer = MemoryOptimizer(self.system_capabilities)
        self.load_balancer = LoadBalancer()
        self.performance_tuner = PerformanceTuner()
    
    def _assess_system_capabilities(self):
        """Assess system capabilities for optimization planning"""
        
        capabilities = {
            "cpu_cores": multiprocessing.cpu_count(),
            "logical_processors": psutil.cpu_count(logical=True),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            "disk_io_capability": self._assess_disk_io_capability()
        }
        
        print(f"System Capabilities Assessment:")
        print(f"  CPU Cores: {capabilities['cpu_cores']}")
        print(f"  Logical Processors: {capabilities['logical_processors']}")
        print(f"  Total Memory: {capabilities['total_memory_gb']:.1f}GB")
        print(f"  Available Memory: {capabilities['available_memory_gb']:.1f}GB")
        print(f"  CPU Frequency: {capabilities['cpu_frequency_mhz']:.0f}MHz")
        print(f"  Disk I/O Rating: {capabilities['disk_io_capability']}")
        
        return capabilities
    
    def _assess_disk_io_capability(self):
        """Assess disk I/O capability for optimization"""
        # Simplified I/O assessment - in practice, would run benchmark
        disk_usage = psutil.disk_usage('/')
        return "SSD" if disk_usage.free > 100 * 1024**3 else "HDD"  # Simplified
    
    def optimize_parallel_processing(self, batch_size, simulation_complexity):
        """Optimize parallel processing configuration"""
        
        print(f"\nOptimizing Parallel Processing Configuration:")
        
        # Calculate optimal worker count
        optimal_workers = self.parallel_optimizer.calculate_optimal_workers(
            batch_size=batch_size,
            simulation_complexity=simulation_complexity,
            target_completion_time=self.target_completion_hours * 3600
        )
        
        # Configure load balancing strategy
        load_balancing_config = self.load_balancer.configure_load_balancing(
            worker_count=optimal_workers,
            task_distribution_strategy="dynamic",
            enable_work_stealing=True
        )
        
        parallel_config = {
            "max_workers": optimal_workers,
            "worker_allocation_strategy": "adaptive",
            "load_balancing": load_balancing_config,
            "worker_timeout": 300,
            "enable_worker_monitoring": True,
            "worker_restart_threshold": 5
        }
        
        print(f"  Optimal Workers: {optimal_workers}")
        print(f"  Load Balancing: {load_balancing_config['strategy']}")
        print(f"  Work Stealing: {load_balancing_config['enable_work_stealing']}")
        
        return parallel_config
    
    def optimize_memory_management(self, batch_size, data_size_per_simulation):
        """Optimize memory management for large-scale processing"""
        
        print(f"\nOptimizing Memory Management:")
        
        # Calculate memory requirements
        total_data_memory = batch_size * data_size_per_simulation
        available_memory = self.system_capabilities["available_memory_gb"]
        
        # Configure memory optimization
        memory_config = self.memory_optimizer.configure_memory_optimization(
            total_data_size_gb=total_data_memory,
            available_memory_gb=available_memory,
            enable_memory_mapping=True,
            enable_swapping=False
        )
        
        print(f"  Memory Mapping: {memory_config['enable_memory_mapping']}")
        print(f"  Cache Size: {memory_config['cache_size_mb']}MB")
        print(f"  Garbage Collection: {memory_config['gc_strategy']}")
        print(f"  Memory Per Worker: {memory_config['memory_per_worker_mb']}MB")
        
        return memory_config
    
    def optimize_io_operations(self, data_characteristics):
        """Optimize I/O operations for video data processing"""
        
        print(f"\nOptimizing I/O Operations:")
        
        io_config = {
            "enable_async_io": True,
            "read_buffer_size_mb": 64,
            "write_buffer_size_mb": 32,
            "enable_read_ahead": True,
            "concurrent_file_operations": 4,
            "enable_compression": False  # For video data
        }
        
        # Adjust based on data characteristics
        if data_characteristics.get("large_files", False):
            io_config["read_buffer_size_mb"] = 128
            io_config["concurrent_file_operations"] = 2
        
        if data_characteristics.get("many_small_files", False):
            io_config["concurrent_file_operations"] = 8
            io_config["read_buffer_size_mb"] = 32
        
        print(f"  Async I/O: {io_config['enable_async_io']}")
        print(f"  Read Buffer: {io_config['read_buffer_size_mb']}MB")
        print(f"  Concurrent Operations: {io_config['concurrent_file_operations']}")
        
        return io_config
    
    def create_optimized_configuration(self, batch_size, simulation_characteristics):
        """Create comprehensive optimized configuration"""
        
        print(f"Creating Optimized Configuration for {batch_size} simulations:")
        
        # Assess simulation complexity
        simulation_complexity = self._assess_simulation_complexity(simulation_characteristics)
        
        # Generate optimization configurations
        parallel_config = self.optimize_parallel_processing(batch_size, simulation_complexity)
        memory_config = self.optimize_memory_management(
            batch_size, 
            simulation_characteristics.get("data_size_per_simulation_mb", 50) / 1024
        )
        io_config = self.optimize_io_operations(simulation_characteristics)
        
        # Calculate performance predictions
        predicted_performance = self._predict_performance(
            batch_size, parallel_config, memory_config, simulation_complexity
        )
        
        optimized_config = {
            "parallel_processing": parallel_config,
            "memory_management": memory_config,
            "io_optimization": io_config,
            "performance_predictions": predicted_performance,
            "monitoring_config": {
                "enable_real_time_monitoring": True,
                "performance_sampling_interval": 10,
                "resource_monitoring_interval": 5,
                "enable_adaptive_optimization": True
            }
        }
        
        print(f"\nPerformance Predictions:")
        print(f"  Estimated Completion Time: {predicted_performance['estimated_completion_hours']:.2f} hours")
        print(f"  Predicted Average Time: {predicted_performance['predicted_avg_time']:.2f} seconds")
        print(f"  Resource Efficiency: {predicted_performance['resource_efficiency']:.1%}")
        
        return optimized_config
    
    def _assess_simulation_complexity(self, characteristics):
        """Assess simulation complexity for optimization planning"""
        
        base_complexity = 1.0
        
        # Adjust based on algorithm complexity
        algorithm_complexity = {
            "infotaxis": 1.2,
            "casting": 0.8,
            "gradient_following": 1.0,
            "plume_tracking": 1.3,
            "hybrid_strategies": 1.5
        }
        
        algorithms = characteristics.get("algorithms", ["infotaxis"])
        avg_algorithm_complexity = sum(algorithm_complexity.get(alg, 1.0) for alg in algorithms) / len(algorithms)
        
        # Adjust based on data characteristics
        data_complexity = 1.0
        if characteristics.get("high_resolution", False):
            data_complexity *= 1.3
        if characteristics.get("long_duration", False):
            data_complexity *= 1.2
        
        total_complexity = base_complexity * avg_algorithm_complexity * data_complexity
        
        return {
            "overall_complexity": total_complexity,
            "algorithm_complexity": avg_algorithm_complexity,
            "data_complexity": data_complexity,
            "complexity_category": self._categorize_complexity(total_complexity)
        }
    
    def _categorize_complexity(self, complexity_score):
        """Categorize complexity for optimization strategy selection"""
        if complexity_score <= 1.0:
            return "low"
        elif complexity_score <= 1.5:
            return "medium"
        else:
            return "high"
    
    def _predict_performance(self, batch_size, parallel_config, memory_config, simulation_complexity):
        """Predict performance based on configuration"""
        
        # Base simulation time estimation
        base_time_per_simulation = 6.0  # seconds
        
        # Adjust for complexity
        complexity_multiplier = simulation_complexity["overall_complexity"]
        adjusted_time = base_time_per_simulation * complexity_multiplier
        
        # Adjust for parallelization efficiency
        workers = parallel_config["max_workers"]
        parallel_efficiency = min(0.95, 0.8 + (workers - 4) * 0.03)  # Diminishing returns
        
        predicted_avg_time = adjusted_time / parallel_efficiency
        
        # Estimate total completion time
        total_simulation_time = batch_size * predicted_avg_time
        parallel_execution_time = total_simulation_time / workers
        overhead_time = batch_size * 0.1  # Setup/teardown overhead
        
        estimated_completion_hours = (parallel_execution_time + overhead_time) / 3600
        
        # Calculate resource efficiency
        cpu_efficiency = min(0.95, workers / self.system_capabilities["cpu_cores"])
        memory_efficiency = memory_config.get("efficiency_score", 0.8)
        resource_efficiency = (cpu_efficiency + memory_efficiency) / 2
        
        return {
            "predicted_avg_time": predicted_avg_time,
            "estimated_completion_hours": estimated_completion_hours,
            "parallel_efficiency": parallel_efficiency,
            "resource_efficiency": resource_efficiency,
            "meets_time_target": predicted_avg_time <= self.target_avg_time,
            "meets_completion_target": estimated_completion_hours <= self.target_completion_hours
        }

def main():
    """Demonstrate batch performance optimization"""
    
    # Initialize optimizer
    optimizer = BatchPerformanceOptimizer(
        target_avg_time=7.2,
        target_completion_hours=8.0
    )
    
    # Simulation characteristics for optimization
    simulation_characteristics = {
        "algorithms": ["infotaxis", "casting", "gradient_following", "plume_tracking"],
        "data_size_per_simulation_mb": 75,
        "high_resolution": True,
        "long_duration": False,
        "complex_algorithms": ["plume_tracking"]
    }
    
    # Create optimized configuration
    optimized_config = optimizer.create_optimized_configuration(
        batch_size=4000,
        simulation_characteristics=simulation_characteristics
    )
    
    # Validate optimization effectiveness
    predictions = optimized_config["performance_predictions"]
    
    print(f"\nOptimization Validation:")
    print(f"  Time Target Met: {'✓' if predictions['meets_time_target'] else '❌'}")
    print(f"  Completion Target Met: {'✓' if predictions['meets_completion_target'] else '❌'}")
    print(f"  Resource Efficiency: {predictions['resource_efficiency']:.1%}")
    
    # Generate optimization recommendations
    if not predictions["meets_time_target"]:
        print(f"\nRecommendations for Time Target:")
        print(f"  - Consider increasing parallel workers")
        print(f"  - Optimize algorithm parameters")
        print(f"  - Enable performance profiling")
    
    if not predictions["meets_completion_target"]:
        print(f"\nRecommendations for Completion Target:")
        print(f"  - Increase system resources")
        print(f"  - Optimize I/O operations")
        print(f"  - Consider batch size reduction")
    
    return optimized_config

if __name__ == "__main__":
    config = main()
```

## Progress Monitoring and Visualization

### Real-Time Progress Tracking

Implementation of real-time progress monitoring using `ProgressBar` and `StatusDisplay` classes with ASCII progress bars, color-coded status indicators, and hierarchical status trees:

```python
#!/usr/bin/env python3
"""
Real-Time Progress Tracking Example

Demonstrates comprehensive real-time progress monitoring using ProgressBar
and StatusDisplay classes with ASCII progress bars, color-coded status
indicators, performance metrics display, and hierarchical status trees.
"""

from src.backend.utils.progress_display import (
    ProgressBar, StatusDisplay, PerformanceMetricsDisplay,
    TERMINAL_COLORS, STATUS_ICONS, create_progress_bar, create_status_display
)
from src.backend.utils.performance_monitoring import collect_system_metrics
import time
import threading
import random
import queue
from datetime import datetime

class RealTimeProgressMonitor:
    """Comprehensive real-time progress monitoring system"""
    
    def __init__(self, monitoring_config):
        self.config = monitoring_config
        self.progress_bars = {}
        self.status_displays = {}
        self.metrics_displays = {}
        self.monitoring_thread = None
        self.metrics_queue = queue.Queue()
        self.is_monitoring = False
        
    def create_batch_progress_monitor(self, batch_id, total_simulations):
        """Create comprehensive progress monitor for batch processing"""
        
        # Main batch progress bar
        main_progress = create_progress_bar(
            bar_id=f"{batch_id}_main",
            total_items=total_simulations,
            description="Batch Simulation Progress",
            bar_width=60,
            display_config={
                "show_percentage": True,
                "show_eta": True,
                "show_rate": True,
                "show_elapsed": True,
                "color_scheme": "scientific",
                "update_interval": 1.0
            }
        )
        
        # Component status display
        status_display = create_status_display(
            display_id=f"{batch_id}_status",
            title="Simulation Components Status",
            display_config={
                "show_timestamps": True,
                "show_performance_metrics": True,
                "auto_scroll": True,
                "max_visible_items": 20,
                "compact_format": False
            }
        )
        
        # Performance thresholds for monitoring
        performance_thresholds = {
            "simulation_time_seconds": self.config.get("time_threshold", 7.2),
            "memory_usage_gb": self.config.get("memory_threshold", 8.0),
            "cpu_utilization": self.config.get("cpu_threshold", 0.8),
            "success_rate": self.config.get("success_rate_threshold", 0.99),
            "correlation_accuracy": self.config.get("correlation_threshold", 0.95)
        }
        
        # Performance metrics display
        metrics_display = PerformanceMetricsDisplay(
            display_id=f"{batch_id}_metrics",
            performance_thresholds=performance_thresholds,
            display_config={
                "show_trends": True,
                "highlight_violations": True,
                "display_format": "detailed_table",
                "update_frequency": 5.0,
                "history_length": 100
            }
        )
        
        # Store monitoring components
        self.progress_bars[batch_id] = main_progress
        self.status_displays[batch_id] = status_display
        self.metrics_displays[batch_id] = metrics_display
        
        # Initialize component hierarchy
        self._initialize_component_hierarchy(batch_id, status_display)
        
        return {
            "progress_bar": main_progress,
            "status_display": status_display,
            "metrics_display": metrics_display
        }
    
    def _initialize_component_hierarchy(self, batch_id, status_display):
        """Initialize hierarchical component status structure"""
        
        # Level 0: Main processing phases
        main_components = [
            ("data_preparation", "Data Preparation", "READY"),
            ("algorithm_execution", "Algorithm Execution", "PENDING"),
            ("result_analysis", "Result Analysis", "PENDING"),
            ("quality_validation", "Quality Validation", "PENDING")
        ]
        
        for comp_id, comp_name, initial_status in main_components:
            status_display.add_component(
                component_id=f"{batch_id}_{comp_id}",
                component_name=comp_name,
                initial_status=initial_status,
                hierarchy_level=0
            )
        
        # Level 1: Sub-components for algorithm execution
        algorithm_components = [
            ("infotaxis_exec", "Infotaxis Algorithm", "PENDING"),
            ("casting_exec", "Casting Algorithm", "PENDING"),
            ("gradient_exec", "Gradient Following", "PENDING"),
            ("tracking_exec", "Plume Tracking", "PENDING"),
            ("hybrid_exec", "Hybrid Strategies", "PENDING")
        ]
        
        for comp_id, comp_name, initial_status in algorithm_components:
            status_display.add_component(
                component_id=f"{batch_id}_{comp_id}",
                component_name=comp_name,
                initial_status=initial_status,
                hierarchy_level=1,
                parent_component=f"{batch_id}_algorithm_execution"
            )
        
        # Level 2: Worker status for parallel processing
        worker_count = self.config.get("max_workers", 8)
        for worker_id in range(worker_count):
            status_display.add_component(
                component_id=f"{batch_id}_worker_{worker_id}",
                component_name=f"Worker {worker_id + 1}",
                initial_status="IDLE",
                hierarchy_level=2,
                parent_component=f"{batch_id}_algorithm_execution"
            )
    
    def start_monitoring(self, batch_id):
        """Start real-time monitoring for specified batch"""
        
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(batch_id,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        print(f"{TERMINAL_COLORS['GREEN']}Real-time monitoring started for batch: {batch_id}{TERMINAL_COLORS['RESET']}")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        print(f"{TERMINAL_COLORS['YELLOW']}Real-time monitoring stopped{TERMINAL_COLORS['RESET']}")
    
    def _monitoring_loop(self, batch_id):
        """Main monitoring loop for real-time updates"""
        
        update_interval = self.config.get("monitoring_interval", 2.0)
        
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = collect_system_metrics()
                
                # Update metrics display
                if batch_id in self.metrics_displays:
                    metrics_display = self.metrics_displays[batch_id]
                    
                    current_metrics = {
                        "memory_usage_gb": system_metrics.get("memory_usage_gb", 0.0),
                        "cpu_utilization": system_metrics.get("cpu_utilization", 0.0),
                        "simulation_time_seconds": random.uniform(5.0, 9.0),  # Simulated
                        "success_rate": random.uniform(0.95, 1.0),  # Simulated
                        "correlation_accuracy": random.uniform(0.94, 0.98)  # Simulated
                    }
                    
                    violations = metrics_display.update_metrics(
                        new_metrics=current_metrics,
                        trigger_threshold_check=True
                    )
                    
                    # Handle threshold violations
                    if violations:
                        self._handle_threshold_violations(batch_id, violations)
                
                time.sleep(update_interval)
                
            except Exception as e:
                print(f"{TERMINAL_COLORS['RED']}Monitoring error: {e}{TERMINAL_COLORS['RESET']}")
                time.sleep(1.0)
    
    def _handle_threshold_violations(self, batch_id, violations):
        """Handle performance threshold violations"""
        
        for violation in violations:
            violation_type = violation["metric"]
            current_value = violation["current_value"]
            threshold = violation["threshold"]
            
            # Update status display with warnings
            if batch_id in self.status_displays:
                status_display = self.status_displays[batch_id]
                
                warning_message = f"{violation_type} threshold exceeded: {current_value:.2f} > {threshold:.2f}"
                
                status_display.add_message(
                    message=warning_message,
                    message_type="WARNING",
                    timestamp=datetime.now()
                )
    
    def update_batch_progress(self, batch_id, completed_simulations, total_simulations, 
                            current_metrics=None, status_updates=None):
        """Update batch progress with comprehensive information"""
        
        # Update main progress bar
        if batch_id in self.progress_bars:
            progress_bar = self.progress_bars[batch_id]
            
            progress_bar.update(
                current_items=completed_simulations,
                status_message=f"Processing simulation {completed_simulations}/{total_simulations}",
                performance_metrics=current_metrics or {}
            )
        
        # Update component status
        if batch_id in self.status_displays and status_updates:
            status_display = self.status_displays[batch_id]
            
            for component_id, status_data in status_updates.items():
                status_display.update_component_status(
                    component_id=f"{batch_id}_{component_id}",
                    new_status=status_data["status"],
                    status_data=status_data.get("data", {}),
                    status_icon=STATUS_ICONS.get(status_data["status"], "❓")
                )
        
        # Queue metrics for background monitoring
        if current_metrics:
            self.metrics_queue.put({
                "batch_id": batch_id,
                "timestamp": time.time(),
                "metrics": current_metrics
            })
    
    def display_comprehensive_status(self, batch_id):
        """Display comprehensive status information"""
        
        print(f"\n{TERMINAL_COLORS['BOLD']}Comprehensive Status for Batch: {batch_id}{TERMINAL_COLORS['RESET']}")
        print(f"{'='*70}")
        
        # Display current progress
        if batch_id in self.progress_bars:
            progress_bar = self.progress_bars[batch_id]
            progress_info = progress_bar.get_current_status()
            
            print(f"\nProgress Information:")
            print(f"  Completed: {progress_info['completed_items']}/{progress_info['total_items']}")
            print(f"  Percentage: {progress_info['percentage']:.1%}")
            print(f"  Elapsed Time: {progress_info['elapsed_time']:.2f} seconds")
            print(f"  Estimated Remaining: {progress_info['eta']:.2f} seconds")
            print(f"  Processing Rate: {progress_info['rate']:.2f} items/second")
        
        # Display component status tree
        if batch_id in self.status_displays:
            status_display = self.status_displays[batch_id]
            
            print(f"\nComponent Status Tree:")
            status_display.display_status_tree(
                include_performance_data=True,
                compact_format=False
            )
        
        # Display current performance metrics
        if batch_id in self.metrics_displays:
            metrics_display = self.metrics_displays[batch_id]
            
            print(f"\nCurrent Performance Metrics:")
            metrics_display.display_metrics_table(
                include_trends=True,
                highlight_violations=True
            )

def simulate_batch_processing_with_monitoring():
    """Simulate batch processing with comprehensive real-time monitoring"""
    
    # Monitoring configuration
    monitoring_config = {
        "time_threshold": 7.2,
        "memory_threshold": 8.0,
        "cpu_threshold": 0.8,
        "success_rate_threshold": 0.99,
        "correlation_threshold": 0.95,
        "max_workers": 8,
        "monitoring_interval": 2.0
    }
    
    # Initialize progress monitor
    progress_monitor = RealTimeProgressMonitor(monitoring_config)
    
    batch_id = "real_time_monitoring_demo"
    total_simulations = 1000
    
    print(f"{TERMINAL_COLORS['BLUE']}Real-Time Progress Monitoring Demonstration{TERMINAL_COLORS['RESET']}")
    print(f"Batch ID: {batch_id}")
    print(f"Total Simulations: {total_simulations}")
    print(f"Target: Real-time monitoring with threshold violations\n")
    
    # Create monitoring components
    monitoring_components = progress_monitor.create_batch_progress_monitor(
        batch_id=batch_id,
        total_simulations=total_simulations
    )
    
    # Start real-time monitoring
    progress_monitor.start_monitoring(batch_id)
    
    # Simulate batch processing phases
    phases = [
        {
            "name": "data_preparation",
            "duration": 100,
            "description": "Preparing and validating input data",
            "status_updates": {
                "data_preparation": {"status": "RUNNING"}
            }
        },
        {
            "name": "algorithm_execution",
            "duration": 800,
            "description": "Executing navigation algorithms",
            "status_updates": {
                "data_preparation": {"status": "COMPLETED"},
                "algorithm_execution": {"status": "RUNNING"}
            }
        },
        {
            "name": "result_analysis",
            "duration": 80,
            "description": "Analyzing simulation results",
            "status_updates": {
                "algorithm_execution": {"status": "COMPLETED"},
                "result_analysis": {"status": "RUNNING"}
            }
        },
        {
            "name": "quality_validation",
            "duration": 20,
            "description": "Validating result quality",
            "status_updates": {
                "result_analysis": {"status": "COMPLETED"},
                "quality_validation": {"status": "RUNNING"}
            }
        }
    ]
    
    current_simulation = 0
    
    try:
        for phase in phases:
            print(f"\n{TERMINAL_COLORS['GREEN']}Starting phase: {phase['description']}{TERMINAL_COLORS['RESET']}")
            
            # Update phase status
            progress_monitor.update_batch_progress(
                batch_id=batch_id,
                completed_simulations=current_simulation,
                total_simulations=total_simulations,
                status_updates=phase["status_updates"]
            )
            
            # Simulate phase execution
            for i in range(phase["duration"]):
                current_simulation += 1
                
                # Simulate variable processing conditions
                processing_time = random.uniform(4.0, 10.0)
                memory_usage = random.uniform(3.0, 9.0)
                cpu_utilization = random.uniform(0.4, 0.9)
                success_rate = random.uniform(0.93, 1.0)
                correlation = random.uniform(0.92, 0.98)
                
                # Create performance metrics
                current_metrics = {
                    "simulation_time_seconds": processing_time,
                    "memory_usage_gb": memory_usage,
                    "cpu_utilization": cpu_utilization,
                    "success_rate": success_rate,
                    "correlation_accuracy": correlation
                }
                
                # Update progress
                progress_monitor.update_batch_progress(
                    batch_id=batch_id,
                    completed_simulations=current_simulation,
                    total_simulations=total_simulations,
                    current_metrics=current_metrics
                )
                
                # Display comprehensive status every 100 simulations
                if current_simulation % 100 == 0:
                    progress_monitor.display_comprehensive_status(batch_id)
                
                time.sleep(0.02)  # Reduced for demonstration
        
        # Complete final phase
        progress_monitor.update_batch_progress(
            batch_id=batch_id,
            completed_simulations=total_simulations,
            total_simulations=total_simulations,
            status_updates={
                "quality_validation": {"status": "COMPLETED"}
            }
        )
        
        # Display final comprehensive status
        print(f"\n{TERMINAL_COLORS['BOLD']}Final Comprehensive Status{TERMINAL_COLORS['RESET']}")
        progress_monitor.display_comprehensive_status(batch_id)
        
        print(f"\n{TERMINAL_COLORS['GREEN']}✅ Batch processing completed successfully with real-time monitoring{TERMINAL_COLORS['RESET']}")
        
    except KeyboardInterrupt:
        print(f"\n{TERMINAL_COLORS['YELLOW']}Batch processing interrupted by user{TERMINAL_COLORS['RESET']}")
    except Exception as e:
        print(f"\n{TERMINAL_COLORS['RED']}Error during batch processing: {e}{TERMINAL_COLORS['RESET']}")
    finally:
        # Stop monitoring
        progress_monitor.stop_monitoring()

def main():
    """Main function for real-time progress tracking demonstration"""
    try:
        simulate_batch_processing_with_monitoring()
        return 0
    except Exception as e:
        print(f"\n{TERMINAL_COLORS['RED']}Demonstration failed: {e}{TERMINAL_COLORS['RESET']}")
        return 1

if __name__ == "__main__":
    exit(main())
```

### Performance Metrics Dashboard

Creation of performance metrics dashboard using `PerformanceMetricsDisplay` for real-time monitoring of execution time, memory usage, throughput, success rates, and resource utilization:

```python
#!/usr/bin/env python3
"""
Performance Metrics Dashboard Example

Demonstrates creation of comprehensive performance metrics dashboard
for real-time monitoring of batch processing performance with threshold
indicators, trend analysis, and automated alerting.
"""

from src.backend.utils.progress_display import (
    PerformanceMetricsDisplay, create_status_table, TERMINAL_COLORS
)
from src.backend.utils.performance_monitoring import (
    collect_system_metrics, calculate_performance_trends
)
import time
import threading
import numpy as np
from collections import deque
from datetime import datetime, timedelta

class PerformanceMetricsDashboard:
    """Comprehensive performance metrics dashboard for batch processing"""
    
    def __init__(self, dashboard_config):
        self.config = dashboard_config
        self.metrics_history = {}
        self.performance_thresholds = dashboard_config["performance_thresholds"]
        self.dashboard_thread = None
        self.is_running = False
        
        # Initialize metrics tracking
        self._initialize_metrics_tracking()
        
        # Create performance metrics display
        self.metrics_display = PerformanceMetricsDisplay(
            display_id="batch_performance_dashboard",
            performance_thresholds=self.performance_thresholds,
            display_config={
                "show_trends": True,
                "highlight_violations": True,
                "display_format": "comprehensive_dashboard",
                "update_frequency": dashboard_config.get("update_frequency", 5.0),
                "history_length": dashboard_config.get("history_length", 200)
            }
        )
    
    def _initialize_metrics_tracking(self):
        """Initialize metrics tracking with historical data storage"""
        
        metric_categories = [
            "execution_performance",
            "resource_utilization", 
            "quality_metrics",
            "throughput_metrics",
            "system_health"
        ]
        
        for category in metric_categories:
            self.metrics_history[category] = {
                "timestamps": deque(maxlen=self.config.get("history_length", 200)),
                "values": {},
                "trends": {},
                "violations": []
            }
    
    def start_dashboard(self, update_interval=5.0):
        """Start real-time performance metrics dashboard"""
        
        if self.is_running:
            return
        
        self.is_running = True
        self.dashboard_thread = threading.Thread(
            target=self._dashboard_update_loop,
            args=(update_interval,),
            daemon=True
        )
        self.dashboard_thread.start()
        
        print(f"{TERMINAL_COLORS['GREEN']}Performance Metrics Dashboard started{TERMINAL_COLORS['RESET']}")
        print(f"Update interval: {update_interval} seconds")
        print(f"Monitoring {len(self.performance_thresholds)} performance metrics\n")
    
    def stop_dashboard(self):
        """Stop performance metrics dashboard"""
        self.is_running = False
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=10.0)
        
        print(f"{TERMINAL_COLORS['YELLOW']}Performance Metrics Dashboard stopped{TERMINAL_COLORS['RESET']}")
    
    def _dashboard_update_loop(self, update_interval):
        """Main dashboard update loop"""
        
        while self.is_running:
            try:
                # Collect current metrics
                current_metrics = self._collect_comprehensive_metrics()
                
                # Update metrics history
                self._update_metrics_history(current_metrics)
                
                # Calculate trends
                trends = self._calculate_performance_trends()
                
                # Check for threshold violations
                violations = self._check_threshold_violations(current_metrics)
                
                # Update display
                self._update_dashboard_display(current_metrics, trends, violations)
                
                time.sleep(update_interval)
                
            except Exception as e:
                print(f"{TERMINAL_COLORS['RED']}Dashboard update error: {e}{TERMINAL_COLORS['RESET']}")
                time.sleep(1.0)
    
    def _collect_comprehensive_metrics(self):
        """Collect comprehensive performance metrics"""
        
        # System metrics
        system_metrics = collect_system_metrics()
        
        # Simulated batch processing metrics (in real implementation, these would come from BatchExecutor)
        batch_metrics = {
            "current_simulation_time": np.random.normal(6.8, 1.2),
            "completed_simulations": int(time.time() % 1000),  # Simulated progress
            "failed_simulations": max(0, int(np.random.poisson(2))),
            "current_success_rate": np.random.uniform(0.96, 1.0),
            "current_correlation": np.random.uniform(0.94, 0.98),
            "throughput_simulations_per_minute": np.random.uniform(8, 12),
            "queue_length": max(0, int(np.random.poisson(5))),
            "active_workers": min(8, max(4, int(np.random.normal(7, 1))))
        }
        
        # Quality metrics
        quality_metrics = {
            "reproducibility_coefficient": np.random.uniform(0.985, 0.999),
            "statistical_significance": np.random.uniform(0.001, 0.05),
            "cross_validation_score": np.random.uniform(0.92, 0.98),
            "error_rate": max(0, np.random.exponential(0.01))
        }
        
        # Combined metrics
        comprehensive_metrics = {
            **system_metrics,
            **batch_metrics,
            **quality_metrics,
            "timestamp": time.time(),
            "datetime": datetime.now()
        }
        
        return comprehensive_metrics
    
    def _update_metrics_history(self, current_metrics):
        """Update metrics history for trend analysis"""
        
        timestamp = current_metrics["timestamp"]
        
        # Execution performance metrics
        exec_metrics = self.metrics_history["execution_performance"]
        exec_metrics["timestamps"].append(timestamp)
        
        if "values" not in exec_metrics or not exec_metrics["values"]:
            exec_metrics["values"] = {key: deque(maxlen=200) for key in [
                "current_simulation_time", "current_success_rate", "current_correlation"
            ]}
        
        exec_metrics["values"]["current_simulation_time"].append(current_metrics.get("current_simulation_time", 0))
        exec_metrics["values"]["current_success_rate"].append(current_metrics.get("current_success_rate", 0))
        exec_metrics["values"]["current_correlation"].append(current_metrics.get("current_correlation", 0))
        
        # Resource utilization metrics
        resource_metrics = self.metrics_history["resource_utilization"]
        resource_metrics["timestamps"].append(timestamp)
        
        if "values" not in resource_metrics or not resource_metrics["values"]:
            resource_metrics["values"] = {key: deque(maxlen=200) for key in [
                "memory_usage_gb", "cpu_utilization", "active_workers"
            ]}
        
        resource_metrics["values"]["memory_usage_gb"].append(current_metrics.get("memory_usage_gb", 0))
        resource_metrics["values"]["cpu_utilization"].append(current_metrics.get("cpu_utilization", 0))
        resource_metrics["values"]["active_workers"].append(current_metrics.get("active_workers", 0))
        
        # Quality metrics
        quality_metrics = self.metrics_history["quality_metrics"]
        quality_metrics["timestamps"].append(timestamp)
        
        if "values" not in quality_metrics or not quality_metrics["values"]:
            quality_metrics["values"] = {key: deque(maxlen=200) for key in [
                "reproducibility_coefficient", "cross_validation_score", "error_rate"
            ]}
        
        quality_metrics["values"]["reproducibility_coefficient"].append(current_metrics.get("reproducibility_coefficient", 0))
        quality_metrics["values"]["cross_validation_score"].append(current_metrics.get("cross_validation_score", 0))
        quality_metrics["values"]["error_rate"].append(current_metrics.get("error_rate", 0))
        
        # Throughput metrics
        throughput_metrics = self.metrics_history["throughput_metrics"]
        throughput_metrics["timestamps"].append(timestamp)
        
        if "values" not in throughput_metrics or not throughput_metrics["values"]:
            throughput_metrics["values"] = {key: deque(maxlen=200) for key in [
                "throughput_simulations_per_minute", "queue_length"
            ]}
        
        throughput_metrics["values"]["throughput_simulations_per_minute"].append(current_metrics.get("throughput_simulations_per_minute", 0))
        throughput_metrics["values"]["queue_length"].append(current_metrics.get("queue_length", 0))
    
    def _calculate_performance_trends(self):
        """Calculate performance trends for dashboard display"""
        
        trends = {}
        
        for category, history in self.metrics_history.items():
            trends[category] = {}
            
            if len(history["timestamps"]) < 10:
                continue
            
            for metric_name, values in history.get("values", {}).items():
                if len(values) < 10:
                    continue
                
                # Calculate trend direction and magnitude
                values_array = np.array(list(values))
                recent_values = values_array[-10:]  # Last 10 values
                older_values = values_array[-20:-10] if len(values_array) >= 20 else values_array[:-10]
                
                if len(older_values) > 0:
                    recent_avg = np.mean(recent_values)
                    older_avg = np.mean(older_values)
                    
                    trend_direction = "increasing" if recent_avg > older_avg else "decreasing"
                    trend_magnitude = abs(recent_avg - older_avg) / older_avg if older_avg != 0 else 0
                    
                    # Calculate trend strength
                    if trend_magnitude < 0.02:
                        trend_strength = "stable"
                    elif trend_magnitude < 0.05:
                        trend_strength = "slight"
                    elif trend_magnitude < 0.15:
                        trend_strength = "moderate"
                    else:
                        trend_strength = "strong"
                    
                    trends[category][metric_name] = {
                        "direction": trend_direction,
                        "strength": trend_strength,
                        "magnitude": trend_magnitude,
                        "recent_avg": recent_avg,
                        "change_rate": trend_magnitude
                    }
        
        return trends
    
    def _check_threshold_violations(self, current_metrics):
        """Check for performance threshold violations"""
        
        violations = []
        
        for metric_name, threshold in self.performance_thresholds.items():
            current_value = current_metrics.get(metric_name, 0)
            
            # Check different threshold types
            if isinstance(threshold, dict):
                if "max" in threshold and current_value > threshold["max"]:
                    violations.append({
                        "metric": metric_name,
                        "type": "exceeds_maximum",
                        "current_value": current_value,
                        "threshold": threshold["max"],
                        "severity": threshold.get("severity", "warning")
                    })
                
                if "min" in threshold and current_value < threshold["min"]:
                    violations.append({
                        "metric": metric_name,
                        "type": "below_minimum",
                        "current_value": current_value,
                        "threshold": threshold["min"],
                        "severity": threshold.get("severity", "warning")
                    })
            else:
                # Simple threshold (assume maximum)
                if current_value > threshold:
                    violations.append({
                        "metric": metric_name,
                        "type": "exceeds_threshold",
                        "current_value": current_value,
                        "threshold": threshold,
                        "severity": "warning"
                    })
        
        return violations
    
    def _update_dashboard_display(self, current_metrics, trends, violations):
        """Update comprehensive dashboard display"""
        
        # Clear screen for dashboard update (in real implementation, use proper terminal control)
        print("\n" * 2)
        
        # Dashboard header
        print(f"{TERMINAL_COLORS['BOLD']}{TERMINAL_COLORS['BLUE']}Performance Metrics Dashboard{TERMINAL_COLORS['RESET']}")
        print(f"{'='*80}")
        print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Current performance overview
        self._display_performance_overview(current_metrics)
        
        # Resource utilization summary
        self._display_resource_utilization(current_metrics)
        
        # Quality metrics summary
        self._display_quality_metrics(current_metrics)
        
        # Throughput and efficiency
        self._display_throughput_metrics(current_metrics)
        
        # Trend analysis
        self._display_trend_analysis(trends)
        
        # Threshold violations
        if violations:
            self._display_violations(violations)
        else:
            print(f"\n{TERMINAL_COLORS['GREEN']}✅ All metrics within acceptable thresholds{TERMINAL_COLORS['RESET']}")
    
    def _display_performance_overview(self, metrics):
        """Display performance overview section"""
        
        print(f"\n{TERMINAL_COLORS['BOLD']}Performance Overview{TERMINAL_COLORS['RESET']}")
        print(f"{'-'*40}")
        
        # Create performance overview table
        performance_data = [
            {
                "Metric": "Simulation Time",
                "Current": f"{metrics.get('current_simulation_time', 0):.2f}s",
                "Target": "≤7.2s",
                "Status": "✓" if metrics.get('current_simulation_time', 0) <= 7.2 else "⚠"
            },
            {
                "Metric": "Success Rate",
                "Current": f"{metrics.get('current_success_rate', 0):.1%}",
                "Target": "≥99%",
                "Status": "✓" if metrics.get('current_success_rate', 0) >= 0.99 else "⚠"
            },
            {
                "Metric": "Correlation",
                "Current": f"{metrics.get('current_correlation', 0):.3f}",
                "Target": "≥0.95",
                "Status": "✓" if metrics.get('current_correlation', 0) >= 0.95 else "⚠"
            },
            {
                "Metric": "Throughput",
                "Current": f"{metrics.get('throughput_simulations_per_minute', 0):.1f}/min",
                "Target": "≥8.3/min",
                "Status": "✓" if metrics.get('throughput_simulations_per_minute', 0) >= 8.3 else "⚠"
            }
        ]
        
        performance_table = create_status_table(
            table_data=performance_data,
            column_headers=["Metric", "Current", "Target", "Status"],
            column_formats={"Metric": "<15", "Current": ">10", "Target": ">10", "Status": ">6"},
            include_borders=True,
            color_scheme="performance"
        )
        
        print(performance_table)
    
    def _display_resource_utilization(self, metrics):
        """Display resource utilization section"""
        
        print(f"\n{TERMINAL_COLORS['BOLD']}Resource Utilization{TERMINAL_COLORS['RESET']}")
        print(f"{'-'*40}")
        
        memory_usage = metrics.get('memory_usage_gb', 0)
        cpu_utilization = metrics.get('cpu_utilization', 0)
        active_workers = metrics.get('active_workers', 0)
        
        # Memory usage bar
        memory_bar = self._create_usage_bar(memory_usage, 8.0, "Memory")
        print(f"Memory Usage: {memory_bar} {memory_usage:.1f}GB / 8.0GB")
        
        # CPU utilization bar
        cpu_bar = self._create_usage_bar(cpu_utilization, 0.8, "CPU")
        print(f"CPU Usage:    {cpu_bar} {cpu_utilization:.1%} / 80%")
        
        # Worker status
        worker_bar = self._create_usage_bar(active_workers, 8, "Workers")
        print(f"Active Workers: {worker_bar} {active_workers} / 8")
    
    def _display_quality_metrics(self, metrics):
        """Display quality metrics section"""
        
        print(f"\n{TERMINAL_COLORS['BOLD']}Quality Metrics{TERMINAL_COLORS['RESET']}")
        print(f"{'-'*40}")
        
        repro_coeff = metrics.get('reproducibility_coefficient', 0)
        cross_val = metrics.get('cross_validation_score', 0)
        error_rate = metrics.get('error_rate', 0)
        
        print(f"Reproducibility: {repro_coeff:.3f} {'✓' if repro_coeff >= 0.99 else '⚠'}")
        print(f"Cross-Validation: {cross_val:.3f} {'✓' if cross_val >= 0.95 else '⚠'}")
        print(f"Error Rate: {error_rate:.3%} {'✓' if error_rate <= 0.01 else '⚠'}")
    
    def _display_throughput_metrics(self, metrics):
        """Display throughput and efficiency metrics"""
        
        print(f"\n{TERMINAL_COLORS['BOLD']}Throughput & Efficiency{TERMINAL_COLORS['RESET']}")
        print(f"{'-'*40}")
        
        throughput = metrics.get('throughput_simulations_per_minute', 0)
        queue_length = metrics.get('queue_length', 0)
        completed = metrics.get('completed_simulations', 0)
        failed = metrics.get('failed_simulations', 0)
        
        print(f"Current Throughput: {throughput:.1f} simulations/minute")
        print(f"Queue Length: {queue_length} pending simulations")
        print(f"Completed: {completed}, Failed: {failed}")
        
        if completed + failed > 0:
            efficiency = completed / (completed + failed)
            print(f"Efficiency: {efficiency:.1%}")
    
    def _display_trend_analysis(self, trends):
        """Display trend analysis section"""
        
        print(f"\n{TERMINAL_COLORS['BOLD']}Trend Analysis{TERMINAL_COLORS['RESET']}")
        print(f"{'-'*40}")
        
        # Display key trends
        key_metrics = [
            ("execution_performance", "current_simulation_time", "Simulation Time"),
            ("resource_utilization", "memory_usage_gb", "Memory Usage"),
            ("quality_metrics", "current_correlation", "Correlation"),
            ("throughput_metrics", "throughput_simulations_per_minute", "Throughput")
        ]
        
        for category, metric, display_name in key_metrics:
            if category in trends and metric in trends[category]:
                trend_info = trends[category][metric]
                
                direction_icon = "↗" if trend_info["direction"] == "increasing" else "↘"
                strength_color = {
                    "stable": "GREEN",
                    "slight": "YELLOW", 
                    "moderate": "YELLOW",
                    "strong": "RED"
                }.get(trend_info["strength"], "RESET")
                
                print(f"{display_name}: {direction_icon} {trend_info['strength']} {trend_info['direction']} "
                      f"({TERMINAL_COLORS[strength_color]}{trend_info['change_rate']:.1%}{TERMINAL_COLORS['RESET']})")
    
    def _display_violations(self, violations):
        """Display threshold violations"""
        
        print(f"\n{TERMINAL_COLORS['BOLD']}{TERMINAL_COLORS['RED']}⚠ Threshold Violations{TERMINAL_COLORS['RESET']}")
        print(f"{'-'*40}")
        
        for violation in violations:
            severity_color = "RED" if violation["severity"] == "critical" else "YELLOW"
            
            print(f"{TERMINAL_COLORS[severity_color]}• {violation['metric']}: "
                  f"{violation['current_value']:.2f} {violation['type']} "
                  f"(threshold: {violation['threshold']:.2f}){TERMINAL_COLORS['RESET']}")
    
    def _create_usage_bar(self, current_value, max_value, metric_type):
        """Create visual usage bar for metrics"""
        
        if max_value == 0:
            return "[" + "█" * 20 + "]"
        
        usage_ratio = min(1.0, current_value / max_value)
        bar_length = 20
        filled_length = int(bar_length * usage_ratio)
        
        # Color based on usage level
        if usage_ratio < 0.7:
            color = "GREEN"
        elif usage_ratio < 0.9:
            color = "YELLOW"
        else:
            color = "RED"
        
        bar = f"{TERMINAL_COLORS[color]}{'█' * filled_length}{'░' * (bar_length - filled_length)}{TERMINAL_COLORS['RESET']}"
        return f"[{bar}]"

def main():
    """Demonstrate performance metrics dashboard"""
    
    # Dashboard configuration
    dashboard_config = {
        "performance_thresholds": {
            "current_simulation_time": {"max": 7.2, "severity": "warning"},
            "memory_usage_gb": {"max": 8.0, "severity": "critical"},
            "cpu_utilization": {"max": 0.8, "severity": "warning"},
            "current_success_rate": {"min": 0.99, "severity": "warning"},
            "current_correlation": {"min": 0.95, "severity": "critical"},
            "error_rate": {"max": 0.01, "severity": "warning"}
        },
        "update_frequency": 3.0,
        "history_length": 100
    }
    
    # Create and start dashboard
    dashboard = PerformanceMetricsDashboard(dashboard_config)
    
    print(f"{TERMINAL_COLORS['BLUE']}Starting Performance Metrics Dashboard Demonstration{TERMINAL_COLORS['RESET']}")
    print(f"This will run for 30 seconds with live metric updates\n")
    
    try:
        dashboard.start_dashboard(update_interval=3.0)
        
        # Run for demonstration period
        time.sleep(30)
        
        print(f"\n{TERMINAL_COLORS['GREEN']}✅ Dashboard demonstration completed{TERMINAL_COLORS['RESET']}")
        
    except KeyboardInterrupt:
        print(f"\n{TERMINAL_COLORS['YELLOW']}Dashboard demonstration interrupted{TERMINAL_COLORS['RESET']}")
    finally:
        dashboard.stop_dashboard()
    
    return 0

if __name__ == "__main__":
    exit(main())
```

### Terminal Output Formatting

Advanced terminal output formatting using `TERMINAL_COLORS` and `STATUS_ICONS` for professional scientific computing presentation:

```python
#!/usr/bin/env python3
"""
Terminal Output Formatting Example

Demonstrates advanced terminal output formatting using TERMINAL_COLORS
and STATUS_ICONS for professional scientific computing presentation with
color coding schemes and structured output layouts.
"""

from src.backend.utils.progress_display import (
    TERMINAL_COLORS, STATUS_ICONS, create_status_table
)
import time
from datetime import datetime

class ScientificTerminalFormatter:
    """Professional terminal output formatter for scientific computing"""
    
    def __init__(self, color_scheme="scientific"):
        self.color_scheme = color_scheme
        self.output_width = 80
        
        # Define scientific computing color scheme
        self.colors = {
            "success": TERMINAL_COLORS["GREEN"],
            "warning": TERMINAL_COLORS["YELLOW"], 
            "error": TERMINAL_COLORS["RED"],
            "info": TERMINAL_COLORS["BLUE"],
            "path": TERMINAL_COLORS["CYAN"],
            "value": TERMINAL_COLORS["MAGENTA"],
            "bold": TERMINAL_COLORS["BOLD"],
            "reset": TERMINAL_COLORS["RESET"]
        }
        
        # Status indicators for scientific processes
        self.status_indicators = {
            "SUCCESS": f"{self.colors['success']}✓{self.colors['reset']}",
            "FAILURE": f"{self.colors['error']}✗{self.colors['reset']}",
            "WARNING": f"{self.colors['warning']}⚠{self.colors['reset']}",
            "INFO": f"{self.colors['info']}ℹ{self.colors['reset']}",
            "PROCESSING": f"{self.colors['info']}⟳{self.colors['reset']}",
            "COMPLETED": f"{self.colors['success']}✓{self.colors['reset']}",
            "FAILED": f"{self.colors['error']}✗{self.colors['reset']}",
            "PENDING": f"{self.colors['warning']}◐{self.colors['reset']}",
            "RUNNING": f"{self.colors['info']}▶{self.colors['reset']}"
        }
    
    def format_header(self, title, subtitle=None, width=None):
        """Format professional header for scientific applications"""
        
        width = width or self.output_width
        
        # Main title
        title_line = f"{self.colors['bold']}{self.colors['info']}{title}{self.colors['reset']}"
        
        # Separator line
        separator = "=" * width
        
        header_lines = [
            "",
            separator,
            title_line.center(width + len(self.colors['bold']) + len(self.colors['info']) + len(self.colors['reset'])),
            separator
        ]
        
        # Add subtitle if provided
        if subtitle:
            subtitle_line = f"{self.colors['info']}{subtitle}{self.colors['reset']}"
            header_lines.insert(-1, subtitle_line.center(width + len(self.colors['info']) + len(self.colors['reset'])))
            header_lines.insert(-1, "-" * width)
        
        return "\n".join(header_lines)
    
    def format_section(self, section_title, content_lines, indent_level=0):
        """Format section with proper indentation and styling"""
        
        indent = "  " * indent_level
        section_header = f"{indent}{self.colors['bold']}{section_title}:{self.colors['reset']}"
        
        formatted_lines = [section_header]
        
        for line in content_lines:
            formatted_lines.append(f"{indent}  {line}")
        
        return "\n".join(formatted_lines)
    
    def format_metric_display(self, metrics_data, include_thresholds=True):
        """Format metrics display with professional scientific presentation"""
        
        metric_lines = []
        
        for metric_name, metric_info in metrics_data.items():
            value = metric_info["value"]
            unit = metric_info.get("unit", "")
            threshold = metric_info.get("threshold")
            
            # Format value with appropriate precision
            if isinstance(value, float):
                if value < 0.01:
                    formatted_value = f"{value:.4f}"
                elif value < 1.0:
                    formatted_value = f"{value:.3f}"
                elif value < 100:
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:.1f}"
            else:
                formatted_value = str(value)
            
            # Determine status based on threshold
            status_indicator = ""
            if threshold and include_thresholds:
                if isinstance(threshold, dict):
                    if "min" in threshold and value < threshold["min"]:
                        status_indicator = self.status_indicators["FAILURE"]
                    elif "max" in threshold and value > threshold["max"]:
                        status_indicator = self.status_indicators["FAILURE"]
                    else:
                        status_indicator = self.status_indicators["SUCCESS"]
                else:
                    status_indicator = self.status_indicators["SUCCESS"] if value <= threshold else self.status_indicators["FAILURE"]
            
            # Format metric line
            metric_line = f"  {metric_name.replace('_', ' ').title()}: {self.colors['value']}{formatted_value}{unit}{self.colors['reset']}"
            
            if threshold and include_thresholds:
                threshold_text = self._format_threshold_text(threshold)
                metric_line += f" {self.colors['info']}({threshold_text}){self.colors['reset']} {status_indicator}"
            
            metric_lines.append(metric_line)
        
        return "\n".join(metric_lines)
    
    def _format_threshold_text(self, threshold):
        """Format threshold text for display"""
        if isinstance(threshold, dict):
            parts = []
            if "min" in threshold:
                parts.append(f"≥{threshold['min']}")
            if "max" in threshold:
                parts.append(f"≤{threshold['max']}")
            return " & ".join(parts)
        else:
            return f"≤{threshold}"
    
    def format_progress_summary(self, progress_data):
        """Format progress summary with visual indicators"""
        
        total = progress_data["total"]
        completed = progress_data["completed"]
        failed = progress_data.get("failed", 0)
        in_progress = progress_data.get("in_progress", 0)
        
        # Calculate percentages
        completion_rate = (completed / total) * 100 if total > 0 else 0
        failure_rate = (failed / total) * 100 if total > 0 else 0
        
        # Create visual progress bar
        bar_width = 40
        completed_width = int((completed / total) * bar_width) if total > 0 else 0
        failed_width = int((failed / total) * bar_width) if total > 0 else 0
        in_progress_width = int((in_progress / total) * bar_width) if total > 0 else 0
        remaining_width = bar_width - completed_width - failed_width - in_progress_width
        
        progress_bar = (
            f"{self.colors['success']}{'█' * completed_width}{self.colors['reset']}"
            f"{self.colors['error']}{'█' * failed_width}{self.colors['reset']}"
            f"{self.colors['warning']}{'█' * in_progress_width}{self.colors['reset']}"
            f"{'░' * remaining_width}"
        )
        
        progress_lines = [
            f"Progress: [{progress_bar}] {completion_rate:.1f}%",
            f"Completed: {self.colors['success']}{completed}{self.colors['reset']} / {total}",
            f"Failed: {self.colors['error']}{failed}{self.colors['reset']} ({failure_rate:.1f}%)",
            f"In Progress: {self.colors['warning']}{in_progress}{self.colors['reset']}"
        ]
        
        return "\n".join(progress_lines)
    
    def format_file_paths(self, file_paths, max_display=10):
        """Format file paths with proper highlighting and truncation"""
        
        formatted_paths = []
        
        for i, path in enumerate(file_paths[:max_display]):
            formatted_path = f"  {self.colors['path']}{path}{self.colors['reset']}"
            formatted_paths.append(formatted_path)
        
        if len(file_paths) > max_display:
            remaining = len(file_paths) - max_display
            formatted_paths.append(f"  {self.colors['info']}... and {remaining} more files{self.colors['reset']}")
        
        return "\n".join(formatted_paths)
    
    def format_error_report(self, error_data):
        """Format comprehensive error report"""
        
        error_lines = [
            f"{self.colors['error']}{self.colors['bold']}Error Report{self.colors['reset']}",
            f"{'-' * 40}"
        ]
        
        # Error summary
        error_lines.extend([
            f"Error Type: {self.colors['error']}{error_data['type']}{self.colors['reset']}",
            f"Error Message: {error_data['message']}",
            f"Timestamp: {error_data.get('timestamp', datetime.now().isoformat())}"
        ])
        
        # Context information
        if "context" in error_data:
            error_lines.extend([
                "",
                f"{self.colors['bold']}Context Information:{self.colors['reset']}",
                f"  Component: {error_data['context'].get('component', 'Unknown')}",
                f"  Operation: {error_data['context'].get('operation', 'Unknown')}",
                f"  Parameters: {error_data['context'].get('parameters', 'None')}"
            ])
        
        # Stack trace (if available)
        if "stack_trace" in error_data:
            error_lines.extend([
                "",
                f"{self.colors['bold']}Stack Trace:{self.colors['reset']}",
                f"{self.colors['path']}{error_data['stack_trace']}{self.colors['reset']}"
            ])
        
        # Recommendations
        if "recommendations" in error_data:
            error_lines.extend([
                "",
                f"{self.colors['bold']}Recommendations:{self.colors['reset']}"
            ])
            for rec in error_data["recommendations"]:
                error_lines.append(f"  • {rec}")
        
        return "\n".join(error_lines)
    
    def format_algorithm_comparison_table(self, algorithm_results):
        """Format algorithm comparison table with scientific precision"""
        
        # Prepare table data
        table_data = []
        
        for algorithm_name, results in algorithm_results.items():
            table_data.append({
                "Algorithm": algorithm_name.replace("_", " ").title(),
                "Success Rate": f"{results['success_rate']:.1%}",
                "Avg Time (s)": f"{results['average_time']:.2f}",
                "Efficiency": f"{results['efficiency']:.3f}",
                "Correlation": f"{results['correlation']:.3f}",
                "Std Dev": f"{results['std_deviation']:.3f}",
                "Rank": f"{results['rank']}"
            })
        
        # Create formatted table
        formatted_table = create_status_table(
            table_data=table_data,
            column_headers=["Algorithm", "Success Rate", "Avg Time (s)", "Efficiency", "Correlation", "Std Dev", "Rank"],
            column_formats={
                "Algorithm": "<15",
                "Success Rate": ">12",
                "Avg Time (s)": ">12",
                "Efficiency": ">10",
                "Correlation": ">11",
                "Std Dev": ">8",
                "Rank": ">4"
            },
            include_borders=True,
            color_scheme="scientific"
        )
        
        return formatted_table
    
    def format_execution_summary(self, execution_data):
        """Format comprehensive execution summary"""
        
        summary_lines = [
            self.format_header("Execution Summary", "Batch Processing Results"),
            ""
        ]
        
        # Basic statistics
        basic_stats = {
            "total_simulations": {"value": execution_data["total_simulations"], "unit": ""},
            "successful_simulations": {"value": execution_data["successful_simulations"], "unit": ""},
            "failed_simulations": {"value": execution_data["failed_simulations"], "unit": ""},
            "success_rate": {"value": execution_data["success_rate"], "unit": "%", "threshold": {"min": 0.99}},
            "total_execution_time": {"value": execution_data["total_execution_time"], "unit": " hours"},
            "average_simulation_time": {"value": execution_data["average_simulation_time"], "unit": " seconds", "threshold": 7.2}
        }
        
        summary_lines.append(
            self.format_section("Execution Statistics", [
                self.format_metric_display(basic_stats)
            ])
        )
        
        # Performance validation
        performance_metrics = {
            "correlation_accuracy": {"value": execution_data["correlation_accuracy"], "unit": "", "threshold": {"min": 0.95}},
            "reproducibility_coefficient": {"value": execution_data["reproducibility_coefficient"], "unit": "", "threshold": {"min": 0.99}},
            "statistical_significance": {"value": execution_data["statistical_significance"], "unit": "", "threshold": {"max": 0.05}}
        }
        
        summary_lines.extend([
            "",
            self.format_section("Performance Validation", [
                self.format_metric_display(performance_metrics)
            ])
        ])
        
        # Resource utilization
        if "resource_utilization" in execution_data:
            resource_data = execution_data["resource_utilization"]
            resource_metrics = {
                "peak_memory_usage": {"value": resource_data["peak_memory_gb"], "unit": " GB", "threshold": {"max": 8.0}},
                "average_cpu_utilization": {"value": resource_data["avg_cpu_utilization"], "unit": "%", "threshold": {"max": 0.8}},
                "io_efficiency": {"value": resource_data["io_efficiency"], "unit": "%"}
            }
            
            summary_lines.extend([
                "",
                self.format_section("Resource Utilization", [
                    self.format_metric_display(resource_metrics)
                ])
            ])
        
        # Target achievement
        targets_met = []
        if execution_data["success_rate"] >= 0.99:
            targets_met.append(f"{self.status_indicators['SUCCESS']} Success Rate Target")
        else:
            targets_met.append(f"{self.status_indicators['FAILURE']} Success Rate Target")
        
        if execution_data["average_simulation_time"] <= 7.2:
            targets_met.append(f"{self.status_indicators['SUCCESS']} Average Time Target")
        else:
            targets_met.append(f"{self.status_indicators['FAILURE']} Average Time Target")
        
        if execution_data["correlation_accuracy"] >= 0.95:
            targets_met.append(f"{self.status_indicators['SUCCESS']} Correlation Target")
        else:
            targets_met.append(f"{self.status_indicators['FAILURE']} Correlation Target")
        
        summary_lines.extend([
            "",
            self.format_section("Target Achievement", targets_met)
        ])
        
        return "\n".join(summary_lines)

def demonstrate_terminal_formatting():
    """Demonstrate comprehensive terminal formatting capabilities"""
    
    formatter = ScientificTerminalFormatter()
    
    # Example execution data
    execution_data = {
        "total_simulations": 4000,
        "successful_simulations": 3976,
        "failed_simulations": 24,
        "success_rate": 0.994,
        "total_execution_time": 7.8,
        "average_simulation_time": 7.02,
        "correlation_accuracy": 0.967,
        "reproducibility_coefficient": 0.996,
        "statistical_significance": 0.003,
        "resource_utilization": {
            "peak_memory_gb": 7.3,
            "avg_cpu_utilization": 0.78,
            "io_efficiency": 0.89
        }
    }
    
    # Algorithm comparison data
    algorithm_results = {
        "infotaxis": {
            "success_rate": 0.996,
            "average_time": 6.8,
            "efficiency": 0.923,
            "correlation": 0.972,
            "std_deviation": 0.045,
            "rank": 1
        },
        "casting": {
            "success_rate": 0.992,
            "average_time": 6.2,
            "efficiency": 0.887,
            "correlation": 0.965,
            "std_deviation": 0.038,
            "rank": 3
        },
        "gradient_following": {
            "success_rate": 0.989,
            "average_time": 7.1,
            "efficiency": 0.901,
            "correlation": 0.961,
            "std_deviation": 0.052,
            "rank": 2
        }
    }
    
    # Progress data
    progress_data = {
        "total": 4000,
        "completed": 3976,
        "failed": 24,
        "in_progress": 0
    }
    
    # File paths
    output_files = [
        "results/batch_001/execution_summary.json",
        "results/batch_001/performance_metrics.csv",
        "results/batch_001/correlation_analysis.json",
        "results/batch_001/algorithm_comparison.html",
        "results/batch_001/visualization/progress_chart.png",
        "results/batch_001/logs/execution.log"
    ]
    
    print(formatter.format_execution_summary(execution_data))
    
    print(f"\n{formatter.colors['bold']}Algorithm Performance Comparison{formatter.colors['reset']}")
    print(formatter.format_algorithm_comparison_table(algorithm_results))
    
    print(f"\n{formatter.colors['bold']}Batch Progress Summary{formatter.colors['reset']}")
    print(formatter.format_progress_summary(progress_data))
    
    print(f"\n{formatter.colors['bold']}Generated Output Files{formatter.colors['reset']}")
    print(formatter.format_file_paths(output_files))
    
    # Example error report
    error_data = {
        "type": "ParameterValidationError",
        "message": "Invalid algorithm parameter: lambda value out of range",
        "timestamp": datetime.now().isoformat(),
        "context": {
            "component": "InfotaxisAlgorithm",
            "operation": "parameter_validation",
            "parameters": "lambda=1.5 (valid range: 0.0-1.0)"
        },
        "recommendations": [
            "Verify algorithm parameter configuration",
            "Check parameter validation thresholds",
            "Review algorithm documentation for valid ranges"
        ]
    }
    
    print(f"\n{formatter.format_error_report(error_data)}")

def main():
    """Main function for terminal formatting demonstration"""
    
    print(f"{TERMINAL_COLORS['BLUE']}Terminal Output Formatting Demonstration{TERMINAL_COLORS['RESET']}")
    print(f"Showcasing professional scientific computing presentation\n")
    
    demonstrate_terminal_formatting()
    
    print(f"\n{TERMINAL_COLORS['GREEN']}✅ Terminal formatting demonstration completed{TERMINAL_COLORS['RESET']}")
    
    return 0

if __name__ == "__main__":
    exit(main())
```

## Algorithm Comparison Workflows

### Multi-Algorithm Batch Execution

Example workflow using `AlgorithmComparisonStudy` class for executing multiple navigation algorithms with comprehensive comparative analysis: