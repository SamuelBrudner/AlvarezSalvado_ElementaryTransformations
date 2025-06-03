# Getting Started with Plume Navigation Simulation System

## Introduction

### Welcome to the Plume Navigation Simulation System

Welcome to the Plume Navigation Simulation System - a comprehensive scientific computing framework for evaluating olfactory navigation algorithms with cross-format compatibility and batch processing capabilities. This system enables automated normalization and calibration of plume recordings, batch simulation execution with 4000+ simulation capabilities, and real-time analysis with >95% correlation accuracy requirements.

### System Overview

The Plume Navigation Simulation System addresses critical challenges in olfactory navigation research by providing standardized methodologies for processing plume recording data, executing navigation algorithms at scale, and performing comprehensive performance analysis. The system supports both Crimaldi plume datasets and custom AVI recordings with automated format conversion and normalization.

**Core System Capabilities:**
- **Automated Normalization Pipeline**: Handles physical scale differences between recordings including arena sizes, pixel resolutions, temporal sampling rates, and intensity units
- **High-Performance Batch Processing**: Executes 4000+ simulations within 8-hour target timeframe with intelligent resource management
- **Cross-Platform Compatibility**: Processes Crimaldi plume data and custom AVI recordings with automated format conversion
- **Real-Time Performance Analysis**: Calculates navigation metrics with statistical validation against reference implementations
- **Scientific Computing Excellence**: >95% correlation with reference implementations, <7.2 seconds average per simulation

### Target Audience

This guide is designed for:
- **Research scientists** new to the plume navigation simulation system
- **Graduate students** learning computational olfactory navigation
- **Algorithm developers** requiring quick system onboarding
- **Data analysts** needing standardized simulation workflows
- **System administrators** deploying the system for research groups

### Learning Objectives

By the end of this guide, you will be able to:
- ✅ Complete system installation and environment validation
- ✅ Execute first successful plume simulation with quality validation
- ✅ Understand basic workflow from data preparation through analysis
- ✅ Achieve >95% correlation with reference implementations
- ✅ Navigate command-line interface and configuration management
- ✅ Process both Crimaldi and custom plume datasets successfully

## Prerequisites

### System Requirements

**Operating System Requirements:**
- **Linux**: Ubuntu 18.04+ or equivalent distribution
- **macOS**: macOS 10.15+ with Python 3.9+ support
- **Windows**: Windows 10+ with Windows Subsystem for Linux (WSL) recommended

**Hardware Requirements:**
- **Memory**: Minimum 8GB RAM (16GB recommended for batch processing)
- **Storage**: Minimum 5GB free space (additional space required for datasets)
- **CPU**: Multi-core processor recommended for parallel processing optimization
- **Disk Space**: Additional space based on dataset size (typically 10-50GB for research datasets)

**Computational Resources:**
- **Processing Capability**: Support for 4000+ simulations within 8 hours
- **Parallel Processing**: Multi-core CPU for optimal performance
- **Memory Bandwidth**: Sufficient for large video file processing
- **Network Storage**: Compatible with distributed dataset storage

### Knowledge Prerequisites

**Technical Background:**
- **Basic Python**: Familiarity with Python programming and command-line interfaces
- **Scientific Computing**: Understanding of scientific data processing concepts
- **Video Processing**: Basic knowledge of video formats and data structures
- **Research Methodology**: Familiarity with experimental data analysis and validation

**Scientific Domain Knowledge:**
- Understanding of olfactory navigation principles
- Familiarity with experimental plume recording methods
- Basic knowledge of navigation algorithm concepts
- Experience with scientific data analysis workflows

### Data Requirements

**Dataset Prerequisites:**
- **Sample Datasets**: Access to plume recording datasets (Crimaldi or custom format)
- **Calibration Parameters**: Spatial and temporal calibration information for custom datasets
- **Reference Implementations**: Optional reference results for validation and comparison

**Data Format Requirements:**
- **Video Files**: AVI format with appropriate codecs (H264, MJPEG, or uncompressed)
- **Resolution**: Minimum 640x480, recommended 1920x1080 for optimal analysis
- **Frame Rate**: 25-60 fps with automatic normalization support
- **Metadata**: Calibration parameters and experimental documentation

## Installation Guide

### Quick Installation

**Standard Installation using PyPI:**

The quickest way to get started is using pip to install the pre-built package:

```bash
# Create and activate virtual environment (recommended)
python -m venv plume_env
source plume_env/bin/activate  # On Windows: plume_env\Scripts\activate

# Install the plume simulation backend
pip install plume-simulation-backend

# Verify installation
plume-simulation --version
```

**Expected Output:**
```
plume-simulation-backend version 1.0.0
```

**Conda Installation (Alternative):**

For users preferring conda package management:

```bash
# Create conda environment
conda create -n plume-sim python=3.11
conda activate plume-sim

# Install from conda-forge
conda install -c conda-forge plume-simulation-backend

# Verify installation
conda list plume-simulation-backend
```

### Development Installation

**Repository Clone and Setup:**

For development work or accessing the latest features:

```bash
# Clone the repository
git clone https://github.com/research-team/plume-simulation.git
cd plume-simulation

# Create development environment
python -m venv plume_dev_env
source plume_dev_env/bin/activate  # On Windows: plume_dev_env\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r src/backend/requirements.txt

# Verify development installation
python -c "import plume_navigation_backend; print('Development installation successful')"
```

### Environment Validation

**Comprehensive System Check:**

Validate your installation and system capabilities:

```bash
# Run comprehensive system health check
plume-simulation status --detailed

# Expected output should show:
# ✓ All core components initialized successfully
# ✓ Scientific computing libraries validated
# ✓ Video processing capabilities confirmed
# ✓ Parallel processing framework operational
```

**Dependency Verification:**

Verify critical scientific computing dependencies:

```bash
# Test core dependencies
python -c "import numpy, scipy, cv2, joblib; print('Dependencies validated')"

# Check versions for compatibility
python -c "
import numpy as np
import scipy as sp
import cv2
print(f'NumPy: {np.__version__}')
print(f'SciPy: {sp.__version__}')
print(f'OpenCV: {cv2.__version__}')
"
```

**Performance Validation:**

Test system performance against target metrics:

```bash
# Run performance validation test
plume-simulation validate --performance-test

# Target metrics validation:
# ✓ Processing speed: <7.2 seconds per simulation
# ✓ Memory usage: <8GB peak during processing
# ✓ Correlation accuracy: >95% with reference implementations
```

**Troubleshooting Installation Issues:**

If you encounter issues during installation:

```bash
# Clean installation approach
pip uninstall plume-simulation-backend
pip cache purge
pip install --no-cache-dir plume-simulation-backend

# For development environment issues
pip install --upgrade pip setuptools wheel
pip install -e . --force-reinstall

# Check for missing system libraries (Linux/macOS)
sudo apt-get install python3-dev libopencv-dev  # Ubuntu
brew install opencv  # macOS
```

## First Steps

### Configuration Setup

**Default Configuration Initialization:**

The system includes scientifically validated default configurations for immediate use:

```bash
# Generate default configuration files
plume-simulation config init --output config/

# This creates:
# config/default_normalization.json - Data normalization parameters
# config/default_simulation.json - Simulation execution settings
# config/default_analysis.json - Performance analysis configuration
```

**Configuration Validation:**

Always validate your configuration before processing:

```bash
# Validate all configuration files
plume-simulation config validate --all

# Validate specific configuration
plume-simulation config validate --config config/custom_config.json

# View current configuration
plume-simulation config show --type normalization
```

**Custom Configuration Creation:**

For specific research requirements:

```bash
# Copy default configuration for customization
cp config/default_simulation.json config/my_research_config.json

# Edit configuration with your preferred editor
nano config/my_research_config.json

# Validate custom configuration
plume-simulation config validate --config config/my_research_config.json
```

### Sample Data Preparation

**Test Datasets Setup:**

The system includes pre-calibrated sample datasets for immediate testing:

```bash
# Create data directory structure
mkdir -p data/raw/crimaldi/test
mkdir -p data/raw/custom/test
mkdir -p data/normalized
mkdir -p results

# Copy sample datasets (if available in installation)
cp src/test/test_fixtures/crimaldi_sample.avi data/raw/crimaldi/test/
cp src/test/test_fixtures/custom_sample.avi data/raw/custom/test/
```

**Data Organization Best Practices:**

Organize your data using the recommended structure:

```
data/
├── raw/
│   ├── crimaldi/
│   │   ├── experiment_1/
│   │   └── experiment_2/
│   └── custom/
│       ├── arena_setup_1/
│       └── arena_setup_2/
├── normalized/
├── results/
└── analysis/
```

### First Simulation Execution

**Step-by-Step Basic Workflow:**

Execute your first complete simulation workflow:

**Step 1: Data Format Detection and Validation**

```bash
# Detect and validate format
plume-simulation detect-format --input data/raw/crimaldi/test/crimaldi_sample.avi

# Expected output:
# Format: crimaldi
# Confidence: 0.98
# Calibration: available
# Status: ✓ Ready for processing
```

**Step 2: Data Normalization with Quality Validation**

```bash
# Normalize data with comprehensive validation
plume-simulation normalize \
  --input data/raw/crimaldi/test/crimaldi_sample.avi \
  --output data/normalized/ \
  --format crimaldi \
  --validate

# Expected output:
# Normalization completed with quality score: 0.97
# ✓ Spatial accuracy: 99.2%
# ✓ Temporal consistency: 96.8%
# ✓ Intensity calibration: 98.1%
# Output: data/normalized/crimaldi_sample.npz
```

**Step 3: Single Algorithm Simulation Execution**

```bash
# Execute simulation with infotaxis algorithm
plume-simulation simulate \
  --input data/normalized/crimaldi_sample.npz \
  --algorithm infotaxis \
  --output results/ \
  --config config/default_simulation.json

# Expected output:
# Simulation completed in 6.8 seconds
# Success rate: 96.2%
# Correlation accuracy: 96.5%
# ✓ Performance targets met
```

**Step 4: Basic Performance Analysis and Visualization**

```bash
# Generate analysis report
plume-simulation analyze \
  --input results/simulation_results.json \
  --output analysis/ \
  --metrics success_rate,path_efficiency,localization_time \
  --visualizations

# Expected output:
# Analysis completed with statistical validation
# ✓ Success rate: 86.3% ± 2.1%
# ✓ Path efficiency: 0.742 ± 0.035
# ✓ Localization time: 45.2s ± 8.3s
# Visualizations generated: analysis/plots/
```

**Success Validation Checklist:**

Verify your first simulation meets quality requirements:

- ✅ **Normalization quality score >0.95**: Indicates excellent data preparation
- ✅ **Simulation correlation accuracy >0.95**: Meets scientific computing standards
- ✅ **Processing time <7.2 seconds per simulation**: Performance target achieved
- ✅ **Analysis statistical significance p<0.05**: Results statistically valid

## Basic Workflows

### Data Preparation Workflow

**Comprehensive Data Preparation Pipeline:**

Data preparation ensures consistent processing across different experimental conditions and recording formats.

**Format Detection and Validation:**

```bash
# Automatic format detection across directory structure
plume-simulation detect-format \
  --input data/raw/ \
  --recursive \
  --output format_report.json

# Review format compatibility matrix
cat format_report.json | jq '.format_analysis'
```

**Batch Normalization with Progress Monitoring:**

```bash
# Normalize multiple videos with parallel processing
plume-simulation normalize-batch \
  --input data/raw/ \
  --output data/normalized/ \
  --workers 4 \
  --progress \
  --config config/normalization_config.json

# Monitor progress in real-time:
# [NORMALIZING] |████████████████████████████████████████| 100% | 12/12 files
# ├─ Crimaldi Format:    [████████████████████] 100% | 8/8  | Avg: 2.3s
# └─ Custom Format:      [████████████████████] 100% | 4/4  | Avg: 3.1s
# 
# Quality Metrics:
# • Average quality score: 96.8% ± 1.2%
# • Processing efficiency: 94.3%
# • Error rate: 0% (0/12 files)
```

**Quality Validation Against Reference Standards:**

```bash
# Validate normalization quality
plume-simulation validate-quality \
  --normalized data/normalized/ \
  --reference data/reference/ \
  --threshold 0.95

# Quality validation results:
# ✓ All files meet >95% correlation threshold
# ✓ Cross-format consistency verified
# ✓ Scientific accuracy standards maintained
```

### Simulation Execution Workflow

**Algorithm Selection and Configuration:**

Choose appropriate algorithms based on research objectives:

**Available Navigation Algorithms:**
- **infotaxis**: Information-theoretic navigation strategy - optimal for sparse information environments
- **casting**: Bio-inspired crosswind casting behavior - effective for turbulent plume conditions  
- **gradient_following**: Direct concentration gradient pursuit - suitable for stable gradient conditions
- **plume_tracking**: Direct plume following strategy - optimal for continuous plume traces
- **hybrid_strategies**: Combined approach algorithms - adaptive strategies for complex environments

**Single Simulation with Detailed Monitoring:**

```bash
# Execute single simulation with comprehensive monitoring
plume-simulation simulate \
  --input data/normalized/video.npz \
  --algorithm infotaxis \
  --config simulation_config.json \
  --output results/single_sim/ \
  --verbose

# Detailed monitoring output:
# Algorithm: infotaxis
# Parameters: step_size=0.1, sensing_radius=0.05
# ⟳ Simulation progress: Step 1250/5000 (25%)
# Current position: (0.45, 0.32)
# Concentration detected: 0.034 ppm
# Information gain: 2.4 bits
# Estimated completion: 4.2 seconds remaining
```

**Batch Simulation with Parallel Processing:**

```bash
# Execute comprehensive batch simulations
plume-simulation batch \
  --input data/normalized/ \
  --algorithms infotaxis,casting,gradient_following \
  --output results/batch_simulation/ \
  --workers 8 \
  --iterations 100 \
  --config config/batch_config.json

# Batch execution monitoring:
# Batch Simulation Progress:
# [████████████████████████████████████████] 100% | 300/300 simulations
# ├─ Infotaxis:          [████████████████████] 100% | 100/100 | Avg: 6.8s
# ├─ Casting:            [████████████████████] 100% | 100/100 | Avg: 7.1s  
# └─ Gradient Following: [████████████████████] 100% | 100/100 | Avg: 6.9s
# 
# Performance Metrics:
# • Overall success rate: 87.3% ± 2.1%
# • Avg processing time: 6.9s (target: <7.2s) ✓
# • Memory usage: 12.3GB / 16.0GB
# • Correlation accuracy: 96.4% (target: >95%) ✓
```

**Algorithm Comparison Study:**

```bash
# Systematic comparison of all available algorithms
plume-simulation compare-algorithms \
  --input data/normalized/video.npz \
  --algorithms all \
  --output comparison/ \
  --statistical-tests \
  --confidence-level 0.95

# Algorithm comparison results:
# 
# Algorithm Performance Summary:
# ┌─────────────────┬─────────────┬──────────────┬─────────────────┐
# │ Algorithm       │ Success Rate│ Path Efficiency│ Avg Time (s)   │
# ├─────────────────┼─────────────┼──────────────┼─────────────────┤
# │ Infotaxis       │ 89.3% ± 3.2%│ 0.78 ± 0.04  │ 6.8 ± 1.2       │
# │ Casting         │ 82.7% ± 4.1%│ 0.65 ± 0.06  │ 7.0 ± 1.8       │
# │ Gradient Follow │ 91.2% ± 2.8%│ 0.84 ± 0.03  │ 6.9 ± 1.0       │
# │ Hybrid Strategy │ 93.1% ± 2.3%│ 0.87 ± 0.02  │ 7.1 ± 0.9       │
# └─────────────────┴─────────────┴──────────────┴─────────────────┘
# 
# Statistical Significance:
# • Hybrid vs Infotaxis: p < 0.001 (highly significant)
# • Gradient vs Casting: p < 0.01 (significant)
```

### Analysis and Visualization Workflow

**Performance Analysis Types:**

**1. Performance Metrics Analysis:**

```bash
# Calculate comprehensive performance metrics
plume-simulation analyze \
  --input results/ \
  --type performance \
  --output analysis/performance/ \
  --metrics success_rate,path_efficiency,localization_time,robustness

# Performance analysis output:
# Performance Metrics Report
# ═══════════════════════════
# 
# Navigation Success:
#   Success Rate: 87.3% ± 2.1% (n=300)
#   Localization Time: 45.2s ± 8.3s
#   95% CI: [44.1s, 46.3s]
# 
# Path Efficiency:
#   Mean Efficiency: 0.742 ± 0.035
#   Optimal Paths: 23.7% of successful trials
#   Path Length Ratio: 1.35 ± 0.18
# 
# Algorithm Robustness:
#   Performance Stability: 94.2%
#   Environmental Adaptation: 0.86
#   Failure Recovery Rate: 78.3%
```

**2. Trajectory Analysis:**

```bash
# Analyze navigation trajectories and search patterns
plume-simulation analyze \
  --input results/ \
  --type trajectory \
  --output analysis/trajectory/ \
  --visualizations \
  --heatmaps

# Trajectory analysis generates:
# • trajectory_plots/: Individual trajectory visualizations
# • heatmaps/: Spatial search pattern analysis
# • efficiency_analysis/: Path optimization metrics
# • temporal_dynamics/: Time-series movement analysis
```

**3. Statistical Comparison with Significance Testing:**

```bash
# Statistical comparison with multiple correction
plume-simulation analyze \
  --input results/ \
  --type statistical \
  --output analysis/statistical/ \
  --baseline infotaxis \
  --correction bonferroni \
  --alpha 0.05

# Statistical Analysis Results:
# ═══════════════════════════
# 
# Pairwise Comparisons (vs Infotaxis):
# ┌─────────────────┬─────────┬────────────┬─────────────┐
# │ Algorithm       │ p-value │ Effect Size│ Significant │
# ├─────────────────┼─────────┼────────────┼─────────────┤
# │ Casting         │ 0.023   │ -0.42      │ Yes*        │
# │ Gradient Follow │ 0.156   │ +0.21      │ No          │
# │ Hybrid Strategy │ 0.001   │ +0.67      │ Yes***      │
# └─────────────────┴─────────┴────────────┴─────────────┘
# 
# Significance levels: * p<0.05, ** p<0.01, *** p<0.001
```

**Automated Report Generation:**

```bash
# Generate comprehensive publication-ready report
plume-simulation report \
  --template scientific_publication \
  --data analysis/ \
  --visualizations \
  --output reports/research_report.pdf \
  --format pdf \
  --dpi 300

# Report includes:
# • Executive summary with key findings
# • Methodology documentation
# • Statistical analysis results
# • Publication-quality figures
# • Reproducibility information
```

## Command Reference

### Core Commands

**Main Command-Line Interface:**

The `plume-simulation` command provides comprehensive access to all system capabilities:

```bash
# Global command structure
plume-simulation [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]

# Global options available for all commands:
--verbose, -v          # Increase output verbosity (use -vv, -vvv for more detail)
--config, -c CONFIG    # Specify custom configuration file
--output-dir, -o DIR   # Set output directory for results
--workers, -w N        # Number of parallel workers for processing
--no-color             # Disable color-coded console output
--no-progress          # Disable progress bars and real-time counters
```

### Data Processing Commands

**normalize - Data Normalization Pipeline:**

```bash
# Basic normalization
plume-simulation normalize INPUT --output OUTPUT --format FORMAT

# Advanced normalization with all options
plume-simulation normalize data/raw/video.avi \
  --output data/normalized/ \
  --format crimaldi \
  --config config/norm_config.json \
  --validate \
  --parallel \
  --quality-threshold 0.95

# Batch normalization
plume-simulation normalize-batch \
  --input data/raw/ \
  --output data/normalized/ \
  --workers 4 \
  --config config/batch_norm.json \
  --progress-bar
```

**Key Normalization Options:**
- `--format {crimaldi,custom,auto}`: Input format specification
- `--validate`: Enable comprehensive quality validation
- `--parallel`: Enable parallel processing for multiple files
- `--quality-threshold FLOAT`: Minimum quality score threshold (default: 0.95)
- `--calibration FILE`: Custom calibration parameters file

### Simulation Execution Commands

**simulate - Execute Navigation Algorithms:**

```bash
# Single algorithm simulation
plume-simulation simulate normalized_data/ \
  --algorithm infotaxis \
  --output results/ \
  --config config/sim_config.json \
  --iterations 100

# Multiple algorithm comparison
plume-simulation simulate normalized_data/ \
  --algorithms infotaxis,casting,gradient_following \
  --output results/multi_algo/ \
  --parameter-grid config/param_grid.json \
  --replications 10

# High-performance batch execution
plume-simulation batch \
  --input data/normalized/ \
  --algorithms all \
  --output results/large_batch/ \
  --workers 16 \
  --max-time 8h \
  --checkpoint-interval 100 \
  --resume-from-checkpoint results/checkpoint.json
```

**Key Simulation Options:**
- `--algorithm ALGO`: Single algorithm name or comma-separated list
- `--iterations N`: Number of simulation iterations per algorithm
- `--parameter-grid FILE`: Parameter sweep configuration
- `--replications N`: Number of replications for statistical reliability
- `--timeout SECONDS`: Maximum time per individual simulation
- `--checkpoint-interval N`: Save progress every N simulations

### Analysis and Reporting Commands

**analyze - Performance Analysis:**

```bash
# Comprehensive performance analysis
plume-simulation analyze results/ \
  --metrics success_rate,path_efficiency,localization_time \
  --statistical-tests \
  --visualizations \
  --output analysis/

# Comparative analysis with baseline
plume-simulation compare \
  --baseline results/infotaxis/ \
  --test results/hybrid/ \
  --significance-level 0.05 \
  --output comparison_report.pdf

# Custom analysis with specific metrics
plume-simulation analyze results/ \
  --metrics custom \
  --metric-config config/custom_metrics.json \
  --export-format csv,json,hdf5 \
  --output analysis/custom/
```

**Analysis Options:**
- `--metrics {all,success_rate,path_efficiency,localization_time,custom}`: Metrics to calculate
- `--statistical-tests`: Enable hypothesis testing and significance analysis
- `--visualizations`: Generate plots and visualizations
- `--baseline DIR`: Baseline results for comparison
- `--export-format {json,csv,hdf5}`: Output format for analysis results

### System Management Commands

**status - System Health Monitoring:**

```bash
# Basic system status
plume-simulation status

# Detailed system diagnostics
plume-simulation status --detailed --performance

# System health check with JSON output
plume-simulation status --health-check --json-output

# Performance benchmarking
plume-simulation benchmark \
  --dataset data/test_dataset.npz \
  --algorithms all \
  --iterations 100 \
  --output benchmark_results.json
```

**Status Information Includes:**
- Core system component health
- Performance metrics and resource utilization
- Configuration validation status
- Dependency verification results
- Hardware capability assessment

**config - Configuration Management:**

```bash
# List available configurations
plume-simulation config list --validate

# Validate specific configuration
plume-simulation config validate config/my_config.json

# Show configuration details
plume-simulation config show --type simulation

# Generate configuration template
plume-simulation config template \
  --output config/new_template.json \
  --include-examples
```

### Progress Visualization

**Real-Time Progress Display:**

The command-line interface provides sophisticated progress visualization with color-coded status indicators:

```bash
# Example of batch simulation progress display:

Batch Simulation Progress:
[████████████████████████████████████████] 100% | 4000/4000 simulations
├─ Infotaxis:        [████████████████████] 100% | 1334/1334 | Avg: 6.8s ✓
├─ Casting:          [██████████████████  ] 90%  | 1200/1334 | Avg: 7.1s ⚠
└─ Gradient Follow:  [████████████        ] 67%  | 894/1334  | Avg: 7.5s ⚠

Performance Metrics:
• Success Rate:     87.3% ± 2.1%          Target: >85% ✓
• Avg Path Length: 12.4m ± 1.8m           
• Localization Time: 45.2s ± 8.3s         Target: <60s ✓
• Memory Usage:     12.3GB / 16.0GB       Usage: 77% ✓
• Correlation Acc:  96.2% ± 0.8%          Target: >95% ✓
• Est. Completion:  2h 15m remaining
```

**Color-Coded Status Indicators:**
- 🟢 **Green**: Successful operations, targets met
- 🟡 **Yellow**: Warnings, non-critical issues
- 🔴 **Red**: Errors, failed operations
- 🔵 **Blue**: Information, status updates
- 🔷 **Cyan**: File paths, configuration values

## Configuration Management

### Configuration Files Structure

**Primary Configuration Files:**

The system uses three main configuration files for different operational aspects:

**1. Normalization Configuration (`config/normalization.json`):**

```json
{
  "normalization": {
    "spatial": {
      "target_resolution": [640, 480],
      "interpolation_method": "bicubic",
      "preserve_aspect_ratio": true,
      "arena_detection_method": "edge_detection"
    },
    "temporal": {
      "target_framerate": 30.0,
      "interpolation_method": "linear",
      "synchronization": "cross_correlation",
      "motion_preservation_threshold": 0.95
    },
    "intensity": {
      "target_range": [0.0, 1.0],
      "calibration_method": "histogram_equalization",
      "background_subtraction": true,
      "noise_reduction": "gaussian"
    },
    "validation": {
      "min_correlation": 0.95,
      "max_error_rate": 0.01,
      "statistical_tests": ["ks_test", "chi_square"]
    }
  }
}
```

**2. Simulation Configuration (`config/simulation.json`):**

```json
{
  "simulation": {
    "execution": {
      "parallel_workers": 16,
      "memory_limit_gb": 32,
      "checkpoint_interval": 100,
      "timeout_seconds": 300
    },
    "algorithms": {
      "infotaxis": {
        "step_size": 0.1,
        "sensing_radius": 0.05,
        "information_threshold": 0.01,
        "exploration_factor": 1.5
      },
      "casting": {
        "step_size": 0.08,
        "casting_angle": 45.0,
        "crosswind_distance": 0.2,
        "success_threshold": 0.05
      },
      "gradient_following": {
        "step_size": 0.09,
        "gradient_threshold": 0.001,
        "adaptation_rate": 0.1,
        "momentum_factor": 0.8
      }
    },
    "performance": {
      "target_simulation_time": 7.2,
      "batch_completion_hours": 8,
      "correlation_accuracy_threshold": 0.95
    }
  }
}
```

**3. Analysis Configuration (`config/analysis.json`):**

```json
{
  "analysis": {
    "metrics": {
      "navigation": [
        "success_rate",
        "path_efficiency", 
        "localization_time",
        "search_coverage"
      ],
      "statistical": {
        "confidence_level": 0.95,
        "significance_threshold": 0.05,
        "effect_size_minimum": 0.2
      }
    },
    "comparison": {
      "baseline_algorithm": "infotaxis",
      "statistical_tests": [
        "t_test",
        "mann_whitney", 
        "anova",
        "kruskal_wallis"
      ]
    },
    "visualization": {
      "trajectory_plots": true,
      "performance_distributions": true,
      "correlation_matrices": true,
      "publication_quality": true
    }
  }
}
```

### Parameter Reference

**Core Processing Parameters:**

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `target_resolution` | tuple | (320,240)-(1920,1080) | (640,480) | Output video resolution |
| `target_framerate` | float | 10.0-120.0 | 30.0 | Normalized frames per second |
| `parallel_workers` | int | 1-64 | CPU count | Parallel simulation workers |
| `memory_limit_gb` | int | 4-128 | 16 | Maximum memory usage |
| `correlation_threshold` | float | 0.90-0.99 | 0.95 | Minimum correlation requirement |

**Algorithm-Specific Parameters:**

| Algorithm | Parameter | Type | Range | Default | Scientific Context |
|-----------|-----------|------|-------|---------|-------------------|
| Infotaxis | `step_size` | float | 0.01-0.5 | 0.1 | Movement distance per time step (meters) |
| Infotaxis | `sensing_radius` | float | 0.01-0.2 | 0.05 | Chemical detection radius (meters) |
| Casting | `casting_angle` | float | 15.0-90.0 | 45.0 | Crosswind casting angle (degrees) |
| Gradient Following | `gradient_threshold` | float | 1e-6-1e-2 | 1e-3 | Minimum gradient for movement |

### Environment Variables

**System Configuration:**

```bash
# Core system paths
export PLUME_SIM_HOME="/opt/plume-simulation"
export PLUME_SIM_DATA="/data/plume-datasets" 
export PLUME_SIM_RESULTS="/results/simulations"
export PLUME_SIM_CONFIG="/etc/plume-sim/config.yaml"

# Performance tuning
export PLUME_SIM_PARALLEL_WORKERS="16"
export PLUME_SIM_MEMORY_LIMIT="32"
export PLUME_SIM_CPU_AFFINITY="0-15"

# Scientific computing
export PLUME_SIM_PRECISION="double"
export PLUME_SIM_RANDOM_SEED="42"
export PLUME_SIM_CORRELATION_THRESHOLD="0.95"

# Logging and debugging
export PLUME_SIM_LOG_LEVEL="INFO"
export PLUME_SIM_DEBUG_MODE="false"
```

### Configuration Validation

**Automated Configuration Validation:**

```bash
# Validate all configuration files
plume-simulation config validate --all

# Validate with strict checking
plume-simulation config validate --strict --file config/production.json

# Generate validation report
plume-simulation config validate --report --output config_validation.json
```

**Configuration Testing:**

```bash
# Test configuration with sample data
plume-simulation test-config \
  --config config/my_config.json \
  --test-data data/sample.avi \
  --validate-performance

# Dry-run with configuration
plume-simulation simulate data/test/ \
  --config config/test_config.json \
  --dry-run \
  --verbose
```

## Troubleshooting

### Common Installation Issues

**Dependency Conflicts:**

*Symptoms*: Import errors, version mismatches during installation
```bash
ImportError: No module named 'cv2'
VersionConflict: plume-simulation-backend 1.0.0 has requirement numpy>=2.1.3, but you have numpy 1.21.0
```

*Solutions*:
```bash
# Create clean virtual environment
python -m venv clean_env
source clean_env/bin/activate

# Update pip and install with no cache
pip install --upgrade pip
pip install --no-cache-dir plume-simulation-backend

# For persistent conflicts
pip uninstall numpy opencv-python scipy
pip install numpy==2.1.3 opencv-python==4.11.0 scipy==1.15.3
pip install plume-simulation-backend
```

**System Library Issues:**

*Symptoms*: OpenCV import errors, missing development headers
```bash
ImportError: libopencv_core.so.4.11: cannot open shared object file
```

*Solutions*:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev libopencv-dev libopencv-contrib-dev

# CentOS/RHEL
sudo yum install python3-devel opencv-devel

# macOS
brew install opencv
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"

# Verify installation
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

### Data Processing Issues

**Video File Format Problems:**

*Issue*: Unsupported video codec or corrupted files
```bash
cv2.error: OpenCV(4.11.0) error: (-2:Unspecified error) in function 'cvReadNextFrameFile'
```

*Diagnosis and Solution*:
```bash
# Check video file integrity
plume-simulation validate-video --input problematic_video.avi --deep-check

# Convert to supported format
ffmpeg -i input_video.mov -c:v libx264 -c:a aac output_video.avi

# Verify conversion success
plume-simulation detect-format --input output_video.avi
```

**Calibration Parameter Issues:**

*Issue*: Invalid or missing calibration parameters
```bash
ValidationError: Calibration parameter 'pixel_to_meter_ratio' must be positive
```

*Solution*:
```bash
# Generate calibration template
plume-simulation generate-calibration \
  --format custom \
  --output calibration_template.json

# Validate calibration parameters
plume-simulation validate-calibration \
  --calibration calibration_parameters.json \
  --format custom

# Auto-detect calibration when possible
plume-simulation extract-calibration \
  --input video_with_metadata.avi \
  --output auto_calibration.json
```

### Performance Issues

**Slow Processing:**

*Symptoms*: Processing time >7.2 seconds per simulation, timeout errors

*Optimization Steps*:
```bash
# Check system performance
plume-simulation benchmark --quick

# Optimize parallel processing
plume-simulation config optimize \
  --workers $(nproc) \
  --memory-limit 16 \
  --output optimized_config.json

# Enable performance monitoring
plume-simulation simulate data/ \
  --algorithm infotaxis \
  --config optimized_config.json \
  --profile-performance \
  --output results/
```

**Memory Exhaustion:**

*Symptoms*: Out of memory errors, system freezing during batch processing

*Solutions*:
```bash
# Reduce batch size and enable memory mapping
plume-simulation batch data/ \
  --algorithms infotaxis \
  --batch-size 10 \
  --memory-map \
  --workers 4 \
  --output results/

# Monitor memory usage
plume-simulation status --performance --memory-details

# Use memory-efficient processing
plume-simulation simulate data/ \
  --algorithm infotaxis \
  --memory-efficient \
  --chunk-size 100 \
  --output results/
```

### Error Recovery

**Checkpoint Recovery:**

When batch processing is interrupted:
```bash
# Resume from last checkpoint
plume-simulation batch data/ \
  --algorithms infotaxis,casting \
  --resume-from-checkpoint results/checkpoint_1000.json \
  --output results/

# Verify checkpoint integrity
plume-simulation validate-checkpoint \
  --checkpoint results/checkpoint_1000.json \
  --verbose
```

**Data Recovery:**

For corrupted or incomplete results:
```bash
# Recover partial results
plume-simulation recover-results \
  --input results/incomplete/ \
  --output results/recovered/ \
  --repair-mode automatic

# Validate recovered data
plume-simulation validate-results \
  --results results/recovered/ \
  --quality-check comprehensive
```

### Getting Help

**Diagnostic Information Collection:**

```bash
# Generate comprehensive diagnostic report
plume-simulation doctor \
  --full-report \
  --output diagnostic_report.json

# System environment information
plume-simulation env-info \
  --detailed \
  --export-format json

# Performance profile for troubleshooting
plume-simulation profile-system \
  --test-dataset data/sample.avi \
  --output performance_profile.json
```

**Support Resources:**

- **Documentation**: Complete user guides at `docs/user_guides/`
- **API Reference**: Detailed API documentation at `docs/api/`
- **Examples**: Working examples at `src/backend/examples/`
- **Community Support**: GitHub issues and discussion forums
- **Professional Support**: Technical consulting and training workshops

## Next Steps

### Immediate Actions

**Complete Your First End-to-End Workflow:**

1. **Validate System Performance**:
```bash
# Run comprehensive system validation
plume-simulation validate --comprehensive --output validation_report.json

# Verify performance targets
plume-simulation benchmark --target-metrics --iterations 100
```

2. **Process Your Own Data**:
```bash
# Prepare your plume recording data
plume-simulation normalize your_data.avi \
  --output normalized_data/ \
  --format auto \
  --validate

# Execute simulation with your data
plume-simulation simulate normalized_data/ \
  --algorithms infotaxis,casting \
  --output your_results/ \
  --config config/your_config.json
```

3. **Generate Research-Quality Analysis**:
```bash
# Create comprehensive analysis
plume-simulation analyze your_results/ \
  --metrics all \
  --statistical-tests \
  --visualizations \
  --output analysis/

# Generate publication-ready report
plume-simulation report \
  --template publication \
  --data analysis/ \
  --output final_report.pdf
```

### Skill Development Path

**1. Master Data Preparation** (Next 1-2 weeks):
- Complete the [Data Preparation Guide](./data_preparation.md)
- Practice with both Crimaldi and custom format datasets
- Achieve consistent >95% normalization quality scores
- Understand calibration parameter optimization

**2. Advanced Simulation Techniques** (Weeks 3-4):
- Explore the [Running Simulations Guide](./running_simulations.md)
- Implement custom algorithm parameters
- Master batch processing for large-scale studies
- Optimize performance for your hardware configuration

**3. Comprehensive Analysis Expertise** (Weeks 5-6):
- Study the [Analyzing Results Guide](./analyzing_results.md)
- Learn advanced statistical analysis techniques
- Create publication-quality visualizations
- Develop custom analysis metrics

**4. Troubleshooting Proficiency** (Ongoing):
- Review the [Troubleshooting Guide](./troubleshooting.md)
- Practice error recovery procedures
- Learn performance optimization techniques
- Contribute to community support

### Advanced Research Applications

**Algorithm Development Projects:**
- Implement custom navigation algorithms using the provided framework
- Conduct comparative studies across different environmental conditions
- Develop adaptive algorithms that switch strategies based on environmental cues
- Integrate machine learning approaches with traditional navigation strategies

**Large-Scale Research Studies:**
- Design experiments utilizing the full 4000+ simulation capacity
- Conduct cross-platform validation studies
- Perform meta-analyses across multiple experimental datasets
- Develop standardized benchmarking protocols for the research community

**Publication and Collaboration:**
- Generate publication-ready results with full reproducibility documentation
- Collaborate with international research groups using standardized workflows
- Contribute to open science initiatives with shared datasets and methodologies
- Develop educational materials for computational olfactory navigation

### Success Criteria Validation

**Technical Competency Checklist:**

- ✅ **System Installation**: Successfully installed with all dependencies verified
- ✅ **Environment Validation**: All health checks pass with optimal performance
- ✅ **First Simulation**: Completed with >95% correlation accuracy in <7.2 seconds
- ✅ **Data Processing**: Successfully processed both Crimaldi and custom formats
- ✅ **Batch Operations**: Executed multi-algorithm batch processing successfully
- ✅ **Configuration Management**: Created and validated custom configurations
- ✅ **Analysis Generation**: Produced statistical analysis with publication-quality outputs
- ✅ **Troubleshooting**: Resolved common issues using provided procedures

**Scientific Standards Validation:**

- ✅ **Correlation Accuracy**: Consistently achieving >95% correlation with references
- ✅ **Processing Performance**: Meeting <7.2 second average simulation time
- ✅ **Statistical Significance**: Generating results with p<0.05 significance levels
- ✅ **Reproducibility**: Achieving >0.99 correlation across different computational environments
- ✅ **Cross-Format Compatibility**: Successfully processing heterogeneous datasets
- ✅ **Quality Assurance**: Implementing comprehensive validation procedures

**Research Readiness Assessment:**

Rate your confidence level (1-5) in each area:
- Data preparation and normalization: ___/5
- Simulation execution and monitoring: ___/5
- Statistical analysis and interpretation: ___/5
- Troubleshooting and optimization: ___/5
- Configuration management: ___/5

**Recommended minimum score: 4/5 in all areas before proceeding to advanced research applications.**

---

## Summary

Congratulations! You have completed the comprehensive getting started guide for the Plume Navigation Simulation System. You now have the knowledge and tools to:

🎯 **Execute high-quality plume simulations** with >95% correlation accuracy
🎯 **Process diverse data formats** with automated normalization and validation
🎯 **Conduct large-scale research studies** using batch processing capabilities  
🎯 **Generate publication-ready analyses** with statistical validation
🎯 **Optimize system performance** for your specific research requirements

**Continue your learning journey** with the specialized user guides:
- [Data Preparation Guide](./data_preparation.md) - Master data normalization techniques
- [Running Simulations Guide](./running_simulations.md) - Advanced simulation strategies  
- [Analyzing Results Guide](./analyzing_results.md) - Comprehensive analysis methods
- [Troubleshooting Guide](./troubleshooting.md) - Expert problem-solving techniques

**Join the research community** and contribute to advancing computational olfactory navigation research with standardized, reproducible, and scientifically rigorous simulation workflows.

---

*This guide represents version 1.0.0 of the Plume Navigation Simulation System documentation. For the latest updates and community contributions, visit the project repository and documentation portal.*