# Plume Navigation Algorithm Simulation Examples

A comprehensive collection of example implementations demonstrating the plume navigation algorithm simulation system for scientific research, algorithm development, and educational purposes. This directory provides production-ready examples showcasing batch simulation execution, cross-format compatibility, algorithm comparison, data normalization, and advanced analysis visualization capabilities.

## Overview

The examples directory serves as the primary entry point for researchers and developers working with the plume navigation algorithm simulation system. These implementations demonstrate essential workflows for processing plume recording data, executing navigation algorithms at scale, and performing comprehensive performance analysis.

### Scientific Purpose and Educational Value

This collection addresses critical challenges in olfactory navigation research by providing standardized methodologies for:

- **Cross-Format Data Processing**: Seamless handling of Crimaldi plume datasets and custom AVI recordings with automated format conversion and normalization
- **Large-Scale Simulation Execution**: Batch processing capabilities supporting 4000+ simulation runs with comprehensive progress monitoring and error handling
- **Algorithm Performance Validation**: Statistical comparison framework ensuring >95% correlation accuracy with reference implementations
- **Reproducible Research Standards**: Implementation of scientific computing best practices with >0.99 reproducibility coefficient across computational environments
- **Educational Workflows**: Step-by-step examples demonstrating complete pipelines from data preparation through publication-ready analysis

### Target Research Applications

- Navigation algorithm development and optimization
- Comparative studies across different experimental conditions
- Performance validation against reference benchmarks
- Educational instruction in computational olfactory navigation
- Cross-platform algorithm testing and validation

## Quick Start Guide

### Environment Requirements

**System Prerequisites:**
- Python 3.9+ with scientific computing libraries
- Minimum 8GB RAM for standard batch operations
- 50GB+ available disk space for plume data and results
- Multi-core CPU recommended for parallel processing optimization

**Required Dependencies:**
```bash
# Core scientific computing stack
numpy>=2.1.3          # Numerical computation framework
scipy>=1.15.3          # Statistical analysis and optimization
pandas>=2.2.0          # Data manipulation and analysis
opencv-python>=4.11.0  # Video processing and computer vision

# Parallel processing and optimization
joblib>=1.6.0          # Memory mapping and process pools
numba>=0.60.0         # Just-in-time compilation for performance

# Visualization and analysis
matplotlib>=3.9.0      # Publication-quality plotting
seaborn>=0.13.2       # Statistical data visualization

# Testing and quality assurance
pytest>=8.3.5         # Unit and integration testing framework
```

### Installation and Setup

1. **Clone the repository and navigate to the examples directory:**
```bash
git clone <repository-url>
cd src/backend/examples
```

2. **Install dependencies using pip:**
```bash
pip install -r requirements.txt
```

3. **Verify installation with environment validation:**
```bash
python -c "import numpy, scipy, cv2, joblib; print('Environment validated successfully')"
```

### Basic Execution

**Run a simple batch simulation:**
```bash
python simple_batch_simulation.py --config config/basic_batch.json --data data/sample_plumes/
```

**Execute cross-format comparison:**
```bash
python cross_format_comparison.py --crimaldi data/crimaldi_dataset/ --custom data/custom_recordings/
```

**Perform algorithm comparison study:**
```bash
python algorithm_comparison.py --algorithms infotaxis,casting,gradient --iterations 1000
```

## Example Descriptions

### Simple Batch Simulation (simple_batch_simulation.py)

**Purpose**: Demonstrates essential batch simulation workflow patterns for educational purposes and rapid prototyping.

**Key Features:**
- Automated batch execution with configurable parameters
- Progress monitoring with real-time performance metrics
- Basic error handling and recovery mechanisms
- Standardized output formats for downstream analysis

**Educational Objectives:**
- Illustrate fundamental simulation pipeline components
- Demonstrate parallel processing optimization techniques
- Show proper error handling and logging practices
- Provide template for custom batch implementations

**Usage Example:**
```bash
python simple_batch_simulation.py \
    --input data/plume_recordings/ \
    --output results/batch_simulation/ \
    --algorithm infotaxis \
    --iterations 500 \
    --parallel-jobs 4
```

**Expected Performance:**
- Processing time: <7.2 seconds average per simulation
- Memory usage: <2GB for standard batch sizes
- Success rate: >99% for validated input data

### Cross-Format Comparison (cross_format_comparison.py)

**Purpose**: Comprehensive demonstration of cross-format algorithm comparison with Crimaldi and custom format processing capabilities.

**Key Features:**
- Automated format detection and conversion
- Physical scale normalization across different recording conditions
- Intensity unit standardization and calibration
- Statistical validation of cross-format consistency

**Technical Implementation:**
- OpenCV-based video processing pipeline
- Automated arena size and pixel resolution normalization
- Temporal sampling rate harmonization
- Comprehensive validation against reference datasets

**Usage Example:**
```bash
python cross_format_comparison.py \
    --crimaldi-data data/crimaldi_dataset/ \
    --custom-data data/custom_recordings/ \
    --output results/cross_format_analysis/ \
    --validation-threshold 0.95
```

**Quality Assurance Metrics:**
- Cross-format correlation: >95% with reference implementations
- Processing accuracy: 6 decimal places for numerical values
- Format compatibility: Automatic detection and conversion

### Algorithm Comparison (algorithm_comparison.py)

**Purpose**: Advanced algorithm comparison study with statistical analysis and publication-ready visualization capabilities.

**Supported Algorithms:**
- **Infotaxis**: Information-theoretic navigation strategy
- **Casting**: Bio-inspired crosswind casting behavior
- **Gradient Following**: Direct concentration gradient pursuit
- **Hybrid Strategies**: Combination approaches with adaptive switching

**Statistical Analysis Framework:**
- Performance metric calculation and comparison
- Statistical significance testing with multiple correction
- Confidence interval estimation and visualization
- Robustness analysis across different experimental conditions

**Usage Example:**
```bash
python algorithm_comparison.py \
    --algorithms infotaxis,casting,gradient_following,hybrid \
    --datasets crimaldi,custom \
    --iterations 2000 \
    --statistical-analysis comprehensive \
    --output results/algorithm_comparison/
```

**Output Artifacts:**
- Performance comparison tables with statistical significance
- Trajectory visualization with algorithm-specific color coding
- Publication-ready figures with scientific formatting
- Comprehensive statistical analysis reports

### Data Normalization (normalization_example.py)

**Purpose**: Comprehensive data normalization pipeline demonstration with cross-format compatibility and validation.

**Normalization Components:**
- **Spatial Normalization**: Arena size and pixel resolution standardization
- **Temporal Normalization**: Sampling rate harmonization and interpolation
- **Intensity Calibration**: Unit conversion and dynamic range optimization
- **Format Standardization**: Automated conversion to common processing format

**Technical Validation:**
- Pre-processing integrity checks and format validation
- Automated detection of calibration parameters
- Quality assurance metrics and validation reporting
- Error handling for incompatible or corrupted data

**Usage Example:**
```bash
python normalization_example.py \
    --input data/raw_recordings/ \
    --output data/normalized/ \
    --calibration config/calibration_parameters.json \
    --validation comprehensive
```

**Quality Metrics:**
- Normalization accuracy: <1e-6 relative error
- Processing reliability: >99% success rate
- Cross-format consistency: >0.99 correlation coefficient

### Analysis and Visualization (analysis_visualization.py)

**Purpose**: Advanced analysis and visualization capabilities with scientific formatting and statistical validation.

**Visualization Components:**
- **Trajectory Analysis**: Path efficiency and navigation pattern visualization
- **Performance Metrics**: Statistical comparison with error bars and confidence intervals
- **Temporal Dynamics**: Time-series analysis of navigation behavior
- **Comparative Studies**: Multi-algorithm performance across different conditions

**Scientific Formatting Standards:**
- Publication-quality figure generation with appropriate sizing
- Statistical annotation with significance levels and effect sizes
- Color schemes optimized for accessibility and print reproduction
- Comprehensive figure legends and captions

**Usage Example:**
```bash
python analysis_visualization.py \
    --results results/simulation_outputs/ \
    --analysis comprehensive \
    --output figures/publication_ready/ \
    --format pdf,png \
    --dpi 300
```

## Configuration and Setup

### Environment Requirements

**Python Environment Specifications:**
```yaml
python_version: ">=3.9"
recommended_version: "3.11"
virtual_environment: recommended
package_manager: pip
```

**System Resource Requirements:**
- **Memory**: 8GB minimum, 16GB recommended for large batch operations
- **Storage**: 50GB for data and results, SSD recommended for performance
- **CPU**: Multi-core processor, parallel processing scales with core count
- **Network**: Internet connectivity for dependency installation only

**Computational Environment Validation:**
```bash
python -m pip install --upgrade pip
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

### Configuration Files

**Primary Configuration Structure:**
```json
{
  "simulation_parameters": {
    "algorithm_types": ["infotaxis", "casting", "gradient_following"],
    "iteration_count": 1000,
    "parallel_workers": 4,
    "timeout_seconds": 300
  },
  "data_processing": {
    "normalization_enabled": true,
    "validation_threshold": 0.95,
    "quality_checks": ["format_validation", "integrity_check", "calibration_validation"]
  },
  "output_configuration": {
    "result_format": ["json", "csv", "hdf5"],
    "visualization_enabled": true,
    "statistical_analysis": "comprehensive"
  }
}
```

**Configuration Parameters:**

| Parameter Category | Required Parameters | Optional Parameters |
|-------------------|-------------------|-------------------|
| Algorithm Settings | algorithm_types, iteration_count | timeout_seconds, random_seed |
| Data Processing | input_directories, normalization_enabled | validation_threshold, calibration_file |
| Performance | parallel_workers, memory_limit | cache_enabled, optimization_level |
| Output | output_directory, result_format | compression_enabled, backup_frequency |

### Data Preparation

**Plume Recording Data Requirements:**
- **Format Support**: AVI video files, custom binary formats, Crimaldi dataset structure
- **Resolution**: Minimum 640x480, recommended 1920x1080 for optimal analysis
- **Frame Rate**: 25-60 fps, automatic normalization for different sampling rates
- **Duration**: Minimum 30 seconds, recommended 2+ minutes for statistical reliability

**Directory Structure:**
```
data/
├── crimaldi_dataset/
│   ├── plume_recordings/
│   ├── calibration_data/
│   └── metadata/
├── custom_recordings/
│   ├── arena_1/
│   ├── arena_2/
│   └── calibration/
└── processed/
    ├── normalized/
    ├── validated/
    └── analysis_ready/
```

**Data Validation Checklist:**
- [ ] Video file integrity verification
- [ ] Calibration parameter availability
- [ ] Metadata completeness check
- [ ] Format compatibility validation
- [ ] Physical scale parameter verification

## Scientific Computing Guidelines

### Reproducibility Standards

**Accuracy Requirements:**
- **Correlation Accuracy**: >95% correlation with reference implementations
- **Numerical Precision**: 6 decimal places for all scientific calculations
- **Reproducibility Coefficient**: >0.99 across different computational environments
- **Statistical Validation**: Comprehensive testing against established benchmarks

**Implementation Practices:**
```python
# Random seed management for reproducibility
import numpy as np
np.random.seed(42)

# Numerical precision configuration
np.set_printoptions(precision=6, suppress=True)

# Reproducible computation settings
import os
os.environ['PYTHONHASHSEED'] = '0'
```

**Version Control for Scientific Reproducibility:**
- Algorithm implementation version tracking
- Configuration parameter documentation
- Input data provenance and validation
- Complete computational environment specification

### Performance Targets

**Processing Performance Standards:**
- **Individual Simulation**: <7.2 seconds average processing time
- **Batch Completion**: 4000+ simulations within 8 hours
- **Memory Efficiency**: <8GB peak memory usage for standard operations
- **Parallel Scaling**: Linear performance improvement with additional CPU cores

**Performance Monitoring Implementation:**
```python
import time
import psutil
from typing import Dict, Any

def monitor_performance(func):
    """Performance monitoring decorator for scientific computing functions."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        metrics = {
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'function_name': func.__name__
        }
        
        return result, metrics
    return wrapper
```

### Quality Assurance

**Validation Procedures:**
1. **Input Validation**: Comprehensive data format and integrity verification
2. **Algorithm Validation**: Statistical comparison with reference implementations
3. **Output Validation**: Result consistency and accuracy verification
4. **Performance Validation**: Processing time and resource usage compliance

**Quality Metrics Framework:**
```python
class QualityAssurance:
    """Scientific computing quality assurance framework."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.validation_results = {}
    
    def validate_numerical_accuracy(self, computed: np.ndarray, reference: np.ndarray) -> bool:
        """Validate numerical accuracy against reference implementation."""
        relative_error = np.abs((computed - reference) / reference)
        max_error = np.max(relative_error)
        
        self.validation_results['max_relative_error'] = max_error
        return max_error < self.tolerance
    
    def validate_statistical_properties(self, data: np.ndarray, expected_stats: Dict) -> bool:
        """Validate statistical properties of simulation results."""
        computed_stats = {
            'mean': np.mean(data),
            'std': np.std(data),
            'median': np.median(data)
        }
        
        for stat_name, expected_value in expected_stats.items():
            if stat_name in computed_stats:
                error = abs(computed_stats[stat_name] - expected_value) / expected_value
                if error > self.tolerance:
                    return False
        
        return True
```

## Usage Examples

### Basic Usage

**Simple Algorithm Execution:**
```bash
# Execute single algorithm with default parameters
python simple_batch_simulation.py --algorithm infotaxis --data data/sample_plumes/

# Specify custom configuration
python simple_batch_simulation.py \
    --config config/custom_parameters.json \
    --data data/plume_recordings/ \
    --output results/custom_analysis/
```

**Cross-Format Data Processing:**
```bash
# Process Crimaldi dataset
python cross_format_comparison.py \
    --input data/crimaldi_dataset/ \
    --format crimaldi \
    --output results/crimaldi_analysis/

# Process custom AVI recordings
python cross_format_comparison.py \
    --input data/custom_recordings/ \
    --format custom \
    --calibration config/arena_calibration.json \
    --output results/custom_analysis/
```

### Advanced Workflows

**Comprehensive Algorithm Comparison Study:**
```bash
# Multi-algorithm performance analysis
python algorithm_comparison.py \
    --algorithms infotaxis,casting,gradient_following,hybrid \
    --datasets crimaldi,custom \
    --iterations 2000 \
    --statistical-tests t_test,mann_whitney,kruskal_wallis \
    --multiple-correction bonferroni \
    --confidence-level 0.95 \
    --output results/comprehensive_comparison/
```

**Large-Scale Batch Processing:**
```bash
# High-throughput batch simulation with monitoring
python simple_batch_simulation.py \
    --config config/large_batch.json \
    --data data/all_recordings/ \
    --parallel-workers 8 \
    --memory-limit 16GB \
    --checkpoint-frequency 100 \
    --progress-monitoring enabled \
    --output results/large_scale_analysis/
```

**Data Normalization Pipeline:**
```bash
# Complete normalization workflow
python normalization_example.py \
    --input data/raw_recordings/ \
    --calibration config/calibration_parameters.json \
    --spatial-normalization enabled \
    --temporal-normalization enabled \
    --intensity-calibration enabled \
    --validation comprehensive \
    --output data/normalized_recordings/
```

### Customization Examples

**Custom Algorithm Integration:**
```python
# Example: Custom navigation algorithm implementation
from src.algorithms.base import NavigationAlgorithm
from typing import Tuple, Optional

class CustomAlgorithm(NavigationAlgorithm):
    """Custom navigation algorithm implementation."""
    
    def __init__(self, parameters: dict):
        super().__init__(parameters)
        self.custom_parameter = parameters.get('custom_param', 1.0)
    
    def navigate_step(self, current_position: Tuple[float, float], 
                     plume_data: np.ndarray) -> Tuple[float, float]:
        """Implement custom navigation logic."""
        # Custom algorithm implementation
        next_position = self._calculate_next_position(current_position, plume_data)
        return next_position
    
    def _calculate_next_position(self, position: Tuple[float, float], 
                               plume_data: np.ndarray) -> Tuple[float, float]:
        """Custom position calculation logic."""
        # Implementation specific to custom algorithm
        pass

# Register custom algorithm for use in examples
from src.algorithms import register_algorithm
register_algorithm('custom_algorithm', CustomAlgorithm)
```

**Custom Analysis Configuration:**
```json
{
  "analysis_configuration": {
    "custom_metrics": [
      {
        "name": "success_efficiency",
        "calculation": "success_rate * path_efficiency",
        "weight": 0.6
      },
      {
        "name": "temporal_consistency",
        "calculation": "std(search_times) / mean(search_times)",
        "weight": 0.4
      }
    ],
    "visualization_options": {
      "custom_plots": ["efficiency_heatmap", "temporal_analysis"],
      "color_scheme": "scientific",
      "figure_size": [12, 8],
      "dpi": 300
    }
  }
}
```

## Output and Results

### Result Formats

**Primary Output Formats:**
- **JSON**: Structured data with full metadata and configuration information
- **CSV**: Tabular data optimized for statistical analysis and spreadsheet import
- **HDF5**: High-performance binary format for large datasets and numerical arrays
- **Visualization Files**: PNG, PDF, and SVG formats for figures and plots

**JSON Output Structure:**
```json
{
  "simulation_metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "algorithm": "infotaxis",
    "configuration": {...},
    "environment_info": {...}
  },
  "performance_metrics": {
    "success_rate": 0.856,
    "average_search_time": 45.2,
    "path_efficiency": 0.742,
    "statistical_significance": 0.001
  },
  "trajectory_data": [...],
  "analysis_results": {...}
}
```

**CSV Output Format:**
```csv
simulation_id,algorithm,success,search_time,path_length,efficiency,plume_dataset
001,infotaxis,True,42.5,156.8,0.78,crimaldi_001
002,infotaxis,True,38.2,142.1,0.82,crimaldi_002
003,casting,False,60.0,245.6,0.45,crimaldi_003
```

### Interpretation Guide

**Performance Metrics Interpretation:**

| Metric | Range | Interpretation | Statistical Significance |
|--------|-------|----------------|-------------------------|
| Success Rate | 0.0 - 1.0 | Fraction of successful source localizations | p < 0.05 for meaningful comparison |
| Search Time | 0 - 300 seconds | Average time to source localization | Log-normal distribution typical |
| Path Efficiency | 0.0 - 1.0 | Ratio of direct distance to actual path length | Higher values indicate more efficient navigation |
| Robustness Index | 0.0 - 1.0 | Performance consistency across different conditions | Standard deviation-based metric |

**Statistical Analysis Results:**
- **Confidence Intervals**: 95% confidence bounds for all performance metrics
- **Effect Sizes**: Cohen's d for practical significance assessment
- **Multiple Comparisons**: Bonferroni correction for family-wise error rate control
- **Normality Testing**: Shapiro-Wilk and Anderson-Darling tests for distribution assumptions

### Publication Guidelines

**Scientific Publication Standards:**
- All numerical results reported with appropriate precision (6 decimal places)
- Statistical significance testing with multiple comparison correction
- Complete methodology description including algorithm parameters
- Reproducibility information including software versions and configuration

**Figure and Table Guidelines:**
```python
# Publication-ready figure configuration
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")

# Configure figure for publication
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.set_xlabel('Search Time (seconds)', fontsize=14)
ax.set_ylabel('Success Rate', fontsize=14)
ax.set_title('Algorithm Performance Comparison', fontsize=16, fontweight='bold')

# Add statistical annotations
from scipy import stats
# Statistical comparison implementation...

# Save in multiple formats
plt.savefig('figures/algorithm_comparison.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/algorithm_comparison.png', bbox_inches='tight', dpi=300)
```

**Citation Requirements:**
When using results from these examples in publications, please include:
- Software version and configuration details
- Dataset specifications and preprocessing methods
- Statistical analysis methodology
- Complete parameter settings for reproducibility

## Troubleshooting

### Common Issues

**Video Processing Errors:**

*Issue*: `OpenCV Error: Unsupported format or codec`
```bash
Error: cv2.error: OpenCV(4.11.0) error: (-2:Unspecified error) in function 'cvReadNextFrameFile'
```
*Solution*:
```bash
# Install additional codec support
pip install opencv-python-headless
# or for full codec support
conda install opencv

# Verify video file integrity
python -c "import cv2; cap = cv2.VideoCapture('path/to/video.avi'); print('Frames:', cap.get(cv2.CAP_PROP_FRAME_COUNT))"
```

**Memory Management Issues:**

*Issue*: `MemoryError: Unable to allocate array`
```python
MemoryError: Unable to allocate 2.51 GiB for an array with shape (89600, 89600) and data type float64
```
*Solutions*:
```python
# Implement memory-efficient processing
import numpy as np
from joblib import Memory

# Configure memory caching
memory = Memory(location='./cache', verbose=0)

@memory.cache
def process_video_chunk(video_path, start_frame, end_frame):
    """Process video in chunks to manage memory usage."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for i in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    return np.array(frames)

# Process large videos in chunks
chunk_size = 1000
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for start in range(0, total_frames, chunk_size):
    end = min(start + chunk_size, total_frames)
    chunk_data = process_video_chunk(video_path, start, end)
    # Process chunk...
```

**Configuration Validation Errors:**

*Issue*: `ValidationError: Invalid algorithm parameters`
```json
{
  "error": "ValidationError",
  "message": "Invalid algorithm parameters",
  "details": "Parameter 'step_size' must be positive float"
}
```
*Solution*:
```python
# Implement comprehensive parameter validation
from typing import Dict, Any
import json

def validate_configuration(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters."""
    required_fields = ['algorithm_types', 'iteration_count', 'output_directory']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate algorithm parameters
    for algorithm in config['algorithm_types']:
        if algorithm not in ['infotaxis', 'casting', 'gradient_following', 'hybrid']:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Validate numerical parameters
    if config['iteration_count'] <= 0:
        raise ValueError("iteration_count must be positive integer")
    
    return True

# Usage
try:
    with open('config/parameters.json', 'r') as f:
        config = json.load(f)
    validate_configuration(config)
except Exception as e:
    print(f"Configuration validation failed: {e}")
```

### Performance Optimization

**Parallel Processing Optimization:**
```python
import multiprocessing as mp
from joblib import Parallel, delayed

def optimize_parallel_processing():
    """Configure optimal parallel processing settings."""
    # Determine optimal worker count
    cpu_count = mp.cpu_count()
    optimal_workers = min(cpu_count - 1, 8)  # Leave one core free, cap at 8
    
    # Configure joblib for scientific computing
    parallel_config = {
        'n_jobs': optimal_workers,
        'backend': 'multiprocessing',
        'batch_size': 'auto',
        'temp_folder': './temp_parallel',
        'max_nbytes': '100M'
    }
    
    return parallel_config

# Example usage
config = optimize_parallel_processing()
results = Parallel(**config)(
    delayed(process_simulation)(params) for params in parameter_list
)
```

**Memory Usage Optimization:**
```python
import gc
import psutil

def monitor_memory_usage():
    """Monitor and optimize memory usage during processing."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"Memory usage: {memory_info.rss / 1024**2:.1f} MB")
    print(f"Memory percent: {process.memory_percent():.1f}%")
    
    # Force garbage collection if memory usage is high
    if process.memory_percent() > 80:
        gc.collect()
        print("Garbage collection performed")

# Configure numpy for memory efficiency
import numpy as np
np.seterr(all='raise')  # Catch numerical errors early
```

### Error Handling

**Comprehensive Error Recovery:**
```python
import logging
import traceback
from typing import Optional, Callable, Any

def robust_execution(func: Callable, *args, **kwargs) -> Optional[Any]:
    """Execute function with comprehensive error handling."""
    try:
        return func(*args, **kwargs)
    
    except MemoryError as e:
        logging.error(f"Memory error in {func.__name__}: {e}")
        # Attempt memory cleanup and retry
        gc.collect()
        try:
            return func(*args, **kwargs)
        except MemoryError:
            logging.error("Memory error persists after cleanup")
            return None
    
    except cv2.error as e:
        logging.error(f"OpenCV error in {func.__name__}: {e}")
        # Handle video processing errors
        return None
    
    except Exception as e:
        logging.error(f"Unexpected error in {func.__name__}: {e}")
        logging.error(traceback.format_exc())
        return None

# Configure logging for troubleshooting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simulation.log'),
        logging.StreamHandler()
    ]
)
```

## Contributing and Development

### Adding New Examples

**Example Development Guidelines:**
1. **Follow Scientific Computing Standards**: Implement robust numerical validation and error handling
2. **Maintain Educational Value**: Include comprehensive documentation and clear learning objectives
3. **Ensure Reproducibility**: Implement deterministic execution with proper random seed management
4. **Performance Compliance**: Meet processing time targets (<7.2 seconds per simulation)

**Example Template Structure:**
```python
#!/usr/bin/env python3
"""
Example Template: [Example Name]

Purpose: [Clear description of educational and scientific objectives]
Target Audience: [Specific user groups and use cases]
Learning Objectives: [List of specific learning outcomes]

Dependencies:
    - numpy>=2.1.3
    - scipy>=1.15.3
    - [additional dependencies]

Usage:
    python example_template.py --config config/parameters.json --data data/input/

Author: [Name]
Date: [Date]
Version: [Version]
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExampleTemplate:
    """Template class for new example implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize example with configuration parameters."""
        self.config = config
        self.validate_configuration()
    
    def validate_configuration(self) -> None:
        """Validate configuration parameters."""
        required_params = ['input_directory', 'output_directory']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required parameter: {param}")
    
    def execute(self) -> Optional[Dict[str, Any]]:
        """Execute example workflow."""
        try:
            logger.info("Starting example execution")
            
            # Implementation steps
            results = self._process_data()
            self._save_results(results)
            
            logger.info("Example execution completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Example execution failed: {e}")
            return None
    
    def _process_data(self) -> Dict[str, Any]:
        """Implement data processing logic."""
        # Example-specific implementation
        pass
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results in standardized format."""
        # Standardized output implementation
        pass

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Example Template")
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--data', required=True, help='Input data directory')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    import json
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config.update({
        'input_directory': args.data,
        'output_directory': args.output
    })
    
    # Execute example
    example = ExampleTemplate(config)
    results = example.execute()
    
    if results is None:
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Testing Examples

**Testing Framework Requirements:**
```python
import pytest
import numpy as np
from pathlib import Path

class TestExampleTemplate:
    """Comprehensive test suite for example implementations."""
    
    def setup_method(self):
        """Setup test fixtures and data."""
        self.test_config = {
            'input_directory': 'tests/data/sample_input',
            'output_directory': 'tests/data/sample_output',
            'algorithm': 'infotaxis',
            'iterations': 10
        }
    
    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        # Test valid configuration
        example = ExampleTemplate(self.test_config)
        assert example.config == self.test_config
        
        # Test invalid configuration
        invalid_config = self.test_config.copy()
        del invalid_config['input_directory']
        
        with pytest.raises(ValueError):
            ExampleTemplate(invalid_config)
    
    def test_execution_success(self):
        """Test successful execution with valid inputs."""
        example = ExampleTemplate(self.test_config)
        results = example.execute()
        
        assert results is not None
        assert 'performance_metrics' in results
        assert 'processing_time' in results
    
    def test_numerical_accuracy(self):
        """Test numerical accuracy against reference implementation."""
        example = ExampleTemplate(self.test_config)
        results = example.execute()
        
        # Load reference results
        reference_file = Path('tests/data/reference_results.json')
        with open(reference_file, 'r') as f:
            reference = json.load(f)
        
        # Validate numerical accuracy
        for metric, reference_value in reference['metrics'].items():
            computed_value = results['performance_metrics'][metric]
            relative_error = abs(computed_value - reference_value) / reference_value
            assert relative_error < 1e-6, f"Numerical accuracy failed for {metric}"
    
    def test_performance_requirements(self):
        """Test performance requirement compliance."""
        import time
        
        example = ExampleTemplate(self.test_config)
        start_time = time.time()
        results = example.execute()
        execution_time = time.time() - start_time
        
        # Performance targets
        max_execution_time = 7.2 * self.test_config['iterations']
        assert execution_time < max_execution_time, "Performance requirement not met"
    
    @pytest.mark.parametrize("algorithm", ["infotaxis", "casting", "gradient_following"])
    def test_algorithm_compatibility(self, algorithm):
        """Test compatibility with different algorithms."""
        config = self.test_config.copy()
        config['algorithm'] = algorithm
        
        example = ExampleTemplate(config)
        results = example.execute()
        
        assert results is not None
        assert results['algorithm'] == algorithm
```

### Documentation Standards

**Documentation Requirements:**
- **Scientific Precision**: All numerical values documented with appropriate precision
- **Code Comments**: Comprehensive inline documentation for complex algorithms
- **Usage Examples**: Multiple complexity levels from basic to advanced
- **Cross-References**: Links to related examples and theoretical background

**Documentation Template:**
```python
def complex_analysis_function(data: np.ndarray, parameters: Dict[str, float]) -> Dict[str, Any]:
    """
    Perform complex statistical analysis on simulation results.
    
    This function implements advanced statistical analysis techniques for
    navigation algorithm performance evaluation, including multi-dimensional
    scaling, cluster analysis, and statistical significance testing.
    
    Args:
        data: Simulation result data array with shape (n_simulations, n_features)
              Features include: [success_rate, search_time, path_efficiency, ...]
        parameters: Analysis parameters dictionary containing:
                   - 'confidence_level': Statistical confidence level (default: 0.95)
                   - 'correction_method': Multiple comparison correction ('bonferroni', 'fdr')
                   - 'clustering_method': Clustering algorithm ('kmeans', 'hierarchical')
    
    Returns:
        Dictionary containing analysis results:
        - 'statistical_tests': Results of hypothesis testing
        - 'cluster_analysis': Clustering results and validation metrics
        - 'visualization_data': Data prepared for plotting functions
        - 'summary_statistics': Descriptive statistics and effect sizes
    
    Raises:
        ValueError: If input data dimensions are incompatible
        StatisticalError: If statistical assumptions are violated
    
    Example:
        >>> import numpy as np
        >>> simulation_data = np.random.random((100, 5))  # 100 simulations, 5 metrics
        >>> params = {'confidence_level': 0.95, 'correction_method': 'bonferroni'}
        >>> results = complex_analysis_function(simulation_data, params)
        >>> print(f"Statistical significance: {results['statistical_tests']['p_value']:.6f}")
        
    References:
        [1] Statistical Methods in Navigation Research, Journal of Computational Biology
        [2] Advanced Analysis Techniques for Algorithm Comparison Studies
        
    Scientific Computing Notes:
        - All statistical tests use 6 decimal place precision
        - Reproducible random seed management for clustering algorithms
        - Comprehensive validation of statistical assumptions
        - Multiple comparison correction applied to family-wise error rate
    """
    # Implementation with extensive inline comments
    pass
```

## References and Further Reading

### Scientific Literature

**Foundational Papers:**
1. **Infotaxis Algorithm**: Vergassola, M., Villermaux, E., & Shraiman, B. I. (2007). "'Infotaxis' as a strategy for searching without gradients." *Nature*, 445(7126), 406-409.

2. **Plume Navigation Theory**: Cardé, R. T., & Willis, M. A. (2008). "Navigational strategies used by insects to find distant, wind-borne sources of odor." *Journal of Chemical Ecology*, 34(7), 854-866.

3. **Computational Olfactory Navigation**: Balkovsky, E., & Shraiman, B. I. (2002). "Olfactory search at high Reynolds number." *Proceedings of the National Academy of Sciences*, 99(20), 12589-12593.

**Algorithmic Development References:**
- Casting Behavior Modeling: Frye, M. A., & Dickinson, M. H. (2004). "Motor output reflects the linear superposition of visual and olfactory inputs in *Drosophila*." *Journal of Experimental Biology*, 207(1), 123-131.
- Gradient Following Strategies: Murlis, J., Elkinton, J. S., & Cardé, R. T. (1992). "Odor plumes and how insects use them." *Annual Review of Entomology*, 37(1), 505-532.

### Technical Documentation

**System Architecture References:**
- **Data Normalization**: Technical Specification Document, Section F-002
- **Batch Processing**: Technical Specification Document, Section F-003  
- **Algorithm Interface**: Technical Specification Document, Section F-004
- **Performance Analysis**: Technical Specification Document, Section F-005

**Implementation Guidelines:**
- **Scientific Computing Standards**: NumPy Documentation on Numerical Precision
- **Parallel Processing**: Joblib Documentation for Scientific Computing
- **Statistical Analysis**: SciPy Documentation on Statistical Functions
- **Visualization**: Matplotlib Documentation for Publication-Quality Figures

### Educational Resources

**Online Courses and Tutorials:**
- **Computational Biology Methods**: Introduction to algorithm development for biological systems
- **Scientific Python Programming**: Advanced techniques for research computing
- **Statistical Analysis in Research**: Hypothesis testing and experimental design
- **High-Performance Scientific Computing**: Optimization techniques for computational research

**Research Community Resources:**
- **Olfactory Navigation Research Group**: International community of researchers
- **Computational Ethology Society**: Interdisciplinary research organization
- **Bio-inspired Robotics Consortium**: Academic and industry collaboration network

### Software Dependencies and Tools

**Core Scientific Computing Stack:**
- **NumPy**: Harris, C. R., et al. (2020). "Array programming with NumPy." *Nature*, 585(7825), 357-362.
- **SciPy**: Virtanen, P., et al. (2020). "SciPy 1.0: fundamental algorithms for scientific computing in Python." *Nature Methods*, 17(3), 261-272.
- **Matplotlib**: Hunter, J. D. (2007). "Matplotlib: A 2D graphics environment." *Computing in Science & Engineering*, 9(3), 90-95.

**Specialized Libraries:**
- **OpenCV**: Bradski, G. (2000). "The OpenCV Library." *Dr. Dobb's Journal of Software Tools*
- **Joblib**: Parallel computing and memory mapping for scientific applications
- **Pandas**: McKinney, W. (2010). "Data structures for statistical computing in Python." *Proceedings of the 9th Python in Science Conference*

---

**Version Information:**
- Documentation Version: 1.0.0
- Last Updated: 2024-01-15
- Compatibility: Python 3.9+, Scientific Computing Stack 2024.1

**Support and Contact:**
For technical support, bug reports, or contributions, please refer to the project repository documentation and issue tracking system.