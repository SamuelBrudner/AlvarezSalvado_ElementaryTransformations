# Crimaldi Dataset Processing Example

## Overview

The Crimaldi dataset represents a cornerstone in plume navigation research, providing high-fidelity recordings of chemical plume dispersal in controlled laboratory environments. This comprehensive guide demonstrates the complete workflow for processing Crimaldi plume dataset recordings using the plume navigation algorithm simulation system, ensuring scientific computing precision and reproducibility standards.

### Crimaldi Format Characteristics

Crimaldi format datasets are characterized by specific technical specifications that distinguish them from other plume recording formats:

- **Video Format**: Typically AVI containers with MJPEG compression for lossless temporal fidelity
- **Spatial Resolution**: High-definition recordings with pixel-to-meter calibration ratios typically around 100.0 pixels per meter
- **Temporal Resolution**: 30-50 Hz frame rates ensuring adequate temporal sampling for navigation algorithm analysis
- **Intensity Encoding**: 16-bit grayscale intensity values representing concentration measurements in parts-per-million (PPM)
- **Coordinate System**: Cartesian coordinate system with bottom-left origin convention
- **Metadata Integration**: Embedded calibration parameters and arena specifications within video metadata

### Physical Scale Parameters

Critical physical scale parameters embedded within Crimaldi recordings include:

- **Arena Dimensions**: Physical laboratory dimensions in meters (typically 1.0m x 1.0m normalized)
- **Pixel Calibration**: Precise pixel-to-meter conversion ratios for spatial accuracy
- **Intensity Calibration**: Concentration measurement scaling from sensor readings to PPM values
- **Temporal Calibration**: Frame rate specifications ensuring temporal consistency

### Calibration Requirements

Successful Crimaldi dataset processing requires extraction and validation of calibration parameters:

- **Spatial Calibration**: Verification of pixel-to-meter ratios with <1% accuracy requirement
- **Intensity Calibration**: Validation of concentration measurement scaling with <2% calibration error threshold
- **Temporal Calibration**: Frame rate consistency validation ensuring >99% frame alignment accuracy
- **Arena Boundary Detection**: Automatic detection and validation of experimental arena boundaries

### Scientific Computing Standards

Processing adheres to rigorous scientific computing standards:

- **Numerical Precision**: float64 precision for all calculations with 1e-6 relative error threshold
- **Reproducibility**: Identical results across different computational environments and platforms
- **Correlation Requirements**: >95% correlation with reference implementations for validation
- **Statistical Significance**: p < 0.05 threshold for performance comparisons and validation tests

## Prerequisites

### Software Dependencies

The Crimaldi dataset processing pipeline requires specific software dependencies with validated version compatibility:

```bash
# Core Python environment (version 3.9+)
python >= 3.9

# Essential numerical computing libraries
numpy >= 2.1.3          # Advanced array operations and numerical precision
scipy >= 1.15.3         # Statistical analysis and signal processing
pandas >= 2.2.0         # Data manipulation and analysis frameworks

# Video processing capabilities
opencv-python >= 4.11.0 # Video codec handling and image processing

# Parallel processing framework
joblib >= 1.6.0         # Memory mapping and batch processing optimization

# Visualization and analysis
matplotlib >= 3.9.0     # Scientific plotting and visualization
seaborn >= 0.13.2       # Statistical data visualization

# Quality assurance and testing
pytest >= 8.3.5         # Comprehensive testing framework
```

### Hardware Requirements

Minimum and recommended hardware specifications for optimal processing performance:

**Minimum Requirements:**
- CPU: 4-core processor with 2.5 GHz base frequency
- RAM: 8 GB system memory for single-file processing
- Storage: 50 GB available space for data and temporary files
- GPU: Not required but beneficial for OpenCV acceleration

**Recommended Configuration:**
- CPU: 8-core processor with 3.0+ GHz base frequency for parallel processing
- RAM: 16 GB system memory for batch processing operations
- Storage: 200 GB high-speed SSD for optimal I/O performance
- GPU: CUDA-compatible GPU for accelerated video processing (optional)

### Environment Configuration

Detailed environment setup ensuring reproducible processing conditions:

```bash
# Create isolated Python environment
python -m venv crimaldi_processing_env

# Activate environment (Linux/macOS)
source crimaldi_processing_env/bin/activate

# Activate environment (Windows)
# crimaldi_processing_env\Scripts\activate

# Install required dependencies
pip install numpy==2.1.3 scipy==1.15.3 pandas==2.2.0
pip install opencv-python==4.11.0 joblib==1.6.0
pip install matplotlib==3.9.0 seaborn==0.13.2 pytest==8.3.5

# Verify installation integrity
python -c "import numpy, scipy, pandas, cv2, joblib, matplotlib, seaborn, pytest; print('All dependencies successfully installed')"
```

### Test Data Preparation

Preparation of sample Crimaldi datasets for validation and testing:

```bash
# Create directory structure for test data
mkdir -p data/crimaldi_samples
mkdir -p data/normalized
mkdir -p results/crimaldi_processing

# Download sample Crimaldi dataset (replace with actual data source)
# wget -P data/crimaldi_samples/ [CRIMALDI_SAMPLE_URL]

# Verify sample data integrity
python -c "
import cv2
video_path = 'data/crimaldi_samples/crimaldi_sample.avi'
cap = cv2.VideoCapture(video_path)
if cap.isOpened():
    print(f'Sample video loaded: {cap.get(cv2.CAP_PROP_FRAME_COUNT)} frames')
    print(f'Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')
    print(f'Frame rate: {cap.get(cv2.CAP_PROP_FPS)} fps')
    cap.release()
else:
    print('Error: Could not load sample video')
"
```

## Data Preparation

### File Format Validation

Comprehensive validation procedures ensuring Crimaldi format compliance and data integrity:

```python
from src.backend.io.crimaldi_format_handler import detect_crimaldi_format, validate_crimaldi_format
from pathlib import Path
import logging

# Configure detailed logging for validation process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_crimaldi_file(file_path):
    """
    Comprehensive Crimaldi file validation with detailed reporting.
    
    Args:
        file_path (Path): Path to Crimaldi video file
        
    Returns:
        dict: Validation results with confidence metrics and issue reports
    """
    file_path = Path(file_path)
    
    # Primary format detection with deep inspection
    format_info = detect_crimaldi_format(file_path, deep_inspection=True)
    logger.info(f"Format detection: {format_info['format']} (confidence: {format_info['confidence']:.3f})")
    
    # Detailed format validation
    validation_result = validate_crimaldi_format(file_path, strict_validation=True)
    
    if format_info['confidence'] < 0.95:
        logger.warning(f"Low format detection confidence: {format_info['confidence']:.3f}")
        logger.warning("Consider manual format verification or deep inspection mode")
    
    # Report validation details
    logger.info(f"Video properties validation: {'PASS' if validation_result['video_valid'] else 'FAIL'}")
    logger.info(f"Metadata extraction: {'PASS' if validation_result['metadata_valid'] else 'FAIL'}")
    logger.info(f"Calibration parameters: {'PASS' if validation_result['calibration_valid'] else 'FAIL'}")
    
    return {
        'format_confidence': format_info['confidence'],
        'validation_passed': validation_result['overall_valid'],
        'issues_detected': validation_result.get('issues', []),
        'recommendations': validation_result.get('recommendations', [])
    }

# Example usage for batch validation
crimaldi_files = [
    'data/crimaldi_samples/experiment_1.avi',
    'data/crimaldi_samples/experiment_2.avi',
    'data/crimaldi_samples/experiment_3.avi'
]

validation_results = {}
for file_path in crimaldi_files:
    if Path(file_path).exists():
        validation_results[file_path] = validate_crimaldi_file(file_path)
    else:
        logger.error(f"File not found: {file_path}")

# Generate validation summary report
valid_files = [f for f, r in validation_results.items() if r['validation_passed']]
logger.info(f"Validation summary: {len(valid_files)}/{len(crimaldi_files)} files passed validation")
```

### Metadata Extraction

Detailed extraction and validation of Crimaldi metadata including calibration parameters:

```python
from src.backend.io.crimaldi_format_handler import CrimaldiFormatHandler
from src.backend.utils.logging_utils import get_logger, LoggingContext
import json

def extract_crimaldi_metadata(file_path, output_path=None):
    """
    Extract comprehensive metadata from Crimaldi format files.
    
    Args:
        file_path (str): Path to Crimaldi video file
        output_path (str, optional): Path to save extracted metadata JSON
        
    Returns:
        dict: Complete metadata including calibration parameters
    """
    with LoggingContext('metadata_extraction'):
        handler = CrimaldiFormatHandler(file_path, enable_caching=True)
        
        # Extract basic video properties
        video_properties = handler.get_video_properties()
        logger = get_logger()
        logger.info(f"Video duration: {video_properties['duration_seconds']:.2f} seconds")
        logger.info(f"Frame dimensions: {video_properties['width']}x{video_properties['height']}")
        logger.info(f"Frame rate: {video_properties['fps']:.2f} fps")
        logger.info(f"Total frames: {video_properties['frame_count']}")
        
        # Extract calibration parameters with validation
        calibration = handler.get_calibration_parameters(validate_accuracy=True)
        logger.info(f"Pixel-to-meter ratio: {calibration['pixel_to_meter_ratio']:.2f}")
        logger.info(f"Arena dimensions: {calibration['arena_width_meters']:.3f}x{calibration['arena_height_meters']:.3f} meters")
        logger.info(f"Intensity units: {calibration['intensity_units']}")
        logger.info(f"Coordinate system: {calibration['coordinate_system']}")
        
        # Extract experimental metadata
        experimental_metadata = handler.get_experimental_metadata()
        logger.info(f"Experimental conditions: {experimental_metadata.get('conditions', 'Not specified')}")
        logger.info(f"Recording date: {experimental_metadata.get('recording_date', 'Not specified')}")
        
        # Compile complete metadata
        complete_metadata = {
            'file_path': str(file_path),
            'format_info': {
                'format': 'Crimaldi',
                'version': handler.get_format_version(),
                'compression': video_properties.get('compression', 'Unknown')
            },
            'video_properties': video_properties,
            'calibration_parameters': calibration,
            'experimental_metadata': experimental_metadata,
            'extraction_timestamp': handler.get_extraction_timestamp(),
            'validation_status': handler.validate_metadata_completeness()
        }
        
        # Save metadata to JSON file if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(complete_metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved to: {output_path}")
        
        return complete_metadata

# Example: Extract metadata from sample files
sample_files = [
    'data/crimaldi_samples/crimaldi_sample.avi'
]

for file_path in sample_files:
    if Path(file_path).exists():
        metadata = extract_crimaldi_metadata(
            file_path, 
            output_path=f"data/metadata/{Path(file_path).stem}_metadata.json"
        )
        print(f"Metadata extraction completed for: {file_path}")
    else:
        print(f"Warning: File not found: {file_path}")
```

### Calibration Parameter Setup

Detailed setup and validation of calibration parameters for accurate physical scale normalization:

```python
from src.backend.io.crimaldi_format_handler import CrimaldiFormatHandler
from src.backend.core.calibration_validator import validate_calibration_parameters
import numpy as np

def setup_crimaldi_calibration(file_path, manual_calibration=None):
    """
    Setup and validate calibration parameters for Crimaldi dataset processing.
    
    Args:
        file_path (str): Path to Crimaldi video file
        manual_calibration (dict, optional): Manual calibration override parameters
        
    Returns:
        dict: Validated calibration parameters ready for normalization
    """
    handler = CrimaldiFormatHandler(file_path)
    
    # Extract embedded calibration parameters
    embedded_calibration = handler.get_calibration_parameters(validate_accuracy=True)
    
    # Use manual calibration if provided, otherwise use embedded
    calibration = manual_calibration if manual_calibration else embedded_calibration
    
    # Validate calibration parameter accuracy
    validation_result = validate_calibration_parameters(calibration, strict_validation=True)
    
    if not validation_result['valid']:
        raise ValueError(f"Calibration validation failed: {validation_result['errors']}")
    
    # Calculate derived calibration parameters
    derived_parameters = {
        'meters_per_pixel': 1.0 / calibration['pixel_to_meter_ratio'],
        'arena_area_square_meters': calibration['arena_width_meters'] * calibration['arena_height_meters'],
        'pixel_area_per_square_meter': calibration['pixel_to_meter_ratio'] ** 2,
        'temporal_resolution_seconds': 1.0 / calibration.get('frame_rate_hz', 30.0)
    }
    
    # Enhanced calibration with derived parameters
    enhanced_calibration = {
        **calibration,
        **derived_parameters,
        'validation_status': validation_result,
        'calibration_accuracy_percent': validation_result.get('accuracy_percent', 0.0)
    }
    
    print(f"Calibration setup completed:")
    print(f"  Pixel-to-meter ratio: {enhanced_calibration['pixel_to_meter_ratio']:.2f}")
    print(f"  Spatial resolution: {enhanced_calibration['meters_per_pixel']:.6f} m/pixel")
    print(f"  Arena area: {enhanced_calibration['arena_area_square_meters']:.3f} square meters")
    print(f"  Temporal resolution: {enhanced_calibration['temporal_resolution_seconds']:.4f} seconds/frame")
    print(f"  Calibration accuracy: {enhanced_calibration['calibration_accuracy_percent']:.1f}%")
    
    return enhanced_calibration

# Example calibration setup with validation
sample_file = 'data/crimaldi_samples/crimaldi_sample.avi'

if Path(sample_file).exists():
    # Setup calibration from embedded parameters
    calibration = setup_crimaldi_calibration(sample_file)
    
    # Example of manual calibration override (if needed)
    manual_calibration_override = {
        'pixel_to_meter_ratio': 100.0,
        'arena_width_meters': 1.0,
        'arena_height_meters': 1.0,
        'intensity_units': 'concentration_ppm',
        'coordinate_system': 'cartesian',
        'coordinate_origin': 'bottom_left',
        'frame_rate_hz': 30.0
    }
    
    # Setup with manual override
    manual_calibration = setup_crimaldi_calibration(sample_file, manual_calibration_override)
    
    print("Calibration setup completed successfully")
else:
    print(f"Error: Sample file not found: {sample_file}")
```

### Quality Assessment

Comprehensive quality assessment procedures for Crimaldi dataset validation:

```python
from src.backend.core.quality_assessment import CrimaldiQualityAssessment
from src.backend.utils.logging_utils import get_logger
import matplotlib.pyplot as plt

def assess_crimaldi_quality(file_path, assessment_config=None):
    """
    Comprehensive quality assessment for Crimaldi dataset files.
    
    Args:
        file_path (str): Path to Crimaldi video file
        assessment_config (dict, optional): Custom assessment configuration
        
    Returns:
        dict: Detailed quality assessment results with metrics and recommendations
    """
    # Default assessment configuration
    default_config = {
        'spatial_quality_threshold': 0.95,
        'temporal_consistency_threshold': 0.99,
        'intensity_calibration_threshold': 0.98,
        'frame_alignment_tolerance': 1e-6,
        'generate_quality_report': True,
        'create_visualization': True
    }
    
    config = {**default_config, **(assessment_config or {})}
    
    # Initialize quality assessor
    assessor = CrimaldiQualityAssessment(file_path, config)
    logger = get_logger()
    
    # Perform comprehensive quality assessment
    logger.info("Starting comprehensive Crimaldi quality assessment...")
    
    # Spatial quality assessment
    spatial_quality = assessor.assess_spatial_quality()
    logger.info(f"Spatial quality score: {spatial_quality['overall_score']:.3f}")
    logger.info(f"Resolution consistency: {spatial_quality['resolution_consistency']:.3f}")
    logger.info(f"Calibration accuracy: {spatial_quality['calibration_accuracy']:.3f}")
    
    # Temporal quality assessment
    temporal_quality = assessor.assess_temporal_quality()
    logger.info(f"Temporal quality score: {temporal_quality['overall_score']:.3f}")
    logger.info(f"Frame rate consistency: {temporal_quality['frame_rate_consistency']:.3f}")
    logger.info(f"Temporal alignment: {temporal_quality['temporal_alignment']:.3f}")
    
    # Intensity calibration assessment
    intensity_quality = assessor.assess_intensity_calibration()
    logger.info(f"Intensity calibration score: {intensity_quality['overall_score']:.3f}")
    logger.info(f"Calibration linearity: {intensity_quality['linearity_score']:.3f}")
    logger.info(f"Dynamic range utilization: {intensity_quality['dynamic_range_score']:.3f}")
    
    # Calculate overall quality score
    overall_quality = assessor.calculate_overall_quality_score()
    logger.info(f"Overall quality score: {overall_quality['composite_score']:.3f}")
    
    # Quality assessment summary
    quality_summary = {
        'file_path': str(file_path),
        'spatial_quality': spatial_quality,
        'temporal_quality': temporal_quality,
        'intensity_quality': intensity_quality,
        'overall_quality': overall_quality,
        'quality_issues': assessor.identify_quality_issues(),
        'recommendations': assessor.generate_quality_recommendations(),
        'assessment_timestamp': assessor.get_assessment_timestamp()
    }
    
    # Generate quality visualization if requested
    if config['create_visualization']:
        assessor.create_quality_visualization(output_path=f"results/quality_assessment_{Path(file_path).stem}.png")
        logger.info("Quality assessment visualization saved")
    
    # Generate detailed quality report if requested
    if config['generate_quality_report']:
        assessor.generate_quality_report(output_path=f"results/quality_report_{Path(file_path).stem}.json")
        logger.info("Detailed quality report saved")
    
    return quality_summary

# Example quality assessment execution
sample_files = [
    'data/crimaldi_samples/crimaldi_sample.avi'
]

quality_results = {}
for file_path in sample_files:
    if Path(file_path).exists():
        quality_results[file_path] = assess_crimaldi_quality(file_path)
        print(f"Quality assessment completed for: {file_path}")
        print(f"Overall quality score: {quality_results[file_path]['overall_quality']['composite_score']:.3f}")
    else:
        print(f"Warning: File not found: {file_path}")

# Quality assessment summary across all files
if quality_results:
    overall_scores = [result['overall_quality']['composite_score'] for result in quality_results.values()]
    average_quality = np.mean(overall_scores)
    min_quality = np.min(overall_scores)
    max_quality = np.max(overall_scores)
    
    print(f"\nQuality Assessment Summary:")
    print(f"  Average quality score: {average_quality:.3f}")
    print(f"  Quality range: {min_quality:.3f} - {max_quality:.3f}")
    print(f"  Files meeting quality threshold (>0.95): {sum(1 for score in overall_scores if score > 0.95)}/{len(overall_scores)}")
```

## Configuration Setup

### Normalization Configuration

Detailed configuration setup for Crimaldi-specific normalization parameters optimized for scientific accuracy:

```json
{
  "format_specific_settings": {
    "crimaldi_format": {
      "pixel_to_meter_ratio": 100.0,
      "intensity_units": "concentration_ppm",
      "coordinate_system": "cartesian",
      "coordinate_origin": "bottom_left",
      "time_units": "seconds",
      "frame_rate_hz": 50.0,
      "bit_depth": 16,
      "color_space": "grayscale",
      "compression": "lossless",
      "metadata_validation": {
        "require_calibration_parameters": true,
        "validate_arena_dimensions": true,
        "check_temporal_consistency": true,
        "verify_intensity_calibration": true
      },
      "quality_thresholds": {
        "minimum_frame_rate": 25.0,
        "maximum_frame_rate": 60.0,
        "minimum_resolution": [320, 240],
        "maximum_resolution": [1920, 1080],
        "minimum_bit_depth": 8,
        "calibration_accuracy_threshold": 0.95
      }
    }
  },
  "arena_normalization": {
    "target_arena_width_meters": 1.0,
    "target_arena_height_meters": 1.0,
    "scaling_method": "bicubic",
    "preserve_aspect_ratio": true,
    "boundary_detection": {
      "automatic_detection": true,
      "edge_detection_threshold": 0.1,
      "morphological_operations": true,
      "validation_tolerance": 0.02
    },
    "coordinate_transformation": {
      "target_origin": "bottom_left",
      "coordinate_scaling": "meters",
      "precision_digits": 6,
      "validation_checks": true
    }
  },
  "intensity_calibration": {
    "target_intensity_range": [0.0, 1.0],
    "calibration_method": "min_max",
    "background_subtraction": false,
    "gamma_correction": 1.0,
    "noise_reduction": {
      "enable_denoising": true,
      "denoising_method": "gaussian",
      "kernel_size": 3,
      "sigma": 0.5
    },
    "calibration_validation": {
      "reference_concentration_check": true,
      "linearity_validation": true,
      "dynamic_range_assessment": true,
      "cross_validation_samples": 100
    }
  },
  "temporal_normalization": {
    "target_frame_rate": 30.0,
    "resampling_method": "cubic_interpolation",
    "temporal_alignment": {
      "enable_frame_alignment": true,
      "alignment_tolerance": 1e-6,
      "synchronization_validation": true
    },
    "temporal_filtering": {
      "enable_temporal_smoothing": false,
      "smoothing_window_size": 3,
      "preserve_temporal_features": true
    }
  },
  "performance_optimization": {
    "enable_parallel_processing": true,
    "max_workers": 8,
    "memory_mapping": true,
    "chunk_size_frames": 1000,
    "disk_caching": {
      "enable_caching": true,
      "cache_directory": "cache/crimaldi_processing",
      "cache_size_limit_gb": 10,
      "cache_expiration_hours": 24
    }
  },
  "validation_settings": {
    "numerical_precision": {
      "floating_point_precision": "float64",
      "relative_error_threshold": 1e-6,
      "absolute_error_threshold": 1e-9
    },
    "cross_format_validation": {
      "enable_reference_comparison": true,
      "correlation_threshold": 0.95,
      "statistical_significance": 0.05
    },
    "quality_assurance": {
      "enable_quality_checks": true,
      "quality_threshold": 0.95,
      "generate_quality_reports": true,
      "automatic_quality_adjustment": false
    }
  },
  "logging_configuration": {
    "log_level": "INFO",
    "enable_detailed_logging": true,
    "log_file_path": "logs/crimaldi_normalization.log",
    "include_performance_metrics": true,
    "log_rotation": {
      "max_file_size_mb": 100,
      "backup_count": 5
    }
  }
}
```

### Simulation Parameters

Comprehensive simulation parameter configuration optimized for Crimaldi dataset characteristics:

```json
{
  "simulation_parameters": {
    "algorithm_timeout_seconds": 300,
    "max_simulation_steps": 10000,
    "convergence_threshold": 0.01,
    "random_seed": 42,
    "numerical_precision": {
      "position_precision": 1e-6,
      "concentration_precision": 1e-9,
      "time_step_precision": 1e-6,
      "gradient_calculation_precision": 1e-8
    },
    "termination_criteria": {
      "source_localization_radius": 0.05,
      "maximum_simulation_time": 300.0,
      "minimum_concentration_threshold": 1e-6,
      "trajectory_convergence_tolerance": 1e-4
    }
  },
  "crimaldi_specific": {
    "coordinate_system": "cartesian",
    "arena_boundaries": [1.0, 1.0],
    "concentration_threshold": 0.1,
    "temporal_resolution": 0.02,
    "physical_parameters": {
      "agent_size_meters": 0.01,
      "sensor_sensitivity": 1e-6,
      "movement_speed_mps": 0.1,
      "turning_rate_rad_per_sec": 1.0
    },
    "environmental_parameters": {
      "wind_field_interpolation": "bicubic",
      "concentration_interpolation": "linear",
      "boundary_conditions": "reflective",
      "source_localization_accuracy": 0.01
    }
  },
  "algorithm_specific_parameters": {
    "infotaxis": {
      "information_gain_threshold": 0.01,
      "exploration_bias": 0.1,
      "memory_decay_rate": 0.95,
      "planning_horizon_steps": 10
    },
    "casting": {
      "casting_radius": 0.2,
      "casting_angle_degrees": 45,
      "upwind_bias": 0.8,
      "surge_distance": 0.1
    },
    "gradient_following": {
      "gradient_estimation_window": 5,
      "gradient_threshold": 1e-6,
      "step_size_multiplier": 0.1,
      "momentum_coefficient": 0.9
    }
  },
  "performance_optimization": {
    "parallel_simulations": true,
    "max_workers": 8,
    "memory_limit_gb": 8,
    "checkpoint_interval": 100,
    "progress_reporting": {
      "update_frequency_seconds": 10,
      "detailed_progress_logging": true,
      "performance_metrics_collection": true
    },
    "resource_management": {
      "memory_monitoring": true,
      "cpu_utilization_target": 0.8,
      "disk_space_monitoring": true,
      "automatic_resource_scaling": false
    }
  },
  "validation_parameters": {
    "cross_validation": {
      "enable_cross_validation": true,
      "validation_splits": 5,
      "stratified_sampling": true,
      "validation_metrics": ["success_rate", "path_efficiency", "time_to_target"]
    },
    "statistical_analysis": {
      "confidence_interval": 0.95,
      "bootstrap_samples": 1000,
      "significance_testing": true,
      "effect_size_calculation": true
    },
    "reproducibility": {
      "deterministic_execution": true,
      "seed_management": true,
      "environment_validation": true,
      "result_verification": true
    }
  },
  "output_configuration": {
    "trajectory_recording": {
      "record_full_trajectories": true,
      "trajectory_sampling_rate": 0.1,
      "include_sensor_data": true,
      "coordinate_precision": 6
    },
    "performance_metrics": {
      "calculate_path_efficiency": true,
      "measure_exploration_coverage": true,
      "track_computational_performance": true,
      "include_statistical_summaries": true
    },
    "result_formats": {
      "primary_format": "numpy_arrays",
      "backup_format": "csv",
      "include_metadata": true,
      "compression": "gzip"
    }
  },
  "error_handling": {
    "simulation_failure_handling": {
      "retry_attempts": 3,
      "retry_delay_seconds": 1,
      "failure_analysis": true,
      "partial_result_preservation": true
    },
    "numerical_stability": {
      "overflow_detection": true,
      "underflow_handling": true,
      "nan_detection": true,
      "infinite_value_handling": true
    },
    "recovery_mechanisms": {
      "checkpoint_restoration": true,
      "graceful_degradation": true,
      "automatic_parameter_adjustment": false,
      "manual_intervention_alerts": true
    }
  }
}
```

### Algorithm Selection

Detailed algorithm selection and configuration for Crimaldi dataset processing:

```python
from src.backend.algorithms.algorithm_registry import list_algorithms, create_algorithm_instance
from src.backend.core.algorithm_configuration import AlgorithmConfigurationManager
import json

def configure_algorithms_for_crimaldi(dataset_characteristics):
    """
    Configure navigation algorithms optimized for Crimaldi dataset characteristics.
    
    Args:
        dataset_characteristics (dict): Crimaldi dataset properties and constraints
        
    Returns:
        dict: Configured algorithm instances with optimized parameters
    """
    # Available algorithms with Crimaldi compatibility
    available_algorithms = list_algorithms(format_compatibility='crimaldi')
    print(f"Available algorithms for Crimaldi processing: {available_algorithms}")
    
    # Algorithm-specific configuration for Crimaldi datasets
    crimaldi_algorithm_configs = {
        'infotaxis': {
            'sensor_noise_std': dataset_characteristics.get('sensor_noise_level', 0.01),
            'concentration_threshold': dataset_characteristics.get('detection_threshold', 1e-6),
            'information_gain_weight': 1.0,
            'exploration_coefficient': 0.1,
            'belief_update_method': 'bayesian',
            'spatial_resolution': dataset_characteristics.get('spatial_resolution', 0.01),
            'temporal_integration_window': dataset_characteristics.get('temporal_window', 5)
        },
        'casting': {
            'surge_distance': dataset_characteristics.get('arena_size', 1.0) * 0.1,
            'casting_radius': dataset_characteristics.get('arena_size', 1.0) * 0.2,
            'upwind_bias_strength': 0.8,
            'crosswind_casting_angle': 45.0,
            'concentration_memory_decay': 0.95,
            'wind_direction_estimation': 'gradient_based',
            'boundary_avoidance_margin': 0.05
        },
        'gradient_following': {
            'gradient_estimation_method': 'finite_difference',
            'step_size': dataset_characteristics.get('spatial_resolution', 0.01),
            'momentum_coefficient': 0.9,
            'gradient_noise_filtering': True,
            'adaptive_step_size': True,
            'local_minima_escape': 'random_walk',
            'convergence_tolerance': 1e-6
        },
        'spiral_search': {
            'spiral_radius_initial': dataset_characteristics.get('arena_size', 1.0) * 0.05,
            'spiral_pitch': dataset_characteristics.get('spatial_resolution', 0.01) * 2,
            'spiral_direction': 'counterclockwise',
            'expansion_rate': 1.2,
            'concentration_triggered_transition': True,
            'transition_threshold': dataset_characteristics.get('detection_threshold', 1e-6)
        },
        'particle_filter': {
            'particle_count': 1000,
            'resampling_threshold': 0.5,
            'process_noise_std': 0.01,
            'measurement_noise_std': dataset_characteristics.get('sensor_noise_level', 0.01),
            'initialization_strategy': 'uniform_random',
            'particle_diversity_maintenance': True,
            'effective_sample_size_threshold': 100
        }
    }
    
    # Initialize configuration manager
    config_manager = AlgorithmConfigurationManager()
    
    # Configure each algorithm with Crimaldi-specific parameters
    configured_algorithms = {}
    for algorithm_name in available_algorithms:
        if algorithm_name in crimaldi_algorithm_configs:
            # Create algorithm instance with configuration
            algorithm_config = crimaldi_algorithm_configs[algorithm_name]
            algorithm_instance = create_algorithm_instance(
                algorithm_name, 
                config=algorithm_config,
                validation_mode=True
            )
            
            # Validate configuration compatibility
            validation_result = config_manager.validate_algorithm_config(
                algorithm_name, 
                algorithm_config, 
                dataset_type='crimaldi'
            )
            
            if validation_result['valid']:
                configured_algorithms[algorithm_name] = {
                    'instance': algorithm_instance,
                    'config': algorithm_config,
                    'validation': validation_result,
                    'crimaldi_compatibility_score': validation_result.get('compatibility_score', 0.0)
                }
                print(f"Algorithm '{algorithm_name}' configured successfully")
                print(f"  Compatibility score: {validation_result.get('compatibility_score', 0.0):.3f}")
            else:
                print(f"Warning: Algorithm '{algorithm_name}' configuration validation failed")
                print(f"  Validation errors: {validation_result.get('errors', [])}")
    
    return configured_algorithms

# Example: Configure algorithms for sample Crimaldi dataset
sample_dataset_characteristics = {
    'arena_size': 1.0,  # meters
    'spatial_resolution': 0.01,  # meters per pixel
    'temporal_resolution': 0.033,  # seconds per frame (30 fps)
    'sensor_noise_level': 0.02,  # relative noise standard deviation
    'detection_threshold': 1e-6,  # minimum detectable concentration
    'concentration_range': [0.0, 1.0],  # normalized concentration range
    'wind_field_complexity': 'moderate',  # qualitative assessment
    'plume_characteristics': {
        'source_strength': 1.0,
        'diffusion_coefficient': 0.1,
        'advection_velocity': 0.2,
        'intermittency_factor': 0.8
    }
}

# Configure algorithms
configured_algorithms = configure_algorithms_for_crimaldi(sample_dataset_characteristics)

# Display configuration summary
print(f"\nAlgorithm Configuration Summary:")
print(f"Successfully configured: {len(configured_algorithms)} algorithms")
for name, config in configured_algorithms.items():
    print(f"  {name}: Compatibility {config['crimaldi_compatibility_score']:.3f}")

# Save algorithm configurations for batch processing
algorithm_config_file = 'config/crimaldi_algorithm_configurations.json'
with open(algorithm_config_file, 'w') as f:
    # Serialize configurations (excluding non-serializable instances)
    serializable_configs = {
        name: {
            'config': config['config'],
            'validation': config['validation'],
            'compatibility_score': config['crimaldi_compatibility_score']
        }
        for name, config in configured_algorithms.items()
    }
    json.dump(serializable_configs, f, indent=2)

print(f"Algorithm configurations saved to: {algorithm_config_file}")
```

### Performance Optimization

Advanced performance optimization strategies for Crimaldi dataset processing:

```python
from src.backend.core.performance_optimizer import CrimaldiPerformanceOptimizer
from src.backend.utils.resource_monitor import ResourceMonitor
import psutil
import numpy as np

def optimize_crimaldi_processing_performance(dataset_info, system_capabilities):
    """
    Optimize processing performance for Crimaldi dataset characteristics.
    
    Args:
        dataset_info (dict): Crimaldi dataset properties and processing requirements
        system_capabilities (dict): Available system resources and capabilities
        
    Returns:
        dict: Optimized configuration parameters and performance recommendations
    """
    # Initialize performance optimizer
    optimizer = CrimaldiPerformanceOptimizer(dataset_info, system_capabilities)
    
    # Analyze dataset processing requirements
    processing_requirements = optimizer.analyze_processing_requirements()
    print(f"Processing Requirements Analysis:")
    print(f"  Estimated memory usage: {processing_requirements['memory_gb']:.2f} GB")
    print(f"  Estimated processing time: {processing_requirements['time_hours']:.2f} hours")
    print(f"  Computational complexity: {processing_requirements['complexity_score']:.2f}")
    
    # Optimize parallel processing configuration
    parallel_config = optimizer.optimize_parallel_processing()
    print(f"\nParallel Processing Optimization:")
    print(f"  Recommended workers: {parallel_config['optimal_workers']}")
    print(f"  Memory per worker: {parallel_config['memory_per_worker_gb']:.2f} GB")
    print(f"  Chunk size: {parallel_config['optimal_chunk_size']} frames")
    print(f"  Expected speedup: {parallel_config['expected_speedup']:.1f}x")
    
    # Memory optimization strategy
    memory_config = optimizer.optimize_memory_usage()
    print(f"\nMemory Optimization:")
    print(f"  Enable memory mapping: {memory_config['use_memory_mapping']}")
    print(f"  Caching strategy: {memory_config['caching_strategy']}")
    print(f"  Buffer size: {memory_config['buffer_size_mb']} MB")
    print(f"  Garbage collection frequency: {memory_config['gc_frequency']}")
    
    # I/O optimization parameters
    io_config = optimizer.optimize_io_operations()
    print(f"\nI/O Optimization:")
    print(f"  Read buffer size: {io_config['read_buffer_size_mb']} MB")
    print(f"  Write buffer size: {io_config['write_buffer_size_mb']} MB")
    print(f"  Prefetch enabled: {io_config['enable_prefetch']}")
    print(f"  Compression level: {io_config['compression_level']}")
    
    # Algorithm-specific optimizations
    algorithm_optimizations = optimizer.optimize_algorithm_performance()
    print(f"\nAlgorithm Performance Optimizations:")
    for algorithm, config in algorithm_optimizations.items():
        print(f"  {algorithm}:")
        print(f"    Vectorization level: {config['vectorization_level']}")
        print(f"    Cache optimization: {config['cache_friendly']}")
        print(f"    Numerical precision: {config['precision_level']}")
    
    # Compile comprehensive optimization configuration
    optimization_config = {
        'parallel_processing': parallel_config,
        'memory_optimization': memory_config,
        'io_optimization': io_config,
        'algorithm_optimizations': algorithm_optimizations,
        'performance_monitoring': {
            'enable_profiling': True,
            'metrics_collection_interval': 10,
            'resource_usage_tracking': True,
            'bottleneck_detection': True
        },
        'adaptive_optimization': {
            'enable_runtime_adaptation': True,
            'performance_threshold_monitoring': True,
            'automatic_parameter_tuning': False,
            'optimization_feedback_loop': True
        }
    }
    
    return optimization_config

# System capability assessment
def assess_system_capabilities():
    """Assess available system resources for optimization."""
    return {
        'cpu_cores': psutil.cpu_count(logical=False),
        'logical_processors': psutil.cpu_count(logical=True),
        'total_memory_gb': psutil.virtual_memory().total / (1024**3),
        'available_memory_gb': psutil.virtual_memory().available / (1024**3),
        'disk_space_gb': psutil.disk_usage('/').free / (1024**3),
        'cpu_frequency_ghz': psutil.cpu_freq().current / 1000 if psutil.cpu_freq() else 2.5
    }

# Dataset information for optimization
crimaldi_dataset_info = {
    'total_files': 10,
    'average_file_size_gb': 2.5,
    'total_frames': 50000,
    'frame_resolution': [640, 480],
    'frame_rate': 30.0,
    'bit_depth': 16,
    'compression_ratio': 0.8,
    'processing_complexity': 'high',
    'target_algorithms': ['infotaxis', 'casting', 'gradient_following'],
    'simulation_count_per_file': 100,
    'statistical_analysis_complexity': 'comprehensive'
}

# Assess system and optimize performance
system_caps = assess_system_capabilities()
optimization_config = optimize_crimaldi_processing_performance(crimaldi_dataset_info, system_caps)

# Save optimization configuration
optimization_config_file = 'config/crimaldi_performance_optimization.json'
with open(optimization_config_file, 'w') as f:
    json.dump(optimization_config, f, indent=2)

print(f"\nPerformance optimization configuration saved to: {optimization_config_file}")

# Resource monitoring setup
resource_monitor = ResourceMonitor(
    monitoring_interval=5,
    log_file='logs/crimaldi_resource_usage.log',
    alert_thresholds={
        'cpu_usage_percent': 90,
        'memory_usage_percent': 85,
        'disk_usage_percent': 90
    }
)

print("Resource monitoring configured for Crimaldi processing")
```

## Single File Processing Example

### File Loading and Validation

Comprehensive example demonstrating single Crimaldi file processing with detailed validation and error handling:

```python
from src.backend.io.crimaldi_format_handler import CrimaldiFormatHandler, detect_crimaldi_format
from src.backend.core.data_normalization import normalize_plume_data
from src.backend.utils.logging_utils import get_logger, LoggingContext
from pathlib import Path
import time
import numpy as np

def process_single_crimaldi_file(file_path, output_path, config_path, verbose=True):
    """
    Complete processing workflow for a single Crimaldi dataset file.
    
    Args:
        file_path (str): Path to input Crimaldi video file
        output_path (str): Path for normalized output file
        config_path (str): Path to normalization configuration file
        verbose (bool): Enable detailed logging output
        
    Returns:
        dict: Comprehensive processing results with quality metrics and timing
    """
    start_time = time.time()
    
    # Initialize logging context for traceability
    with LoggingContext('single_crimaldi_processing', enable_detailed_logging=verbose):
        logger = get_logger()
        logger.info(f"Starting Crimaldi file processing: {file_path}")
        
        # Step 1: File existence and accessibility validation
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Crimaldi file not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        logger.info(f"File validation passed: {file_path.name}")
        logger.info(f"File size: {file_path.stat().st_size / (1024**2):.2f} MB")
        
        # Step 2: Crimaldi format detection with confidence assessment
        logger.info("Performing Crimaldi format detection...")
        format_detection_start = time.time()
        
        format_info = detect_crimaldi_format(file_path, deep_inspection=True)
        format_detection_time = time.time() - format_detection_start
        
        logger.info(f"Format detection completed in {format_detection_time:.3f} seconds")
        logger.info(f"Detected format: {format_info['format']}")
        logger.info(f"Detection confidence: {format_info['confidence']:.4f}")
        
        if format_info['confidence'] < 0.95:
            logger.warning(f"Low format detection confidence: {format_info['confidence']:.4f}")
            logger.warning("Proceeding with manual format validation...")
        
        # Step 3: Initialize Crimaldi format handler with caching
        logger.info("Initializing Crimaldi format handler...")
        handler_init_start = time.time()
        
        handler = CrimaldiFormatHandler(
            file_path, 
            enable_caching=True,
            cache_directory='cache/crimaldi_handlers',
            validation_mode=True
        )
        
        handler_init_time = time.time() - handler_init_start
        logger.info(f"Handler initialization completed in {handler_init_time:.3f} seconds")
        
        # Step 4: Extract and validate video properties
        logger.info("Extracting video properties...")
        video_properties = handler.get_video_properties()
        
        logger.info(f"Video properties extracted:")
        logger.info(f"  Duration: {video_properties['duration_seconds']:.2f} seconds")
        logger.info(f"  Dimensions: {video_properties['width']}x{video_properties['height']}")
        logger.info(f"  Frame rate: {video_properties['fps']:.2f} fps")
        logger.info(f"  Total frames: {video_properties['frame_count']}")
        logger.info(f"  Bit depth: {video_properties.get('bit_depth', 'Unknown')}")
        logger.info(f"  Codec: {video_properties.get('codec', 'Unknown')}")
        
        # Step 5: Extract calibration parameters with accuracy validation
        logger.info("Extracting calibration parameters...")
        calibration_extraction_start = time.time()
        
        calibration = handler.get_calibration_parameters(validate_accuracy=True)
        calibration_extraction_time = time.time() - calibration_extraction_start
        
        logger.info(f"Calibration extraction completed in {calibration_extraction_time:.3f} seconds")
        logger.info(f"Calibration parameters:")
        logger.info(f"  Pixel-to-meter ratio: {calibration['pixel_to_meter_ratio']:.2f}")
        logger.info(f"  Arena dimensions: {calibration['arena_width_meters']:.3f}x{calibration['arena_height_meters']:.3f} meters")
        logger.info(f"  Intensity units: {calibration['intensity_units']}")
        logger.info(f"  Coordinate system: {calibration['coordinate_system']}")
        logger.info(f"  Coordinate origin: {calibration['coordinate_origin']}")
        
        # Validate calibration accuracy
        calibration_accuracy = calibration.get('accuracy_score', 0.0)
        if calibration_accuracy < 0.95:
            logger.warning(f"Calibration accuracy below threshold: {calibration_accuracy:.3f}")
        else:
            logger.info(f"Calibration accuracy: {calibration_accuracy:.3f} (PASS)")
        
        # Step 6: Normalization pipeline execution
        logger.info("Starting normalization pipeline...")
        normalization_start = time.time()
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Execute normalization with comprehensive configuration
        normalization_result = normalize_plume_data(
            input_file_path=file_path,
            output_path=output_path,
            config_path=config_path,
            validation_mode=True,
            preserve_metadata=True,
            generate_quality_report=True
        )
        
        normalization_time = time.time() - normalization_start
        logger.info(f"Normalization completed in {normalization_time:.3f} seconds")
        
        # Step 7: Quality assessment and validation
        logger.info("Performing quality assessment...")
        quality_assessment_start = time.time()
        
        overall_quality_score = normalization_result.calculate_overall_quality_score()
        spatial_quality = normalization_result.get_spatial_quality_metrics()
        temporal_quality = normalization_result.get_temporal_quality_metrics()
        intensity_quality = normalization_result.get_intensity_quality_metrics()
        
        quality_assessment_time = time.time() - quality_assessment_start
        
        logger.info(f"Quality assessment completed in {quality_assessment_time:.3f} seconds")
        logger.info(f"Quality metrics:")
        logger.info(f"  Overall quality score: {overall_quality_score:.4f}")
        logger.info(f"  Spatial quality: {spatial_quality.get('overall_score', 0.0):.4f}")
        logger.info(f"  Temporal quality: {temporal_quality.get('overall_score', 0.0):.4f}")
        logger.info(f"  Intensity quality: {intensity_quality.get('overall_score', 0.0):.4f}")
        
        # Validate quality thresholds
        quality_thresholds = {
            'overall_quality': 0.95,
            'spatial_quality': 0.95,
            'temporal_quality': 0.99,
            'intensity_quality': 0.98
        }
        
        quality_validation = {
            'overall_quality': overall_quality_score >= quality_thresholds['overall_quality'],
            'spatial_quality': spatial_quality.get('overall_score', 0.0) >= quality_thresholds['spatial_quality'],
            'temporal_quality': temporal_quality.get('overall_score', 0.0) >= quality_thresholds['temporal_quality'],
            'intensity_quality': intensity_quality.get('overall_score', 0.0) >= quality_thresholds['intensity_quality']
        }
        
        all_quality_checks_passed = all(quality_validation.values())
        
        if all_quality_checks_passed:
            logger.info("All quality checks PASSED")
        else:
            failed_checks = [check for check, passed in quality_validation.items() if not passed]
            logger.warning(f"Quality checks FAILED: {failed_checks}")
        
        # Step 8: Generate processing summary
        total_processing_time = time.time() - start_time
        
        processing_summary = {
            'input_file': str(file_path),
            'output_file': str(output_path),
            'config_file': str(config_path),
            'format_detection': {
                'format': format_info['format'],
                'confidence': format_info['confidence'],
                'detection_time_seconds': format_detection_time
            },
            'video_properties': video_properties,
            'calibration_parameters': calibration,
            'calibration_accuracy': calibration_accuracy,
            'normalization_results': {
                'success': normalization_result.success,
                'processing_time_seconds': normalization_time,
                'output_file_size_mb': output_path.stat().st_size / (1024**2) if output_path.exists() else 0.0
            },
            'quality_assessment': {
                'overall_quality_score': overall_quality_score,
                'spatial_quality_score': spatial_quality.get('overall_score', 0.0),
                'temporal_quality_score': temporal_quality.get('overall_score', 0.0),
                'intensity_quality_score': intensity_quality.get('overall_score', 0.0),
                'quality_validation': quality_validation,
                'all_checks_passed': all_quality_checks_passed,
                'assessment_time_seconds': quality_assessment_time
            },
            'timing_breakdown': {
                'format_detection_seconds': format_detection_time,
                'handler_initialization_seconds': handler_init_time,
                'calibration_extraction_seconds': calibration_extraction_time,
                'normalization_seconds': normalization_time,
                'quality_assessment_seconds': quality_assessment_time,
                'total_processing_seconds': total_processing_time
            },
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processing_environment': {
                'python_version': handler.get_python_version(),
                'opencv_version': handler.get_opencv_version(),
                'numpy_version': handler.get_numpy_version()
            }
        }
        
        logger.info(f"Single file processing completed successfully")
        logger.info(f"Total processing time: {total_processing_time:.2f} seconds")
        logger.info(f"Processing rate: {video_properties['frame_count'] / total_processing_time:.1f} frames/second")
        
        return processing_summary

# Example usage: Process a single Crimaldi file
if __name__ == "__main__":
    # Define file paths
    sample_file = 'data/crimaldi_samples/crimaldi_sample.avi'
    normalized_output = 'data/normalized/crimaldi_sample_normalized.npy'
    config_file = 'config/crimaldi_normalization_config.json'
    
    try:
        # Process single file with comprehensive validation
        result = process_single_crimaldi_file(
            file_path=sample_file,
            output_path=normalized_output,
            config_path=config_file,
            verbose=True
        )
        
        # Display processing summary
        print(f"\n{'='*60}")
        print("SINGLE FILE PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Input file: {result['input_file']}")
        print(f"Output file: {result['output_file']}")
        print(f"Format detection confidence: {result['format_detection']['confidence']:.4f}")
        print(f"Calibration accuracy: {result['calibration_accuracy']:.4f}")
        print(f"Overall quality score: {result['quality_assessment']['overall_quality_score']:.4f}")
        print(f"Quality validation: {'PASS' if result['quality_assessment']['all_checks_passed'] else 'FAIL'}")
        print(f"Processing time: {result['timing_breakdown']['total_processing_seconds']:.2f} seconds")
        print(f"Processing rate: {result['video_properties']['frame_count'] / result['timing_breakdown']['total_processing_seconds']:.1f} fps")
        
        # Save detailed results
        import json
        results_file = 'results/single_file_processing_results.json'
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"Detailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during single file processing: {e}")
        import traceback
        traceback.print_exc()
```

### Format Detection

Advanced format detection with deep inspection and confidence assessment:

```python
from src.backend.io.crimaldi_format_handler import detect_crimaldi_format, CrimaldiFormatValidator
from src.backend.utils.format_detection import VideoFormatAnalyzer
import cv2
import numpy as np

def comprehensive_format_detection(file_path, enable_deep_analysis=True):
    """
    Comprehensive Crimaldi format detection with detailed analysis and validation.
    
    Args:
        file_path (str): Path to video file for format detection
        enable_deep_analysis (bool): Enable deep inspection of video content
        
    Returns:
        dict: Detailed format detection results with confidence metrics and recommendations
    """
    file_path = Path(file_path)
    
    # Step 1: Basic file validation
    if not file_path.exists():
        return {
            'format': 'unknown',
            'confidence': 0.0,
            'error': 'File not found',
            'recommendations': ['Verify file path and accessibility']
        }
    
    # Step 2: File extension and container analysis
    container_analysis = VideoFormatAnalyzer.analyze_container(file_path)
    print(f"Container analysis:")
    print(f"  File extension: {container_analysis['extension']}")
    print(f"  Container format: {container_analysis['container_format']}")
    print(f"  Container compatibility: {container_analysis['crimaldi_compatibility']:.3f}")
    
    # Step 3: Primary format detection
    primary_detection = detect_crimaldi_format(
        file_path, 
        deep_inspection=enable_deep_analysis,
        confidence_threshold=0.8
    )
    
    print(f"Primary format detection:")
    print(f"  Detected format: {primary_detection['format']}")
    print(f"  Confidence score: {primary_detection['confidence']:.4f}")
    print(f"  Detection method: {primary_detection.get('detection_method', 'standard')}")
    
    if enable_deep_analysis:
        # Step 4: Deep content analysis
        deep_analysis_result = perform_deep_content_analysis(file_path)
        
        # Step 5: Metadata signature verification
        metadata_verification = verify_crimaldi_metadata_signature(file_path)
        
        # Step 6: Video stream analysis
        stream_analysis = analyze_video_streams(file_path)
        
        # Step 7: Calibration marker detection
        calibration_markers = detect_calibration_markers(file_path)
        
        # Comprehensive confidence calculation
        confidence_factors = {
            'container_compatibility': container_analysis['crimaldi_compatibility'] * 0.15,
            'primary_detection': primary_detection['confidence'] * 0.25,
            'content_analysis': deep_analysis_result['crimaldi_likelihood'] * 0.20,
            'metadata_signature': metadata_verification['signature_match'] * 0.15,
            'stream_analysis': stream_analysis['compatibility_score'] * 0.15,
            'calibration_markers': calibration_markers['detection_confidence'] * 0.10
        }
        
        composite_confidence = sum(confidence_factors.values())
        
        # Detailed analysis summary
        detailed_results = {
            'format': 'crimaldi' if composite_confidence > 0.85 else 'unknown',
            'confidence': composite_confidence,
            'confidence_breakdown': confidence_factors,
            'container_analysis': container_analysis,
            'primary_detection': primary_detection,
            'deep_content_analysis': deep_analysis_result,
            'metadata_verification': metadata_verification,
            'stream_analysis': stream_analysis,
            'calibration_markers': calibration_markers,
            'detection_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_recommendations': generate_detection_recommendations(
                composite_confidence, confidence_factors
            )
        }
        
    else:
        # Standard detection without deep analysis
        detailed_results = {
            'format': primary_detection['format'],
            'confidence': primary_detection['confidence'],
            'container_analysis': container_analysis,
            'primary_detection': primary_detection,
            'detection_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_recommendations': generate_basic_recommendations(primary_detection['confidence'])
        }
    
    return detailed_results

def perform_deep_content_analysis(file_path):
    """Perform deep analysis of video content for Crimaldi format characteristics."""
    cap = cv2.VideoCapture(str(file_path))
    
    if not cap.isOpened():
        return {'crimaldi_likelihood': 0.0, 'error': 'Could not open video file'}
    
    # Sample frames for analysis
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, frame_count - 1, min(50, frame_count), dtype=int)
    
    content_indicators = {
        'grayscale_consistency': 0.0,
        'intensity_distribution': 0.0,
        'spatial_patterns': 0.0,
        'temporal_consistency': 0.0
    }
    
    frames_analyzed = 0
    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Analyze frame characteristics
            content_indicators['grayscale_consistency'] += analyze_grayscale_consistency(frame)
            content_indicators['intensity_distribution'] += analyze_intensity_distribution(frame)
            content_indicators['spatial_patterns'] += analyze_spatial_patterns(frame)
            frames_analyzed += 1
    
    cap.release()
    
    if frames_analyzed > 0:
        # Average the indicators
        for key in content_indicators:
            content_indicators[key] /= frames_analyzed
        
        # Calculate composite likelihood
        crimaldi_likelihood = np.mean(list(content_indicators.values()))
    else:
        crimaldi_likelihood = 0.0
    
    return {
        'crimaldi_likelihood': crimaldi_likelihood,
        'content_indicators': content_indicators,
        'frames_analyzed': frames_analyzed,
        'total_frames': frame_count
    }

def analyze_grayscale_consistency(frame):
    """Analyze grayscale consistency indicative of Crimaldi format."""
    if len(frame.shape) == 3:
        # Check if image is effectively grayscale
        b, g, r = cv2.split(frame)
        grayscale_diff = np.mean(np.abs(b.astype(float) - g.astype(float))) + \
                        np.mean(np.abs(g.astype(float) - r.astype(float))) + \
                        np.mean(np.abs(r.astype(float) - b.astype(float)))
        grayscale_consistency = max(0.0, 1.0 - grayscale_diff / 255.0)
    else:
        grayscale_consistency = 1.0  # Already grayscale
    
    return grayscale_consistency

def analyze_intensity_distribution(frame):
    """Analyze intensity distribution patterns characteristic of Crimaldi recordings."""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_normalized = hist / np.sum(hist)
    
    # Analyze distribution characteristics
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
    dynamic_range = np.max(gray) - np.min(gray)
    
    # Crimaldi recordings typically have moderate entropy and good dynamic range
    entropy_score = min(1.0, entropy / 8.0)  # Normalize to 0-1
    dynamic_range_score = min(1.0, dynamic_range / 255.0)
    
    return (entropy_score + dynamic_range_score) / 2.0

def analyze_spatial_patterns(frame):
    """Analyze spatial patterns indicative of plume structures."""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Calculate gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Analyze gradient distribution
    gradient_mean = np.mean(gradient_magnitude)
    gradient_std = np.std(gradient_magnitude)
    
    # Plume structures typically have moderate gradient variation
    spatial_complexity = min(1.0, gradient_std / (gradient_mean + 1e-10))
    
    return min(1.0, spatial_complexity)

def verify_crimaldi_metadata_signature(file_path):
    """Verify Crimaldi-specific metadata signatures."""
    try:
        cap = cv2.VideoCapture(str(file_path))
        
        # Extract codec information
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        # Check for Crimaldi-compatible codecs
        crimaldi_codecs = ['MJPG', 'H264', 'XVID', 'DIV3']
        codec_match = codec in crimaldi_codecs
        
        # Additional metadata checks would go here
        # (This is a simplified example)
        
        cap.release()
        
        return {
            'signature_match': 1.0 if codec_match else 0.3,
            'codec_detected': codec,
            'codec_compatibility': codec_match
        }
        
    except Exception as e:
        return {
            'signature_match': 0.0,
            'error': str(e)
        }

def analyze_video_streams(file_path):
    """Analyze video stream properties for Crimaldi compatibility."""
    cap = cv2.VideoCapture(str(file_path))
    
    properties = {
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    
    cap.release()
    
    # Evaluate compatibility based on typical Crimaldi specifications
    compatibility_factors = {
        'resolution': evaluate_resolution_compatibility(properties['width'], properties['height']),
        'frame_rate': evaluate_frame_rate_compatibility(properties['fps']),
        'duration': evaluate_duration_compatibility(properties['frame_count'], properties['fps'])
    }
    
    compatibility_score = np.mean(list(compatibility_factors.values()))
    
    return {
        'compatibility_score': compatibility_score,
        'compatibility_factors': compatibility_factors,
        'stream_properties': properties
    }

def evaluate_resolution_compatibility(width, height):
    """Evaluate resolution compatibility with Crimaldi standards."""
    # Common Crimaldi resolutions
    standard_resolutions = [(640, 480), (800, 600), (1024, 768), (320, 240)]
    
    for std_w, std_h in standard_resolutions:
        if width == std_w and height == std_h:
            return 1.0
    
    # Check aspect ratio compatibility
    aspect_ratio = width / height
    standard_aspect = 4.0 / 3.0  # Common for Crimaldi recordings
    
    aspect_compatibility = max(0.0, 1.0 - abs(aspect_ratio - standard_aspect) / standard_aspect)
    
    return aspect_compatibility

def evaluate_frame_rate_compatibility(fps):
    """Evaluate frame rate compatibility with Crimaldi standards."""
    # Typical Crimaldi frame rates
    if 25 <= fps <= 60:
        return 1.0
    elif 15 <= fps < 25 or 60 < fps <= 100:
        return 0.7
    else:
        return 0.3

def evaluate_duration_compatibility(frame_count, fps):
    """Evaluate video duration compatibility."""
    duration = frame_count / fps if fps > 0 else 0
    
    # Typical experimental durations
    if 30 <= duration <= 600:  # 30 seconds to 10 minutes
        return 1.0
    elif 10 <= duration < 30 or 600 < duration <= 1800:
        return 0.8
    else:
        return 0.5

def detect_calibration_markers(file_path):
    """Detect calibration markers or patterns in video."""
    cap = cv2.VideoCapture(str(file_path))
    
    # Sample first few frames for calibration marker detection
    calibration_indicators = []
    
    for frame_idx in range(min(10, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if ret:
            # Look for geometric patterns that might indicate calibration
            calibration_score = detect_geometric_patterns(frame)
            calibration_indicators.append(calibration_score)
    
    cap.release()
    
    detection_confidence = np.mean(calibration_indicators) if calibration_indicators else 0.0
    
    return {
        'detection_confidence': detection_confidence,
        'frames_analyzed': len(calibration_indicators),
        'calibration_patterns_detected': np.sum(np.array(calibration_indicators) > 0.5)
    }

def detect_geometric_patterns(frame):
    """Detect geometric patterns that might indicate calibration markers."""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Simple geometric pattern detection (circles, rectangles)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 20,
        param1=50, param2=30, minRadius=5, maxRadius=50
    )
    
    # Detect rectangles/squares
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = 0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            rectangles += 1
    
    # Calculate calibration pattern score
    circle_score = 0.5 if circles is not None and len(circles[0]) > 0 else 0.0
    rectangle_score = min(0.5, rectangles / 10.0)
    
    return circle_score + rectangle_score

def generate_detection_recommendations(confidence, confidence_factors):
    """Generate recommendations based on detection results."""
    recommendations = []
    
    if confidence < 0.5:
        recommendations.append("Low overall confidence - verify file is Crimaldi format")
    
    if confidence_factors.get('container_compatibility', 0) < 0.5:
        recommendations.append("Container format may not be standard Crimaldi - check file extension and codec")
    
    if confidence_factors.get('metadata_signature', 0) < 0.5:
        recommendations.append("Metadata signature doesn't match expected Crimaldi patterns")
    
    if confidence_factors.get('calibration_markers', 0) < 0.3:
        recommendations.append("No clear calibration markers detected - manual calibration may be required")
    
    if not recommendations:
        recommendations.append("Format detection successful - proceed with processing")
    
    return recommendations

def generate_basic_recommendations(confidence):
    """Generate basic recommendations for standard detection."""
    if confidence > 0.9:
        return ["High confidence Crimaldi format detection - proceed with processing"]
    elif confidence > 0.7:
        return ["Moderate confidence detection - consider additional validation"]
    else:
        return ["Low confidence detection - manual format verification recommended"]

# Example usage
if __name__ == "__main__":
    sample_file = 'data/crimaldi_samples/crimaldi_sample.avi'
    
    if Path(sample_file).exists():
        # Perform comprehensive format detection
        detection_result = comprehensive_format_detection(sample_file, enable_deep_analysis=True)
        
        print(f"\n{'='*60}")
        print("COMPREHENSIVE FORMAT DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"File: {sample_file}")
        print(f"Detected format: {detection_result['format']}")
        print(f"Overall confidence: {detection_result['confidence']:.4f}")
        
        if 'confidence_breakdown' in detection_result:
            print(f"\nConfidence breakdown:")
            for factor, score in detection_result['confidence_breakdown'].items():
                print(f"  {factor}: {score:.4f}")
        
        print(f"\nRecommendations:")
        for rec in detection_result['analysis_recommendations']:
            print(f"  - {rec}")
        
        # Save detailed detection results
        import json
        results_file = 'results/format_detection_results.json'
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(detection_result, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
    else:
        print(f"Sample file not found: {sample_file}")
```

### Normalization Pipeline

Advanced normalization pipeline with comprehensive quality control and validation:

```python
from src.backend.core.data_normalization import normalize_plume_data, NormalizationQualityAssessment
from src.backend.core.calibration_manager import CalibrationManager
from src.backend.utils.progress_display import create_progress_bar
import json
import numpy as np

def execute_normalization_pipeline(input_file, output_file, config_file, quality_threshold=0.95):
    """
    Execute comprehensive normalization pipeline with quality control and validation.
    
    Args:
        input_file (str): Path to input Crimaldi video file
        output_file (str): Path for normalized output file
        config_file (str): Path to normalization configuration
        quality_threshold (float): Minimum quality threshold for acceptance
        
    Returns:
        dict: Comprehensive normalization results with quality metrics
    """
    print(f"Starting normalization pipeline for: {input_file}")
    
    # Load normalization configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Initialize calibration manager
    calibration_manager = CalibrationManager()
    
    # Step 1: Pre-normalization validation
    print("Step 1: Pre-normalization validation...")
    pre_validation = perform_pre_normalization_validation(input_file, config)
    
    if not pre_validation['valid']:
        raise ValueError(f"Pre-normalization validation failed: {pre_validation['errors']}")
    
    print(f"  Pre-validation passed: {pre_validation['score']:.3f}")
    
    # Step 2: Calibration parameter setup and validation
    print("Step 2: Calibration parameter setup...")
    calibration_result = calibration_manager.setup_calibration_parameters(
        input_file, 
        config.get('format_specific_settings', {}).get('crimaldi_format', {}),
        validate_accuracy=True
    )
    
    print(f"  Calibration setup completed: accuracy {calibration_result['accuracy_score']:.3f}")
    
    # Step 3: Execute normalization with progress monitoring
    print("Step 3: Executing normalization pipeline...")
    
    # Create progress bar for normalization
    progress_bar = create_progress_bar(
        total_steps=100,
        description="Normalizing plume data",
        show_percentage=True,
        show_timing=True
    )
    
    # Execute normalization with callback for progress updates
    normalization_result = normalize_plume_data(
        input_file_path=input_file,
        output_path=output_file,
        config_path=config_file,
        validation_mode=True,
        preserve_metadata=True,
        progress_callback=progress_bar.update,
        enable_quality_assessment=True
    )
    
    progress_bar.close()
    
    # Step 4: Post-normalization quality assessment
    print("Step 4: Post-normalization quality assessment...")
    quality_assessor = NormalizationQualityAssessment(
        original_file=input_file,
        normalized_file=output_file,
        calibration_parameters=calibration_result['parameters']
    )
    
    quality_metrics = quality_assessor.perform_comprehensive_assessment()
    
    print(f"  Quality assessment completed:")
    print(f"    Overall quality: {quality_metrics['overall_quality']:.4f}")
    print(f"    Spatial fidelity: {quality_metrics['spatial_fidelity']:.4f}")
    print(f"    Temporal consistency: {quality_metrics['temporal_consistency']:.4f}")
    print(f"    Intensity calibration: {quality_metrics['intensity_calibration']:.4f}")
    
    # Step 5: Quality threshold validation
    print("Step 5: Quality threshold validation...")
    quality_validation = validate_quality_thresholds(quality_metrics, quality_threshold)
    
    if not quality_validation['passes_threshold']:
        print(f"  WARNING: Quality below threshold ({quality_threshold})")
        print(f"  Failed metrics: {quality_validation['failed_metrics']}")
        
        if config.get('validation_settings', {}).get('automatic_quality_adjustment', False):
            print("  Attempting automatic quality adjustment...")
            adjusted_result = attempt_quality_adjustment(input_file, output_file, config, quality_metrics)
            if adjusted_result['success']:
                quality_metrics = adjusted_result['improved_quality']
                print(f"  Quality adjustment successful: {quality_metrics['overall_quality']:.4f}")
            else:
                print(f"  Quality adjustment failed: {adjusted_result['error']}")
    else:
        print(f"  Quality validation PASSED: {quality_metrics['overall_quality']:.4f}")
    
    # Step 6: Generate normalization report
    print("Step 6: Generating normalization report...")
    normalization_report = generate_normalization_report(
        input_file, output_file, config_file,
        pre_validation, calibration_result, normalization_result, quality_metrics
    )
    
    # Save normalization report
    report_file = output_file.replace('.npy', '_normalization_report.json')
    with open(report_file, 'w') as f:
        json.dump(normalization_report, f, indent=2, default=str)
    
    print(f"  Normalization report saved: {report_file}")
    
    # Step 7: Validation against reference data (if available)
    reference_validation = None
    if config.get('validation_settings', {}).get('enable_reference_comparison', False):
        print("Step 7: Reference data validation...")
        reference_file = config.get('validation_settings', {}).get('reference_file')
        if reference_file and Path(reference_file).exists():
            reference_validation = validate_against_reference(output_file, reference_file)
            print(f"  Reference correlation: {reference_validation['correlation']:.4f}")
        else:
            print("  Reference file not available - skipping reference validation")
    
    # Compile comprehensive results
    pipeline_results = {
        'input_file': str(input_file),
        'output_file': str(output_file),
        'config_file': str(config_file),
        'pre_validation': pre_validation,
        'calibration_result': calibration_result,
        'normalization_result': normalization_result.to_dict(),
        'quality_metrics': quality_metrics,
        'quality_validation': quality_validation,
        'reference_validation': reference_validation,
        'normalization_report_file': report_file,
        'pipeline_success': quality_validation['passes_threshold'],
        'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(f"Normalization pipeline completed: {'SUCCESS' if pipeline_results['pipeline_success'] else 'QUALITY_ISSUES'}")
    
    return pipeline_results

def perform_pre_normalization_validation(input_file, config):
    """Perform comprehensive pre-normalization validation."""
    validation_checks = {
        'file_accessibility': validate_file_accessibility(input_file),
        'format_compatibility': validate_format_compatibility(input_file),
        'resolution_requirements': validate_resolution_requirements(input_file, config),
        'temporal_requirements': validate_temporal_requirements(input_file, config),
        'calibration_availability': validate_calibration_availability(input_file),
        'disk_space_availability': validate_disk_space_availability(input_file, config)
    }
    
    # Calculate overall validation score
    validation_scores = [check['score'] for check in validation_checks.values()]
    overall_score = np.mean(validation_scores)
    
    # Identify failed checks
    failed_checks = [name for name, check in validation_checks.items() if not check['valid']]
    errors = [check['error'] for check in validation_checks.values() if 'error' in check]
    
    return {
        'valid': len(failed_checks) == 0,
        'score': overall_score,
        'checks': validation_checks,
        'failed_checks': failed_checks,
        'errors': errors
    }

def validate_file_accessibility(file_path):
    """Validate file accessibility and basic properties."""
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'valid': False, 'score': 0.0, 'error': 'File does not exist'}
        
        if not file_path.is_file():
            return {'valid': False, 'score': 0.0, 'error': 'Path is not a file'}
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024**2)
        if file_size_mb < 0.1:
            return {'valid': False, 'score': 0.3, 'error': 'File too small (likely corrupted)'}
        
        # Check read permissions
        if not os.access(file_path, os.R_OK):
            return {'valid': False, 'score': 0.0, 'error': 'No read permission'}
        
        return {'valid': True, 'score': 1.0, 'file_size_mb': file_size_mb}
        
    except Exception as e:
        return {'valid': False, 'score': 0.0, 'error': f'File validation error: {str(e)}'}

def validate_format_compatibility(file_path):
    """Validate format compatibility for normalization."""
    try:
        # Use previously defined format detection
        format_result = detect_crimaldi_format(file_path, deep_inspection=False)
        
        if format_result['confidence'] >= 0.8:
            return {'valid': True, 'score': format_result['confidence'], 'format': format_result['format']}
        else:
            return {
                'valid': False, 
                'score': format_result['confidence'], 
                'error': f'Low format confidence: {format_result["confidence"]:.3f}'
            }
            
    except Exception as e:
        return {'valid': False, 'score': 0.0, 'error': f'Format validation error: {str(e)}'}

def validate_resolution_requirements(file_path, config):
    """Validate video resolution meets requirements."""
    try:
        cap = cv2.VideoCapture(str(file_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Check against configuration requirements
        quality_thresholds = config.get('format_specific_settings', {}) \
                                   .get('crimaldi_format', {}) \
                                   .get('quality_thresholds', {})
        
        min_resolution = quality_thresholds.get('minimum_resolution', [320, 240])
        max_resolution = quality_thresholds.get('maximum_resolution', [1920, 1080])
        
        if width < min_resolution[0] or height < min_resolution[1]:
            return {
                'valid': False, 
                'score': 0.3, 
                'error': f'Resolution too low: {width}x{height} < {min_resolution[0]}x{min_resolution[1]}'
            }
        
        if width > max_resolution[0] or height > max_resolution[1]:
            return {
                'valid': False, 
                'score': 0.7, 
                'error': f'Resolution too high: {width}x{height} > {max_resolution[0]}x{max_resolution[1]}'
            }
        
        return {'valid': True, 'score': 1.0, 'resolution': [width, height]}
        
    except Exception as e:
        return {'valid': False, 'score': 0.0, 'error': f'Resolution validation error: {str(e)}'}

def validate_temporal_requirements(file_path, config):
    """Validate temporal properties meet requirements."""
    try:
        cap = cv2.VideoCapture(str(file_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Check against configuration requirements
        quality_thresholds = config.get('format_specific_settings', {}) \
                                   .get('crimaldi_format', {}) \
                                   .get('quality_thresholds', {})
        
        min_fps = quality_thresholds.get('minimum_frame_rate', 15.0)
        max_fps = quality_thresholds.get('maximum_frame_rate', 120.0)
        
        if fps < min_fps:
            return {
                'valid': False, 
                'score': 0.3, 
                'error': f'Frame rate too low: {fps:.2f} < {min_fps}'
            }
        
        if fps > max_fps:
            return {
                'valid': False, 
                'score': 0.7, 
                'error': f'Frame rate too high: {fps:.2f} > {max_fps}'
            }
        
        if duration < 5.0:  # Minimum 5 seconds
            return {
                'valid': False, 
                'score': 0.4, 
                'error': f'Video too short: {duration:.1f} seconds'
            }
        
        return {
            'valid': True, 
            'score': 1.0, 
            'fps': fps, 
            'duration': duration, 
            'frame_count': frame_count
        }
        
    except Exception as e:
        return {'valid': False, 'score': 0.0, 'error': f'Temporal validation error: {str(e)}'}

def validate_calibration_availability(file_path):
    """Validate availability of calibration parameters."""
    try:
        handler = CrimaldiFormatHandler(file_path)
        calibration = handler.get_calibration_parameters(validate_accuracy=False)
        
        required_params = ['pixel_to_meter_ratio', 'arena_width_meters', 'arena_height_meters']
        missing_params = [param for param in required_params if param not in calibration or calibration[param] is None]
        
        if missing_params:
            return {
                'valid': False, 
                'score': 0.5, 
                'error': f'Missing calibration parameters: {missing_params}'
            }
        
        return {'valid': True, 'score': 1.0, 'calibration_available': True}
        
    except Exception as e:
        return {'valid': False, 'score': 0.0, 'error': f'Calibration validation error: {str(e)}'}

def validate_disk_space_availability(input_file, config):
    """Validate sufficient disk space for processing."""
    try:
        input_size = Path(input_file).stat().st_size
        
        # Estimate output size (typically 2-3x input size for normalized data)
        estimated_output_size = input_size * 3
        
        # Check available disk space
        disk_usage = psutil.disk_usage(Path(input_file).parent)
        available_space = disk_usage.free
        
        if available_space < estimated_output_size * 1.5:  # 50% safety margin
            return {
                'valid': False, 
                'score': 0.2, 
                'error': f'Insufficient disk space: {available_space / (1024**3):.1f} GB available, {estimated_output_size * 1.5 / (1024**3):.1f} GB required'
            }
        
        return {
            'valid': True, 
            'score': 1.0, 
            'available_space_gb': available_space / (1024**3),
            'estimated_requirement_gb': estimated_output_size / (1024**3)
        }
        
    except Exception as e:
        return {'valid': False, 'score': 0.0, 'error': f'Disk space validation error: {str(e)}'}

def validate_quality_thresholds(quality_metrics, quality_threshold):
    """Validate quality metrics against thresholds."""
    quality_checks = {
        'overall_quality': quality_metrics['overall_quality'] >= quality_threshold,
        'spatial_fidelity': quality_metrics['spatial_fidelity'] >= quality_threshold * 0.95,
        'temporal_consistency': quality_metrics['temporal_consistency'] >= quality_threshold * 0.98,
        'intensity_calibration': quality_metrics['intensity_calibration'] >= quality_threshold * 0.96
    }
    
    failed_metrics = [metric for metric, passed in quality_checks.items() if not passed]
    passes_threshold = len(failed_metrics) == 0
    
    return {
        'passes_threshold': passes_threshold,
        'quality_checks': quality_checks,
        'failed_metrics': failed_metrics,
        'quality_score': quality_metrics['overall_quality']
    }

def attempt_quality_adjustment(input_file, output_file, config, quality_metrics):
    """Attempt automatic quality adjustment if enabled."""
    try:
        # Identify quality issues and adjust parameters
        adjusted_config = config.copy()
        
        if quality_metrics['spatial_fidelity'] < 0.95:
            # Improve spatial processing
            adjusted_config['arena_normalization']['scaling_method'] = 'lanczos'
            adjusted_config['intensity_calibration']['noise_reduction']['enable_denoising'] = True
        
        if quality_metrics['temporal_consistency'] < 0.98:
            # Improve temporal processing
            adjusted_config['temporal_normalization']['enable_frame_alignment'] = True
            adjusted_config['temporal_normalization']['temporal_filtering']['enable_temporal_smoothing'] = True
        
        # Re-run normalization with adjusted parameters
        adjusted_result = normalize_plume_data(
            input_file_path=input_file,
            output_path=output_file.replace('.npy', '_adjusted.npy'),
            config=adjusted_config,
            validation_mode=True
        )
        
        # Re-assess quality
        quality_assessor = NormalizationQualityAssessment(
            original_file=input_file,
            normalized_file=output_file.replace('.npy', '_adjusted.npy'),
            calibration_parameters={}  # Would need calibration parameters
        )
        
        improved_quality = quality_assessor.perform_comprehensive_assessment()
        
        return {
            'success': True,
            'improved_quality': improved_quality,
            'adjustment_applied': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'adjustment_applied': False
        }

def generate_normalization_report(input_file, output_file, config_file, 
                                 pre_validation, calibration_result, 
                                 normalization_result, quality_metrics):
    """Generate comprehensive normalization report."""
    return {
        'normalization_summary': {
            'input_file': str(input_file),
            'output_file': str(output_file),
            'config_file': str(config_file),
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processing_success': normalization_result.success,
            'overall_quality_score': quality_metrics['overall_quality']
        },
        'pre_validation_results': pre_validation,
        'calibration_results': calibration_result,
        'normalization_performance': {
            'processing_time_seconds': normalization_result.processing_time_seconds,
            'memory_usage_peak_mb': getattr(normalization_result, 'peak_memory_mb', 0),
            'frames_processed': getattr(normalization_result, 'frames_processed', 0),
            'processing_rate_fps': getattr(normalization_result, 'processing_rate_fps', 0)
        },
        'quality_assessment': quality_metrics,
        'recommendations': generate_normalization_recommendations(quality_metrics),
        'technical_details': {
            'input_file_size_mb': Path(input_file).stat().st_size / (1024**2),
            'output_file_size_mb': Path(output_file).stat().st_size / (1024**2) if Path(output_file).exists() else 0,
            'compression_ratio': calculate_compression_ratio(input_file, output_file),
            'format_conversion': 'AVI to NumPy array',
            'precision_level': 'float64'
        }
    }

def generate_normalization_recommendations(quality_metrics):
    """Generate recommendations based on quality metrics."""
    recommendations = []
    
    if quality_metrics['overall_quality'] >= 0.98:
        recommendations.append("Excellent normalization quality - ready for simulation")
    elif quality_metrics['overall_quality'] >= 0.95:
        recommendations.append("Good normalization quality - suitable for most analyses")
    else:
        recommendations.append("Normalization quality below optimal - consider parameter adjustment")
    
    if quality_metrics['spatial_fidelity'] < 0.95:
        recommendations.append("Consider higher-quality spatial interpolation method")
    
    if quality_metrics['temporal_consistency'] < 0.98:
        recommendations.append("Enable temporal alignment and filtering for better consistency")
    
    if quality_metrics['intensity_calibration'] < 0.96:
        recommendations.append("Review calibration parameters and consider manual adjustment")
    
    return recommendations

def calculate_compression_ratio(input_file, output_file):
    """Calculate compression ratio between input and output files."""
    try:
        input_size = Path(input_file).stat().st_size
        output_size = Path(output_file).stat().st_size if Path(output_file).exists() else 0
        
        if output_size > 0:
            return input_size / output_size
        else:
            return 0.0
    except:
        return 0.0

def validate_against_reference(normalized_file, reference_file):
    """Validate normalized data against reference implementation."""
    try:
        # Load normalized and reference data
        normalized_data = np.load(normalized_file)
        reference_data = np.load(reference_file)
        
        # Calculate correlation
        correlation = np.corrcoef(normalized_data.flatten(), reference_data.flatten())[0, 1]
        
        # Calculate additional metrics
        mse = np.mean((normalized_data - reference_data) ** 2)
        mae = np.mean(np.abs(normalized_data - reference_data))
        
        return {
            'correlation': correlation,
            'mse': mse,
            'mae': mae,
            'passes_correlation_threshold': correlation >= 0.95
        }
        
    except Exception as e:
        return {
            'correlation': 0.0,
            'error': str(e),
            'passes_correlation_threshold': False
        }

# Example usage
if __name__ == "__main__":
    # Define processing parameters
    input_file = 'data/crimaldi_samples/crimaldi_sample.avi'
    output_file = 'data/normalized/crimaldi_sample_normalized.npy'
    config_file = 'config/crimaldi_normalization_config.json'
    
    try:
        # Execute normalization pipeline
        pipeline_results = execute_normalization_pipeline(
            input_file=input_file,
            output_file=output_file,
            config_file=config_file,
            quality_threshold=0.95
        )
        
        print(f"\n{'='*60}")
        print("NORMALIZATION PIPELINE RESULTS")
        print(f"{'='*60}")
        print(f"Pipeline success: {'YES' if pipeline_results['pipeline_success'] else 'NO'}")
        print(f"Overall quality: {pipeline_results['quality_metrics']['overall_quality']:.4f}")
        print(f"Processing time: {pipeline_results['normalization_result']['processing_time_seconds']:.2f} seconds")
        print(f"Report file: {pipeline_results['normalization_report_file']}")
        
        if pipeline_results['reference_validation']:
            print(f"Reference correlation: {pipeline_results['reference_validation']['correlation']:.4f}")
        
    except Exception as e:
        print(f"Error in normalization pipeline: {e}")
        import traceback
        traceback.print_exc()
```

### Quality Validation

Comprehensive quality validation framework ensuring scientific computing standards: