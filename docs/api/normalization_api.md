# Data Normalization API Reference

**Version:** 1.0.0  
**Last Updated:** 2024-01-15  
**Supported Formats:** crimaldi, custom, avi, mp4, mov

## Overview

The Data Normalization API provides comprehensive tools for normalizing plume recording data across different experimental setups and formats. This module ensures >95% correlation accuracy with reference implementations while supporting cross-format compatibility for Crimaldi and custom datasets. The API is designed for scientific computing workflows requiring reproducible results and efficient batch processing of 4000+ simulations.

### Key Features

- **Cross-format video processing** (Crimaldi, custom AVI, MP4, MOV)
- **Automated scale calibration** and spatial normalization  
- **Temporal normalization** with motion preservation
- **Intensity calibration** and unit conversion
- **Comprehensive validation** and quality assurance
- **Batch processing** with parallel execution
- **Scientific computing precision** (>95% correlation)
- **Performance optimization** (<7.2s average processing time)

### Supported Formats

- **Crimaldi Dataset:** Standard research dataset with embedded calibration parameters
- **Custom AVI:** User-provided AVI files with automatic parameter detection
- **MP4/MOV:** Modern video formats with metadata extraction
- **Cross-format compatibility:** Seamless conversion between formats

## Core Pipeline API

The `DataNormalizationPipeline` class provides the main interface for comprehensive data normalization operations.

### DataNormalizationPipeline

**Class:** `DataNormalizationPipeline`

Comprehensive data normalization pipeline orchestrator providing unified interface for all normalization operations including video processing, calibration, validation, and format conversion with scientific computing precision.

#### Constructor

```python
DataNormalizationPipeline(
    pipeline_config: Dict[str, Any],
    enable_caching: bool = True,
    enable_validation: bool = True
)
```

**Parameters:**
- `pipeline_config` (Dict[str, Any]): Pipeline configuration including processing options, quality thresholds, and component settings
- `enable_caching` (bool): Enable multi-level caching for performance optimization (default: True)
- `enable_validation` (bool): Enable comprehensive validation and quality assurance (default: True)

#### Methods

##### normalize_single_file

```python
normalize_single_file(
    input_path: str,
    output_path: str,
    processing_options: Dict[str, Any] = None
) -> NormalizationResult
```

Normalize single plume recording file with comprehensive processing pipeline including format detection, calibration, and validation.

**Parameters:**
- `input_path` (str): Path to input video file (supports Crimaldi, AVI, MP4, MOV formats)
- `output_path` (str): Path for normalized output file
- `processing_options` (Dict[str, Any], optional): Optional processing configuration overrides

**Returns:**
- `NormalizationResult`: Comprehensive normalization result with processed data, quality metrics, and validation status

**Example:**
```python
pipeline = DataNormalizationPipeline(config)
result = pipeline.normalize_single_file(
    'input/crimaldi_sample.avi',
    'output/normalized_sample.avi'
)
print(f'Quality score: {result.calculate_overall_quality_score()}')
```

##### normalize_batch_files

```python
normalize_batch_files(
    input_paths: List[str],
    output_directory: str,
    batch_options: Dict[str, Any] = None
) -> BatchNormalizationResult
```

Normalize multiple plume recording files with parallel processing and comprehensive error handling for large-scale operations.

**Parameters:**
- `input_paths` (List[str]): List of input video file paths for batch processing
- `output_directory` (str): Directory for normalized output files
- `batch_options` (Dict[str, Any], optional): Batch processing configuration including parallel processing settings

**Returns:**
- `BatchNormalizationResult`: Batch normalization result with aggregate statistics and error analysis

**Example:**
```python
batch_result = pipeline.normalize_batch_files(
    ['video1.avi', 'video2.avi', 'video3.avi'],
    'output_directory/',
    {'enable_parallel': True, 'max_workers': 4}
)
print(f'Success rate: {batch_result.success_rate:.2%}')
```

##### validate_pipeline_configuration

```python
validate_pipeline_configuration(
    strict_validation: bool = True
) -> ValidationResult
```

Validate pipeline configuration and component compatibility with comprehensive analysis and recommendations.

**Parameters:**
- `strict_validation` (bool): Whether to apply strict validation criteria

**Returns:**
- `ValidationResult`: Pipeline configuration validation result with analysis and recommendations

## Video Processing API

The `VideoProcessor` class handles multi-format video reading, processing, and optimization.

### VideoProcessor

**Class:** `VideoProcessor`

Comprehensive video processing orchestrator with multi-format support, intelligent caching, and performance optimization for scientific plume recording analysis.

#### Methods

##### process_video

```python
process_video(
    video_path: str,
    output_path: str,
    processing_options: Dict[str, Any] = None
) -> VideoProcessingResult
```

Process single video file with comprehensive normalization pipeline including format detection, calibration, and quality validation.

**Parameters:**
- `video_path` (str): Path to input video file
- `output_path` (str): Path for processed output video
- `processing_options` (Dict[str, Any], optional): Processing configuration options

**Returns:**
- `VideoProcessingResult`: Video processing result with quality metrics and performance data

##### process_video_batch

```python
process_video_batch(
    video_paths: List[str],
    output_directory: str,
    batch_options: Dict[str, Any] = None
) -> List[VideoProcessingResult]
```

Process multiple video files with batch optimization and parallel processing support.

**Parameters:**
- `video_paths` (List[str]): List of video file paths to process
- `output_directory` (str): Directory for processed output videos
- `batch_options` (Dict[str, Any], optional): Batch processing configuration

**Returns:**
- `List[VideoProcessingResult]`: List of processing results for each video

##### validate_processing_quality

```python
validate_processing_quality(
    original_path: str,
    processed_path: str,
    quality_thresholds: Dict[str, float] = None
) -> ValidationResult
```

Validate video processing quality against scientific computing standards.

**Parameters:**
- `original_path` (str): Path to original video file
- `processed_path` (str): Path to processed video file
- `quality_thresholds` (Dict[str, float], optional): Quality validation thresholds

**Returns:**
- `ValidationResult`: Quality validation result with correlation analysis

## Scale Calibration API

The `ScaleCalibration` class provides spatial normalization and coordinate transformation capabilities.

### ScaleCalibration

**Class:** `ScaleCalibration`

Comprehensive scale calibration container providing spatial normalization, coordinate transformation, and calibration parameter management for plume recording data.

#### Methods

##### extract_calibration_parameters

```python
extract_calibration_parameters(
    force_reextraction: bool = False,
    extraction_hints: Dict[str, Any] = None
) -> Dict[str, float]
```

Extract comprehensive calibration parameters from video including arena detection, ratio calculation, and validation.

**Parameters:**
- `force_reextraction` (bool): Force re-extraction even if parameters already exist
- `extraction_hints` (Dict[str, Any], optional): Hints to improve parameter extraction accuracy

**Returns:**
- `Dict[str, float]`: Extracted calibration parameters with confidence levels

##### apply_to_coordinates

```python
apply_to_coordinates(
    coordinates: np.ndarray,
    source_system: str,
    target_system: str,
    validate_transformation: bool = True
) -> np.ndarray
```

Apply scale calibration to coordinate data with transformation validation and performance optimization.

**Parameters:**
- `coordinates` (np.ndarray): Input coordinates to transform
- `source_system` (str): Source coordinate system ('pixel', 'meter', 'normalized')
- `target_system` (str): Target coordinate system ('pixel', 'meter', 'normalized')
- `validate_transformation` (bool): Enable transformation validation

**Returns:**
- `np.ndarray`: Transformed coordinates with calibration applied

##### calculate_scaling_factors

```python
calculate_scaling_factors(
    target_coordinate_system: str = 'meter',
    validate_factors: bool = True
) -> Dict[str, float]
```

Calculate comprehensive scaling factors for spatial normalization including pixel-to-meter ratios and validation metrics.

**Parameters:**
- `target_coordinate_system` (str): Target coordinate system for scaling calculations
- `validate_factors` (bool): Whether to validate calculated scaling factors

**Returns:**
- `Dict[str, float]`: Scaling factors with validation metrics and confidence assessment

##### validate_calibration

```python
validate_calibration(
    validation_thresholds: Dict[str, float] = None,
    strict_validation: bool = False
) -> ValidationResult
```

Validate calibration parameters against scientific requirements and accuracy thresholds.

**Parameters:**
- `validation_thresholds` (Dict[str, float], optional): Custom validation thresholds
- `strict_validation` (bool): Enable strict validation criteria for scientific computing

**Returns:**
- `ValidationResult`: Calibration validation result with error analysis and recommendations

### ScaleCalibrationManager

**Class:** `ScaleCalibrationManager`

Manager class for handling multiple scale calibrations with cross-format compatibility, batch operations, and performance optimization.

#### Methods

##### create_calibration

```python
create_calibration(
    video_path: str,
    calibration_config: Dict[str, Any] = None,
    validate_creation: bool = True
) -> ScaleCalibration
```

Create new scale calibration for video file with format detection, parameter extraction, and validation.

**Parameters:**
- `video_path` (str): Path to video file for calibration creation
- `calibration_config` (Dict[str, Any], optional): Configuration for calibration creation
- `validate_creation` (bool): Whether to validate calibration creation

**Returns:**
- `ScaleCalibration`: Created scale calibration with extracted parameters and validation status

##### batch_calibrate

```python
batch_calibrate(
    video_paths: List[str],
    batch_config: Dict[str, Any] = None,
    enable_parallel_processing: bool = None
) -> Dict[str, ScaleCalibration]
```

Perform batch calibration for multiple video files with parallel processing and error handling.

**Parameters:**
- `video_paths` (List[str]): List of video file paths for batch calibration
- `batch_config` (Dict[str, Any], optional): Configuration for batch processing
- `enable_parallel_processing` (bool, optional): Whether to enable parallel processing

**Returns:**
- `Dict[str, ScaleCalibration]`: Batch calibration results with individual calibration status

##### validate_cross_format_compatibility

```python
validate_cross_format_compatibility(
    format_types: List[str],
    tolerance_thresholds: Dict[str, float] = None,
    detailed_analysis: bool = True
) -> ValidationResult
```

Validate calibration compatibility across different video formats with conversion accuracy assessment.

**Parameters:**
- `format_types` (List[str]): List of format types to validate compatibility
- `tolerance_thresholds` (Dict[str, float], optional): Tolerance thresholds for cross-format consistency
- `detailed_analysis` (bool): Whether to perform detailed compatibility analysis

**Returns:**
- `ValidationResult`: Cross-format compatibility validation result with conversion accuracy assessment

## Temporal Normalization API

The `TemporalNormalizer` class handles frame rate normalization and temporal processing.

### TemporalNormalizer

**Class:** `TemporalNormalizer`

Comprehensive temporal normalization class providing advanced temporal processing capabilities including frame rate normalization, temporal interpolation, and motion preservation.

#### Constructor

```python
TemporalNormalizer(
    normalization_config: Dict[str, Any] = None,
    enable_performance_monitoring: bool = True,
    enable_quality_validation: bool = True
)
```

**Parameters:**
- `normalization_config` (Dict[str, Any], optional): Configuration dictionary for temporal normalization
- `enable_performance_monitoring` (bool): Whether to enable performance monitoring
- `enable_quality_validation` (bool): Whether to enable quality validation

#### Methods

##### normalize_video_temporal

```python
normalize_video_temporal(
    video_input: Union[str, VideoReader],
    source_fps: float,
    processing_options: Dict[str, Any] = None
) -> TemporalNormalizationResult
```

Normalize video temporal characteristics including frame rate conversion, synchronization, and quality validation.

**Parameters:**
- `video_input` (Union[str, VideoReader]): Video file path or VideoReader instance
- `source_fps` (float): Source video frame rate
- `processing_options` (Dict[str, Any], optional): Temporal processing configuration options

**Returns:**
- `TemporalNormalizationResult`: Temporal normalization result with processed video data and quality metrics

##### validate_temporal_quality

```python
validate_temporal_quality(
    original_sequence: np.ndarray,
    processed_sequence: np.ndarray,
    validation_context: Dict[str, Any] = None
) -> ValidationResult
```

Validate temporal processing quality with correlation analysis and motion preservation assessment.

**Parameters:**
- `original_sequence` (np.ndarray): Original frame sequence before processing
- `processed_sequence` (np.ndarray): Processed frame sequence after temporal normalization
- `validation_context` (Dict[str, Any], optional): Additional context for validation

**Returns:**
- `ValidationResult`: Comprehensive temporal quality validation result with detailed analysis

##### synchronize_sequences

```python
synchronize_sequences(
    video_sequences: List[np.ndarray],
    sequence_fps: List[float],
    sync_options: Dict[str, Any] = None
) -> Tuple[List[np.ndarray], Dict[str, Any]]
```

Synchronize multiple video sequences with drift correction and alignment validation.

**Parameters:**
- `video_sequences` (List[np.ndarray]): List of video sequences to synchronize
- `sequence_fps` (List[float]): List of frame rates for each sequence
- `sync_options` (Dict[str, Any], optional): Options for synchronization processing

**Returns:**
- `Tuple[List[np.ndarray], Dict[str, Any]]`: Synchronized sequences with analysis and quality metrics

### TemporalNormalizationResult

**Class:** `TemporalNormalizationResult`

Comprehensive result container for temporal normalization operations providing processed video data, quality metrics, and validation results.

#### Methods

##### calculate_quality_score

```python
calculate_quality_score() -> float
```

Calculate overall quality score based on temporal correlation, motion preservation, and validation metrics.

**Returns:**
- `float`: Overall quality score between 0.0 and 1.0 representing temporal processing quality

##### to_dict

```python
to_dict(include_frame_data: bool = False) -> Dict[str, Any]
```

Convert result to dictionary format for serialization and reporting.

**Parameters:**
- `include_frame_data` (bool): Whether to include actual frame data (memory intensive)

**Returns:**
- `Dict[str, Any]`: Complete result as dictionary with optional frame data inclusion

## Intensity Calibration API

The `IntensityCalibration` class provides intensity normalization and unit conversion.

### IntensityCalibration

**Class:** `IntensityCalibration`

Comprehensive intensity calibration container providing intensity parameter extraction, conversion calculation, and cross-format compatibility with scientific computing precision.

#### Methods

##### extract_intensity_parameters

```python
extract_intensity_parameters(
    intensity_data: np.ndarray,
    extraction_options: Dict[str, Any] = None
) -> Dict[str, Any]
```

Extract comprehensive intensity parameters from video data including dynamic range, noise characteristics, and statistical properties.

**Parameters:**
- `intensity_data` (np.ndarray): Input intensity data array
- `extraction_options` (Dict[str, Any], optional): Parameter extraction configuration options

**Returns:**
- `Dict[str, Any]`: Extracted intensity parameters with statistical analysis

##### apply_to_intensity_data

```python
apply_to_intensity_data(
    intensity_data: np.ndarray,
    validate_results: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]
```

Apply comprehensive intensity calibration to video data with noise reduction, background subtraction, and quality validation.

**Parameters:**
- `intensity_data` (np.ndarray): Input intensity data to calibrate
- `validate_results` (bool): Enable calibration result validation

**Returns:**
- `Tuple[np.ndarray, Dict[str, Any]]`: Calibrated intensity data and calibration metadata

### IntensityCalibrationManager

**Class:** `IntensityCalibrationManager`

Manager class for handling multiple intensity calibrations with batch processing and cross-format compatibility.

#### Methods

##### create_calibration

```python
create_calibration(
    video_path: str,
    calibration_config: Dict[str, Any] = None,
    validate_creation: bool = True
) -> IntensityCalibration
```

Create new intensity calibration for video file with parameter extraction and validation.

**Parameters:**
- `video_path` (str): Path to video file for calibration creation
- `calibration_config` (Dict[str, Any], optional): Configuration for calibration creation
- `validate_creation` (bool): Whether to validate calibration creation

**Returns:**
- `IntensityCalibration`: Created intensity calibration with extracted parameters

## Validation and Quality Assurance

Comprehensive validation framework ensuring scientific computing standards and reproducible results.

### Quality Metrics

- **Correlation Accuracy:** >95% correlation with reference implementations
- **Spatial Accuracy:** Pixel-to-meter conversion precision
- **Temporal Accuracy:** Frame rate conversion quality  
- **Intensity Accuracy:** Unit conversion and calibration precision
- **Cross-format Consistency:** Compatibility across different formats

### Validation Functions

#### validate_normalization_pipeline

```python
validate_normalization_pipeline(
    pipeline_config: Dict[str, Any],
    strict_validation: bool = True,
    include_performance_validation: bool = True
) -> ValidationResult
```

Validate complete normalization pipeline configuration and components ensuring scientific computing standards.

**Parameters:**
- `pipeline_config` (Dict[str, Any]): Pipeline configuration dictionary for validation
- `strict_validation` (bool): Whether to apply strict validation criteria
- `include_performance_validation` (bool): Whether to include performance validation

**Returns:**
- `ValidationResult`: Pipeline validation result with configuration analysis and recommendations

#### validate_cross_format_consistency

```python
validate_cross_format_consistency(
    supported_formats: List[str],
    compatibility_config: Dict[str, Any] = None,
    strict_validation: bool = True
) -> ValidationResult
```

Validate cross-format compatibility and conversion accuracy across different video formats.

**Parameters:**
- `supported_formats` (List[str]): List of supported formats to validate
- `compatibility_config` (Dict[str, Any], optional): Configuration for compatibility testing
- `strict_validation` (bool): Whether to apply strict validation criteria

**Returns:**
- `ValidationResult`: Cross-format compatibility validation result

## Error Handling and Recovery

Robust error handling with graceful degradation and recovery mechanisms.

### Error Types

- **ValidationError:** Configuration and parameter validation failures
- **ProcessingError:** Video processing and calibration errors
- **FormatError:** Unsupported or corrupted video formats
- **QualityError:** Quality threshold violations

### Recovery Strategies

- **Fail-fast validation:** Early detection of incompatible parameters
- **Graceful degradation:** Partial processing with error reporting
- **Automatic retry:** Transient error recovery with exponential backoff
- **Checkpoint resumption:** Resume interrupted batch operations

## Performance Optimization

Performance features for efficient large-scale processing.

### Caching System

- **Level 1:** In-memory caching for active data
- **Level 2:** Disk-based caching for processed videos
- **Level 3:** Result caching for completed normalizations
- **Cache management:** Automatic eviction and optimization

### Parallel Processing

- **Batch processing:** Parallel execution of independent operations
- **Resource management:** Optimal CPU and memory utilization
- **Progress monitoring:** Real-time batch progress tracking
- **Load balancing:** Dynamic resource allocation

## Usage Examples

### Basic Single File Normalization

```python
from src.backend.core.data_normalization import (
    create_normalization_pipeline,
    normalize_plume_data
)

# Create pipeline with default configuration
pipeline = create_normalization_pipeline({
    'target_format': 'normalized',
    'quality_threshold': 0.95,
    'enable_validation': True
})

# Normalize single file
result = normalize_plume_data(
    'input/crimaldi_sample.avi',
    'output/normalized_sample.avi',
    {'enable_caching': True}
)

print(f'Normalization successful: {result.normalization_successful}')
print(f'Quality score: {result.calculate_overall_quality_score():.3f}')
```

### Batch Processing with Parallel Execution

```python
from src.backend.core.data_normalization import batch_normalize_plume_data

# Batch normalize multiple files
input_files = [
    'data/crimaldi_001.avi',
    'data/crimaldi_002.avi', 
    'data/custom_001.avi',
    'data/custom_002.avi'
]

batch_result = batch_normalize_plume_data(
    input_files,
    'output_directory/',
    {
        'enable_parallel_processing': True,
        'max_workers': 4,
        'batch_size': 50,
        'quality_threshold': 0.95
    }
)

print(f'Processed {batch_result.total_files} files')
print(f'Success rate: {batch_result.success_rate:.2%}')
print(f'Average quality: {batch_result.aggregate_quality_metrics["mean_quality"]:.3f}')
```

### Cross-Format Compatibility Validation

```python
from src.backend.core.data_normalization import (
    DataNormalizationPipeline,
    validate_cross_format_consistency
)

# Validate cross-format compatibility
validation_result = validate_cross_format_consistency(
    ['crimaldi', 'custom', 'avi'],
    test_data_path='test/sample_data.avi'
)

if validation_result.validation_passed:
    print('Cross-format compatibility validated')
    print(f'Correlation: {validation_result.correlation_with_reference:.3f}')
else:
    print('Compatibility issues detected:')
    for error in validation_result.validation_errors:
        print(f'  - {error}')
```

### Custom Configuration and Advanced Options

```python
from src.backend.core.data_normalization import (
    DataNormalizationPipeline,
    VideoProcessingConfig,
    TemporalNormalizationConfig,
    IntensityCalibrationConfig
)

# Create custom configuration
video_config = VideoProcessingConfig(
    target_format='normalized',
    quality_thresholds={'correlation': 0.98, 'spatial_accuracy': 0.95},
    enable_validation=True
)

temporal_config = TemporalNormalizationConfig(
    target_fps=30.0,
    interpolation_method='cubic',
    temporal_smoothing=True
)

intensity_config = IntensityCalibrationConfig(
    calibration_method='robust_scaling',
    target_range=(0.0, 1.0),
    background_subtraction=True,
    noise_reduction_method='bilateral'
)

# Create pipeline with custom configuration
pipeline = DataNormalizationPipeline({
    'video_processing': video_config.to_dict(),
    'temporal_normalization': temporal_config.to_dict(),
    'intensity_calibration': intensity_config.to_dict(),
    'enable_caching': True,
    'enable_parallel_processing': True
})

# Process with custom configuration
result = pipeline.normalize_single_file(
    'input/complex_sample.avi',
    'output/normalized_complex.avi'
)
```

## Configuration Reference

Detailed configuration options for all normalization components.

### Pipeline Configuration

- `enable_validation` (bool): Enable comprehensive validation (default: True)
- `enable_caching` (bool): Enable multi-level caching (default: True)
- `enable_parallel_processing` (bool): Enable parallel batch processing (default: True)
- `quality_threshold` (float): Overall quality threshold (default: 0.95)
- `processing_timeout` (float): Processing timeout in seconds (default: 300.0)
- `batch_size` (int): Batch processing size (default: 50)
- `max_workers` (int): Maximum parallel workers (default: 8)

### Video Processing Configuration

- `target_format` (str): Target video format ('normalized', 'avi', 'mp4')
- `quality_thresholds` (Dict[str, float]): Quality validation thresholds
- `enable_format_conversion` (bool): Enable automatic format conversion
- `memory_optimization` (bool): Enable memory usage optimization
- `frame_buffer_size` (int): Frame buffer size for processing

### Scale Calibration Configuration

- `arena_detection_method` (str): Arena detection method ('contour', 'edge', 'template')
- `pixel_to_meter_ratio` (float): Manual pixel-to-meter ratio (optional)
- `coordinate_system` (str): Target coordinate system ('pixel', 'meter', 'normalized')
- `validation_threshold` (float): Calibration validation threshold

### Temporal Normalization Configuration

- `target_fps` (float): Target frame rate (default: 30.0)
- `interpolation_method` (str): Interpolation method ('linear', 'cubic', 'quintic')
- `frame_alignment` (str): Frame alignment ('start', 'center', 'end')
- `temporal_smoothing` (bool): Enable temporal smoothing
- `motion_preservation_threshold` (float): Motion preservation quality threshold

### Intensity Calibration Configuration

- `calibration_method` (str): Calibration method ('min_max', 'z_score', 'robust_scaling')
- `target_range` (Tuple[float, float]): Target intensity range
- `background_subtraction` (bool): Enable background subtraction
- `noise_reduction_method` (str): Noise reduction method ('gaussian', 'median', 'bilateral')
- `gamma_correction` (float): Gamma correction value
- `outlier_removal` (bool): Enable outlier detection and removal

---

## API Constants

### Supported Formats
```python
SUPPORTED_FORMATS = ['crimaldi', 'custom', 'avi', 'mp4', 'mov']
```

### Quality Thresholds
```python
QUALITY_THRESHOLD = 0.95
PROCESSING_TIME_TARGET = 7.2  # seconds
```

### Performance Limits
```python
MAX_PARALLEL_WORKERS = 8
DEFAULT_BATCH_SIZE = 50
```

This comprehensive API documentation provides detailed reference for all data normalization functionality with complete method signatures, parameters, return values, and usage examples for scientific plume recording data processing with cross-format compatibility and >95% correlation accuracy.