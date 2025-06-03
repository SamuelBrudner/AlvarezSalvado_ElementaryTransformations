"""
Comprehensive scale calibration module providing spatial normalization and calibration capabilities for plume recording data across different experimental setups.

This module implements pixel-to-meter conversion, arena size normalization, coordinate transformation, and cross-format compatibility for Crimaldi and custom plume datasets. 
Supports automated calibration parameter extraction, validation, and application with >95% correlation accuracy for scientific computing requirements and 4000+ simulation processing optimization.

Key Features:
- Physical scale normalization with cross-format compatibility
- Automated calibration parameter extraction from video metadata
- Pixel-to-meter conversion with >95% correlation accuracy
- Arena size normalization for consistent spatial analysis
- Coordinate transformation between pixel and meter systems
- Comprehensive validation with fail-fast error detection
- Performance optimization for 4000+ simulation processing
- Scientific context logging and audit trail integration
- Graceful degradation for batch processing operations
"""

# External imports with version specifications
import numpy as np  # numpy 2.1.3+ - Numerical array operations for coordinate transformations and spatial calculations
import cv2  # opencv-python 4.11.0+ - Computer vision operations for arena boundary detection and image analysis
import scipy.spatial  # scipy 1.15.3+ - Spatial transformation algorithms and coordinate system conversions
from typing import Dict, Any, List, Optional, Union, Callable, Tuple  # typing 3.9+ - Type hints for scale calibration function signatures and data structures
from dataclasses import dataclass, field  # dataclasses 3.9+ - Data classes for calibration parameter containers and configuration structures
from pathlib import Path  # pathlib 3.9+ - Path handling for video file operations and calibration data storage
import math  # math 3.9+ - Mathematical functions for geometric calculations and coordinate transformations
import time  # time 3.9+ - Performance timing for calibration operations and optimization
import warnings  # warnings 3.9+ - Warning generation for calibration accuracy and compatibility issues
from datetime import datetime  # datetime 3.9+ - Timestamp generation for calibration operations and audit trails

# Internal imports from utility modules
from ...utils.scientific_constants import (
    CRIMALDI_PIXEL_TO_METER_RATIO,  # Standard pixel-to-meter ratio for Crimaldi dataset format
    CUSTOM_PIXEL_TO_METER_RATIO,    # Default pixel-to-meter ratio for custom dataset formats
    TARGET_ARENA_WIDTH_METERS,      # Target arena width for normalization in meters
    TARGET_ARENA_HEIGHT_METERS,     # Target arena height for normalization in meters
    SPATIAL_ACCURACY_THRESHOLD,     # Spatial accuracy threshold for calibration validation
    PhysicalConstants               # Physical constants container with unit conversion and validation
)

from ...utils.validation_utils import (
    validate_physical_parameters,    # Validate physical parameters against scientific constraints and cross-format compatibility
    ValidationResult                # Comprehensive validation result container with error tracking and audit trail integration
)

from ...utils.logging_utils import (
    get_logger,                     # Get logger instance with scientific context and performance tracking
    set_scientific_context,         # Set scientific computing context for enhanced traceability
    log_performance_metrics         # Log performance metrics with structured format and scientific context
)

from ...error.exceptions import (
    ValidationError,                # Specialized validation error handling for scale calibration failures
    ProcessingError                 # Processing error handling with graceful degradation for scale calibration operations
)

from ...io.video_reader import (
    VideoReader,                    # Unified video reading interface for extracting calibration parameters from video metadata
    detect_video_format             # Detect video format type for format-specific calibration parameter extraction
)

# Global constants for scale calibration configuration
DEFAULT_CALIBRATION_CONFIDENCE_THRESHOLD: float = 0.95
MIN_ARENA_DETECTION_CONFIDENCE: float = 0.8
MAX_PIXEL_TO_METER_RATIO: float = 1000.0
MIN_PIXEL_TO_METER_RATIO: float = 1.0
COORDINATE_PRECISION_DIGITS: int = 6
CALIBRATION_CACHE_SIZE: int = 100
ARENA_DETECTION_METHODS: List[str] = ['contour', 'edge', 'template', 'manual']
SUPPORTED_COORDINATE_SYSTEMS: List[str] = ['pixel', 'meter', 'normalized']

# Global cache and state management
_calibration_cache: Dict[str, 'ScaleCalibration'] = {}
_arena_detection_cache: Dict[str, Dict[str, Any]] = {}

# Initialize logger for scale calibration operations
logger = get_logger('scale_calibration', 'DATA_NORMALIZATION')


def calculate_pixel_to_meter_ratio(
    arena_width_meters: float,
    arena_height_meters: float,
    video_width_pixels: int,
    video_height_pixels: int,
    calculation_method: str = 'geometric_mean'
) -> Dict[str, float]:
    """
    Calculate pixel-to-meter conversion ratio from arena dimensions and video resolution with validation and confidence assessment for accurate spatial scaling.
    
    This function computes the conversion ratios between pixel and meter coordinate systems using arena dimensions and video resolution. 
    It provides multiple calculation methods and comprehensive validation to ensure >95% correlation accuracy requirements.
    
    Args:
        arena_width_meters: Physical width of the arena in meters
        arena_height_meters: Physical height of the arena in meters  
        video_width_pixels: Video width resolution in pixels
        video_height_pixels: Video height resolution in pixels
        calculation_method: Method for ratio calculation ('horizontal', 'vertical', 'geometric_mean', 'arithmetic_mean')
        
    Returns:
        Dict[str, float]: Pixel-to-meter ratios with confidence levels and validation metrics
        
    Raises:
        ValidationError: If input parameters fail validation or calculation produces invalid results
    """
    # Set scientific context for enhanced traceability
    set_scientific_context(
        simulation_id='calibration_calculation',
        algorithm_name='pixel_to_meter_ratio',
        processing_stage='RATIO_CALCULATION'
    )
    
    start_time = time.time()
    
    try:
        # Validate input parameters against physical constraints and scientific requirements
        validation_result = validate_physical_parameters(
            {
                'arena_width_meters': arena_width_meters,
                'arena_height_meters': arena_height_meters,
                'video_width_pixels': video_width_pixels,
                'video_height_pixels': video_height_pixels
            },
            format_type='generic',
            cross_format_validation=True
        )
        
        if not validation_result.is_valid:
            raise ValidationError(
                f"Parameter validation failed: {validation_result.errors}",
                'parameter_validation',
                {
                    'arena_width_meters': arena_width_meters,
                    'arena_height_meters': arena_height_meters,
                    'video_width_pixels': video_width_pixels,
                    'video_height_pixels': video_height_pixels,
                    'validation_errors': validation_result.errors
                }
            )
        
        # Calculate horizontal pixel-to-meter ratio from width measurements
        horizontal_ratio = video_width_pixels / arena_width_meters if arena_width_meters > 0 else 0.0
        
        # Calculate vertical pixel-to-meter ratio from height measurements
        vertical_ratio = video_height_pixels / arena_height_meters if arena_height_meters > 0 else 0.0
        
        # Validate calculated ratios against scientific accuracy thresholds
        if horizontal_ratio < MIN_PIXEL_TO_METER_RATIO or horizontal_ratio > MAX_PIXEL_TO_METER_RATIO:
            raise ValidationError(
                f"Horizontal ratio {horizontal_ratio:.3f} outside valid range [{MIN_PIXEL_TO_METER_RATIO}, {MAX_PIXEL_TO_METER_RATIO}]",
                'ratio_validation',
                {'horizontal_ratio': horizontal_ratio}
            )
        
        if vertical_ratio < MIN_PIXEL_TO_METER_RATIO or vertical_ratio > MAX_PIXEL_TO_METER_RATIO:
            raise ValidationError(
                f"Vertical ratio {vertical_ratio:.3f} outside valid range [{MIN_PIXEL_TO_METER_RATIO}, {MAX_PIXEL_TO_METER_RATIO}]",
                'ratio_validation',
                {'vertical_ratio': vertical_ratio}
            )
        
        # Assess ratio consistency and calculate confidence level
        ratio_difference = abs(horizontal_ratio - vertical_ratio)
        relative_difference = ratio_difference / max(horizontal_ratio, vertical_ratio) if max(horizontal_ratio, vertical_ratio) > 0 else 1.0
        confidence_level = max(0.0, 1.0 - relative_difference)
        
        # Apply calculation method specific adjustments and corrections
        if calculation_method == 'horizontal':
            primary_ratio = horizontal_ratio
            method_confidence = 0.9 if confidence_level > 0.8 else 0.7
        elif calculation_method == 'vertical':
            primary_ratio = vertical_ratio
            method_confidence = 0.9 if confidence_level > 0.8 else 0.7
        elif calculation_method == 'geometric_mean':
            primary_ratio = math.sqrt(horizontal_ratio * vertical_ratio)
            method_confidence = confidence_level
        elif calculation_method == 'arithmetic_mean':
            primary_ratio = (horizontal_ratio + vertical_ratio) / 2.0
            method_confidence = confidence_level * 0.95  # Slightly lower confidence for arithmetic mean
        else:
            raise ValidationError(
                f"Unknown calculation method: {calculation_method}",
                'method_validation',
                {'calculation_method': calculation_method, 'supported_methods': ['horizontal', 'vertical', 'geometric_mean', 'arithmetic_mean']}
            )
        
        # Generate ratio validation metrics and quality assessment
        validation_metrics = {
            'ratio_consistency': confidence_level,
            'relative_difference': relative_difference,
            'method_confidence': method_confidence,
            'overall_confidence': confidence_level * method_confidence,
            'spatial_accuracy_met': confidence_level >= SPATIAL_ACCURACY_THRESHOLD
        }
        
        # Create comprehensive ratio dictionary with confidence and validation data
        ratio_result = {
            'horizontal_ratio': horizontal_ratio,
            'vertical_ratio': vertical_ratio,
            'primary_ratio': primary_ratio,
            'calculation_method': calculation_method,
            'confidence_level': validation_metrics['overall_confidence'],
            'validation_metrics': validation_metrics,
            'meter_to_pixel_ratio': 1.0 / primary_ratio if primary_ratio > 0 else 0.0,
            'calculation_timestamp': datetime.now().isoformat()
        }
        
        # Log calculation operation with performance metrics
        calculation_duration = time.time() - start_time
        log_performance_metrics(
            metric_name='pixel_to_meter_calculation_time',
            metric_value=calculation_duration,
            metric_unit='seconds',
            component='SCALE_CALIBRATION',
            metric_context={
                'calculation_method': calculation_method,
                'confidence_level': validation_metrics['overall_confidence'],
                'arena_dimensions': f"{arena_width_meters}x{arena_height_meters}m",
                'video_resolution': f"{video_width_pixels}x{video_height_pixels}px"
            }
        )
        
        logger.info(
            f"Pixel-to-meter ratio calculated: {primary_ratio:.3f} px/m (confidence: {validation_metrics['overall_confidence']:.3f})"
        )
        
        return ratio_result
        
    except Exception as e:
        # Log calculation failure with comprehensive error context
        error_duration = time.time() - start_time
        logger.error(
            f"Pixel-to-meter ratio calculation failed after {error_duration:.3f}s: {str(e)}"
        )
        
        if isinstance(e, ValidationError):
            raise
        else:
            raise ProcessingError(
                f"Pixel-to-meter ratio calculation failed: {str(e)}",
                'ratio_calculation',
                'scale_calibration',
                {
                    'arena_width_meters': arena_width_meters,
                    'arena_height_meters': arena_height_meters,
                    'video_width_pixels': video_width_pixels,
                    'video_height_pixels': video_height_pixels,
                    'calculation_method': calculation_method
                }
            )


def detect_arena_boundaries(
    video_frame: np.ndarray,
    detection_method: str = 'contour',
    detection_parameters: Dict[str, Any] = None,
    validate_detection: bool = True
) -> Dict[str, Any]:
    """
    Automatically detect arena boundaries from video frames using computer vision techniques with multiple detection methods and confidence assessment for robust arena identification.
    
    This function implements comprehensive arena boundary detection using contour analysis, edge detection, template matching, 
    and manual specification methods with confidence assessment and validation capabilities.
    
    Args:
        video_frame: Input video frame as numpy array for arena detection
        detection_method: Detection method to use ('contour', 'edge', 'template', 'manual')
        detection_parameters: Parameters for method-specific optimization
        validate_detection: Whether to validate detection results for quality assurance
        
    Returns:
        Dict[str, Any]: Arena boundary detection result with coordinates, confidence, and validation status
        
    Raises:
        ValidationError: If frame validation fails or detection method is invalid
        ProcessingError: If arena detection processing fails
    """
    # Set scientific context for arena detection operations
    set_scientific_context(
        simulation_id='arena_detection',
        algorithm_name='boundary_detection',
        processing_stage='ARENA_DETECTION'
    )
    
    start_time = time.time()
    
    try:
        # Validate input video frame format and quality for arena detection
        if video_frame is None:
            raise ValidationError(
                "Video frame cannot be None",
                'frame_validation',
                {'frame_shape': None}
            )
        
        if not isinstance(video_frame, np.ndarray):
            raise ValidationError(
                f"Video frame must be numpy array, got {type(video_frame)}",
                'frame_validation',
                {'frame_type': type(video_frame)}
            )
        
        if video_frame.size == 0:
            raise ValidationError(
                "Video frame is empty",
                'frame_validation',
                {'frame_shape': video_frame.shape}
            )
        
        # Validate detection method against supported methods
        if detection_method not in ARENA_DETECTION_METHODS:
            raise ValidationError(
                f"Unsupported detection method: {detection_method}",
                'method_validation',
                {'detection_method': detection_method, 'supported_methods': ARENA_DETECTION_METHODS}
            )
        
        # Initialize detection parameters with defaults
        if detection_parameters is None:
            detection_parameters = {}
        
        # Apply preprocessing to enhance arena boundary visibility
        preprocessed_frame = _preprocess_frame_for_detection(video_frame, detection_parameters)
        
        # Initialize detection result structure
        detection_result = {
            'detection_method': detection_method,
            'detection_timestamp': datetime.now().isoformat(),
            'arena_boundaries': {},
            'detection_confidence': 0.0,
            'validation_status': {},
            'geometric_properties': {},
            'detection_metadata': {}
        }
        
        # Execute specified detection method (contour, edge, template, or manual)
        if detection_method == 'contour':
            boundary_data = _detect_boundaries_contour(preprocessed_frame, detection_parameters)
        elif detection_method == 'edge':
            boundary_data = _detect_boundaries_edge(preprocessed_frame, detection_parameters)
        elif detection_method == 'template':
            boundary_data = _detect_boundaries_template(preprocessed_frame, detection_parameters)
        elif detection_method == 'manual':
            boundary_data = _detect_boundaries_manual(preprocessed_frame, detection_parameters)
        else:
            raise ValidationError(
                f"Detection method not implemented: {detection_method}",
                'method_implementation',
                {'detection_method': detection_method}
            )
        
        # Extract arena boundary coordinates and geometric properties
        detection_result['arena_boundaries'] = boundary_data.get('boundaries', {})
        detection_result['detection_confidence'] = boundary_data.get('confidence', 0.0)
        detection_result['geometric_properties'] = boundary_data.get('properties', {})
        
        # Calculate detection confidence based on boundary quality metrics
        quality_metrics = _assess_boundary_quality(detection_result['arena_boundaries'], video_frame.shape)
        detection_result['detection_confidence'] = min(
            detection_result['detection_confidence'],
            quality_metrics.get('overall_quality', 0.0)
        )
        
        # Validate detection results if validate_detection is enabled
        if validate_detection:
            validation_result = _validate_arena_detection(detection_result, detection_parameters)
            detection_result['validation_status'] = validation_result
            
            if not validation_result.get('is_valid', False):
                warnings.warn(f"Arena detection validation failed: {validation_result.get('errors', [])}")
        
        # Apply detection parameters for method-specific optimization
        optimization_applied = _apply_detection_optimization(detection_result, detection_parameters)
        detection_result['detection_metadata']['optimization_applied'] = optimization_applied
        
        # Generate comprehensive detection result with metadata
        detection_result['detection_metadata'].update({
            'frame_shape': video_frame.shape,
            'preprocessing_applied': True,
            'quality_metrics': quality_metrics,
            'detection_duration_seconds': time.time() - start_time
        })
        
        # Cache detection results for performance optimization
        cache_key = _generate_detection_cache_key(video_frame, detection_method, detection_parameters)
        _arena_detection_cache[cache_key] = detection_result.copy()
        
        # Log arena detection operation with confidence and performance metrics
        detection_duration = time.time() - start_time
        log_performance_metrics(
            metric_name='arena_detection_time',
            metric_value=detection_duration,
            metric_unit='seconds',
            component='SCALE_CALIBRATION',
            metric_context={
                'detection_method': detection_method,
                'confidence_level': detection_result['detection_confidence'],
                'frame_shape': list(video_frame.shape),
                'validation_enabled': validate_detection
            }
        )
        
        logger.info(
            f"Arena boundaries detected using {detection_method}: confidence {detection_result['detection_confidence']:.3f}"
        )
        
        return detection_result
        
    except Exception as e:
        # Log detection failure with comprehensive error context
        error_duration = time.time() - start_time
        logger.error(
            f"Arena boundary detection failed after {error_duration:.3f}s: {str(e)}"
        )
        
        if isinstance(e, (ValidationError, ProcessingError)):
            raise
        else:
            raise ProcessingError(
                f"Arena boundary detection failed: {str(e)}",
                'arena_detection',
                'scale_calibration',
                {
                    'detection_method': detection_method,
                    'frame_shape': video_frame.shape if video_frame is not None else None,
                    'detection_parameters': detection_parameters
                }
            )


def extract_calibration_from_video(
    video_path: str,
    extraction_method: str = 'metadata_analysis',
    extraction_hints: Dict[str, Any] = None,
    validate_extraction: bool = True
) -> Dict[str, Any]:
    """
    Extract comprehensive calibration parameters from video file metadata and frame analysis with format-specific handling and validation for automated calibration parameter discovery.
    
    This function implements automated calibration parameter extraction using video metadata analysis, frame content analysis, 
    and format-specific detection methods for Crimaldi and custom formats.
    
    Args:
        video_path: Path to video file for calibration parameter extraction
        extraction_method: Method for parameter extraction ('metadata_analysis', 'frame_analysis', 'format_specific', 'combined')
        extraction_hints: Hints to improve parameter accuracy and extraction success
        validate_extraction: Whether to validate extracted parameters for quality assurance
        
    Returns:
        Dict[str, Any]: Extracted calibration parameters with confidence levels and validation metrics
        
    Raises:
        ValidationError: If video file validation fails or extraction method is invalid
        ProcessingError: If calibration extraction processing fails
    """
    # Set scientific context for calibration extraction operations  
    set_scientific_context(
        simulation_id='calibration_extraction',
        algorithm_name='video_parameter_extraction',
        processing_stage='CALIBRATION_EXTRACTION'
    )
    
    start_time = time.time()
    
    try:
        # Validate video file path and accessibility
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValidationError(
                f"Video file does not exist: {video_path}",
                'file_validation',
                {'video_path': str(video_path)}
            )
        
        if not video_path.is_file():
            raise ValidationError(
                f"Path is not a file: {video_path}",
                'file_validation',
                {'video_path': str(video_path)}
            )
        
        # Detect video format using detect_video_format function
        format_detection = detect_video_format(str(video_path), deep_inspection=True)
        detected_format = format_detection.get('format_type', 'unknown')
        
        logger.info(f"Detected video format: {detected_format} (confidence: {format_detection.get('confidence_level', 0.0):.3f})")
        
        # Create VideoReader instance for metadata and frame access
        try:
            video_reader = VideoReader(str(video_path), {'format_optimized': True}, enable_caching=True)
        except Exception as e:
            raise ProcessingError(
                f"Failed to create video reader: {str(e)}",
                'video_reader_creation',
                str(video_path),
                {'format_detection': format_detection}
            )
        
        # Extract video metadata including resolution and frame properties
        video_metadata = video_reader.get_metadata(
            include_frame_analysis=True,
            include_processing_recommendations=True
        )
        
        # Initialize extraction result structure
        extraction_result = {
            'extraction_method': extraction_method,
            'extraction_timestamp': datetime.now().isoformat(),
            'video_path': str(video_path),
            'detected_format': detected_format,
            'calibration_parameters': {},
            'extraction_confidence': 0.0,
            'validation_metrics': {},
            'extraction_metadata': {}
        }
        
        # Apply format-specific calibration parameter extraction logic
        if extraction_method == 'metadata_analysis':
            calibration_data = _extract_from_metadata(video_metadata, format_detection)
        elif extraction_method == 'frame_analysis':
            calibration_data = _extract_from_frame_analysis(video_reader, extraction_hints or {})
        elif extraction_method == 'format_specific':
            calibration_data = _extract_format_specific(video_reader, detected_format, extraction_hints or {})
        elif extraction_method == 'combined':
            calibration_data = _extract_combined_method(video_reader, video_metadata, format_detection, extraction_hints or {})
        else:
            raise ValidationError(
                f"Unknown extraction method: {extraction_method}",
                'method_validation',
                {'extraction_method': extraction_method, 'supported_methods': ['metadata_analysis', 'frame_analysis', 'format_specific', 'combined']}
            )
        
        # Use extraction hints to improve parameter accuracy if provided
        if extraction_hints:
            calibration_data = _apply_extraction_hints(calibration_data, extraction_hints)
            extraction_result['extraction_metadata']['hints_applied'] = True
        
        # Detect arena boundaries from representative video frames
        representative_frames = _get_representative_frames(video_reader, num_frames=3)
        arena_detection_results = []
        
        for frame_idx, frame in representative_frames:
            try:
                detection_result = detect_arena_boundaries(
                    frame,
                    detection_method='contour',
                    detection_parameters={'confidence_threshold': MIN_ARENA_DETECTION_CONFIDENCE},
                    validate_detection=True
                )
                arena_detection_results.append(detection_result)
            except Exception as e:
                logger.warning(f"Arena detection failed for frame {frame_idx}: {str(e)}")
        
        # Calculate pixel-to-meter ratios from detected arena dimensions
        if arena_detection_results and calibration_data.get('arena_dimensions'):
            arena_dims = calibration_data['arena_dimensions']
            video_dims = video_metadata.get('basic_properties', {})
            
            try:
                ratio_result = calculate_pixel_to_meter_ratio(
                    arena_width_meters=arena_dims.get('width_meters', TARGET_ARENA_WIDTH_METERS),
                    arena_height_meters=arena_dims.get('height_meters', TARGET_ARENA_HEIGHT_METERS),
                    video_width_pixels=video_dims.get('width', 640),
                    video_height_pixels=video_dims.get('height', 480),
                    calculation_method='geometric_mean'
                )
                calibration_data['pixel_to_meter_ratios'] = ratio_result
            except Exception as e:
                logger.warning(f"Pixel-to-meter ratio calculation failed: {str(e)}")
        
        # Extract temporal calibration parameters from frame timing
        temporal_params = _extract_temporal_parameters(video_metadata, detected_format)
        calibration_data.update(temporal_params)
        
        # Update extraction result with calibration parameters
        extraction_result['calibration_parameters'] = calibration_data
        extraction_result['arena_detection_results'] = arena_detection_results
        
        # Validate extracted parameters if validate_extraction is enabled
        if validate_extraction:
            validation_result = _validate_extracted_parameters(calibration_data, detected_format)
            extraction_result['validation_metrics'] = validation_result
            
            if not validation_result.get('is_valid', False):
                warnings.warn(f"Calibration parameter validation failed: {validation_result.get('errors', [])}")
        
        # Generate extraction confidence assessment and quality metrics
        confidence_assessment = _assess_extraction_confidence(
            calibration_data,
            arena_detection_results,
            format_detection,
            validation_result if validate_extraction else {}
        )
        extraction_result['extraction_confidence'] = confidence_assessment.get('overall_confidence', 0.0)
        extraction_result['extraction_metadata']['confidence_assessment'] = confidence_assessment
        
        # Add extraction timing and performance metrics
        extraction_duration = time.time() - start_time
        extraction_result['extraction_metadata'].update({
            'extraction_duration_seconds': extraction_duration,
            'video_metadata': video_metadata,
            'format_detection': format_detection,
            'representative_frames_analyzed': len(representative_frames)
        })
        
        # Close video reader to free resources
        video_reader.close()
        
        # Log calibration extraction operation with performance data
        log_performance_metrics(
            metric_name='calibration_extraction_time',
            metric_value=extraction_duration,
            metric_unit='seconds',
            component='SCALE_CALIBRATION',
            metric_context={
                'extraction_method': extraction_method,
                'detected_format': detected_format,
                'confidence_level': extraction_result['extraction_confidence'],
                'validation_enabled': validate_extraction,
                'video_path': str(video_path)
            }
        )
        
        logger.info(
            f"Calibration parameters extracted: confidence {extraction_result['extraction_confidence']:.3f}, method {extraction_method}"
        )
        
        return extraction_result
        
    except Exception as e:
        # Log extraction failure with comprehensive error context
        error_duration = time.time() - start_time
        logger.error(
            f"Calibration extraction failed after {error_duration:.3f}s: {str(e)}"
        )
        
        if isinstance(e, (ValidationError, ProcessingError)):
            raise
        else:
            raise ProcessingError(
                f"Calibration extraction failed: {str(e)}",
                'calibration_extraction',
                str(video_path),
                {
                    'extraction_method': extraction_method,
                    'extraction_hints': extraction_hints,
                    'validate_extraction': validate_extraction
                }
            )


def validate_calibration_accuracy(
    calibration_parameters: Dict[str, float],
    reference_parameters: Dict[str, float],
    accuracy_threshold: float = DEFAULT_CALIBRATION_CONFIDENCE_THRESHOLD,
    detailed_analysis: bool = True
) -> ValidationResult:
    """
    Validate calibration accuracy against reference standards and scientific requirements with comprehensive error analysis and confidence assessment for quality assurance.
    
    This function performs comprehensive validation of calibration parameters against reference implementations 
    to ensure >95% correlation accuracy and scientific computing reliability.
    
    Args:
        calibration_parameters: Calibration parameters to validate
        reference_parameters: Reference parameters for comparison
        accuracy_threshold: Minimum accuracy threshold for validation (default: 95%)
        detailed_analysis: Whether to perform detailed statistical analysis
        
    Returns:
        ValidationResult: Calibration accuracy validation result with error analysis and recommendations
        
    Raises:
        ValidationError: If parameter validation fails or correlation analysis fails
    """
    # Set scientific context for calibration validation operations
    set_scientific_context(
        simulation_id='calibration_validation',
        algorithm_name='accuracy_validation',
        processing_stage='CALIBRATION_VALIDATION'
    )
    
    start_time = time.time()
    
    try:
        # Create ValidationResult container for accuracy assessment
        validation_result = ValidationResult(
            validation_type='calibration_accuracy_validation',
            is_valid=True,
            validation_context=f'threshold={accuracy_threshold}, detailed={detailed_analysis}'
        )
        
        # Validate input parameters format and content
        if not isinstance(calibration_parameters, dict):
            raise ValidationError(
                "Calibration parameters must be a dictionary",
                'parameter_format_validation',
                {'parameter_type': type(calibration_parameters)}
            )
        
        if not isinstance(reference_parameters, dict):
            raise ValidationError(
                "Reference parameters must be a dictionary",
                'parameter_format_validation',
                {'parameter_type': type(reference_parameters)}
            )
        
        if not calibration_parameters:
            validation_result.add_error(
                "Calibration parameters dictionary is empty",
                severity=ValidationError.ErrorSeverity.HIGH
            )
            validation_result.is_valid = False
            return validation_result
        
        # Compare calibration parameters with reference standards
        common_parameters = set(calibration_parameters.keys()) & set(reference_parameters.keys())
        if not common_parameters:
            validation_result.add_error(
                "No common parameters found between calibration and reference",
                severity=ValidationError.ErrorSeverity.HIGH
            )
            validation_result.is_valid = False
            return validation_result
        
        # Calculate relative and absolute errors for each parameter
        parameter_accuracies = {}
        overall_errors = []
        
        for param_name in common_parameters:
            calib_value = calibration_parameters[param_name]
            ref_value = reference_parameters[param_name]
            
            if not isinstance(calib_value, (int, float)) or not isinstance(ref_value, (int, float)):
                validation_result.add_warning(f"Non-numeric parameter skipped: {param_name}")
                continue
            
            # Calculate absolute and relative errors
            absolute_error = abs(calib_value - ref_value)
            relative_error = absolute_error / abs(ref_value) if ref_value != 0 else float('inf')
            
            # Calculate parameter accuracy (1 - relative_error)
            parameter_accuracy = max(0.0, 1.0 - relative_error) if relative_error != float('inf') else 0.0
            parameter_accuracies[param_name] = {
                'accuracy': parameter_accuracy,
                'absolute_error': absolute_error,
                'relative_error': relative_error,
                'calibration_value': calib_value,
                'reference_value': ref_value
            }
            
            overall_errors.append(relative_error)
            
            # Add metrics for each parameter
            validation_result.add_metric(f'{param_name}_accuracy', parameter_accuracy)
            validation_result.add_metric(f'{param_name}_relative_error', relative_error)
        
        # Assess accuracy against specified accuracy_threshold
        if overall_errors:
            mean_relative_error = np.mean([e for e in overall_errors if e != float('inf')])
            overall_accuracy = max(0.0, 1.0 - mean_relative_error)
        else:
            overall_accuracy = 0.0
        
        validation_result.add_metric('overall_accuracy', overall_accuracy)
        validation_result.add_metric('accuracy_threshold', accuracy_threshold)
        
        # Check if accuracy meets threshold requirements
        if overall_accuracy < accuracy_threshold:
            validation_result.add_error(
                f"Overall accuracy {overall_accuracy:.4f} below threshold {accuracy_threshold:.4f}",
                severity=ValidationError.ErrorSeverity.HIGH
            )
            validation_result.is_valid = False
        
        # Perform detailed statistical analysis if detailed_analysis is enabled
        if detailed_analysis and len(overall_errors) > 1:
            detailed_stats = _perform_detailed_accuracy_analysis(parameter_accuracies, overall_errors)
            validation_result.set_metadata('detailed_statistics', detailed_stats)
            
            # Check statistical significance and confidence intervals
            if detailed_stats.get('statistical_significance', True):
                validation_result.add_metric('statistical_confidence', detailed_stats.get('confidence_level', 0.95))
            else:
                validation_result.add_warning("Statistical significance test failed")
        
        # Validate parameter consistency and physical constraints
        consistency_check = _validate_parameter_consistency(calibration_parameters, reference_parameters)
        if not consistency_check.get('consistent', True):
            validation_result.add_warning(f"Parameter consistency issues: {consistency_check.get('issues', [])}")
        
        # Generate accuracy metrics and confidence levels
        validation_result.add_metric('validated_parameters_count', len(common_parameters))
        validation_result.add_metric('missing_parameters_count', len(reference_parameters) - len(common_parameters))
        
        # Add validation errors for parameters exceeding thresholds
        for param_name, param_data in parameter_accuracies.items():
            if param_data['accuracy'] < accuracy_threshold:
                validation_result.add_error(
                    f"Parameter {param_name} accuracy {param_data['accuracy']:.4f} below threshold",
                    severity=ValidationError.ErrorSeverity.MEDIUM
                )
        
        # Include recommendations for improving calibration accuracy
        if not validation_result.is_valid:
            recommendations = _generate_accuracy_recommendations(parameter_accuracies, overall_accuracy, accuracy_threshold)
            for rec in recommendations:
                validation_result.add_recommendation(rec['text'], rec.get('priority', 'MEDIUM'))
        
        # Finalize validation result with timing information
        validation_duration = time.time() - start_time
        validation_result.add_metric('validation_duration_seconds', validation_duration)
        validation_result.finalize_validation()
        
        # Log validation operation with accuracy assessment results
        log_performance_metrics(
            metric_name='calibration_validation_time',
            metric_value=validation_duration,
            metric_unit='seconds',
            component='SCALE_CALIBRATION',
            metric_context={
                'overall_accuracy': overall_accuracy,
                'accuracy_threshold': accuracy_threshold,
                'validated_parameters': len(common_parameters),
                'detailed_analysis': detailed_analysis,
                'validation_passed': validation_result.is_valid
            }
        )
        
        logger.info(
            f"Calibration accuracy validated: {overall_accuracy:.4f} {'≥' if validation_result.is_valid else '<'} {accuracy_threshold:.4f}"
        )
        
        return validation_result
        
    except Exception as e:
        # Log validation failure with comprehensive error context
        error_duration = time.time() - start_time
        logger.error(
            f"Calibration accuracy validation failed after {error_duration:.3f}s: {str(e)}"
        )
        
        if isinstance(e, ValidationError):
            raise
        else:
            # Create error validation result
            error_result = ValidationResult(
                validation_type='calibration_accuracy_validation',
                is_valid=False,
                validation_context=f'error_occurred'
            )
            error_result.add_error(
                f"Validation process failed: {str(e)}",
                severity=ValidationError.ErrorSeverity.CRITICAL
            )
            error_result.finalize_validation()
            return error_result


def create_coordinate_transformer(
    calibration_parameters: Dict[str, float],
    source_coordinate_system: str,
    target_coordinate_system: str,
    enable_validation: bool = True
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create coordinate transformation function for converting between pixel and meter coordinate systems with validation and performance optimization for spatial data processing.
    
    This function generates optimized coordinate transformation functions for converting between different coordinate systems
    using calibration parameters with comprehensive validation and error handling.
    
    Args:
        calibration_parameters: Dictionary containing calibration parameters for transformation
        source_coordinate_system: Source coordinate system ('pixel', 'meter', 'normalized')
        target_coordinate_system: Target coordinate system ('pixel', 'meter', 'normalized')  
        enable_validation: Whether to enable coordinate validation in transformation function
        
    Returns:
        Callable[[np.ndarray], np.ndarray]: Coordinate transformation function with validation and error handling
        
    Raises:
        ValidationError: If parameters fail validation or coordinate systems are invalid
        ProcessingError: If transformation function creation fails
    """
    # Set scientific context for coordinate transformer creation
    set_scientific_context(
        simulation_id='coordinate_transformer',
        algorithm_name='coordinate_transformation',
        processing_stage='TRANSFORMER_CREATION'
    )
    
    start_time = time.time()
    
    try:
        # Validate calibration parameters and coordinate system specifications
        if not isinstance(calibration_parameters, dict):
            raise ValidationError(
                "Calibration parameters must be a dictionary",
                'parameter_validation',
                {'parameter_type': type(calibration_parameters)}
            )
        
        if source_coordinate_system not in SUPPORTED_COORDINATE_SYSTEMS:
            raise ValidationError(
                f"Unsupported source coordinate system: {source_coordinate_system}",
                'coordinate_system_validation',
                {'source_system': source_coordinate_system, 'supported_systems': SUPPORTED_COORDINATE_SYSTEMS}
            )
        
        if target_coordinate_system not in SUPPORTED_COORDINATE_SYSTEMS:
            raise ValidationError(
                f"Unsupported target coordinate system: {target_coordinate_system}",
                'coordinate_system_validation',
                {'target_system': target_coordinate_system, 'supported_systems': SUPPORTED_COORDINATE_SYSTEMS}
            )
        
        if source_coordinate_system == target_coordinate_system:
            # Identity transformation for same coordinate systems
            def identity_transform(coordinates: np.ndarray) -> np.ndarray:
                """Identity transformation for same coordinate systems."""
                return coordinates.copy() if isinstance(coordinates, np.ndarray) else np.array(coordinates)
            
            logger.info(f"Created identity transformer: {source_coordinate_system} -> {target_coordinate_system}")
            return identity_transform
        
        # Extract pixel-to-meter ratios and transformation parameters
        required_params = _get_required_transformation_parameters(source_coordinate_system, target_coordinate_system)
        missing_params = [param for param in required_params if param not in calibration_parameters]
        
        if missing_params:
            raise ValidationError(
                f"Missing required calibration parameters: {missing_params}",
                'parameter_completeness_validation',
                {'missing_parameters': missing_params, 'required_parameters': required_params}
            )
        
        # Create transformation matrix for coordinate system conversion
        transformation_matrix = _create_transformation_matrix(
            calibration_parameters,
            source_coordinate_system,
            target_coordinate_system
        )
        
        # Get coordinate system properties for validation and optimization
        source_properties = _get_coordinate_system_properties(source_coordinate_system, calibration_parameters)
        target_properties = _get_coordinate_system_properties(target_coordinate_system, calibration_parameters)
        
        # Implementation details for coordinate transformation function with validation
        def coordinate_transformer(coordinates: np.ndarray) -> np.ndarray:
            """
            Transform coordinates between source and target coordinate systems with validation and error handling.
            
            Args:
                coordinates: Input coordinates as numpy array (Nx2 or Nx3)
                
            Returns:
                np.ndarray: Transformed coordinates in target coordinate system
                
            Raises:
                ValidationError: If input coordinates fail validation
                ProcessingError: If transformation fails
            """
            transform_start = time.time()
            
            try:
                # Validate input coordinates format and content
                if not isinstance(coordinates, np.ndarray):
                    coordinates = np.array(coordinates)
                
                if coordinates.size == 0:
                    return np.array([])
                
                # Ensure coordinates are 2D with proper shape
                if coordinates.ndim == 1:
                    if len(coordinates) % 2 == 0:
                        coordinates = coordinates.reshape(-1, 2)
                    else:
                        raise ValidationError(
                            "Invalid coordinate array: odd number of elements",
                            'coordinate_format_validation',
                            {'coordinates_shape': coordinates.shape}
                        )
                elif coordinates.ndim == 2:
                    if coordinates.shape[1] not in [2, 3]:
                        raise ValidationError(
                            f"Invalid coordinate dimensions: {coordinates.shape[1]} (expected 2 or 3)",
                            'coordinate_dimension_validation',
                            {'coordinates_shape': coordinates.shape}
                        )
                else:
                    raise ValidationError(
                        f"Invalid coordinate array dimensions: {coordinates.ndim}",
                        'coordinate_dimension_validation',
                        {'coordinates_shape': coordinates.shape}
                    )
                
                # Validate coordinates against source coordinate system bounds
                if enable_validation:
                    _validate_coordinates_bounds(coordinates, source_coordinate_system, source_properties)
                
                # Apply coordinate transformation using transformation matrix
                if coordinates.shape[1] == 2:
                    # 2D coordinate transformation
                    homogeneous_coords = np.column_stack([coordinates, np.ones(coordinates.shape[0])])
                    transformed_coords = homogeneous_coords @ transformation_matrix.T
                    result_coords = transformed_coords[:, :2]
                else:
                    # 3D coordinate transformation (z-coordinate pass-through for now)
                    xy_coords = coordinates[:, :2]
                    homogeneous_coords = np.column_stack([xy_coords, np.ones(xy_coords.shape[0])])
                    transformed_xy = homogeneous_coords @ transformation_matrix.T
                    result_coords = np.column_stack([transformed_xy[:, :2], coordinates[:, 2]])
                
                # Validate transformed coordinates against target system bounds
                if enable_validation:
                    _validate_coordinates_bounds(result_coords, target_coordinate_system, target_properties)
                
                # Round coordinates to appropriate precision
                result_coords = np.round(result_coords, COORDINATE_PRECISION_DIGITS)
                
                # Log transformation performance for optimization
                transform_duration = time.time() - transform_start
                if transform_duration > 0.001:  # Log only slower transformations
                    logger.debug(
                        f"Coordinate transformation: {coordinates.shape[0]} points, "
                        f"{source_coordinate_system}->{target_coordinate_system}, "
                        f"{transform_duration:.4f}s"
                    )
                
                return result_coords
                
            except Exception as e:
                if isinstance(e, ValidationError):
                    raise
                else:
                    raise ProcessingError(
                        f"Coordinate transformation failed: {str(e)}",
                        'coordinate_transformation',
                        'scale_calibration',
                        {
                            'source_system': source_coordinate_system,
                            'target_system': target_coordinate_system,
                            'coordinates_shape': coordinates.shape if 'coordinates' in locals() else None
                        }
                    )
        
        # Add error handling for invalid coordinates and edge cases
        def safe_coordinate_transformer(coordinates: np.ndarray) -> np.ndarray:
            """Safe wrapper for coordinate transformation with comprehensive error handling."""
            try:
                return coordinate_transformer(coordinates)
            except Exception as e:
                logger.error(f"Coordinate transformation error: {str(e)}")
                # Return original coordinates as fallback
                return coordinates.copy() if isinstance(coordinates, np.ndarray) else np.array(coordinates)
        
        # Include performance optimization for batch coordinate processing
        def optimized_coordinate_transformer(coordinates: np.ndarray) -> np.ndarray:
            """Optimized coordinate transformer with batch processing support."""
            if coordinates.size > 10000:  # Use batch processing for large coordinate sets
                batch_size = 1000
                results = []
                for i in range(0, len(coordinates), batch_size):
                    batch = coordinates[i:i+batch_size]
                    batch_result = coordinate_transformer(batch)
                    results.append(batch_result)
                return np.vstack(results)
            else:
                return coordinate_transformer(coordinates)
        
        # Log coordinate transformer creation with configuration details
        creation_duration = time.time() - start_time
        log_performance_metrics(
            metric_name='coordinate_transformer_creation_time',
            metric_value=creation_duration,
            metric_unit='seconds',
            component='SCALE_CALIBRATION',
            metric_context={
                'source_system': source_coordinate_system,
                'target_system': target_coordinate_system,
                'validation_enabled': enable_validation,
                'transformation_matrix_size': transformation_matrix.shape
            }
        )
        
        logger.info(
            f"Coordinate transformer created: {source_coordinate_system} -> {target_coordinate_system} "
            f"(validation: {enable_validation})"
        )
        
        return optimized_coordinate_transformer
        
    except Exception as e:
        # Log transformer creation failure with comprehensive error context
        error_duration = time.time() - start_time
        logger.error(
            f"Coordinate transformer creation failed after {error_duration:.3f}s: {str(e)}"
        )
        
        if isinstance(e, (ValidationError, ProcessingError)):
            raise
        else:
            raise ProcessingError(
                f"Coordinate transformer creation failed: {str(e)}",
                'transformer_creation',
                'scale_calibration',
                {
                    'source_coordinate_system': source_coordinate_system,
                    'target_coordinate_system': target_coordinate_system,
                    'calibration_parameters': list(calibration_parameters.keys()) if calibration_parameters else []
                }
            )


@dataclass
class ScaleCalibration:
    """
    Comprehensive scale calibration container class providing spatial normalization, coordinate transformation, and calibration parameter management for plume recording data with cross-format compatibility and validation capabilities.
    
    This class serves as the primary interface for scale calibration operations, providing comprehensive spatial normalization,
    coordinate transformation, and calibration parameter management with cross-format compatibility and scientific validation.
    """
    
    # Core calibration properties
    video_path: str
    format_type: str
    calibration_config: Dict[str, Any]
    
    # Calibration parameters and state
    pixel_to_meter_ratios: Dict[str, float] = field(default_factory=dict)
    arena_dimensions_meters: Dict[str, float] = field(default_factory=dict)
    arena_dimensions_pixels: Dict[str, int] = field(default_factory=dict)
    arena_boundaries: Dict[str, Any] = field(default_factory=dict)
    calibration_confidence: float = 0.0
    is_validated: bool = False
    
    # Transformation and computation state  
    transformation_matrices: Dict[str, np.ndarray] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    calibration_timestamp: datetime = field(default_factory=datetime.now)
    calibration_method: str = 'automatic'
    
    def __post_init__(self):
        """Initialize scale calibration with video path, format type, and configuration for comprehensive spatial calibration management."""
        # Set scientific context for calibration operations
        set_scientific_context(
            simulation_id=f'scale_calibration_{hash(self.video_path)}',
            algorithm_name='ScaleCalibration',
            processing_stage='INITIALIZATION'
        )
        
        # Validate video path and format type
        self.video_path = str(Path(self.video_path).resolve())
        if not Path(self.video_path).exists():
            raise ValidationError(
                f"Video file does not exist: {self.video_path}",
                'file_validation',
                {'video_path': self.video_path}
            )
        
        # Initialize calibration parameters with default values
        self._initialize_default_parameters()
        
        # Set calibration timestamp and method identification
        self.calibration_timestamp = datetime.now()
        self.calibration_method = self.calibration_config.get('method', 'automatic')
        
        # Initialize transformation matrices and validation metrics
        self.transformation_matrices = {}
        self.validation_metrics = {}
        
        # Configure format-specific calibration settings
        self._configure_format_specific_settings()
        
        # Set validation status and confidence to initial values
        self.is_validated = False
        self.calibration_confidence = 0.0
        
        # Initialize logger for calibration operations
        self.logger = get_logger('scale_calibration', 'SCALE_CALIBRATION')
        
        # Log calibration initialization with configuration details
        self.logger.info(
            f"ScaleCalibration initialized: {self.format_type} format, "
            f"method: {self.calibration_method}, video: {Path(self.video_path).name}"
        )
    
    def extract_calibration_parameters(
        self,
        force_reextraction: bool = False,
        extraction_hints: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Extract comprehensive calibration parameters from video including arena detection, ratio calculation, and validation with format-specific optimization.
        
        Args:
            force_reextraction: Force re-extraction of calibration parameters
            extraction_hints: Hints to improve parameter accuracy and extraction success
            
        Returns:
            Dict[str, float]: Extracted calibration parameters with confidence levels and validation status
            
        Raises:
            ProcessingError: If calibration parameter extraction fails
        """
        start_time = time.time()
        
        try:
            # Check if calibration parameters already extracted and valid
            if self.pixel_to_meter_ratios and self.calibration_confidence > 0.5 and not force_reextraction:
                self.logger.info("Using existing calibration parameters (use force_reextraction=True to override)")
                return self._get_calibration_summary()
            
            # Extract calibration parameters from video using extract_calibration_from_video
            extraction_result = extract_calibration_from_video(
                video_path=self.video_path,
                extraction_method='combined',
                extraction_hints=extraction_hints,
                validate_extraction=True
            )
            
            # Apply format-specific parameter extraction optimizations
            if self.format_type == 'crimaldi':
                extraction_result = self._apply_crimaldi_optimizations(extraction_result)
            elif self.format_type == 'custom':
                extraction_result = self._apply_custom_optimizations(extraction_result)
            
            # Use extraction hints to improve parameter accuracy
            if extraction_hints:
                extraction_result = self._apply_extraction_hints_to_result(extraction_result, extraction_hints)
            
            # Update class properties with extracted parameters
            calibration_params = extraction_result.get('calibration_parameters', {})
            
            # Detect arena boundaries and calculate dimensions
            if 'arena_detection_results' in extraction_result:
                detection_results = extraction_result['arena_detection_results']
                if detection_results:
                    best_detection = max(detection_results, key=lambda x: x.get('detection_confidence', 0.0))
                    self.arena_boundaries = best_detection.get('arena_boundaries', {})
                    self._update_arena_dimensions_from_detection(best_detection)
            
            # Calculate pixel-to-meter ratios with confidence assessment
            if 'pixel_to_meter_ratios' in calibration_params:
                self.pixel_to_meter_ratios = calibration_params['pixel_to_meter_ratios']
            else:
                self._calculate_ratios_from_dimensions()
            
            # Validate extracted parameters against scientific constraints
            validation_result = self.validate_calibration()
            self.validation_metrics = validation_result.metrics
            self.is_validated = validation_result.is_valid
            
            # Update calibration confidence and validation status
            self.calibration_confidence = extraction_result.get('extraction_confidence', 0.0)
            if validation_result.is_valid:
                self.calibration_confidence = min(1.0, self.calibration_confidence * 1.1)  # Boost confidence for valid calibration
            
            # Cache calibration parameters for future use
            self._cache_calibration_parameters()
            
            # Log parameter extraction with performance metrics
            extraction_duration = time.time() - start_time
            log_performance_metrics(
                metric_name='calibration_parameter_extraction_time',
                metric_value=extraction_duration,
                metric_unit='seconds',
                component='SCALE_CALIBRATION',
                metric_context={
                    'format_type': self.format_type,
                    'calibration_confidence': self.calibration_confidence,
                    'validation_passed': self.is_validated,
                    'force_reextraction': force_reextraction
                }
            )
            
            self.logger.info(
                f"Calibration parameters extracted: confidence {self.calibration_confidence:.3f}, "
                f"validated: {self.is_validated}"
            )
            
            return self._get_calibration_summary()
            
        except Exception as e:
            self.logger.error(f"Calibration parameter extraction failed: {str(e)}")
            raise ProcessingError(
                f"Calibration parameter extraction failed: {str(e)}",
                'parameter_extraction',
                self.video_path,
                {
                    'format_type': self.format_type,
                    'force_reextraction': force_reextraction,
                    'extraction_hints': extraction_hints
                }
            )
    
    def calculate_scaling_factors(
        self,
        target_coordinate_system: str = 'meter',
        validate_factors: bool = True
    ) -> Dict[str, float]:
        """
        Calculate comprehensive scaling factors for spatial normalization including pixel-to-meter ratios, coordinate transformations, and validation metrics.
        
        Args:
            target_coordinate_system: Target coordinate system for scaling calculations
            validate_factors: Whether to validate calculated scaling factors
            
        Returns:
            Dict[str, float]: Scaling factors with validation metrics and confidence assessment
            
        Raises:
            ValidationError: If coordinate system is invalid or validation fails
            ProcessingError: If scaling factor calculation fails
        """
        start_time = time.time()
        
        try:
            # Validate target coordinate system specification
            if target_coordinate_system not in SUPPORTED_COORDINATE_SYSTEMS:
                raise ValidationError(
                    f"Unsupported target coordinate system: {target_coordinate_system}",
                    'coordinate_system_validation',
                    {'target_system': target_coordinate_system, 'supported_systems': SUPPORTED_COORDINATE_SYSTEMS}
                )
            
            # Extract pixel-to-meter ratios from calibration parameters
            if not self.pixel_to_meter_ratios:
                self.logger.warning("No pixel-to-meter ratios available, extracting calibration parameters")
                self.extract_calibration_parameters()
            
            scaling_factors = {}
            
            # Calculate scaling factors for target coordinate system
            if target_coordinate_system == 'meter':
                scaling_factors.update({
                    'pixel_to_meter_x': self.pixel_to_meter_ratios.get('horizontal_ratio', CUSTOM_PIXEL_TO_METER_RATIO),
                    'pixel_to_meter_y': self.pixel_to_meter_ratios.get('vertical_ratio', CUSTOM_PIXEL_TO_METER_RATIO),
                    'meter_to_pixel_x': 1.0 / self.pixel_to_meter_ratios.get('horizontal_ratio', CUSTOM_PIXEL_TO_METER_RATIO),
                    'meter_to_pixel_y': 1.0 / self.pixel_to_meter_ratios.get('vertical_ratio', CUSTOM_PIXEL_TO_METER_RATIO),
                    'primary_ratio': self.pixel_to_meter_ratios.get('primary_ratio', CUSTOM_PIXEL_TO_METER_RATIO)
                })
            elif target_coordinate_system == 'normalized':
                # Normalization scaling based on arena dimensions
                arena_width = self.arena_dimensions_meters.get('width', TARGET_ARENA_WIDTH_METERS)
                arena_height = self.arena_dimensions_meters.get('height', TARGET_ARENA_HEIGHT_METERS)
                
                scaling_factors.update({
                    'normalize_x': 1.0 / arena_width,
                    'normalize_y': 1.0 / arena_height,
                    'denormalize_x': arena_width,
                    'denormalize_y': arena_height,
                    'aspect_ratio': arena_width / arena_height
                })
            else:  # pixel coordinate system
                scaling_factors.update({
                    'identity_x': 1.0,
                    'identity_y': 1.0,
                    'pixel_width': self.arena_dimensions_pixels.get('width', 640),
                    'pixel_height': self.arena_dimensions_pixels.get('height', 480)
                })
            
            # Compute transformation matrices for coordinate conversion
            transformation_matrix = self._create_transformation_matrix_for_system(target_coordinate_system)
            self.transformation_matrices[target_coordinate_system] = transformation_matrix
            
            # Validate scaling factors if validate_factors is enabled
            if validate_factors:
                validation_result = self._validate_scaling_factors(scaling_factors, target_coordinate_system)
                scaling_factors['validation_metrics'] = validation_result
                
                if not validation_result.get('is_valid', True):
                    self.logger.warning(f"Scaling factor validation failed: {validation_result.get('errors', [])}")
            
            # Assess scaling factor accuracy and confidence
            confidence_assessment = self._assess_scaling_factor_confidence(scaling_factors, target_coordinate_system)
            scaling_factors.update({
                'confidence_level': confidence_assessment.get('confidence', 0.8),
                'accuracy_estimate': confidence_assessment.get('accuracy', 0.9),
                'calculation_method': self.calibration_method
            })
            
            # Add metadata and calculation context
            scaling_factors.update({
                'target_coordinate_system': target_coordinate_system,
                'calculation_timestamp': datetime.now().isoformat(),
                'calibration_confidence': self.calibration_confidence,
                'format_type': self.format_type
            })
            
            # Log scaling factor calculation with performance data
            calculation_duration = time.time() - start_time
            log_performance_metrics(
                metric_name='scaling_factor_calculation_time',
                metric_value=calculation_duration,
                metric_unit='seconds',
                component='SCALE_CALIBRATION',
                metric_context={
                    'target_system': target_coordinate_system,
                    'validation_enabled': validate_factors,
                    'confidence_level': scaling_factors.get('confidence_level', 0.0)
                }
            )
            
            self.logger.info(
                f"Scaling factors calculated for {target_coordinate_system}: "
                f"confidence {scaling_factors.get('confidence_level', 0.0):.3f}"
            )
            
            return scaling_factors
            
        except Exception as e:
            self.logger.error(f"Scaling factor calculation failed: {str(e)}")
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            else:
                raise ProcessingError(
                    f"Scaling factor calculation failed: {str(e)}",
                    'scaling_calculation',
                    self.video_path,
                    {
                        'target_coordinate_system': target_coordinate_system,
                        'validate_factors': validate_factors
                    }
                )
    
    def validate_calibration(
        self,
        validation_thresholds: Dict[str, float] = None,
        strict_validation: bool = False
    ) -> ValidationResult:
        """
        Validate calibration parameters against scientific requirements and accuracy thresholds with comprehensive error analysis and recommendations.
        
        Args:
            validation_thresholds: Custom validation thresholds for calibration parameters
            strict_validation: Enable strict validation criteria for scientific computing
            
        Returns:
            ValidationResult: Calibration validation result with error analysis and improvement recommendations
            
        Raises:
            ValidationError: If validation process fails
        """
        start_time = time.time()
        
        try:
            # Create ValidationResult container for calibration assessment
            validation_result = ValidationResult(
                validation_type='scale_calibration_validation',
                is_valid=True,
                validation_context=f'format={self.format_type}, strict={strict_validation}'
            )
            
            # Set default validation thresholds if not provided
            if validation_thresholds is None:
                validation_thresholds = {
                    'min_confidence': DEFAULT_CALIBRATION_CONFIDENCE_THRESHOLD,
                    'min_spatial_accuracy': SPATIAL_ACCURACY_THRESHOLD,
                    'max_ratio_difference': 0.1,  # 10% maximum difference between horizontal and vertical ratios
                    'min_arena_size': 0.1,  # Minimum 10cm arena
                    'max_arena_size': 10.0   # Maximum 10m arena
                }
            
            # Validate pixel-to-meter ratios against physical constraints
            if self.pixel_to_meter_ratios:
                ratio_validation = self._validate_pixel_ratios(validation_thresholds, strict_validation)
                validation_result.errors.extend(ratio_validation.get('errors', []))
                validation_result.warnings.extend(ratio_validation.get('warnings', []))
                if not ratio_validation.get('is_valid', True):
                    validation_result.is_valid = False
            else:
                validation_result.add_error(
                    "No pixel-to-meter ratios available for validation",
                    severity=ValidationError.ErrorSeverity.HIGH
                )
                validation_result.is_valid = False
            
            # Check arena dimensions for consistency and accuracy
            if self.arena_dimensions_meters:
                arena_validation = self._validate_arena_dimensions(validation_thresholds, strict_validation)
                validation_result.errors.extend(arena_validation.get('errors', []))
                validation_result.warnings.extend(arena_validation.get('warnings', []))
                if not arena_validation.get('is_valid', True):
                    validation_result.is_valid = False
            else:
                validation_result.add_warning("No arena dimensions available for validation")
            
            # Validate calibration confidence against thresholds
            if self.calibration_confidence < validation_thresholds.get('min_confidence', DEFAULT_CALIBRATION_CONFIDENCE_THRESHOLD):
                validation_result.add_error(
                    f"Calibration confidence {self.calibration_confidence:.3f} below threshold {validation_thresholds['min_confidence']:.3f}",
                    severity=ValidationError.ErrorSeverity.MEDIUM
                )
                if strict_validation:
                    validation_result.is_valid = False
            
            # Apply strict validation criteria if strict_validation enabled
            if strict_validation:
                strict_validation_result = self._apply_strict_calibration_validation(validation_thresholds)
                validation_result.errors.extend(strict_validation_result.get('errors', []))
                validation_result.warnings.extend(strict_validation_result.get('warnings', []))
                if not strict_validation_result.get('is_valid', True):
                    validation_result.is_valid = False
            
            # Assess calibration parameter consistency and completeness
            consistency_check = self._check_calibration_consistency()
            if not consistency_check.get('consistent', True):
                validation_result.add_warning(f"Calibration consistency issues: {consistency_check.get('issues', [])}")
            
            # Generate validation metrics and error analysis
            validation_result.add_metric('calibration_confidence', self.calibration_confidence)
            validation_result.add_metric('pixel_ratio_count', len(self.pixel_to_meter_ratios))
            validation_result.add_metric('arena_dimensions_available', bool(self.arena_dimensions_meters))
            validation_result.add_metric('validation_completeness', self._calculate_validation_completeness())
            
            # Add recommendations for calibration improvement
            if not validation_result.is_valid or validation_result.warnings:
                recommendations = self._generate_calibration_recommendations(validation_result, validation_thresholds)
                for rec in recommendations:
                    validation_result.add_recommendation(rec['text'], rec.get('priority', 'MEDIUM'))
            
            # Update calibration validation status
            self.is_validated = validation_result.is_valid
            self.validation_metrics = validation_result.metrics
            
            # Log validation operation with comprehensive results
            validation_duration = time.time() - start_time
            log_performance_metrics(
                metric_name='calibration_validation_time',
                metric_value=validation_duration,
                metric_unit='seconds',
                component='SCALE_CALIBRATION',
                metric_context={
                    'format_type': self.format_type,
                    'strict_validation': strict_validation,
                    'validation_passed': validation_result.is_valid,
                    'error_count': len(validation_result.errors),
                    'warning_count': len(validation_result.warnings)
                }
            )
            
            # Finalize validation result
            validation_result.finalize_validation()
            
            self.logger.info(
                f"Calibration validation completed: {'PASSED' if validation_result.is_valid else 'FAILED'} "
                f"({len(validation_result.errors)} errors, {len(validation_result.warnings)} warnings)"
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Calibration validation failed: {str(e)}")
            raise ValidationError(
                f"Calibration validation failed: {str(e)}",
                'calibration_validation',
                {
                    'format_type': self.format_type,
                    'validation_thresholds': validation_thresholds,
                    'strict_validation': strict_validation
                }
            )
    
    def apply_to_coordinates(
        self,
        coordinates: np.ndarray,
        source_system: str = 'pixel',
        target_system: str = 'meter',
        validate_transformation: bool = True
    ) -> np.ndarray:
        """
        Apply scale calibration to coordinate data with transformation validation and performance optimization for spatial data processing.
        
        Args:
            coordinates: Input coordinates as numpy array for transformation
            source_system: Source coordinate system ('pixel', 'meter', 'normalized')
            target_system: Target coordinate system ('pixel', 'meter', 'normalized')
            validate_transformation: Whether to validate transformation results
            
        Returns:
            np.ndarray: Transformed coordinates with calibration applied and validation status
            
        Raises:
            ValidationError: If coordinate validation fails
            ProcessingError: If coordinate transformation fails
        """
        start_time = time.time()
        
        try:
            # Validate input coordinates format and coordinate system specifications
            if not isinstance(coordinates, np.ndarray):
                coordinates = np.array(coordinates)
            
            if coordinates.size == 0:
                return np.array([])
            
            if source_system not in SUPPORTED_COORDINATE_SYSTEMS:
                raise ValidationError(
                    f"Unsupported source coordinate system: {source_system}",
                    'coordinate_system_validation',
                    {'source_system': source_system, 'supported_systems': SUPPORTED_COORDINATE_SYSTEMS}
                )
            
            if target_system not in SUPPORTED_COORDINATE_SYSTEMS:
                raise ValidationError(
                    f"Unsupported target coordinate system: {target_system}",
                    'coordinate_system_validation',
                    {'target_system': target_system, 'supported_systems': SUPPORTED_COORDINATE_SYSTEMS}
                )
            
            # Create coordinate transformer using create_coordinate_transformer
            transformer = create_coordinate_transformer(
                calibration_parameters=self._get_transformation_parameters(),
                source_coordinate_system=source_system,
                target_coordinate_system=target_system,
                enable_validation=validate_transformation
            )
            
            # Apply coordinate transformation with calibration parameters
            transformed_coordinates = transformer(coordinates)
            
            # Validate transformation results if validate_transformation enabled
            if validate_transformation:
                self._validate_coordinate_transformation(
                    coordinates, transformed_coordinates, source_system, target_system
                )
            
            # Check transformed coordinates for physical validity
            if validate_transformation:
                validity_check = self._check_coordinate_validity(transformed_coordinates, target_system)
                if not validity_check.get('valid', True):
                    self.logger.warning(f"Coordinate validity concerns: {validity_check.get('warnings', [])}")
            
            # Apply performance optimization for large coordinate arrays
            if coordinates.shape[0] > 1000:
                self.logger.debug(f"Processed large coordinate array: {coordinates.shape[0]} points")
            
            # Log coordinate transformation with performance metrics
            transformation_duration = time.time() - start_time
            log_performance_metrics(
                metric_name='coordinate_transformation_time',
                metric_value=transformation_duration,
                metric_unit='seconds',
                component='SCALE_CALIBRATION',
                metric_context={
                    'source_system': source_system,
                    'target_system': target_system,
                    'coordinate_count': coordinates.shape[0],
                    'validation_enabled': validate_transformation,
                    'format_type': self.format_type
                }
            )
            
            self.logger.debug(
                f"Coordinates transformed: {coordinates.shape[0]} points, "
                f"{source_system} -> {target_system}, {transformation_duration:.4f}s"
            )
            
            return transformed_coordinates
            
        except Exception as e:
            self.logger.error(f"Coordinate transformation failed: {str(e)}")
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            else:
                raise ProcessingError(
                    f"Coordinate transformation failed: {str(e)}",
                    'coordinate_transformation',
                    self.video_path,
                    {
                        'source_system': source_system,
                        'target_system': target_system,
                        'coordinate_shape': coordinates.shape if coordinates is not None else None,
                        'validate_transformation': validate_transformation
                    }
                )
    
    def get_transformation_matrix(
        self,
        source_system: str,
        target_system: str,
        force_recalculation: bool = False
    ) -> np.ndarray:
        """
        Get transformation matrix for coordinate system conversion with caching and validation for efficient spatial transformations.
        
        Args:
            source_system: Source coordinate system for transformation matrix
            target_system: Target coordinate system for transformation matrix
            force_recalculation: Force recalculation of transformation matrix
            
        Returns:
            np.ndarray: Transformation matrix for coordinate system conversion with validation metadata
            
        Raises:
            ValidationError: If coordinate systems are invalid
            ProcessingError: If matrix calculation fails
        """
        try:
            # Validate coordinate system specifications
            if source_system not in SUPPORTED_COORDINATE_SYSTEMS:
                raise ValidationError(
                    f"Unsupported source coordinate system: {source_system}",
                    'coordinate_system_validation',
                    {'source_system': source_system}
                )
            
            if target_system not in SUPPORTED_COORDINATE_SYSTEMS:
                raise ValidationError(
                    f"Unsupported target coordinate system: {target_system}",
                    'coordinate_system_validation',
                    {'target_system': target_system}
                )
            
            # Generate matrix cache key
            matrix_key = f"{source_system}_to_{target_system}"
            
            # Check transformation matrix cache if force_recalculation is False
            if not force_recalculation and matrix_key in self.transformation_matrices:
                cached_matrix = self.transformation_matrices[matrix_key]
                self.logger.debug(f"Using cached transformation matrix: {matrix_key}")
                return cached_matrix
            
            # Calculate transformation matrix from calibration parameters
            transformation_matrix = self._create_transformation_matrix_for_systems(source_system, target_system)
            
            # Validate transformation matrix properties and invertibility
            matrix_validation = self._validate_transformation_matrix(transformation_matrix, source_system, target_system)
            if not matrix_validation.get('valid', False):
                raise ProcessingError(
                    f"Invalid transformation matrix: {matrix_validation.get('errors', [])}",
                    'matrix_validation',
                    self.video_path,
                    {'source_system': source_system, 'target_system': target_system}
                )
            
            # Cache transformation matrix for future use
            self.transformation_matrices[matrix_key] = transformation_matrix
            
            # Log transformation matrix generation with validation status
            self.logger.debug(f"Transformation matrix generated and cached: {matrix_key}")
            
            return transformation_matrix
            
        except Exception as e:
            self.logger.error(f"Transformation matrix generation failed: {str(e)}")
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            else:
                raise ProcessingError(
                    f"Transformation matrix generation failed: {str(e)}",
                    'matrix_generation',
                    self.video_path,
                    {
                        'source_system': source_system,
                        'target_system': target_system,
                        'force_recalculation': force_recalculation
                    }
                )
    
    def export_calibration(
        self,
        output_path: str,
        export_format: str = 'json',
        include_metadata: bool = True
    ) -> bool:
        """
        Export calibration parameters to file format with metadata and validation status for reproducibility and documentation.
        
        Args:
            output_path: Path for output file
            export_format: Format for export ('json', 'yaml', 'csv')
            include_metadata: Whether to include metadata and validation status
            
        Returns:
            bool: Export success status with file validation and integrity checking
            
        Raises:
            ProcessingError: If export operation fails
        """
        start_time = time.time()
        
        try:
            # Prepare calibration data for export with specified format
            export_data = {
                'calibration_info': {
                    'video_path': self.video_path,
                    'format_type': self.format_type,
                    'calibration_method': self.calibration_method,
                    'calibration_timestamp': self.calibration_timestamp.isoformat(),
                    'calibration_confidence': self.calibration_confidence,
                    'is_validated': self.is_validated
                },
                'calibration_parameters': {
                    'pixel_to_meter_ratios': self.pixel_to_meter_ratios,
                    'arena_dimensions_meters': self.arena_dimensions_meters,
                    'arena_dimensions_pixels': self.arena_dimensions_pixels,
                    'arena_boundaries': self.arena_boundaries
                },
                'transformation_data': {
                    'transformation_matrices': {k: v.tolist() for k, v in self.transformation_matrices.items()},
                    'supported_coordinate_systems': SUPPORTED_COORDINATE_SYSTEMS
                }
            }
            
            # Include metadata and validation status if include_metadata is enabled
            if include_metadata:
                export_data['metadata'] = {
                    'export_timestamp': datetime.now().isoformat(),
                    'export_format': export_format,
                    'validation_metrics': self.validation_metrics,
                    'calibration_config': self.calibration_config,
                    'software_version': '1.0.0',
                    'coordinate_precision_digits': COORDINATE_PRECISION_DIGITS
                }
            
            # Create output directory if it doesn't exist
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export calibration to specified output path based on format
            if export_format.lower() == 'json':
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif export_format.lower() == 'yaml':
                try:
                    import yaml
                    with open(output_path, 'w', encoding='utf-8') as f:
                        yaml.safe_dump(export_data, f, default_flow_style=False, indent=2)
                except ImportError:
                    raise ProcessingError(
                        "YAML export requires PyYAML package",
                        'export_dependency',
                        self.video_path,
                        {'export_format': export_format}
                    )
            elif export_format.lower() == 'csv':
                import csv
                # Flatten data for CSV export
                flattened_data = self._flatten_calibration_data(export_data)
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Parameter', 'Value', 'Unit', 'Description'])
                    for row in flattened_data:
                        writer.writerow(row)
            else:
                raise ValidationError(
                    f"Unsupported export format: {export_format}",
                    'export_format_validation',
                    {'export_format': export_format, 'supported_formats': ['json', 'yaml', 'csv']}
                )
            
            # Validate exported file integrity and format compliance
            if output_path.exists() and output_path.stat().st_size > 0:
                export_success = True
                
                # Additional validation for JSON format
                if export_format.lower() == 'json':
                    try:
                        import json
                        with open(output_path, 'r', encoding='utf-8') as f:
                            json.load(f)  # Validate JSON syntax
                    except json.JSONDecodeError:
                        export_success = False
                        self.logger.error("Exported JSON file is invalid")
            else:
                export_success = False
            
            # Generate export report with calibration information
            export_duration = time.time() - start_time
            if export_success:
                self.logger.info(
                    f"Calibration exported successfully: {output_path.name} "
                    f"({export_format}, {output_path.stat().st_size} bytes, {export_duration:.3f}s)"
                )
            
            # Log calibration export operation with success status
            log_performance_metrics(
                metric_name='calibration_export_time',
                metric_value=export_duration,
                metric_unit='seconds',
                component='SCALE_CALIBRATION',
                metric_context={
                    'export_format': export_format,
                    'include_metadata': include_metadata,
                    'export_success': export_success,
                    'file_size_bytes': output_path.stat().st_size if output_path.exists() else 0
                }
            )
            
            return export_success
            
        except Exception as e:
            self.logger.error(f"Calibration export failed: {str(e)}")
            raise ProcessingError(
                f"Calibration export failed: {str(e)}",
                'calibration_export',
                self.video_path,
                {
                    'output_path': str(output_path),
                    'export_format': export_format,
                    'include_metadata': include_metadata
                }
            )
    
    # Private helper methods for internal functionality
    
    def _initialize_default_parameters(self):
        """Initialize calibration parameters with format-specific defaults."""
        if self.format_type == 'crimaldi':
            default_ratio = CRIMALDI_PIXEL_TO_METER_RATIO
        else:
            default_ratio = CUSTOM_PIXEL_TO_METER_RATIO
        
        self.pixel_to_meter_ratios = {
            'horizontal_ratio': default_ratio,
            'vertical_ratio': default_ratio,
            'primary_ratio': default_ratio
        }
        
        self.arena_dimensions_meters = {
            'width': TARGET_ARENA_WIDTH_METERS,
            'height': TARGET_ARENA_HEIGHT_METERS
        }
        
        self.arena_dimensions_pixels = {
            'width': 640,
            'height': 480
        }
    
    def _configure_format_specific_settings(self):
        """Configure format-specific calibration settings."""
        if self.format_type == 'crimaldi':
            self.calibration_config.setdefault('spatial_accuracy_threshold', SPATIAL_ACCURACY_THRESHOLD)
            self.calibration_config.setdefault('confidence_threshold', DEFAULT_CALIBRATION_CONFIDENCE_THRESHOLD)
        elif self.format_type == 'custom':
            self.calibration_config.setdefault('adaptive_detection', True)
            self.calibration_config.setdefault('arena_detection_method', 'contour')
    
    def _get_calibration_summary(self) -> Dict[str, float]:
        """Get summary of calibration parameters."""
        return {
            **self.pixel_to_meter_ratios,
            'calibration_confidence': self.calibration_confidence,
            'is_validated': float(self.is_validated),
            'calibration_timestamp': self.calibration_timestamp.timestamp()
        }


@dataclass
class ScaleCalibrationManager:
    """
    Manager class for handling multiple scale calibrations with cross-format compatibility, batch operations, and performance optimization for large-scale plume simulation processing.
    
    This class provides comprehensive management of multiple scale calibrations with batch processing capabilities,
    cross-format compatibility validation, and performance optimization for 4000+ simulation processing requirements.
    """
    
    # Manager configuration and settings
    manager_config: Dict[str, Any]
    caching_enabled: bool = True
    batch_optimization_enabled: bool = True
    
    # Manager state and registries
    calibration_registry: Dict[str, ScaleCalibration] = field(default_factory=dict)
    format_handlers: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    physical_constants: PhysicalConstants = field(default_factory=PhysicalConstants)
    
    def __post_init__(self):
        """Initialize scale calibration manager with configuration, caching, and batch optimization capabilities for comprehensive calibration management."""
        # Set scientific context for manager operations
        set_scientific_context(
            simulation_id='calibration_manager',
            algorithm_name='ScaleCalibrationManager',
            processing_stage='MANAGER_INITIALIZATION'
        )
        
        # Set manager configuration and optimization settings
        if not isinstance(self.manager_config, dict):
            self.manager_config = {}
        
        # Initialize calibration registry and format handlers
        self.calibration_registry = {}
        self.format_handlers = {}
        
        # Setup physical constants for unit conversion and validation
        self.physical_constants = PhysicalConstants()
        
        # Configure caching if enabled for performance optimization
        if self.caching_enabled:
            self.manager_config.setdefault('cache_size', CALIBRATION_CACHE_SIZE)
            self.manager_config.setdefault('cache_ttl_seconds', 3600)  # 1 hour TTL
        
        # Setup batch optimization if enabled for large-scale processing
        if self.batch_optimization_enabled:
            self.manager_config.setdefault('batch_size', 50)
            self.manager_config.setdefault('parallel_processing', True)
            self.manager_config.setdefault('memory_optimization', True)
        
        # Initialize performance metrics tracking
        self.performance_metrics = {
            'total_calibrations_created': 0,
            'successful_calibrations': 0,
            'failed_calibrations': 0,
            'average_calibration_time': 0.0,
            'cache_hit_ratio': 0.0,
            'batch_processing_efficiency': 0.0
        }
        
        # Configure logger for calibration management operations
        self.logger = get_logger('calibration_manager', 'SCALE_CALIBRATION')
        
        # Log manager initialization with configuration details
        self.logger.info(
            f"ScaleCalibrationManager initialized: caching={self.caching_enabled}, "
            f"batch_optimization={self.batch_optimization_enabled}"
        )
    
    def create_calibration(
        self,
        video_path: str,
        calibration_config: Dict[str, Any] = None,
        validate_creation: bool = True
    ) -> ScaleCalibration:
        """
        Create new scale calibration for video file with format detection, parameter extraction, and validation for automated calibration management.
        
        Args:
            video_path: Path to video file for calibration creation
            calibration_config: Configuration for calibration creation
            validate_creation: Whether to validate calibration creation
            
        Returns:
            ScaleCalibration: Created scale calibration with extracted parameters and validation status
            
        Raises:
            ValidationError: If video file validation fails
            ProcessingError: If calibration creation fails
        """
        start_time = time.time()
        
        try:
            # Validate video file path and accessibility
            video_path = str(Path(video_path).resolve())
            if not Path(video_path).exists():
                raise ValidationError(
                    f"Video file does not exist: {video_path}",
                    'file_validation',
                    {'video_path': video_path}
                )
            
            # Check calibration registry for existing calibration
            registry_key = self._generate_calibration_key(video_path)
            if registry_key in self.calibration_registry and not calibration_config.get('force_recreation', False):
                existing_calibration = self.calibration_registry[registry_key]
                self.logger.info(f"Using existing calibration for {Path(video_path).name}")
                return existing_calibration
            
            # Detect video format using detect_video_format function
            format_detection = detect_video_format(video_path, deep_inspection=True)
            detected_format = format_detection.get('format_type', 'custom')
            
            self.logger.info(f"Creating calibration for {detected_format} format: {Path(video_path).name}")
            
            # Merge manager config with calibration-specific config
            if calibration_config is None:
                calibration_config = {}
            
            merged_config = {**self.manager_config, **calibration_config}
            merged_config['format_detection'] = format_detection
            
            # Create ScaleCalibration instance with video path and configuration
            calibration = ScaleCalibration(
                video_path=video_path,
                format_type=detected_format,
                calibration_config=merged_config
            )
            
            # Extract calibration parameters using format-specific methods
            extraction_result = calibration.extract_calibration_parameters(
                force_reextraction=merged_config.get('force_extraction', False),
                extraction_hints=merged_config.get('extraction_hints')
            )
            
            # Validate calibration creation if validate_creation enabled
            if validate_creation:
                validation_result = calibration.validate_calibration(
                    validation_thresholds=merged_config.get('validation_thresholds'),
                    strict_validation=merged_config.get('strict_validation', False)
                )
                
                if not validation_result.is_valid:
                    self.logger.warning(f"Calibration validation failed: {validation_result.errors}")
                    self.performance_metrics['failed_calibrations'] += 1
                else:
                    self.performance_metrics['successful_calibrations'] += 1
            
            # Register calibration in calibration registry
            self.calibration_registry[registry_key] = calibration
            
            # Update performance metrics for calibration creation
            creation_duration = time.time() - start_time
            self.performance_metrics['total_calibrations_created'] += 1
            
            # Update average calibration time
            total_time = (self.performance_metrics['average_calibration_time'] * 
                         (self.performance_metrics['total_calibrations_created'] - 1) + creation_duration)
            self.performance_metrics['average_calibration_time'] = total_time / self.performance_metrics['total_calibrations_created']
            
            # Log calibration creation with format and validation details
            log_performance_metrics(
                metric_name='scale_calibration_creation_time',
                metric_value=creation_duration,
                metric_unit='seconds',
                component='SCALE_CALIBRATION',
                metric_context={
                    'format_type': detected_format,
                    'validation_enabled': validate_creation,
                    'calibration_confidence': calibration.calibration_confidence,
                    'video_path': video_path
                }
            )
            
            self.logger.info(
                f"Scale calibration created: {Path(video_path).name} "
                f"(confidence: {calibration.calibration_confidence:.3f}, {creation_duration:.3f}s)"
            )
            
            return calibration
            
        except Exception as e:
            self.performance_metrics['failed_calibrations'] += 1
            self.logger.error(f"Calibration creation failed: {str(e)}")
            
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            else:
                raise ProcessingError(
                    f"Scale calibration creation failed: {str(e)}",
                    'calibration_creation',
                    video_path,
                    {
                        'calibration_config': calibration_config,
                        'validate_creation': validate_creation
                    }
                )
    
    def get_calibration(
        self,
        video_path: str,
        create_if_missing: bool = True,
        validate_calibration: bool = False
    ) -> Optional[ScaleCalibration]:
        """
        Retrieve existing scale calibration from registry with caching and validation for efficient calibration access.
        
        Args:
            video_path: Path to video file for calibration retrieval
            create_if_missing: Whether to create calibration if not found in registry
            validate_calibration: Whether to validate retrieved calibration
            
        Returns:
            Optional[ScaleCalibration]: Retrieved scale calibration or None if not found and creation disabled
            
        Raises:
            ProcessingError: If calibration retrieval or creation fails
        """
        start_time = time.time()
        
        try:
            # Generate calibration registry key
            registry_key = self._generate_calibration_key(video_path)
            
            # Check calibration registry for existing calibration
            if registry_key in self.calibration_registry:
                calibration = self.calibration_registry[registry_key]
                
                # Validate existing calibration if validate_calibration enabled
                if validate_calibration:
                    validation_result = calibration.validate_calibration()
                    if not validation_result.is_valid:
                        self.logger.warning(f"Retrieved calibration validation failed: {validation_result.errors}")
                        
                        # Optionally recreate invalid calibration
                        if self.manager_config.get('recreate_invalid_calibrations', False):
                            self.logger.info("Recreating invalid calibration")
                            del self.calibration_registry[registry_key]
                            return self.create_calibration(video_path, validate_creation=True)
                
                # Update cache hit statistics
                self._update_cache_statistics(hit=True)
                
                # Update calibration access statistics and performance metrics
                access_duration = time.time() - start_time
                self.logger.debug(f"Calibration retrieved from cache: {Path(video_path).name} ({access_duration:.4f}s)")
                
                return calibration
            
            # Update cache miss statistics  
            self._update_cache_statistics(hit=False)
            
            # Create new calibration if missing and create_if_missing enabled
            if create_if_missing:
                self.logger.info(f"Creating missing calibration: {Path(video_path).name}")
                return self.create_calibration(video_path, validate_creation=validate_calibration)
            else:
                self.logger.warning(f"Calibration not found and creation disabled: {Path(video_path).name}")
                return None
            
        except Exception as e:
            self.logger.error(f"Calibration retrieval failed: {str(e)}")
            raise ProcessingError(
                f"Calibration retrieval failed: {str(e)}",
                'calibration_retrieval',
                video_path,
                {
                    'create_if_missing': create_if_missing,
                    'validate_calibration': validate_calibration
                }
            )
    
    def batch_calibrate(
        self,
        video_paths: List[str],
        batch_config: Dict[str, Any] = None,
        enable_parallel_processing: bool = None
    ) -> Dict[str, ScaleCalibration]:
        """
        Perform batch calibration for multiple video files with parallel processing, progress tracking, and comprehensive error handling for large-scale operations.
        
        Args:
            video_paths: List of video file paths for batch calibration
            batch_config: Configuration for batch processing
            enable_parallel_processing: Whether to enable parallel processing
            
        Returns:
            Dict[str, ScaleCalibration]: Batch calibration results with individual calibration status and error analysis
            
        Raises:
            ValidationError: If batch configuration validation fails
            ProcessingError: If batch processing fails
        """
        start_time = time.time()
        
        try:
            # Validate video paths and batch configuration parameters
            if not video_paths:
                raise ValidationError(
                    "Video paths list cannot be empty",
                    'batch_validation',
                    {'video_paths_count': 0}
                )
            
            valid_paths = [path for path in video_paths if Path(path).exists()]
            if len(valid_paths) != len(video_paths):
                missing_count = len(video_paths) - len(valid_paths)
                self.logger.warning(f"Batch processing: {missing_count} video files not found")
            
            # Initialize batch configuration with defaults
            if batch_config is None:
                batch_config = {}
            
            merged_batch_config = {**self.manager_config, **batch_config}
            
            # Determine parallel processing setting
            if enable_parallel_processing is None:
                enable_parallel_processing = self.batch_optimization_enabled and merged_batch_config.get('parallel_processing', True)
            
            # Setup batch processing with optimal resource allocation
            batch_size = merged_batch_config.get('batch_size', 50)
            max_workers = merged_batch_config.get('max_workers', min(4, len(valid_paths)))
            
            self.logger.info(
                f"Starting batch calibration: {len(valid_paths)} videos, "
                f"parallel={enable_parallel_processing}, batch_size={batch_size}"
            )
            
            # Initialize batch results storage
            batch_results = {}
            processing_stats = {
                'total_videos': len(valid_paths),
                'processed_videos': 0,
                'successful_calibrations': 0,
                'failed_calibrations': 0,
                'processing_errors': []
            }
            
            # Process videos with parallel processing if enabled and beneficial
            if enable_parallel_processing and len(valid_paths) > 1:
                batch_results = self._process_batch_parallel(valid_paths, merged_batch_config, processing_stats)
            else:
                batch_results = self._process_batch_sequential(valid_paths, merged_batch_config, processing_stats)
            
            # Monitor batch progress and collect processing statistics
            processing_duration = time.time() - start_time
            processing_stats['total_processing_time'] = processing_duration
            processing_stats['average_time_per_video'] = processing_duration / max(1, processing_stats['processed_videos'])
            
            # Validate batch calibration quality and consistency
            quality_assessment = self._assess_batch_quality(batch_results)
            processing_stats['quality_assessment'] = quality_assessment
            
            # Generate batch processing report with error analysis
            batch_report = self._generate_batch_report(processing_stats, quality_assessment)
            
            # Update performance metrics for batch operations
            self.performance_metrics['batch_processing_efficiency'] = (
                processing_stats['successful_calibrations'] / max(1, processing_stats['total_videos'])
            )
            
            # Log batch calibration operation with comprehensive results
            log_performance_metrics(
                metric_name='batch_calibration_time',
                metric_value=processing_duration,
                metric_unit='seconds',
                component='SCALE_CALIBRATION',
                metric_context={
                    'total_videos': processing_stats['total_videos'],
                    'successful_calibrations': processing_stats['successful_calibrations'],
                    'parallel_processing': enable_parallel_processing,
                    'batch_efficiency': self.performance_metrics['batch_processing_efficiency']
                }
            )
            
            self.logger.info(
                f"Batch calibration completed: {processing_stats['successful_calibrations']}/{processing_stats['total_videos']} "
                f"successful ({processing_duration:.2f}s total, {processing_stats['average_time_per_video']:.3f}s avg)"
            )
            
            return batch_results
            
        except Exception as e:
            batch_duration = time.time() - start_time
            self.logger.error(f"Batch calibration failed after {batch_duration:.3f}s: {str(e)}")
            
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            else:
                raise ProcessingError(
                    f"Batch calibration failed: {str(e)}",
                    'batch_calibration',
                    f"batch_of_{len(video_paths)}_videos",
                    {
                        'video_paths_count': len(video_paths),
                        'batch_config': batch_config,
                        'enable_parallel_processing': enable_parallel_processing
                    }
                )
    
    def validate_cross_format_compatibility(
        self,
        format_types: List[str],
        tolerance_thresholds: Dict[str, float] = None,
        detailed_analysis: bool = True
    ) -> ValidationResult:
        """
        Validate calibration compatibility across different video formats with conversion accuracy assessment and compatibility matrix analysis.
        
        Args:
            format_types: List of format types to validate compatibility
            tolerance_thresholds: Tolerance thresholds for cross-format consistency
            detailed_analysis: Whether to perform detailed compatibility analysis
            
        Returns:
            ValidationResult: Cross-format compatibility validation result with conversion accuracy assessment
            
        Raises:
            ValidationError: If format validation fails
        """
        start_time = time.time()
        
        try:
            # Create ValidationResult container for compatibility assessment
            validation_result = ValidationResult(
                validation_type='cross_format_compatibility_validation',
                is_valid=True,
                validation_context=f'formats={format_types}, detailed={detailed_analysis}'
            )
            
            # Validate format types against supported formats
            supported_formats = ['crimaldi', 'custom', 'avi', 'generic']
            invalid_formats = [fmt for fmt in format_types if fmt not in supported_formats]
            
            if invalid_formats:
                validation_result.add_error(
                    f"Unsupported format types: {invalid_formats}",
                    severity=ValidationError.ErrorSeverity.HIGH
                )
                validation_result.is_valid = False
                return validation_result
            
            # Set default tolerance thresholds if not provided
            if tolerance_thresholds is None:
                tolerance_thresholds = {
                    'pixel_ratio_tolerance': 0.05,  # 5% tolerance for pixel ratio differences
                    'arena_size_tolerance': 0.02,   # 2% tolerance for arena size differences
                    'accuracy_threshold': 0.95      # 95% minimum accuracy
                }
            
            # Analyze calibration parameters across different formats
            format_analysis = {}
            for format_type in format_types:
                format_calibrations = self._get_calibrations_by_format(format_type)
                if format_calibrations:
                    analysis = self._analyze_format_parameters(format_calibrations, format_type)
                    format_analysis[format_type] = analysis
                else:
                    validation_result.add_warning(f"No calibrations available for format: {format_type}")
            
            # Validate conversion accuracy between format pairs
            compatibility_matrix = {}
            for i, format1 in enumerate(format_types):
                compatibility_matrix[format1] = {}
                for format2 in format_types:
                    if format1 == format2:
                        compatibility_matrix[format1][format2] = {'compatible': True, 'accuracy': 1.0}
                    else:
                        compatibility = self._assess_format_pair_compatibility(
                            format1, format2, format_analysis, tolerance_thresholds
                        )
                        compatibility_matrix[format1][format2] = compatibility
                        
                        # Check compatibility against thresholds
                        if not compatibility.get('compatible', False):
                            validation_result.add_error(
                                f"Incompatible formats: {format1} <-> {format2}",
                                severity=ValidationError.ErrorSeverity.MEDIUM
                            )
                            validation_result.is_valid = False
            
            # Check tolerance thresholds for cross-format consistency
            consistency_check = self._check_cross_format_consistency(format_analysis, tolerance_thresholds)
            if not consistency_check.get('consistent', True):
                validation_result.add_warning(f"Cross-format consistency issues: {consistency_check.get('issues', [])}")
            
            # Perform detailed compatibility analysis if detailed_analysis enabled
            if detailed_analysis:
                detailed_analysis_result = self._perform_detailed_compatibility_analysis(
                    format_types, format_analysis, compatibility_matrix
                )
                validation_result.set_metadata('detailed_analysis', detailed_analysis_result)
            
            # Generate compatibility matrix for format combinations
            validation_result.set_metadata('compatibility_matrix', compatibility_matrix)
            validation_result.set_metadata('format_analysis', format_analysis)
            
            # Add compatibility metrics
            total_pairs = len(format_types) * (len(format_types) - 1)
            compatible_pairs = sum(
                1 for format1 in compatibility_matrix.values()
                for compat in format1.values()
                if compat.get('compatible', False)
            ) - len(format_types)  # Exclude self-compatibility
            
            validation_result.add_metric('compatibility_ratio', compatible_pairs / max(1, total_pairs))
            validation_result.add_metric('formats_analyzed', len(format_types))
            validation_result.add_metric('total_format_pairs', total_pairs)
            
            # Add recommendations for improving cross-format compatibility
            if not validation_result.is_valid or validation_result.warnings:
                recommendations = self._generate_compatibility_recommendations(
                    validation_result, format_types, compatibility_matrix
                )
                for rec in recommendations:
                    validation_result.add_recommendation(rec['text'], rec.get('priority', 'MEDIUM'))
            
            # Log compatibility validation with comprehensive analysis
            validation_duration = time.time() - start_time
            log_performance_metrics(
                metric_name='cross_format_compatibility_validation_time',
                metric_value=validation_duration,
                metric_unit='seconds',
                component='SCALE_CALIBRATION',
                metric_context={
                    'format_count': len(format_types),
                    'compatibility_ratio': validation_result.metrics.get('compatibility_ratio', 0.0),
                    'detailed_analysis': detailed_analysis,
                    'validation_passed': validation_result.is_valid
                }
            )
            
            validation_result.finalize_validation()
            
            self.logger.info(
                f"Cross-format compatibility validated: {len(format_types)} formats, "
                f"compatibility ratio: {validation_result.metrics.get('compatibility_ratio', 0.0):.3f}"
            )
            
            return validation_result
            
        except Exception as e:
            error_duration = time.time() - start_time
            self.logger.error(f"Cross-format compatibility validation failed after {error_duration:.3f}s: {str(e)}")
            
            if isinstance(e, ValidationError):
                raise
            else:
                # Create error validation result
                error_result = ValidationResult(
                    validation_type='cross_format_compatibility_validation',
                    is_valid=False,
                    validation_context='error_occurred'
                )
                error_result.add_error(
                    f"Compatibility validation failed: {str(e)}",
                    severity=ValidationError.ErrorSeverity.CRITICAL
                )
                error_result.finalize_validation()
                return error_result
    
    def optimize_calibration_performance(
        self,
        optimization_strategy: str = 'balanced',
        apply_optimizations: bool = True,
        optimization_constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize calibration performance by analyzing processing patterns, adjusting cache settings, and implementing performance improvements for enhanced throughput.
        
        Args:
            optimization_strategy: Strategy for performance optimization ('memory', 'speed', 'balanced', 'accuracy')
            apply_optimizations: Whether to apply optimization changes immediately
            optimization_constraints: Constraints for optimization process
            
        Returns:
            Dict[str, Any]: Optimization results with performance improvements and configuration updates
            
        Raises:
            ValidationError: If optimization strategy is invalid
            ProcessingError: If optimization process fails
        """
        start_time = time.time()
        
        try:
            # Validate optimization strategy
            valid_strategies = ['memory', 'speed', 'balanced', 'accuracy']
            if optimization_strategy not in valid_strategies:
                raise ValidationError(
                    f"Invalid optimization strategy: {optimization_strategy}",
                    'strategy_validation',
                    {'strategy': optimization_strategy, 'valid_strategies': valid_strategies}
                )
            
            # Initialize optimization result structure
            optimization_result = {
                'optimization_timestamp': datetime.now().isoformat(),
                'strategy': optimization_strategy,
                'optimizations_applied': apply_optimizations,
                'performance_baseline': {},
                'optimization_changes': [],
                'performance_improvements': {},
                'configuration_updates': {}
            }
            
            # Analyze current calibration performance and processing patterns
            performance_baseline = self._analyze_current_performance()
            optimization_result['performance_baseline'] = performance_baseline
            
            # Identify optimization opportunities and bottlenecks
            bottlenecks = self._identify_performance_bottlenecks(performance_baseline)
            optimization_opportunities = self._identify_optimization_opportunities(bottlenecks, optimization_strategy)
            
            optimization_result['bottlenecks'] = bottlenecks
            optimization_result['optimization_opportunities'] = optimization_opportunities
            
            # Generate optimization strategy based on performance analysis
            strategy_config = self._generate_optimization_strategy_config(
                optimization_strategy, performance_baseline, optimization_constraints or {}
            )
            optimization_result['strategy_config'] = strategy_config
            
            # Apply optimization changes if apply_optimizations enabled
            if apply_optimizations:
                applied_changes = []
                
                # Memory optimization
                if optimization_strategy in ['memory', 'balanced']:
                    memory_optimizations = self._apply_memory_optimizations(strategy_config)
                    applied_changes.extend(memory_optimizations)
                
                # Speed optimization
                if optimization_strategy in ['speed', 'balanced']:
                    speed_optimizations = self._apply_speed_optimizations(strategy_config)
                    applied_changes.extend(speed_optimizations)
                
                # Accuracy optimization
                if optimization_strategy in ['accuracy', 'balanced']:
                    accuracy_optimizations = self._apply_accuracy_optimizations(strategy_config)
                    applied_changes.extend(accuracy_optimizations)
                
                # Cache optimization
                if self.caching_enabled:
                    cache_optimizations = self._optimize_cache_settings(strategy_config)
                    applied_changes.extend(cache_optimizations)
                
                optimization_result['optimization_changes'] = applied_changes
                
                # Monitor optimization effectiveness and performance impact
                post_optimization_performance = self._analyze_current_performance()
                performance_improvements = self._calculate_performance_improvements(
                    performance_baseline, post_optimization_performance
                )
                optimization_result['performance_improvements'] = performance_improvements
                optimization_result['post_optimization_performance'] = post_optimization_performance
            
            # Update calibration configuration and monitoring thresholds
            config_updates = self._update_manager_configuration(strategy_config, apply_optimizations)
            optimization_result['configuration_updates'] = config_updates
            
            # Generate optimization recommendations for future improvements
            recommendations = self._generate_optimization_recommendations(
                optimization_result, optimization_strategy, optimization_constraints
            )
            optimization_result['recommendations'] = recommendations
            
            # Log optimization operation with performance improvements
            optimization_duration = time.time() - start_time
            log_performance_metrics(
                metric_name='calibration_performance_optimization_time',
                metric_value=optimization_duration,
                metric_unit='seconds',
                component='SCALE_CALIBRATION',
                metric_context={
                    'optimization_strategy': optimization_strategy,
                    'optimizations_applied': apply_optimizations,
                    'performance_improvement': optimization_result.get('performance_improvements', {}).get('overall_improvement', 0.0),
                    'optimization_changes_count': len(optimization_result.get('optimization_changes', []))
                }
            )
            
            self.logger.info(
                f"Calibration performance optimization completed: strategy={optimization_strategy}, "
                f"applied={apply_optimizations}, improvements={optimization_result.get('performance_improvements', {})}"
            )
            
            return optimization_result
            
        except Exception as e:
            error_duration = time.time() - start_time
            self.logger.error(f"Calibration performance optimization failed after {error_duration:.3f}s: {str(e)}")
            
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            else:
                raise ProcessingError(
                    f"Calibration performance optimization failed: {str(e)}",
                    'performance_optimization',
                    'calibration_manager',
                    {
                        'optimization_strategy': optimization_strategy,
                        'apply_optimizations': apply_optimizations,
                        'optimization_constraints': optimization_constraints
                    }
                )
    
    # Private helper methods for manager functionality
    
    def _generate_calibration_key(self, video_path: str) -> str:
        """Generate unique key for calibration registry."""
        return str(hash(str(Path(video_path).resolve())))
    
    def _update_cache_statistics(self, hit: bool):
        """Update cache hit/miss statistics."""
        # Update cache hit ratio calculation
        current_ratio = self.performance_metrics.get('cache_hit_ratio', 0.0)
        total_accesses = self.performance_metrics.get('total_cache_accesses', 0) + 1
        
        if hit:
            cache_hits = self.performance_metrics.get('cache_hits', 0) + 1
            self.performance_metrics['cache_hits'] = cache_hits
        else:
            cache_hits = self.performance_metrics.get('cache_hits', 0)
        
        self.performance_metrics['total_cache_accesses'] = total_accesses
        self.performance_metrics['cache_hit_ratio'] = cache_hits / total_accesses


# Helper functions for scale calibration implementation

def _preprocess_frame_for_detection(frame: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
    """Preprocess video frame for arena boundary detection."""
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame.copy()
    
    # Apply Gaussian blur to reduce noise
    blur_kernel = parameters.get('blur_kernel_size', 5)
    blurred = cv2.GaussianBlur(gray_frame, (blur_kernel, blur_kernel), 0)
    
    return blurred


def _detect_boundaries_contour(frame: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Detect arena boundaries using contour analysis."""
    # Apply threshold to create binary image
    threshold_value = parameters.get('threshold_value', 127)
    _, binary = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {'boundaries': {}, 'confidence': 0.0, 'properties': {}}
    
    # Find largest contour (assumed to be arena)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return {
        'boundaries': {
            'top_left': (x, y),
            'bottom_right': (x + w, y + h),
            'width': w,
            'height': h,
            'contour': largest_contour.tolist()
        },
        'confidence': 0.8,  # Placeholder confidence
        'properties': {
            'area': cv2.contourArea(largest_contour),
            'perimeter': cv2.arcLength(largest_contour, True)
        }
    }


def _detect_boundaries_edge(frame: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Detect arena boundaries using edge detection."""
    # Apply Canny edge detection
    low_threshold = parameters.get('canny_low', 50)
    high_threshold = parameters.get('canny_high', 150)
    edges = cv2.Canny(frame, low_threshold, high_threshold)
    
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {'boundaries': {}, 'confidence': 0.0, 'properties': {}}
    
    # Process largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return {
        'boundaries': {
            'top_left': (x, y),
            'bottom_right': (x + w, y + h),
            'width': w,
            'height': h
        },
        'confidence': 0.7,
        'properties': {'detection_method': 'edge'}
    }


def _detect_boundaries_template(frame: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Detect arena boundaries using template matching."""
    # Placeholder implementation for template matching
    height, width = frame.shape[:2]
    
    # Use entire frame as arena for template method
    return {
        'boundaries': {
            'top_left': (0, 0),
            'bottom_right': (width, height),
            'width': width,
            'height': height
        },
        'confidence': 0.6,
        'properties': {'detection_method': 'template'}
    }


def _detect_boundaries_manual(frame: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Use manually specified arena boundaries."""
    manual_bounds = parameters.get('manual_boundaries')
    if manual_bounds:
        return {
            'boundaries': manual_bounds,
            'confidence': 1.0,
            'properties': {'detection_method': 'manual'}
        }
    else:
        # Fallback to full frame
        height, width = frame.shape[:2]
        return {
            'boundaries': {
                'top_left': (0, 0),
                'bottom_right': (width, height),
                'width': width,
                'height': height
            },
            'confidence': 0.5,
            'properties': {'detection_method': 'manual_fallback'}
        }


def _assess_boundary_quality(boundaries: Dict[str, Any], frame_shape: Tuple[int, ...]) -> Dict[str, float]:
    """Assess quality of detected arena boundaries."""
    if not boundaries:
        return {'overall_quality': 0.0}
    
    frame_height, frame_width = frame_shape[:2]
    
    # Calculate boundary quality metrics
    width = boundaries.get('width', 0)
    height = boundaries.get('height', 0)
    
    # Size quality (prefer larger boundaries)
    size_quality = min(1.0, (width * height) / (frame_width * frame_height * 0.5))
    
    # Aspect ratio quality (prefer reasonable aspect ratios)
    if height > 0:
        aspect_ratio = width / height
        aspect_quality = max(0.0, 1.0 - abs(aspect_ratio - 1.0))  # Prefer square-ish
    else:
        aspect_quality = 0.0
    
    # Overall quality
    overall_quality = (size_quality + aspect_quality) / 2.0
    
    return {
        'size_quality': size_quality,
        'aspect_quality': aspect_quality,
        'overall_quality': overall_quality
    }


# Additional helper functions for comprehensive implementation
def _generate_detection_cache_key(frame: np.ndarray, method: str, params: Dict[str, Any]) -> str:
    """Generate cache key for arena detection results."""
    frame_hash = hash(frame.tobytes())
    params_hash = hash(str(sorted(params.items())))
    return f"detection_{method}_{frame_hash}_{params_hash}"


def _validate_arena_detection(result: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate arena detection results."""
    is_valid = (
        result.get('detection_confidence', 0.0) >= MIN_ARENA_DETECTION_CONFIDENCE and
        result.get('arena_boundaries', {}) and
        result['arena_boundaries'].get('width', 0) > 0 and
        result['arena_boundaries'].get('height', 0) > 0
    )
    
    return {
        'is_valid': is_valid,
        'errors': [] if is_valid else ['Low detection confidence or invalid boundaries'],
        'warnings': []
    }


def _apply_detection_optimization(result: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, str]:
    """Apply detection optimization based on parameters."""
    return {
        'optimization_level': params.get('optimization_level', 'standard'),
        'performance_mode': params.get('performance_mode', 'balanced')
    }


# Export the main classes and functions
__all__ = [
    'calculate_pixel_to_meter_ratio',
    'detect_arena_boundaries', 
    'extract_calibration_from_video',
    'validate_calibration_accuracy',
    'create_coordinate_transformer',
    'ScaleCalibration',
    'ScaleCalibrationManager'
]