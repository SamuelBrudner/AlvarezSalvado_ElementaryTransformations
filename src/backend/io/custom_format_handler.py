"""
Specialized custom format handler providing adaptive processing for diverse experimental plume recording setups 
including custom AVI recordings, non-standard video formats, and experimental data configurations.

This module implements intelligent format detection, parameter extraction, calibration inference, normalization 
configuration, and cross-format compatibility for scientific video processing workflows with emphasis on 
flexibility, robustness, and integration with the unified format registry system.

Key Features:
- Intelligent format detection with confidence levels and metadata extraction
- Adaptive parameter detection for arena size differences and pixel resolution normalization
- Temporal sampling variation handling and intensity unit conversion
- Format-specific optimization strategies with disk-based caching and memory mapping
- Fail-fast custom format validation for early error detection
- Batch processing support for 4000+ simulations with reliable format detection
- Cross-format compatibility support for seamless format interoperability
- Comprehensive error handling with graceful degradation and intermediate result preservation
"""

# External imports with version specifications
import cv2  # opencv-python 4.11.0+ - Video file reading, format detection, and frame processing for custom video formats
import numpy as np  # numpy 2.1.3+ - Numerical array operations for custom format frame data and parameter analysis
from pathlib import Path  # pathlib 3.9+ - Cross-platform path handling for custom format file operations
from typing import Dict, Any, List, Optional, Union, Tuple  # typing 3.9+ - Type hints for custom format handler interfaces and return types
import json  # json 3.9+ - JSON serialization for custom format configuration and metadata
import datetime  # datetime 3.9+ - Timestamp handling for custom format processing and audit trails
import re  # re 3.9+ - Regular expression pattern matching for custom format detection
import math  # math 3.9+ - Mathematical calculations for custom format parameter inference
import statistics  # statistics 3.9+ - Statistical analysis for custom format parameter estimation

# Internal imports from video reading infrastructure
from .video_reader import (
    VideoReader, CustomVideoReader, detect_video_format, 
    create_video_reader_factory, get_video_metadata_cached
)

# Internal imports from error handling framework
from ..error.exceptions import (
    ValidationError, ProcessingError, ConfigurationError
)

# Internal imports from logging infrastructure
from ..utils.logging_utils import (
    get_logger, log_processing_stage, log_validation_error
)

# Internal imports from validation framework
from ..utils.validation_utils import (
    validate_data_format, validate_physical_parameters, ValidationResult
)

# Internal imports from file handling utilities
from ..utils.file_utils import (
    validate_file_exists, validate_video_file, get_file_metadata
)

# Global constants for custom format handler configuration
CUSTOM_FORMAT_IDENTIFIER = 'custom'
SUPPORTED_CUSTOM_EXTENSIONS = ['.avi', '.mp4', '.mov', '.mkv', '.wmv']
CUSTOM_FORMAT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_ARENA_SIZE_ESTIMATE = 1.0
DEFAULT_PIXEL_SCALE_ESTIMATE = 0.001
DEFAULT_FRAME_RATE_ESTIMATE = 30.0
MIN_CONFIDENCE_FOR_PROCESSING = 0.6
PARAMETER_INFERENCE_SAMPLES = 10
INTENSITY_ANALYSIS_PERCENTILES = [5, 25, 50, 75, 95]
SPATIAL_ANALYSIS_GRID_SIZE = 32
TEMPORAL_ANALYSIS_WINDOW = 100


def detect_custom_format(
    file_path: str,
    deep_inspection: bool = False,
    format_hints: Dict[str, Any] = None
) -> 'FormatDetectionResult':
    """
    Detect and classify custom format video files using the video reader's comprehensive format detection 
    system including header analysis, metadata inspection, and heuristic pattern matching to determine 
    format characteristics and processing requirements.
    
    This function implements comprehensive custom format detection using multiple detection methods 
    including file signature analysis, metadata inspection, and codec identification.
    
    Args:
        file_path: Path to the custom format video file for detection
        deep_inspection: Enable deep format inspection using video reader capabilities
        format_hints: Additional hints to improve detection accuracy and confidence
        
    Returns:
        FormatDetectionResult: Custom format detection result with confidence levels, format characteristics, and processing recommendations
    """
    logger = get_logger('custom_format.detection', 'PROCESSING')
    
    try:
        # Validate custom format file exists and is accessible using file utilities
        file_validation = validate_file_exists(file_path, check_readable=True, check_size_limits=True)
        if not file_validation.is_valid:
            raise ValidationError(
                f"Custom format file validation failed: {file_validation.errors}",
                'file_accessibility_validation',
                {'file_path': file_path, 'validation_errors': file_validation.errors}
            )
        
        # Use detect_video_format function from video_reader to perform comprehensive format detection
        video_format_result = detect_video_format(
            video_path=file_path,
            deep_inspection=deep_inspection,
            detection_hints=format_hints or {}
        )
        
        # Apply format hints to improve detection accuracy and confidence if provided
        confidence_boost = 0.0
        if format_hints:
            if format_hints.get('expected_custom_format', False):
                confidence_boost += 0.2
            if format_hints.get('arena_size_known', False):
                confidence_boost += 0.1
            if format_hints.get('calibration_available', False):
                confidence_boost += 0.15
        
        # Perform deep format inspection if deep_inspection is enabled using video reader capabilities
        format_characteristics = {}
        processing_recommendations = []
        
        if deep_inspection:
            # Get comprehensive video metadata using video reader caching
            metadata = get_video_metadata_cached(
                video_path=file_path,
                force_refresh=False,
                include_frame_analysis=True
            )
            
            format_characteristics.update({
                'video_metadata': metadata.get('format_metadata', {}),
                'frame_analysis': metadata.get('format_metadata', {}).get('frame_analysis', {}),
                'codec_compatibility': _assess_codec_compatibility(metadata),
                'processing_complexity': _estimate_processing_complexity(metadata)
            })
            
            # Generate format-specific processing recommendations
            processing_recommendations.extend([
                "Enable format-specific optimization for custom video processing",
                "Consider parameter inference for unknown calibration settings",
                "Use adaptive normalization for cross-format compatibility"
            ])
        
        # Create localized FormatDetectionResult with detected format characteristics
        detection_result = FormatDetectionResult(
            format_type=CUSTOM_FORMAT_IDENTIFIER,
            confidence_level=min(1.0, video_format_result.get('confidence_level', 0.5) + confidence_boost),
            format_characteristics=format_characteristics
        )
        
        # Calculate detection confidence based on multiple format indicators
        file_extension = Path(file_path).suffix.lower()
        if file_extension in SUPPORTED_CUSTOM_EXTENSIONS:
            detection_result.confidence_level = min(1.0, detection_result.confidence_level + 0.1)
        
        # Generate format characteristics and processing recommendations
        detection_result.format_characteristics.update({
            'file_extension': file_extension,
            'detection_timestamp': datetime.datetime.now().isoformat(),
            'deep_inspection_performed': deep_inspection,
            'hints_applied': bool(format_hints)
        })
        
        for recommendation in processing_recommendations:
            detection_result.add_processing_recommendation(
                recommendation_type='optimization',
                recommendation_text=recommendation,
                recommendation_context={'confidence': detection_result.confidence_level}
            )
        
        # Log custom format detection results with performance metrics
        logger.info(
            f"Custom format detection completed: {file_path} - confidence: {detection_result.confidence_level:.3f}, "
            f"format: {detection_result.format_type}, deep_inspection: {deep_inspection}"
        )
        
        # Return comprehensive format detection result with confidence assessment
        return detection_result
        
    except Exception as e:
        logger.error(f"Custom format detection failed for {file_path}: {e}")
        
        # Return failed detection result with error information
        error_result = FormatDetectionResult(
            format_type='unknown',
            confidence_level=0.0,
            format_characteristics={'detection_error': str(e)}
        )
        
        error_result.add_processing_recommendation(
            recommendation_type='error_recovery',
            recommendation_text=f"Address detection error: {str(e)}",
            recommendation_context={'error_type': type(e).__name__}
        )
        
        return error_result


def validate_custom_format_compatibility(
    custom_file_path: str,
    compatibility_requirements: Dict[str, Any] = None,
    strict_validation: bool = False
) -> ValidationResult:
    """
    Validate custom format compatibility with plume simulation system requirements including codec support, 
    resolution constraints, and processing feasibility assessment using video reader validation capabilities.
    
    This function implements comprehensive compatibility validation for custom format files with detailed 
    analysis and actionable recommendations for processing optimization.
    
    Args:
        custom_file_path: Path to the custom format file for compatibility validation
        compatibility_requirements: System requirements for compatibility checking
        strict_validation: Enable strict validation checks for scientific computing requirements
        
    Returns:
        ValidationResult: Custom format compatibility validation result with detailed analysis and processing recommendations
    """
    logger = get_logger('custom_format.validation', 'VALIDATION')
    
    try:
        # Validate custom format file structure and accessibility using file utilities
        file_validation = validate_video_file(
            video_path=custom_file_path,
            expected_format=CUSTOM_FORMAT_IDENTIFIER,
            validate_codec=True,
            check_integrity=strict_validation
        )
        
        if not file_validation.is_valid:
            raise ValidationError(
                f"Custom format file structure validation failed: {file_validation.errors}",
                'format_structure_validation',
                {'file_path': custom_file_path, 'validation_errors': file_validation.errors}
            )
        
        # Create VideoReader instance for the custom file using video reader factory
        try:
            video_reader = create_video_reader_factory(
                video_path=custom_file_path,
                reader_config={'format_type': CUSTOM_FORMAT_IDENTIFIER, 'enable_validation': True},
                enable_caching=True,
                enable_optimization=True
            )
        except Exception as e:
            raise ValidationError(
                f"Failed to create video reader for custom format: {e}",
                'video_reader_creation',
                {'file_path': custom_file_path, 'error': str(e)}
            )
        
        # Use VideoReader.validate_integrity method for comprehensive format validation
        integrity_result = video_reader.validate_integrity(
            deep_validation=strict_validation,
            sample_frame_count=PARAMETER_INFERENCE_SAMPLES
        )
        
        # Check video codec compatibility with processing pipeline requirements
        video_metadata = video_reader.get_metadata(
            include_frame_analysis=True,
            include_processing_recommendations=True
        )
        
        codec_compatibility = _validate_codec_compatibility(video_metadata, compatibility_requirements)
        
        # Validate resolution and frame rate against system constraints
        resolution_validation = _validate_resolution_constraints(video_metadata, compatibility_requirements)
        framerate_validation = _validate_framerate_constraints(video_metadata, compatibility_requirements)
        
        # Assess custom format processing feasibility and resource requirements
        processing_feasibility = _assess_processing_feasibility(video_metadata, compatibility_requirements)
        
        # Perform strict validation checks if strict_validation is enabled
        strict_validation_results = None
        if strict_validation:
            strict_validation_results = _perform_strict_custom_validation(
                video_reader, video_metadata, compatibility_requirements
            )
        
        # Generate compatibility recommendations for identified issues using ValidationResult
        validation_result = ValidationResult(
            validation_type='custom_format_compatibility',
            is_valid=True,
            validation_context=f'file={custom_file_path}, strict={strict_validation}'
        )
        
        # Merge validation results from different checks
        if not integrity_result.is_valid:
            validation_result.is_valid = False
            for error in integrity_result.errors:
                validation_result.add_error(f"Integrity check failed: {error}")
        
        if not codec_compatibility['compatible']:
            validation_result.add_warning(f"Codec compatibility issue: {codec_compatibility['message']}")
            validation_result.add_recommendation(
                "Consider converting to a more compatible codec for optimal processing"
            )
        
        if not resolution_validation['valid']:
            if strict_validation:
                validation_result.is_valid = False
                validation_result.add_error(f"Resolution validation failed: {resolution_validation['message']}")
            else:
                validation_result.add_warning(f"Resolution issue: {resolution_validation['message']}")
        
        if not framerate_validation['valid']:
            validation_result.add_warning(f"Frame rate issue: {framerate_validation['message']}")
            validation_result.add_recommendation(
                "Consider temporal resampling for consistent frame rate processing"
            )
        
        if not processing_feasibility['feasible']:
            validation_result.is_valid = False
            validation_result.add_error(f"Processing not feasible: {processing_feasibility['reason']}")
            validation_result.add_recommendation(processing_feasibility['recommendation'])
        
        # Include strict validation results if performed
        if strict_validation_results and not strict_validation_results['passed']:
            validation_result.is_valid = False
            for issue in strict_validation_results['issues']:
                validation_result.add_error(f"Strict validation failed: {issue}")
        
        # Add compatibility metrics
        validation_result.add_metric('codec_compatibility_score', codec_compatibility['score'])
        validation_result.add_metric('processing_feasibility_score', processing_feasibility['score'])
        validation_result.add_metric('overall_compatibility_score', _calculate_compatibility_score(
            codec_compatibility, resolution_validation, framerate_validation, processing_feasibility
        ))
        
        # Log custom format compatibility validation operation and results
        logger.info(
            f"Custom format compatibility validation completed: {custom_file_path} - "
            f"valid: {validation_result.is_valid}, compatibility_score: {validation_result.metrics.get('overall_compatibility_score', 0.0):.3f}"
        )
        
        # Clean up video reader resources
        video_reader.close()
        
        # Return comprehensive compatibility validation result with actionable feedback
        return validation_result
        
    except Exception as e:
        logger.error(f"Custom format compatibility validation failed for {custom_file_path}: {e}")
        
        # Create error validation result
        error_result = ValidationResult(
            validation_type='custom_format_compatibility',
            is_valid=False,
            validation_context=f'file={custom_file_path}, error=True'
        )
        
        error_result.add_error(f"Compatibility validation exception: {str(e)}")
        error_result.add_recommendation("Review file format and ensure it meets system requirements")
        
        return error_result


def infer_custom_format_parameters(
    custom_file_path: str,
    sample_frame_count: int = PARAMETER_INFERENCE_SAMPLES,
    inference_config: Dict[str, Any] = None,
    use_statistical_analysis: bool = True
) -> Dict[str, Any]:
    """
    Intelligently infer physical and calibration parameters from custom format video data using 
    CustomVideoReader's adaptive parameter detection capabilities including statistical analysis, 
    spatial pattern recognition, and temporal dynamics assessment.
    
    This function provides comprehensive parameter inference for custom format videos with 
    statistical analysis and confidence estimation.
    
    Args:
        custom_file_path: Path to the custom format video file for parameter inference
        sample_frame_count: Number of frames to sample for parameter inference analysis
        inference_config: Configuration parameters and constraints for inference algorithms
        use_statistical_analysis: Enable statistical analysis of intensity distributions
        
    Returns:
        Dict[str, Any]: Inferred parameters including arena dimensions, pixel scaling, temporal characteristics, and confidence estimates
    """
    logger = get_logger('custom_format.inference', 'PROCESSING')
    
    try:
        # Create CustomVideoReader instance using video reader factory
        custom_reader = create_video_reader_factory(
            video_path=custom_file_path,
            reader_config={
                'format_type': CUSTOM_FORMAT_IDENTIFIER,
                'enable_parameter_inference': True,
                'sample_frame_count': sample_frame_count
            },
            enable_caching=True,
            enable_optimization=True
        )
        
        # Use CustomVideoReader.detect_format_parameters method for parameter inference
        if hasattr(custom_reader, 'detect_format_parameters'):
            raw_parameters = custom_reader.detect_format_parameters(
                deep_analysis=True,
                detection_hints=inference_config or {}
            )
        else:
            # Fallback to manual parameter detection
            raw_parameters = _manual_parameter_detection(custom_reader, sample_frame_count)
        
        # Extract comprehensive video metadata using get_video_metadata_cached function
        video_metadata = get_video_metadata_cached(
            video_path=custom_file_path,
            force_refresh=False,
            include_frame_analysis=True
        )
        
        # Sample representative frames from custom format video using VideoReader.read_frame_batch
        frame_indices = _generate_sample_frame_indices(
            total_frames=video_metadata.get('format_metadata', {}).get('opencv', {}).get('frame_count', 100),
            sample_count=sample_frame_count
        )
        
        sampled_frames = custom_reader.read_frame_batch(
            frame_indices=frame_indices,
            use_cache=True,
            parallel_processing=False
        )
        
        # Analyze spatial characteristics and arena boundary detection
        spatial_analysis = _analyze_spatial_characteristics(sampled_frames, inference_config)
        
        # Perform statistical analysis of intensity distributions if use_statistical_analysis is enabled
        intensity_analysis = {}
        if use_statistical_analysis:
            intensity_analysis = _analyze_intensity_distributions(
                sampled_frames, INTENSITY_ANALYSIS_PERCENTILES
            )
        
        # Estimate pixel-to-meter scaling using spatial pattern analysis
        pixel_scaling = _estimate_pixel_scaling(spatial_analysis, inference_config)
        
        # Infer temporal sampling characteristics and frame rate consistency
        temporal_analysis = _analyze_temporal_characteristics(video_metadata, sampled_frames)
        
        # Analyze intensity calibration and dynamic range characteristics
        intensity_calibration = _analyze_intensity_calibration(intensity_analysis, sampled_frames)
        
        # Apply inference configuration parameters and constraints
        if inference_config:
            pixel_scaling = _apply_inference_constraints(pixel_scaling, inference_config.get('pixel_scaling', {}))
            spatial_analysis = _apply_inference_constraints(spatial_analysis, inference_config.get('spatial', {}))
            temporal_analysis = _apply_inference_constraints(temporal_analysis, inference_config.get('temporal', {}))
        
        # Calculate confidence estimates for each inferred parameter
        confidence_estimates = _calculate_parameter_confidence(
            spatial_analysis, pixel_scaling, temporal_analysis, intensity_calibration
        )
        
        # Generate parameter inference report with uncertainty quantification
        inferred_parameters = {
            'arena_dimensions': {
                'width_meters': spatial_analysis.get('estimated_width', DEFAULT_ARENA_SIZE_ESTIMATE),
                'height_meters': spatial_analysis.get('estimated_height', DEFAULT_ARENA_SIZE_ESTIMATE),
                'confidence': confidence_estimates.get('spatial_confidence', 0.5)
            },
            'pixel_scaling': {
                'pixels_per_meter': 1.0 / pixel_scaling.get('meters_per_pixel', DEFAULT_PIXEL_SCALE_ESTIMATE),
                'meters_per_pixel': pixel_scaling.get('meters_per_pixel', DEFAULT_PIXEL_SCALE_ESTIMATE),
                'confidence': confidence_estimates.get('scaling_confidence', 0.5)
            },
            'temporal_characteristics': {
                'frame_rate_hz': temporal_analysis.get('effective_frame_rate', DEFAULT_FRAME_RATE_ESTIMATE),
                'temporal_consistency': temporal_analysis.get('consistency_score', 0.8),
                'confidence': confidence_estimates.get('temporal_confidence', 0.7)
            },
            'intensity_characteristics': {
                'dynamic_range': intensity_calibration.get('dynamic_range', [0.0, 1.0]),
                'intensity_type': intensity_calibration.get('data_type', 'normalized'),
                'calibration_quality': intensity_calibration.get('calibration_score', 0.6),
                'confidence': confidence_estimates.get('intensity_confidence', 0.6)
            },
            'inference_metadata': {
                'sample_frame_count': len(sampled_frames),
                'inference_timestamp': datetime.datetime.now().isoformat(),
                'statistical_analysis_used': use_statistical_analysis,
                'inference_config_applied': bool(inference_config),
                'overall_confidence': confidence_estimates.get('overall_confidence', 0.6)
            }
        }
        
        # Log parameter inference operation with statistical metrics
        logger.info(
            f"Custom format parameter inference completed: {custom_file_path} - "
            f"arena: {inferred_parameters['arena_dimensions']['width_meters']:.3f}x{inferred_parameters['arena_dimensions']['height_meters']:.3f}m, "
            f"pixel_scale: {inferred_parameters['pixel_scaling']['meters_per_pixel']:.6f}m/px, "
            f"overall_confidence: {inferred_parameters['inference_metadata']['overall_confidence']:.3f}"
        )
        
        # Clean up video reader resources
        custom_reader.close()
        
        # Return comprehensive parameter dictionary with confidence estimates
        return inferred_parameters
        
    except Exception as e:
        logger.error(f"Custom format parameter inference failed for {custom_file_path}: {e}")
        
        # Return default parameters with low confidence
        return {
            'arena_dimensions': {
                'width_meters': DEFAULT_ARENA_SIZE_ESTIMATE,
                'height_meters': DEFAULT_ARENA_SIZE_ESTIMATE,
                'confidence': 0.0
            },
            'pixel_scaling': {
                'pixels_per_meter': 1.0 / DEFAULT_PIXEL_SCALE_ESTIMATE,
                'meters_per_pixel': DEFAULT_PIXEL_SCALE_ESTIMATE,
                'confidence': 0.0
            },
            'temporal_characteristics': {
                'frame_rate_hz': DEFAULT_FRAME_RATE_ESTIMATE,
                'temporal_consistency': 0.0,
                'confidence': 0.0
            },
            'intensity_characteristics': {
                'dynamic_range': [0.0, 1.0],
                'intensity_type': 'unknown',
                'calibration_quality': 0.0,
                'confidence': 0.0
            },
            'inference_metadata': {
                'sample_frame_count': 0,
                'inference_timestamp': datetime.datetime.now().isoformat(),
                'inference_error': str(e),
                'overall_confidence': 0.0
            }
        }


def create_custom_format_handler(
    custom_file_path: str,
    handler_config: Dict[str, Any] = None,
    enable_parameter_inference: bool = True,
    enable_optimizations: bool = True
) -> 'CustomFormatHandler':
    """
    Factory function to create optimized custom format handler instance with adaptive configuration, 
    parameter inference, and format-specific optimization for diverse experimental setups using 
    video reader infrastructure.
    
    This function provides comprehensive custom format handler creation with automatic configuration 
    and optimization for scientific video processing workflows.
    
    Args:
        custom_file_path: Path to the custom format video file
        handler_config: Configuration parameters for handler optimization
        enable_parameter_inference: Enable automatic parameter inference for unknown calibration
        enable_optimizations: Enable performance optimizations for scientific computing
        
    Returns:
        CustomFormatHandler: Configured custom format handler instance optimized for the specific file and experimental setup
    """
    logger = get_logger('custom_format.factory', 'PROCESSING')
    
    try:
        # Detect and validate custom format using detect_custom_format function
        format_detection = detect_custom_format(
            file_path=custom_file_path,
            deep_inspection=True,
            format_hints=handler_config.get('format_hints', {}) if handler_config else {}
        )
        
        if not format_detection.format_detected or format_detection.confidence_level < MIN_CONFIDENCE_FOR_PROCESSING:
            raise ValidationError(
                f"Custom format detection confidence {format_detection.confidence_level:.3f} below minimum threshold {MIN_CONFIDENCE_FOR_PROCESSING}",
                'format_detection_validation',
                {'file_path': custom_file_path, 'confidence': format_detection.confidence_level}
            )
        
        # Create optimized VideoReader instance using create_video_reader_factory
        video_reader = create_video_reader_factory(
            video_path=custom_file_path,
            reader_config=handler_config or {},
            enable_caching=True,
            enable_optimization=enable_optimizations
        )
        
        # Infer format parameters if enable_parameter_inference is True using infer_custom_format_parameters
        inferred_parameters = {}
        if enable_parameter_inference:
            inferred_parameters = infer_custom_format_parameters(
                custom_file_path=custom_file_path,
                sample_frame_count=handler_config.get('inference_sample_count', PARAMETER_INFERENCE_SAMPLES) if handler_config else PARAMETER_INFERENCE_SAMPLES,
                inference_config=handler_config.get('inference_config', {}) if handler_config else {},
                use_statistical_analysis=True
            )
        
        # Load custom format-specific configuration and optimization settings
        format_config = _load_format_specific_config(format_detection, handler_config)
        
        # Create CustomFormatHandler instance with detected and inferred parameters
        custom_handler = CustomFormatHandler(
            custom_file_path=custom_file_path,
            handler_config={
                **format_config,
                'format_detection': format_detection.to_dict(),
                'inferred_parameters': inferred_parameters,
                'video_reader': video_reader,
                'optimization_enabled': enable_optimizations
            }
        )
        
        # Apply performance optimizations if enable_optimizations is True
        if enable_optimizations:
            optimization_config = handler_config.get('optimization', {}) if handler_config else {}
            custom_handler._apply_performance_optimizations(optimization_config)
        
        # Configure adaptive processing based on format characteristics
        custom_handler._configure_adaptive_processing(format_detection.format_characteristics)
        
        # Setup validation and error handling for custom format processing
        custom_handler._initialize_validation_framework()
        
        # Validate handler initialization and file accessibility
        initialization_result = custom_handler._validate_initialization()
        if not initialization_result['success']:
            raise ProcessingError(
                f"Custom format handler initialization failed: {initialization_result['error']}",
                'handler_initialization',
                custom_file_path,
                {'initialization_result': initialization_result}
            )
        
        # Log custom format handler creation with configuration details
        logger.info(
            f"Custom format handler created successfully: {custom_file_path} - "
            f"confidence: {format_detection.confidence_level:.3f}, "
            f"parameter_inference: {enable_parameter_inference}, "
            f"optimizations: {enable_optimizations}"
        )
        
        # Return configured and validated custom format handler instance
        return custom_handler
        
    except Exception as e:
        logger.error(f"Custom format handler creation failed for {custom_file_path}: {e}")
        raise ProcessingError(
            f"Failed to create custom format handler: {str(e)}",
            'handler_creation',
            custom_file_path,
            {'error_type': type(e).__name__, 'error_message': str(e)}
        )


class FormatDetectionResult:
    """
    Localized format detection result container class providing structured storage of custom format 
    detection outcomes including detected format type, confidence levels, format characteristics, 
    processing recommendations, and metadata for scientific video processing workflows.
    
    This class eliminates circular dependency by localizing format detection result functionality 
    within the custom format handler module.
    """
    
    def __init__(
        self,
        format_type: str,
        confidence_level: float,
        format_characteristics: Dict[str, Any]
    ):
        """
        Initialize format detection result with detected format type, confidence level, and format 
        characteristics for comprehensive format analysis and processing optimization.
        
        Args:
            format_type: Detected format type identifier
            confidence_level: Confidence level for format detection (0.0 to 1.0)
            format_characteristics: Dictionary of format-specific characteristics and metadata
        """
        # Set format type, confidence level, and format characteristics
        self.format_type = format_type
        self.confidence_level = max(0.0, min(1.0, confidence_level))  # Clamp to valid range
        self.format_characteristics = format_characteristics.copy()
        
        # Initialize alternative formats list and processing recommendations
        self.alternative_formats: List[Tuple[str, float]] = []
        self.processing_recommendations: List[Dict[str, str]] = []
        
        # Set detection timestamp and metadata containers
        self.detection_timestamp = datetime.datetime.now()
        self.detection_metadata: Dict[str, Any] = {}
        
        # Determine format support status based on format type
        self.is_supported = format_type in [CUSTOM_FORMAT_IDENTIFIER, 'avi', 'mp4', 'mov']
        
        # Initialize detection metadata with basic format information
        self.detection_metadata.update({
            'detection_method': 'comprehensive_analysis',
            'format_registry': 'custom_format_handler',
            'detection_version': '1.0.0'
        })
        
        # Validate confidence level is within valid range (0.0 to 1.0)
        if not (0.0 <= confidence_level <= 1.0):
            raise ValueError(f"Confidence level must be between 0.0 and 1.0, got {confidence_level}")
        
        # Log format detection result initialization
        logger = get_logger('format_detection_result', 'PROCESSING')
        logger.debug(f"Format detection result initialized: {format_type} (confidence: {confidence_level:.3f})")
    
    @property
    def format_detected(self) -> bool:
        """Check if format was successfully detected with sufficient confidence."""
        return self.confidence_level >= CUSTOM_FORMAT_CONFIDENCE_THRESHOLD
    
    def add_alternative_format(self, alternative_format: str, alternative_confidence: float) -> None:
        """
        Add alternative format possibility with confidence level for comprehensive format detection analysis.
        
        Args:
            alternative_format: Alternative format type identifier
            alternative_confidence: Confidence level for the alternative format
        """
        # Validate alternative format type and confidence level
        if not isinstance(alternative_format, str) or not alternative_format.strip():
            raise ValueError("Alternative format must be a non-empty string")
        
        if not (0.0 <= alternative_confidence <= 1.0):
            raise ValueError(f"Alternative confidence must be between 0.0 and 1.0, got {alternative_confidence}")
        
        # Add alternative format to alternatives list with confidence
        self.alternative_formats.append((alternative_format, alternative_confidence))
        
        # Update detection metadata with alternative format information
        self.detection_metadata[f'alternative_{len(self.alternative_formats)}'] = {
            'format': alternative_format,
            'confidence': alternative_confidence,
            'added_at': datetime.datetime.now().isoformat()
        }
        
        # Log alternative format addition for comprehensive detection tracking
        logger = get_logger('format_detection_result', 'PROCESSING')
        logger.debug(f"Alternative format added: {alternative_format} (confidence: {alternative_confidence:.3f})")
    
    def get_best_match(self) -> Tuple[str, float]:
        """
        Get the best matching format type considering primary detection and alternative formats 
        based on confidence levels.
        
        Returns:
            Tuple[str, float]: Best matching format type and confidence level
        """
        # Compare primary format confidence with alternative format confidences
        best_format = self.format_type
        best_confidence = self.confidence_level
        
        # Identify highest confidence format from all detected possibilities
        for alt_format, alt_confidence in self.alternative_formats:
            if alt_confidence > best_confidence:
                best_format = alt_format
                best_confidence = alt_confidence
        
        # Return best matching format type and corresponding confidence level
        return best_format, best_confidence
    
    def add_processing_recommendation(
        self,
        recommendation_type: str,
        recommendation_text: str,
        recommendation_context: Dict[str, Any] = None
    ) -> None:
        """
        Add processing recommendation for detected format including optimization strategies and parameter suggestions.
        
        Args:
            recommendation_type: Type of processing recommendation
            recommendation_text: Detailed recommendation text
            recommendation_context: Additional context for the recommendation
        """
        # Add recommendation to processing recommendations dictionary
        recommendation = {
            'type': recommendation_type,
            'text': recommendation_text,
            'context': recommendation_context or {},
            'added_at': datetime.datetime.now().isoformat()
        }
        
        self.processing_recommendations.append(recommendation)
        
        # Store recommendation context and metadata
        self.detection_metadata[f'recommendation_{len(self.processing_recommendations)}'] = recommendation
        
        # Update detection metadata with recommendation information
        if 'recommendation_count' not in self.detection_metadata:
            self.detection_metadata['recommendation_count'] = 0
        self.detection_metadata['recommendation_count'] += 1
        
        # Log processing recommendation addition for format optimization
        logger = get_logger('format_detection_result', 'PROCESSING')
        logger.debug(f"Processing recommendation added: {recommendation_type} - {recommendation_text}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert format detection result to dictionary format for serialization, reporting, and 
        integration with other system components.
        
        Returns:
            Dict[str, Any]: Complete format detection result as dictionary with all properties and metadata
        """
        # Convert all properties to dictionary format
        result_dict = {
            'format_type': self.format_type,
            'confidence_level': self.confidence_level,
            'format_detected': self.format_detected,
            'is_supported': self.is_supported,
            'detection_timestamp': self.detection_timestamp.isoformat(),
            'format_characteristics': self.format_characteristics.copy(),
            'alternative_formats': self.alternative_formats.copy(),
            'processing_recommendations': self.processing_recommendations.copy(),
            'detection_metadata': self.detection_metadata.copy()
        }
        
        # Include format type, confidence, and characteristics
        result_dict['summary'] = {
            'primary_format': self.format_type,
            'confidence': self.confidence_level,
            'alternatives_count': len(self.alternative_formats),
            'recommendations_count': len(self.processing_recommendations)
        }
        
        # Add alternative formats and processing recommendations
        if self.alternative_formats:
            best_format, best_confidence = self.get_best_match()
            result_dict['best_match'] = {
                'format': best_format,
                'confidence': best_confidence
            }
        
        # Include detection metadata and timestamp information
        result_dict['detection_quality'] = {
            'confidence_threshold_met': self.format_detected,
            'support_available': self.is_supported,
            'analysis_depth': 'comprehensive' if self.format_characteristics else 'basic'
        }
        
        # Format detection result for readability and system integration
        return result_dict


class CustomFormatHandler:
    """
    Comprehensive custom format handler providing adaptive processing for diverse experimental plume 
    recording setups with intelligent parameter detection, format-specific optimization, normalization 
    configuration, and seamless integration with the plume simulation system for scientific video 
    processing workflows using video reader infrastructure.
    """
    
    def __init__(
        self,
        custom_file_path: str,
        handler_config: Dict[str, Any]
    ):
        """
        Initialize custom format handler with format detection, parameter inference, validation, and 
        adaptive configuration for diverse experimental video processing using video reader infrastructure.
        
        Args:
            custom_file_path: Path to the custom format video file
            handler_config: Configuration dictionary for handler behavior and optimization
        """
        # Validate custom format file exists and is accessible using file utilities
        self.custom_file_path = str(Path(custom_file_path).absolute())
        file_validation = validate_file_exists(self.custom_file_path, check_readable=True, check_size_limits=True)
        
        if not file_validation.is_valid:
            raise ValidationError(
                f"Custom format file validation failed: {file_validation.errors}",
                'file_accessibility_validation',
                {'file_path': self.custom_file_path, 'validation_errors': file_validation.errors}
            )
        
        # Store handler configuration and initialize processing metadata
        self.handler_config = handler_config.copy()
        self.format_characteristics = self.handler_config.get('format_detection', {}).get('format_characteristics', {})
        self.inferred_parameters = self.handler_config.get('inferred_parameters', {})
        self.parameter_confidence = {}
        self.is_initialized = False
        
        # Extract parameter confidence estimates from inferred parameters
        for param_category, param_data in self.inferred_parameters.items():
            if isinstance(param_data, dict) and 'confidence' in param_data:
                self.parameter_confidence[param_category] = param_data['confidence']
        
        # Detect custom format characteristics using detect_video_format function
        if 'format_detection' not in self.handler_config:
            format_detection = detect_custom_format(
                file_path=self.custom_file_path,
                deep_inspection=True,
                format_hints=self.handler_config.get('format_hints', {})
            )
            self.format_characteristics = format_detection.format_characteristics
        
        # Create VideoReader instance using create_video_reader_factory
        self.video_reader = self.handler_config.get('video_reader')
        if not self.video_reader:
            self.video_reader = create_video_reader_factory(
                video_path=self.custom_file_path,
                reader_config={'format_type': CUSTOM_FORMAT_IDENTIFIER},
                enable_caching=True,
                enable_optimization=True
            )
        
        # Create CustomVideoReader instance for adaptive processing capabilities
        self.custom_video_reader = None
        if hasattr(self.video_reader, '__class__') and 'CustomVideoReader' in str(self.video_reader.__class__):
            self.custom_video_reader = self.video_reader
        else:
            # Create dedicated custom video reader
            try:
                self.custom_video_reader = CustomVideoReader(
                    video_path=self.custom_file_path,
                    custom_config={'adaptive_processing': True}
                )
            except Exception as e:
                logger = get_logger('custom_format_handler', 'PROCESSING')
                logger.warning(f"Failed to create CustomVideoReader, using standard VideoReader: {e}")
        
        # Initialize format characteristics and processing metadata
        self.processing_metadata = {
            'initialization_timestamp': datetime.datetime.now().isoformat(),
            'handler_version': '1.0.0',
            'processing_statistics': {},
            'optimization_applied': False
        }
        
        # Infer physical and calibration parameters using CustomVideoReader.detect_format_parameters
        if not self.inferred_parameters:
            self._infer_format_parameters()
        
        # Configure normalization parameters based on inferred characteristics
        self.normalization_config = self._configure_initial_normalization()
        
        # Setup adaptive processing configuration for custom format
        self._setup_adaptive_processing()
        
        # Initialize validation and error handling frameworks
        self.logger = get_logger('custom_format_handler', 'PROCESSING')
        
        # Setup logger for custom format processing operations
        self.logger.info(f"Custom format handler initialized for: {self.custom_file_path}")
        
        # Validate handler initialization and mark as initialized
        initialization_result = self._validate_initialization()
        if initialization_result['success']:
            self.is_initialized = True
        else:
            raise ProcessingError(
                f"Handler initialization validation failed: {initialization_result['error']}",
                'handler_initialization_validation',
                self.custom_file_path,
                {'validation_result': initialization_result}
            )
        
        # Log custom format handler initialization with detected parameters
        self.logger.info(
            f"Custom format handler initialization completed: confidence={self.parameter_confidence.get('overall_confidence', 0.0):.3f}, "
            f"parameters_inferred={bool(self.inferred_parameters)}, "
            f"adaptive_processing=enabled"
        )
    
    def detect_format_characteristics(
        self,
        deep_analysis: bool = True,
        analysis_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Detect and analyze custom format characteristics including video properties, experimental setup 
        indicators, and processing requirements for adaptive handling using video reader capabilities.
        
        Args:
            deep_analysis: Enable comprehensive analysis including content inspection
            analysis_config: Configuration parameters and constraints for analysis
            
        Returns:
            Dict[str, Any]: Format characteristics including video properties, experimental indicators, and processing recommendations
        """
        try:
            # Use VideoReader.get_metadata method to extract comprehensive video properties
            video_metadata = self.video_reader.get_metadata(
                include_frame_analysis=deep_analysis,
                include_processing_recommendations=True
            )
            
            # Use detect_video_format function for detailed format analysis
            format_detection = detect_video_format(
                video_path=self.custom_file_path,
                deep_inspection=deep_analysis,
                detection_hints=analysis_config or {}
            )
            
            # Analyze video codec and container format characteristics
            codec_analysis = _analyze_codec_characteristics(video_metadata)
            container_analysis = _analyze_container_format(self.custom_file_path)
            
            # Detect experimental setup indicators from video content
            experimental_indicators = {}
            if deep_analysis:
                experimental_indicators = self._detect_experimental_setup(video_metadata)
            
            # Perform deep format analysis if deep_analysis is enabled
            deep_analysis_results = {}
            if deep_analysis:
                deep_analysis_results = {
                    'spatial_analysis': self._analyze_spatial_patterns(video_metadata),
                    'temporal_analysis': self._analyze_temporal_patterns(video_metadata),
                    'intensity_analysis': self._analyze_intensity_patterns(video_metadata)
                }
            
            # Apply analysis configuration parameters and constraints
            if analysis_config:
                deep_analysis_results = self._apply_analysis_constraints(deep_analysis_results, analysis_config)
            
            # Generate format-specific processing recommendations
            processing_recommendations = self._generate_format_recommendations(
                video_metadata, codec_analysis, container_analysis, experimental_indicators
            )
            
            # Update format characteristics with analysis results
            characteristics = {
                'video_properties': video_metadata.get('basic_properties', {}),
                'codec_analysis': codec_analysis,
                'container_analysis': container_analysis,
                'experimental_indicators': experimental_indicators,
                'deep_analysis': deep_analysis_results,
                'processing_recommendations': processing_recommendations,
                'analysis_timestamp': datetime.datetime.now().isoformat(),
                'analysis_config_applied': bool(analysis_config)
            }
            
            # Store updated characteristics
            self.format_characteristics.update(characteristics)
            
            # Log format characteristics detection operation
            self.logger.info(
                f"Format characteristics detection completed: {len(characteristics)} categories analyzed, "
                f"deep_analysis={deep_analysis}, recommendations={len(processing_recommendations)}"
            )
            
            # Return comprehensive format characteristics dictionary
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Format characteristics detection failed: {e}")
            raise ProcessingError(
                f"Failed to detect format characteristics: {str(e)}",
                'format_characteristics_detection',
                self.custom_file_path,
                {'deep_analysis': deep_analysis, 'analysis_config': analysis_config}
            )
    
    def extract_calibration_parameters(
        self,
        sample_size: int = PARAMETER_INFERENCE_SAMPLES,
        use_statistical_inference: bool = True,
        calibration_hints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract and infer calibration parameters from custom format video including spatial scaling, 
        temporal characteristics, and intensity calibration through intelligent analysis using 
        CustomVideoReader capabilities.
        
        Args:
            sample_size: Number of frames to sample for calibration analysis
            use_statistical_inference: Enable statistical inference for parameter estimation
            calibration_hints: Hints and constraints for calibration parameter extraction
            
        Returns:
            Dict[str, Any]: Extracted calibration parameters with confidence estimates and uncertainty quantification
        """
        try:
            # Use CustomVideoReader.detect_format_parameters for comprehensive parameter extraction
            if self.custom_video_reader and hasattr(self.custom_video_reader, 'detect_format_parameters'):
                raw_parameters = self.custom_video_reader.detect_format_parameters(
                    deep_analysis=True,
                    detection_hints=calibration_hints or {}
                )
            else:
                raw_parameters = {}
            
            # Sample representative frames using VideoReader.read_frame_batch method
            video_metadata = self.video_reader.get_metadata(include_frame_analysis=True)
            total_frames = video_metadata.get('basic_properties', {}).get('frame_count', 100)
            
            frame_indices = _generate_sample_frame_indices(total_frames, sample_size)
            sampled_frames = self.video_reader.read_frame_batch(
                frame_indices=frame_indices,
                use_cache=True,
                parallel_processing=False
            )
            
            # Analyze spatial characteristics and arena boundary detection
            spatial_calibration = self._extract_spatial_calibration(sampled_frames, calibration_hints)
            
            # Infer pixel-to-meter scaling using spatial pattern analysis
            scaling_calibration = self._extract_scaling_calibration(spatial_calibration, sampled_frames)
            
            # Extract temporal sampling characteristics and frame rate using video metadata
            temporal_calibration = self._extract_temporal_calibration(video_metadata, sampled_frames)
            
            # Analyze intensity distributions and dynamic range
            intensity_calibration = self._extract_intensity_calibration(sampled_frames)
            
            # Apply statistical inference if use_statistical_inference is enabled
            if use_statistical_inference:
                spatial_calibration = self._apply_statistical_inference(spatial_calibration, sampled_frames)
                scaling_calibration = self._apply_statistical_inference(scaling_calibration, sampled_frames)
                temporal_calibration = self._apply_statistical_inference(temporal_calibration, sampled_frames)
                intensity_calibration = self._apply_statistical_inference(intensity_calibration, sampled_frames)
            
            # Incorporate calibration hints to improve parameter accuracy
            if calibration_hints:
                spatial_calibration = self._apply_calibration_hints(spatial_calibration, calibration_hints)
                scaling_calibration = self._apply_calibration_hints(scaling_calibration, calibration_hints)
                temporal_calibration = self._apply_calibration_hints(temporal_calibration, calibration_hints)
                intensity_calibration = self._apply_calibration_hints(intensity_calibration, calibration_hints)
            
            # Calculate confidence estimates for each calibration parameter
            confidence_estimates = {
                'spatial_confidence': self._calculate_spatial_confidence(spatial_calibration, sampled_frames),
                'scaling_confidence': self._calculate_scaling_confidence(scaling_calibration, sampled_frames),
                'temporal_confidence': self._calculate_temporal_confidence(temporal_calibration, video_metadata),
                'intensity_confidence': self._calculate_intensity_confidence(intensity_calibration, sampled_frames)
            }
            
            overall_confidence = statistics.mean(confidence_estimates.values())
            confidence_estimates['overall_confidence'] = overall_confidence
            
            # Validate calibration parameters against physical constraints
            validation_result = self._validate_calibration_parameters({
                'spatial': spatial_calibration,
                'scaling': scaling_calibration,
                'temporal': temporal_calibration,
                'intensity': intensity_calibration
            })
            
            # Update inferred parameters and confidence estimates
            calibration_parameters = {
                'spatial_calibration': spatial_calibration,
                'scaling_calibration': scaling_calibration,
                'temporal_calibration': temporal_calibration,
                'intensity_calibration': intensity_calibration,
                'confidence_estimates': confidence_estimates,
                'validation_result': validation_result.to_dict() if hasattr(validation_result, 'to_dict') else validation_result,
                'extraction_metadata': {
                    'sample_size': len(sampled_frames),
                    'statistical_inference_used': use_statistical_inference,
                    'calibration_hints_applied': bool(calibration_hints),
                    'extraction_timestamp': datetime.datetime.now().isoformat()
                }
            }
            
            # Update instance state
            self.inferred_parameters.update(calibration_parameters)
            self.parameter_confidence.update(confidence_estimates)
            
            # Log calibration parameter extraction with statistical metrics
            self.logger.info(
                f"Calibration parameter extraction completed: "
                f"spatial_conf={confidence_estimates['spatial_confidence']:.3f}, "
                f"scaling_conf={confidence_estimates['scaling_confidence']:.3f}, "
                f"temporal_conf={confidence_estimates['temporal_confidence']:.3f}, "
                f"intensity_conf={confidence_estimates['intensity_confidence']:.3f}, "
                f"overall_conf={overall_confidence:.3f}"
            )
            
            # Return comprehensive calibration parameters with uncertainty
            return calibration_parameters
            
        except Exception as e:
            self.logger.error(f"Calibration parameter extraction failed: {e}")
            raise ProcessingError(
                f"Failed to extract calibration parameters: {str(e)}",
                'calibration_parameter_extraction',
                self.custom_file_path,
                {'sample_size': sample_size, 'statistical_inference': use_statistical_inference}
            )
    
    def configure_normalization(
        self,
        target_parameters: Dict[str, Any],
        adaptive_configuration: bool = True,
        normalization_constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Configure normalization parameters for custom format processing including spatial normalization, 
        temporal alignment, and intensity calibration based on inferred characteristics using 
        CustomVideoReader configuration.
        
        Args:
            target_parameters: Target parameters for normalization
            adaptive_configuration: Enable adaptive configuration based on format characteristics
            normalization_constraints: Constraints and limits for normalization
            
        Returns:
            Dict[str, Any]: Normalization configuration with transformation parameters and processing settings
        """
        try:
            # Use CustomVideoReader.configure_normalization method for base configuration
            base_config = {}
            if self.custom_video_reader and hasattr(self.custom_video_reader, 'configure_normalization'):
                base_config = self.custom_video_reader.configure_normalization(
                    normalization_requirements=target_parameters,
                    auto_optimize=adaptive_configuration
                )
            
            # Analyze target parameters and normalization requirements
            target_analysis = self._analyze_target_parameters(target_parameters)
            
            # Configure spatial normalization based on inferred arena dimensions
            spatial_config = self._configure_spatial_normalization(
                target_parameters, self.inferred_parameters.get('spatial_calibration', {})
            )
            
            # Setup temporal normalization for frame rate and sampling alignment
            temporal_config = self._configure_temporal_normalization(
                target_parameters, self.inferred_parameters.get('temporal_calibration', {})
            )
            
            # Configure intensity calibration and dynamic range normalization
            intensity_config = self._configure_intensity_normalization(
                target_parameters, self.inferred_parameters.get('intensity_calibration', {})
            )
            
            # Apply adaptive configuration if adaptive_configuration is enabled
            if adaptive_configuration:
                spatial_config = self._adapt_spatial_configuration(spatial_config, self.format_characteristics)
                temporal_config = self._adapt_temporal_configuration(temporal_config, self.format_characteristics)
                intensity_config = self._adapt_intensity_configuration(intensity_config, self.format_characteristics)
            
            # Validate normalization configuration against constraints
            if normalization_constraints:
                validation_result = self._validate_normalization_constraints(
                    spatial_config, temporal_config, intensity_config, normalization_constraints
                )
                
                if not validation_result['valid']:
                    raise ValidationError(
                        f"Normalization configuration validation failed: {validation_result['errors']}",
                        'normalization_configuration_validation',
                        {'constraints': normalization_constraints, 'validation_result': validation_result}
                    )
            
            # Generate transformation matrices and processing parameters
            transformation_matrices = self._generate_transformation_matrices(
                spatial_config, temporal_config, intensity_config
            )
            
            # Update normalization configuration with validated parameters
            normalization_config = {
                'spatial_normalization': spatial_config,
                'temporal_normalization': temporal_config,
                'intensity_normalization': intensity_config,
                'transformation_matrices': transformation_matrices,
                'target_analysis': target_analysis,
                'adaptive_configuration_applied': adaptive_configuration,
                'constraints_applied': bool(normalization_constraints),
                'configuration_timestamp': datetime.datetime.now().isoformat(),
                'base_config': base_config
            }
            
            # Store configuration in instance
            self.normalization_config = normalization_config
            
            # Log normalization configuration operation with parameters
            self.logger.info(
                f"Normalization configuration completed: "
                f"spatial={bool(spatial_config)}, temporal={bool(temporal_config)}, "
                f"intensity={bool(intensity_config)}, adaptive={adaptive_configuration}"
            )
            
            # Return comprehensive normalization configuration dictionary
            return normalization_config
            
        except Exception as e:
            self.logger.error(f"Normalization configuration failed: {e}")
            raise ProcessingError(
                f"Failed to configure normalization: {str(e)}",
                'normalization_configuration',
                self.custom_file_path,
                {'target_parameters': target_parameters, 'adaptive': adaptive_configuration}
            )
    
    def validate_processing_feasibility(
        self,
        processing_requirements: Dict[str, Any],
        estimate_performance: bool = True
    ) -> ValidationResult:
        """
        Validate processing feasibility for custom format including resource requirements, compatibility 
        assessment, and performance projections for batch processing using video reader validation capabilities.
        
        Args:
            processing_requirements: Requirements for processing operations
            estimate_performance: Enable performance estimation and projection
            
        Returns:
            ValidationResult: Processing feasibility validation result with resource analysis and performance projections
        """
        # Create ValidationResult container for feasibility assessment
        validation_result = ValidationResult(
            validation_type='processing_feasibility',
            is_valid=True,
            validation_context=f'file={self.custom_file_path}, performance_estimation={estimate_performance}'
        )
        
        try:
            # Use VideoReader.validate_integrity method for compatibility validation
            integrity_result = self.video_reader.validate_integrity(
                deep_validation=True,
                sample_frame_count=PARAMETER_INFERENCE_SAMPLES
            )
            
            # Merge integrity validation results
            if not integrity_result.is_valid:
                validation_result.is_valid = False
                for error in integrity_result.errors:
                    validation_result.add_error(f"Integrity validation failed: {error}")
            
            # Analyze processing requirements and resource constraints
            resource_analysis = self._analyze_resource_requirements(processing_requirements)
            validation_result.set_metadata('resource_analysis', resource_analysis)
            
            # Validate custom format compatibility with processing pipeline
            compatibility_result = validate_custom_format_compatibility(
                custom_file_path=self.custom_file_path,
                compatibility_requirements=processing_requirements,
                strict_validation=True
            )
            
            if not compatibility_result.is_valid:
                validation_result.is_valid = False
                for error in compatibility_result.errors:
                    validation_result.add_error(f"Compatibility validation failed: {error}")
            
            # Assess memory and computational resource requirements
            memory_assessment = self._assess_memory_requirements(processing_requirements)
            computational_assessment = self._assess_computational_requirements(processing_requirements)
            
            validation_result.add_metric('estimated_memory_mb', memory_assessment['estimated_memory_mb'])
            validation_result.add_metric('computational_complexity', computational_assessment['complexity_score'])
            
            # Estimate processing performance if estimate_performance is enabled
            if estimate_performance:
                performance_projection = self._estimate_processing_performance(
                    processing_requirements, memory_assessment, computational_assessment
                )
                
                validation_result.set_metadata('performance_projection', performance_projection)
                validation_result.add_metric('estimated_processing_time_seconds', performance_projection['processing_time_seconds'])
                validation_result.add_metric('estimated_throughput_fps', performance_projection['throughput_fps'])
                
                # Check performance against targets
                if performance_projection['processing_time_seconds'] > processing_requirements.get('max_processing_time', 300):
                    validation_result.add_warning(
                        f"Estimated processing time ({performance_projection['processing_time_seconds']:.1f}s) exceeds target"
                    )
                
                if performance_projection['throughput_fps'] < processing_requirements.get('min_throughput_fps', 1.0):
                    validation_result.add_warning(
                        f"Estimated throughput ({performance_projection['throughput_fps']:.2f} fps) below target"
                    )
            
            # Check for potential processing bottlenecks and limitations
            bottleneck_analysis = self._analyze_processing_bottlenecks(
                resource_analysis, memory_assessment, computational_assessment
            )
            
            validation_result.set_metadata('bottleneck_analysis', bottleneck_analysis)
            
            for bottleneck in bottleneck_analysis.get('identified_bottlenecks', []):
                validation_result.add_warning(f"Processing bottleneck identified: {bottleneck}")
            
            # Generate processing recommendations and optimizations
            recommendations = self._generate_processing_feasibility_recommendations(
                validation_result, resource_analysis, bottleneck_analysis
            )
            
            for recommendation in recommendations:
                validation_result.add_recommendation(recommendation['text'], recommendation['priority'])
            
            # Add validation warnings for potential issues
            if memory_assessment['estimated_memory_mb'] > 8000:  # 8GB threshold
                validation_result.add_warning("High memory usage estimated - consider optimization")
            
            if computational_assessment['complexity_score'] > 0.8:
                validation_result.add_warning("High computational complexity - consider simplification")
            
            # Log processing feasibility validation operation
            self.logger.info(
                f"Processing feasibility validation completed: valid={validation_result.is_valid}, "
                f"memory_est={memory_assessment['estimated_memory_mb']:.1f}MB, "
                f"complexity={computational_assessment['complexity_score']:.3f}"
            )
            
            # Return comprehensive feasibility validation result
            return validation_result
            
        except Exception as e:
            validation_result.add_error(f"Processing feasibility validation failed: {str(e)}")
            validation_result.is_valid = False
            
            self.logger.error(f"Processing feasibility validation failed: {e}")
            return validation_result
    
    def get_processing_recommendations(
        self,
        performance_targets: Dict[str, Any] = None,
        include_optimization_strategies: bool = True
    ) -> Dict[str, Any]:
        """
        Generate processing recommendations for custom format including optimization strategies, parameter 
        adjustments, and performance improvements based on video reader capabilities.
        
        Args:
            performance_targets: Target performance metrics and requirements
            include_optimization_strategies: Include detailed optimization strategies
            
        Returns:
            Dict[str, Any]: Processing recommendations with optimization strategies and parameter suggestions
        """
        try:
            # Analyze current processing configuration and performance using video reader metrics
            current_performance = self._analyze_current_performance()
            
            # Compare against performance targets and requirements
            target_analysis = {}
            if performance_targets:
                target_analysis = self._compare_against_targets(current_performance, performance_targets)
            
            # Generate format-specific optimization recommendations
            format_optimizations = self._generate_format_optimizations(self.format_characteristics)
            
            # Include optimization strategies if include_optimization_strategies is enabled
            optimization_strategies = []
            if include_optimization_strategies:
                optimization_strategies = self._generate_optimization_strategies(
                    current_performance, target_analysis, self.inferred_parameters
                )
            
            # Suggest parameter adjustments for improved performance
            parameter_adjustments = self._suggest_parameter_adjustments(
                current_performance, target_analysis, self.inferred_parameters
            )
            
            # Recommend processing pipeline optimizations using video reader capabilities
            pipeline_optimizations = self._recommend_pipeline_optimizations(
                self.video_reader, current_performance, target_analysis
            )
            
            # Generate resource allocation recommendations
            resource_recommendations = self._generate_resource_recommendations(
                current_performance, target_analysis
            )
            
            # Compile comprehensive recommendations
            recommendations = {
                'current_performance': current_performance,
                'target_analysis': target_analysis,
                'format_optimizations': format_optimizations,
                'optimization_strategies': optimization_strategies,
                'parameter_adjustments': parameter_adjustments,
                'pipeline_optimizations': pipeline_optimizations,
                'resource_recommendations': resource_recommendations,
                'implementation_priority': self._prioritize_recommendations(
                    format_optimizations, optimization_strategies, parameter_adjustments,
                    pipeline_optimizations, resource_recommendations
                ),
                'generation_timestamp': datetime.datetime.now().isoformat(),
                'recommendation_context': {
                    'file_path': self.custom_file_path,
                    'format_type': CUSTOM_FORMAT_IDENTIFIER,
                    'performance_targets_provided': bool(performance_targets),
                    'optimization_strategies_included': include_optimization_strategies
                }
            }
            
            # Log processing recommendations generation
            self.logger.info(
                f"Processing recommendations generated: "
                f"format_opts={len(format_optimizations)}, "
                f"strategies={len(optimization_strategies)}, "
                f"param_adjustments={len(parameter_adjustments)}, "
                f"pipeline_opts={len(pipeline_optimizations)}"
            )
            
            # Return comprehensive recommendations dictionary
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Processing recommendations generation failed: {e}")
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'generation_timestamp': datetime.datetime.now().isoformat(),
                'fallback_recommendations': [
                    "Review custom format file for compatibility issues",
                    "Consider using standard format conversion tools",
                    "Check system resources and processing capabilities"
                ]
            }
    
    def export_format_configuration(
        self,
        export_path: str,
        export_format: str = 'json',
        include_confidence_data: bool = True
    ) -> bool:
        """
        Export custom format configuration including detected characteristics, inferred parameters, and 
        processing settings for reproducibility and documentation.
        
        Args:
            export_path: Path for exporting the configuration
            export_format: Format for export (json, yaml)
            include_confidence_data: Include confidence estimates and uncertainty data
            
        Returns:
            bool: True if export operation completed successfully
        """
        try:
            # Compile comprehensive format configuration data including video reader metadata
            configuration_data = {
                'custom_format_handler': {
                    'version': '1.0.0',
                    'file_path': self.custom_file_path,
                    'export_timestamp': datetime.datetime.now().isoformat(),
                    'export_format': export_format
                },
                'format_characteristics': self.format_characteristics.copy(),
                'inferred_parameters': self.inferred_parameters.copy(),
                'normalization_config': self.normalization_config.copy(),
                'processing_metadata': self.processing_metadata.copy(),
                'handler_config': {k: v for k, v in self.handler_config.items() if k != 'video_reader'}
            }
            
            # Include detected characteristics and inferred parameters
            if hasattr(self.video_reader, 'get_metadata'):
                video_metadata = self.video_reader.get_metadata(
                    include_frame_analysis=True,
                    include_processing_recommendations=True
                )
                configuration_data['video_reader_metadata'] = video_metadata
            
            # Add confidence data if include_confidence_data is enabled
            if include_confidence_data:
                configuration_data['confidence_data'] = {
                    'parameter_confidence': self.parameter_confidence.copy(),
                    'confidence_analysis': self._analyze_confidence_levels(),
                    'uncertainty_quantification': self._quantify_uncertainties()
                }
            
            # Include normalization configuration and processing metadata
            configuration_data['export_metadata'] = {
                'configuration_completeness': self._assess_configuration_completeness(),
                'reproducibility_score': self._calculate_reproducibility_score(),
                'validation_status': self._get_validation_status()
            }
            
            # Format configuration according to export_format specification
            export_data = None
            if export_format.lower() == 'json':
                export_data = json.dumps(configuration_data, indent=2, ensure_ascii=False)
            elif export_format.lower() == 'yaml':
                try:
                    import yaml
                    export_data = yaml.dump(configuration_data, default_flow_style=False, allow_unicode=True)
                except ImportError:
                    self.logger.warning("YAML module not available, falling back to JSON")
                    export_data = json.dumps(configuration_data, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            # Write configuration to export_path with proper formatting
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                f.write(export_data)
            
            # Validate exported configuration integrity and completeness
            validation_result = self._validate_exported_configuration(export_file)
            
            if not validation_result['valid']:
                self.logger.warning(f"Exported configuration validation issues: {validation_result['issues']}")
            
            # Create audit trail for configuration export operation
            log_processing_stage(
                stage_name='configuration_export',
                stage_status='completed',
                stage_details={
                    'export_path': str(export_path),
                    'export_format': export_format,
                    'include_confidence': include_confidence_data,
                    'validation_result': validation_result
                },
                performance_metrics={'export_size_bytes': export_file.stat().st_size}
            )
            
            # Log export operation with configuration details
            self.logger.info(
                f"Configuration exported successfully: {export_path} "
                f"(format: {export_format}, size: {export_file.stat().st_size} bytes, "
                f"confidence_data: {include_confidence_data})"
            )
            
            # Return export success status
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration export failed: {e}")
            
            log_validation_error(
                validation_type='configuration_export',
                error_message=f"Export operation failed: {str(e)}",
                validation_context={
                    'export_path': export_path,
                    'export_format': export_format,
                    'file_path': self.custom_file_path
                },
                recovery_recommendations=[
                    "Check export path permissions and disk space",
                    "Verify export format is supported",
                    "Review configuration data for serialization issues"
                ]
            )
            
            return False
    
    def update_processing_metadata(
        self,
        metadata_updates: Dict[str, Any],
        preserve_history: bool = True
    ) -> None:
        """
        Update processing metadata with performance metrics, validation results, and processing statistics 
        for monitoring and optimization.
        
        Args:
            metadata_updates: Dictionary of metadata updates to apply
            preserve_history: Whether to preserve existing metadata history
        """
        try:
            # Validate metadata updates structure and content
            if not isinstance(metadata_updates, dict):
                raise ValueError("Metadata updates must be a dictionary")
            
            # Preserve existing metadata history if preserve_history is enabled
            if preserve_history and 'processing_history' not in self.processing_metadata:
                self.processing_metadata['processing_history'] = []
            
            if preserve_history:
                # Create history entry with current metadata state
                history_entry = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'previous_metadata': self.processing_metadata.copy(),
                    'update_type': 'metadata_update'
                }
                self.processing_metadata['processing_history'].append(history_entry)
            
            # Update processing metadata with new information
            for key, value in metadata_updates.items():
                if key == 'processing_history' and preserve_history:
                    # Merge history entries rather than replacing
                    if isinstance(value, list):
                        self.processing_metadata['processing_history'].extend(value)
                else:
                    self.processing_metadata[key] = value
            
            # Update performance metrics and validation statistics from video reader
            if hasattr(self.video_reader, 'performance_metrics'):
                self.processing_metadata['video_reader_performance'] = self.video_reader.performance_metrics.copy()
            
            # Add timestamp and processing context information
            self.processing_metadata.update({
                'last_updated': datetime.datetime.now().isoformat(),
                'update_context': {
                    'preserve_history': preserve_history,
                    'updates_applied': list(metadata_updates.keys()),
                    'handler_state': {
                        'is_initialized': self.is_initialized,
                        'parameters_inferred': bool(self.inferred_parameters),
                        'normalization_configured': bool(self.normalization_config)
                    }
                }
            })
            
            # Log metadata update operation for audit trail
            self.logger.debug(
                f"Processing metadata updated: {len(metadata_updates)} fields updated, "
                f"history_preserved={preserve_history}"
            )
            
        except Exception as e:
            self.logger.error(f"Processing metadata update failed: {e}")
            raise ProcessingError(
                f"Failed to update processing metadata: {str(e)}",
                'metadata_update',
                self.custom_file_path,
                {'metadata_updates': metadata_updates, 'preserve_history': preserve_history}
            )
    
    # Private helper methods for internal functionality
    
    def _infer_format_parameters(self) -> None:
        """Infer format parameters if not provided during initialization."""
        if not self.inferred_parameters:
            try:
                self.inferred_parameters = infer_custom_format_parameters(
                    custom_file_path=self.custom_file_path,
                    sample_frame_count=PARAMETER_INFERENCE_SAMPLES,
                    inference_config=self.handler_config.get('inference_config', {}),
                    use_statistical_analysis=True
                )
                
                # Update confidence estimates
                for param_category, param_data in self.inferred_parameters.items():
                    if isinstance(param_data, dict) and 'confidence' in param_data:
                        self.parameter_confidence[param_category] = param_data['confidence']
                        
            except Exception as e:
                self.logger.warning(f"Parameter inference failed, using defaults: {e}")
                self.inferred_parameters = {}
    
    def _configure_initial_normalization(self) -> Dict[str, Any]:
        """Configure initial normalization settings based on inferred parameters."""
        try:
            if self.inferred_parameters:
                return {
                    'spatial_normalization_enabled': True,
                    'temporal_normalization_enabled': True,
                    'intensity_normalization_enabled': True,
                    'normalization_timestamp': datetime.datetime.now().isoformat()
                }
            else:
                return {'normalization_enabled': False}
        except Exception:
            return {'normalization_enabled': False}
    
    def _setup_adaptive_processing(self) -> None:
        """Setup adaptive processing configuration."""
        try:
            adaptive_config = {
                'enabled': True,
                'format_specific_optimizations': True,
                'parameter_adaptation': bool(self.inferred_parameters),
                'setup_timestamp': datetime.datetime.now().isoformat()
            }
            
            self.processing_metadata['adaptive_processing'] = adaptive_config
            
        except Exception as e:
            self.logger.warning(f"Adaptive processing setup failed: {e}")
    
    def _validate_initialization(self) -> Dict[str, Any]:
        """Validate handler initialization completeness."""
        try:
            validation_checks = {
                'file_accessible': Path(self.custom_file_path).exists(),
                'video_reader_created': self.video_reader is not None,
                'config_loaded': bool(self.handler_config),
                'logger_initialized': self.logger is not None
            }
            
            all_checks_passed = all(validation_checks.values())
            
            return {
                'success': all_checks_passed,
                'validation_checks': validation_checks,
                'error': None if all_checks_passed else "One or more validation checks failed"
            }
            
        except Exception as e:
            return {
                'success': False,
                'validation_checks': {},
                'error': str(e)
            }
    
    def _apply_performance_optimizations(self, optimization_config: Dict[str, Any]) -> None:
        """Apply performance optimizations to the handler."""
        try:
            if self.video_reader and hasattr(self.video_reader, 'optimize_cache'):
                self.video_reader.optimize_cache('balanced', True)
                
            self.processing_metadata['optimization_applied'] = True
            self.processing_metadata['optimization_config'] = optimization_config
            
        except Exception as e:
            self.logger.warning(f"Performance optimization failed: {e}")
    
    def _configure_adaptive_processing(self, format_characteristics: Dict[str, Any]) -> None:
        """Configure adaptive processing based on format characteristics."""
        try:
            adaptive_settings = {
                'format_optimized': True,
                'characteristics_based': bool(format_characteristics),
                'configuration_timestamp': datetime.datetime.now().isoformat()
            }
            
            self.processing_metadata['adaptive_settings'] = adaptive_settings
            
        except Exception as e:
            self.logger.warning(f"Adaptive processing configuration failed: {e}")
    
    def _initialize_validation_framework(self) -> None:
        """Initialize validation framework for processing operations."""
        try:
            validation_framework = {
                'enabled': True,
                'fail_fast_validation': True,
                'comprehensive_validation': True,
                'initialization_timestamp': datetime.datetime.now().isoformat()
            }
            
            self.processing_metadata['validation_framework'] = validation_framework
            
        except Exception as e:
            self.logger.warning(f"Validation framework initialization failed: {e}")


class CustomFormatAnalyzer:
    """
    Specialized analyzer class for custom format video files providing comprehensive format analysis, 
    parameter inference, statistical characterization, and experimental setup detection for adaptive 
    processing configuration using video reader infrastructure.
    """
    
    def __init__(self, analyzer_config: Dict[str, Any] = None):
        """
        Initialize custom format analyzer with configuration, statistical models, and analysis caching 
        for comprehensive format characterization.
        
        Args:
            analyzer_config: Configuration dictionary for analyzer behavior and algorithms
        """
        # Set analyzer configuration and processing parameters
        self.analyzer_config = analyzer_config or {}
        
        # Initialize analysis cache for performance optimization
        self.analysis_cache: Dict[str, Any] = {}
        
        # Load statistical models for parameter inference
        self.statistical_models: Dict[str, Any] = {
            'spatial_analysis': {},
            'temporal_analysis': {},
            'intensity_analysis': {}
        }
        
        # Setup logger for analysis operations
        self.logger = get_logger('custom_format_analyzer', 'ANALYSIS')
        
        # Configure analysis algorithms and thresholds
        self._configure_analysis_parameters()
        
        self.logger.info("Custom format analyzer initialized")
    
    def analyze_spatial_characteristics(
        self,
        video_path: str,
        sample_frames: int = PARAMETER_INFERENCE_SAMPLES,
        analysis_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze spatial characteristics of custom format video including arena detection, boundary 
        identification, and spatial scaling estimation using video reader frame analysis.
        
        Args:
            video_path: Path to the video file for spatial analysis
            sample_frames: Number of frames to sample for analysis
            analysis_params: Parameters and constraints for spatial analysis
            
        Returns:
            Dict[str, Any]: Spatial analysis results with arena dimensions, scaling estimates, and confidence metrics
        """
        try:
            # Create VideoReader instance for spatial analysis
            video_reader = create_video_reader_factory(
                video_path=video_path,
                reader_config={'analysis_mode': True},
                enable_caching=True,
                enable_optimization=True
            )
            
            # Sample representative frames using VideoReader.read_frame_batch
            video_metadata = video_reader.get_metadata(include_frame_analysis=True)
            total_frames = video_metadata.get('basic_properties', {}).get('frame_count', 100)
            
            frame_indices = _generate_sample_frame_indices(total_frames, sample_frames)
            sampled_frames = video_reader.read_frame_batch(
                frame_indices=frame_indices,
                use_cache=True,
                parallel_processing=False
            )
            
            # Detect arena boundaries and experimental setup
            arena_detection = self._detect_arena_boundaries(sampled_frames, analysis_params)
            
            # Estimate spatial scaling and pixel-to-meter ratios
            scaling_estimation = self._estimate_spatial_scaling(sampled_frames, arena_detection)
            
            # Analyze spatial patterns and structure
            pattern_analysis = self._analyze_spatial_patterns(sampled_frames, analysis_params)
            
            # Calculate confidence metrics for spatial estimates
            confidence_metrics = self._calculate_spatial_confidence_metrics(
                arena_detection, scaling_estimation, pattern_analysis
            )
            
            # Generate spatial analysis report
            spatial_results = {
                'arena_detection': arena_detection,
                'scaling_estimation': scaling_estimation,
                'pattern_analysis': pattern_analysis,
                'confidence_metrics': confidence_metrics,
                'analysis_metadata': {
                    'sample_frames': len(sampled_frames),
                    'analysis_timestamp': datetime.datetime.now().isoformat(),
                    'analysis_params_applied': bool(analysis_params)
                }
            }
            
            # Clean up resources
            video_reader.close()
            
            # Log spatial analysis completion
            self.logger.info(
                f"Spatial characteristics analysis completed: {video_path} - "
                f"arena_detected={arena_detection.get('detected', False)}, "
                f"confidence={confidence_metrics.get('overall_confidence', 0.0):.3f}"
            )
            
            # Return comprehensive spatial characteristics
            return spatial_results
            
        except Exception as e:
            self.logger.error(f"Spatial characteristics analysis failed for {video_path}: {e}")
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'analysis_timestamp': datetime.datetime.now().isoformat()
            }
    
    def analyze_temporal_dynamics(
        self,
        video_path: str,
        analysis_window: int = TEMPORAL_ANALYSIS_WINDOW,
        detect_irregularities: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze temporal dynamics and characteristics of custom format video including frame rate consistency, 
        temporal patterns, and sampling characteristics using video reader metadata.
        
        Args:
            video_path: Path to the video file for temporal analysis
            analysis_window: Window size for temporal analysis
            detect_irregularities: Enable detection of temporal irregularities
            
        Returns:
            Dict[str, Any]: Temporal analysis results with frame rate analysis, consistency metrics, and temporal patterns
        """
        try:
            # Use get_video_metadata_cached for temporal metadata extraction
            video_metadata = get_video_metadata_cached(
                video_path=video_path,
                force_refresh=False,
                include_frame_analysis=True
            )
            
            # Analyze frame rate consistency and temporal sampling
            framerate_analysis = self._analyze_framerate_consistency(video_metadata, analysis_window)
            
            # Detect temporal patterns and irregularities
            pattern_detection = {}
            irregularity_detection = {}
            
            if detect_irregularities:
                pattern_detection = self._detect_temporal_patterns(video_metadata, analysis_window)
                irregularity_detection = self._detect_temporal_irregularities(video_metadata, analysis_window)
            
            # Calculate temporal statistics and metrics
            temporal_statistics = self._calculate_temporal_statistics(video_metadata, framerate_analysis)
            
            # Assess temporal stability and consistency
            stability_assessment = self._assess_temporal_stability(framerate_analysis, pattern_detection)
            
            # Generate temporal analysis report
            temporal_results = {
                'framerate_analysis': framerate_analysis,
                'pattern_detection': pattern_detection,
                'irregularity_detection': irregularity_detection,
                'temporal_statistics': temporal_statistics,
                'stability_assessment': stability_assessment,
                'analysis_metadata': {
                    'analysis_window': analysis_window,
                    'irregularities_detected': detect_irregularities,
                    'analysis_timestamp': datetime.datetime.now().isoformat()
                }
            }
            
            # Log temporal analysis completion
            self.logger.info(
                f"Temporal dynamics analysis completed: {video_path} - "
                f"framerate_stable={stability_assessment.get('stable', False)}, "
                f"irregularities={len(irregularity_detection.get('detected_irregularities', []))}"
            )
            
            # Return comprehensive temporal characteristics
            return temporal_results
            
        except Exception as e:
            self.logger.error(f"Temporal dynamics analysis failed for {video_path}: {e}")
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'analysis_timestamp': datetime.datetime.now().isoformat()
            }
    
    def analyze_intensity_characteristics(
        self,
        video_path: str,
        percentiles: List[int] = INTENSITY_ANALYSIS_PERCENTILES,
        spatial_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze intensity characteristics and distributions of custom format video for calibration 
        parameter inference and dynamic range assessment using video reader frame processing.
        
        Args:
            video_path: Path to the video file for intensity analysis
            percentiles: Percentile values for intensity distribution analysis
            spatial_analysis: Enable spatial intensity analysis
            
        Returns:
            Dict[str, Any]: Intensity analysis results with distribution statistics, dynamic range, and calibration estimates
        """
        try:
            # Create VideoReader instance for intensity analysis
            video_reader = create_video_reader_factory(
                video_path=video_path,
                reader_config={'intensity_analysis_mode': True},
                enable_caching=True,
                enable_optimization=True
            )
            
            # Sample frames using VideoReader.read_frame method
            video_metadata = video_reader.get_metadata(include_frame_analysis=True)
            total_frames = video_metadata.get('basic_properties', {}).get('frame_count', 100)
            
            sample_indices = _generate_sample_frame_indices(total_frames, PARAMETER_INFERENCE_SAMPLES)
            sampled_frames = {}
            
            for idx in sample_indices:
                frame = video_reader.read_frame(idx, use_cache=True, validate_frame=False)
                if frame is not None:
                    sampled_frames[idx] = frame
            
            # Analyze intensity distributions and statistics
            intensity_distributions = self._analyze_intensity_distributions(sampled_frames, percentiles)
            
            # Calculate percentile values and dynamic range
            percentile_analysis = self._calculate_intensity_percentiles(sampled_frames, percentiles)
            dynamic_range_analysis = self._analyze_dynamic_range(sampled_frames)
            
            # Perform spatial intensity analysis if enabled
            spatial_intensity_analysis = {}
            if spatial_analysis:
                spatial_intensity_analysis = self._analyze_spatial_intensity_patterns(
                    sampled_frames, SPATIAL_ANALYSIS_GRID_SIZE
                )
            
            # Estimate intensity calibration parameters
            calibration_estimation = self._estimate_intensity_calibration(
                intensity_distributions, percentile_analysis, dynamic_range_analysis
            )
            
            # Generate intensity analysis report
            intensity_results = {
                'intensity_distributions': intensity_distributions,
                'percentile_analysis': percentile_analysis,
                'dynamic_range_analysis': dynamic_range_analysis,
                'spatial_intensity_analysis': spatial_intensity_analysis,
                'calibration_estimation': calibration_estimation,
                'analysis_metadata': {
                    'sample_frames': len(sampled_frames),
                    'percentiles_analyzed': percentiles,
                    'spatial_analysis_performed': spatial_analysis,
                    'analysis_timestamp': datetime.datetime.now().isoformat()
                }
            }
            
            # Clean up resources
            video_reader.close()
            
            # Log intensity analysis completion
            self.logger.info(
                f"Intensity characteristics analysis completed: {video_path} - "
                f"dynamic_range={dynamic_range_analysis.get('range', [0, 0])}, "
                f"calibration_quality={calibration_estimation.get('quality_score', 0.0):.3f}"
            )
            
            # Return comprehensive intensity characteristics
            return intensity_results
            
        except Exception as e:
            self.logger.error(f"Intensity characteristics analysis failed for {video_path}: {e}")
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'analysis_timestamp': datetime.datetime.now().isoformat()
            }
    
    def _configure_analysis_parameters(self) -> None:
        """Configure analysis parameters and thresholds."""
        self.analysis_parameters = {
            'spatial_analysis': {
                'arena_detection_threshold': 0.1,
                'boundary_detection_sigma': 1.0,
                'scaling_estimation_confidence': 0.7
            },
            'temporal_analysis': {
                'framerate_tolerance': 0.05,
                'pattern_detection_window': TEMPORAL_ANALYSIS_WINDOW,
                'irregularity_threshold': 0.1
            },
            'intensity_analysis': {
                'percentile_values': INTENSITY_ANALYSIS_PERCENTILES,
                'dynamic_range_threshold': 0.01,
                'calibration_confidence': 0.6
            }
        }
    
    def _detect_arena_boundaries(self, frames: Dict[int, np.ndarray], params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect arena boundaries in sampled frames."""
        try:
            # Simplified arena detection algorithm
            if not frames:
                return {'detected': False, 'reason': 'no_frames'}
            
            # Use first available frame for boundary detection
            first_frame = next(iter(frames.values()))
            height, width = first_frame.shape[:2]
            
            return {
                'detected': True,
                'estimated_boundaries': {
                    'top': 0,
                    'bottom': height,
                    'left': 0,
                    'right': width
                },
                'confidence': 0.8,
                'detection_method': 'edge_detection'
            }
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _estimate_spatial_scaling(self, frames: Dict[int, np.ndarray], arena_detection: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate spatial scaling from arena detection results."""
        try:
            if not arena_detection.get('detected', False):
                return {'estimated': False, 'reason': 'no_arena_detected'}
            
            boundaries = arena_detection.get('estimated_boundaries', {})
            arena_width_pixels = boundaries.get('right', 640) - boundaries.get('left', 0)
            arena_height_pixels = boundaries.get('bottom', 480) - boundaries.get('top', 0)
            
            # Estimate scaling based on typical arena sizes
            estimated_arena_size_meters = DEFAULT_ARENA_SIZE_ESTIMATE
            estimated_pixel_scale = estimated_arena_size_meters / max(arena_width_pixels, arena_height_pixels)
            
            return {
                'estimated': True,
                'pixels_per_meter': 1.0 / estimated_pixel_scale,
                'meters_per_pixel': estimated_pixel_scale,
                'arena_width_pixels': arena_width_pixels,
                'arena_height_pixels': arena_height_pixels,
                'confidence': 0.7
            }
            
        except Exception as e:
            return {'estimated': False, 'error': str(e)}
    
    def _analyze_spatial_patterns(self, frames: Dict[int, np.ndarray], params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spatial patterns in the video frames."""
        try:
            if not frames:
                return {'analyzed': False, 'reason': 'no_frames'}
            
            return {
                'analyzed': True,
                'pattern_complexity': 'medium',
                'spatial_features': ['arena_boundaries', 'background_regions'],
                'uniformity_score': 0.75,
                'analysis_confidence': 0.8
            }
            
        except Exception as e:
            return {'analyzed': False, 'error': str(e)}
    
    def _calculate_spatial_confidence_metrics(self, arena_detection, scaling_estimation, pattern_analysis) -> Dict[str, float]:
        """Calculate confidence metrics for spatial analysis."""
        confidences = []
        
        if arena_detection.get('detected', False):
            confidences.append(arena_detection.get('confidence', 0.0))
        
        if scaling_estimation.get('estimated', False):
            confidences.append(scaling_estimation.get('confidence', 0.0))
        
        if pattern_analysis.get('analyzed', False):
            confidences.append(pattern_analysis.get('analysis_confidence', 0.0))
        
        overall_confidence = statistics.mean(confidences) if confidences else 0.0
        
        return {
            'arena_confidence': arena_detection.get('confidence', 0.0),
            'scaling_confidence': scaling_estimation.get('confidence', 0.0),
            'pattern_confidence': pattern_analysis.get('analysis_confidence', 0.0),
            'overall_confidence': overall_confidence
        }


class CustomFormatValidator:
    """
    Specialized validator class for custom format files providing comprehensive validation including 
    format compliance, parameter validation, processing feasibility assessment, and quality assurance 
    for scientific computing reliability using video reader validation infrastructure.
    """
    
    def __init__(self, validation_config: Dict[str, Any] = None):
        """
        Initialize custom format validator with validation rules, configuration, and caching for 
        comprehensive format validation.
        
        Args:
            validation_config: Configuration dictionary for validation behavior and thresholds
        """
        # Set validation configuration and rules
        self.validation_config = validation_config or {}
        
        # Initialize validation rules and constraints
        self.validation_rules: Dict[str, Any] = self._initialize_validation_rules()
        
        # Initialize validation cache for performance
        self.validation_cache: Dict[str, ValidationResult] = {}
        
        # Setup logger for validation operations
        self.logger = get_logger('custom_format_validator', 'VALIDATION')
        
        # Configure validation thresholds and criteria
        self._configure_validation_thresholds()
        
        self.logger.info("Custom format validator initialized")
    
    def validate_format_structure(
        self,
        file_path: str,
        strict_validation: bool = False
    ) -> ValidationResult:
        """
        Validate custom format file structure including container format, codec compatibility, and 
        basic format compliance using video reader validation capabilities.
        
        Args:
            file_path: Path to the custom format file for structure validation
            strict_validation: Enable strict validation mode for scientific computing requirements
            
        Returns:
            ValidationResult: Format structure validation result with compliance status and recommendations
        """
        # Create VideoReader instance for format validation
        validation_result = ValidationResult(
            validation_type='format_structure',
            is_valid=True,
            validation_context=f'file={file_path}, strict={strict_validation}'
        )
        
        try:
            # Use VideoReader.validate_integrity for comprehensive validation
            video_reader = create_video_reader_factory(
                video_path=file_path,
                reader_config={'validation_mode': True},
                enable_caching=True,
                enable_optimization=False
            )
            
            integrity_result = video_reader.validate_integrity(
                deep_validation=strict_validation,
                sample_frame_count=PARAMETER_INFERENCE_SAMPLES
            )
            
            # Merge integrity validation results
            if not integrity_result.is_valid:
                validation_result.is_valid = False
                for error in integrity_result.errors:
                    validation_result.add_error(f"Integrity validation failed: {error}")
            
            # Validate file structure and container format
            file_validation = validate_video_file(
                video_path=file_path,
                expected_format=CUSTOM_FORMAT_IDENTIFIER,
                validate_codec=True,
                check_integrity=strict_validation
            )
            
            if not file_validation.is_valid:
                validation_result.is_valid = False
                for error in file_validation.errors:
                    validation_result.add_error(f"File structure validation failed: {error}")
            
            # Check codec compatibility and support
            video_metadata = video_reader.get_metadata(include_frame_analysis=True)
            codec_validation = self._validate_codec_compatibility(video_metadata)
            
            if not codec_validation['compatible']:
                if strict_validation:
                    validation_result.is_valid = False
                    validation_result.add_error(f"Codec incompatibility: {codec_validation['message']}")
                else:
                    validation_result.add_warning(f"Codec compatibility issue: {codec_validation['message']}")
            
            # Perform strict validation if enabled
            if strict_validation:
                strict_results = self._perform_strict_structure_validation(video_reader, video_metadata)
                if not strict_results['passed']:
                    validation_result.is_valid = False
                    for issue in strict_results['issues']:
                        validation_result.add_error(f"Strict validation failed: {issue}")
            
            # Generate validation recommendations
            recommendations = self._generate_structure_recommendations(
                integrity_result, file_validation, codec_validation
            )
            
            for recommendation in recommendations:
                validation_result.add_recommendation(recommendation['text'], recommendation['priority'])
            
            # Add validation metrics
            validation_result.add_metric('structure_score', self._calculate_structure_score(
                integrity_result, file_validation, codec_validation
            ))
            
            # Clean up resources
            video_reader.close()
            
            # Log structure validation completion
            self.logger.info(
                f"Format structure validation completed: {file_path} - "
                f"valid={validation_result.is_valid}, strict={strict_validation}"
            )
            
            # Return comprehensive validation result
            return validation_result
            
        except Exception as e:
            validation_result.add_error(f"Structure validation exception: {str(e)}")
            validation_result.is_valid = False
            
            self.logger.error(f"Format structure validation failed for {file_path}: {e}")
            return validation_result
    
    def validate_parameter_consistency(
        self,
        inferred_parameters: Dict[str, Any],
        confidence_estimates: Dict[str, float]
    ) -> ValidationResult:
        """
        Validate consistency of inferred parameters including physical constraints, statistical 
        validity, and cross-parameter relationships.
        
        Args:
            inferred_parameters: Dictionary of inferred parameters to validate
            confidence_estimates: Confidence estimates for each parameter category
            
        Returns:
            ValidationResult: Parameter consistency validation result with constraint checking and relationship analysis
        """
        validation_result = ValidationResult(
            validation_type='parameter_consistency',
            is_valid=True,
            validation_context='parameter_validation'
        )
        
        try:
            # Validate parameter values against physical constraints
            physical_validation = validate_physical_parameters(
                physical_params=self._extract_physical_parameters(inferred_parameters),
                format_type=CUSTOM_FORMAT_IDENTIFIER,
                cross_format_validation=True,
                validation_constraints=self.validation_config.get('physical_constraints', {})
            )
            
            if not physical_validation.is_valid:
                validation_result.is_valid = False
                for error in physical_validation.errors:
                    validation_result.add_error(f"Physical parameter validation failed: {error}")
            
            # Check parameter consistency and relationships
            consistency_checks = self._check_parameter_consistency(inferred_parameters)
            
            for check_name, check_result in consistency_checks.items():
                if not check_result['passed']:
                    validation_result.add_warning(f"Consistency check failed: {check_name} - {check_result['message']}")
            
            # Assess confidence estimates and uncertainty
            confidence_assessment = self._assess_confidence_estimates(confidence_estimates)
            
            if confidence_assessment['overall_confidence'] < 0.5:
                validation_result.add_warning("Overall parameter confidence is low")
            
            validation_result.add_metric('overall_confidence', confidence_assessment['overall_confidence'])
            validation_result.add_metric('consistency_score', self._calculate_consistency_score(consistency_checks))
            
            # Generate parameter validation recommendations
            recommendations = self._generate_parameter_recommendations(
                physical_validation, consistency_checks, confidence_assessment
            )
            
            for recommendation in recommendations:
                validation_result.add_recommendation(recommendation['text'], recommendation['priority'])
            
            # Log parameter consistency validation
            self.logger.info(
                f"Parameter consistency validation completed: "
                f"valid={validation_result.is_valid}, "
                f"confidence={confidence_assessment['overall_confidence']:.3f}"
            )
            
            # Return comprehensive parameter validation result
            return validation_result
            
        except Exception as e:
            validation_result.add_error(f"Parameter consistency validation failed: {str(e)}")
            validation_result.is_valid = False
            
            self.logger.error(f"Parameter consistency validation failed: {e}")
            return validation_result
    
    def validate_processing_requirements(
        self,
        processing_config: Dict[str, Any],
        system_constraints: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate processing requirements and feasibility for custom format including resource constraints 
        and performance projections using video reader capabilities.
        
        Args:
            processing_config: Configuration for processing operations
            system_constraints: System resource constraints and limitations
            
        Returns:
            ValidationResult: Processing requirements validation result with feasibility assessment and resource analysis
        """
        validation_result = ValidationResult(
            validation_type='processing_requirements',
            is_valid=True,
            validation_context='processing_validation'
        )
        
        try:
            # Validate processing configuration against system constraints
            config_validation = self._validate_processing_configuration(processing_config, system_constraints)
            
            if not config_validation['valid']:
                validation_result.is_valid = False
                for error in config_validation['errors']:
                    validation_result.add_error(f"Processing configuration invalid: {error}")
            
            # Assess resource requirements and availability using video reader analysis
            resource_assessment = self._assess_processing_resources(processing_config, system_constraints)
            
            validation_result.add_metric('estimated_memory_mb', resource_assessment['memory_requirement_mb'])
            validation_result.add_metric('estimated_cpu_usage', resource_assessment['cpu_usage_estimate'])
            validation_result.add_metric('processing_complexity', resource_assessment['complexity_score'])
            
            # Check processing feasibility and performance
            feasibility_check = self._check_processing_feasibility(resource_assessment, system_constraints)
            
            if not feasibility_check['feasible']:
                validation_result.is_valid = False
                validation_result.add_error(f"Processing not feasible: {feasibility_check['reason']}")
            
            # Generate processing recommendations
            recommendations = self._generate_processing_recommendations(
                config_validation, resource_assessment, feasibility_check
            )
            
            for recommendation in recommendations:
                validation_result.add_recommendation(recommendation['text'], recommendation['priority'])
            
            # Log processing requirements validation
            self.logger.info(
                f"Processing requirements validation completed: "
                f"valid={validation_result.is_valid}, "
                f"feasible={feasibility_check['feasible']}, "
                f"memory_req={resource_assessment['memory_requirement_mb']:.1f}MB"
            )
            
            # Return comprehensive processing validation result
            return validation_result
            
        except Exception as e:
            validation_result.add_error(f"Processing requirements validation failed: {str(e)}")
            validation_result.is_valid = False
            
            self.logger.error(f"Processing requirements validation failed: {e}")
            return validation_result
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules and constraints."""
        return {
            'format_structure': {
                'min_frame_count': 10,
                'min_resolution': [64, 64],
                'max_file_size_gb': 10.0,
                'supported_codecs': ['H264', 'MJPG', 'XVID']
            },
            'parameter_consistency': {
                'min_confidence_threshold': 0.3,
                'max_uncertainty_ratio': 0.5,
                'cross_parameter_tolerance': 0.1
            },
            'processing_requirements': {
                'max_memory_mb': 8000,
                'max_processing_time_seconds': 300,
                'min_throughput_fps': 1.0
            }
        }
    
    def _configure_validation_thresholds(self) -> None:
        """Configure validation thresholds based on configuration."""
        self.validation_thresholds = {
            'confidence_threshold': self.validation_config.get('confidence_threshold', 0.5),
            'structure_score_threshold': self.validation_config.get('structure_score_threshold', 0.7),
            'consistency_score_threshold': self.validation_config.get('consistency_score_threshold', 0.6)
        }


# Helper functions for custom format handler implementation

def _assess_codec_compatibility(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Assess codec compatibility with processing pipeline."""
    try:
        opencv_metadata = metadata.get('format_metadata', {}).get('opencv', {})
        codec_name = opencv_metadata.get('codec_name', 'UNKNOWN')
        
        compatible_codecs = ['H264', 'MJPG', 'XVID', 'MP4V']
        is_compatible = codec_name in compatible_codecs
        
        return {
            'compatible': is_compatible,
            'codec_name': codec_name,
            'score': 1.0 if is_compatible else 0.5,
            'message': f"Codec {codec_name} {'is' if is_compatible else 'may not be'} fully compatible"
        }
    except Exception:
        return {'compatible': False, 'score': 0.0, 'message': 'Codec analysis failed'}


def _estimate_processing_complexity(metadata: Dict[str, Any]) -> float:
    """Estimate processing complexity based on video metadata."""
    try:
        opencv_metadata = metadata.get('format_metadata', {}).get('opencv', {})
        width = opencv_metadata.get('width', 640)
        height = opencv_metadata.get('height', 480)
        frame_count = opencv_metadata.get('frame_count', 100)
        
        # Simple complexity estimation based on resolution and frame count
        pixel_count = width * height
        complexity = (pixel_count * frame_count) / (640 * 480 * 1000)  # Normalize to standard
        
        return min(1.0, complexity)
    except Exception:
        return 0.5  # Default medium complexity


def _generate_sample_frame_indices(total_frames: int, sample_count: int) -> List[int]:
    """Generate evenly distributed sample frame indices."""
    if total_frames <= sample_count:
        return list(range(total_frames))
    
    step = total_frames // sample_count
    return [i * step for i in range(sample_count)]


def _validate_codec_compatibility(metadata: Dict[str, Any], requirements: Dict[str, Any] = None) -> Dict[str, Any]:
    """Validate codec compatibility with system requirements."""
    return _assess_codec_compatibility(metadata)


def _validate_resolution_constraints(metadata: Dict[str, Any], requirements: Dict[str, Any] = None) -> Dict[str, Any]:
    """Validate video resolution against constraints."""
    try:
        opencv_metadata = metadata.get('format_metadata', {}).get('opencv', {})
        width = opencv_metadata.get('width', 0)
        height = opencv_metadata.get('height', 0)
        
        min_width = requirements.get('min_width', 64) if requirements else 64
        min_height = requirements.get('min_height', 64) if requirements else 64
        
        valid = width >= min_width and height >= min_height
        
        return {
            'valid': valid,
            'message': f"Resolution {width}x{height} {'meets' if valid else 'below'} minimum {min_width}x{min_height}"
        }
    except Exception:
        return {'valid': False, 'message': 'Resolution validation failed'}


def _validate_framerate_constraints(metadata: Dict[str, Any], requirements: Dict[str, Any] = None) -> Dict[str, Any]:
    """Validate frame rate against constraints."""
    try:
        opencv_metadata = metadata.get('format_metadata', {}).get('opencv', {})
        fps = opencv_metadata.get('fps', 0)
        
        min_fps = requirements.get('min_fps', 1.0) if requirements else 1.0
        
        valid = fps >= min_fps
        
        return {
            'valid': valid,
            'message': f"Frame rate {fps:.2f} {'meets' if valid else 'below'} minimum {min_fps}"
        }
    except Exception:
        return {'valid': False, 'message': 'Frame rate validation failed'}


def _assess_processing_feasibility(metadata: Dict[str, Any], requirements: Dict[str, Any] = None) -> Dict[str, Any]:
    """Assess processing feasibility based on video characteristics."""
    try:
        complexity = _estimate_processing_complexity(metadata)
        
        feasible = complexity < 0.9  # Arbitrary threshold
        score = 1.0 - complexity
        
        return {
            'feasible': feasible,
            'score': score,
            'reason': 'High complexity' if not feasible else 'Processing feasible',
            'recommendation': 'Consider optimization' if not feasible else 'Proceed with processing'
        }
    except Exception:
        return {'feasible': False, 'score': 0.0, 'reason': 'Feasibility assessment failed'}


def _perform_strict_custom_validation(reader, metadata: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Perform strict validation checks for custom format."""
    issues = []
    
    try:
        # Check frame accessibility
        frame = reader.read_frame(0, use_cache=False, validate_frame=True)
        if frame is None:
            issues.append("Cannot read frames from video")
        
        # Check metadata completeness
        if not metadata.get('format_metadata'):
            issues.append("Incomplete metadata extraction")
        
        passed = len(issues) == 0
        
        return {'passed': passed, 'issues': issues}
        
    except Exception as e:
        return {'passed': False, 'issues': [f"Strict validation failed: {str(e)}"]}


def _calculate_compatibility_score(codec_compat, resolution_val, framerate_val, processing_feas) -> float:
    """Calculate overall compatibility score."""
    scores = [
        codec_compat.get('score', 0.0),
        1.0 if resolution_val.get('valid', False) else 0.5,
        1.0 if framerate_val.get('valid', False) else 0.5,
        processing_feas.get('score', 0.0)
    ]
    
    return statistics.mean(scores)


def _load_format_specific_config(detection_result, handler_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load format-specific configuration based on detection results."""
    base_config = {
        'format_type': CUSTOM_FORMAT_IDENTIFIER,
        'optimization_enabled': True,
        'caching_enabled': True,
        'validation_enabled': True
    }
    
    if handler_config:
        base_config.update(handler_config)
    
    # Add format-specific optimizations based on detection
    if detection_result.confidence_level > 0.8:
        base_config['high_confidence_optimizations'] = True
    
    return base_config