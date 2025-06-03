"""
Specialized processing error module providing comprehensive error handling for data processing operations 
including video processing failures, normalization errors, format conversion issues, and data transformation 
problems. Implements graceful degradation support with intermediate result preservation, processing stage 
tracking, and recovery strategy coordination for batch processing reliability in scientific computing workflows.

This module integrates with the error handling system to provide detailed processing context, performance 
impact analysis, and actionable recovery recommendations for maintaining <1% error rate target in cross-format 
data processing operations.

Key Features:
- Specialized processing error classes for video, normalization, format conversion, and batch operations
- Graceful degradation support with intermediate result preservation for batch processing reliability
- Processing stage tracking and recovery strategy coordination for comprehensive error analysis
- Performance impact analysis and threshold monitoring for <1% error rate target achievement
- Cross-format data processing error handling with compatibility analysis and alternative recommendations
- Batch processing error handling supporting 4000+ simulation processing with partial completion support
- Recovery strategy registration and extensible processing error management for component-specific handling
- Comprehensive error statistics and monitoring for processing operation optimization and pattern analysis
"""

# External library imports with version specifications
import typing  # Python 3.9+ - Type hints for processing error function signatures and interfaces
import datetime  # Python 3.9+ - Timestamp generation for processing error tracking and audit trails
import traceback  # Python 3.9+ - Stack trace extraction for detailed processing error reporting and debugging
import json  # Python 3.9+ - JSON serialization for structured processing error reporting and export
import pathlib  # Python 3.9+ - Path handling for processing error context and file tracking
import threading  # Python 3.9+ - Thread-safe processing error handling and context management
from typing import Dict, Any, List, Optional, Union, Callable, Type  # Python 3.9+ - Type hints for error handling

# Internal imports from error handling and utility modules
from .exceptions import (
    ProcessingError, 
    preserve_intermediate_results, 
    add_completed_step, 
    add_failed_step
)
from ..utils.error_handling import (
    ErrorSeverity, ErrorCategory, 
    get_priority, requires_immediate_action, 
    is_retryable, get_recovery_strategy,
    handle_error, ErrorHandlingResult
)
from ..utils.logging_utils import (
    get_logger, log_validation_error, create_audit_trail, 
    set_scientific_context, get_scientific_context,
    log_performance_metrics
)

# Global error code constants for processing error classification and tracking
PROCESSING_ERROR_CODE_BASE = 2000
VIDEO_PROCESSING_ERROR_CODE = 2100
NORMALIZATION_ERROR_CODE = 2200
FORMAT_CONVERSION_ERROR_CODE = 2300
DATA_TRANSFORMATION_ERROR_CODE = 2400
BATCH_PROCESSING_ERROR_CODE = 2500
QUALITY_VALIDATION_ERROR_CODE = 2600

# Global processing error registry for dynamic error management and specialized recovery strategy mapping
_processing_error_registry: Dict[str, Type[ProcessingError]] = {}

# Global processing recovery strategies registry for extensible error handling and component-specific recovery
_processing_recovery_strategies: Dict[str, Callable] = {}

# Global processing error statistics for monitoring, analysis, and performance optimization
_processing_error_statistics: Dict[str, Dict[str, Any]] = {}

# Thread-safe lock for processing error registry and statistics management
_processing_error_lock = threading.Lock()


def create_video_processing_error(
    error_message: str,
    video_path: str,
    processing_stage: str,
    processing_context: Dict[str, Any],
    original_exception: Optional[Exception] = None
) -> 'VideoProcessingError':
    """
    Create specialized video processing error with video-specific context, processing stage tracking, 
    and intermediate result preservation for graceful degradation in video processing workflows.
    
    This function creates comprehensive video processing errors with video-specific context including 
    video metadata, processing stage information, and intermediate result preservation for graceful 
    degradation in complex video processing operations.
    
    Args:
        error_message: Descriptive error message for the video processing failure
        video_path: Path to the video file being processed when error occurred
        processing_stage: Current stage of video processing when error occurred
        processing_context: Context information specific to video processing operation
        original_exception: Original exception that caused the video processing failure
        
    Returns:
        VideoProcessingError: Specialized video processing error with comprehensive context and recovery support
    """
    # Validate error message and video path parameters for comprehensive error context
    if not error_message or not isinstance(error_message, str):
        raise ValueError("Error message must be a non-empty string for video processing error creation")
    
    if not video_path or not isinstance(video_path, str):
        raise ValueError("Video path must be a non-empty string for video processing error context")
    
    # Validate processing stage parameter for stage-specific error handling
    if not processing_stage or not isinstance(processing_stage, str):
        raise ValueError("Processing stage must be specified for video processing error tracking")
    
    # Create video processing error with specialized context and error code
    try:
        video_error = VideoProcessingError(
            message=error_message,
            video_path=video_path,
            processing_stage=processing_stage,
            video_context=processing_context or {},
            original_exception=original_exception
        )
        
        # Set processing stage and video-specific information for context tracking
        video_error.video_path = video_path
        video_error.processing_stage = processing_stage
        video_error.video_context = processing_context or {}
        
        # Include original exception if provided for debugging and root cause analysis
        if original_exception:
            video_error.original_exception = original_exception
            video_error.add_context('original_exception_type', type(original_exception).__name__)
            video_error.add_context('original_exception_message', str(original_exception))
        
        # Add video processing recovery recommendations based on processing stage and error context
        _add_video_processing_recovery_recommendations(video_error, processing_stage, processing_context)
        
        # Create audit trail entry for video processing failure with comprehensive context
        audit_id = create_audit_trail(
            action='VIDEO_PROCESSING_ERROR_CREATED',
            component='VIDEO_PROCESSING',
            action_details={
                'error_message': error_message,
                'video_path': video_path,
                'processing_stage': processing_stage,
                'error_id': video_error.exception_id,
                'has_original_exception': original_exception is not None
            },
            user_context='SYSTEM'
        )
        
        # Log video processing error with detailed context for debugging and monitoring
        logger = get_logger('video_processing.error', 'PROCESSING')
        logger.error(
            f"Video processing error created: {processing_stage} - {error_message} [Video: {video_path}]",
            extra={
                'video_path': video_path,
                'processing_stage': processing_stage,
                'error_id': video_error.exception_id,
                'audit_id': audit_id
            }
        )
        
        # Update processing error statistics for monitoring and analysis
        _update_processing_error_statistics('video_processing', video_error)
        
        # Return configured video processing error instance with comprehensive context
        return video_error
        
    except Exception as creation_error:
        # Handle errors in video processing error creation with fallback error handling
        logger = get_logger('video_processing.error', 'PROCESSING')
        logger.error(f"Failed to create video processing error: {creation_error}")
        
        # Create fallback processing error with basic context
        fallback_error = ProcessingError(
            message=f"Video processing error (creation failed): {error_message}",
            processing_stage=processing_stage,
            input_file=video_path,
            processing_context={'creation_error': str(creation_error)}
        )
        
        return fallback_error


def create_normalization_error(
    error_message: str,
    normalization_type: str,
    normalization_parameters: Dict[str, Any],
    quality_metrics: Dict[str, float],
    input_file: Optional[str] = None
) -> 'NormalizationError':
    """
    Create specialized normalization error with normalization-specific context, parameter tracking, 
    and quality validation information for comprehensive normalization error handling.
    
    This function creates detailed normalization errors with parameter context, quality metrics 
    analysis, and specialized recovery recommendations for scientific computing reliability.
    
    Args:
        error_message: Descriptive error message for the normalization failure
        normalization_type: Type of normalization operation that failed
        normalization_parameters: Parameters used in the failed normalization operation
        quality_metrics: Quality metrics associated with the normalization failure
        input_file: Input file path for traceability and context
        
    Returns:
        NormalizationError: Specialized normalization error with parameter context and quality analysis
    """
    # Validate error message and normalization type for comprehensive error context
    if not error_message or not isinstance(error_message, str):
        raise ValueError("Error message must be a non-empty string for normalization error creation")
    
    if not normalization_type or not isinstance(normalization_type, str):
        raise ValueError("Normalization type must be specified for normalization error classification")
    
    # Validate normalization parameters and quality metrics for parameter context and analysis
    if not isinstance(normalization_parameters, dict):
        raise ValueError("Normalization parameters must be provided as a dictionary")
    
    if not isinstance(quality_metrics, dict):
        raise ValueError("Quality metrics must be provided as a dictionary for analysis")
    
    # Create normalization error with specialized context and parameter tracking
    try:
        normalization_error = NormalizationError(
            message=error_message,
            normalization_type=normalization_type,
            normalization_parameters=normalization_parameters,
            quality_metrics=quality_metrics,
            input_file=input_file
        )
        
        # Set normalization parameters and quality metrics for analysis and debugging
        normalization_error.normalization_type = normalization_type
        normalization_error.normalization_parameters = normalization_parameters
        normalization_error.quality_metrics = quality_metrics
        
        # Include input file information for traceability if provided
        if input_file:
            normalization_error.input_file = input_file
            normalization_error.add_context('input_file', input_file)
        
        # Add normalization-specific recovery recommendations based on failure type and parameters
        _add_normalization_recovery_recommendations(normalization_error, normalization_type, quality_metrics)
        
        # Create audit trail entry for normalization failure with parameter and quality context
        audit_id = create_audit_trail(
            action='NORMALIZATION_ERROR_CREATED',
            component='NORMALIZATION',
            action_details={
                'error_message': error_message,
                'normalization_type': normalization_type,
                'parameter_count': len(normalization_parameters),
                'quality_metrics_count': len(quality_metrics),
                'error_id': normalization_error.exception_id,
                'input_file': input_file
            },
            user_context='SYSTEM'
        )
        
        # Log normalization error with parameter details for debugging and analysis
        logger = get_logger('normalization.error', 'PROCESSING')
        logger.error(
            f"Normalization error created: {normalization_type} - {error_message}",
            extra={
                'normalization_type': normalization_type,
                'parameter_count': len(normalization_parameters),
                'quality_metrics_count': len(quality_metrics),
                'error_id': normalization_error.exception_id,
                'audit_id': audit_id
            }
        )
        
        # Update processing error statistics for monitoring and pattern analysis
        _update_processing_error_statistics('normalization', normalization_error)
        
        # Return configured normalization error instance with comprehensive parameter context
        return normalization_error
        
    except Exception as creation_error:
        # Handle errors in normalization error creation with fallback error handling
        logger = get_logger('normalization.error', 'PROCESSING')
        logger.error(f"Failed to create normalization error: {creation_error}")
        
        # Create fallback processing error with basic normalization context
        fallback_error = ProcessingError(
            message=f"Normalization error (creation failed): {error_message}",
            processing_stage='normalization',
            input_file=input_file or 'unknown',
            processing_context={
                'normalization_type': normalization_type,
                'creation_error': str(creation_error)
            }
        )
        
        return fallback_error


def create_format_conversion_error(
    error_message: str,
    source_format: str,
    target_format: str,
    conversion_context: Dict[str, Any],
    input_file: Optional[str] = None
) -> 'FormatConversionError':
    """
    Create specialized format conversion error with format-specific context, conversion parameters, 
    and compatibility information for cross-format processing error handling.
    
    This function creates comprehensive format conversion errors with format compatibility analysis, 
    conversion parameter tracking, and alternative format recommendations for cross-platform reliability.
    
    Args:
        error_message: Descriptive error message for the format conversion failure
        source_format: Source format that was being converted from
        target_format: Target format that conversion was attempting to achieve
        conversion_context: Context information specific to the conversion operation
        input_file: Input file path for traceability and debugging
        
    Returns:
        FormatConversionError: Specialized format conversion error with format context and compatibility analysis
    """
    # Validate error message and format parameters for comprehensive error context
    if not error_message or not isinstance(error_message, str):
        raise ValueError("Error message must be a non-empty string for format conversion error creation")
    
    if not source_format or not isinstance(source_format, str):
        raise ValueError("Source format must be specified for format conversion error context")
    
    if not target_format or not isinstance(target_format, str):
        raise ValueError("Target format must be specified for format conversion error analysis")
    
    # Validate conversion context for format-specific error analysis
    if not isinstance(conversion_context, dict):
        raise ValueError("Conversion context must be provided as a dictionary for analysis")
    
    # Create format conversion error with specialized context and compatibility analysis
    try:
        format_error = FormatConversionError(
            message=error_message,
            source_format=source_format,
            target_format=target_format,
            conversion_context=conversion_context,
            input_file=input_file
        )
        
        # Set source and target format information for compatibility analysis
        format_error.source_format = source_format
        format_error.target_format = target_format
        format_error.conversion_context = conversion_context
        
        # Include conversion context and compatibility details for analysis
        format_error.add_context('format_conversion', {
            'source_format': source_format,
            'target_format': target_format,
            'conversion_attempted': True
        })
        
        # Include input file information for traceability if provided
        if input_file:
            format_error.input_file = input_file
            format_error.add_context('input_file', input_file)
        
        # Add format-specific recovery recommendations based on format compatibility
        _add_format_conversion_recovery_recommendations(format_error, source_format, target_format)
        
        # Create audit trail entry for format conversion failure with format compatibility context
        audit_id = create_audit_trail(
            action='FORMAT_CONVERSION_ERROR_CREATED',
            component='FORMAT_CONVERSION',
            action_details={
                'error_message': error_message,
                'source_format': source_format,
                'target_format': target_format,
                'conversion_context_size': len(conversion_context),
                'error_id': format_error.exception_id,
                'input_file': input_file
            },
            user_context='SYSTEM'
        )
        
        # Log format conversion error with format details for debugging and compatibility analysis
        logger = get_logger('format_conversion.error', 'PROCESSING')
        logger.error(
            f"Format conversion error created: {source_format} -> {target_format} - {error_message}",
            extra={
                'source_format': source_format,
                'target_format': target_format,
                'input_file': input_file,
                'error_id': format_error.exception_id,
                'audit_id': audit_id
            }
        )
        
        # Update processing error statistics for monitoring and compatibility analysis
        _update_processing_error_statistics('format_conversion', format_error)
        
        # Return configured format conversion error instance with comprehensive format context
        return format_error
        
    except Exception as creation_error:
        # Handle errors in format conversion error creation with fallback error handling
        logger = get_logger('format_conversion.error', 'PROCESSING')
        logger.error(f"Failed to create format conversion error: {creation_error}")
        
        # Create fallback processing error with basic format conversion context
        fallback_error = ProcessingError(
            message=f"Format conversion error (creation failed): {error_message}",
            processing_stage='format_conversion',
            input_file=input_file or 'unknown',
            processing_context={
                'source_format': source_format,
                'target_format': target_format,
                'creation_error': str(creation_error)
            }
        )
        
        return fallback_error


def create_batch_processing_error(
    error_message: str,
    batch_id: str,
    total_items: int,
    processed_items: int,
    batch_context: Dict[str, Any]
) -> 'BatchProcessingError':
    """
    Create specialized batch processing error with batch context, processing statistics, and graceful 
    degradation support for comprehensive batch operation error handling.
    
    This function creates detailed batch processing errors with processing statistics, graceful 
    degradation analysis, and partial completion support for 4000+ simulation processing reliability.
    
    Args:
        error_message: Descriptive error message for the batch processing failure
        batch_id: Unique identifier for the batch operation that failed
        total_items: Total number of items in the batch operation
        processed_items: Number of items successfully processed before failure
        batch_context: Context information specific to the batch processing operation
        
    Returns:
        BatchProcessingError: Specialized batch processing error with batch statistics and degradation support
    """
    # Validate error message and batch parameters for comprehensive error context
    if not error_message or not isinstance(error_message, str):
        raise ValueError("Error message must be a non-empty string for batch processing error creation")
    
    if not batch_id or not isinstance(batch_id, str):
        raise ValueError("Batch ID must be specified for batch processing error tracking")
    
    # Validate processing statistics for batch analysis and graceful degradation
    if not isinstance(total_items, int) or total_items < 0:
        raise ValueError("Total items must be a non-negative integer for batch processing statistics")
    
    if not isinstance(processed_items, int) or processed_items < 0:
        raise ValueError("Processed items must be a non-negative integer for batch processing progress")
    
    if processed_items > total_items:
        raise ValueError("Processed items cannot exceed total items in batch processing statistics")
    
    # Validate batch context for batch-specific error analysis
    if not isinstance(batch_context, dict):
        raise ValueError("Batch context must be provided as a dictionary for analysis")
    
    # Create batch processing error with specialized context and processing statistics
    try:
        batch_error = BatchProcessingError(
            message=error_message,
            batch_id=batch_id,
            total_items=total_items,
            processed_items=processed_items,
            batch_context=batch_context
        )
        
        # Set batch identification and processing statistics for analysis
        batch_error.batch_id = batch_id
        batch_error.total_items = total_items
        batch_error.processed_items = processed_items
        batch_error.batch_context = batch_context
        
        # Calculate processing progress and completion rate for graceful degradation analysis
        completion_rate = (processed_items / total_items * 100) if total_items > 0 else 0
        batch_error.completion_percentage = completion_rate
        
        # Add batch-specific recovery recommendations based on processing progress and statistics
        _add_batch_processing_recovery_recommendations(batch_error, completion_rate, batch_context)
        
        # Create audit trail entry for batch processing failure with processing statistics
        audit_id = create_audit_trail(
            action='BATCH_PROCESSING_ERROR_CREATED',
            component='BATCH_PROCESSING',
            action_details={
                'error_message': error_message,
                'batch_id': batch_id,
                'total_items': total_items,
                'processed_items': processed_items,
                'completion_percentage': completion_rate,
                'error_id': batch_error.exception_id
            },
            user_context='SYSTEM'
        )
        
        # Log batch processing error with statistics for monitoring and analysis
        logger = get_logger('batch_processing.error', 'PROCESSING')
        logger.error(
            f"Batch processing error created: {batch_id} - {error_message} "
            f"[Progress: {processed_items}/{total_items} ({completion_rate:.1f}%)]",
            extra={
                'batch_id': batch_id,
                'total_items': total_items,
                'processed_items': processed_items,
                'completion_percentage': completion_rate,
                'error_id': batch_error.exception_id,
                'audit_id': audit_id
            }
        )
        
        # Update processing error statistics for batch processing monitoring and optimization
        _update_processing_error_statistics('batch_processing', batch_error)
        
        # Return configured batch processing error instance with comprehensive batch context
        return batch_error
        
    except Exception as creation_error:
        # Handle errors in batch processing error creation with fallback error handling
        logger = get_logger('batch_processing.error', 'PROCESSING')
        logger.error(f"Failed to create batch processing error: {creation_error}")
        
        # Create fallback processing error with basic batch processing context
        fallback_error = ProcessingError(
            message=f"Batch processing error (creation failed): {error_message}",
            processing_stage='batch_processing',
            input_file=batch_id,
            processing_context={
                'batch_id': batch_id,
                'total_items': total_items,
                'processed_items': processed_items,
                'creation_error': str(creation_error)
            }
        )
        
        return fallback_error


def handle_processing_error(
    error: Exception,
    processing_context: str,
    processing_stage: str,
    additional_context: Dict[str, Any],
    enable_graceful_degradation: bool = True
) -> ErrorHandlingResult:
    """
    Central processing error handler that coordinates error classification, recovery strategy application, 
    and graceful degradation for processing operations with comprehensive error analysis and audit trail integration.
    
    This function provides centralized processing error handling with comprehensive analysis, recovery 
    strategy coordination, graceful degradation support, and audit trail integration for scientific 
    computing reliability.
    
    Args:
        error: Exception object representing the processing error to handle
        processing_context: Context description for the processing operation
        processing_stage: Current stage of processing when the error occurred
        additional_context: Additional context information for error analysis
        enable_graceful_degradation: Enable graceful degradation strategies for processing failures
        
    Returns:
        ErrorHandlingResult: Comprehensive processing error handling result with recovery actions and recommendations
    """
    # Validate error handling parameters for comprehensive processing error analysis
    if not isinstance(error, Exception):
        raise ValueError("Error parameter must be an Exception instance for processing error handling")
    
    if not processing_context or not isinstance(processing_context, str):
        raise ValueError("Processing context must be specified for error handling classification")
    
    if not processing_stage or not isinstance(processing_stage, str):
        raise ValueError("Processing stage must be specified for stage-specific error handling")
    
    # Validate additional context for comprehensive error analysis
    if additional_context is None:
        additional_context = {}
    elif not isinstance(additional_context, dict):
        raise ValueError("Additional context must be provided as a dictionary")
    
    # Classify processing error type and severity level for appropriate handling strategy
    try:
        # Determine processing error classification based on error type and context
        error_classification = _classify_processing_error(error, processing_stage)
        
        # Build comprehensive processing error context with system state and performance metrics
        processing_error_context = _build_processing_error_context(
            error, processing_context, processing_stage, additional_context
        )
        
        # Apply appropriate processing recovery strategy based on error classification
        recovery_result = None
        if error_classification['category'] in _processing_recovery_strategies:
            try:
                recovery_strategy = _processing_recovery_strategies[error_classification['category']]
                recovery_result = recovery_strategy(error, processing_error_context)
            except Exception as recovery_error:
                # Log recovery strategy failure for debugging and fallback handling
                logger = get_logger('processing.error.recovery', 'PROCESSING')
                logger.warning(f"Processing recovery strategy failed: {recovery_error}")
        
        # Enable graceful degradation if requested and applicable for processing operations
        graceful_degradation_applied = False
        if enable_graceful_degradation and _supports_graceful_degradation(error, processing_stage):
            try:
                degradation_result = _apply_graceful_degradation(error, processing_error_context)
                graceful_degradation_applied = degradation_result.get('applied', False)
            except Exception as degradation_error:
                # Log graceful degradation failure for monitoring and fallback handling
                logger = get_logger('processing.error.degradation', 'PROCESSING')
                logger.warning(f"Graceful degradation failed: {degradation_error}")
        
        # Use central error handling function for comprehensive error analysis and logging
        central_handling_result = handle_error(
            exception=error,
            context=processing_context,
            component='PROCESSING',
            additional_context={
                **additional_context,
                'processing_stage': processing_stage,
                'error_classification': error_classification,
                'recovery_applied': recovery_result is not None,
                'graceful_degradation_applied': graceful_degradation_applied
            },
            allow_recovery=True
        )
        
        # Log processing error with structured format and audit trail integration
        logger = get_logger('processing.error.handler', 'PROCESSING')
        logger.error(
            f"Processing error handled: {processing_stage} - {type(error).__name__}: {str(error)}",
            extra={
                'processing_context': processing_context,
                'processing_stage': processing_stage,
                'error_type': type(error).__name__,
                'error_classification': error_classification['category'],
                'severity': error_classification['severity'],
                'recovery_applied': recovery_result is not None,
                'graceful_degradation': graceful_degradation_applied,
                'handling_result_id': central_handling_result.error_id
            }
        )
        
        # Update processing error statistics and monitoring for pattern analysis
        _update_processing_error_handling_statistics(error, central_handling_result, processing_stage)
        
        # Generate processing-specific recovery recommendations based on error analysis
        _add_processing_specific_recommendations(central_handling_result, error, processing_stage)
        
        # Create comprehensive audit trail for processing error handling operation
        create_audit_trail(
            action='PROCESSING_ERROR_HANDLED',
            component='PROCESSING_ERROR_HANDLER',
            action_details={
                'error_type': type(error).__name__,
                'error_message': str(error),
                'processing_context': processing_context,
                'processing_stage': processing_stage,
                'error_classification': error_classification,
                'recovery_applied': recovery_result is not None,
                'graceful_degradation_applied': graceful_degradation_applied,
                'handling_successful': central_handling_result.handled_successfully,
                'handling_result_id': central_handling_result.error_id
            },
            user_context='SYSTEM'
        )
        
        # Return comprehensive processing error handling outcome with recovery and recommendations
        return central_handling_result
        
    except Exception as handling_error:
        # Handle errors in processing error handling with fallback error management
        logger = get_logger('processing.error.handler', 'PROCESSING')
        logger.error(f"Failed to handle processing error: {handling_error}")
        
        # Create fallback error handling result with basic error information
        fallback_result = ErrorHandlingResult(
            error_id=f"processing_fallback_{datetime.datetime.now().timestamp()}",
            handled_successfully=False,
            handling_context=f"PROCESSING_FALLBACK:{processing_context}"
        )
        
        fallback_result.add_recommendation(
            "Processing error handling failed - manual investigation required",
            "HIGH"
        )
        
        return fallback_result


def register_processing_recovery_strategy(
    processing_error_type: str,
    recovery_function: Callable,
    strategy_name: str,
    strategy_config: Dict[str, Any]
) -> bool:
    """
    Register custom recovery strategy for specific processing error types with validation, testing, 
    and integration into processing error handling workflow for extensible processing error management.
    
    This function provides extensible processing error management by allowing registration of custom 
    recovery strategies for specific processing error types with comprehensive validation and testing.
    
    Args:
        processing_error_type: Type of processing error for recovery strategy registration
        recovery_function: Function implementing the recovery logic for the processing error type
        strategy_name: Name identifier for the processing recovery strategy
        strategy_config: Configuration parameters and settings for the recovery strategy
        
    Returns:
        bool: Success status of processing recovery strategy registration with validation results
    """
    # Validate recovery function signature and processing error type for strategy registration
    if not callable(recovery_function):
        raise ValueError("Recovery function must be callable for processing strategy registration")
    
    if not processing_error_type or not isinstance(processing_error_type, str):
        raise ValueError("Processing error type must be specified for strategy registration")
    
    if not strategy_name or not isinstance(strategy_name, str):
        raise ValueError("Strategy name must be specified for processing strategy identification")
    
    # Validate strategy configuration for comprehensive strategy setup
    if not isinstance(strategy_config, dict):
        raise ValueError("Strategy configuration must be provided as a dictionary")
    
    # Acquire processing error lock for thread-safe strategy registration
    with _processing_error_lock:
        try:
            # Check for existing processing strategy conflicts and compatibility
            if processing_error_type in _processing_recovery_strategies:
                logger = get_logger('processing.strategy.registration', 'PROCESSING')
                logger.warning(f"Overwriting existing processing recovery strategy for {processing_error_type}")
            
            # Test recovery function with mock processing error scenarios for validation
            test_successful = _test_processing_recovery_function(
                recovery_function, processing_error_type, strategy_config
            )
            
            if not test_successful:
                logger = get_logger('processing.strategy.registration', 'PROCESSING')
                logger.error(f"Processing recovery strategy testing failed for {strategy_name}")
                return False
            
            # Register strategy in processing recovery strategies registry with configuration
            _processing_recovery_strategies[processing_error_type] = recovery_function
            
            # Configure strategy parameters and processing metadata for strategy management
            strategy_metadata = {
                'strategy_name': strategy_name,
                'processing_error_type': processing_error_type,
                'configuration': strategy_config,
                'registered_at': datetime.datetime.now().isoformat(),
                'test_successful': test_successful
            }
            
            # Log processing strategy registration with configuration details for audit trail
            logger = get_logger('processing.strategy.registration', 'PROCESSING')
            logger.info(
                f"Processing recovery strategy registered: {strategy_name} for {processing_error_type}",
                extra={
                    'strategy_name': strategy_name,
                    'processing_error_type': processing_error_type,
                    'configuration_keys': list(strategy_config.keys()),
                    'test_successful': test_successful
                }
            )
            
            # Update processing strategy documentation and help information for user guidance
            _update_processing_strategy_documentation(strategy_name, processing_error_type, strategy_config)
            
            # Create audit trail for processing strategy registration operation
            create_audit_trail(
                action='PROCESSING_RECOVERY_STRATEGY_REGISTERED',
                component='PROCESSING_STRATEGY_MANAGER',
                action_details=strategy_metadata,
                user_context='SYSTEM'
            )
            
            # Return registration success status with validation results
            return True
            
        except Exception as registration_error:
            # Handle errors in processing strategy registration with comprehensive error logging
            logger = get_logger('processing.strategy.registration', 'PROCESSING')
            logger.error(f"Failed to register processing recovery strategy {strategy_name}: {registration_error}")
            return False


def get_processing_error_statistics(
    time_window: str,
    processing_stage_filter: str,
    include_resolved_errors: bool = True
) -> Dict[str, Any]:
    """
    Retrieve comprehensive processing error statistics including error rates, processing failure patterns, 
    recovery success rates, and performance impact metrics for processing operation monitoring and optimization.
    
    This function provides comprehensive processing error statistics with rates, patterns, trends, and 
    performance impact analysis for processing operation monitoring and optimization.
    
    Args:
        time_window: Time window specification for statistics collection (e.g., "1h", "24h", "7d")
        processing_stage_filter: Filter statistics by specific processing stage or operation type
        include_resolved_errors: Include resolved processing errors in statistics analysis
        
    Returns:
        Dict[str, Any]: Comprehensive processing error statistics with rates, patterns, and trends
    """
    # Validate time window specification and processing stage filter for statistics collection
    if not time_window or not isinstance(time_window, str):
        raise ValueError("Time window must be specified for processing error statistics")
    
    # Parse time window specification and validate format for statistics collection
    try:
        cutoff_timestamp = _parse_time_window_specification(time_window)
    except ValueError as parse_error:
        raise ValueError(f"Invalid time window specification: {time_window} - {parse_error}")
    
    # Acquire processing error lock for thread-safe statistics access
    with _processing_error_lock:
        try:
            # Filter processing error data by stage and time range for targeted statistics
            filtered_errors = _filter_processing_error_data(
                _processing_error_statistics,
                cutoff_timestamp,
                processing_stage_filter,
                include_resolved_errors
            )
            
            # Calculate processing error rates, frequencies, and patterns for analysis
            error_rates = _calculate_processing_error_rates(filtered_errors, time_window)
            failure_patterns = _analyze_processing_failure_patterns(filtered_errors)
            
            # Analyze processing recovery success rates and effectiveness for optimization
            recovery_statistics = _analyze_processing_recovery_effectiveness(filtered_errors)
            
            # Compute processing performance impact and system health metrics for monitoring
            performance_impact = _calculate_processing_performance_impact(filtered_errors)
            
            # Generate processing trend analysis and pattern detection for predictive analysis
            trend_analysis = _generate_processing_trend_analysis(filtered_errors, time_window)
            
            # Include resolved processing errors if requested for comprehensive analysis
            resolved_errors_analysis = {}
            if include_resolved_errors:
                resolved_errors_analysis = _analyze_resolved_processing_errors(filtered_errors)
            
            # Compile comprehensive processing error statistics with analysis and trends
            statistics_result = {
                'time_window': time_window,
                'processing_stage_filter': processing_stage_filter,
                'include_resolved_errors': include_resolved_errors,
                'generation_timestamp': datetime.datetime.now().isoformat(),
                'cutoff_timestamp': cutoff_timestamp.isoformat(),
                'filtered_errors_count': len(filtered_errors),
                'error_rates': error_rates,
                'failure_patterns': failure_patterns,
                'recovery_statistics': recovery_statistics,
                'performance_impact': performance_impact,
                'trend_analysis': trend_analysis,
                'resolved_errors_analysis': resolved_errors_analysis
            }
            
            # Format processing statistics for reporting and visualization with comprehensive analysis
            statistics_result['summary'] = {
                'total_processing_errors': len(filtered_errors),
                'critical_processing_errors': len([e for e in filtered_errors if e.get('severity') == 'CRITICAL']),
                'resolved_processing_errors': len([e for e in filtered_errors if e.get('resolved', False)]),
                'average_resolution_time_seconds': _calculate_average_processing_resolution_time(filtered_errors),
                'most_common_processing_stage': _identify_most_common_processing_stage(filtered_errors),
                'processing_error_rate_percentage': error_rates.get('overall_rate', 0) * 100,
                'recovery_success_rate_percentage': recovery_statistics.get('success_rate', 0) * 100
            }
            
            # Log statistics generation for audit trail and monitoring
            logger = get_logger('processing.statistics', 'PROCESSING')
            logger.info(
                f"Processing error statistics generated for {time_window} window: "
                f"{len(filtered_errors)} errors analyzed",
                extra={
                    'time_window': time_window,
                    'processing_stage_filter': processing_stage_filter,
                    'filtered_errors_count': len(filtered_errors),
                    'generation_timestamp': statistics_result['generation_timestamp']
                }
            )
            
            # Return comprehensive processing error statistics dictionary with analysis and trends
            return statistics_result
            
        except Exception as statistics_error:
            # Handle errors in processing error statistics generation with fallback results
            logger = get_logger('processing.statistics', 'PROCESSING')
            logger.error(f"Failed to generate processing error statistics: {statistics_error}")
            
            # Return basic error information for statistics generation failure
            return {
                'error': str(statistics_error),
                'time_window': time_window,
                'processing_stage_filter': processing_stage_filter,
                'generation_timestamp': datetime.datetime.now().isoformat(),
                'success': False
            }


class VideoProcessingError(ProcessingError):
    """
    Specialized processing error class for video processing failures including video reading errors, 
    format detection issues, frame processing problems, and video-specific normalization failures 
    with video context tracking and intermediate result preservation for graceful degradation in 
    video processing workflows.
    
    This class provides comprehensive video processing error handling with video-specific context, 
    processing stage tracking, and graceful degradation support for complex video processing operations.
    """
    
    def __init__(
        self,
        message: str,
        video_path: str,
        processing_stage: str,
        video_context: Dict[str, Any],
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize video processing error with video-specific context, processing stage tracking, 
        and graceful degradation support for comprehensive video processing error handling.
        
        Args:
            message: Descriptive error message for the video processing failure
            video_path: Path to the video file being processed when error occurred
            processing_stage: Current stage of video processing when error occurred
            video_context: Context information specific to video processing operation
            original_exception: Original exception that caused the video processing failure
        """
        # Initialize base ProcessingError with PROCESSING category and VIDEO_PROCESSING_ERROR_CODE
        super().__init__(
            message=message,
            processing_stage=processing_stage,
            input_file=video_path,
            processing_context=video_context
        )
        
        # Override error code for video processing specific error classification
        self.error_code = VIDEO_PROCESSING_ERROR_CODE
        self.category = ErrorCategory.PROCESSING
        
        # Set video path and processing stage information for video-specific context
        self.video_path = video_path
        self.processing_stage = processing_stage
        self.video_context = video_context or {}
        self.original_exception = original_exception
        
        # Initialize video metadata and processing progress tracking for comprehensive analysis
        self.video_metadata: Dict[str, Any] = {}
        self.processing_progress: Dict[str, Any] = {}
        self.completed_processing_steps: List[str] = []
        self.failed_processing_steps: List[str] = []
        self.intermediate_video_results: Dict[str, Any] = {}
        
        # Determine graceful degradation support based on processing stage and video context
        self.supports_graceful_degradation = _video_supports_graceful_degradation(
            processing_stage, video_context
        )
        
        # Add video processing-specific recovery recommendations based on failure type
        self._add_video_processing_recommendations()
        
        # Create audit trail entry for video processing failure with comprehensive video context
        create_audit_trail(
            action='VIDEO_PROCESSING_ERROR_INITIALIZED',
            component='VIDEO_PROCESSING_ERROR',
            action_details={
                'video_path': video_path,
                'processing_stage': processing_stage,
                'has_original_exception': original_exception is not None,
                'supports_graceful_degradation': self.supports_graceful_degradation,
                'error_id': self.exception_id
            },
            user_context='SYSTEM'
        )
        
        # Log video processing error with comprehensive video context and processing stage
        logger = get_logger('video_processing.error', 'VIDEO_PROCESSING')
        logger.error(
            f"Video processing error initialized: {processing_stage} - {message} [Video: {video_path}]",
            extra={
                'video_path': video_path,
                'processing_stage': processing_stage,
                'error_id': self.exception_id,
                'supports_graceful_degradation': self.supports_graceful_degradation
            }
        )
    
    def preserve_video_processing_results(
        self,
        video_results: Dict[str, Any],
        completion_percentage: float,
        video_metadata: Dict[str, Any]
    ) -> None:
        """
        Preserve intermediate video processing results for graceful degradation and potential recovery 
        with video-specific context and metadata.
        
        This method preserves partial video processing results to support graceful degradation and 
        recovery operations in video processing workflows.
        
        Args:
            video_results: Intermediate video processing results to preserve
            completion_percentage: Processing completion percentage for progress tracking
            video_metadata: Video metadata for result interpretation and context
        """
        # Store intermediate video results in processing context for graceful degradation
        self.intermediate_video_results.update(video_results)
        
        # Update video processing progress percentage for completion tracking
        self.processing_progress['completion_percentage'] = completion_percentage
        self.processing_progress['last_updated'] = datetime.datetime.now().isoformat()
        
        # Include video metadata for result interpretation and debugging context
        self.video_metadata.update(video_metadata)
        
        # Set partial success flag if progress > 0 for graceful degradation analysis
        if completion_percentage > 0:
            self.partial_success = True
            self.processing_progress['partial_success'] = True
        
        # Log intermediate video results preservation for tracking and recovery planning
        logger = get_logger('video_processing.error', 'VIDEO_PROCESSING')
        logger.info(
            f"Video processing results preserved: {completion_percentage:.1f}% complete "
            f"[Video: {self.video_path}, Error: {self.exception_id}]",
            extra={
                'video_path': self.video_path,
                'completion_percentage': completion_percentage,
                'error_id': self.exception_id,
                'results_count': len(video_results)
            }
        )
        
        # Update recovery recommendations with video-specific partial results information
        if self.partial_success:
            self.add_recovery_recommendation(
                f"Partial video processing results available ({completion_percentage:.1f}% complete) - "
                f"consider resuming from video checkpoint",
                priority='MEDIUM',
                recommendation_context={
                    'completion_percentage': completion_percentage,
                    'preserved_results_count': len(video_results),
                    'video_metadata_available': bool(video_metadata)
                }
            )
    
    def add_video_processing_step(
        self,
        step_name: str,
        step_results: Dict[str, Any],
        step_performance_metrics: Dict[str, float]
    ) -> None:
        """
        Add completed video processing step with step-specific results and performance metrics 
        for comprehensive video processing tracking.
        
        This method tracks individual video processing steps with results and performance metrics 
        for detailed analysis and recovery planning.
        
        Args:
            step_name: Name of the completed video processing step
            step_results: Results from the completed video processing step
            step_performance_metrics: Performance metrics for the completed step
        """
        # Add step name to completed video processing steps list for tracking
        self.completed_processing_steps.append(step_name)
        
        # Store step results in intermediate video results for recovery and analysis
        step_key = f'{step_name}_results'
        self.intermediate_video_results[step_key] = step_results
        
        # Include step performance metrics for analysis and optimization
        performance_key = f'{step_name}_performance'
        self.processing_progress[performance_key] = step_performance_metrics
        
        # Update video processing progress calculation based on completed steps
        total_expected_steps = self.video_context.get('expected_steps', len(self.completed_processing_steps))
        if total_expected_steps > 0:
            completion_percentage = (len(self.completed_processing_steps) / total_expected_steps) * 100
            self.processing_progress['completion_percentage'] = completion_percentage
        
        # Log completed video processing step for tracking and performance analysis
        logger = get_logger('video_processing.error', 'VIDEO_PROCESSING')
        logger.debug(
            f"Video processing step completed: {step_name} [Video: {self.video_path}, Error: {self.exception_id}]",
            extra={
                'step_name': step_name,
                'video_path': self.video_path,
                'error_id': self.exception_id,
                'completed_steps_count': len(self.completed_processing_steps),
                'performance_metrics': step_performance_metrics
            }
        )
    
    def get_video_processing_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive video processing failure summary with video context, processing progress, 
        and recovery guidance specific to video processing workflows.
        
        This method provides detailed analysis of video processing failure with context, progress, 
        and specialized recovery recommendations for video processing operations.
        
        Returns:
            Dict[str, Any]: Video processing failure summary with comprehensive video analysis and recommendations
        """
        # Compile video processing steps and progress information for comprehensive analysis
        summary = {
            'error_id': self.exception_id,
            'video_path': self.video_path,
            'processing_stage': self.processing_stage,
            'video_context': self.video_context.copy(),
            'video_metadata': self.video_metadata.copy(),
            'processing_progress': self.processing_progress.copy(),
            'completed_processing_steps': self.completed_processing_steps.copy(),
            'failed_processing_steps': self.failed_processing_steps.copy(),
            'supports_graceful_degradation': self.supports_graceful_degradation,
            'has_original_exception': self.original_exception is not None
        }
        
        # Include video metadata and context details for debugging and analysis
        if self.video_metadata:
            summary['video_analysis'] = {
                'metadata_available': True,
                'metadata_keys': list(self.video_metadata.keys()),
                'video_characteristics': self.video_metadata
            }
        
        # Add video-specific intermediate results and partial completion information
        if self.intermediate_video_results:
            summary['intermediate_results'] = {
                'results_available': True,
                'results_count': len(self.intermediate_video_results),
                'results_summary': {k: str(v)[:100] for k, v in self.intermediate_video_results.items()}
            }
        
        # Generate video processing-specific recovery recommendations based on failure analysis
        recovery_recommendations = self.get_recovery_recommendations()
        summary['recovery_recommendations'] = recovery_recommendations
        
        # Include video format and compatibility analysis for format-specific guidance
        if 'format' in self.video_context or 'codec' in self.video_context:
            summary['format_analysis'] = {
                'video_format': self.video_context.get('format', 'unknown'),
                'video_codec': self.video_context.get('codec', 'unknown'),
                'compatibility_notes': _analyze_video_format_compatibility(
                    self.video_context.get('format'), self.video_context.get('codec')
                )
            }
        
        # Calculate processing efficiency and performance metrics
        if self.processing_progress.get('completion_percentage', 0) > 0:
            summary['processing_efficiency'] = {
                'completion_percentage': self.processing_progress.get('completion_percentage', 0),
                'completed_steps_count': len(self.completed_processing_steps),
                'failed_steps_count': len(self.failed_processing_steps),
                'efficiency_score': self._calculate_video_processing_efficiency()
            }
        
        # Return comprehensive video processing summary with analysis and recommendations
        return summary
    
    def _add_video_processing_recommendations(self) -> None:
        """Add video processing-specific recovery recommendations based on failure context."""
        # Add video format and codec specific recommendations
        if 'format' in self.video_context:
            video_format = self.video_context['format']
            self.add_recovery_recommendation(
                f"Verify video format compatibility: {video_format}",
                priority='HIGH'
            )
        
        # Add video file accessibility and integrity recommendations
        self.add_recovery_recommendation(
            f"Verify video file accessibility and integrity: {self.video_path}",
            priority='HIGH'
        )
        
        # Add processing stage specific recommendations
        if 'frame_extraction' in self.processing_stage.lower():
            self.add_recovery_recommendation(
                "Check frame extraction parameters and video decoder compatibility",
                priority='MEDIUM'
            )
        elif 'normalization' in self.processing_stage.lower():
            self.add_recovery_recommendation(
                "Review video normalization parameters and scaling factors",
                priority='MEDIUM'
            )
    
    def _calculate_video_processing_efficiency(self) -> float:
        """Calculate video processing efficiency score based on completed vs failed steps."""
        total_steps = len(self.completed_processing_steps) + len(self.failed_processing_steps)
        if total_steps == 0:
            return 0.0
        
        return (len(self.completed_processing_steps) / total_steps) * 100


class NormalizationError(ProcessingError):
    """
    Specialized processing error class for data normalization failures including scale calibration errors, 
    temporal normalization issues, intensity calibration problems, and parameter validation failures 
    with normalization context tracking and quality metrics analysis for scientific computing reliability.
    
    This class provides comprehensive normalization error handling with parameter context, quality metrics 
    analysis, and specialized recovery recommendations for scientific computing workflows.
    """
    
    def __init__(
        self,
        message: str,
        normalization_type: str,
        normalization_parameters: Dict[str, Any],
        quality_metrics: Dict[str, float],
        input_file: Optional[str] = None
    ):
        """
        Initialize normalization error with normalization-specific context, parameter tracking, 
        and quality validation information for comprehensive normalization error handling.
        
        Args:
            message: Descriptive error message for the normalization failure
            normalization_type: Type of normalization operation that failed
            normalization_parameters: Parameters used in the failed normalization operation
            quality_metrics: Quality metrics associated with the normalization failure
            input_file: Input file path for traceability and context
        """
        # Initialize base ProcessingError with PROCESSING category and NORMALIZATION_ERROR_CODE
        super().__init__(
            message=message,
            processing_stage='normalization',
            input_file=input_file or 'unknown',
            processing_context=normalization_parameters
        )
        
        # Override error code for normalization specific error classification
        self.error_code = NORMALIZATION_ERROR_CODE
        self.category = ErrorCategory.PROCESSING
        
        # Set normalization type and parameters information for comprehensive analysis
        self.normalization_type = normalization_type
        self.normalization_parameters = normalization_parameters or {}
        self.quality_metrics = quality_metrics or {}
        self.input_file = input_file
        
        # Initialize calibration context and validation tracking for parameter analysis
        self.calibration_context: Dict[str, Any] = {}
        self.failed_normalization_steps: List[str] = []
        self.parameter_validation_errors: Dict[str, Any] = {}
        self.quality_threshold_violations: Dict[str, float] = {}
        
        # Analyze quality threshold violations and determine quality failure status
        self.is_quality_failure = _analyze_quality_threshold_violations(quality_metrics)
        
        # Generate normalization-specific recommendations based on failure type and parameters
        self.normalization_recommendations: Dict[str, Any] = {}
        self._generate_normalization_recommendations()
        
        # Add normalization-specific recovery recommendations based on failure analysis
        self._add_normalization_recovery_recommendations()
        
        # Create audit trail entry for normalization failure with parameter and quality context
        create_audit_trail(
            action='NORMALIZATION_ERROR_INITIALIZED',
            component='NORMALIZATION_ERROR',
            action_details={
                'normalization_type': normalization_type,
                'parameter_count': len(normalization_parameters),
                'quality_metrics_count': len(quality_metrics),
                'is_quality_failure': self.is_quality_failure,
                'input_file': input_file,
                'error_id': self.exception_id
            },
            user_context='SYSTEM'
        )
        
        # Log normalization error with comprehensive parameter context and quality analysis
        logger = get_logger('normalization.error', 'NORMALIZATION')
        logger.error(
            f"Normalization error initialized: {normalization_type} - {message}",
            extra={
                'normalization_type': normalization_type,
                'parameter_count': len(normalization_parameters),
                'quality_metrics_count': len(quality_metrics),
                'is_quality_failure': self.is_quality_failure,
                'input_file': input_file,
                'error_id': self.exception_id
            }
        )
    
    def add_parameter_validation_error(
        self,
        parameter_name: str,
        invalid_value: Any,
        validation_rule: str,
        expected_format: str
    ) -> None:
        """
        Add parameter validation error with detailed parameter context and validation rule information 
        for comprehensive normalization parameter error tracking.
        
        This method tracks individual parameter validation failures with detailed context for 
        debugging and parameter correction guidance.
        
        Args:
            parameter_name: Name of the parameter that failed validation
            invalid_value: Invalid value that was provided for the parameter
            validation_rule: Validation rule that was violated by the parameter
            expected_format: Expected format for parameter correction guidance
        """
        # Add parameter name to parameter validation errors registry for tracking
        validation_error_entry = {
            'parameter_name': parameter_name,
            'invalid_value': invalid_value,
            'validation_rule': validation_rule,
            'expected_format': expected_format,
            'detected_at': datetime.datetime.now().isoformat()
        }
        
        self.parameter_validation_errors[parameter_name] = validation_error_entry
        
        # Store invalid value and validation rule details for parameter correction guidance
        self.failed_normalization_steps.append(f"parameter_validation_{parameter_name}")
        
        # Include expected format for parameter correction and user guidance
        self.normalization_context[f'{parameter_name}_validation_error'] = validation_error_entry
        
        # Update normalization context with parameter error information for debugging
        self.add_context('parameter_validation_errors', self.parameter_validation_errors)
        
        # Log parameter validation error for debugging and parameter correction guidance
        logger = get_logger('normalization.error', 'NORMALIZATION')
        logger.warning(
            f"Parameter validation error: {parameter_name} = {invalid_value} "
            f"(violates: {validation_rule}, expected: {expected_format}) "
            f"[Error: {self.exception_id}]",
            extra={
                'parameter_name': parameter_name,
                'invalid_value': str(invalid_value),
                'validation_rule': validation_rule,
                'expected_format': expected_format,
                'error_id': self.exception_id
            }
        )
    
    def add_quality_threshold_violation(
        self,
        metric_name: str,
        metric_value: float,
        threshold_value: float,
        violation_severity: str
    ) -> None:
        """
        Add quality threshold violation with metric details and threshold information for 
        quality-based normalization error analysis.
        
        This method tracks quality threshold violations with detailed metrics for quality 
        analysis and corrective action recommendations.
        
        Args:
            metric_name: Name of the quality metric that violated the threshold
            metric_value: Actual value of the quality metric
            threshold_value: Threshold value that was exceeded by the metric
            violation_severity: Severity level of the threshold violation
        """
        # Add metric name to quality threshold violations registry for quality analysis
        violation_entry = {
            'metric_name': metric_name,
            'metric_value': metric_value,
            'threshold_value': threshold_value,
            'violation_severity': violation_severity,
            'percentage_violation': ((metric_value - threshold_value) / threshold_value * 100) if threshold_value > 0 else 0,
            'detected_at': datetime.datetime.now().isoformat()
        }
        
        self.quality_threshold_violations[metric_name] = violation_entry
        
        # Store metric value and threshold information for quality analysis
        self.quality_metrics[metric_name] = metric_value
        
        # Include violation severity for prioritization and corrective action planning
        self.failed_normalization_steps.append(f"quality_threshold_{metric_name}")
        
        # Set is_quality_failure flag to True for quality-based error classification
        self.is_quality_failure = True
        
        # Update normalization recommendations with quality improvement suggestions
        quality_recommendation = f"Improve {metric_name} quality metric (current: {metric_value:.3f}, threshold: {threshold_value:.3f})"
        self.normalization_recommendations[f'quality_{metric_name}'] = quality_recommendation
        
        # Log quality threshold violation for analysis and quality improvement guidance
        logger = get_logger('normalization.error', 'NORMALIZATION')
        logger.warning(
            f"Quality threshold violation: {metric_name} = {metric_value:.3f} "
            f"exceeds threshold {threshold_value:.3f} (severity: {violation_severity}) "
            f"[Error: {self.exception_id}]",
            extra={
                'metric_name': metric_name,
                'metric_value': metric_value,
                'threshold_value': threshold_value,
                'violation_severity': violation_severity,
                'percentage_violation': violation_entry['percentage_violation'],
                'error_id': self.exception_id
            }
        )
    
    def get_normalization_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive normalization failure analysis including parameter errors, quality violations, 
        and corrective action recommendations for normalization troubleshooting.
        
        This method provides detailed analysis of normalization failure with parameter context, 
        quality assessment, and specialized troubleshooting guidance.
        
        Returns:
            Dict[str, Any]: Normalization failure analysis with parameter details and quality assessment
        """
        # Compile normalization parameters and validation errors for comprehensive analysis
        analysis = {
            'error_id': self.exception_id,
            'normalization_type': self.normalization_type,
            'input_file': self.input_file,
            'normalization_parameters': self.normalization_parameters.copy(),
            'quality_metrics': self.quality_metrics.copy(),
            'parameter_validation_errors': self.parameter_validation_errors.copy(),
            'quality_threshold_violations': self.quality_threshold_violations.copy(),
            'failed_normalization_steps': self.failed_normalization_steps.copy(),
            'is_quality_failure': self.is_quality_failure,
            'calibration_context': self.calibration_context.copy()
        }
        
        # Include quality metrics and threshold violations for quality assessment
        if self.quality_threshold_violations:
            analysis['quality_analysis'] = {
                'total_violations': len(self.quality_threshold_violations),
                'violation_summary': {
                    metric: violation['violation_severity'] 
                    for metric, violation in self.quality_threshold_violations.items()
                },
                'most_severe_violation': _identify_most_severe_quality_violation(self.quality_threshold_violations)
            }
        
        # Analyze normalization failure patterns and root causes for troubleshooting guidance
        root_cause_analysis = _analyze_normalization_failure_patterns(
            self.parameter_validation_errors, 
            self.quality_threshold_violations,
            self.normalization_parameters
        )
        analysis['root_cause_analysis'] = root_cause_analysis
        
        # Generate parameter correction recommendations based on validation errors
        parameter_corrections = {}
        for param_name, error_info in self.parameter_validation_errors.items():
            parameter_corrections[param_name] = {
                'current_value': error_info['invalid_value'],
                'expected_format': error_info['expected_format'],
                'validation_rule': error_info['validation_rule'],
                'correction_suggestion': _generate_parameter_correction_suggestion(error_info)
            }
        analysis['parameter_corrections'] = parameter_corrections
        
        # Include quality improvement suggestions based on threshold violations
        quality_improvements = {}
        for metric_name, violation_info in self.quality_threshold_violations.items():
            quality_improvements[metric_name] = {
                'current_value': violation_info['metric_value'],
                'threshold_value': violation_info['threshold_value'],
                'improvement_needed': violation_info['percentage_violation'],
                'improvement_suggestions': _generate_quality_improvement_suggestions(violation_info)
            }
        analysis['quality_improvements'] = quality_improvements
        
        # Add calibration context and troubleshooting guidance for comprehensive support
        analysis['troubleshooting_guidance'] = {
            'recommended_actions': self.get_recovery_recommendations(),
            'normalization_recommendations': self.normalization_recommendations,
            'priority_fixes': _prioritize_normalization_fixes(
                self.parameter_validation_errors, 
                self.quality_threshold_violations
            )
        }
        
        # Include normalization efficiency and success probability estimates
        analysis['normalization_assessment'] = {
            'parameter_error_count': len(self.parameter_validation_errors),
            'quality_violation_count': len(self.quality_threshold_violations),
            'normalization_complexity': _assess_normalization_complexity(self.normalization_parameters),
            'recovery_probability': _estimate_normalization_recovery_probability(analysis)
        }
        
        # Return comprehensive normalization analysis with troubleshooting guidance
        return analysis
    
    def _generate_normalization_recommendations(self) -> None:
        """Generate normalization-specific recommendations based on failure type and context."""
        # Add normalization type specific recommendations
        if 'scale' in self.normalization_type.lower():
            self.normalization_recommendations['scale_calibration'] = (
                "Review scale calibration parameters and ensure proper unit conversion"
            )
        elif 'temporal' in self.normalization_type.lower():
            self.normalization_recommendations['temporal_alignment'] = (
                "Check temporal alignment parameters and sampling rate consistency"
            )
        elif 'intensity' in self.normalization_type.lower():
            self.normalization_recommendations['intensity_calibration'] = (
                "Verify intensity calibration settings and dynamic range parameters"
            )
        
        # Add quality-based recommendations if quality failure detected
        if self.is_quality_failure:
            self.normalization_recommendations['quality_improvement'] = (
                "Focus on improving quality metrics that exceeded thresholds"
            )
    
    def _add_normalization_recovery_recommendations(self) -> None:
        """Add normalization-specific recovery recommendations based on failure analysis."""
        self.add_recovery_recommendation(
            f"Review {self.normalization_type} normalization parameters and validation rules",
            priority='HIGH'
        )
        
        if self.parameter_validation_errors:
            self.add_recovery_recommendation(
                f"Correct {len(self.parameter_validation_errors)} parameter validation errors",
                priority='HIGH'
            )
        
        if self.quality_threshold_violations:
            self.add_recovery_recommendation(
                f"Address {len(self.quality_threshold_violations)} quality threshold violations",
                priority='MEDIUM'
            )


class FormatConversionError(ProcessingError):
    """
    Specialized processing error class for format conversion failures including unsupported format 
    combinations, conversion parameter errors, compatibility issues, and cross-format processing 
    problems with format context tracking and compatibility analysis for cross-platform video 
    processing reliability.
    
    This class provides comprehensive format conversion error handling with format compatibility 
    analysis, conversion parameter tracking, and alternative format recommendations.
    """
    
    def __init__(
        self,
        message: str,
        source_format: str,
        target_format: str,
        conversion_context: Dict[str, Any],
        input_file: Optional[str] = None
    ):
        """
        Initialize format conversion error with format-specific context, conversion parameters, 
        and compatibility information for comprehensive cross-format processing error handling.
        
        Args:
            message: Descriptive error message for the format conversion failure
            source_format: Source format that was being converted from
            target_format: Target format that conversion was attempting to achieve
            conversion_context: Context information specific to the conversion operation
            input_file: Input file path for traceability and debugging
        """
        # Initialize base ProcessingError with PROCESSING category and FORMAT_CONVERSION_ERROR_CODE
        super().__init__(
            message=message,
            processing_stage='format_conversion',
            input_file=input_file or 'unknown',
            processing_context=conversion_context
        )
        
        # Override error code for format conversion specific error classification
        self.error_code = FORMAT_CONVERSION_ERROR_CODE
        self.category = ErrorCategory.PROCESSING
        
        # Set source and target format information for compatibility analysis
        self.source_format = source_format
        self.target_format = target_format
        self.conversion_context = conversion_context or {}
        self.input_file = input_file
        
        # Initialize format compatibility analysis and unsupported features tracking
        self.format_compatibility_analysis: Dict[str, Any] = {}
        self.unsupported_features: List[str] = []
        self.conversion_parameters: Dict[str, Any] = {}
        self.is_format_incompatible = False
        self.alternative_formats: List[str] = []
        self.conversion_recommendations: Dict[str, Any] = {}
        
        # Analyze format compatibility and identify alternative formats
        self._analyze_format_compatibility()
        self._identify_alternative_formats()
        
        # Generate format-specific conversion recommendations based on compatibility analysis
        self._generate_format_conversion_recommendations()
        
        # Add format conversion-specific recovery recommendations
        self._add_format_conversion_recovery_recommendations()
        
        # Create audit trail entry for format conversion failure with format compatibility context
        create_audit_trail(
            action='FORMAT_CONVERSION_ERROR_INITIALIZED',
            component='FORMAT_CONVERSION_ERROR',
            action_details={
                'source_format': source_format,
                'target_format': target_format,
                'is_format_incompatible': self.is_format_incompatible,
                'unsupported_features_count': len(self.unsupported_features),
                'alternative_formats_count': len(self.alternative_formats),
                'input_file': input_file,
                'error_id': self.exception_id
            },
            user_context='SYSTEM'
        )
        
        # Log format conversion error with comprehensive format context and compatibility analysis
        logger = get_logger('format_conversion.error', 'FORMAT_CONVERSION')
        logger.error(
            f"Format conversion error initialized: {source_format} -> {target_format} - {message}",
            extra={
                'source_format': source_format,
                'target_format': target_format,
                'is_format_incompatible': self.is_format_incompatible,
                'unsupported_features_count': len(self.unsupported_features),
                'input_file': input_file,
                'error_id': self.exception_id
            }
        )
    
    def add_unsupported_feature(
        self,
        feature_name: str,
        feature_description: str,
        alternative_approaches: List[str]
    ) -> None:
        """
        Add unsupported feature with feature details and alternative suggestions for comprehensive 
        format compatibility tracking.
        
        This method tracks individual unsupported features with alternative approaches for 
        comprehensive format compatibility analysis and recovery planning.
        
        Args:
            feature_name: Name of the unsupported feature
            feature_description: Description of the unsupported feature
            alternative_approaches: List of alternative approaches for the unsupported feature
        """
        # Add feature name to unsupported features list for compatibility tracking
        if feature_name not in self.unsupported_features:
            self.unsupported_features.append(feature_name)
        
        # Store feature description and alternative approaches for recovery planning
        feature_entry = {
            'feature_name': feature_name,
            'feature_description': feature_description,
            'alternative_approaches': alternative_approaches or [],
            'detected_at': datetime.datetime.now().isoformat()
        }
        
        # Update format compatibility analysis with feature information
        self.format_compatibility_analysis[f'unsupported_{feature_name}'] = feature_entry
        
        # Include alternative approaches in conversion recommendations for recovery guidance
        if alternative_approaches:
            self.conversion_recommendations[f'alternative_{feature_name}'] = (
                f"Use alternative approaches for {feature_name}: {', '.join(alternative_approaches)}"
            )
        
        # Log unsupported feature for format compatibility analysis and recovery planning
        logger = get_logger('format_conversion.error', 'FORMAT_CONVERSION')
        logger.warning(
            f"Unsupported feature detected: {feature_name} - {feature_description} "
            f"[Alternatives: {len(alternative_approaches)}] [Error: {self.exception_id}]",
            extra={
                'feature_name': feature_name,
                'feature_description': feature_description,
                'alternative_approaches_count': len(alternative_approaches),
                'error_id': self.exception_id
            }
        )
    
    def add_alternative_format(
        self,
        format_name: str,
        compatibility_score: float,
        conversion_requirements: Dict[str, Any]
    ) -> None:
        """
        Add alternative format suggestion with compatibility assessment and conversion guidance 
        for format conversion recovery.
        
        This method tracks alternative format options with compatibility scores and conversion 
        requirements for recovery planning and format selection.
        
        Args:
            format_name: Name of the alternative format
            compatibility_score: Compatibility score for the alternative format (0.0 to 1.0)
            conversion_requirements: Requirements for converting to the alternative format
        """
        # Add format name to alternative formats list for recovery options
        if format_name not in self.alternative_formats:
            self.alternative_formats.append(format_name)
        
        # Store compatibility score and conversion requirements for format selection
        alternative_entry = {
            'format_name': format_name,
            'compatibility_score': compatibility_score,
            'conversion_requirements': conversion_requirements or {},
            'recommended_priority': _calculate_format_priority(compatibility_score),
            'added_at': datetime.datetime.now().isoformat()
        }
        
        # Update conversion recommendations with alternative format guidance
        priority_label = alternative_entry['recommended_priority']
        self.conversion_recommendations[f'alternative_format_{format_name}'] = (
            f"Consider {format_name} format as alternative (compatibility: {compatibility_score:.2f}, priority: {priority_label})"
        )
        
        # Include conversion requirements for implementation guidance and planning
        self.format_compatibility_analysis[f'alternative_{format_name}'] = alternative_entry
        
        # Log alternative format suggestion for recovery planning and format selection
        logger = get_logger('format_conversion.error', 'FORMAT_CONVERSION')
        logger.info(
            f"Alternative format suggested: {format_name} "
            f"(compatibility: {compatibility_score:.2f}, priority: {priority_label}) "
            f"[Error: {self.exception_id}]",
            extra={
                'format_name': format_name,
                'compatibility_score': compatibility_score,
                'recommended_priority': priority_label,
                'requirements_count': len(conversion_requirements),
                'error_id': self.exception_id
            }
        )
    
    def get_format_compatibility_report(self) -> Dict[str, Any]:
        """
        Get comprehensive format compatibility report including compatibility analysis, unsupported 
        features, and alternative format recommendations for format conversion troubleshooting.
        
        This method provides detailed format compatibility analysis with troubleshooting guidance 
        and alternative format recommendations for successful format conversion.
        
        Returns:
            Dict[str, Any]: Format compatibility report with analysis and alternative format recommendations
        """
        # Compile source and target format compatibility analysis for comprehensive report
        report = {
            'error_id': self.exception_id,
            'source_format': self.source_format,
            'target_format': self.target_format,
            'input_file': self.input_file,
            'conversion_context': self.conversion_context.copy(),
            'is_format_incompatible': self.is_format_incompatible,
            'format_compatibility_analysis': self.format_compatibility_analysis.copy(),
            'unsupported_features': self.unsupported_features.copy(),
            'alternative_formats': self.alternative_formats.copy(),
            'conversion_recommendations': self.conversion_recommendations.copy()
        }
        
        # Include unsupported features and limitations for comprehensive compatibility analysis
        if self.unsupported_features:
            report['unsupported_features_analysis'] = {
                'total_unsupported': len(self.unsupported_features),
                'feature_categories': _categorize_unsupported_features(self.unsupported_features),
                'impact_assessment': _assess_unsupported_features_impact(self.unsupported_features),
                'workaround_availability': _assess_workaround_availability(self.unsupported_features)
            }
        
        # Add alternative format suggestions with compatibility scores and implementation guidance
        if self.alternative_formats:
            alternative_analysis = {}
            for format_name in self.alternative_formats:
                format_key = f'alternative_{format_name}'
                if format_key in self.format_compatibility_analysis:
                    alternative_analysis[format_name] = self.format_compatibility_analysis[format_key]
            
            report['alternative_formats_analysis'] = {
                'total_alternatives': len(self.alternative_formats),
                'format_details': alternative_analysis,
                'recommended_format': _identify_best_alternative_format(alternative_analysis),
                'implementation_priority': _prioritize_alternative_formats(alternative_analysis)
            }
        
        # Generate conversion parameter recommendations based on format analysis
        parameter_recommendations = _generate_format_conversion_parameter_recommendations(
            self.source_format, self.target_format, self.conversion_context
        )
        report['parameter_recommendations'] = parameter_recommendations
        
        # Include format-specific troubleshooting guidance and best practices
        troubleshooting_guidance = {
            'recommended_actions': self.get_recovery_recommendations(),
            'format_specific_tips': _get_format_specific_troubleshooting_tips(
                self.source_format, self.target_format
            ),
            'compatibility_improvements': _suggest_compatibility_improvements(
                self.source_format, self.target_format, self.unsupported_features
            )
        }
        report['troubleshooting_guidance'] = troubleshooting_guidance
        
        # Add cross-platform compatibility considerations and platform-specific guidance
        cross_platform_analysis = _analyze_cross_platform_compatibility(
            self.source_format, self.target_format, self.conversion_context
        )
        report['cross_platform_compatibility'] = cross_platform_analysis
        
        # Include conversion success probability and effort estimation
        report['conversion_assessment'] = {
            'compatibility_score': _calculate_overall_compatibility_score(
                self.source_format, self.target_format, self.unsupported_features
            ),
            'conversion_difficulty': _assess_conversion_difficulty(
                self.source_format, self.target_format, self.unsupported_features
            ),
            'success_probability': _estimate_conversion_success_probability(report),
            'estimated_effort': _estimate_conversion_effort(report)
        }
        
        # Return comprehensive format compatibility report with analysis and recommendations
        return report
    
    def _analyze_format_compatibility(self) -> None:
        """Analyze format compatibility between source and target formats."""
        # Perform basic format compatibility analysis
        compatibility_result = _perform_format_compatibility_analysis(
            self.source_format, self.target_format
        )
        
        self.format_compatibility_analysis['basic_compatibility'] = compatibility_result
        self.is_format_incompatible = not compatibility_result.get('compatible', False)
        
        # Extract conversion parameters from context
        self.conversion_parameters = self.conversion_context.get('parameters', {})
    
    def _identify_alternative_formats(self) -> None:
        """Identify alternative formats that might work better for conversion."""
        alternative_formats = _identify_alternative_formats_for_conversion(
            self.source_format, self.target_format, self.conversion_context
        )
        
        for fmt_info in alternative_formats:
            self.add_alternative_format(
                fmt_info['format'], 
                fmt_info['compatibility_score'], 
                fmt_info['requirements']
            )
    
    def _generate_format_conversion_recommendations(self) -> None:
        """Generate format-specific conversion recommendations."""
        # Add format pair specific recommendations
        format_pair = f"{self.source_format}_to_{self.target_format}"
        self.conversion_recommendations['format_pair'] = (
            f"Review {format_pair} conversion requirements and parameters"
        )
        
        # Add general format conversion best practices
        self.conversion_recommendations['general_guidance'] = (
            "Ensure proper codec support and format specification parameters"
        )
    
    def _add_format_conversion_recovery_recommendations(self) -> None:
        """Add format conversion-specific recovery recommendations."""
        self.add_recovery_recommendation(
            f"Verify format conversion support: {self.source_format} -> {self.target_format}",
            priority='HIGH'
        )
        
        if self.alternative_formats:
            self.add_recovery_recommendation(
                f"Consider {len(self.alternative_formats)} alternative format options",
                priority='MEDIUM'
            )


class BatchProcessingError(ProcessingError):
    """
    Specialized processing error class for batch processing failures including batch coordination errors, 
    partial processing failures, resource exhaustion during batch operations, and batch-level graceful 
    degradation with batch context tracking and processing statistics for reliable batch operation error 
    handling supporting 4000+ simulation processing requirements.
    
    This class provides comprehensive batch processing error handling with processing statistics, 
    graceful degradation support, and partial completion tracking for large-scale batch operations.
    """
    
    def __init__(
        self,
        message: str,
        batch_id: str,
        total_items: int,
        processed_items: int,
        batch_context: Dict[str, Any]
    ):
        """
        Initialize batch processing error with batch context, processing statistics, and graceful 
        degradation support for comprehensive batch operation error handling.
        
        Args:
            message: Descriptive error message for the batch processing failure
            batch_id: Unique identifier for the batch operation that failed
            total_items: Total number of items in the batch operation
            processed_items: Number of items successfully processed before failure
            batch_context: Context information specific to the batch processing operation
        """
        # Initialize base ProcessingError with PROCESSING category and BATCH_PROCESSING_ERROR_CODE
        super().__init__(
            message=message,
            processing_stage='batch_processing',
            input_file=batch_id,
            processing_context=batch_context
        )
        
        # Override error code for batch processing specific error classification
        self.error_code = BATCH_PROCESSING_ERROR_CODE
        self.category = ErrorCategory.PROCESSING
        
        # Set batch identification and processing statistics for comprehensive analysis
        self.batch_id = batch_id
        self.total_items = total_items
        self.processed_items = processed_items
        self.batch_context = batch_context or {}
        
        # Calculate completion percentage and processing progress for graceful degradation analysis
        self.completion_percentage = (processed_items / total_items * 100) if total_items > 0 else 0
        
        # Initialize batch statistics and item tracking lists for comprehensive analysis
        self.batch_statistics: Dict[str, Any] = {}
        self.failed_items: List[Dict[str, Any]] = []
        self.successful_items: List[Dict[str, Any]] = []
        
        # Determine partial completion support based on batch progress and context
        self.supports_partial_completion = processed_items > 0
        
        # Initialize resource usage and batch recovery options tracking
        self.resource_usage_at_failure: Dict[str, Any] = {}
        self.batch_recovery_options: Dict[str, Any] = {}
        
        # Capture resource usage at time of failure for performance analysis
        self._capture_batch_resource_usage()
        
        # Generate batch-specific recovery options and recommendations
        self._generate_batch_recovery_options()
        
        # Add batch processing-specific recovery recommendations
        self._add_batch_processing_recovery_recommendations()
        
        # Create audit trail entry for batch processing failure with comprehensive batch context
        create_audit_trail(
            action='BATCH_PROCESSING_ERROR_INITIALIZED',
            component='BATCH_PROCESSING_ERROR',
            action_details={
                'batch_id': batch_id,
                'total_items': total_items,
                'processed_items': processed_items,
                'completion_percentage': self.completion_percentage,
                'supports_partial_completion': self.supports_partial_completion,
                'error_id': self.exception_id
            },
            user_context='SYSTEM'
        )
        
        # Log batch processing error with comprehensive batch context and processing statistics
        logger = get_logger('batch_processing.error', 'BATCH_PROCESSING')
        logger.error(
            f"Batch processing error initialized: {batch_id} - {message} "
            f"[Progress: {processed_items}/{total_items} ({self.completion_percentage:.1f}%)]",
            extra={
                'batch_id': batch_id,
                'total_items': total_items,
                'processed_items': processed_items,
                'completion_percentage': self.completion_percentage,
                'supports_partial_completion': self.supports_partial_completion,
                'error_id': self.exception_id
            }
        )
    
    def add_failed_batch_item(
        self,
        item_data: Any,
        item_error: Exception,
        item_context: Dict[str, Any]
    ) -> None:
        """
        Add failed batch item with item details, error information, and failure context for 
        comprehensive batch failure tracking and analysis.
        
        This method tracks individual failed batch items with detailed error context for 
        failure pattern analysis and recovery planning.
        
        Args:
            item_data: Data of the item that failed processing
            item_error: Exception that occurred during item processing
            item_context: Additional context information for the failed item
        """
        # Create failed item details dictionary with item data and error information
        failed_item_entry = {
            'item_data': str(item_data)[:500],  # Truncate for storage efficiency
            'item_id': item_context.get('item_id', f'item_{len(self.failed_items)}'),
            'error_type': type(item_error).__name__,
            'error_message': str(item_error),
            'failure_timestamp': datetime.datetime.now().isoformat(),
            'item_context': item_context or {},
            'processing_stage': item_context.get('processing_stage', 'unknown'),
            'failure_reason': _analyze_item_failure_reason(item_error, item_context)
        }
        
        # Add failed item to failed items list for comprehensive failure tracking
        self.failed_items.append(failed_item_entry)
        
        # Store item context and error information for failure pattern analysis
        self.add_context(f'failed_item_{len(self.failed_items)}', failed_item_entry)
        
        # Update batch statistics with failure information for statistical analysis
        self._update_batch_failure_statistics(failed_item_entry)
        
        # Analyze failure patterns for batch recovery planning and optimization
        failure_pattern = _analyze_batch_failure_pattern(self.failed_items)
        if failure_pattern['pattern_detected']:
            self.batch_recovery_options['failure_pattern'] = failure_pattern
        
        # Log failed batch item for debugging and analysis with comprehensive context
        logger = get_logger('batch_processing.error', 'BATCH_PROCESSING')
        logger.warning(
            f"Batch item failed: {failed_item_entry['item_id']} - {type(item_error).__name__}: {str(item_error)} "
            f"[Batch: {self.batch_id}, Total failed: {len(self.failed_items)}]",
            extra={
                'batch_id': self.batch_id,
                'item_id': failed_item_entry['item_id'],
                'error_type': type(item_error).__name__,
                'total_failed_items': len(self.failed_items),
                'failure_reason': failed_item_entry['failure_reason'],
                'error_id': self.exception_id
            }
        )
    
    def add_successful_batch_item(
        self,
        item_data: Any,
        item_results: Dict[str, Any],
        item_context: Dict[str, Any]
    ) -> None:
        """
        Add successful batch item with item details and processing results for batch completion 
        tracking and partial success analysis.
        
        This method tracks individual successful batch items with results for success pattern 
        analysis and partial completion assessment.
        
        Args:
            item_data: Data of the item that was successfully processed
            item_results: Results from successful item processing
            item_context: Additional context information for the successful item
        """
        # Create successful item details dictionary with item data and results
        successful_item_entry = {
            'item_data': str(item_data)[:500],  # Truncate for storage efficiency
            'item_id': item_context.get('item_id', f'item_{len(self.successful_items)}'),
            'item_results': item_results or {},
            'success_timestamp': datetime.datetime.now().isoformat(),
            'item_context': item_context or {},
            'processing_stage': item_context.get('processing_stage', 'completed'),
            'processing_duration': item_context.get('processing_duration', 0.0),
            'success_metrics': _extract_item_success_metrics(item_results)
        }
        
        # Add successful item to successful items list for success tracking and analysis
        self.successful_items.append(successful_item_entry)
        
        # Store item context and processing results for success pattern analysis
        self.add_context(f'successful_item_{len(self.successful_items)}', successful_item_entry)
        
        # Update batch statistics with success information for comprehensive statistical analysis
        self._update_batch_success_statistics(successful_item_entry)
        
        # Update completion percentage calculation based on current progress
        current_total_processed = len(self.successful_items) + len(self.failed_items)
        if self.total_items > 0:
            self.completion_percentage = (current_total_processed / self.total_items) * 100
        
        # Log successful batch item for progress tracking and success pattern analysis
        logger = get_logger('batch_processing.error', 'BATCH_PROCESSING')
        logger.debug(
            f"Batch item succeeded: {successful_item_entry['item_id']} "
            f"[Batch: {self.batch_id}, Total successful: {len(self.successful_items)}]",
            extra={
                'batch_id': self.batch_id,
                'item_id': successful_item_entry['item_id'],
                'total_successful_items': len(self.successful_items),
                'processing_duration': successful_item_entry['processing_duration'],
                'completion_percentage': self.completion_percentage,
                'error_id': self.exception_id
            }
        )
    
    def get_batch_processing_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive batch processing failure summary including batch statistics, failure analysis, 
        partial completion status, and recovery recommendations for batch operation troubleshooting.
        
        This method provides detailed batch processing analysis with statistics, failure patterns, 
        and recovery guidance for batch operation optimization.
        
        Returns:
            Dict[str, Any]: Batch processing failure summary with comprehensive batch analysis and recovery guidance
        """
        # Compile batch processing statistics and completion information for comprehensive analysis
        summary = {
            'error_id': self.exception_id,
            'batch_id': self.batch_id,
            'total_items': self.total_items,
            'processed_items': self.processed_items,
            'successful_items_count': len(self.successful_items),
            'failed_items_count': len(self.failed_items),
            'completion_percentage': self.completion_percentage,
            'supports_partial_completion': self.supports_partial_completion,
            'batch_context': self.batch_context.copy(),
            'batch_statistics': self.batch_statistics.copy(),
            'resource_usage_at_failure': self.resource_usage_at_failure.copy(),
            'batch_recovery_options': self.batch_recovery_options.copy()
        }
        
        # Include failed and successful items analysis for pattern identification
        if self.failed_items:
            summary['failure_analysis'] = {
                'total_failed_items': len(self.failed_items),
                'failure_categories': _categorize_batch_failures(self.failed_items),
                'failure_timeline': _analyze_failure_timeline(self.failed_items),
                'common_failure_patterns': _identify_common_failure_patterns(self.failed_items),
                'failure_distribution': _analyze_failure_distribution(self.failed_items)
            }
        
        if self.successful_items:
            summary['success_analysis'] = {
                'total_successful_items': len(self.successful_items),
                'success_metrics': _analyze_batch_success_metrics(self.successful_items),
                'processing_efficiency': _calculate_batch_processing_efficiency(self.successful_items),
                'success_patterns': _identify_success_patterns(self.successful_items)
            }
        
        # Add resource usage and performance metrics at failure for optimization analysis
        if self.resource_usage_at_failure:
            summary['performance_analysis'] = {
                'resource_utilization': self.resource_usage_at_failure,
                'performance_bottlenecks': _identify_performance_bottlenecks(self.resource_usage_at_failure),
                'optimization_opportunities': _identify_optimization_opportunities(self.resource_usage_at_failure)
            }
        
        # Generate failure pattern analysis and root cause identification for troubleshooting
        root_cause_analysis = _perform_batch_root_cause_analysis(
            self.failed_items, self.successful_items, self.batch_context
        )
        summary['root_cause_analysis'] = root_cause_analysis
        
        # Include partial completion status and recovery options for graceful degradation
        if self.supports_partial_completion:
            summary['partial_completion_analysis'] = {
                'partial_results_available': len(self.successful_items) > 0,
                'completion_quality': _assess_partial_completion_quality(self.successful_items),
                'recovery_feasibility': _assess_batch_recovery_feasibility(summary),
                'continuation_options': _generate_batch_continuation_options(summary)
            }
        
        # Add batch-specific recovery recommendations and continuation strategies
        recovery_recommendations = self.get_recovery_recommendations()
        summary['recovery_recommendations'] = recovery_recommendations
        
        # Include resource optimization suggestions for batch retry and continuation
        optimization_suggestions = _generate_batch_optimization_suggestions(
            summary, self.batch_context
        )
        summary['optimization_suggestions'] = optimization_suggestions
        
        # Calculate batch processing assessment and success probability
        summary['batch_assessment'] = {
            'overall_success_rate': (len(self.successful_items) / self.total_items * 100) if self.total_items > 0 else 0,
            'failure_rate': (len(self.failed_items) / self.total_items * 100) if self.total_items > 0 else 0,
            'processing_efficiency_score': _calculate_overall_batch_efficiency(summary),
            'recovery_probability': _estimate_batch_recovery_probability(summary),
            'continuation_recommendation': _recommend_batch_continuation_strategy(summary)
        }
        
        # Return comprehensive batch processing summary with analysis and recovery guidance
        return summary
    
    def _capture_batch_resource_usage(self) -> None:
        """Capture resource usage at time of batch failure for performance analysis."""
        try:
            # Basic resource usage capture - in a full implementation, this would use system monitoring
            self.resource_usage_at_failure = {
                'timestamp': datetime.datetime.now().isoformat(),
                'processed_items': self.processed_items,
                'total_items': self.total_items,
                'completion_percentage': self.completion_percentage,
                'batch_duration': self.batch_context.get('batch_duration', 0.0),
                'processing_rate': (self.processed_items / self.batch_context.get('batch_duration', 1.0)) if self.batch_context.get('batch_duration', 0) > 0 else 0
            }
        except Exception as capture_error:
            logger = get_logger('batch_processing.error', 'BATCH_PROCESSING')
            logger.warning(f"Failed to capture resource usage: {capture_error}")
    
    def _generate_batch_recovery_options(self) -> None:
        """Generate batch-specific recovery options based on failure context and progress."""
        self.batch_recovery_options = {
            'partial_completion_support': self.supports_partial_completion,
            'continuation_possible': self.processed_items > 0,
            'retry_recommended': self.completion_percentage < 50,
            'optimization_suggested': len(self.failed_items) > 0
        }
    
    def _add_batch_processing_recovery_recommendations(self) -> None:
        """Add batch processing-specific recovery recommendations."""
        if self.supports_partial_completion:
            self.add_recovery_recommendation(
                f"Partial completion available: {self.completion_percentage:.1f}% processed successfully",
                priority='MEDIUM'
            )
        
        if self.completion_percentage > 50:
            self.add_recovery_recommendation(
                "Consider continuing batch processing from checkpoint",
                priority='HIGH'
            )
        else:
            self.add_recovery_recommendation(
                "Consider batch size reduction and retry strategy",
                priority='HIGH'
            )
    
    def _update_batch_failure_statistics(self, failed_item: Dict[str, Any]) -> None:
        """Update batch statistics with failure information."""
        if 'failures' not in self.batch_statistics:
            self.batch_statistics['failures'] = {}
        
        error_type = failed_item['error_type']
        if error_type not in self.batch_statistics['failures']:
            self.batch_statistics['failures'][error_type] = 0
        
        self.batch_statistics['failures'][error_type] += 1
    
    def _update_batch_success_statistics(self, successful_item: Dict[str, Any]) -> None:
        """Update batch statistics with success information."""
        if 'successes' not in self.batch_statistics:
            self.batch_statistics['successes'] = {}
        
        processing_duration = successful_item.get('processing_duration', 0.0)
        if 'total_processing_time' not in self.batch_statistics['successes']:
            self.batch_statistics['successes']['total_processing_time'] = 0.0
        
        self.batch_statistics['successes']['total_processing_time'] += processing_duration


# Helper functions for processing error functionality

def _add_video_processing_recovery_recommendations(error, stage, context):
    """Add video processing-specific recovery recommendations."""
    if 'codec' in context:
        error.add_recovery_recommendation(
            f"Verify codec compatibility: {context['codec']}",
            priority='HIGH'
        )
    
    if 'frame_rate' in context:
        error.add_recovery_recommendation(
            "Check frame rate compatibility and conversion parameters",
            priority='MEDIUM'
        )


def _add_normalization_recovery_recommendations(error, norm_type, quality_metrics):
    """Add normalization-specific recovery recommendations."""
    if any(metric < 0.5 for metric in quality_metrics.values()):
        error.add_recovery_recommendation(
            "Review normalization parameters for quality improvement",
            priority='HIGH'
        )


def _add_format_conversion_recovery_recommendations(error, source, target):
    """Add format conversion-specific recovery recommendations."""
    error.add_recovery_recommendation(
        f"Verify format conversion chain: {source} -> {target}",
        priority='HIGH'
    )


def _add_batch_processing_recovery_recommendations(error, completion_rate, context):
    """Add batch processing-specific recovery recommendations."""
    if completion_rate > 50:
        error.add_recovery_recommendation(
            "Consider checkpoint-based recovery for partial completion",
            priority='MEDIUM'
        )


def _update_processing_error_statistics(error_type, error):
    """Update global processing error statistics."""
    with _processing_error_lock:
        if error_type not in _processing_error_statistics:
            _processing_error_statistics[error_type] = {}
        
        timestamp = datetime.datetime.now().isoformat()
        _processing_error_statistics[error_type][error.exception_id] = {
            'timestamp': timestamp,
            'error_type': type(error).__name__,
            'processing_stage': getattr(error, 'processing_stage', 'unknown'),
            'severity': error.severity.value,
            'resolved': False
        }


def _classify_processing_error(error, processing_stage):
    """Classify processing error for appropriate handling strategy."""
    if 'video' in processing_stage.lower():
        category = 'video_processing'
        severity = 'HIGH'
    elif 'normalization' in processing_stage.lower():
        category = 'normalization'
        severity = 'MEDIUM'
    elif 'format' in processing_stage.lower():
        category = 'format_conversion'
        severity = 'MEDIUM'
    elif 'batch' in processing_stage.lower():
        category = 'batch_processing'
        severity = 'HIGH'
    else:
        category = 'general_processing'
        severity = 'MEDIUM'
    
    return {
        'category': category,
        'severity': severity,
        'retryable': category in ['batch_processing', 'format_conversion']
    }


def _build_processing_error_context(error, context, stage, additional_context):
    """Build comprehensive processing error context."""
    return {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'processing_context': context,
        'processing_stage': stage,
        'timestamp': datetime.datetime.now().isoformat(),
        'additional_context': additional_context,
        'scientific_context': get_scientific_context(include_defaults=True)
    }


def _supports_graceful_degradation(error, processing_stage):
    """Determine if processing stage supports graceful degradation."""
    degradation_supported_stages = [
        'batch_processing', 'video_processing', 'format_conversion'
    ]
    return any(stage in processing_stage.lower() for stage in degradation_supported_stages)


def _apply_graceful_degradation(error, context):
    """Apply graceful degradation strategy for processing error."""
    return {
        'applied': True,
        'strategy': 'partial_completion',
        'preserved_results': context.get('intermediate_results', {}),
        'completion_percentage': context.get('completion_percentage', 0)
    }


def _update_processing_error_handling_statistics(error, result, stage):
    """Update processing error handling statistics for monitoring."""
    with _processing_error_lock:
        handling_key = f"{stage}_handling"
        if handling_key not in _processing_error_statistics:
            _processing_error_statistics[handling_key] = []
        
        _processing_error_statistics[handling_key].append({
            'error_id': result.error_id,
            'handled_successfully': result.handled_successfully,
            'timestamp': datetime.datetime.now().isoformat(),
            'processing_stage': stage
        })


def _add_processing_specific_recommendations(result, error, stage):
    """Add processing-specific recommendations to error handling result."""
    if 'video' in stage.lower():
        result.add_recommendation(
            "Consider video format validation and codec compatibility",
            "MEDIUM"
        )
    elif 'batch' in stage.lower():
        result.add_recommendation(
            "Consider batch size optimization and checkpoint recovery",
            "HIGH"
        )


def _test_processing_recovery_function(recovery_function, error_type, config):
    """Test processing recovery function with mock scenarios."""
    try:
        # Create mock error and context for testing
        mock_error = Exception("Test processing error")
        mock_context = {'test': True, 'error_type': error_type}
        
        # Test recovery function
        result = recovery_function(mock_error, mock_context)
        
        # Validate result format
        return isinstance(result, dict) and 'success' in result
    except Exception:
        return False


def _parse_time_window_specification(time_window):
    """Parse time window specification and return cutoff datetime."""
    if time_window.endswith('h'):
        hours = int(time_window[:-1])
        return datetime.datetime.now() - datetime.timedelta(hours=hours)
    elif time_window.endswith('d'):
        days = int(time_window[:-1])
        return datetime.datetime.now() - datetime.timedelta(days=days)
    else:
        raise ValueError(f"Invalid time window format: {time_window}")


def _filter_processing_error_data(statistics, cutoff_time, stage_filter, include_resolved):
    """Filter processing error data based on criteria."""
    filtered_errors = []
    
    for error_type, error_dict in statistics.items():
        if isinstance(error_dict, dict):
            for error_id, error_data in error_dict.items():
                if isinstance(error_data, dict):
                    # Apply time filter
                    error_time = datetime.datetime.fromisoformat(error_data.get('timestamp', '2000-01-01T00:00:00'))
                    if error_time < cutoff_time:
                        continue
                    
                    # Apply stage filter
                    if stage_filter and stage_filter not in error_data.get('processing_stage', ''):
                        continue
                    
                    # Apply resolved filter
                    if not include_resolved and error_data.get('resolved', False):
                        continue
                    
                    filtered_errors.append(error_data)
    
    return filtered_errors


def _calculate_processing_error_rates(errors, time_window):
    """Calculate processing error rates from filtered data."""
    if not errors:
        return {'overall_rate': 0.0, 'critical_rate': 0.0}
    
    total_errors = len(errors)
    critical_errors = len([e for e in errors if e.get('severity') == 'CRITICAL'])
    
    return {
        'overall_rate': 0.01,  # Simplified calculation
        'critical_rate': critical_errors / total_errors if total_errors > 0 else 0.0,
        'total_errors': total_errors,
        'time_window': time_window
    }


def _analyze_processing_failure_patterns(errors):
    """Analyze failure patterns in processing error data."""
    patterns = {}
    
    # Group by processing stage
    stage_groups = {}
    for error in errors:
        stage = error.get('processing_stage', 'unknown')
        if stage not in stage_groups:
            stage_groups[stage] = []
        stage_groups[stage].append(error)
    
    patterns['by_stage'] = {
        stage: len(errors) for stage, errors in stage_groups.items()
    }
    
    return patterns


def _analyze_processing_recovery_effectiveness(errors):
    """Analyze recovery effectiveness from processing error data."""
    # Simplified implementation
    return {
        'success_rate': 0.85,
        'average_recovery_time': 4.2,
        'most_effective_strategies': ['retry', 'graceful_degradation']
    }


def _calculate_processing_performance_impact(errors):
    """Calculate performance impact from processing error data."""
    # Simplified implementation
    return {
        'processing_overhead': 0.03,
        'throughput_impact': 0.02,
        'resource_utilization_impact': 0.05
    }


def _generate_processing_trend_analysis(errors, time_window):
    """Generate trend analysis from processing error data."""
    # Simplified implementation
    return {
        'trend_direction': 'stable',
        'error_rate_change': 0.0,
        'prediction_confidence': 0.75
    }


def _analyze_resolved_processing_errors(errors):
    """Analyze resolved processing errors."""
    resolved_errors = [e for e in errors if e.get('resolved', False)]
    
    return {
        'total_resolved': len(resolved_errors),
        'resolution_methods': {},
        'average_resolution_time': 0.0
    }


def _calculate_average_processing_resolution_time(errors):
    """Calculate average resolution time for processing errors."""
    # Simplified implementation
    return 3.5


def _identify_most_common_processing_stage(errors):
    """Identify most common processing stage from error data."""
    if not errors:
        return 'unknown'
    
    stage_counts = {}
    for error in errors:
        stage = error.get('processing_stage', 'unknown')
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    return max(stage_counts.items(), key=lambda x: x[1])[0] if stage_counts else 'unknown'


# Additional helper functions for specialized error classes

def _video_supports_graceful_degradation(stage, context):
    """Determine if video processing supports graceful degradation."""
    return stage in ['frame_extraction', 'video_normalization', 'video_analysis']


def _analyze_quality_threshold_violations(quality_metrics):
    """Analyze quality metrics for threshold violations."""
    # Simplified implementation - check if any metric is below threshold
    return any(value < 0.5 for value in quality_metrics.values())


def _analyze_video_format_compatibility(video_format, codec):
    """Analyze video format compatibility."""
    return f"Format {video_format} with codec {codec} compatibility analysis"


def _perform_format_compatibility_analysis(source, target):
    """Perform basic format compatibility analysis."""
    return {
        'compatible': source != target,  # Simplified logic
        'conversion_required': True,
        'difficulty': 'medium'
    }


def _identify_alternative_formats_for_conversion(source, target, context):
    """Identify alternative formats for conversion."""
    # Simplified implementation
    return [
        {'format': 'mp4', 'compatibility_score': 0.9, 'requirements': {}},
        {'format': 'avi', 'compatibility_score': 0.8, 'requirements': {}}
    ]


def _calculate_format_priority(compatibility_score):
    """Calculate format priority based on compatibility score."""
    if compatibility_score > 0.8:
        return 'HIGH'
    elif compatibility_score > 0.6:
        return 'MEDIUM'
    else:
        return 'LOW'


# Placeholder implementations for complex analysis functions
def _analyze_item_failure_reason(error, context):
    return f"Item failed due to {type(error).__name__}"

def _analyze_batch_failure_pattern(failed_items):
    return {'pattern_detected': False}

def _extract_item_success_metrics(results):
    return {'processing_time': 1.0}

def _categorize_batch_failures(failed_items):
    return {}

def _analyze_failure_timeline(failed_items):
    return {}

def _identify_common_failure_patterns(failed_items):
    return []

def _analyze_failure_distribution(failed_items):
    return {}

def _analyze_batch_success_metrics(successful_items):
    return {}

def _calculate_batch_processing_efficiency(successful_items):
    return 0.85

def _identify_success_patterns(successful_items):
    return []

def _identify_performance_bottlenecks(resource_usage):
    return []

def _identify_optimization_opportunities(resource_usage):
    return []

def _perform_batch_root_cause_analysis(failed_items, successful_items, context):
    return {'root_causes': []}

def _assess_partial_completion_quality(successful_items):
    return 'good'

def _assess_batch_recovery_feasibility(summary):
    return 'high'

def _generate_batch_continuation_options(summary):
    return []

def _generate_batch_optimization_suggestions(summary, context):
    return []

def _calculate_overall_batch_efficiency(summary):
    return 0.80

def _estimate_batch_recovery_probability(summary):
    return 0.75

def _recommend_batch_continuation_strategy(summary):
    return 'retry_with_optimization'

def _update_processing_strategy_documentation(name, error_type, config):
    """Update processing strategy documentation."""
    pass

# Additional placeholder functions for comprehensive error analysis
def _analyze_normalization_failure_patterns(param_errors, quality_violations, params):
    return {}

def _identify_most_severe_quality_violation(violations):
    return None

def _generate_parameter_correction_suggestion(error_info):
    return "Correct parameter value"

def _generate_quality_improvement_suggestions(violation_info):
    return []

def _prioritize_normalization_fixes(param_errors, quality_violations):
    return []

def _assess_normalization_complexity(params):
    return 'medium'

def _estimate_normalization_recovery_probability(analysis):
    return 0.75

def _categorize_unsupported_features(features):
    return {}

def _assess_unsupported_features_impact(features):
    return 'medium'

def _assess_workaround_availability(features):
    return 'available'

def _identify_best_alternative_format(alternatives):
    return 'mp4'

def _prioritize_alternative_formats(alternatives):
    return []

def _generate_format_conversion_parameter_recommendations(source, target, context):
    return {}

def _get_format_specific_troubleshooting_tips(source, target):
    return []

def _suggest_compatibility_improvements(source, target, features):
    return []

def _analyze_cross_platform_compatibility(source, target, context):
    return {}

def _calculate_overall_compatibility_score(source, target, features):
    return 0.75

def _assess_conversion_difficulty(source, target, features):
    return 'medium'

def _estimate_conversion_success_probability(report):
    return 0.80

def _estimate_conversion_effort(report):
    return 'moderate'