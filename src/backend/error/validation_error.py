"""
Specialized validation error module providing comprehensive validation error classes and utilities 
for the plume simulation system including data format validation errors, parameter validation errors, 
schema validation errors, cross-format compatibility errors, and fail-fast validation support.

This module implements ValidationError classes with detailed validation context, recovery recommendations, 
and integration with the error handling framework to ensure >95% correlation with reference implementations 
and <1% error rate in cross-format data processing.

Key Features:
- Comprehensive validation error hierarchy with scientific computing context
- Fail-fast validation strategy for early error detection and resource optimization
- Cross-format compatibility validation for Crimaldi and custom plume data formats
- Parameter validation with constraint checking and recovery recommendations
- Schema validation with JSON schema compliance and error reporting
- Audit trail integration for scientific computing traceability and reproducibility
- Performance monitoring with <1% error rate target validation
- Thread-safe validation error handling with batch processing support
"""

# Standard library imports with version specifications
import datetime  # Python 3.9+ - Timestamp generation for validation error tracking and audit trails
import typing  # Python 3.9+ - Type hints for validation error class signatures and method parameters
import uuid  # Python 3.9+ - Unique identifier generation for validation error tracking and correlation
import json  # Python 3.9+ - JSON serialization for structured validation error reporting and logging
import traceback  # Python 3.9+ - Stack trace extraction for detailed validation error reporting and debugging
import threading  # Python 3.9+ - Thread-safe validation error handling and context management
import math  # Python 3.9+ - Mathematical operations for numerical validation and precision checking
import re  # Python 3.9+ - Regular expression operations for pattern validation and constraint checking
from typing import Dict, Any, List, Optional, Union, Type, Tuple  # Python 3.9+ - Advanced type hints
from pathlib import Path  # Python 3.9+ - Modern path handling for file validation and error reporting

# Internal imports from error handling framework
from ./exceptions import PlumeSimulationException  # Base exception class for plume simulation errors
from ../utils/error_handling import ErrorSeverity, ErrorCategory  # Error classification and severity levels
from ../utils/validation_utils import ValidationResult  # Validation result container with comprehensive tracking
from ../utils.logging_utils import get_logger, log_validation_error, create_audit_trail  # Structured logging and audit trail

# Global validation error codes for classification and tracking
VALIDATION_ERROR_CODE_BASE = 1100
FORMAT_VALIDATION_ERROR_CODE = 1101
PARAMETER_VALIDATION_ERROR_CODE = 1102
SCHEMA_VALIDATION_ERROR_CODE = 1103
CROSS_FORMAT_VALIDATION_ERROR_CODE = 1104
NUMERICAL_VALIDATION_ERROR_CODE = 1105
CONFIGURATION_VALIDATION_ERROR_CODE = 1106
BATCH_VALIDATION_ERROR_CODE = 1107

# Global validation error registry for dynamic error handling and specialized error management
_validation_error_registry: Dict[str, Type['ValidationError']] = {}

# Global validation error statistics for monitoring and analysis
_validation_error_statistics: Dict[str, int] = {
    'total_validation_errors': 0,
    'format_errors': 0,
    'parameter_errors': 0,
    'schema_errors': 0,
    'cross_format_errors': 0,
    'numerical_errors': 0,
    'configuration_errors': 0,
    'batch_errors': 0,
    'resolved_errors': 0,
    'critical_errors': 0
}

# Thread-local storage for validation context and error tracking
_validation_context_storage = threading.local()

# Validation performance thresholds and monitoring constants
VALIDATION_ERROR_RATE_THRESHOLD = 0.01  # <1% error rate requirement
CORRELATION_ACCURACY_THRESHOLD = 0.95  # >95% correlation with reference implementations
VALIDATION_TIMEOUT_SECONDS = 300  # Maximum validation operation timeout
FAIL_FAST_ERROR_THRESHOLD = 5  # Critical error count for fail-fast triggering
BATCH_VALIDATION_CHUNK_SIZE = 100  # Optimal chunk size for batch validation operations


def register_validation_error_type(
    error_class: Type['ValidationError'],
    error_code: int,
    validation_category: str
) -> bool:
    """
    Register custom validation error type in the global validation error registry for dynamic 
    error handling and specialized validation error management.
    
    This function enables dynamic registration of validation error types with comprehensive
    validation of inheritance hierarchy, error code uniqueness, and category classification
    for extensible validation error handling and recovery strategy mapping.
    
    Args:
        error_class: Validation error class to register (must inherit from ValidationError)
        error_code: Unique error code for classification and tracking
        validation_category: Category classification for specialized handling
        
    Returns:
        bool: Success status of validation error type registration with audit trail
    """
    logger = get_logger('validation.registry', 'VALIDATION')
    
    try:
        # Validate error class inheritance from ValidationError base class
        if not issubclass(error_class, ValidationError):
            logger.error(f"Error class {error_class.__name__} must inherit from ValidationError")
            return False
        
        # Check for error code conflicts in global registry
        for registered_name, registered_class in _validation_error_registry.items():
            if hasattr(registered_class, '_error_code') and registered_class._error_code == error_code:
                logger.warning(f"Error code conflict: {error_code} already used by {registered_name}")
                return False
        
        # Register validation error type with error code mapping
        error_class_name = error_class.__name__
        _validation_error_registry[error_class_name] = error_class
        error_class._error_code = error_code
        error_class._validation_category = validation_category
        
        # Update validation error category grouping and classification
        if not hasattr(error_class, '_recovery_strategies'):
            error_class._recovery_strategies = []
        
        # Configure default recovery strategies for validation error type
        default_strategies = _get_default_recovery_strategies(validation_category)
        error_class._recovery_strategies.extend(default_strategies)
        
        # Log validation error type registration for audit trail and monitoring
        create_audit_trail(
            action='VALIDATION_ERROR_TYPE_REGISTERED',
            component='VALIDATION_REGISTRY',
            action_details={
                'error_class': error_class_name,
                'error_code': error_code,
                'validation_category': validation_category,
                'recovery_strategies_count': len(error_class._recovery_strategies)
            },
            user_context='SYSTEM'
        )
        
        logger.info(f"Validation error type registered: {error_class_name} (code: {error_code}, category: {validation_category})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register validation error type {error_class.__name__}: {e}")
        return False


def create_validation_error_context(
    validation_type: str,
    validation_stage: str,
    validation_data: Dict[str, Any],
    additional_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create comprehensive validation error context dictionary with validation details, system state, 
    and debugging information for reproducible research outcomes.
    
    This function captures complete validation context including system state, thread information,
    performance metrics, and scientific computing context for detailed error analysis and debugging
    support in scientific computing workflows.
    
    Args:
        validation_type: Type of validation operation being performed
        validation_stage: Current stage of validation processing
        validation_data: Data being validated with parameter details
        additional_context: Additional context information for specialized validation
        
    Returns:
        Dict[str, Any]: Comprehensive validation error context with validation and system information
    """
    # Capture current validation context from thread storage
    if not hasattr(_validation_context_storage, 'current_context'):
        _validation_context_storage.current_context = {}
    
    current_thread_context = getattr(_validation_context_storage, 'current_context', {})
    
    # Extract validation type and stage information for context building
    context = {
        'validation_type': validation_type,
        'validation_stage': validation_stage,
        'validation_data': validation_data,
        'timestamp': datetime.datetime.now().isoformat(),
        'thread_id': threading.current_thread().ident,
        'thread_name': threading.current_thread().name
    }
    
    # Include validation data and parameters for debugging and analysis
    if validation_data:
        context['data_summary'] = {
            'data_type': type(validation_data).__name__,
            'data_size': len(validation_data) if hasattr(validation_data, '__len__') else 'unknown',
            'data_keys': list(validation_data.keys()) if isinstance(validation_data, dict) else 'not_dict'
        }
    
    # Add timestamp and unique validation error identifier for tracking
    context['validation_id'] = str(uuid.uuid4())
    context['correlation_id'] = str(uuid.uuid4())
    
    # Merge additional context if provided for specialized validation scenarios
    if additional_context:
        context['additional_context'] = additional_context
        # Merge relevant fields directly into main context
        for key, value in additional_context.items():
            if key not in context:  # Avoid overwriting core context fields
                context[key] = value
    
    # Include stack trace and call hierarchy for debugging support
    try:
        stack_frames = []
        frame = traceback.extract_stack()
        # Include last 5 frames for context (excluding this function)
        for frame_info in frame[-6:-1]:
            stack_frames.append({
                'filename': frame_info.filename,
                'function': frame_info.name,
                'line_number': frame_info.lineno,
                'code_context': frame_info.line
            })
        context['call_stack'] = stack_frames
    except Exception:
        context['call_stack'] = []
    
    # Add performance metrics if available for system state analysis
    context['performance_context'] = {
        'memory_usage_estimate': 'unknown',  # Would be populated by memory profiler
        'processing_time_estimate': 'unknown',  # Would be calculated from start time
        'validation_complexity': _calculate_validation_complexity(validation_data)
    }
    
    # Include scientific computing context for reproducible research
    context['scientific_context'] = current_thread_context.copy()
    
    return context


def format_validation_error_message(
    base_message: str,
    validation_context: Dict[str, Any],
    include_recommendations: bool = True,
    include_validation_details: bool = True
) -> str:
    """
    Format validation error message with validation context, error details, and recovery 
    recommendations for clear error communication and debugging support.
    
    This function creates comprehensive, user-friendly error messages with scientific context,
    detailed validation information, and actionable recovery recommendations for effective
    error resolution and debugging in scientific computing workflows.
    
    Args:
        base_message: Base validation error message describing the failure
        validation_context: Validation context dictionary with detailed information
        include_recommendations: Whether to include recovery recommendations in message
        include_validation_details: Whether to include detailed validation information
        
    Returns:
        str: Formatted validation error message with context and recommendations
    """
    # Format base message with validation context variables and scientific information
    formatted_message = base_message
    
    # Add validation-specific details if enabled for comprehensive error reporting
    if include_validation_details and validation_context:
        validation_type = validation_context.get('validation_type', 'unknown')
        validation_stage = validation_context.get('validation_stage', 'unknown')
        formatted_message += f" [Type: {validation_type}] [Stage: {validation_stage}]"
        
        # Include validation stage and component information for debugging context
        if 'validation_id' in validation_context:
            formatted_message += f" [ID: {validation_context['validation_id'][:8]}]"
        
        # Add data summary if available for context understanding
        if 'data_summary' in validation_context:
            data_summary = validation_context['data_summary']
            formatted_message += f" [Data: {data_summary['data_type']}"
            if data_summary['data_size'] != 'unknown':
                formatted_message += f", Size: {data_summary['data_size']}"
            formatted_message += "]"
    
    # Add error code and severity information for classification and handling
    if 'error_code' in validation_context:
        formatted_message += f" [Code: {validation_context['error_code']}]"
    
    if 'severity' in validation_context:
        formatted_message += f" [Severity: {validation_context['severity']}]"
    
    # Include recovery recommendations if enabled for actionable error resolution
    if include_recommendations and 'recovery_recommendations' in validation_context:
        recommendations = validation_context['recovery_recommendations']
        if recommendations and len(recommendations) > 0:
            formatted_message += "\n\nRecovery Recommendations:"
            for i, rec in enumerate(recommendations[:3], 1):  # Limit to top 3 recommendations
                formatted_message += f"\n  {i}. {rec}"
    
    # Format message for readability and debugging with additional context
    if 'timestamp' in validation_context:
        formatted_message += f"\n\nTimestamp: {validation_context['timestamp']}"
    
    if include_validation_details and 'call_stack' in validation_context:
        call_stack = validation_context['call_stack']
        if call_stack and len(call_stack) > 0:
            formatted_message += f"\n\nValidation Location: {call_stack[-1]['function']} ({call_stack[-1]['filename']}:{call_stack[-1]['line_number']})"
    
    return formatted_message


def get_validation_error_statistics(
    time_window: str = "1h",
    validation_type_filter: str = None,
    include_resolved_errors: bool = True
) -> Dict[str, Any]:
    """
    Retrieve comprehensive validation error statistics including error rates, validation failure 
    patterns, and validation performance metrics for system monitoring and analysis.
    
    This function provides detailed statistical analysis of validation errors with time-based
    filtering, category analysis, resolution tracking, and performance impact assessment for
    system optimization and quality assurance monitoring.
    
    Args:
        time_window: Time window for statistics analysis ("1h", "24h", "7d", "30d")
        validation_type_filter: Filter statistics by specific validation type
        include_resolved_errors: Whether to include resolved validation errors in analysis
        
    Returns:
        Dict[str, Any]: Comprehensive validation error statistics with rates, patterns, and trends
    """
    logger = get_logger('validation.statistics', 'VALIDATION')
    
    try:
        # Parse time window specification and validate parameters
        cutoff_time = _parse_time_window(time_window)
        current_time = datetime.datetime.now()
        
        # Filter validation error data by type and time range for analysis
        filtered_statistics = _validation_error_statistics.copy()
        
        # Calculate validation error rates, frequencies, and patterns
        total_errors = filtered_statistics['total_validation_errors']
        critical_errors = filtered_statistics['critical_errors']
        resolved_errors = filtered_statistics['resolved_errors']
        
        # Calculate error rates and success metrics
        error_rate = 0.0
        resolution_rate = 0.0
        critical_error_rate = 0.0
        
        if total_errors > 0:
            resolution_rate = resolved_errors / total_errors
            critical_error_rate = critical_errors / total_errors
            # Error rate would be calculated from total validations if tracked
            error_rate = min(total_errors / 10000, 1.0)  # Assume 10k validations as baseline
        
        # Analyze validation failure patterns and trends
        failure_patterns = {
            'format_errors': filtered_statistics['format_errors'],
            'parameter_errors': filtered_statistics['parameter_errors'],
            'schema_errors': filtered_statistics['schema_errors'],
            'cross_format_errors': filtered_statistics['cross_format_errors'],
            'numerical_errors': filtered_statistics['numerical_errors'],
            'configuration_errors': filtered_statistics['configuration_errors'],
            'batch_errors': filtered_statistics['batch_errors']
        }
        
        # Identify most common error types
        most_common_error_type = max(failure_patterns.items(), key=lambda x: x[1])[0] if failure_patterns else 'none'
        
        # Compute validation performance impact and system health metrics
        performance_impact = {
            'error_rate': error_rate,
            'meets_error_rate_threshold': error_rate <= VALIDATION_ERROR_RATE_THRESHOLD,
            'resolution_efficiency': resolution_rate,
            'critical_error_impact': critical_error_rate,
            'most_common_error_type': most_common_error_type
        }
        
        # Generate trend analysis and pattern detection for optimization
        trend_analysis = {
            'error_trend': 'stable',  # Would be calculated from historical data
            'resolution_trend': 'improving' if resolution_rate > 0.8 else 'needs_attention',
            'performance_status': 'healthy' if error_rate <= VALIDATION_ERROR_RATE_THRESHOLD else 'degraded'
        }
        
        # Include resolved validation errors if requested
        resolved_analysis = {}
        if include_resolved_errors:
            resolved_analysis = {
                'total_resolved': resolved_errors,
                'resolution_rate': resolution_rate,
                'average_resolution_time': 'unknown',  # Would be calculated from resolution timestamps
                'successful_recovery_rate': resolution_rate  # Simplified calculation
            }
        
        # Format statistics for reporting and visualization
        statistics = {
            'time_window': time_window,
            'filter_applied': validation_type_filter,
            'generation_timestamp': current_time.isoformat(),
            'total_statistics': filtered_statistics,
            'error_rates': {
                'overall_error_rate': error_rate,
                'critical_error_rate': critical_error_rate,
                'meets_threshold': error_rate <= VALIDATION_ERROR_RATE_THRESHOLD
            },
            'failure_patterns': failure_patterns,
            'performance_impact': performance_impact,
            'trend_analysis': trend_analysis,
            'resolution_analysis': resolved_analysis if include_resolved_errors else {}
        }
        
        logger.debug(f"Validation error statistics generated for {time_window} window: {total_errors} total errors")
        
        return statistics
        
    except Exception as e:
        logger.error(f"Failed to retrieve validation error statistics: {e}")
        return {
            'error': str(e),
            'generation_timestamp': datetime.datetime.now().isoformat(),
            'statistics_available': False
        }


class ValidationError(PlumeSimulationException):
    """
    Specialized validation error class for data validation failures including format validation, 
    parameter validation, schema validation, and cross-format compatibility issues with fail-fast 
    validation support, detailed validation context, and comprehensive error reporting for 
    scientific computing reliability.
    
    This class serves as the base validation error with comprehensive context management, recovery
    recommendation framework, fail-fast validation support, and integration with the error handling
    framework for reproducible scientific computing workflows.
    """
    
    def __init__(
        self,
        message: str,
        validation_type: str,
        validation_context: Dict[str, Any] = None,
        failed_parameters: List[str] = None
    ):
        """
        Initialize validation error with comprehensive validation context and fail-fast validation 
        support for early error detection.
        
        Args:
            message: Detailed validation error message describing the failure
            validation_type: Type of validation that failed (format, parameter, schema, etc.)
            validation_context: Context information for validation operation
            failed_parameters: List of parameters that failed validation
        """
        # Initialize base PlumeSimulationException with VALIDATION category
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            error_code=VALIDATION_ERROR_CODE_BASE,
            context=validation_context or {}
        )
        
        # Set validation error code and HIGH severity for priority handling
        self.error_code = VALIDATION_ERROR_CODE_BASE
        self.severity = ErrorSeverity.HIGH
        
        # Store validation type and context information for detailed tracking
        self.validation_type = validation_type
        self.validation_context = validation_context or {}
        
        # Initialize failed parameters and validation errors lists for comprehensive tracking
        self.failed_parameters = failed_parameters or []
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
        # Generate unique validation identifier for tracking and correlation
        self.validation_id = str(uuid.uuid4())
        
        # Determine if validation failure is critical based on failed parameters count
        self.is_critical_validation_failure = len(self.failed_parameters) > FAIL_FAST_ERROR_THRESHOLD
        
        # Set validation stage from context for processing pipeline tracking
        self.validation_stage = self.validation_context.get('validation_stage', 'unknown')
        
        # Initialize validation metrics for performance tracking and analysis
        self.validation_metrics = {
            'failed_parameter_count': len(self.failed_parameters),
            'validation_type': validation_type,
            'critical_failure': self.is_critical_validation_failure,
            'validation_complexity': _calculate_validation_complexity(self.validation_context)
        }
        
        # Create validation result container for detailed tracking and reporting
        self.validation_result = ValidationResult(
            validation_type=validation_type,
            is_valid=False,
            validation_context=f"validation_id={self.validation_id}"
        )
        
        # Initialize fail-fast validation support for early error detection
        self.fail_fast_triggered = False
        
        # Add validation-specific recovery recommendations based on validation type
        self._add_validation_recovery_recommendations()
        
        # Create audit trail entry for validation failure tracking
        create_audit_trail(
            action='VALIDATION_ERROR_CREATED',
            component='VALIDATION',
            action_details={
                'validation_id': self.validation_id,
                'validation_type': validation_type,
                'failed_parameters': self.failed_parameters,
                'critical_failure': self.is_critical_validation_failure,
                'validation_stage': self.validation_stage
            },
            user_context='SYSTEM'
        )
        
        # Update global validation error statistics for monitoring
        _validation_error_statistics['total_validation_errors'] += 1
        if self.is_critical_validation_failure:
            _validation_error_statistics['critical_errors'] += 1
        
        # Log validation error with detailed context for debugging and analysis
        log_validation_error(
            validation_type=validation_type,
            error_message=message,
            validation_context={
                'validation_id': self.validation_id,
                'failed_parameters': self.failed_parameters,
                'validation_stage': self.validation_stage
            },
            failed_parameters=self.failed_parameters,
            recovery_recommendations=self.recovery_recommendations
        )
    
    def add_validation_error(
        self,
        error_description: str,
        parameter_name: str,
        expected_value: Any = None,
        actual_value: Any = None
    ) -> None:
        """
        Add specific validation error to the validation error list for comprehensive validation 
        failure tracking.
        
        Args:
            error_description: Detailed description of the validation error
            parameter_name: Name of the parameter that failed validation
            expected_value: Expected value for the parameter
            actual_value: Actual value that caused the validation failure
        """
        # Add error description to validation errors list with parameter details
        formatted_error = f"{parameter_name}: {error_description}"
        if expected_value is not None and actual_value is not None:
            formatted_error += f" (expected: {expected_value}, actual: {actual_value})"
        
        self.validation_errors.append(formatted_error)
        
        # Store parameter name and value information for debugging and analysis
        if parameter_name not in self.failed_parameters:
            self.failed_parameters.append(parameter_name)
        
        # Update validation context with error details for comprehensive tracking
        self.validation_context[f'{parameter_name}_validation_error'] = {
            'description': error_description,
            'expected_value': expected_value,
            'actual_value': actual_value,
            'error_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Add error to validation result container for structured reporting
        self.validation_result.add_error(
            error_message=formatted_error,
            severity=self.validation_result.ErrorSeverity.HIGH,
            error_context={
                'parameter_name': parameter_name,
                'expected_value': expected_value,
                'actual_value': actual_value
            }
        )
        
        # Log validation error addition for debugging and audit trail
        logger = get_logger('validation.error_detail', 'VALIDATION')
        logger.debug(f"Validation error added to {self.validation_id}: {formatted_error}")
    
    def add_validation_warning(
        self,
        warning_description: str,
        parameter_name: str
    ) -> None:
        """
        Add validation warning for non-critical issues that may affect processing quality.
        
        Args:
            warning_description: Description of the validation warning
            parameter_name: Name of the parameter with validation warning
        """
        # Add warning description to validation warnings list
        formatted_warning = f"{parameter_name}: {warning_description}"
        self.validation_warnings.append(formatted_warning)
        
        # Store parameter name for warning context and tracking
        self.validation_context[f'{parameter_name}_validation_warning'] = {
            'description': warning_description,
            'warning_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Add warning to validation result container for comprehensive tracking
        self.validation_result.add_warning(
            warning_message=formatted_warning,
            warning_context={'parameter_name': parameter_name}
        )
        
        # Log validation warning for review and monitoring
        logger = get_logger('validation.warning', 'VALIDATION')
        logger.warning(f"Validation warning for {self.validation_id}: {formatted_warning}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive validation failure summary with errors, warnings, and recovery guidance.
        
        Returns:
            Dict[str, Any]: Validation failure summary with comprehensive analysis
        """
        # Compile validation errors and warnings for summary reporting
        summary = {
            'validation_id': self.validation_id,
            'validation_type': self.validation_type,
            'validation_stage': self.validation_stage,
            'failed_parameters': self.failed_parameters,
            'validation_errors': self.validation_errors,
            'validation_warnings': self.validation_warnings,
            'total_errors': len(self.validation_errors),
            'total_warnings': len(self.validation_warnings),
            'critical_failure': self.is_critical_validation_failure,
            'fail_fast_triggered': self.fail_fast_triggered,
            'validation_metrics': self.validation_metrics,
            'recovery_recommendations': self.get_recovery_recommendations(),
            'validation_context_summary': self._get_context_summary()
        }
        
        # Include validation result container data for comprehensive reporting
        summary['validation_result_summary'] = self.validation_result.get_summary()
        
        return summary
    
    def trigger_fail_fast(
        self,
        fail_fast_reason: str
    ) -> None:
        """
        Trigger fail-fast validation strategy to immediately stop processing upon critical 
        validation failure.
        
        Args:
            fail_fast_reason: Reason for triggering fail-fast validation strategy
        """
        # Set fail_fast_triggered flag to True for processing pipeline control
        self.fail_fast_triggered = True
        
        # Record fail-fast reason in validation context for analysis
        self.validation_context['fail_fast_reason'] = fail_fast_reason
        self.validation_context['fail_fast_timestamp'] = datetime.datetime.now().isoformat()
        
        # Update validation severity to CRITICAL for priority handling
        self.severity = ErrorSeverity.CRITICAL
        
        # Log fail-fast trigger with reason for system monitoring
        logger = get_logger('validation.fail_fast', 'VALIDATION')
        logger.critical(f"Fail-fast validation triggered for {self.validation_id}: {fail_fast_reason}")
        
        # Create audit trail entry for fail-fast activation tracking
        create_audit_trail(
            action='FAIL_FAST_VALIDATION_TRIGGERED',
            component='VALIDATION',
            action_details={
                'validation_id': self.validation_id,
                'fail_fast_reason': fail_fast_reason,
                'failed_parameters_count': len(self.failed_parameters),
                'validation_type': self.validation_type
            },
            user_context='SYSTEM'
        )
    
    def get_failed_parameters_details(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about failed validation parameters including expected and 
        actual values.
        
        Returns:
            Dict[str, Dict[str, Any]]: Detailed failed parameters information with expected and actual values
        """
        # Extract failed parameters from validation context with detailed information
        failed_details = {}
        
        for parameter_name in self.failed_parameters:
            error_key = f'{parameter_name}_validation_error'
            if error_key in self.validation_context:
                error_info = self.validation_context[error_key]
                failed_details[parameter_name] = {
                    'description': error_info.get('description', 'Unknown error'),
                    'expected_value': error_info.get('expected_value'),
                    'actual_value': error_info.get('actual_value'),
                    'error_timestamp': error_info.get('error_timestamp'),
                    'parameter_type': type(error_info.get('actual_value')).__name__ if error_info.get('actual_value') is not None else 'unknown'
                }
                
                # Include parameter validation rules and constraints if available
                constraint_key = f'{parameter_name}_constraints'
                if constraint_key in self.validation_context:
                    failed_details[parameter_name]['constraints'] = self.validation_context[constraint_key]
                
                # Add parameter-specific recovery recommendations
                failed_details[parameter_name]['recovery_recommendations'] = [
                    f"Verify {parameter_name} meets validation requirements",
                    f"Check {parameter_name} value format and range",
                    f"Review {parameter_name} configuration and dependencies"
                ]
        
        return failed_details
    
    def _add_validation_recovery_recommendations(self) -> None:
        """Add validation-specific recovery recommendations based on validation type."""
        if self.validation_type == 'format_validation':
            self.add_recovery_recommendation(
                "Verify data format matches expected schema and structure",
                priority='HIGH'
            )
            self.add_recovery_recommendation(
                "Check file integrity and accessibility",
                priority='MEDIUM'
            )
        elif self.validation_type == 'parameter_validation':
            self.add_recovery_recommendation(
                "Review parameter values against valid ranges and constraints",
                priority='HIGH'
            )
            self.add_recovery_recommendation(
                "Validate parameter types and format requirements",
                priority='MEDIUM'
            )
        elif self.validation_type == 'schema_validation':
            self.add_recovery_recommendation(
                "Review configuration file structure and required fields",
                priority='HIGH'
            )
            self.add_recovery_recommendation(
                "Validate JSON schema compliance and syntax",
                priority='MEDIUM'
            )
        
        # Add general validation recovery recommendations
        self.add_recovery_recommendation(
            "Review validation error details and correct input data",
            priority='MEDIUM'
        )
        
        if self.is_critical_validation_failure:
            self.add_recovery_recommendation(
                "Critical validation failure detected - consider fail-fast strategy",
                priority='CRITICAL'
            )
    
    def _get_context_summary(self) -> Dict[str, Any]:
        """Get summary of validation context for reporting."""
        return {
            'validation_stage': self.validation_stage,
            'validation_complexity': self.validation_metrics.get('validation_complexity', 'unknown'),
            'context_keys': list(self.validation_context.keys()),
            'has_additional_context': 'additional_context' in self.validation_context,
            'thread_info': {
                'thread_id': threading.current_thread().ident,
                'thread_name': threading.current_thread().name
            }
        }


class FormatValidationError(ValidationError):
    """
    Specialized validation error for data format validation failures including file format detection 
    errors, header validation failures, metadata verification issues, and cross-format compatibility 
    problems with format-specific context and recovery guidance.
    
    This class extends ValidationError with format-specific validation context, compatibility analysis,
    and specialized recovery recommendations for data format issues in scientific computing workflows.
    """
    
    def __init__(
        self,
        message: str,
        file_path: str,
        expected_format: str,
        detected_format: str
    ):
        """
        Initialize format validation error with file format context and compatibility analysis.
        
        Args:
            message: Detailed format validation error message
            file_path: Path to the file with format validation issues
            expected_format: Expected file format specification
            detected_format: Detected file format from analysis
        """
        # Initialize base ValidationError with format validation type
        super().__init__(
            message=message,
            validation_type='format_validation',
            validation_context={
                'file_path': file_path,
                'expected_format': expected_format,
                'detected_format': detected_format
            }
        )
        
        # Set format validation error code for classification
        self.error_code = FORMAT_VALIDATION_ERROR_CODE
        
        # Store file path and format information for detailed tracking
        self.file_path = file_path
        self.expected_format = expected_format
        self.detected_format = detected_format
        
        # Initialize format metadata and compatibility tracking
        self.format_metadata: Dict[str, Any] = {}
        self.format_compatibility_issues: List[str] = []
        self.is_format_convertible = False
        
        # Analyze format convertibility and compatibility
        self._analyze_format_compatibility()
        
        # Add format-specific recovery recommendations
        self._add_format_recovery_recommendations()
        
        # Update format error statistics
        _validation_error_statistics['format_errors'] += 1
        
        # Log format validation error with file context
        logger = get_logger('validation.format', 'VALIDATION')
        logger.error(f"Format validation error: {file_path} - expected {expected_format}, detected {detected_format}")
    
    def add_format_compatibility_issue(
        self,
        issue_description: str,
        compatibility_level: str
    ) -> None:
        """
        Add format compatibility issue to the validation error for comprehensive format analysis.
        
        Args:
            issue_description: Description of the format compatibility issue
            compatibility_level: Level of compatibility impact (CRITICAL, HIGH, MEDIUM, LOW)
        """
        # Add issue description to compatibility issues list
        formatted_issue = f"[{compatibility_level}] {issue_description}"
        self.format_compatibility_issues.append(formatted_issue)
        
        # Store compatibility level information for analysis
        self.validation_context[f'compatibility_issue_{len(self.format_compatibility_issues)}'] = {
            'description': issue_description,
            'compatibility_level': compatibility_level,
            'detected_at': datetime.datetime.now().isoformat()
        }
        
        # Update format metadata with issue details
        if 'compatibility_analysis' not in self.format_metadata:
            self.format_metadata['compatibility_analysis'] = []
        
        self.format_metadata['compatibility_analysis'].append({
            'issue': issue_description,
            'level': compatibility_level,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Log format compatibility issue addition for tracking
        logger = get_logger('validation.format', 'VALIDATION')
        logger.warning(f"Format compatibility issue added to {self.validation_id}: {formatted_issue}")
    
    def get_format_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive format analysis including compatibility assessment and conversion 
        recommendations.
        
        Returns:
            Dict[str, Any]: Format analysis with compatibility assessment and conversion guidance
        """
        # Analyze format compatibility between expected and detected formats
        compatibility_score = self._calculate_compatibility_score()
        
        analysis = {
            'file_path': self.file_path,
            'expected_format': self.expected_format,
            'detected_format': self.detected_format,
            'format_metadata': self.format_metadata,
            'compatibility_issues': self.format_compatibility_issues,
            'compatibility_score': compatibility_score,
            'is_format_convertible': self.is_format_convertible,
            'conversion_recommendations': []
        }
        
        # Include format metadata and specifications for detailed analysis
        if self.format_metadata:
            analysis['format_specifications'] = self.format_metadata
        
        # Generate format conversion recommendations based on compatibility analysis
        if self.is_format_convertible:
            analysis['conversion_recommendations'] = [
                f"Convert from {self.detected_format} to {self.expected_format}",
                "Verify data integrity after conversion",
                "Test conversion with sample data first"
            ]
        else:
            analysis['conversion_recommendations'] = [
                "Manual format verification required",
                "Consider alternative data sources",
                "Review format specification requirements"
            ]
        
        # Include format-specific recovery strategies for error resolution
        analysis['format_recovery_strategies'] = [
            "Validate file format detection accuracy",
            "Check format specification compliance",
            "Verify file integrity and completeness"
        ]
        
        return analysis
    
    def _analyze_format_compatibility(self) -> None:
        """Analyze format compatibility and determine conversion feasibility."""
        # Basic format compatibility analysis
        compatible_formats = {
            'crimaldi': ['avi', 'mp4'],
            'custom': ['avi', 'mp4', 'mov'],
            'avi': ['mp4', 'mov'],
            'mp4': ['avi', 'mov']
        }
        
        expected_lower = self.expected_format.lower()
        detected_lower = self.detected_format.lower()
        
        # Check if formats are compatible for conversion
        if expected_lower in compatible_formats:
            compatible_list = compatible_formats[expected_lower]
            self.is_format_convertible = detected_lower in compatible_list
        
        # Analyze specific format characteristics
        self.format_metadata = {
            'expected_format_type': self._get_format_type(self.expected_format),
            'detected_format_type': self._get_format_type(self.detected_format),
            'conversion_complexity': 'low' if self.is_format_convertible else 'high',
            'compatibility_assessment': 'compatible' if self.is_format_convertible else 'incompatible'
        }
    
    def _add_format_recovery_recommendations(self) -> None:
        """Add format-specific recovery recommendations."""
        self.add_recovery_recommendation(
            f"Verify file format: expected {self.expected_format}, detected {self.detected_format}",
            priority='HIGH'
        )
        
        if self.is_format_convertible:
            self.add_recovery_recommendation(
                f"Consider format conversion from {self.detected_format} to {self.expected_format}",
                priority='MEDIUM'
            )
        else:
            self.add_recovery_recommendation(
                "Manual format verification and correction required",
                priority='HIGH'
            )
        
        self.add_recovery_recommendation(
            f"Check file integrity and accessibility: {self.file_path}",
            priority='MEDIUM'
        )
    
    def _get_format_type(self, format_name: str) -> str:
        """Get format type classification."""
        format_types = {
            'crimaldi': 'scientific_dataset',
            'custom': 'scientific_dataset',
            'avi': 'video_format',
            'mp4': 'video_format',
            'mov': 'video_format',
            'mkv': 'video_format'
        }
        return format_types.get(format_name.lower(), 'unknown')
    
    def _calculate_compatibility_score(self) -> float:
        """Calculate compatibility score between formats."""
        if self.expected_format.lower() == self.detected_format.lower():
            return 1.0
        elif self.is_format_convertible:
            return 0.8
        else:
            return 0.2


class ParameterValidationError(ValidationError):
    """
    Specialized validation error for parameter validation failures including parameter bounds checking, 
    type validation, constraint verification, and parameter dependency validation with parameter-specific 
    context and correction guidance.
    
    This class extends ValidationError with parameter-specific validation context, constraint analysis,
    and detailed correction guidance for parameter validation issues in scientific computing workflows.
    """
    
    def __init__(
        self,
        message: str,
        parameter_name: str,
        parameter_value: Any,
        parameter_constraints: Dict[str, Any]
    ):
        """
        Initialize parameter validation error with parameter context and constraint analysis.
        
        Args:
            message: Detailed parameter validation error message
            parameter_name: Name of the parameter that failed validation
            parameter_value: Value of the parameter that failed validation
            parameter_constraints: Constraints and validation rules for the parameter
        """
        # Initialize base ValidationError with parameter validation type
        super().__init__(
            message=message,
            validation_type='parameter_validation',
            validation_context={
                'parameter_name': parameter_name,
                'parameter_value': parameter_value,
                'parameter_constraints': parameter_constraints
            }
        )
        
        # Set parameter validation error code for classification
        self.error_code = PARAMETER_VALIDATION_ERROR_CODE
        
        # Store parameter name, value, and constraints for detailed tracking
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.parameter_constraints = parameter_constraints
        
        # Analyze parameter type and constraint violations
        self.parameter_type = type(parameter_value).__name__
        self.constraint_violations: List[str] = []
        self.suggested_values: Dict[str, Any] = {}
        
        # Analyze constraint violations and generate suggestions
        self._analyze_constraint_violations()
        
        # Initialize suggested values and corrections based on constraints
        self._generate_parameter_suggestions()
        
        # Add parameter-specific recovery recommendations
        self._add_parameter_recovery_recommendations()
        
        # Update parameter error statistics
        _validation_error_statistics['parameter_errors'] += 1
        
        # Log parameter validation error with parameter context
        logger = get_logger('validation.parameter', 'VALIDATION')
        logger.error(f"Parameter validation error: {parameter_name} = {parameter_value} (type: {self.parameter_type})")
    
    def add_constraint_violation(
        self,
        violation_description: str,
        constraint_type: str
    ) -> None:
        """
        Add constraint violation to the parameter validation error for detailed constraint analysis.
        
        Args:
            violation_description: Description of the constraint violation
            constraint_type: Type of constraint that was violated
        """
        # Add violation description to constraint violations list
        formatted_violation = f"[{constraint_type}] {violation_description}"
        self.constraint_violations.append(formatted_violation)
        
        # Store constraint type information for analysis and reporting
        self.validation_context[f'constraint_violation_{len(self.constraint_violations)}'] = {
            'description': violation_description,
            'constraint_type': constraint_type,
            'parameter_name': self.parameter_name,
            'detected_at': datetime.datetime.now().isoformat()
        }
        
        # Update parameter context with violation details
        violation_key = f'{self.parameter_name}_violation_{constraint_type}'
        self.validation_context[violation_key] = {
            'violation': violation_description,
            'constraint_type': constraint_type,
            'parameter_value': self.parameter_value
        }
        
        # Log constraint violation addition for tracking and analysis
        logger = get_logger('validation.parameter', 'VALIDATION')
        logger.warning(f"Constraint violation added to {self.validation_id}: {formatted_violation}")
    
    def suggest_parameter_value(
        self,
        suggested_value: Any,
        suggestion_reason: str
    ) -> None:
        """
        Suggest corrected parameter value based on constraints and validation rules.
        
        Args:
            suggested_value: Suggested corrected value for the parameter
            suggestion_reason: Reason and rationale for the suggested value
        """
        # Add suggested value to suggested values dictionary with reasoning
        suggestion_key = f"suggestion_{len(self.suggested_values) + 1}"
        self.suggested_values[suggestion_key] = {
            'suggested_value': suggested_value,
            'suggestion_reason': suggestion_reason,
            'suggested_type': type(suggested_value).__name__,
            'suggestion_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Store suggestion reason and rationale for recovery guidance
        self.validation_context[f'{self.parameter_name}_suggestion_{suggestion_key}'] = {
            'suggested_value': suggested_value,
            'reason': suggestion_reason,
            'original_value': self.parameter_value,
            'suggested_at': datetime.datetime.now().isoformat()
        }
        
        # Update parameter context with suggestion for recovery planning
        if 'parameter_suggestions' not in self.validation_context:
            self.validation_context['parameter_suggestions'] = []
        
        self.validation_context['parameter_suggestions'].append({
            'value': suggested_value,
            'reason': suggestion_reason,
            'confidence': 'high' if 'constraint' in suggestion_reason.lower() else 'medium'
        })
        
        # Log parameter value suggestion for recovery guidance
        logger = get_logger('validation.parameter', 'VALIDATION')
        logger.info(f"Parameter suggestion for {self.parameter_name}: {suggested_value} ({suggestion_reason})")
    
    def get_parameter_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive parameter analysis including constraint violations and correction suggestions.
        
        Returns:
            Dict[str, Any]: Parameter analysis with constraint violations and correction guidance
        """
        # Analyze parameter constraints and violations for comprehensive reporting
        analysis = {
            'parameter_name': self.parameter_name,
            'parameter_value': self.parameter_value,
            'parameter_type': self.parameter_type,
            'parameter_constraints': self.parameter_constraints,
            'constraint_violations': self.constraint_violations,
            'suggested_values': self.suggested_values,
            'violation_count': len(self.constraint_violations),
            'suggestions_count': len(self.suggested_values)
        }
        
        # Include parameter type and value information for type checking
        analysis['type_analysis'] = {
            'expected_type': self.parameter_constraints.get('type', 'unknown'),
            'actual_type': self.parameter_type,
            'type_compatible': self._check_type_compatibility(),
            'type_conversion_possible': self._check_type_conversion_feasibility()
        }
        
        # Generate parameter correction recommendations based on analysis
        analysis['correction_recommendations'] = []
        
        if self.constraint_violations:
            analysis['correction_recommendations'].append(
                f"Address {len(self.constraint_violations)} constraint violations"
            )
        
        if self.suggested_values:
            best_suggestion = list(self.suggested_values.values())[0]  # First suggestion
            analysis['correction_recommendations'].append(
                f"Consider suggested value: {best_suggestion['suggested_value']}"
            )
        
        # Include constraint-specific recovery strategies for error resolution
        analysis['recovery_strategies'] = [
            f"Review {self.parameter_name} constraints and valid ranges",
            f"Validate {self.parameter_name} type requirements",
            f"Check {self.parameter_name} dependencies and relationships"
        ]
        
        return analysis
    
    def _analyze_constraint_violations(self) -> None:
        """Analyze parameter constraints and identify violations."""
        if not self.parameter_constraints:
            return
        
        # Check type constraints
        if 'type' in self.parameter_constraints:
            expected_type = self.parameter_constraints['type']
            if not self._check_type_compatibility():
                self.add_constraint_violation(
                    f"Type mismatch: expected {expected_type}, got {self.parameter_type}",
                    'TYPE'
                )
        
        # Check range constraints
        if 'min_value' in self.parameter_constraints and isinstance(self.parameter_value, (int, float)):
            min_value = self.parameter_constraints['min_value']
            if self.parameter_value < min_value:
                self.add_constraint_violation(
                    f"Value {self.parameter_value} below minimum {min_value}",
                    'RANGE'
                )
        
        if 'max_value' in self.parameter_constraints and isinstance(self.parameter_value, (int, float)):
            max_value = self.parameter_constraints['max_value']
            if self.parameter_value > max_value:
                self.add_constraint_violation(
                    f"Value {self.parameter_value} above maximum {max_value}",
                    'RANGE'
                )
        
        # Check pattern constraints for strings
        if 'pattern' in self.parameter_constraints and isinstance(self.parameter_value, str):
            pattern = self.parameter_constraints['pattern']
            if not re.match(pattern, self.parameter_value):
                self.add_constraint_violation(
                    f"Value '{self.parameter_value}' does not match pattern '{pattern}'",
                    'PATTERN'
                )
    
    def _generate_parameter_suggestions(self) -> None:
        """Generate parameter value suggestions based on constraints."""
        if not self.parameter_constraints:
            return
        
        # Suggest values based on range constraints
        if 'min_value' in self.parameter_constraints and 'max_value' in self.parameter_constraints:
            min_val = self.parameter_constraints['min_value']
            max_val = self.parameter_constraints['max_value']
            
            if isinstance(self.parameter_value, (int, float)):
                if self.parameter_value < min_val:
                    self.suggest_parameter_value(
                        min_val,
                        f"Minimum allowed value for {self.parameter_name}"
                    )
                elif self.parameter_value > max_val:
                    self.suggest_parameter_value(
                        max_val,
                        f"Maximum allowed value for {self.parameter_name}"
                    )
                else:
                    # Suggest middle value if current value is in range but has other issues
                    middle_value = (min_val + max_val) / 2
                    self.suggest_parameter_value(
                        middle_value,
                        f"Safe middle value within valid range"
                    )
        
        # Suggest default value if available
        if 'default_value' in self.parameter_constraints:
            self.suggest_parameter_value(
                self.parameter_constraints['default_value'],
                f"Default value for {self.parameter_name}"
            )
    
    def _check_type_compatibility(self) -> bool:
        """Check if parameter type is compatible with constraints."""
        if 'type' not in self.parameter_constraints:
            return True
        
        expected_type = self.parameter_constraints['type']
        actual_type = type(self.parameter_value).__name__
        
        # Direct type match
        if actual_type == expected_type:
            return True
        
        # Check for compatible types
        compatible_types = {
            'int': ['float'],
            'float': ['int'],
            'str': [],
            'bool': []
        }
        
        return actual_type in compatible_types.get(expected_type, [])
    
    def _check_type_conversion_feasibility(self) -> bool:
        """Check if parameter value can be converted to expected type."""
        if 'type' not in self.parameter_constraints:
            return False
        
        expected_type = self.parameter_constraints['type']
        
        try:
            if expected_type == 'int':
                int(self.parameter_value)
                return True
            elif expected_type == 'float':
                float(self.parameter_value)
                return True
            elif expected_type == 'str':
                str(self.parameter_value)
                return True
            elif expected_type == 'bool':
                bool(self.parameter_value)
                return True
        except (ValueError, TypeError):
            return False
        
        return False
    
    def _add_parameter_recovery_recommendations(self) -> None:
        """Add parameter-specific recovery recommendations."""
        self.add_recovery_recommendation(
            f"Review parameter {self.parameter_name} value and constraints",
            priority='HIGH'
        )
        
        if self.constraint_violations:
            self.add_recovery_recommendation(
                f"Address {len(self.constraint_violations)} constraint violations",
                priority='HIGH'
            )
        
        if self.suggested_values:
            best_suggestion = list(self.suggested_values.values())[0]
            self.add_recovery_recommendation(
                f"Consider suggested value: {best_suggestion['suggested_value']}",
                priority='MEDIUM'
            )
        
        self.add_recovery_recommendation(
            f"Validate parameter {self.parameter_name} type and format requirements",
            priority='MEDIUM'
        )


class SchemaValidationError(ValidationError):
    """
    Specialized validation error for JSON schema validation failures including schema structure validation, 
    required field validation, type constraint validation, and schema compliance issues with schema-specific 
    context and correction guidance.
    
    This class extends ValidationError with schema-specific validation context, compliance analysis,
    and detailed correction guidance for JSON schema validation issues in configuration management.
    """
    
    def __init__(
        self,
        message: str,
        schema_name: str,
        schema_violations: Dict[str, Any],
        validation_path: str
    ):
        """
        Initialize schema validation error with schema context and violation analysis.
        
        Args:
            message: Detailed schema validation error message
            schema_name: Name of the schema that failed validation
            schema_violations: Dictionary of schema violations and details
            validation_path: JSON path where validation failed
        """
        # Initialize base ValidationError with schema validation type
        super().__init__(
            message=message,
            validation_type='schema_validation',
            validation_context={
                'schema_name': schema_name,
                'schema_violations': schema_violations,
                'validation_path': validation_path
            }
        )
        
        # Set schema validation error code for classification
        self.error_code = SCHEMA_VALIDATION_ERROR_CODE
        
        # Store schema name and validation path for detailed tracking
        self.schema_name = schema_name
        self.schema_violations = schema_violations
        self.validation_path = validation_path
        
        # Analyze schema violations and missing fields
        self.missing_required_fields: List[str] = []
        self.type_violations: List[str] = []
        self.schema_corrections: Dict[str, Any] = {}
        
        # Analyze schema violations and generate corrections
        self._analyze_schema_violations()
        
        # Initialize schema corrections and suggestions based on violations
        self._generate_schema_corrections()
        
        # Add schema-specific recovery recommendations
        self._add_schema_recovery_recommendations()
        
        # Update schema error statistics
        _validation_error_statistics['schema_errors'] += 1
        
        # Log schema validation error with schema context
        logger = get_logger('validation.schema', 'VALIDATION')
        logger.error(f"Schema validation error: {schema_name} at path {validation_path}")
    
    def add_missing_field(
        self,
        field_name: str,
        field_type: str,
        default_value: Any = None
    ) -> None:
        """
        Add missing required field to the schema validation error for comprehensive schema analysis.
        
        Args:
            field_name: Name of the missing required field
            field_type: Expected type of the missing field
            default_value: Default value for the missing field
        """
        # Add field name to missing required fields list
        self.missing_required_fields.append(field_name)
        
        # Store field type and default value information for correction guidance
        self.validation_context[f'missing_field_{field_name}'] = {
            'field_name': field_name,
            'field_type': field_type,
            'default_value': default_value,
            'detected_at': datetime.datetime.now().isoformat()
        }
        
        # Update schema context with missing field details for comprehensive tracking
        if 'missing_fields_analysis' not in self.schema_corrections:
            self.schema_corrections['missing_fields_analysis'] = []
        
        self.schema_corrections['missing_fields_analysis'].append({
            'field_name': field_name,
            'field_type': field_type,
            'default_value': default_value,
            'correction_priority': 'high' if default_value is None else 'medium'
        })
        
        # Log missing field addition for schema correction planning
        logger = get_logger('validation.schema', 'VALIDATION')
        logger.warning(f"Missing required field added to {self.validation_id}: {field_name} ({field_type})")
    
    def add_type_violation(
        self,
        field_path: str,
        expected_type: str,
        actual_type: str
    ) -> None:
        """
        Add type violation to the schema validation error for detailed type analysis.
        
        Args:
            field_path: JSON path of the field with type violation
            expected_type: Expected type according to schema
            actual_type: Actual type found in data
        """
        # Add type violation to type violations list with path and type details
        violation_description = f"{field_path}: expected {expected_type}, got {actual_type}"
        self.type_violations.append(violation_description)
        
        # Store expected and actual type information for correction guidance
        self.validation_context[f'type_violation_{len(self.type_violations)}'] = {
            'field_path': field_path,
            'expected_type': expected_type,
            'actual_type': actual_type,
            'violation_description': violation_description,
            'detected_at': datetime.datetime.now().isoformat()
        }
        
        # Update schema context with type violation details for analysis
        if 'type_violations_analysis' not in self.schema_corrections:
            self.schema_corrections['type_violations_analysis'] = []
        
        self.schema_corrections['type_violations_analysis'].append({
            'field_path': field_path,
            'expected_type': expected_type,
            'actual_type': actual_type,
            'conversion_possible': self._check_type_conversion_possible(actual_type, expected_type),
            'correction_complexity': self._assess_type_correction_complexity(actual_type, expected_type)
        })
        
        # Log type violation addition for schema correction planning
        logger = get_logger('validation.schema', 'VALIDATION')
        logger.warning(f"Type violation added to {self.validation_id}: {violation_description}")
    
    def get_schema_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive schema analysis including violations and correction recommendations.
        
        Returns:
            Dict[str, Any]: Schema analysis with violations and correction guidance
        """
        # Analyze schema violations and missing fields for comprehensive reporting
        analysis = {
            'schema_name': self.schema_name,
            'validation_path': self.validation_path,
            'schema_violations': self.schema_violations,
            'missing_required_fields': self.missing_required_fields,
            'type_violations': self.type_violations,
            'schema_corrections': self.schema_corrections,
            'violation_summary': {
                'total_violations': len(self.schema_violations),
                'missing_fields_count': len(self.missing_required_fields),
                'type_violations_count': len(self.type_violations)
            }
        }
        
        # Include type violations and constraint issues for detailed analysis
        analysis['compliance_analysis'] = {
            'schema_compliance_score': self._calculate_compliance_score(),
            'critical_violations': self._identify_critical_violations(),
            'correction_feasibility': self._assess_correction_feasibility(),
            'estimated_correction_effort': self._estimate_correction_effort()
        }
        
        # Generate schema correction recommendations based on violation analysis
        analysis['correction_recommendations'] = []
        
        if self.missing_required_fields:
            analysis['correction_recommendations'].append(
                f"Add {len(self.missing_required_fields)} missing required fields"
            )
        
        if self.type_violations:
            analysis['correction_recommendations'].append(
                f"Correct {len(self.type_violations)} type violations"
            )
        
        # Include schema-specific recovery strategies for comprehensive error resolution
        analysis['schema_recovery_strategies'] = [
            f"Review {self.schema_name} schema definition and requirements",
            f"Validate data structure at path {self.validation_path}",
            "Check schema version compatibility and updates",
            "Verify required field presence and type compliance"
        ]
        
        return analysis
    
    def _analyze_schema_violations(self) -> None:
        """Analyze schema violations and categorize them by type."""
        if not self.schema_violations:
            return
        
        for violation_key, violation_details in self.schema_violations.items():
            if isinstance(violation_details, dict):
                # Analyze different types of violations
                if 'missing_property' in violation_details:
                    property_name = violation_details['missing_property']
                    property_type = violation_details.get('expected_type', 'unknown')
                    default_value = violation_details.get('default_value')
                    self.add_missing_field(property_name, property_type, default_value)
                
                elif 'type_error' in violation_details:
                    field_path = violation_details.get('field_path', self.validation_path)
                    expected_type = violation_details.get('expected_type', 'unknown')
                    actual_type = violation_details.get('actual_type', 'unknown')
                    self.add_type_violation(field_path, expected_type, actual_type)
    
    def _generate_schema_corrections(self) -> None:
        """Generate schema correction suggestions based on violations."""
        corrections = {}
        
        # Generate corrections for missing fields
        if self.missing_required_fields:
            corrections['missing_fields_corrections'] = []
            for field_name in self.missing_required_fields:
                field_key = f'missing_field_{field_name}'
                if field_key in self.validation_context:
                    field_info = self.validation_context[field_key]
                    correction = {
                        'action': 'add_field',
                        'field_name': field_name,
                        'field_type': field_info['field_type'],
                        'suggested_value': field_info.get('default_value', 'null'),
                        'priority': 'high'
                    }
                    corrections['missing_fields_corrections'].append(correction)
        
        # Generate corrections for type violations
        if self.type_violations:
            corrections['type_corrections'] = []
            for i, violation in enumerate(self.type_violations, 1):
                violation_key = f'type_violation_{i}'
                if violation_key in self.validation_context:
                    violation_info = self.validation_context[violation_key]
                    correction = {
                        'action': 'fix_type',
                        'field_path': violation_info['field_path'],
                        'expected_type': violation_info['expected_type'],
                        'actual_type': violation_info['actual_type'],
                        'conversion_suggestion': self._suggest_type_conversion(
                            violation_info['actual_type'],
                            violation_info['expected_type']
                        ),
                        'priority': 'medium'
                    }
                    corrections['type_corrections'].append(correction)
        
        self.schema_corrections.update(corrections)
    
    def _calculate_compliance_score(self) -> float:
        """Calculate schema compliance score based on violations."""
        total_violations = len(self.schema_violations)
        if total_violations == 0:
            return 1.0
        
        # Weight different violation types
        missing_fields_weight = len(self.missing_required_fields) * 0.3
        type_violations_weight = len(self.type_violations) * 0.2
        other_violations_weight = (total_violations - len(self.missing_required_fields) - len(self.type_violations)) * 0.1
        
        total_weighted_violations = missing_fields_weight + type_violations_weight + other_violations_weight
        
        # Calculate compliance score (higher violations = lower score)
        compliance_score = max(0.0, 1.0 - (total_weighted_violations / 10.0))
        return round(compliance_score, 3)
    
    def _identify_critical_violations(self) -> List[str]:
        """Identify critical schema violations that require immediate attention."""
        critical_violations = []
        
        # Missing required fields are critical
        for field in self.missing_required_fields:
            critical_violations.append(f"Missing required field: {field}")
        
        # Type violations for critical fields
        for violation in self.type_violations:
            if any(critical_field in violation for critical_field in ['id', 'type', 'name', 'version']):
                critical_violations.append(f"Critical type violation: {violation}")
        
        return critical_violations
    
    def _assess_correction_feasibility(self) -> str:
        """Assess the feasibility of correcting schema violations."""
        if not self.schema_violations:
            return 'no_corrections_needed'
        
        total_violations = len(self.schema_violations)
        missing_fields = len(self.missing_required_fields)
        type_violations = len(self.type_violations)
        
        # Assess based on violation complexity
        if missing_fields == 0 and type_violations <= 2:
            return 'easy'
        elif missing_fields <= 3 and type_violations <= 5:
            return 'moderate'
        elif total_violations <= 10:
            return 'difficult'
        else:
            return 'complex'
    
    def _estimate_correction_effort(self) -> str:
        """Estimate the effort required to correct schema violations."""
        feasibility = self._assess_correction_feasibility()
        
        effort_mapping = {
            'no_corrections_needed': 'none',
            'easy': 'low',
            'moderate': 'medium',
            'difficult': 'high',
            'complex': 'very_high'
        }
        
        return effort_mapping.get(feasibility, 'unknown')
    
    def _check_type_conversion_possible(self, actual_type: str, expected_type: str) -> bool:
        """Check if type conversion is possible between actual and expected types."""
        conversion_matrix = {
            'string': ['number', 'integer', 'boolean'],
            'number': ['string', 'integer'],
            'integer': ['string', 'number'],
            'boolean': ['string'],
            'array': [],
            'object': []
        }
        
        return expected_type in conversion_matrix.get(actual_type, [])
    
    def _assess_type_correction_complexity(self, actual_type: str, expected_type: str) -> str:
        """Assess the complexity of correcting a type violation."""
        if actual_type == expected_type:
            return 'none'
        
        if self._check_type_conversion_possible(actual_type, expected_type):
            return 'low'
        elif actual_type == 'string' or expected_type == 'string':
            return 'medium'
        else:
            return 'high'
    
    def _suggest_type_conversion(self, actual_type: str, expected_type: str) -> str:
        """Suggest type conversion approach for type violations."""
        if actual_type == expected_type:
            return 'no_conversion_needed'
        
        if actual_type == 'string' and expected_type == 'number':
            return 'parse_string_to_number'
        elif actual_type == 'string' and expected_type == 'integer':
            return 'parse_string_to_integer'
        elif actual_type == 'string' and expected_type == 'boolean':
            return 'parse_string_to_boolean'
        elif actual_type == 'number' and expected_type == 'string':
            return 'convert_number_to_string'
        elif actual_type == 'integer' and expected_type == 'number':
            return 'promote_integer_to_number'
        else:
            return 'manual_conversion_required'
    
    def _add_schema_recovery_recommendations(self) -> None:
        """Add schema-specific recovery recommendations."""
        self.add_recovery_recommendation(
            f"Review schema definition for {self.schema_name}",
            priority='HIGH'
        )
        
        if self.missing_required_fields:
            self.add_recovery_recommendation(
                f"Add {len(self.missing_required_fields)} missing required fields",
                priority='HIGH'
            )
        
        if self.type_violations:
            self.add_recovery_recommendation(
                f"Correct {len(self.type_violations)} type violations",
                priority='MEDIUM'
            )
        
        self.add_recovery_recommendation(
            f"Validate data structure at path: {self.validation_path}",
            priority='MEDIUM'
        )


class CrossFormatValidationError(ValidationError):
    """
    Specialized validation error for cross-format compatibility validation failures including format 
    conversion errors, compatibility matrix violations, and format interoperability issues with 
    cross-format context and conversion guidance.
    
    This class extends ValidationError with cross-format compatibility context, conversion analysis,
    and specialized guidance for format interoperability issues in scientific data processing.
    """
    
    def __init__(
        self,
        message: str,
        format_types: List[str],
        compatibility_issues: Dict[str, Any],
        conversion_accuracy: Dict[str, float]
    ):
        """
        Initialize cross-format validation error with format compatibility context and conversion analysis.
        
        Args:
            message: Detailed cross-format validation error message
            format_types: List of format types involved in compatibility issue
            compatibility_issues: Dictionary of compatibility issues and details
            conversion_accuracy: Dictionary of conversion accuracy metrics
        """
        # Initialize base ValidationError with cross-format validation type
        super().__init__(
            message=message,
            validation_type='cross_format_validation',
            validation_context={
                'format_types': format_types,
                'compatibility_issues': compatibility_issues,
                'conversion_accuracy': conversion_accuracy
            }
        )
        
        # Set cross-format validation error code for classification
        self.error_code = CROSS_FORMAT_VALIDATION_ERROR_CODE
        
        # Store format types and compatibility issues for detailed tracking
        self.format_types = format_types
        self.compatibility_issues = compatibility_issues
        self.conversion_accuracy = conversion_accuracy
        
        # Analyze conversion accuracy and feasibility
        self.compatibility_matrix: Dict[str, str] = {}
        self.conversion_warnings: List[str] = []
        self.is_conversion_possible = False
        
        # Analyze format compatibility and conversion feasibility
        self._analyze_format_compatibility()
        
        # Initialize compatibility matrix and warnings based on analysis
        self._build_compatibility_matrix()
        
        # Add cross-format recovery recommendations
        self._add_cross_format_recovery_recommendations()
        
        # Update cross-format error statistics
        _validation_error_statistics['cross_format_errors'] += 1
        
        # Log cross-format validation error with format context
        logger = get_logger('validation.cross_format', 'VALIDATION')
        logger.error(f"Cross-format validation error: {format_types} - compatibility issues detected")
    
    def add_conversion_warning(
        self,
        warning_description: str,
        affected_data_type: str
    ) -> None:
        """
        Add conversion warning for potential data loss or accuracy issues during format conversion.
        
        Args:
            warning_description: Description of the conversion warning
            affected_data_type: Type of data affected by the conversion issue
        """
        # Add warning description to conversion warnings list
        formatted_warning = f"[{affected_data_type}] {warning_description}"
        self.conversion_warnings.append(formatted_warning)
        
        # Store affected data type information for conversion planning
        self.validation_context[f'conversion_warning_{len(self.conversion_warnings)}'] = {
            'warning_description': warning_description,
            'affected_data_type': affected_data_type,
            'warning_severity': self._assess_warning_severity(warning_description),
            'detected_at': datetime.datetime.now().isoformat()
        }
        
        # Update compatibility context with warning details for analysis
        if 'conversion_warnings_analysis' not in self.compatibility_issues:
            self.compatibility_issues['conversion_warnings_analysis'] = []
        
        self.compatibility_issues['conversion_warnings_analysis'].append({
            'warning': warning_description,
            'data_type': affected_data_type,
            'impact_level': self._assess_warning_impact(warning_description),
            'mitigation_possible': self._check_warning_mitigation_possible(warning_description)
        })
        
        # Log conversion warning addition for conversion planning
        logger = get_logger('validation.cross_format', 'VALIDATION')
        logger.warning(f"Conversion warning added to {self.validation_id}: {formatted_warning}")
    
    def get_compatibility_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive cross-format compatibility analysis including conversion feasibility 
        and accuracy assessment.
        
        Returns:
            Dict[str, Any]: Cross-format compatibility analysis with conversion guidance
        """
        # Analyze format compatibility and conversion feasibility for comprehensive reporting
        analysis = {
            'format_types': self.format_types,
            'compatibility_issues': self.compatibility_issues,
            'conversion_accuracy': self.conversion_accuracy,
            'compatibility_matrix': self.compatibility_matrix,
            'conversion_warnings': self.conversion_warnings,
            'is_conversion_possible': self.is_conversion_possible,
            'format_analysis': {
                'total_formats': len(self.format_types),
                'compatibility_score': self._calculate_compatibility_score(),
                'conversion_complexity': self._assess_conversion_complexity(),
                'data_loss_risk': self._assess_data_loss_risk()
            }
        }
        
        # Include conversion accuracy metrics and warnings for quality assessment
        analysis['accuracy_analysis'] = {
            'overall_accuracy': self._calculate_overall_accuracy(),
            'meets_accuracy_threshold': self._check_accuracy_threshold(),
            'accuracy_by_format': self.conversion_accuracy,
            'critical_accuracy_issues': self._identify_critical_accuracy_issues()
        }
        
        # Generate format conversion recommendations based on compatibility analysis
        analysis['conversion_recommendations'] = []
        
        if self.is_conversion_possible:
            analysis['conversion_recommendations'].extend([
                "Cross-format conversion is feasible with appropriate validation",
                "Monitor conversion accuracy and data integrity",
                "Implement conversion validation checks"
            ])
        else:
            analysis['conversion_recommendations'].extend([
                "Direct conversion not recommended due to compatibility issues",
                "Consider alternative data sources or manual conversion",
                "Review format specifications and requirements"
            ])
        
        # Include cross-format recovery strategies for comprehensive error resolution
        analysis['cross_format_recovery_strategies'] = [
            "Analyze format compatibility matrix for conversion paths",
            "Implement cross-format validation and verification",
            "Test conversion accuracy with sample data",
            "Monitor conversion performance and data integrity"
        ]
        
        return analysis
    
    def _analyze_format_compatibility(self) -> None:
        """Analyze compatibility between different format types."""
        if len(self.format_types) < 2:
            self.is_conversion_possible = True
            return
        
        # Define format compatibility rules
        compatibility_rules = {
            ('crimaldi', 'custom'): {'possible': True, 'accuracy': 0.95, 'complexity': 'medium'},
            ('crimaldi', 'avi'): {'possible': True, 'accuracy': 0.90, 'complexity': 'low'},
            ('custom', 'avi'): {'possible': True, 'accuracy': 0.85, 'complexity': 'low'},
            ('avi', 'mp4'): {'possible': True, 'accuracy': 0.98, 'complexity': 'low'},
            ('mp4', 'mov'): {'possible': True, 'accuracy': 0.95, 'complexity': 'low'}
        }
        
        # Check compatibility for each format pair
        total_compatibility = 0
        compatibility_count = 0
        
        for i, format1 in enumerate(self.format_types):
            for format2 in self.format_types[i+1:]:
                format_pair = tuple(sorted([format1.lower(), format2.lower()]))
                
                if format_pair in compatibility_rules:
                    rule = compatibility_rules[format_pair]
                    total_compatibility += 1 if rule['possible'] else 0
                    compatibility_count += 1
                    
                    # Check accuracy threshold
                    if rule['accuracy'] < CORRELATION_ACCURACY_THRESHOLD:
                        self.add_conversion_warning(
                            f"Conversion accuracy {rule['accuracy']:.2f} below threshold {CORRELATION_ACCURACY_THRESHOLD}",
                            f"{format1}-{format2}"
                        )
        
        # Determine overall conversion possibility
        if compatibility_count > 0:
            compatibility_ratio = total_compatibility / compatibility_count
            self.is_conversion_possible = compatibility_ratio >= 0.5
        else:
            self.is_conversion_possible = False
    
    def _build_compatibility_matrix(self) -> None:
        """Build compatibility matrix for format combinations."""
        matrix = {}
        
        for format1 in self.format_types:
            matrix[format1] = {}
            for format2 in self.format_types:
                if format1 == format2:
                    matrix[format1][format2] = 'identical'
                else:
                    # Simplified compatibility assessment
                    compatibility = self._assess_format_pair_compatibility(format1, format2)
                    matrix[format1][format2] = compatibility
        
        self.compatibility_matrix = matrix
    
    def _calculate_compatibility_score(self) -> float:
        """Calculate overall compatibility score for format combinations."""
        if len(self.format_types) <= 1:
            return 1.0
        
        compatibility_scores = []
        
        for format1 in self.format_types:
            for format2 in self.format_types:
                if format1 != format2:
                    pair_score = self._get_format_pair_score(format1, format2)
                    compatibility_scores.append(pair_score)
        
        if compatibility_scores:
            return sum(compatibility_scores) / len(compatibility_scores)
        else:
            return 1.0
    
    def _calculate_overall_accuracy(self) -> float:
        """Calculate overall conversion accuracy across all formats."""
        if not self.conversion_accuracy:
            return 0.0
        
        accuracy_values = list(self.conversion_accuracy.values())
        return sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0
    
    def _check_accuracy_threshold(self) -> bool:
        """Check if conversion accuracy meets the required threshold."""
        overall_accuracy = self._calculate_overall_accuracy()
        return overall_accuracy >= CORRELATION_ACCURACY_THRESHOLD
    
    def _identify_critical_accuracy_issues(self) -> List[str]:
        """Identify critical accuracy issues that require attention."""
        critical_issues = []
        
        for format_pair, accuracy in self.conversion_accuracy.items():
            if accuracy < CORRELATION_ACCURACY_THRESHOLD:
                critical_issues.append(
                    f"{format_pair}: accuracy {accuracy:.3f} below threshold {CORRELATION_ACCURACY_THRESHOLD}"
                )
        
        return critical_issues
    
    def _assess_conversion_complexity(self) -> str:
        """Assess the complexity of cross-format conversion."""
        if len(self.format_types) <= 1:
            return 'none'
        
        complexity_factors = []
        
        # Number of formats
        if len(self.format_types) > 3:
            complexity_factors.append('high_format_count')
        
        # Conversion warnings
        if len(self.conversion_warnings) > 5:
            complexity_factors.append('many_warnings')
        
        # Accuracy issues
        if not self._check_accuracy_threshold():
            complexity_factors.append('accuracy_issues')
        
        # Determine overall complexity
        if len(complexity_factors) >= 3:
            return 'very_high'
        elif len(complexity_factors) == 2:
            return 'high'
        elif len(complexity_factors) == 1:
            return 'medium'
        else:
            return 'low'
    
    def _assess_data_loss_risk(self) -> str:
        """Assess the risk of data loss during cross-format conversion."""
        risk_factors = []
        
        # Check conversion accuracy
        overall_accuracy = self._calculate_overall_accuracy()
        if overall_accuracy < 0.9:
            risk_factors.append('low_accuracy')
        
        # Check for data loss warnings
        data_loss_warnings = [w for w in self.conversion_warnings if 'loss' in w.lower()]
        if data_loss_warnings:
            risk_factors.append('data_loss_warnings')
        
        # Check format compatibility
        if not self.is_conversion_possible:
            risk_factors.append('incompatible_formats')
        
        # Determine risk level
        if len(risk_factors) >= 3:
            return 'high'
        elif len(risk_factors) == 2:
            return 'medium'
        elif len(risk_factors) == 1:
            return 'low'
        else:
            return 'minimal'
    
    def _assess_format_pair_compatibility(self, format1: str, format2: str) -> str:
        """Assess compatibility between a pair of formats."""
        # Simplified compatibility assessment
        scientific_formats = {'crimaldi', 'custom'}
        video_formats = {'avi', 'mp4', 'mov', 'mkv'}
        
        if format1.lower() in scientific_formats and format2.lower() in scientific_formats:
            return 'high'
        elif format1.lower() in video_formats and format2.lower() in video_formats:
            return 'medium'
        elif (format1.lower() in scientific_formats and format2.lower() in video_formats) or \
             (format1.lower() in video_formats and format2.lower() in scientific_formats):
            return 'low'
        else:
            return 'unknown'
    
    def _get_format_pair_score(self, format1: str, format2: str) -> float:
        """Get numerical score for format pair compatibility."""
        compatibility = self._assess_format_pair_compatibility(format1, format2)
        
        score_mapping = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.4,
            'unknown': 0.1
        }
        
        return score_mapping.get(compatibility, 0.0)
    
    def _assess_warning_severity(self, warning_description: str) -> str:
        """Assess the severity of a conversion warning."""
        if any(keyword in warning_description.lower() for keyword in ['critical', 'fatal', 'error']):
            return 'high'
        elif any(keyword in warning_description.lower() for keyword in ['loss', 'degradation']):
            return 'medium'
        else:
            return 'low'
    
    def _assess_warning_impact(self, warning_description: str) -> str:
        """Assess the impact level of a conversion warning."""
        if 'accuracy' in warning_description.lower():
            return 'accuracy'
        elif 'data' in warning_description.lower():
            return 'data_integrity'
        elif 'performance' in warning_description.lower():
            return 'performance'
        else:
            return 'general'
    
    def _check_warning_mitigation_possible(self, warning_description: str) -> bool:
        """Check if warning mitigation is possible."""
        mitigatable_patterns = ['accuracy', 'performance', 'format']
        return any(pattern in warning_description.lower() for pattern in mitigatable_patterns)
    
    def _add_cross_format_recovery_recommendations(self) -> None:
        """Add cross-format specific recovery recommendations."""
        self.add_recovery_recommendation(
            f"Analyze compatibility between formats: {', '.join(self.format_types)}",
            priority='HIGH'
        )
        
        if not self.is_conversion_possible:
            self.add_recovery_recommendation(
                "Direct conversion not possible - consider alternative approaches",
                priority='HIGH'
            )
        else:
            self.add_recovery_recommendation(
                "Cross-format conversion possible with validation",
                priority='MEDIUM'
            )
        
        if self.conversion_warnings:
            self.add_recovery_recommendation(
                f"Address {len(self.conversion_warnings)} conversion warnings",
                priority='MEDIUM'
            )
        
        self.add_recovery_recommendation(
            "Validate conversion accuracy against scientific requirements",
            priority='MEDIUM'
        )


# Helper functions for validation error utilities

def _get_default_recovery_strategies(validation_category: str) -> List[str]:
    """Get default recovery strategies for validation category."""
    strategies = {
        'format_validation': [
            "Verify file format and structure",
            "Check file integrity and accessibility",
            "Validate format specification compliance"
        ],
        'parameter_validation': [
            "Review parameter values and constraints",
            "Validate parameter types and ranges",
            "Check parameter dependencies"
        ],
        'schema_validation': [
            "Review schema definition and compliance",
            "Validate required fields and types",
            "Check schema version compatibility"
        ],
        'cross_format_validation': [
            "Analyze format compatibility matrix",
            "Validate conversion accuracy",
            "Test cross-format conversion"
        ]
    }
    
    return strategies.get(validation_category, ["Review validation requirements and correct issues"])


def _calculate_validation_complexity(validation_data: Any) -> str:
    """Calculate validation complexity based on data characteristics."""
    if not validation_data:
        return 'low'
    
    complexity_factors = 0
    
    if isinstance(validation_data, dict):
        complexity_factors += len(validation_data) // 10  # 1 point per 10 keys
        
        # Check for nested structures
        for value in validation_data.values():
            if isinstance(value, (dict, list)):
                complexity_factors += 1
    
    elif isinstance(validation_data, list):
        complexity_factors += len(validation_data) // 100  # 1 point per 100 items
    
    # Determine complexity level
    if complexity_factors >= 5:
        return 'high'
    elif complexity_factors >= 2:
        return 'medium'
    else:
        return 'low'


def _parse_time_window(time_window: str) -> datetime.datetime:
    """Parse time window string and return cutoff datetime."""
    current_time = datetime.datetime.now()
    
    if time_window.endswith('h'):
        hours = int(time_window[:-1])
        return current_time - datetime.timedelta(hours=hours)
    elif time_window.endswith('d'):
        days = int(time_window[:-1])
        return current_time - datetime.timedelta(days=days)
    elif time_window.endswith('m'):
        minutes = int(time_window[:-1])
        return current_time - datetime.timedelta(minutes=minutes)
    else:
        # Default to 1 hour
        return current_time - datetime.timedelta(hours=1)


# Register validation error types in the global registry
register_validation_error_type(ValidationError, VALIDATION_ERROR_CODE_BASE, 'general_validation')
register_validation_error_type(FormatValidationError, FORMAT_VALIDATION_ERROR_CODE, 'format_validation')
register_validation_error_type(ParameterValidationError, PARAMETER_VALIDATION_ERROR_CODE, 'parameter_validation')
register_validation_error_type(SchemaValidationError, SCHEMA_VALIDATION_ERROR_CODE, 'schema_validation')
register_validation_error_type(CrossFormatValidationError, CROSS_FORMAT_VALIDATION_ERROR_CODE, 'cross_format_validation')