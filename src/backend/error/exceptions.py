"""
Core exception classes module providing the foundational exception hierarchy for the plume simulation system 
including base exception classes, error categorization, context management, recovery recommendation framework, 
and audit trail integration.

This module implements comprehensive error classification with severity levels, scientific computing context 
enhancement, performance impact tracking, and structured error reporting to support reproducible research 
outcomes and reliable scientific computing operations with <1% error rate target.

Key Features:
- Comprehensive exception hierarchy with scientific computing context
- Error severity and category classification for prioritized handling
- Recovery recommendation framework with actionable guidance
- Audit trail integration for scientific computing traceability
- Performance impact tracking and threshold monitoring
- Structured error reporting for debugging and analysis
- Thread-safe exception handling and context management
- Fail-fast validation strategy with early error detection
- Graceful degradation support for batch processing operations
- Error recovery mechanisms with automatic retry logic
"""

# Standard library imports with version specifications
import datetime  # Python 3.9+ - Timestamp generation for exception tracking and audit trails
import traceback  # Python 3.9+ - Stack trace extraction for detailed exception reporting and debugging
import threading  # Python 3.9+ - Thread-safe exception handling and context management
import uuid  # Python 3.9+ - Unique identifier generation for exception tracking and correlation
import json  # Python 3.9+ - JSON serialization for structured exception reporting and logging
import sys  # Python 3.9+ - System-specific exception information and error handling
from typing import Dict, Any, List, Optional, Union, Type  # Python 3.9+ - Type hints for exception signatures

# Internal imports from utility modules
from ..utils.error_handling import (
    ErrorSeverity, ErrorCategory, 
    get_recovery_strategy, is_retryable,
    get_priority, requires_immediate_action
)
from ..utils.logging_utils import (
    get_logger, log_validation_error, create_audit_trail, get_scientific_context
)
from ..utils.validation_utils import ValidationResult
from ..utils.scientific_constants import (
    NUMERICAL_PRECISION_THRESHOLD, DEFAULT_CORRELATION_THRESHOLD,
    ERROR_RATE_THRESHOLD, PROCESSING_TIME_TARGET_SECONDS
)

# Global exception registry and recovery strategies for dynamic exception management
_exception_registry: Dict[str, Type['PlumeSimulationException']] = {}
_recovery_recommendations: Dict[str, List[str]] = {}
_error_context_defaults: Dict[str, Any] = {
    'simulation_id': 'unknown',
    'algorithm_name': 'unknown', 
    'processing_stage': 'unknown',
    'batch_id': None
}

# Global error codes for exception classification and tracking
BASE_ERROR_CODE = 1000
VALIDATION_ERROR_CODE = 1100
PROCESSING_ERROR_CODE = 2000
SIMULATION_ERROR_CODE = 3000
ANALYSIS_ERROR_CODE = 4000
CONFIGURATION_ERROR_CODE = 5000
RESOURCE_ERROR_CODE = 6000
SYSTEM_ERROR_CODE = 7000

# Thread-local storage for scientific context and error tracking
_context_storage = threading.local()


def register_exception_type(
    exception_class: Type['PlumeSimulationException'],
    error_code: int,
    category: str
) -> bool:
    """
    Register custom exception type in the global exception registry for dynamic exception 
    handling and specialized recovery strategy mapping.
    
    Args:
        exception_class: Exception class to register
        error_code: Unique error code for the exception type
        category: Error category for classification
        
    Returns:
        bool: Success status of exception type registration
    """
    logger = get_logger('exception_registry', 'ERROR_HANDLING')
    
    try:
        # Validate exception class inheritance from PlumeSimulationException
        if not issubclass(exception_class, PlumeSimulationException):
            raise TypeError("Exception class must inherit from PlumeSimulationException")
        
        # Check for error code conflicts in registry
        for registered_name, registered_class in _exception_registry.items():
            if hasattr(registered_class, '_error_code') and registered_class._error_code == error_code:
                logger.warning(f"Error code conflict: {error_code} already used by {registered_name}")
        
        # Register exception type with error code mapping
        exception_name = exception_class.__name__
        _exception_registry[exception_name] = exception_class
        exception_class._error_code = error_code
        exception_class._category = category
        
        # Configure default recovery strategies for exception type
        if category in ['VALIDATION', 'CONFIGURATION']:
            _recovery_recommendations[exception_name] = [
                "Review and correct input parameters",
                "Validate configuration settings", 
                "Check data format compatibility"
            ]
        elif category in ['PROCESSING', 'SIMULATION']:
            _recovery_recommendations[exception_name] = [
                "Retry operation with different parameters",
                "Check system resources and availability",
                "Verify algorithm configuration"
            ]
        else:
            _recovery_recommendations[exception_name] = [
                "Review error details and context",
                "Contact system administrator if needed"
            ]
        
        # Log exception type registration for audit trail
        create_audit_trail(
            action='EXCEPTION_TYPE_REGISTERED',
            component='EXCEPTION_REGISTRY',
            action_details={
                'exception_class': exception_name,
                'error_code': error_code,
                'category': category
            },
            user_context='SYSTEM'
        )
        
        logger.info(f"Exception type registered: {exception_name} (code: {error_code}, category: {category})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register exception type {exception_class.__name__}: {e}")
        return False


def create_exception_context(
    component: str,
    operation: str,
    additional_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create comprehensive exception context dictionary with scientific computing context, 
    system state, and debugging information for reproducible research outcomes.
    
    Args:
        component: Component name where exception occurred
        operation: Operation being performed when exception occurred
        additional_context: Additional context information
        
    Returns:
        Dict[str, Any]: Comprehensive exception context with scientific and system information
    """
    # Capture current scientific computing context from thread storage
    scientific_context = get_scientific_context(include_defaults=True)
    
    # Extract system state and resource usage information
    context = {
        'component': component,
        'operation': operation,
        'timestamp': datetime.datetime.now().isoformat(),
        'thread_id': threading.current_thread().ident,
        'thread_name': threading.current_thread().name,
        'scientific_context': scientific_context
    }
    
    # Include component and operation identification
    context['error_location'] = f"{component}.{operation}"
    
    # Add timestamp and unique exception identifier
    context['exception_id'] = str(uuid.uuid4())
    context['correlation_id'] = str(uuid.uuid4())
    
    # Merge additional context if provided
    if additional_context:
        context['additional_context'] = additional_context
    
    # Include stack trace and call hierarchy for debugging
    try:
        stack_frames = []
        frame = sys._getframe(1)  # Skip this function
        frame_count = 0
        
        while frame and frame_count < 5:  # Limit stack depth for performance
            stack_frames.append({
                'filename': frame.f_code.co_filename,
                'function': frame.f_code.co_name,
                'line_number': frame.f_lineno
            })
            frame = frame.f_back
            frame_count += 1
        
        context['call_stack'] = stack_frames
        
    except Exception:
        # Ignore errors in stack trace extraction
        context['call_stack'] = []
    
    # Add performance metrics if available
    context['performance_context'] = {
        'memory_usage_estimate': 'unknown',
        'processing_time_estimate': 'unknown'
    }
    
    return context


def format_exception_message(
    base_message: str,
    context: Dict[str, Any],
    include_recommendations: bool = True,
    include_scientific_context: bool = True
) -> str:
    """
    Format exception message with scientific context, error details, and recovery 
    recommendations for clear error communication and debugging support.
    
    Args:
        base_message: Base exception message
        context: Exception context dictionary
        include_recommendations: Whether to include recovery recommendations
        include_scientific_context: Whether to include scientific context
        
    Returns:
        str: Formatted exception message with context and recommendations
    """
    # Format base message with context variables
    formatted_message = base_message
    
    # Add scientific computing specific details if enabled
    if include_scientific_context and 'scientific_context' in context:
        sci_context = context['scientific_context']
        formatted_message += f" [Simulation: {sci_context.get('simulation_id', 'unknown')}]"
        formatted_message += f" [Algorithm: {sci_context.get('algorithm_name', 'unknown')}]"
        formatted_message += f" [Stage: {sci_context.get('processing_stage', 'unknown')}]"
    
    # Include component and operation information
    if 'component' in context and 'operation' in context:
        formatted_message += f" [Location: {context['component']}.{context['operation']}]"
    
    # Add error code and severity information
    if 'error_code' in context:
        formatted_message += f" [Code: {context['error_code']}]"
    
    if 'severity' in context:
        formatted_message += f" [Severity: {context['severity']}]"
    
    # Include recovery recommendations if enabled
    if include_recommendations and 'recovery_recommendations' in context:
        recommendations = context['recovery_recommendations']
        if recommendations:
            formatted_message += "\n\nRecovery Recommendations:"
            for i, rec in enumerate(recommendations[:3], 1):  # Limit to top 3
                formatted_message += f"\n  {i}. {rec}"
    
    # Format message for readability and user guidance
    if 'exception_id' in context:
        formatted_message += f"\n\nException ID: {context['exception_id']}"
    
    return formatted_message


def get_default_recovery_recommendations(
    error_category: str,
    severity_level: str,
    error_context: Dict[str, Any]
) -> List[str]:
    """
    Get default recovery recommendations for exception category and severity level 
    to provide actionable guidance for error resolution.
    
    Args:
        error_category: Error category string
        severity_level: Error severity level
        error_context: Error context for specific recommendations
        
    Returns:
        List[str]: List of recovery recommendations specific to error category and severity
    """
    # Map error category to default recovery strategies
    category_recommendations = {
        'VALIDATION': [
            "Verify input data format and structure",
            "Check parameter values against valid ranges",
            "Validate configuration file syntax and completeness"
        ],
        'PROCESSING': [
            "Retry operation with modified parameters",
            "Check available system resources",
            "Verify input file integrity and accessibility"
        ],
        'SIMULATION': [
            "Review simulation algorithm configuration",
            "Check convergence criteria and parameters",
            "Verify input data quality and completeness"
        ],
        'ANALYSIS': [
            "Review analysis configuration settings",
            "Check data availability and format",
            "Verify statistical analysis parameters"
        ],
        'CONFIGURATION': [
            "Review configuration file structure",
            "Check for missing required parameters",
            "Validate configuration against schema"
        ],
        'RESOURCE': [
            "Check available memory and disk space",
            "Monitor CPU usage and system load",
            "Consider reducing batch size or complexity"
        ],
        'SYSTEM': [
            "Check system logs for additional information",
            "Verify system dependencies and services",
            "Contact system administrator if persistent"
        ]
    }
    
    recommendations = category_recommendations.get(error_category, [
        "Review error details and context",
        "Check system logs for additional information",
        "Consider alternative approaches"
    ])
    
    # Filter recommendations by severity level
    if severity_level in ['CRITICAL', 'HIGH']:
        recommendations.append("Immediate intervention required - stop current operation")
    
    # Include context-specific recommendations
    if 'batch_id' in error_context:
        recommendations.append("Consider reducing batch size or processing subset")
    
    if 'algorithm_name' in error_context:
        recommendations.append(f"Review {error_context['algorithm_name']} algorithm parameters")
    
    # Add general troubleshooting guidance
    recommendations.extend([
        "Verify system requirements and dependencies",
        "Check for recent system or configuration changes"
    ])
    
    return recommendations[:5]  # Return top 5 recommendations


class PlumeSimulationException(Exception):
    """
    Base exception class for all plume simulation system errors providing comprehensive error 
    context management, recovery recommendation framework, audit trail integration, and 
    scientific computing context enhancement for reproducible research outcomes and reliable 
    error handling.
    
    This class serves as the foundation for all exceptions in the plume simulation system with 
    comprehensive context management, audit trail support, and recovery recommendation framework.
    """
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: int = BASE_ERROR_CODE,
        context: Dict[str, Any] = None
    ):
        """
        Initialize base plume simulation exception with comprehensive context management, 
        audit trail creation, and recovery recommendation framework.
        
        Args:
            message: Exception message describing the error
            category: Error category for classification
            severity: Error severity level
            error_code: Unique error code for tracking
            context: Additional context information
        """
        # Initialize base Exception with formatted message
        super().__init__(message)
        
        # Set error category, severity, and error code
        self.category = category
        self.severity = severity
        self.error_code = error_code
        self.message = message
        
        # Generate unique exception identifier
        self.exception_id = str(uuid.uuid4())
        
        # Set timestamp for exception occurrence
        self.timestamp = datetime.datetime.now()
        
        # Capture scientific computing context from thread storage
        self.scientific_context = get_scientific_context(include_defaults=True)
        
        # Create comprehensive error context dictionary
        self.context = create_exception_context(
            component=context.get('component', 'unknown') if context else 'unknown',
            operation=context.get('operation', 'unknown') if context else 'unknown',
            additional_context=context
        )
        
        # Initialize recovery recommendations list
        self.recovery_recommendations: List[str] = []
        
        # Create audit trail entry for exception
        self.audit_trail_id = create_audit_trail(
            action='EXCEPTION_CREATED',
            component='EXCEPTION_HANDLING',
            action_details={
                'exception_type': self.__class__.__name__,
                'exception_id': self.exception_id,
                'message': message,
                'category': category.value,
                'severity': severity.value,
                'error_code': error_code,
                'scientific_context': self.scientific_context
            },
            user_context='SYSTEM'
        )
        
        # Determine if immediate action is required
        self.requires_immediate_action = requires_immediate_action(severity.value)
        
        # Initialize performance impact tracking
        self.performance_impact = {
            'processing_time_impact': 'unknown',
            'memory_impact': 'unknown',
            'throughput_impact': 'unknown'
        }
        
        # Log exception creation with structured format
        logger = get_logger('exception_base', 'ERROR_HANDLING')
        logger.log(
            getattr(logger, severity.value.lower(), logger.error),
            f"Exception created: {self.__class__.__name__} - {message} [ID: {self.exception_id}]"
        )
    
    def add_context(
        self,
        key: str,
        value: Any,
        context_category: str = 'additional'
    ) -> None:
        """
        Add additional context information to the exception for enhanced debugging 
        and error analysis.
        
        Args:
            key: Context key
            value: Context value
            context_category: Category for organizing context information
        """
        # Add key-value pair to exception context dictionary
        if context_category not in self.context:
            self.context[context_category] = {}
        
        self.context[context_category][key] = value
        
        # Update audit trail with context addition
        create_audit_trail(
            action='EXCEPTION_CONTEXT_ADDED',
            component='EXCEPTION_HANDLING',
            action_details={
                'exception_id': self.exception_id,
                'context_key': key,
                'context_category': context_category
            },
            user_context='SYSTEM'
        )
        
        # Log context addition for debugging and traceability
        logger = get_logger('exception_base', 'ERROR_HANDLING')
        logger.debug(f"Context added to exception {self.exception_id}: {context_category}.{key}")
    
    def add_recovery_recommendation(
        self,
        recommendation: str,
        priority: str = 'MEDIUM',
        recommendation_context: Dict[str, Any] = None
    ) -> None:
        """
        Add specific recovery recommendation to the exception for actionable error 
        resolution guidance.
        
        Args:
            recommendation: Recommendation text
            priority: Priority level for the recommendation
            recommendation_context: Additional context for the recommendation
        """
        # Add recommendation to recovery recommendations list
        formatted_recommendation = f"[{priority}] {recommendation}"
        self.recovery_recommendations.append(formatted_recommendation)
        
        # Store recommendation context for implementation guidance
        if recommendation_context:
            self.context[f'recommendation_{len(self.recovery_recommendations)}'] = {
                'text': recommendation,
                'priority': priority,
                'context': recommendation_context,
                'added_at': datetime.datetime.now().isoformat()
            }
        
        # Update audit trail with recommendation addition
        create_audit_trail(
            action='RECOVERY_RECOMMENDATION_ADDED',
            component='EXCEPTION_HANDLING',
            action_details={
                'exception_id': self.exception_id,
                'recommendation': recommendation,
                'priority': priority
            },
            user_context='SYSTEM'
        )
        
        # Log recommendation addition for tracking
        logger = get_logger('exception_base', 'ERROR_HANDLING')
        logger.debug(f"Recovery recommendation added to exception {self.exception_id}: {recommendation}")
    
    def get_recovery_recommendations(self) -> List[str]:
        """
        Get comprehensive list of recovery recommendations prioritized by effectiveness 
        and feasibility.
        
        Returns:
            List[str]: Prioritized list of recovery recommendations
        """
        # Combine custom and default recovery recommendations
        all_recommendations = self.recovery_recommendations.copy()
        
        # Get default recommendations for category and severity
        default_recommendations = get_default_recovery_recommendations(
            error_category=self.category.value,
            severity_level=self.severity.value,
            error_context=self.context
        )
        
        # Add default recommendations that aren't already present
        for rec in default_recommendations:
            if not any(rec.lower() in existing.lower() for existing in all_recommendations):
                all_recommendations.append(f"[DEFAULT] {rec}")
        
        # Sort recommendations by priority and effectiveness
        priority_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'DEFAULT': 1}
        
        def get_priority(rec: str) -> int:
            for priority, value in priority_order.items():
                if f'[{priority}]' in rec:
                    return value
            return 1
        
        sorted_recommendations = sorted(all_recommendations, key=get_priority, reverse=True)
        
        # Return prioritized recovery recommendations list
        return sorted_recommendations
    
    def to_dict(
        self,
        include_stack_trace: bool = False,
        include_scientific_context: bool = True
    ) -> Dict[str, Any]:
        """
        Convert exception to dictionary format for serialization, logging, and integration 
        with monitoring systems.
        
        Args:
            include_stack_trace: Whether to include stack trace
            include_scientific_context: Whether to include scientific context
            
        Returns:
            Dict[str, Any]: Complete exception information as dictionary with all properties and context
        """
        # Convert all exception properties to dictionary format
        exception_dict = {
            'exception_type': self.__class__.__name__,
            'exception_id': self.exception_id,
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'error_code': self.error_code,
            'timestamp': self.timestamp.isoformat(),
            'requires_immediate_action': self.requires_immediate_action,
            'audit_trail_id': self.audit_trail_id,
            'recovery_recommendations': self.get_recovery_recommendations(),
            'context': self.context,
            'performance_impact': self.performance_impact
        }
        
        # Include scientific context if requested
        if include_scientific_context:
            exception_dict['scientific_context'] = self.scientific_context
        
        # Add stack trace if requested
        if include_stack_trace:
            exception_dict['stack_trace'] = traceback.format_exception(
                type(self), self, self.__traceback__
            )
        
        return exception_dict
    
    def get_formatted_message(
        self,
        include_context: bool = True,
        include_recommendations: bool = True
    ) -> str:
        """
        Get formatted exception message with context, severity, and recovery guidance 
        for user-friendly error reporting.
        
        Args:
            include_context: Whether to include context information
            include_recommendations: Whether to include recovery recommendations
            
        Returns:
            str: Formatted exception message with enhanced information
        """
        # Format base exception message with severity and category
        message_context = self.context.copy()
        message_context['severity'] = self.severity.value
        message_context['error_code'] = self.error_code
        message_context['recovery_recommendations'] = self.get_recovery_recommendations()
        
        # Format message for readability and user guidance
        formatted_message = format_exception_message(
            base_message=self.message,
            context=message_context,
            include_recommendations=include_recommendations,
            include_scientific_context=include_context
        )
        
        return formatted_message


class ValidationError(PlumeSimulationException):
    """
    Specialized exception class for data validation errors including format validation, 
    parameter validation, schema validation, and cross-format compatibility issues with 
    fail-fast validation support and detailed validation context for scientific computing reliability.
    """
    
    def __init__(
        self,
        message: str,
        validation_type: str,
        validation_context: Dict[str, Any] = None,
        failed_parameters: List[str] = None
    ):
        """
        Initialize validation error with comprehensive validation context and fail-fast 
        validation support for early error detection.
        
        Args:
            message: Validation error message
            validation_type: Type of validation that failed
            validation_context: Context information for validation
            failed_parameters: List of parameters that failed validation
        """
        # Initialize base PlumeSimulationException with VALIDATION category
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            error_code=VALIDATION_ERROR_CODE,
            context=validation_context
        )
        
        # Store validation type and context information
        self.validation_type = validation_type
        self.validation_context = validation_context or {}
        
        # Initialize failed parameters and validation errors lists
        self.failed_parameters = failed_parameters or []
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
        # Determine if validation failure is critical
        self.is_critical_validation_failure = len(self.failed_parameters) > 5
        
        # Set validation stage from context
        self.validation_stage = self.validation_context.get('stage', 'unknown')
        
        # Initialize validation metrics
        self.validation_metrics = {
            'failed_parameter_count': len(self.failed_parameters),
            'validation_type': validation_type,
            'critical_failure': self.is_critical_validation_failure
        }
        
        # Add validation-specific recovery recommendations
        self._add_validation_recommendations()
        
        # Create audit trail entry for validation failure
        create_audit_trail(
            action='VALIDATION_ERROR_CREATED',
            component='VALIDATION',
            action_details={
                'exception_id': self.exception_id,
                'validation_type': validation_type,
                'failed_parameters': self.failed_parameters,
                'critical_failure': self.is_critical_validation_failure
            },
            user_context='SYSTEM'
        )
        
        # Log validation error with detailed context
        logger = get_logger('validation_error', 'VALIDATION')
        logger.error(
            f"Validation error: {validation_type} - {message} "
            f"[Failed parameters: {len(self.failed_parameters)}]"
        )
    
    def add_validation_error(
        self,
        error_description: str,
        parameter_name: str,
        expected_value: Any = None,
        actual_value: Any = None
    ) -> None:
        """
        Add specific validation error to the validation error list for comprehensive 
        validation failure tracking.
        
        Args:
            error_description: Description of the validation error
            parameter_name: Name of the parameter that failed
            expected_value: Expected value for the parameter
            actual_value: Actual value that caused the failure
        """
        # Add error description to validation errors list
        formatted_error = f"{parameter_name}: {error_description}"
        if expected_value is not None and actual_value is not None:
            formatted_error += f" (expected: {expected_value}, actual: {actual_value})"
        
        self.validation_errors.append(formatted_error)
        
        # Store parameter name and value information
        if parameter_name not in self.failed_parameters:
            self.failed_parameters.append(parameter_name)
        
        # Update validation context with error details
        self.validation_context[f'{parameter_name}_error'] = {
            'description': error_description,
            'expected_value': expected_value,
            'actual_value': actual_value,
            'error_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Log validation error addition for debugging
        logger = get_logger('validation_error', 'VALIDATION')
        logger.debug(f"Validation error added: {formatted_error}")
    
    def add_validation_warning(
        self,
        warning_description: str,
        parameter_name: str
    ) -> None:
        """
        Add validation warning for non-critical issues that may affect processing quality.
        
        Args:
            warning_description: Description of the validation warning
            parameter_name: Name of the parameter with warning
        """
        # Add warning description to validation warnings list
        formatted_warning = f"{parameter_name}: {warning_description}"
        self.validation_warnings.append(formatted_warning)
        
        # Store parameter name for warning context
        self.validation_context[f'{parameter_name}_warning'] = {
            'description': warning_description,
            'warning_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Log validation warning for review
        logger = get_logger('validation_error', 'VALIDATION')
        logger.warning(f"Validation warning: {formatted_warning}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive validation failure summary with errors, warnings, and recovery guidance.
        
        Returns:
            Dict[str, Any]: Validation failure summary with comprehensive analysis
        """
        # Compile validation errors and warnings
        summary = {
            'validation_type': self.validation_type,
            'validation_stage': self.validation_stage,
            'failed_parameters': self.failed_parameters,
            'validation_errors': self.validation_errors,
            'validation_warnings': self.validation_warnings,
            'total_errors': len(self.validation_errors),
            'total_warnings': len(self.validation_warnings),
            'critical_failure': self.is_critical_validation_failure,
            'validation_metrics': self.validation_metrics,
            'recovery_recommendations': self.get_recovery_recommendations()
        }
        
        return summary
    
    def _add_validation_recommendations(self) -> None:
        """Add validation-specific recovery recommendations."""
        if self.validation_type == 'format_validation':
            self.add_recovery_recommendation(
                "Verify data format matches expected schema",
                priority='HIGH'
            )
        elif self.validation_type == 'parameter_validation':
            self.add_recovery_recommendation(
                "Check parameter values against valid ranges",
                priority='HIGH'
            )
        elif self.validation_type == 'schema_validation':
            self.add_recovery_recommendation(
                "Review configuration file structure and required fields",
                priority='HIGH'
            )
        
        # Add general validation recommendations
        self.add_recovery_recommendation(
            "Review validation error details and correct input data",
            priority='MEDIUM'
        )


class ProcessingError(PlumeSimulationException):
    """
    Specialized exception class for data processing errors including video processing failures, 
    normalization errors, format conversion issues, and data transformation problems with 
    graceful degradation support and intermediate result preservation for batch processing reliability.
    """
    
    def __init__(
        self,
        message: str,
        processing_stage: str,
        input_file: str,
        processing_context: Dict[str, Any] = None
    ):
        """
        Initialize processing error with processing stage context and intermediate result 
        preservation for graceful degradation support.
        
        Args:
            message: Processing error message
            processing_stage: Stage of processing where error occurred
            input_file: Input file being processed
            processing_context: Context information for processing
        """
        # Initialize base PlumeSimulationException with PROCESSING category
        super().__init__(
            message=message,
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            error_code=PROCESSING_ERROR_CODE,
            context=processing_context
        )
        
        # Store processing stage and input file information
        self.processing_stage = processing_stage
        self.input_file = input_file
        self.processing_context = processing_context or {}
        
        # Initialize intermediate results and progress tracking
        self.intermediate_results: Dict[str, Any] = {}
        self.processing_progress = 0.0
        self.partial_success = False
        
        # Initialize completed and failed steps lists
        self.completed_steps: List[str] = []
        self.failed_steps: List[str] = []
        
        # Initialize performance metrics
        self.performance_metrics: Dict[str, float] = {}
        
        # Add processing-specific recovery recommendations
        self._add_processing_recommendations()
        
        # Create audit trail entry for processing failure
        create_audit_trail(
            action='PROCESSING_ERROR_CREATED',
            component='PROCESSING',
            action_details={
                'exception_id': self.exception_id,
                'processing_stage': processing_stage,
                'input_file': input_file,
                'partial_success': self.partial_success
            },
            user_context='SYSTEM'
        )
        
        # Log processing error with stage and file context
        logger = get_logger('processing_error', 'PROCESSING')
        logger.error(f"Processing error in {processing_stage}: {message} [File: {input_file}]")
    
    def preserve_intermediate_results(
        self,
        results: Dict[str, Any],
        completion_percentage: float
    ) -> None:
        """
        Preserve intermediate processing results for graceful degradation and potential recovery.
        
        Args:
            results: Intermediate results to preserve
            completion_percentage: Processing completion percentage
        """
        # Store intermediate results in processing context
        self.intermediate_results.update(results)
        self.processing_progress = completion_percentage
        
        # Set partial success flag if progress > 0
        if completion_percentage > 0:
            self.partial_success = True
        
        # Log intermediate results preservation
        logger = get_logger('processing_error', 'PROCESSING')
        logger.info(
            f"Intermediate results preserved: {completion_percentage:.1f}% complete "
            f"[Exception: {self.exception_id}]"
        )
        
        # Update recovery recommendations with partial results
        if self.partial_success:
            self.add_recovery_recommendation(
                f"Partial results available ({completion_percentage:.1f}% complete) - consider resuming from checkpoint",
                priority='MEDIUM'
            )
    
    def add_completed_step(
        self,
        step_name: str,
        step_results: Dict[str, Any] = None
    ) -> None:
        """
        Add successfully completed processing step for progress tracking and recovery planning.
        
        Args:
            step_name: Name of the completed step
            step_results: Results from the completed step
        """
        # Add step name to completed steps list
        self.completed_steps.append(step_name)
        
        # Store step results in intermediate results
        if step_results:
            self.intermediate_results[f'{step_name}_results'] = step_results
        
        # Update processing progress calculation
        total_steps = len(self.completed_steps) + len(self.failed_steps)
        if total_steps > 0:
            self.processing_progress = len(self.completed_steps) / total_steps * 100
        
        # Log completed step for tracking
        logger = get_logger('processing_error', 'PROCESSING')
        logger.debug(f"Processing step completed: {step_name} [Exception: {self.exception_id}]")
    
    def add_failed_step(
        self,
        step_name: str,
        failure_reason: str
    ) -> None:
        """
        Add failed processing step with failure details for debugging and recovery planning.
        
        Args:
            step_name: Name of the failed step
            failure_reason: Reason for step failure
        """
        # Add step name to failed steps list
        self.failed_steps.append(step_name)
        
        # Store failure reason in processing context
        self.processing_context[f'{step_name}_failure'] = {
            'reason': failure_reason,
            'failed_at': datetime.datetime.now().isoformat()
        }
        
        # Update processing error analysis
        # Log failed step for debugging
        logger = get_logger('processing_error', 'PROCESSING')
        logger.warning(f"Processing step failed: {step_name} - {failure_reason}")
    
    def _add_processing_recommendations(self) -> None:
        """Add processing-specific recovery recommendations."""
        if 'video' in self.processing_stage.lower():
            self.add_recovery_recommendation(
                "Verify video file format and codec compatibility",
                priority='HIGH'
            )
        elif 'normalization' in self.processing_stage.lower():
            self.add_recovery_recommendation(
                "Check normalization parameters and scaling factors",
                priority='HIGH'
            )
        elif 'conversion' in self.processing_stage.lower():
            self.add_recovery_recommendation(
                "Verify format conversion settings and target format",
                priority='HIGH'
            )
        
        # Add general processing recommendations
        self.add_recovery_recommendation(
            "Retry processing with modified parameters or smaller data subset",
            priority='MEDIUM'
        )


class SimulationError(PlumeSimulationException):
    """
    Specialized exception class for simulation execution errors including algorithm failures, 
    convergence issues, batch processing problems, and performance threshold violations with 
    comprehensive simulation context and performance tracking for scientific computing reliability.
    """
    
    def __init__(
        self,
        message: str,
        simulation_id: str,
        algorithm_name: str,
        simulation_context: Dict[str, Any] = None
    ):
        """
        Initialize simulation error with comprehensive simulation context and performance 
        tracking for batch processing reliability.
        
        Args:
            message: Simulation error message
            simulation_id: Unique identifier for the simulation
            algorithm_name: Name of the navigation algorithm
            simulation_context: Context information for simulation
        """
        # Initialize base PlumeSimulationException with SIMULATION category
        super().__init__(
            message=message,
            category=ErrorCategory.SIMULATION,
            severity=ErrorSeverity.HIGH,
            error_code=SIMULATION_ERROR_CODE,
            context=simulation_context
        )
        
        # Store simulation identification and algorithm information
        self.simulation_id = simulation_id
        self.algorithm_name = algorithm_name
        self.simulation_context = simulation_context or {}
        
        # Extract batch ID and simulation step from context
        self.batch_id = self.simulation_context.get('batch_id')
        self.simulation_step = self.simulation_context.get('simulation_step', 0)
        
        # Initialize performance metrics and execution time
        self.execution_time = 0.0
        self.performance_metrics: Dict[str, float] = {}
        self.algorithm_state: Dict[str, Any] = {}
        
        # Determine retryability based on failure type
        self.is_retryable = is_retryable(ErrorCategory.SIMULATION.value)
        
        # Set failure mode from simulation context
        self.failure_mode = self.simulation_context.get('failure_mode', 'unknown')
        
        # Add simulation-specific recovery recommendations
        self._add_simulation_recommendations()
        
        # Create audit trail entry for simulation failure
        create_audit_trail(
            action='SIMULATION_ERROR_CREATED',
            component='SIMULATION',
            action_details={
                'exception_id': self.exception_id,
                'simulation_id': simulation_id,
                'algorithm_name': algorithm_name,
                'failure_mode': self.failure_mode,
                'is_retryable': self.is_retryable
            },
            user_context='SYSTEM'
        )
        
        # Log simulation error with comprehensive context
        logger = get_logger('simulation_error', 'SIMULATION')
        logger.error(
            f"Simulation error: {algorithm_name} [{simulation_id}] - {message} "
            f"[Step: {self.simulation_step}, Retryable: {self.is_retryable}]"
        )
    
    def record_performance_metrics(
        self,
        metrics: Dict[str, float],
        execution_time: float = None
    ) -> None:
        """
        Record performance metrics at the time of simulation failure for analysis and optimization.
        
        Args:
            metrics: Performance metrics dictionary
            execution_time: Execution time for the simulation
        """
        # Store performance metrics in simulation context
        self.performance_metrics.update(metrics)
        
        # Update execution time if provided
        if execution_time is not None:
            self.execution_time = execution_time
        
        # Log performance metrics for failure analysis
        logger = get_logger('simulation_error', 'SIMULATION')
        logger.info(
            f"Performance metrics recorded for simulation {self.simulation_id}: "
            f"execution_time={self.execution_time:.3f}s"
        )
        
        # Update simulation failure statistics
        if self.execution_time > PROCESSING_TIME_TARGET_SECONDS:
            self.add_recovery_recommendation(
                f"Simulation exceeded target time ({PROCESSING_TIME_TARGET_SECONDS}s) - consider optimization",
                priority='MEDIUM'
            )
    
    def preserve_algorithm_state(
        self,
        algorithm_state: Dict[str, Any]
    ) -> None:
        """
        Preserve algorithm state at the time of failure for debugging and potential recovery.
        
        Args:
            algorithm_state: Algorithm state information
        """
        # Store algorithm state in simulation context
        self.algorithm_state = algorithm_state
        
        # Include algorithm parameters and configuration
        if 'parameters' not in self.algorithm_state:
            self.algorithm_state['parameters'] = self.simulation_context.get('parameters', {})
        
        # Log algorithm state preservation for debugging
        logger = get_logger('simulation_error', 'SIMULATION')
        logger.debug(f"Algorithm state preserved for simulation {self.simulation_id}")
        
        # Update recovery options with preserved state
        self.add_recovery_recommendation(
            "Algorithm state preserved - consider resuming from last checkpoint",
            priority='MEDIUM'
        )
    
    def get_failure_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive failure analysis including root cause identification and recovery recommendations.
        
        Returns:
            Dict[str, Any]: Simulation failure analysis with root cause and recovery guidance
        """
        # Analyze failure mode and simulation context
        analysis = {
            'simulation_id': self.simulation_id,
            'algorithm_name': self.algorithm_name,
            'failure_mode': self.failure_mode,
            'execution_time': self.execution_time,
            'simulation_step': self.simulation_step,
            'performance_metrics': self.performance_metrics,
            'algorithm_state': self.algorithm_state,
            'is_retryable': self.is_retryable
        }
        
        # Identify potential root causes
        root_causes = []
        if self.execution_time > PROCESSING_TIME_TARGET_SECONDS * 2:
            root_causes.append("Performance degradation - execution time exceeded limits")
        
        if 'convergence' in self.failure_mode.lower():
            root_causes.append("Algorithm convergence failure - adjust parameters")
        
        if 'memory' in str(self.performance_metrics).lower():
            root_causes.append("Memory-related issues - consider resource optimization")
        
        analysis['potential_root_causes'] = root_causes
        
        # Generate recovery recommendations
        recovery_recommendations = self.get_recovery_recommendations()
        analysis['recovery_recommendations'] = recovery_recommendations
        
        # Include performance impact assessment
        analysis['performance_impact'] = {
            'processing_time_ratio': self.execution_time / PROCESSING_TIME_TARGET_SECONDS if PROCESSING_TIME_TARGET_SECONDS > 0 else 0,
            'failure_severity': self.severity.value,
            'batch_impact': 'high' if self.batch_id else 'single_simulation'
        }
        
        # Add debugging information and suggestions
        analysis['debugging_info'] = {
            'exception_id': self.exception_id,
            'timestamp': self.timestamp.isoformat(),
            'context_available': bool(self.simulation_context),
            'state_preserved': bool(self.algorithm_state)
        }
        
        return analysis
    
    def _add_simulation_recommendations(self) -> None:
        """Add simulation-specific recovery recommendations."""
        if 'convergence' in self.failure_mode.lower():
            self.add_recovery_recommendation(
                "Adjust convergence criteria or increase iteration limits",
                priority='HIGH'
            )
        elif 'timeout' in self.failure_mode.lower():
            self.add_recovery_recommendation(
                "Increase timeout limits or optimize algorithm parameters",
                priority='HIGH'
            )
        elif 'memory' in self.failure_mode.lower():
            self.add_recovery_recommendation(
                "Reduce simulation complexity or increase available memory",
                priority='HIGH'
            )
        
        # Add algorithm-specific recommendations
        if self.algorithm_name:
            self.add_recovery_recommendation(
                f"Review {self.algorithm_name} algorithm configuration and parameters",
                priority='MEDIUM'
            )


class AnalysisError(PlumeSimulationException):
    """
    Specialized exception class for analysis pipeline errors including statistical analysis failures, 
    visualization errors, report generation issues, and performance analysis problems with partial 
    result preservation and analysis context tracking.
    """
    
    def __init__(
        self,
        message: str,
        analysis_type: str,
        analysis_context: Dict[str, Any] = None,
        input_data: Dict[str, Any] = None
    ):
        """
        Initialize analysis error with analysis context and partial result preservation 
        for graceful degradation in analysis pipeline.
        
        Args:
            message: Analysis error message
            analysis_type: Type of analysis that failed
            analysis_context: Context information for analysis
            input_data: Input data for analysis
        """
        # Initialize base PlumeSimulationException with ANALYSIS category
        super().__init__(
            message=message,
            category=ErrorCategory.ANALYSIS,
            severity=ErrorSeverity.MEDIUM,
            error_code=ANALYSIS_ERROR_CODE,
            context=analysis_context
        )
        
        # Store analysis type and context information
        self.analysis_type = analysis_type
        self.analysis_context = analysis_context or {}
        self.input_data = input_data or {}
        
        # Initialize partial results and progress tracking
        self.partial_results: Dict[str, Any] = {}
        self.completed_analyses: List[str] = []
        self.failed_analyses: List[str] = []
        self.analysis_progress = 0.0
        
        # Extract statistical context from analysis data
        self.statistical_context = self._extract_statistical_context()
        
        # Add analysis-specific recovery recommendations
        self._add_analysis_recommendations()
        
        # Create audit trail entry for analysis failure
        create_audit_trail(
            action='ANALYSIS_ERROR_CREATED',
            component='ANALYSIS',
            action_details={
                'exception_id': self.exception_id,
                'analysis_type': analysis_type,
                'input_data_size': len(self.input_data),
                'partial_results_available': bool(self.partial_results)
            },
            user_context='SYSTEM'
        )
        
        # Log analysis error with type and context
        logger = get_logger('analysis_error', 'ANALYSIS')
        logger.error(f"Analysis error: {analysis_type} - {message}")
    
    def preserve_partial_analysis(
        self,
        partial_results: Dict[str, Any],
        completion_percentage: float
    ) -> None:
        """
        Preserve partial analysis results for graceful degradation and potential recovery.
        
        Args:
            partial_results: Partial analysis results to preserve
            completion_percentage: Analysis completion percentage
        """
        # Store partial results in analysis context
        self.partial_results.update(partial_results)
        self.analysis_progress = completion_percentage
        
        # Log partial results preservation
        logger = get_logger('analysis_error', 'ANALYSIS')
        logger.info(
            f"Partial analysis results preserved: {completion_percentage:.1f}% complete "
            f"[Exception: {self.exception_id}]"
        )
        
        # Update recovery recommendations with partial analysis
        if completion_percentage > 0:
            self.add_recovery_recommendation(
                f"Partial analysis results available ({completion_percentage:.1f}% complete)",
                priority='MEDIUM'
            )
    
    def add_completed_analysis(
        self,
        analysis_name: str,
        analysis_results: Dict[str, Any] = None
    ) -> None:
        """
        Add successfully completed analysis component for progress tracking.
        
        Args:
            analysis_name: Name of the completed analysis
            analysis_results: Results from the completed analysis
        """
        # Add analysis name to completed analyses list
        self.completed_analyses.append(analysis_name)
        
        # Store analysis results in partial results
        if analysis_results:
            self.partial_results[f'{analysis_name}_results'] = analysis_results
        
        # Update analysis progress calculation
        total_analyses = len(self.completed_analyses) + len(self.failed_analyses)
        if total_analyses > 0:
            self.analysis_progress = len(self.completed_analyses) / total_analyses * 100
        
        # Log completed analysis for tracking
        logger = get_logger('analysis_error', 'ANALYSIS')
        logger.debug(f"Analysis component completed: {analysis_name}")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis failure summary with partial results and recovery guidance.
        
        Returns:
            Dict[str, Any]: Analysis failure summary with partial results and recommendations
        """
        # Compile completed and failed analyses
        summary = {
            'analysis_type': self.analysis_type,
            'completed_analyses': self.completed_analyses,
            'failed_analyses': self.failed_analyses,
            'analysis_progress': self.analysis_progress,
            'partial_results_available': bool(self.partial_results),
            'statistical_context': self.statistical_context,
            'input_data_summary': {
                'data_points': len(self.input_data),
                'data_types': list(self.input_data.keys()) if self.input_data else []
            },
            'recovery_recommendations': self.get_recovery_recommendations()
        }
        
        # Include partial results and progress information
        if self.partial_results:
            summary['partial_results_summary'] = {
                'available_results': list(self.partial_results.keys()),
                'completion_percentage': self.analysis_progress
            }
        
        return summary
    
    def _extract_statistical_context(self) -> Dict[str, Any]:
        """Extract statistical context from analysis data."""
        context = {}
        
        if self.input_data:
            # Basic statistical information
            context['data_points'] = len(self.input_data)
            context['data_complexity'] = 'high' if len(self.input_data) > 1000 else 'medium' if len(self.input_data) > 100 else 'low'
        
        if self.analysis_type:
            # Analysis type context
            context['analysis_category'] = 'statistical' if 'stat' in self.analysis_type.lower() else 'visualization' if 'plot' in self.analysis_type.lower() else 'general'
        
        return context
    
    def _add_analysis_recommendations(self) -> None:
        """Add analysis-specific recovery recommendations."""
        if 'statistical' in self.analysis_type.lower():
            self.add_recovery_recommendation(
                "Check statistical analysis parameters and data distribution",
                priority='HIGH'
            )
        elif 'visualization' in self.analysis_type.lower():
            self.add_recovery_recommendation(
                "Verify visualization parameters and data format",
                priority='HIGH'
            )
        elif 'report' in self.analysis_type.lower():
            self.add_recovery_recommendation(
                "Check report template and output configuration",
                priority='HIGH'
            )
        
        # Add general analysis recommendations
        self.add_recovery_recommendation(
            "Review analysis input data quality and completeness",
            priority='MEDIUM'
        )


class ConfigurationError(PlumeSimulationException):
    """
    Specialized exception class for configuration errors including invalid configuration parameters, 
    missing configuration files, schema validation failures, and configuration compatibility issues 
    with configuration context and correction guidance.
    """
    
    def __init__(
        self,
        message: str,
        config_file: str,
        config_section: str,
        config_context: Dict[str, Any] = None
    ):
        """
        Initialize configuration error with configuration context and parameter validation 
        details for configuration correction guidance.
        
        Args:
            message: Configuration error message
            config_file: Configuration file where error occurred
            config_section: Configuration section with error
            config_context: Context information for configuration
        """
        # Initialize base PlumeSimulationException with CONFIGURATION category
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            error_code=CONFIGURATION_ERROR_CODE,
            context=config_context
        )
        
        # Store configuration file and section information
        self.config_file = config_file
        self.config_section = config_section
        self.config_context = config_context or {}
        
        # Initialize invalid and missing parameters lists
        self.invalid_parameters: List[str] = []
        self.missing_parameters: List[str] = []
        self.default_values: Dict[str, Any] = {}
        
        # Extract configuration schema version if available
        self.config_schema_version = self.config_context.get('schema_version', 'unknown')
        
        # Determine if error is schema validation related
        self.is_schema_validation_error = 'schema' in message.lower()
        
        # Add configuration-specific recovery recommendations
        self._add_configuration_recommendations()
        
        # Create audit trail entry for configuration failure
        create_audit_trail(
            action='CONFIGURATION_ERROR_CREATED',
            component='CONFIGURATION',
            action_details={
                'exception_id': self.exception_id,
                'config_file': config_file,
                'config_section': config_section,
                'schema_validation_error': self.is_schema_validation_error
            },
            user_context='SYSTEM'
        )
        
        # Log configuration error with file and section context
        logger = get_logger('configuration_error', 'CONFIGURATION')
        logger.error(f"Configuration error in {config_file}[{config_section}]: {message}")
    
    def add_invalid_parameter(
        self,
        parameter_name: str,
        invalid_value: Any,
        validation_rule: str
    ) -> None:
        """
        Add invalid configuration parameter with details for configuration correction.
        
        Args:
            parameter_name: Name of the invalid parameter
            invalid_value: Invalid value that was provided
            validation_rule: Validation rule that was violated
        """
        # Add parameter name to invalid parameters list
        self.invalid_parameters.append(parameter_name)
        
        # Store invalid value and validation rule
        self.config_context[f'{parameter_name}_invalid'] = {
            'value': invalid_value,
            'validation_rule': validation_rule,
            'detected_at': datetime.datetime.now().isoformat()
        }
        
        # Log invalid parameter for debugging
        logger = get_logger('configuration_error', 'CONFIGURATION')
        logger.warning(
            f"Invalid parameter detected: {parameter_name} = {invalid_value} "
            f"(violates: {validation_rule})"
        )
    
    def add_missing_parameter(
        self,
        parameter_name: str,
        default_value: Any = None,
        parameter_description: str = None
    ) -> None:
        """
        Add missing required configuration parameter for configuration completion guidance.
        
        Args:
            parameter_name: Name of the missing parameter
            default_value: Default value for the parameter
            parameter_description: Description of the parameter
        """
        # Add parameter name to missing parameters list
        self.missing_parameters.append(parameter_name)
        
        # Store default value and description
        if default_value is not None:
            self.default_values[parameter_name] = default_value
        
        self.config_context[f'{parameter_name}_missing'] = {
            'default_value': default_value,
            'description': parameter_description,
            'detected_at': datetime.datetime.now().isoformat()
        }
        
        # Log missing parameter for configuration guidance
        logger = get_logger('configuration_error', 'CONFIGURATION')
        logger.warning(f"Missing parameter: {parameter_name}")
    
    def get_configuration_guidance(self) -> Dict[str, Any]:
        """
        Get comprehensive configuration correction guidance with parameter details and examples.
        
        Returns:
            Dict[str, Any]: Configuration correction guidance with parameter details and examples
        """
        # Compile invalid and missing parameters
        guidance = {
            'config_file': self.config_file,
            'config_section': self.config_section,
            'invalid_parameters': self.invalid_parameters,
            'missing_parameters': self.missing_parameters,
            'default_values': self.default_values,
            'schema_version': self.config_schema_version,
            'correction_steps': []
        }
        
        # Generate configuration correction examples
        if self.invalid_parameters:
            guidance['correction_steps'].append({
                'step': 'correct_invalid_parameters',
                'description': 'Review and correct invalid parameter values',
                'parameters': self.invalid_parameters
            })
        
        if self.missing_parameters:
            guidance['correction_steps'].append({
                'step': 'add_missing_parameters',
                'description': 'Add missing required parameters',
                'parameters': self.missing_parameters,
                'suggested_defaults': self.default_values
            })
        
        # Add schema validation guidance if applicable
        if self.is_schema_validation_error:
            guidance['correction_steps'].append({
                'step': 'validate_schema_compliance',
                'description': 'Ensure configuration follows schema requirements',
                'schema_version': self.config_schema_version
            })
        
        guidance['recovery_recommendations'] = self.get_recovery_recommendations()
        
        return guidance
    
    def _add_configuration_recommendations(self) -> None:
        """Add configuration-specific recovery recommendations."""
        if self.is_schema_validation_error:
            self.add_recovery_recommendation(
                "Validate configuration file against schema requirements",
                priority='HIGH'
            )
        
        self.add_recovery_recommendation(
            f"Review configuration file structure: {self.config_file}",
            priority='HIGH'
        )
        
        self.add_recovery_recommendation(
            f"Check section [{self.config_section}] for missing or invalid parameters",
            priority='MEDIUM'
        )


class ResourceError(PlumeSimulationException):
    """
    Specialized exception class for resource-related errors including memory exhaustion, 
    disk space issues, CPU overload, and system resource conflicts with system state 
    capture and resource optimization recommendations.
    """
    
    def __init__(
        self,
        message: str,
        resource_type: str,
        resource_context: Dict[str, Any] = None,
        resource_usage: Dict[str, float] = None
    ):
        """
        Initialize resource error with system state capture and resource usage analysis 
        for optimization recommendations.
        
        Args:
            message: Resource error message
            resource_type: Type of resource (memory, disk, cpu, network)
            resource_context: Context information for resource
            resource_usage: Current resource usage metrics
        """
        # Initialize base PlumeSimulationException with RESOURCE category
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.CRITICAL,
            error_code=RESOURCE_ERROR_CODE,
            context=resource_context
        )
        
        # Store resource type and usage information
        self.resource_type = resource_type
        self.resource_context = resource_context or {}
        self.resource_usage = resource_usage or {}
        
        # Initialize resource limits and system state
        self.resource_limits: Dict[str, float] = {}
        self.system_state: Dict[str, Any] = {}
        
        # Determine if resource error is recoverable
        self.is_recoverable = resource_type in ['disk', 'memory', 'cpu']
        
        # Initialize optimization recommendations
        self.optimization_recommendations: List[str] = []
        
        # Analyze resource exhaustion type and patterns
        self.resource_exhaustion_type = self._analyze_resource_exhaustion()
        
        # Capture current system state and resource limits
        self.capture_system_state(include_process_details=True)
        
        # Generate resource optimization recommendations
        self._generate_optimization_recommendations()
        
        # Add resource-specific recovery recommendations
        self._add_resource_recommendations()
        
        # Create audit trail entry for resource failure
        create_audit_trail(
            action='RESOURCE_ERROR_CREATED',
            component='RESOURCE_MANAGEMENT',
            action_details={
                'exception_id': self.exception_id,
                'resource_type': resource_type,
                'resource_usage': self.resource_usage,
                'is_recoverable': self.is_recoverable,
                'exhaustion_type': self.resource_exhaustion_type
            },
            user_context='SYSTEM'
        )
        
        # Log resource error with system state context
        logger = get_logger('resource_error', 'RESOURCE_MANAGEMENT')
        logger.critical(f"Resource error: {resource_type} - {message}")
    
    def capture_system_state(
        self,
        include_process_details: bool = False
    ) -> None:
        """
        Capture comprehensive system state at the time of resource error for analysis and optimization.
        
        Args:
            include_process_details: Whether to include detailed process information
        """
        import psutil
        
        try:
            # Capture current memory usage and availability
            memory_info = psutil.virtual_memory()
            self.system_state['memory'] = {
                'total_gb': memory_info.total / (1024**3),
                'available_gb': memory_info.available / (1024**3),
                'used_gb': memory_info.used / (1024**3),
                'percent_used': memory_info.percent
            }
            
            # Record CPU usage and load averages
            cpu_info = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'cpu_count': psutil.cpu_count(),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
            self.system_state['cpu'] = cpu_info
            
            # Monitor disk space and I/O statistics
            disk_info = psutil.disk_usage('/')
            self.system_state['disk'] = {
                'total_gb': disk_info.total / (1024**3),
                'free_gb': disk_info.free / (1024**3),
                'used_gb': disk_info.used / (1024**3),
                'percent_used': (disk_info.used / disk_info.total) * 100
            }
            
            # Include process details if requested
            if include_process_details:
                current_process = psutil.Process()
                self.system_state['current_process'] = {
                    'pid': current_process.pid,
                    'memory_mb': current_process.memory_info().rss / (1024**2),
                    'cpu_percent': current_process.cpu_percent(),
                    'num_threads': current_process.num_threads()
                }
        
        except ImportError:
            # psutil not available - use basic system information
            self.system_state['note'] = 'psutil not available for detailed system monitoring'
        except Exception as e:
            self.system_state['capture_error'] = str(e)
        
        # Store system state in resource context
        self.resource_context['system_state_capture'] = datetime.datetime.now().isoformat()
        
        # Log system state capture for analysis
        logger = get_logger('resource_error', 'RESOURCE_MANAGEMENT')
        logger.info(f"System state captured for resource error analysis: {self.exception_id}")
    
    def analyze_resource_bottleneck(self) -> Dict[str, Any]:
        """
        Analyze resource bottleneck patterns and generate optimization recommendations.
        
        Returns:
            Dict[str, Any]: Resource bottleneck analysis with optimization recommendations
        """
        # Analyze resource usage patterns and trends
        analysis = {
            'resource_type': self.resource_type,
            'exhaustion_type': self.resource_exhaustion_type,
            'system_state': self.system_state,
            'bottleneck_analysis': {},
            'optimization_strategies': []
        }
        
        # Identify primary resource bottlenecks
        if self.resource_type == 'memory':
            if self.system_state.get('memory', {}).get('percent_used', 0) > 90:
                analysis['bottleneck_analysis']['primary_issue'] = 'memory_exhaustion'
                analysis['optimization_strategies'].append('Reduce memory-intensive operations')
                analysis['optimization_strategies'].append('Implement memory-efficient algorithms')
        
        elif self.resource_type == 'cpu':
            if self.system_state.get('cpu', {}).get('cpu_percent', 0) > 90:
                analysis['bottleneck_analysis']['primary_issue'] = 'cpu_overload'
                analysis['optimization_strategies'].append('Reduce computational complexity')
                analysis['optimization_strategies'].append('Implement parallel processing')
        
        elif self.resource_type == 'disk':
            if self.system_state.get('disk', {}).get('percent_used', 0) > 95:
                analysis['bottleneck_analysis']['primary_issue'] = 'disk_space_exhaustion'
                analysis['optimization_strategies'].append('Clean up temporary files')
                analysis['optimization_strategies'].append('Compress or archive old data')
        
        # Generate resource optimization strategies
        analysis['optimization_strategies'].extend(self.optimization_recommendations)
        
        # Include system configuration recommendations
        analysis['system_recommendations'] = [
            'Monitor resource usage trends',
            'Implement resource usage alerts',
            'Consider system resource upgrades if persistent'
        ]
        
        return analysis
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive resource error summary with usage analysis and optimization recommendations.
        
        Returns:
            Dict[str, Any]: Resource error summary with usage analysis and optimization recommendations
        """
        # Compile resource usage and limits information
        summary = {
            'resource_type': self.resource_type,
            'resource_usage': self.resource_usage,
            'resource_limits': self.resource_limits,
            'system_state': self.system_state,
            'is_recoverable': self.is_recoverable,
            'exhaustion_type': self.resource_exhaustion_type,
            'optimization_recommendations': self.optimization_recommendations,
            'bottleneck_analysis': self.analyze_resource_bottleneck(),
            'recovery_recommendations': self.get_recovery_recommendations()
        }
        
        # Include recovery feasibility assessment
        summary['recovery_feasibility'] = {
            'automatic_recovery_possible': self.is_recoverable,
            'manual_intervention_required': not self.is_recoverable,
            'estimated_recovery_time': 'immediate' if self.is_recoverable else 'requires_admin_action'
        }
        
        return summary
    
    def _analyze_resource_exhaustion(self) -> str:
        """Analyze the type of resource exhaustion from context and usage."""
        if self.resource_type == 'memory':
            if any('allocation' in str(v).lower() for v in self.resource_usage.values()):
                return 'memory_allocation_failure'
            else:
                return 'memory_exhaustion'
        elif self.resource_type == 'cpu':
            return 'cpu_overload'
        elif self.resource_type == 'disk':
            return 'disk_space_exhaustion'
        else:
            return 'unknown_resource_exhaustion'
    
    def _generate_optimization_recommendations(self) -> None:
        """Generate resource-specific optimization recommendations."""
        if self.resource_type == 'memory':
            self.optimization_recommendations.extend([
                "Reduce batch size to lower memory requirements",
                "Implement streaming processing for large datasets",
                "Use memory-efficient data structures",
                "Clear intermediate results when no longer needed"
            ])
        elif self.resource_type == 'cpu':
            self.optimization_recommendations.extend([
                "Optimize algorithm complexity and efficiency",
                "Implement parallel processing where possible",
                "Reduce computational overhead in loops",
                "Consider using more efficient algorithms"
            ])
        elif self.resource_type == 'disk':
            self.optimization_recommendations.extend([
                "Clean up temporary and intermediate files",
                "Compress large data files",
                "Implement data archiving strategies",
                "Monitor and manage disk space usage"
            ])
    
    def _add_resource_recommendations(self) -> None:
        """Add resource-specific recovery recommendations."""
        self.add_recovery_recommendation(
            f"Address {self.resource_type} resource constraints immediately",
            priority='CRITICAL'
        )
        
        if self.is_recoverable:
            self.add_recovery_recommendation(
                "Reduce resource usage and retry operation",
                priority='HIGH'
            )
        else:
            self.add_recovery_recommendation(
                "Contact system administrator for resource allocation",
                priority='HIGH'
            )


class SystemError(PlumeSimulationException):
    """
    Specialized exception class for system-level errors including platform compatibility issues, 
    dependency failures, environment configuration problems, and critical system failures with 
    diagnostic capabilities and system health assessment.
    """
    
    def __init__(
        self,
        message: str,
        system_component: str,
        system_context: Dict[str, Any] = None,
        error_source: str = 'unknown'
    ):
        """
        Initialize system error with comprehensive system diagnostics and health assessment 
        for critical system failure handling.
        
        Args:
            message: System error message
            system_component: System component where error occurred
            system_context: Context information for system
            error_source: Source of the system error
        """
        # Initialize base PlumeSimulationException with SYSTEM category
        super().__init__(
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            error_code=SYSTEM_ERROR_CODE,
            context=system_context
        )
        
        # Store system component and error source information
        self.system_component = system_component
        self.system_context = system_context or {}
        self.error_source = error_source
        
        # Initialize environment and dependency information
        self.environment_info: Dict[str, str] = {}
        self.dependency_issues: List[str] = []
        self.diagnostic_data: Dict[str, Any] = {}
        
        # Assess system health status and restart requirements
        self.requires_system_restart = False
        self.system_health_status = 'unknown'
        
        # Collect system diagnostic information
        self._collect_environment_info()
        self._collect_diagnostic_data()
        
        # Check dependency health
        dependency_health = self.check_dependency_health()
        self._process_dependency_health(dependency_health)
        
        # Generate system-specific recovery recommendations
        self._generate_system_recovery_plan()
        
        # Add critical system failure handling guidance
        self._add_system_recommendations()
        
        # Create audit trail entry for system failure
        create_audit_trail(
            action='SYSTEM_ERROR_CREATED',
            component='SYSTEM_MANAGEMENT',
            action_details={
                'exception_id': self.exception_id,
                'system_component': system_component,
                'error_source': error_source,
                'requires_restart': self.requires_system_restart,
                'health_status': self.system_health_status
            },
            user_context='SYSTEM'
        )
        
        # Log system error with comprehensive diagnostics
        logger = get_logger('system_error', 'SYSTEM_MANAGEMENT')
        logger.critical(
            f"System error in {system_component}: {message} "
            f"[Source: {error_source}, Health: {self.system_health_status}]"
        )
    
    def generate_diagnostic_report(
        self,
        include_environment_details: bool = True,
        include_dependency_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive system diagnostic report for troubleshooting and system health assessment.
        
        Args:
            include_environment_details: Whether to include environment details
            include_dependency_analysis: Whether to include dependency analysis
            
        Returns:
            Dict[str, Any]: Comprehensive system diagnostic report with health assessment
        """
        # Compile system component and error information
        report = {
            'diagnostic_id': str(uuid.uuid4()),
            'generation_timestamp': datetime.datetime.now().isoformat(),
            'system_component': self.system_component,
            'error_source': self.error_source,
            'system_health_status': self.system_health_status,
            'requires_system_restart': self.requires_system_restart,
            'diagnostic_data': self.diagnostic_data
        }
        
        # Include environment details if requested
        if include_environment_details:
            report['environment_info'] = self.environment_info
            report['system_context'] = self.system_context
        
        # Add dependency analysis if requested
        if include_dependency_analysis:
            report['dependency_issues'] = self.dependency_issues
            report['dependency_health'] = self.check_dependency_health()
        
        # Generate system health assessment
        report['health_assessment'] = {
            'overall_status': self.system_health_status,
            'critical_issues': len([issue for issue in self.dependency_issues if 'critical' in issue.lower()]),
            'total_issues': len(self.dependency_issues),
            'restart_required': self.requires_system_restart
        }
        
        # Include troubleshooting recommendations
        report['troubleshooting_recommendations'] = self.get_recovery_recommendations()
        
        return report
    
    def check_dependency_health(self) -> Dict[str, bool]:
        """
        Check health and availability of system dependencies for dependency issue identification.
        
        Returns:
            Dict[str, bool]: Dependency health status with availability information
        """
        dependency_health = {}
        
        # Check critical Python dependencies
        critical_dependencies = [
            'numpy', 'opencv-python', 'jsonschema', 'psutil'
        ]
        
        for dependency in critical_dependencies:
            try:
                __import__(dependency.replace('-', '_'))
                dependency_health[dependency] = True
            except ImportError:
                dependency_health[dependency] = False
                self.dependency_issues.append(f"Missing critical dependency: {dependency}")
        
        # Check system-level dependencies
        system_dependencies = {
            'python_version': sys.version_info >= (3, 9),
            'threading_support': hasattr(threading, 'Lock'),
            'json_support': hasattr(json, 'loads')
        }
        
        dependency_health.update(system_dependencies)
        
        # Test dependency functionality
        try:
            # Test JSON functionality
            json.loads('{"test": true}')
            dependency_health['json_functional'] = True
        except Exception:
            dependency_health['json_functional'] = False
            self.dependency_issues.append("JSON functionality not working")
        
        # Identify dependency issues and conflicts
        failed_dependencies = [dep for dep, status in dependency_health.items() if not status]
        if failed_dependencies:
            self.dependency_issues.extend([f"Failed dependency: {dep}" for dep in failed_dependencies])
        
        return dependency_health
    
    def get_system_recovery_plan(self) -> Dict[str, Any]:
        """
        Get comprehensive system recovery plan with step-by-step recovery guidance.
        
        Returns:
            Dict[str, Any]: System recovery plan with step-by-step guidance and priority actions
        """
        # Analyze system error and component failure
        recovery_plan = {
            'recovery_id': str(uuid.uuid4()),
            'system_component': self.system_component,
            'error_source': self.error_source,
            'recovery_steps': [],
            'priority_actions': [],
            'verification_steps': []
        }
        
        # Generate priority recovery actions
        if self.requires_system_restart:
            recovery_plan['priority_actions'].append({
                'action': 'system_restart',
                'description': 'Restart system services and components',
                'priority': 'CRITICAL',
                'estimated_time': '5-10 minutes'
            })
        
        # Include dependency resolution steps
        if self.dependency_issues:
            recovery_plan['recovery_steps'].append({
                'step': 'resolve_dependencies',
                'description': 'Install or repair missing dependencies',
                'actions': [f"Install {issue.split(': ')[1]}" for issue in self.dependency_issues if 'Missing' in issue],
                'priority': 'HIGH'
            })
        
        # Add system restart guidance if required
        if self.requires_system_restart:
            recovery_plan['recovery_steps'].append({
                'step': 'restart_services',
                'description': 'Restart affected system services',
                'actions': [
                    f"Restart {self.system_component} service",
                    "Verify service status after restart",
                    "Check service logs for errors"
                ],
                'priority': 'CRITICAL'
            })
        
        # Include verification and testing steps
        recovery_plan['verification_steps'].extend([
            {
                'step': 'verify_dependencies',
                'description': 'Verify all dependencies are available and functional',
                'test_command': 'python -c "import sys; print(sys.version)"'
            },
            {
                'step': 'test_system_functionality',
                'description': 'Test basic system functionality',
                'test_command': 'Basic system health check'
            }
        ])
        
        recovery_plan['estimated_total_time'] = '15-30 minutes'
        recovery_plan['success_probability'] = 'high' if not self.requires_system_restart else 'medium'
        
        return recovery_plan
    
    def _collect_environment_info(self) -> None:
        """Collect comprehensive environment information."""
        self.environment_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'python_executable': sys.executable,
            'python_path': str(sys.path[:3]),  # First 3 entries for brevity
            'working_directory': str(pathlib.Path.cwd()) if 'pathlib' in globals() else 'unknown'
        }
        
        # Add environment variables relevant to the system
        import os
        relevant_env_vars = ['PATH', 'PYTHONPATH', 'HOME', 'USER']
        for var in relevant_env_vars:
            if var in os.environ:
                self.environment_info[f'env_{var}'] = os.environ[var][:100]  # Truncate for security
    
    def _collect_diagnostic_data(self) -> None:
        """Collect comprehensive diagnostic data."""
        self.diagnostic_data = {
            'error_timestamp': self.timestamp.isoformat(),
            'system_component': self.system_component,
            'error_source': self.error_source,
            'thread_info': {
                'current_thread': threading.current_thread().name,
                'thread_count': threading.active_count()
            }
        }
        
        # Add memory and system information if available
        try:
            import psutil
            self.diagnostic_data['system_info'] = {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3)
            }
        except ImportError:
            self.diagnostic_data['system_info'] = 'psutil not available'
    
    def _process_dependency_health(self, dependency_health: Dict[str, bool]) -> None:
        """Process dependency health results and update system status."""
        failed_dependencies = sum(1 for status in dependency_health.values() if not status)
        total_dependencies = len(dependency_health)
        
        if failed_dependencies == 0:
            self.system_health_status = 'healthy'
        elif failed_dependencies <= total_dependencies * 0.25:
            self.system_health_status = 'minor_issues'
        elif failed_dependencies <= total_dependencies * 0.5:
            self.system_health_status = 'degraded'
        else:
            self.system_health_status = 'critical'
            self.requires_system_restart = True
    
    def _generate_system_recovery_plan(self) -> None:
        """Generate initial system recovery plan."""
        # This method sets up the foundation for recovery planning
        # Actual recovery plan is generated by get_system_recovery_plan()
        pass
    
    def _add_system_recommendations(self) -> None:
        """Add system-specific recovery recommendations."""
        if self.system_health_status == 'critical':
            self.add_recovery_recommendation(
                "Immediate system attention required - critical health status",
                priority='CRITICAL'
            )
        
        if self.requires_system_restart:
            self.add_recovery_recommendation(
                "System restart required to resolve issues",
                priority='CRITICAL'
            )
        
        if self.dependency_issues:
            self.add_recovery_recommendation(
                f"Resolve {len(self.dependency_issues)} dependency issues",
                priority='HIGH'
            )
        
        self.add_recovery_recommendation(
            f"Check {self.system_component} component logs for additional details",
            priority='MEDIUM'
        )


# Register all exception types in the global registry
register_exception_type(PlumeSimulationException, BASE_ERROR_CODE, 'SYSTEM')
register_exception_type(ValidationError, VALIDATION_ERROR_CODE, 'VALIDATION')
register_exception_type(ProcessingError, PROCESSING_ERROR_CODE, 'PROCESSING')
register_exception_type(SimulationError, SIMULATION_ERROR_CODE, 'SIMULATION')
register_exception_type(AnalysisError, ANALYSIS_ERROR_CODE, 'ANALYSIS')
register_exception_type(ConfigurationError, CONFIGURATION_ERROR_CODE, 'CONFIGURATION')
register_exception_type(ResourceError, RESOURCE_ERROR_CODE, 'RESOURCE')
register_exception_type(SystemError, SYSTEM_ERROR_CODE, 'SYSTEM')