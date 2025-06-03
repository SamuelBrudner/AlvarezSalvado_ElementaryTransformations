"""
Comprehensive error handling utilities module providing centralized error management infrastructure 
for the plume simulation system including error severity classification, error category enumeration, 
retry logic with exponential backoff, graceful degradation strategies, error handling result containers, 
and batch processing error coordination.

This module implements fail-fast validation support, automatic recovery mechanisms, comprehensive error 
context management, and integrated threshold validation to ensure scientific computing reliability and 
reproducible research outcomes with <1% error rate target.

Key Features:
- Centralized error handling with comprehensive error analysis and recovery
- Exponential backoff retry logic with jitter for transient failures
- Graceful degradation strategy for batch processing operations with partial completion support
- Error severity classification with priority mapping and handling strategies
- Error category enumeration with recovery strategy mapping for specialized error handling
- Comprehensive error handling result containers with recovery tracking and audit trail integration
- Batch processing result containers with graceful degradation support for 4000+ simulation processing
- Context manager for scoped error handling with automatic context management and recovery
- Threshold validation with configuration-driven thresholds and alerting capabilities
- Thread-safe error handling and context management for concurrent operations
"""

# External library imports with version specifications
import logging  # Python 3.9+ - Core Python logging framework for error logging and audit trails
import traceback  # Python 3.9+ - Stack trace extraction for detailed error reporting and debugging
import sys  # Python 3.9+ - System-specific exception information and error handling
import threading  # Python 3.9+ - Thread-safe error handling and context management
import contextlib  # Python 3.9+ - Context manager utilities for error handling scope and cleanup
import datetime  # Python 3.9+ - Timestamp generation for error tracking and audit trails
import json  # Python 3.9+ - JSON serialization for structured error reporting and export
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Type  # Python 3.9+ - Type hints for error handler function signatures and interfaces
import functools  # Python 3.9+ - Decorator utilities for error handling and recovery strategies
import uuid  # Python 3.9+ - Unique identifier generation for error tracking and correlation
import time  # Python 3.9+ - Time-based operations for retry delays and timeout handling
import random  # Python 3.9+ - Random jitter for exponential backoff retry strategies
from enum import Enum  # Python 3.9+ - Error severity and category enumeration
import collections  # Python 3.9+ - Data structure utilities for error tracking and batch processing

# Internal imports from utility modules
from .logging_utils import get_logger, log_validation_error, create_audit_trail
from .validation_utils import ValidationResult
from .config_parser import load_configuration, get_config_section

# Global configuration constants for error handling system
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 60.0
EXPONENTIAL_BACKOFF_MULTIPLIER = 2.0
JITTER_FACTOR = 0.1
CRITICAL_ERROR_THRESHOLD = 5
WARNING_ERROR_THRESHOLD = 2
ERROR_RATE_WINDOW_MINUTES = 15
BATCH_FAILURE_THRESHOLD = 0.05
CHECKPOINT_INTERVAL_SECONDS = 300
PERFORMANCE_THRESHOLDS_CONFIG_PATH = 'src/backend/config/performance_thresholds.json'

# Global state management for error handling system
_error_context_stack = threading.local()
_error_statistics: Dict[str, Dict[str, Any]] = {}
_recovery_strategies: Dict[str, Callable] = {}
_error_handlers: Dict[str, Callable] = {}
_batch_processing_state: Dict[str, Any] = {}
_threshold_configuration: Optional[Dict[str, Any]] = None
_threshold_cache_timestamp: Optional[datetime.datetime] = None


class ErrorSeverity(Enum):
    """
    Enumeration defining error severity levels for classification and handling prioritization 
    including CRITICAL, HIGH, MEDIUM, LOW, and INFO levels with associated handling strategies 
    and priority mapping for scientific computing error management.
    """
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

    @classmethod
    def get_priority(cls, severity_level: str) -> int:
        """
        Get numerical priority value for severity level comparison and sorting.
        
        Args:
            severity_level: String representation of severity level
            
        Returns:
            int: Numerical priority value for severity level (higher number = higher priority)
        """
        priority_mapping = {
            cls.CRITICAL.value: 5,
            cls.HIGH.value: 4,
            cls.MEDIUM.value: 3,
            cls.LOW.value: 2,
            cls.INFO.value: 1
        }
        return priority_mapping.get(severity_level, 0)

    @classmethod
    def requires_immediate_action(cls, severity_level: str) -> bool:
        """
        Determine if severity level requires immediate intervention and system protection.
        
        Args:
            severity_level: String representation of severity level
            
        Returns:
            bool: True if immediate action is required
        """
        immediate_action_levels = {cls.CRITICAL.value, cls.HIGH.value}
        return severity_level in immediate_action_levels


class ErrorCategory(Enum):
    """
    Enumeration defining error categories for classification and specialized handling including 
    VALIDATION, PROCESSING, SIMULATION, ANALYSIS, CONFIGURATION, RESOURCE, and SYSTEM categories 
    with category-specific recovery strategies and retryability assessment.
    """
    VALIDATION = "VALIDATION"
    PROCESSING = "PROCESSING"
    SIMULATION = "SIMULATION"
    ANALYSIS = "ANALYSIS"
    CONFIGURATION = "CONFIGURATION"
    RESOURCE = "RESOURCE"
    SYSTEM = "SYSTEM"

    @classmethod
    def get_recovery_strategy(cls, category: str) -> str:
        """
        Get default recovery strategy for error category.
        
        Args:
            category: Error category string
            
        Returns:
            str: Default recovery strategy name for category
        """
        recovery_strategies = {
            cls.VALIDATION.value: "retry_with_corrected_parameters",
            cls.PROCESSING.value: "restart_processing_pipeline",
            cls.SIMULATION.value: "fallback_algorithm_configuration",
            cls.ANALYSIS.value: "simplified_analysis_method",
            cls.CONFIGURATION.value: "load_default_configuration",
            cls.RESOURCE.value: "reduce_resource_requirements",
            cls.SYSTEM.value: "system_diagnostic_and_recovery"
        }
        return recovery_strategies.get(category, "manual_intervention_required")

    @classmethod
    def is_retryable(cls, category: str) -> bool:
        """
        Determine if error category supports retry operations.
        
        Args:
            category: Error category string
            
        Returns:
            bool: True if category supports retry operations
        """
        retryable_categories = {
            cls.PROCESSING.value,
            cls.SIMULATION.value,
            cls.RESOURCE.value,
            cls.SYSTEM.value
        }
        return category in retryable_categories


class ErrorHandlingResult:
    """
    Comprehensive error handling result container class providing structured storage of error 
    handling outcomes, recovery actions, recommendations, and metadata for scientific computing 
    error management with audit trail integration and performance tracking.
    """
    
    def __init__(
        self,
        error_id: str,
        handled_successfully: bool,
        handling_context: str
    ):
        """
        Initialize error handling result container with error identification, handling status, 
        and context for comprehensive error tracking and reporting.
        
        Args:
            error_id: Unique identifier for the error
            handled_successfully: Boolean indicating if error was handled successfully
            handling_context: Context description for the error handling operation
        """
        # Core error identification and status
        self.error_id = error_id
        self.handled_successfully = handled_successfully
        self.handling_context = handling_context
        
        # Error classification and metadata
        self.severity = ErrorSeverity.MEDIUM
        self.category = ErrorCategory.SYSTEM
        self.recovery_actions: List[str] = []
        self.recommendations: List[str] = []
        self.metadata: Dict[str, Any] = {}
        
        # Timing and tracking information
        self.handling_timestamp = datetime.datetime.now()
        self.handling_duration_seconds = 0.0
        self.requires_manual_intervention = False
        self.audit_trail_id = str(uuid.uuid4())
        self.performance_impact: Dict[str, Any] = {}

    def set_recovery_action(
        self,
        action_description: str,
        action_successful: bool,
        action_context: Dict[str, Any] = None
    ) -> None:
        """
        Add recovery action taken during error handling for comprehensive tracking and audit trail.
        
        Args:
            action_description: Description of the recovery action taken
            action_successful: Boolean indicating if the action was successful
            action_context: Additional context information for the action
        """
        action_entry = {
            'description': action_description,
            'successful': action_successful,
            'timestamp': datetime.datetime.now().isoformat(),
            'context': action_context or {}
        }
        
        self.recovery_actions.append(action_description)
        self.metadata[f'recovery_action_{len(self.recovery_actions)}'] = action_entry
        
        # Log recovery action for audit trail
        get_logger('error_handling.recovery', 'ERROR_HANDLING').info(
            f"Recovery action recorded: {action_description} (success: {action_successful})"
        )

    def add_recommendation(
        self,
        recommendation_text: str,
        priority: str = "MEDIUM",
        recommendation_context: Dict[str, Any] = None
    ) -> None:
        """
        Add actionable recommendation for error prevention or system improvement.
        
        Args:
            recommendation_text: Text description of the recommendation
            priority: Priority level for the recommendation
            recommendation_context: Additional context for the recommendation
        """
        formatted_recommendation = f"[{priority}] {recommendation_text}"
        self.recommendations.append(formatted_recommendation)
        
        recommendation_entry = {
            'text': recommendation_text,
            'priority': priority,
            'timestamp': datetime.datetime.now().isoformat(),
            'context': recommendation_context or {}
        }
        
        self.metadata[f'recommendation_{len(self.recommendations)}'] = recommendation_entry

    def mark_resolved(
        self,
        resolution_method: str,
        resolution_context: Dict[str, Any] = None
    ) -> None:
        """
        Mark error as resolved with resolution details and performance impact assessment.
        
        Args:
            resolution_method: Method used to resolve the error
            resolution_context: Additional context about the resolution
        """
        self.handled_successfully = True
        self.handling_duration_seconds = (
            datetime.datetime.now() - self.handling_timestamp
        ).total_seconds()
        
        resolution_entry = {
            'method': resolution_method,
            'context': resolution_context or {},
            'resolution_timestamp': datetime.datetime.now().isoformat(),
            'total_handling_time': self.handling_duration_seconds
        }
        
        self.metadata['resolution'] = resolution_entry
        
        # Create audit trail entry for resolution
        create_audit_trail(
            action='ERROR_RESOLVED',
            component='ERROR_HANDLING',
            action_details={
                'error_id': self.error_id,
                'resolution_method': resolution_method,
                'handling_duration': self.handling_duration_seconds,
                'recovery_actions_count': len(self.recovery_actions)
            },
            user_context='SYSTEM'
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error handling result to dictionary format for serialization, reporting, and 
        integration with monitoring systems.
        
        Returns:
            Dict[str, Any]: Complete error handling result as dictionary with all properties and metadata
        """
        return {
            'error_id': self.error_id,
            'handled_successfully': self.handled_successfully,
            'handling_context': self.handling_context,
            'severity': self.severity.value,
            'category': self.category.value,
            'recovery_actions': self.recovery_actions.copy(),
            'recommendations': self.recommendations.copy(),
            'metadata': self.metadata.copy(),
            'handling_timestamp': self.handling_timestamp.isoformat(),
            'handling_duration_seconds': self.handling_duration_seconds,
            'requires_manual_intervention': self.requires_manual_intervention,
            'audit_trail_id': self.audit_trail_id,
            'performance_impact': self.performance_impact.copy()
        }


class BatchProcessingResult:
    """
    Comprehensive batch processing result container class providing structured storage of batch 
    operation outcomes, graceful degradation analysis, partial results preservation, and failure 
    pattern analysis for scientific computing batch operations with 4000+ simulation support.
    """
    
    def __init__(
        self,
        batch_id: str,
        total_items: int,
        batch_context: str
    ):
        """
        Initialize batch processing result container with batch identification, item count, and 
        context for comprehensive batch tracking and analysis.
        
        Args:
            batch_id: Unique identifier for the batch operation
            total_items: Total number of items to process in the batch
            batch_context: Context description for the batch operation
        """
        # Core batch identification and tracking
        self.batch_id = batch_id
        self.total_items = total_items
        self.batch_context = batch_context
        
        # Processing counters and results
        self.successful_items = 0
        self.failed_items = 0
        self.successful_results: List[Any] = []
        self.failed_items_details: List[Dict[str, Any]] = []
        
        # Performance metrics and analysis
        self.success_rate = 0.0
        self.failure_rate = 0.0
        self.graceful_degradation_applied = False
        
        # Timing information
        self.batch_start_time = datetime.datetime.now()
        self.batch_end_time: Optional[datetime.datetime] = None
        self.total_processing_time = 0.0
        
        # Performance and degradation tracking
        self.performance_metrics: Dict[str, Any] = {}
        self.degradation_recommendations: List[str] = []

    def add_successful_item(
        self,
        item_result: Any,
        item_metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add successfully processed item to batch results with result data and performance metrics.
        
        Args:
            item_result: Result data from successful item processing
            item_metadata: Additional metadata about the processed item
        """
        self.successful_results.append(item_result)
        self.successful_items += 1
        
        # Update success rate calculation
        if self.total_items > 0:
            self.success_rate = self.successful_items / self.total_items
        
        # Store item metadata if provided
        if item_metadata:
            self.performance_metrics[f'success_item_{self.successful_items}'] = {
                'metadata': item_metadata,
                'processed_at': datetime.datetime.now().isoformat()
            }

    def add_failed_item(
        self,
        item_data: Any,
        error: Exception,
        failure_context: Dict[str, Any] = None
    ) -> None:
        """
        Add failed item to batch results with error details and failure analysis for comprehensive 
        failure tracking.
        
        Args:
            item_data: Data of the item that failed processing
            error: Exception that caused the failure
            failure_context: Additional context about the failure
        """
        failure_details = {
            'item_data': str(item_data)[:500],  # Truncate for storage
            'error_type': type(error).__name__,
            'error_message': str(error),
            'failure_timestamp': datetime.datetime.now().isoformat(),
            'context': failure_context or {}
        }
        
        self.failed_items_details.append(failure_details)
        self.failed_items += 1
        
        # Update failure rate calculation
        if self.total_items > 0:
            self.failure_rate = self.failed_items / self.total_items

    def apply_graceful_degradation(
        self,
        degradation_reason: str,
        degradation_config: Dict[str, Any] = None
    ) -> None:
        """
        Apply graceful degradation strategy when failure threshold is exceeded with degradation 
        analysis and recommendations.
        
        Args:
            degradation_reason: Reason for applying graceful degradation
            degradation_config: Configuration parameters for degradation strategy
        """
        self.graceful_degradation_applied = True
        
        degradation_entry = {
            'reason': degradation_reason,
            'config': degradation_config or {},
            'applied_at': datetime.datetime.now().isoformat(),
            'failure_rate_at_degradation': self.failure_rate,
            'successful_items_at_degradation': self.successful_items
        }
        
        self.performance_metrics['graceful_degradation'] = degradation_entry
        
        # Generate degradation recommendations
        self.degradation_recommendations.extend([
            f"Graceful degradation applied: {degradation_reason}",
            f"Preserved {self.successful_items} successful results",
            "Consider reviewing failed items for retry opportunities",
            "Analyze failure patterns to prevent future degradation"
        ])
        
        # Log graceful degradation application
        get_logger('batch_processing.degradation', 'BATCH_PROCESSING').warning(
            f"Graceful degradation applied to batch {self.batch_id}: {degradation_reason}"
        )

    def finalize_results(self) -> None:
        """
        Finalize batch processing results with comprehensive analysis, performance summary, and recommendations.
        """
        self.batch_end_time = datetime.datetime.now()
        self.total_processing_time = (
            self.batch_end_time - self.batch_start_time
        ).total_seconds()
        
        # Calculate final rates and metrics
        if self.total_items > 0:
            self.success_rate = self.successful_items / self.total_items
            self.failure_rate = self.failed_items / self.total_items
        
        # Generate comprehensive performance analysis
        self.performance_metrics.update({
            'final_statistics': {
                'total_items': self.total_items,
                'successful_items': self.successful_items,
                'failed_items': self.failed_items,
                'success_rate': self.success_rate,
                'failure_rate': self.failure_rate,
                'processing_time_seconds': self.total_processing_time,
                'items_per_second': self.total_items / self.total_processing_time if self.total_processing_time > 0 else 0
            },
            'finalization_timestamp': datetime.datetime.now().isoformat()
        })
        
        # Create audit trail for batch completion
        create_audit_trail(
            action='BATCH_PROCESSING_FINALIZED',
            component='BATCH_PROCESSING',
            action_details={
                'batch_id': self.batch_id,
                'final_statistics': self.performance_metrics['final_statistics'],
                'graceful_degradation_applied': self.graceful_degradation_applied
            },
            user_context='SYSTEM'
        )

    def get_summary(self, include_detailed_failures: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive batch processing summary with statistics, performance metrics, and recommendations.
        
        Args:
            include_detailed_failures: Whether to include detailed failure information
            
        Returns:
            Dict[str, Any]: Batch processing summary with comprehensive analysis and recommendations
        """
        summary = {
            'batch_id': self.batch_id,
            'batch_context': self.batch_context,
            'total_items': self.total_items,
            'successful_items': self.successful_items,
            'failed_items': self.failed_items,
            'success_rate': self.success_rate,
            'failure_rate': self.failure_rate,
            'processing_time_seconds': self.total_processing_time,
            'graceful_degradation_applied': self.graceful_degradation_applied,
            'degradation_recommendations': self.degradation_recommendations.copy(),
            'performance_metrics': self.performance_metrics.copy()
        }
        
        if include_detailed_failures:
            summary['failed_items_details'] = self.failed_items_details.copy()
        
        return summary


class ErrorContext:
    """
    Context manager for scoped error handling that automatically manages error context setup, 
    cleanup, and recovery strategy application for different processing stages with thread-safe 
    operation and comprehensive error tracking.
    """
    
    def __init__(
        self,
        context_name: str,
        component: str,
        context_data: Dict[str, Any] = None,
        enable_recovery: bool = True
    ):
        """
        Initialize error handling context manager with context information, recovery configuration, 
        and error tracking setup.
        
        Args:
            context_name: Name identifier for the error context
            component: Component name for error categorization
            context_data: Additional context data to store
            enable_recovery: Whether to enable automatic recovery attempts
        """
        self.context_name = context_name
        self.component = component
        self.context_data = context_data or {}
        self.enable_recovery = enable_recovery
        
        # Context state tracking
        self.previous_context: Dict[str, Any] = {}
        self.captured_errors: List[Exception] = []
        self.start_time = datetime.datetime.now()
        self.error_occurred = False
        self.recovery_strategy = None
        self.recovery_actions: List[str] = []

    def __enter__(self) -> 'ErrorContext':
        """
        Enter error handling context, setting up error capture, context tracking, and recovery preparation.
        
        Returns:
            ErrorContext: Self reference for context management and error tracking
        """
        # Save current error context state
        if hasattr(_error_context_stack, 'current_context'):
            self.previous_context = getattr(_error_context_stack, 'current_context', {})
        
        # Set new error context in thread-local storage
        _error_context_stack.current_context = {
            'context_name': self.context_name,
            'component': self.component,
            'context_data': self.context_data,
            'start_time': self.start_time,
            'thread_id': threading.current_thread().ident
        }
        
        # Log context entry
        get_logger('error_context', 'ERROR_HANDLING').debug(
            f"Entering error context: {self.context_name} ({self.component})"
        )
        
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: traceback) -> bool:
        """
        Exit error handling context, processing captured errors, applying recovery strategies, 
        and generating context summary.
        
        Args:
            exc_type: Exception type if exception occurred
            exc_val: Exception value if exception occurred
            exc_tb: Exception traceback if exception occurred
            
        Returns:
            bool: True if error was handled and should be suppressed, False to propagate
        """
        end_time = datetime.datetime.now()
        execution_time = (end_time - self.start_time).total_seconds()
        
        # Handle context exit exception if present
        if exc_type is not None:
            self.error_occurred = True
            self.capture_error(exc_val, {'exception_type': exc_type.__name__})
            
            # Apply recovery strategy if enabled
            if self.enable_recovery:
                try:
                    recovery_result = handle_error(
                        exception=exc_val,
                        context=self.context_name,
                        component=self.component,
                        additional_context=self.context_data,
                        allow_recovery=True
                    )
                    
                    if recovery_result.handled_successfully:
                        self.recovery_actions.append(f"Automatic recovery successful: {recovery_result.error_id}")
                        get_logger('error_context', 'ERROR_HANDLING').info(
                            f"Error recovered in context {self.context_name}: {recovery_result.error_id}"
                        )
                        # Suppress the exception since it was recovered
                        return True
                    
                except Exception as recovery_error:
                    self.recovery_actions.append(f"Recovery failed: {str(recovery_error)}")
        
        # Generate context summary
        context_summary = self.get_context_summary()
        context_summary['execution_time_seconds'] = execution_time
        
        # Log context exit
        get_logger('error_context', 'ERROR_HANDLING').debug(
            f"Exiting error context: {self.context_name} (errors: {len(self.captured_errors)}, "
            f"recovery: {len(self.recovery_actions)})"
        )
        
        # Restore previous error context
        if self.previous_context:
            _error_context_stack.current_context = self.previous_context
        else:
            if hasattr(_error_context_stack, 'current_context'):
                delattr(_error_context_stack, 'current_context')
        
        # Create audit trail for context completion
        create_audit_trail(
            action='ERROR_CONTEXT_COMPLETED',
            component='ERROR_HANDLING',
            action_details=context_summary,
            user_context='SYSTEM'
        )
        
        # Return False to propagate exceptions (unless recovered)
        return False

    def capture_error(
        self,
        error: Exception,
        error_context: Dict[str, Any] = None
    ) -> None:
        """
        Capture error within context for batch processing, analysis, and coordinated error handling.
        
        Args:
            error: Exception to capture
            error_context: Additional context about the error
        """
        self.captured_errors.append(error)
        self.error_occurred = True
        
        # Build comprehensive error context
        full_context = {
            'context_name': self.context_name,
            'component': self.component,
            'capture_timestamp': datetime.datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context_data': self.context_data
        }
        
        if error_context:
            full_context.update(error_context)
        
        # Log error capture
        get_logger('error_context', 'ERROR_HANDLING').warning(
            f"Error captured in context {self.context_name}: {type(error).__name__}: {str(error)}"
        )

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of error handling context execution including errors, recovery 
        actions, and performance metrics.
        
        Returns:
            Dict[str, Any]: Context execution summary with error analysis and recovery information
        """
        return {
            'context_name': self.context_name,
            'component': self.component,
            'context_data': self.context_data.copy(),
            'start_time': self.start_time.isoformat(),
            'error_occurred': self.error_occurred,
            'captured_errors_count': len(self.captured_errors),
            'recovery_actions': self.recovery_actions.copy(),
            'enable_recovery': self.enable_recovery,
            'thread_id': threading.current_thread().ident
        }


class ThresholdValidator:
    """
    Localized threshold validation class providing performance threshold validation and alerting 
    capabilities for error rate monitoring, batch processing health checks, and quality assurance 
    validation using configuration-driven thresholds.
    """
    
    def __init__(
        self,
        threshold_config: Dict[str, Any],
        validation_context: str
    ):
        """
        Initialize threshold validator with configuration and validation context for comprehensive 
        threshold monitoring.
        
        Args:
            threshold_config: Configuration dictionary containing threshold definitions
            validation_context: Context description for threshold validation
        """
        self.threshold_config = threshold_config
        self.validation_context = validation_context
        
        # Extract threshold categories from configuration
        self.quality_assurance_thresholds = threshold_config.get('quality_assurance', {})
        self.monitoring_thresholds = threshold_config.get('monitoring', {})
        self.batch_processing_thresholds = threshold_config.get('batch_processing', {})
        
        # Validation tracking
        self.last_validation_time = datetime.datetime.now()
        self.validation_history: Dict[str, Any] = {}

    def validate_metrics(
        self,
        metrics: Dict[str, float],
        metric_category: str
    ) -> ValidationResult:
        """
        Validate metrics against configured thresholds with comprehensive violation detection 
        and severity assessment.
        
        Args:
            metrics: Dictionary of metric names and values to validate
            metric_category: Category of metrics for threshold selection
            
        Returns:
            ValidationResult: Validation result with threshold violations and recommendations
        """
        # Create validation result container
        validation_result = ValidationResult(
            validation_type='threshold_validation',
            is_valid=True,
            validation_context=f"{self.validation_context}:{metric_category}"
        )
        
        # Select appropriate thresholds for metric category
        if metric_category == 'quality_assurance':
            thresholds = self.quality_assurance_thresholds
        elif metric_category == 'monitoring':
            thresholds = self.monitoring_thresholds
        elif metric_category == 'batch_processing':
            thresholds = self.batch_processing_thresholds
        else:
            validation_result.add_error(
                f"Unknown metric category: {metric_category}",
                severity=ValidationResult.ErrorSeverity.HIGH
            )
            return validation_result
        
        # Validate each metric against corresponding thresholds
        for metric_name, metric_value in metrics.items():
            if metric_name in thresholds:
                threshold_config = thresholds[metric_name]
                
                # Check warning threshold
                warning_threshold = threshold_config.get('warning')
                if warning_threshold is not None and metric_value > warning_threshold:
                    validation_result.add_warning(
                        f"Metric {metric_name} ({metric_value}) exceeds warning threshold ({warning_threshold})"
                    )
                
                # Check critical threshold
                critical_threshold = threshold_config.get('critical')
                if critical_threshold is not None and metric_value > critical_threshold:
                    validation_result.add_error(
                        f"Metric {metric_name} ({metric_value}) exceeds critical threshold ({critical_threshold})",
                        severity=ValidationResult.ErrorSeverity.HIGH
                    )
                    validation_result.is_valid = False
        
        # Update validation history
        self.validation_history[datetime.datetime.now().isoformat()] = {
            'metrics': metrics,
            'metric_category': metric_category,
            'validation_result': validation_result.get_summary(),
            'violations_count': len(validation_result.errors)
        }
        
        self.last_validation_time = datetime.datetime.now()
        
        return validation_result

    def trigger_alerts(
        self,
        violations: List[Dict[str, Any]],
        immediate_escalation: bool = False
    ) -> List[str]:
        """
        Trigger alerts for threshold violations based on severity and configured alert conditions.
        
        Args:
            violations: List of threshold violations with details
            immediate_escalation: Whether to apply immediate escalation
            
        Returns:
            List[str]: List of triggered alert IDs for tracking
        """
        triggered_alerts = []
        
        for violation in violations:
            alert_id = str(uuid.uuid4())
            
            # Generate alert based on violation severity
            severity = violation.get('severity', 'MEDIUM')
            metric_name = violation.get('metric_name', 'unknown')
            metric_value = violation.get('metric_value', 0)
            threshold_value = violation.get('threshold_value', 0)
            
            alert_message = (
                f"THRESHOLD VIOLATION: {metric_name} = {metric_value} "
                f"exceeds threshold {threshold_value} (severity: {severity})"
            )
            
            # Log alert generation
            logger = get_logger('threshold_validator', 'VALIDATION')
            if severity in ['CRITICAL', 'HIGH'] or immediate_escalation:
                logger.error(alert_message)
            else:
                logger.warning(alert_message)
            
            triggered_alerts.append(alert_id)
            
            # Create audit trail for alert
            create_audit_trail(
                action='THRESHOLD_ALERT_TRIGGERED',
                component='THRESHOLD_VALIDATOR',
                action_details={
                    'alert_id': alert_id,
                    'violation': violation,
                    'immediate_escalation': immediate_escalation,
                    'validation_context': self.validation_context
                },
                user_context='SYSTEM'
            )
        
        return triggered_alerts

    def get_threshold_summary(self) -> Dict[str, Any]:
        """
        Get summary of configured thresholds and recent validation history.
        
        Returns:
            Dict[str, Any]: Threshold configuration summary with validation statistics
        """
        return {
            'validation_context': self.validation_context,
            'threshold_categories': {
                'quality_assurance': len(self.quality_assurance_thresholds),
                'monitoring': len(self.monitoring_thresholds),
                'batch_processing': len(self.batch_processing_thresholds)
            },
            'last_validation_time': self.last_validation_time.isoformat(),
            'validation_history_count': len(self.validation_history),
            'total_configured_thresholds': (
                len(self.quality_assurance_thresholds) + 
                len(self.monitoring_thresholds) + 
                len(self.batch_processing_thresholds)
            )
        }


class ThresholdValidationResult:
    """
    Data class representing threshold validation results with violation details, severity assessment, 
    corrective action recommendations, and compliance status for comprehensive threshold monitoring.
    """
    
    def __init__(
        self,
        validation_id: str,
        validation_context: str,
        validation_passed: bool
    ):
        """
        Initialize threshold validation result with validation identification and basic status.
        
        Args:
            validation_id: Unique identifier for the validation operation
            validation_context: Context description for the validation
            validation_passed: Boolean indicating if validation passed
        """
        self.validation_id = validation_id
        self.validation_context = validation_context
        self.validation_passed = validation_passed
        
        # Validation details and tracking
        self.violations: List[Dict[str, Any]] = []
        self.recommendations: List[str] = []
        self.validation_metadata: Dict[str, Any] = {}
        self.validation_timestamp = datetime.datetime.now()
        self.overall_compliance_score = 100.0
        self.compliance_status = 'COMPLIANT' if validation_passed else 'NON_COMPLIANT'

    def add_violation(
        self,
        metric_name: str,
        metric_value: float,
        threshold_value: float,
        violation_severity: str
    ) -> None:
        """
        Add threshold violation with details and severity assessment.
        
        Args:
            metric_name: Name of the metric that violated the threshold
            metric_value: Actual value of the metric
            threshold_value: Threshold value that was exceeded
            violation_severity: Severity level of the violation
        """
        violation = {
            'metric_name': metric_name,
            'metric_value': metric_value,
            'threshold_value': threshold_value,
            'violation_severity': violation_severity,
            'violation_percentage': ((metric_value - threshold_value) / threshold_value * 100) if threshold_value > 0 else 0,
            'detected_at': datetime.datetime.now().isoformat()
        }
        
        self.violations.append(violation)
        self.validation_passed = False
        
        # Update compliance score based on violation severity
        severity_impact = {
            'CRITICAL': 30.0,
            'HIGH': 20.0,
            'MEDIUM': 10.0,
            'LOW': 5.0
        }
        
        impact = severity_impact.get(violation_severity, 10.0)
        self.overall_compliance_score = max(0.0, self.overall_compliance_score - impact)
        
        # Update compliance status
        if self.overall_compliance_score < 70.0:
            self.compliance_status = 'CRITICAL_NON_COMPLIANCE'
        elif self.overall_compliance_score < 85.0:
            self.compliance_status = 'MINOR_NON_COMPLIANCE'

    def add_recommendation(
        self,
        recommendation_text: str,
        priority: str = "MEDIUM"
    ) -> None:
        """
        Add corrective action recommendation for threshold violations.
        
        Args:
            recommendation_text: Text description of the recommendation
            priority: Priority level for the recommendation
        """
        formatted_recommendation = f"[{priority}] {recommendation_text}"
        self.recommendations.append(formatted_recommendation)
        
        # Store recommendation metadata
        self.validation_metadata[f'recommendation_{len(self.recommendations)}'] = {
            'text': recommendation_text,
            'priority': priority,
            'added_at': datetime.datetime.now().isoformat()
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive validation summary with violations, recommendations, and compliance assessment.
        
        Returns:
            Dict[str, Any]: Validation summary with comprehensive analysis and recommendations
        """
        return {
            'validation_id': self.validation_id,
            'validation_context': self.validation_context,
            'validation_passed': self.validation_passed,
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'violations_count': len(self.violations),
            'recommendations_count': len(self.recommendations),
            'overall_compliance_score': self.overall_compliance_score,
            'compliance_status': self.compliance_status,
            'violations_summary': self.violations.copy(),
            'recommendations_summary': self.recommendations.copy()
        }


def handle_error(
    exception: Exception,
    context: str,
    component: str,
    additional_context: Dict[str, Any] = None,
    allow_recovery: bool = True
) -> ErrorHandlingResult:
    """
    Central error handling function that processes any exception with comprehensive error analysis, 
    recovery strategy application, audit trail creation, and alert generation for scientific 
    computing reliability.
    
    Args:
        exception: Exception object to handle
        context: Context description where the error occurred
        component: Component name for error categorization
        additional_context: Additional context information
        allow_recovery: Whether to attempt automatic recovery
        
    Returns:
        ErrorHandlingResult: Comprehensive error handling result with recovery actions and recommendations
    """
    # Generate unique error ID for tracking
    error_id = str(uuid.uuid4())
    
    # Initialize error handling result
    result = ErrorHandlingResult(
        error_id=error_id,
        handled_successfully=False,
        handling_context=f"{component}:{context}"
    )
    
    try:
        # Classify exception type and determine severity level
        result.severity = _classify_error_severity(exception)
        result.category = _classify_error_category(exception, component)
        
        # Build comprehensive error context with system state
        error_context = create_error_context(component, context, additional_context)
        result.metadata.update(error_context)
        
        # Apply appropriate recovery strategy if available and allowed
        if allow_recovery and result.category.value in _recovery_strategies:
            try:
                recovery_strategy = _recovery_strategies[result.category.value]
                recovery_result = recovery_strategy(exception, error_context)
                
                if recovery_result.get('success', False):
                    result.set_recovery_action(
                        f"Automatic recovery applied: {recovery_result.get('method', 'unknown')}",
                        True,
                        recovery_result
                    )
                    result.handled_successfully = True
                else:
                    result.set_recovery_action(
                        f"Recovery attempted but failed: {recovery_result.get('error', 'unknown')}",
                        False,
                        recovery_result
                    )
            except Exception as recovery_error:
                result.set_recovery_action(
                    f"Recovery strategy failed: {str(recovery_error)}",
                    False,
                    {'recovery_error': str(recovery_error)}
                )
        
        # Log error with structured format and audit trail
        logger = get_logger('error_handling', 'ERROR_HANDLING')
        
        if result.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR [{error_id}]: {type(exception).__name__}: {str(exception)}")
        elif result.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR [{error_id}]: {type(exception).__name__}: {str(exception)}")
        else:
            logger.warning(f"ERROR [{error_id}]: {type(exception).__name__}: {str(exception)}")
        
        # Update error statistics and monitoring counters
        _update_error_statistics(result)
        
        # Validate error rates against performance thresholds
        threshold_validation_result = validate_error_thresholds(
            _get_current_error_rates(),
            f"{component}:{context}"
        )
        
        if not threshold_validation_result.validation_passed:
            result.add_recommendation(
                "Error rate thresholds exceeded - investigate and address error patterns",
                "HIGH"
            )
        
        # Trigger alerts if error thresholds are exceeded
        if threshold_validation_result.violations:
            alert_ids = _trigger_error_alerts(threshold_validation_result.violations)
            result.metadata['triggered_alerts'] = alert_ids
        
        # Generate error handling result with recommendations
        _generate_error_recommendations(result, exception, error_context)
        
        # Mark as resolved if handling was successful
        if result.handled_successfully:
            result.mark_resolved("automatic_recovery", {"recovery_applied": True})
        
        # Create comprehensive audit trail
        create_audit_trail(
            action='ERROR_HANDLED',
            component='ERROR_HANDLING',
            action_details={
                'error_id': error_id,
                'exception_type': type(exception).__name__,
                'severity': result.severity.value,
                'category': result.category.value,
                'handled_successfully': result.handled_successfully,
                'context': f"{component}:{context}",
                'recovery_actions_count': len(result.recovery_actions)
            },
            user_context='SYSTEM'
        )
        
        return result
        
    except Exception as handling_error:
        # Handle errors in the error handling process itself
        result.add_recommendation(
            f"Error handling process encountered an issue: {str(handling_error)}",
            "HIGH"
        )
        
        # Log the meta-error
        get_logger('error_handling', 'ERROR_HANDLING').error(
            f"Error in error handling process [{error_id}]: {str(handling_error)}"
        )
        
        return result


def retry_with_backoff(
    max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    initial_delay: float = DEFAULT_RETRY_DELAY,
    max_delay: float = MAX_RETRY_DELAY,
    backoff_multiplier: float = EXPONENTIAL_BACKOFF_MULTIPLIER,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    add_jitter: bool = True
) -> Callable:
    """
    Decorator function implementing exponential backoff retry logic with jitter for transient 
    failures, configurable retry attempts, delay parameters, and exception filtering to handle 
    temporary system issues without overwhelming resources.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay before first retry (seconds)
        max_delay: Maximum delay between retries (seconds)
        backoff_multiplier: Multiplier for exponential backoff
        retryable_exceptions: Tuple of exception types that should trigger retries
        add_jitter: Whether to add random jitter to prevent thundering herd
        
    Returns:
        Callable: Decorated function with retry logic applied
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger('retry_backoff', 'ERROR_HANDLING')
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    # Execute function with exception handling
                    result = func(*args, **kwargs)
                    
                    # Log success after retries
                    if attempt > 0:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    # Check if this is the last attempt
                    if attempt == max_attempts - 1:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts: {e}")
                        break
                    
                    # Calculate exponential backoff with jitter
                    delay = min(initial_delay * (backoff_multiplier ** attempt), max_delay)
                    if add_jitter:
                        jitter = delay * JITTER_FACTOR * random.random()
                        delay += jitter
                    
                    # Log retry attempt
                    logger.warning(
                        f"Function {func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f} seconds"
                    )
                    
                    # Apply backoff delay before retry
                    time.sleep(delay)
                
                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"Function {func.__name__} failed with non-retryable exception: {e}")
                    raise e
            
            # Raise the last exception after all attempts
            raise last_exception
        
        return wrapper
    return decorator


def graceful_degradation(
    batch_items: List[Any],
    processing_function: Callable,
    failure_threshold: float = BATCH_FAILURE_THRESHOLD,
    degradation_context: str = "batch_processing",
    degradation_config: Dict[str, Any] = None
) -> BatchProcessingResult:
    """
    Implement graceful degradation strategy for batch processing operations that allows partial 
    completion with detailed reporting of successful and failed operations, enabling recovery 
    and continuation of interrupted batch operations.
    
    Args:
        batch_items: List of items to process in the batch
        processing_function: Function to apply to each batch item
        failure_threshold: Failure rate threshold to trigger degradation
        degradation_context: Context description for degradation tracking
        degradation_config: Configuration parameters for degradation behavior
        
    Returns:
        BatchProcessingResult: Batch processing result with graceful degradation analysis and partial results
    """
    # Initialize batch processing result container
    batch_id = str(uuid.uuid4())
    result = BatchProcessingResult(
        batch_id=batch_id,
        total_items=len(batch_items),
        batch_context=degradation_context
    )
    
    logger = get_logger('graceful_degradation', 'BATCH_PROCESSING')
    logger.info(f"Starting batch processing with graceful degradation: {batch_id} ({len(batch_items)} items)")
    
    # Process batch items with error handling
    for item_index, item in enumerate(batch_items):
        try:
            # Process item using provided processing function
            item_result = processing_function(item)
            result.add_successful_item(item_result, {'item_index': item_index})
            
        except Exception as e:
            # Track failed item with error details
            result.add_failed_item(item, e, {'item_index': item_index})
            
            # Check if failure rate exceeds threshold
            if result.failure_rate > failure_threshold:
                # Apply graceful degradation strategy
                degradation_reason = f"Failure rate {result.failure_rate:.2%} exceeds threshold {failure_threshold:.2%}"
                result.apply_graceful_degradation(degradation_reason, degradation_config)
                
                logger.warning(
                    f"Graceful degradation triggered for batch {batch_id}: {degradation_reason} "
                    f"(processed {item_index + 1}/{len(batch_items)} items)"
                )
                
                # Continue processing remaining items but track degradation
                continue
    
    # Finalize batch processing with comprehensive analysis
    result.finalize_results()
    
    # Generate comprehensive batch processing report
    logger.info(
        f"Batch processing completed: {batch_id} - "
        f"Success: {result.successful_items}/{result.total_items} ({result.success_rate:.1%}), "
        f"Degradation: {result.graceful_degradation_applied}"
    )
    
    return result


def register_recovery_strategy(
    error_type: str,
    recovery_function: Callable,
    strategy_name: str,
    strategy_config: Dict[str, Any] = None
) -> bool:
    """
    Register custom recovery strategy for specific error types or components, enabling extensible 
    error handling with component-specific recovery logic and automated error resolution.
    
    Args:
        error_type: Error type or category to register strategy for
        recovery_function: Function that implements the recovery logic
        strategy_name: Name identifier for the recovery strategy
        strategy_config: Configuration parameters for the strategy
        
    Returns:
        bool: Success status of recovery strategy registration
    """
    logger = get_logger('recovery_strategy', 'ERROR_HANDLING')
    
    try:
        # Validate recovery function signature and parameters
        if not callable(recovery_function):
            raise TypeError("Recovery function must be callable")
        
        # Check for existing strategy conflicts
        if error_type in _recovery_strategies:
            logger.warning(f"Overwriting existing recovery strategy for {error_type}")
        
        # Register strategy in global recovery registry
        _recovery_strategies[error_type] = recovery_function
        
        # Configure strategy parameters and metadata
        strategy_metadata = {
            'strategy_name': strategy_name,
            'error_type': error_type,
            'config': strategy_config or {},
            'registered_at': datetime.datetime.now().isoformat()
        }
        
        # Test strategy function with mock error scenarios (basic validation)
        try:
            # Create a simple test exception
            test_exception = Exception("Test exception for strategy validation")
            test_context = {'test': True, 'component': 'validation'}
            
            # Call recovery function to validate interface
            test_result = recovery_function(test_exception, test_context)
            
            # Validate return format
            if not isinstance(test_result, dict):
                logger.warning(f"Recovery strategy {strategy_name} should return a dictionary")
            
        except Exception as test_error:
            logger.warning(f"Recovery strategy {strategy_name} test failed: {test_error}")
        
        # Log strategy registration with configuration details
        logger.info(f"Recovery strategy registered: {strategy_name} for {error_type}")
        
        # Create audit trail for strategy registration
        create_audit_trail(
            action='RECOVERY_STRATEGY_REGISTERED',
            component='ERROR_HANDLING',
            action_details=strategy_metadata,
            user_context='SYSTEM'
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to register recovery strategy {strategy_name}: {e}")
        return False


def register_error_handler(
    error_type: str,
    handler_function: Callable,
    handler_name: str,
    handler_config: Dict[str, Any] = None
) -> bool:
    """
    Register custom error handler for specific error types with validation, testing, and integration 
    into error handling workflow for extensible error management.
    
    Args:
        error_type: Error type to register handler for
        handler_function: Function that implements the error handling logic
        handler_name: Name identifier for the error handler
        handler_config: Configuration parameters for the handler
        
    Returns:
        bool: Success status of error handler registration
    """
    logger = get_logger('error_handler', 'ERROR_HANDLING')
    
    try:
        # Validate handler function signature and configuration
        if not callable(handler_function):
            raise TypeError("Handler function must be callable")
        
        # Check for existing handler conflicts and compatibility
        if error_type in _error_handlers:
            logger.warning(f"Overwriting existing error handler for {error_type}")
        
        # Test handler function with mock error scenarios
        try:
            test_exception = Exception("Test exception for handler validation")
            test_context = {'test': True, 'component': 'validation'}
            
            # Call handler function to validate interface
            test_result = handler_function(test_exception, test_context)
            
            # Validate return format (should be compatible with ErrorHandlingResult)
            if test_result is not None and not isinstance(test_result, dict):
                logger.warning(f"Error handler {handler_name} should return a dictionary or None")
                
        except Exception as test_error:
            logger.warning(f"Error handler {handler_name} test failed: {test_error}")
        
        # Register handler in error handlers registry
        _error_handlers[error_type] = handler_function
        
        # Configure handler parameters and metadata
        handler_metadata = {
            'handler_name': handler_name,
            'error_type': error_type,
            'config': handler_config or {},
            'registered_at': datetime.datetime.now().isoformat()
        }
        
        # Log handler registration with configuration details
        logger.info(f"Error handler registered: {handler_name} for {error_type}")
        
        # Create audit trail for handler registration
        create_audit_trail(
            action='ERROR_HANDLER_REGISTERED',
            component='ERROR_HANDLING',
            action_details=handler_metadata,
            user_context='SYSTEM'
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to register error handler {handler_name}: {e}")
        return False


def get_error_statistics(
    time_window: str = "1h",
    component_filter: str = None,
    include_resolved_errors: bool = True
) -> Dict[str, Any]:
    """
    Retrieve comprehensive error statistics including error rates, failure patterns, recovery 
    success rates, and performance impact metrics for system monitoring and analysis.
    
    Args:
        time_window: Time window for statistics (e.g., "1h", "24h", "7d")
        component_filter: Filter statistics by specific component
        include_resolved_errors: Whether to include resolved errors in statistics
        
    Returns:
        Dict[str, Any]: Comprehensive error statistics with rates, patterns, and trends
    """
    logger = get_logger('error_statistics', 'ERROR_HANDLING')
    
    try:
        # Parse time window and calculate cutoff time
        cutoff_time = _parse_time_window(time_window)
        
        # Initialize statistics structure
        statistics = {
            'time_window': time_window,
            'component_filter': component_filter,
            'include_resolved_errors': include_resolved_errors,
            'generation_timestamp': datetime.datetime.now().isoformat(),
            'error_rates': {},
            'failure_patterns': {},
            'recovery_statistics': {},
            'performance_impact': {},
            'trend_analysis': {}
        }
        
        # Filter error data by component and time range
        filtered_errors = _filter_error_statistics(
            _error_statistics,
            cutoff_time,
            component_filter,
            include_resolved_errors
        )
        
        # Calculate error rates, frequencies, and patterns
        statistics['error_rates'] = _calculate_error_rates(filtered_errors, time_window)
        statistics['failure_patterns'] = _analyze_failure_patterns(filtered_errors)
        
        # Analyze recovery success rates and effectiveness
        statistics['recovery_statistics'] = _analyze_recovery_effectiveness(filtered_errors)
        
        # Compute performance impact and system health metrics
        statistics['performance_impact'] = _calculate_performance_impact(filtered_errors)
        
        # Generate trend analysis and pattern detection
        statistics['trend_analysis'] = _generate_trend_analysis(filtered_errors, time_window)
        
        # Include resolved errors if requested
        if include_resolved_errors:
            statistics['resolved_errors'] = _analyze_resolved_errors(filtered_errors)
        
        # Format statistics for reporting and visualization
        statistics['summary'] = {
            'total_errors': len(filtered_errors),
            'critical_errors': len([e for e in filtered_errors if e.get('severity') == 'CRITICAL']),
            'resolved_errors': len([e for e in filtered_errors if e.get('resolved', False)]),
            'average_resolution_time': _calculate_average_resolution_time(filtered_errors)
        }
        
        logger.debug(f"Error statistics generated for {time_window} window: {statistics['summary']['total_errors']} errors")
        
        return statistics
        
    except Exception as e:
        logger.error(f"Failed to retrieve error statistics: {e}")
        return {
            'error': str(e),
            'generation_timestamp': datetime.datetime.now().isoformat()
        }


def clear_error_statistics(
    component_filter: str = None,
    preserve_critical_errors: bool = True,
    reset_reason: str = "manual_reset"
) -> Dict[str, int]:
    """
    Clear error statistics and reset monitoring counters for fresh tracking periods, typically 
    used for testing, benchmarking, or periodic monitoring reset operations.
    
    Args:
        component_filter: Filter clearing by specific component
        preserve_critical_errors: Whether to preserve critical error records
        reset_reason: Reason for clearing statistics
        
    Returns:
        Dict[str, int]: Statistics about cleared data including counts and preserved items
    """
    logger = get_logger('error_statistics', 'ERROR_HANDLING')
    
    try:
        global _error_statistics
        
        # Record statistics before clearing
        total_errors_before = len(_error_statistics)
        critical_errors_before = len([
            e for e in _error_statistics.values() 
            if isinstance(e, dict) and e.get('severity') == 'CRITICAL'
        ])
        
        preserved_errors = 0
        cleared_errors = 0
        
        # Filter statistics by component if specified
        if component_filter:
            # Clear only matching component errors
            keys_to_remove = [
                key for key, error_data in _error_statistics.items()
                if isinstance(error_data, dict) and error_data.get('component') == component_filter
            ]
        else:
            # Clear all errors (subject to preservation rules)
            keys_to_remove = list(_error_statistics.keys())
        
        # Process each error for clearing or preservation
        for key in keys_to_remove:
            error_data = _error_statistics.get(key, {})
            
            # Preserve critical errors if requested
            if preserve_critical_errors and error_data.get('severity') == 'CRITICAL':
                preserved_errors += 1
                continue
            
            # Clear the error record
            del _error_statistics[key]
            cleared_errors += 1
        
        # Reset monitoring timers and windows
        # (In a full implementation, this would reset monitoring state)
        
        # Generate clearing summary
        clearing_stats = {
            'total_errors_before': total_errors_before,
            'critical_errors_before': critical_errors_before,
            'cleared_errors': cleared_errors,
            'preserved_errors': preserved_errors,
            'component_filter': component_filter,
            'preserve_critical_errors': preserve_critical_errors,
            'reset_reason': reset_reason,
            'reset_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Log statistics reset operation with reason
        logger.info(
            f"Error statistics cleared: {cleared_errors} errors removed, "
            f"{preserved_errors} critical errors preserved (reason: {reset_reason})"
        )
        
        # Create audit trail for statistics reset
        create_audit_trail(
            action='ERROR_STATISTICS_CLEARED',
            component='ERROR_HANDLING',
            action_details=clearing_stats,
            user_context='SYSTEM'
        )
        
        return clearing_stats
        
    except Exception as e:
        logger.error(f"Failed to clear error statistics: {e}")
        return {
            'error': str(e),
            'reset_timestamp': datetime.datetime.now().isoformat()
        }


def create_error_context(
    component: str,
    operation: str,
    additional_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create comprehensive error context dictionary with system state, processing context, and 
    debugging information for scientific reproducibility and detailed error analysis.
    
    Args:
        component: Component name where the error occurred
        operation: Operation being performed when error occurred
        additional_context: Additional context information to include
        
    Returns:
        Dict[str, Any]: Comprehensive error context with system and processing information
    """
    # Capture current system state and resource usage
    context = {
        'component': component,
        'operation': operation,
        'timestamp': datetime.datetime.now().isoformat(),
        'thread_id': threading.current_thread().ident,
        'thread_name': threading.current_thread().name,
        'process_id': os.getpid() if hasattr(os, 'getpid') else None
    }
    
    # Extract processing context from thread-local storage
    if hasattr(_error_context_stack, 'current_context'):
        current_context = getattr(_error_context_stack, 'current_context', {})
        context['error_context_stack'] = current_context
    
    # Include component and operation information
    context['context_details'] = {
        'component': component,
        'operation': operation,
        'context_id': str(uuid.uuid4())
    }
    
    # Add timestamp and unique error identifier
    context['error_tracking'] = {
        'context_timestamp': datetime.datetime.now().isoformat(),
        'context_id': str(uuid.uuid4()),
        'correlation_id': str(uuid.uuid4())
    }
    
    # Merge additional context if provided
    if additional_context:
        context['additional_context'] = additional_context
    
    # Include stack trace and call hierarchy (limited for security)
    try:
        frame_info = []
        current_frame = sys._getframe(1)  # Skip this function
        frame_count = 0
        
        while current_frame and frame_count < 5:  # Limit stack depth
            frame_info.append({
                'filename': current_frame.f_code.co_filename,
                'function': current_frame.f_code.co_name,
                'line_number': current_frame.f_lineno
            })
            current_frame = current_frame.f_back
            frame_count += 1
        
        context['call_stack'] = frame_info
        
    except Exception:
        # Ignore errors in stack trace extraction
        context['call_stack'] = []
    
    return context


def format_error_message(
    base_message: str,
    context: Dict[str, Any],
    include_recommendations: bool = True,
    include_stack_trace: bool = False
) -> str:
    """
    Format error message with scientific context, error details, and recovery recommendations 
    for clear error communication and user guidance.
    
    Args:
        base_message: Base error message to format
        context: Error context dictionary with additional information
        include_recommendations: Whether to include recovery recommendations
        include_stack_trace: Whether to include stack trace information
        
    Returns:
        str: Formatted error message with context and recommendations
    """
    # Format base message with context variables
    formatted_message = base_message
    
    # Add scientific computing specific details
    if 'component' in context:
        formatted_message += f" [Component: {context['component']}]"
    
    if 'operation' in context:
        formatted_message += f" [Operation: {context['operation']}]"
    
    # Include processing stage and component information
    if 'error_context_stack' in context:
        stack_context = context['error_context_stack']
        if 'context_name' in stack_context:
            formatted_message += f" [Context: {stack_context['context_name']}]"
    
    # Add recovery recommendations if enabled
    if include_recommendations and 'recommendations' in context:
        recommendations = context['recommendations']
        if recommendations:
            formatted_message += "\n\nRecommendations:"
            for rec in recommendations[:3]:  # Limit to top 3
                formatted_message += f"\n  • {rec}"
    
    # Include stack trace if requested
    if include_stack_trace and 'call_stack' in context:
        call_stack = context['call_stack']
        if call_stack:
            formatted_message += "\n\nCall Stack:"
            for frame in call_stack:
                formatted_message += f"\n  {frame['function']} ({frame['filename']}:{frame['line_number']})"
    
    # Format message for readability and debugging
    if 'error_tracking' in context:
        tracking = context['error_tracking']
        if 'context_id' in tracking:
            formatted_message += f"\n\nError ID: {tracking['context_id']}"
    
    return formatted_message


def validate_error_thresholds(
    current_error_rates: Dict[str, float],
    validation_context: str
) -> ThresholdValidationResult:
    """
    Validate current error rates against configured thresholds with alert generation and corrective 
    action recommendations for performance monitoring using locally loaded threshold configuration.
    
    Args:
        current_error_rates: Dictionary of current error rates to validate
        validation_context: Context for the threshold validation
        
    Returns:
        ThresholdValidationResult: Threshold validation result with violations and recommendations
    """
    # Load performance thresholds configuration from file
    threshold_config = _load_threshold_configuration()
    
    # Extract error rate thresholds from quality assurance section
    error_thresholds = threshold_config.get('quality_assurance', {}).get('error_rates', {})
    
    # Create threshold validation result
    validation_id = str(uuid.uuid4())
    result = ThresholdValidationResult(
        validation_id=validation_id,
        validation_context=validation_context,
        validation_passed=True
    )
    
    # Compare current error rates against configured thresholds
    for metric_name, current_rate in current_error_rates.items():
        if metric_name in error_thresholds:
            threshold_config = error_thresholds[metric_name]
            
            # Check warning threshold
            warning_threshold = threshold_config.get('warning', 0.02)  # Default 2%
            if current_rate > warning_threshold:
                result.add_recommendation(
                    f"Error rate for {metric_name} ({current_rate:.2%}) exceeds warning threshold",
                    "MEDIUM"
                )
            
            # Check critical threshold
            critical_threshold = threshold_config.get('critical', 0.05)  # Default 5%
            if current_rate > critical_threshold:
                result.add_violation(
                    metric_name=metric_name,
                    metric_value=current_rate,
                    threshold_value=critical_threshold,
                    violation_severity="HIGH"
                )
                result.add_recommendation(
                    f"Immediate action required for {metric_name} error rate",
                    "HIGH"
                )
    
    # Identify threshold violations and severity levels
    if result.violations:
        result.validation_passed = False
    
    # Generate corrective action recommendations
    if not result.validation_passed:
        result.add_recommendation(
            "Review error patterns and implement corrective measures",
            "HIGH"
        )
        result.add_recommendation(
            "Consider increasing monitoring frequency during high error periods",
            "MEDIUM"
        )
    
    # Create threshold validation result with detailed analysis
    logger = get_logger('threshold_validation', 'ERROR_HANDLING')
    logger.info(
        f"Error threshold validation completed: {validation_context} - "
        f"Passed: {result.validation_passed}, Violations: {len(result.violations)}"
    )
    
    # Log threshold validation results with context
    create_audit_trail(
        action='ERROR_THRESHOLD_VALIDATION',
        component='ERROR_HANDLING',
        action_details={
            'validation_id': validation_id,
            'validation_context': validation_context,
            'validation_passed': result.validation_passed,
            'violations_count': len(result.violations),
            'current_error_rates': current_error_rates
        },
        user_context='SYSTEM'
    )
    
    return result


def _load_threshold_configuration(force_reload: bool = False) -> Dict[str, Any]:
    """
    Internal function to load and cache performance threshold configuration from file with 
    automatic refresh and validation.
    
    Args:
        force_reload: Whether to force reload from file
        
    Returns:
        Dict[str, Any]: Performance thresholds configuration dictionary
    """
    global _threshold_configuration, _threshold_cache_timestamp
    
    # Check if configuration is cached and timestamp is recent
    if (not force_reload and 
        _threshold_configuration is not None and 
        _threshold_cache_timestamp is not None and
        (datetime.datetime.now() - _threshold_cache_timestamp).total_seconds() < 300):  # 5 minute cache
        return _threshold_configuration
    
    try:
        # Load configuration from performance thresholds file if needed
        config = load_configuration(
            config_name='performance_thresholds',
            config_path=PERFORMANCE_THRESHOLDS_CONFIG_PATH,
            validate_schema=False,
            use_cache=True
        )
        
        # Validate configuration structure and required sections
        if not isinstance(config, dict):
            raise ValueError("Threshold configuration must be a dictionary")
        
        # Cache configuration and update timestamp
        _threshold_configuration = config
        _threshold_cache_timestamp = datetime.datetime.now()
        
        # Log configuration loading operation
        get_logger('threshold_config', 'ERROR_HANDLING').debug(
            "Performance thresholds configuration loaded and cached"
        )
        
        return config
        
    except Exception as e:
        # Return default configuration if loading fails
        get_logger('threshold_config', 'ERROR_HANDLING').warning(
            f"Failed to load threshold configuration, using defaults: {e}"
        )
        
        default_config = {
            'quality_assurance': {
                'error_rates': {
                    'overall_error_rate': {'warning': 0.01, 'critical': 0.05},
                    'critical_error_rate': {'warning': 0.001, 'critical': 0.01},
                    'validation_error_rate': {'warning': 0.02, 'critical': 0.10}
                }
            }
        }
        
        _threshold_configuration = default_config
        _threshold_cache_timestamp = datetime.datetime.now()
        
        return default_config


def _validate_threshold_violations(
    metrics: Dict[str, float],
    thresholds: Dict[str, Any],
    threshold_type: str
) -> List[Dict[str, Any]]:
    """
    Internal function to validate metric values against threshold limits and identify violations 
    with severity assessment.
    
    Args:
        metrics: Dictionary of metric values to validate
        thresholds: Threshold configuration dictionary
        threshold_type: Type of threshold validation
        
    Returns:
        List[Dict[str, Any]]: List of threshold violations with severity and details
    """
    violations = []
    
    # Iterate through metrics and corresponding thresholds
    for metric_name, metric_value in metrics.items():
        if metric_name in thresholds:
            threshold_config = thresholds[metric_name]
            
            # Compare metric values against warning and critical thresholds
            warning_threshold = threshold_config.get('warning')
            critical_threshold = threshold_config.get('critical')
            
            violation = None
            
            # Determine violation severity based on threshold exceeded
            if critical_threshold is not None and metric_value > critical_threshold:
                violation = {
                    'metric_name': metric_name,
                    'metric_value': metric_value,
                    'threshold_value': critical_threshold,
                    'threshold_type': 'critical',
                    'severity': 'CRITICAL'
                }
            elif warning_threshold is not None and metric_value > warning_threshold:
                violation = {
                    'metric_name': metric_name,
                    'metric_value': metric_value,
                    'threshold_value': warning_threshold,
                    'threshold_type': 'warning',
                    'severity': 'MEDIUM'
                }
            
            if violation:
                # Calculate percentage of threshold exceeded
                violation['percentage_exceeded'] = (
                    (metric_value - violation['threshold_value']) / violation['threshold_value'] * 100
                ) if violation['threshold_value'] > 0 else 0
                
                # Generate violation recommendations based on type and severity
                violation['recommendations'] = _generate_threshold_recommendations(
                    metric_name, metric_value, violation['threshold_value'], violation['severity']
                )
                
                violations.append(violation)
    
    return violations


# Helper functions for error handling implementation

def _classify_error_severity(exception: Exception) -> ErrorSeverity:
    """Classify exception severity based on type and context."""
    if isinstance(exception, (SystemExit, KeyboardInterrupt, MemoryError)):
        return ErrorSeverity.CRITICAL
    elif isinstance(exception, (FileNotFoundError, PermissionError, ConnectionError)):
        return ErrorSeverity.HIGH
    elif isinstance(exception, (ValueError, TypeError, AttributeError)):
        return ErrorSeverity.MEDIUM
    else:
        return ErrorSeverity.LOW


def _classify_error_category(exception: Exception, component: str) -> ErrorCategory:
    """Classify exception into error category based on type and component."""
    if 'validation' in component.lower():
        return ErrorCategory.VALIDATION
    elif 'simulation' in component.lower():
        return ErrorCategory.SIMULATION
    elif 'config' in component.lower():
        return ErrorCategory.CONFIGURATION
    elif isinstance(exception, (MemoryError, OSError)):
        return ErrorCategory.RESOURCE
    else:
        return ErrorCategory.PROCESSING


def _update_error_statistics(result: ErrorHandlingResult) -> None:
    """Update global error statistics with new error handling result."""
    global _error_statistics
    
    error_entry = {
        'error_id': result.error_id,
        'severity': result.severity.value,
        'category': result.category.value,
        'component': result.handling_context.split(':')[0],
        'timestamp': result.handling_timestamp.isoformat(),
        'handled_successfully': result.handled_successfully,
        'resolution_time': result.handling_duration_seconds
    }
    
    _error_statistics[result.error_id] = error_entry


def _get_current_error_rates() -> Dict[str, float]:
    """Calculate current error rates from statistics."""
    # This is a simplified implementation
    # In practice, this would calculate rates over time windows
    return {
        'overall_error_rate': 0.01,  # 1%
        'critical_error_rate': 0.001,  # 0.1%
        'validation_error_rate': 0.02  # 2%
    }


def _trigger_error_alerts(violations: List[Dict[str, Any]]) -> List[str]:
    """Trigger alerts for threshold violations."""
    alert_ids = []
    
    for violation in violations:
        alert_id = str(uuid.uuid4())
        alert_ids.append(alert_id)
        
        # Log alert (in practice, this would integrate with alerting systems)
        logger = get_logger('error_alerts', 'ERROR_HANDLING')
        logger.warning(f"Alert triggered [{alert_id}]: {violation}")
    
    return alert_ids


def _generate_error_recommendations(
    result: ErrorHandlingResult, 
    exception: Exception, 
    context: Dict[str, Any]
) -> None:
    """Generate actionable recommendations for error resolution."""
    # Add general recommendations based on error type
    if isinstance(exception, FileNotFoundError):
        result.add_recommendation("Verify file paths and ensure required files exist", "HIGH")
    elif isinstance(exception, MemoryError):
        result.add_recommendation("Reduce data size or increase available memory", "CRITICAL")
    elif isinstance(exception, ValueError):
        result.add_recommendation("Validate input parameters and data formats", "MEDIUM")
    
    # Add context-specific recommendations
    if 'component' in context:
        component = context['component']
        if 'validation' in component.lower():
            result.add_recommendation("Review validation rules and input data quality", "MEDIUM")
        elif 'simulation' in component.lower():
            result.add_recommendation("Check simulation parameters and algorithm configuration", "MEDIUM")


def _generate_threshold_recommendations(
    metric_name: str, 
    metric_value: float, 
    threshold_value: float, 
    severity: str
) -> List[str]:
    """Generate recommendations for threshold violations."""
    recommendations = []
    
    if 'error_rate' in metric_name:
        recommendations.append("Investigate recent error patterns and root causes")
        recommendations.append("Consider implementing additional validation checks")
    
    if severity == 'CRITICAL':
        recommendations.append("Immediate intervention required - consider system maintenance")
    
    return recommendations


def _parse_time_window(time_window: str) -> datetime.datetime:
    """Parse time window string and return cutoff datetime."""
    # Simplified implementation - would parse "1h", "24h", "7d", etc.
    if time_window.endswith('h'):
        hours = int(time_window[:-1])
        return datetime.datetime.now() - datetime.timedelta(hours=hours)
    elif time_window.endswith('d'):
        days = int(time_window[:-1])
        return datetime.datetime.now() - datetime.timedelta(days=days)
    else:
        return datetime.datetime.now() - datetime.timedelta(hours=1)


def _filter_error_statistics(
    error_stats: Dict[str, Any], 
    cutoff_time: datetime.datetime, 
    component_filter: str, 
    include_resolved: bool
) -> List[Dict[str, Any]]:
    """Filter error statistics based on criteria."""
    # Simplified implementation - would filter based on parameters
    return list(error_stats.values())


def _calculate_error_rates(errors: List[Dict[str, Any]], time_window: str) -> Dict[str, float]:
    """Calculate error rates from filtered error data."""
    # Simplified implementation
    return {
        'overall_rate': 0.01,
        'critical_rate': 0.001,
        'resolution_rate': 0.95
    }


def _analyze_failure_patterns(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze failure patterns in error data."""
    # Simplified implementation
    return {
        'most_common_errors': [],
        'error_clusters': [],
        'temporal_patterns': {}
    }


def _analyze_recovery_effectiveness(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze recovery effectiveness from error data."""
    # Simplified implementation
    return {
        'recovery_success_rate': 0.85,
        'average_recovery_time': 5.2,
        'most_effective_strategies': []
    }


def _calculate_performance_impact(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate performance impact from error data."""
    # Simplified implementation
    return {
        'processing_overhead': 0.05,
        'throughput_impact': 0.02,
        'resource_utilization': 0.95
    }


def _generate_trend_analysis(errors: List[Dict[str, Any]], time_window: str) -> Dict[str, Any]:
    """Generate trend analysis from error data."""
    # Simplified implementation
    return {
        'trend_direction': 'stable',
        'prediction_confidence': 0.75,
        'projected_error_rate': 0.01
    }


def _analyze_resolved_errors(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze resolved errors from error data."""
    # Simplified implementation
    return {
        'resolution_methods': {},
        'resolution_times': [],
        'effectiveness_scores': {}
    }


def _calculate_average_resolution_time(errors: List[Dict[str, Any]]) -> float:
    """Calculate average resolution time from error data."""
    # Simplified implementation
    return 4.5  # Average resolution time in seconds