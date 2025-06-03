"""
Comprehensive algorithm interface module providing standardized interface abstraction, algorithm lifecycle 
management, parameter validation, execution context management, and performance tracking for navigation 
algorithms in the plume simulation system.

This module implements algorithm discovery, instantiation, validation, execution coordination, and result 
standardization to ensure >95% correlation with reference implementations and support reproducible research 
outcomes across different plume recording formats with comprehensive error handling and audit trail integration.

Key Features:
- Centralized algorithm interface management with thread-safe operations
- Comprehensive parameter validation with scientific computing constraints
- Performance tracking with <7.2 seconds target execution time
- Cross-platform compatibility for different plume formats
- Algorithm discovery and metadata management with dynamic loading
- Batch processing support for 4000+ simulation requirements
- Scientific computing standards compliance and reproducibility assessment
- Audit trail integration for scientific traceability
- Graceful degradation and error recovery mechanisms
- Interface statistics and performance monitoring
"""

# External imports with version specifications
from abc import ABC, abstractmethod  # Python 3.9+ - Abstract base class functionality
import numpy as np  # version: 2.1.3+ - Numerical array operations and scientific computing
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Type  # Python 3.9+ - Type hints
import dataclasses  # Python 3.9+ - Data classes for interface configuration and results
import datetime  # Python 3.9+ - Timestamp generation for execution tracking
import uuid  # Python 3.9+ - Unique identifier generation for correlation tracking
import copy  # Python 3.9+ - Deep copying for state preservation and parameter isolation
import threading  # Python 3.9+ - Thread-safe interface operations and concurrent execution
import contextlib  # Python 3.9+ - Context manager utilities for execution scope
import weakref  # Python 3.9+ - Weak references for instance management and memory optimization
import time  # Python 3.9+ - High-precision timing for performance measurement
import json  # Python 3.9+ - JSON serialization for interface configuration
import collections  # Python 3.9+ - Efficient data structures for interface management

# Internal imports from algorithm framework
from ...algorithms.base_algorithm import (
    BaseAlgorithm, AlgorithmParameters, AlgorithmResult, AlgorithmContext,
    validate_plume_data, create_algorithm_context, calculate_performance_metrics
)
from ...algorithms.algorithm_registry import (
    get_algorithm, create_algorithm_instance, validate_algorithm_interface, list_algorithms
)

# Internal imports from utility modules
from ...utils.validation_utils import (
    ValidationResult, validate_algorithm_parameters, validate_numerical_accuracy
)
from ...utils.logging_utils import (
    get_logger, set_scientific_context, create_audit_trail, log_simulation_event
)
from ...error.exceptions import (
    PlumeSimulationException, ValidationError, SimulationError
)

# Global constants for algorithm interface execution and validation
ALGORITHM_INTERFACE_VERSION = '1.0.0'
DEFAULT_EXECUTION_TIMEOUT = 300.0
PERFORMANCE_TRACKING_ENABLED = True
VALIDATION_ENABLED = True
SCIENTIFIC_CONTEXT_ENABLED = True
DEFAULT_CORRELATION_THRESHOLD = 0.95
REPRODUCIBILITY_THRESHOLD = 0.99
MAX_ALGORITHM_INSTANCES = 50
ALGORITHM_CACHE_TTL = 3600.0

# Global interface management state with thread-safe operations
_algorithm_instance_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_interface_statistics: Dict[str, int] = {
    'total_executions': 0,
    'successful_executions': 0,
    'failed_executions': 0
}
_execution_context_stack = threading.local()
_interface_lock = threading.RLock()


def extract_algorithm_parameters(
    config_data: Dict[str, Any],
    algorithm_name: str,
    validate_types: bool = True
) -> AlgorithmParameters:
    """
    Extract algorithm-specific parameters from configuration with validation and type conversion, 
    localized functionality to avoid circular dependency while ensuring comprehensive parameter 
    extraction and constraint validation.
    
    Args:
        config_data: Configuration data dictionary containing algorithm parameters
        algorithm_name: Name of the algorithm to extract parameters for
        validate_types: Whether to enable type validation and conversion
        
    Returns:
        AlgorithmParameters: Extracted and validated algorithm parameters with type conversion and constraint validation
    """
    logger = get_logger('algorithm_interface.extract_parameters', 'ALGORITHM')
    
    try:
        # Extract algorithm-specific section from configuration data
        algorithm_section = config_data.get('algorithms', {}).get(algorithm_name, {})
        if not algorithm_section:
            # Fallback to root level parameters for algorithm
            algorithm_section = config_data.get(algorithm_name, {})
        
        # Initialize parameter extraction with validation context
        extracted_params = {}
        param_constraints = {}
        
        # Extract core algorithm parameters with type conversion
        for param_name, param_value in algorithm_section.items():
            if param_name in ['constraints', 'metadata', 'version']:
                continue  # Skip metadata fields
            
            # Validate parameter types and convert values if validation enabled
            if validate_types:
                converted_value = _convert_parameter_type(param_name, param_value, algorithm_name)
                extracted_params[param_name] = converted_value
            else:
                extracted_params[param_name] = param_value
        
        # Extract parameter constraints and bounds from configuration
        if 'constraints' in algorithm_section:
            param_constraints = algorithm_section['constraints']
        
        # Apply default constraints for common algorithm parameters
        default_constraints = _get_default_algorithm_constraints(algorithm_name)
        param_constraints.update(default_constraints)
        
        # Create AlgorithmParameters instance with extracted values
        algorithm_parameters = AlgorithmParameters(
            algorithm_name=algorithm_name,
            version=algorithm_section.get('version', ALGORITHM_INTERFACE_VERSION),
            parameters=extracted_params,
            constraints=param_constraints,
            convergence_tolerance=algorithm_section.get('convergence_tolerance', 1e-6),
            max_iterations=algorithm_section.get('max_iterations', 10000),
            enable_performance_tracking=algorithm_section.get('enable_performance_tracking', PERFORMANCE_TRACKING_ENABLED)
        )
        
        # Validate parameter consistency and dependencies
        if validate_types:
            validation_result = algorithm_parameters.validate(strict_validation=True)
            if not validation_result.is_valid:
                raise ValidationError(
                    message=f"Parameter extraction validation failed for {algorithm_name}",
                    validation_type="parameter_extraction",
                    validation_context={'algorithm_name': algorithm_name},
                    failed_parameters=[error.split(':')[0] for error in validation_result.errors]
                )
        
        # Log parameter extraction operation for audit trail
        logger.info(f"Algorithm parameters extracted successfully: {algorithm_name} ({len(extracted_params)} parameters)")
        
        # Create audit trail entry for parameter extraction
        create_audit_trail(
            action='ALGORITHM_PARAMETERS_EXTRACTED',
            component='ALGORITHM_INTERFACE',
            action_details={
                'algorithm_name': algorithm_name,
                'parameter_count': len(extracted_params),
                'validation_enabled': validate_types,
                'extracted_parameters': list(extracted_params.keys())
            },
            user_context='SYSTEM'
        )
        
        return algorithm_parameters
        
    except Exception as e:
        logger.error(f"Parameter extraction failed for {algorithm_name}: {e}", exc_info=True)
        raise SimulationError(
            message=f"Algorithm parameter extraction failed: {str(e)}",
            simulation_id="parameter_extraction",
            algorithm_name=algorithm_name,
            simulation_context={'error': str(e), 'stage': 'parameter_extraction'}
        )


def create_algorithm_interface(
    algorithm_name: str,
    interface_config: Dict[str, Any],
    enable_validation: bool = True,
    enable_performance_tracking: bool = True
) -> 'AlgorithmInterface':
    """
    Create algorithm interface instance with comprehensive configuration, validation setup, and performance 
    tracking for standardized algorithm management and execution coordination with scientific computing compliance.
    
    Args:
        algorithm_name: Name of the navigation algorithm to create interface for
        interface_config: Configuration dictionary for interface setup
        enable_validation: Whether to enable comprehensive validation
        enable_performance_tracking: Whether to enable performance tracking
        
    Returns:
        AlgorithmInterface: Configured algorithm interface instance with validation and performance tracking
    """
    logger = get_logger('algorithm_interface.create_interface', 'ALGORITHM')
    
    try:
        # Validate algorithm name and interface configuration
        if not algorithm_name or not isinstance(algorithm_name, str):
            raise ValueError("Algorithm name must be a non-empty string")
        
        if not isinstance(interface_config, dict):
            raise TypeError("Interface configuration must be a dictionary")
        
        # Retrieve algorithm class from registry with dynamic loading support
        try:
            algorithm_class = get_algorithm(
                algorithm_name=algorithm_name,
                validate_availability=True,
                enable_dynamic_loading=True
            )
        except Exception as e:
            raise SimulationError(
                message=f"Failed to retrieve algorithm from registry: {str(e)}",
                simulation_id="interface_creation",
                algorithm_name=algorithm_name,
                simulation_context={'error': str(e), 'stage': 'algorithm_retrieval'}
            )
        
        # Create algorithm interface instance with configuration
        interface_instance = AlgorithmInterface(
            algorithm_name=algorithm_name,
            interface_config=interface_config,
            enable_validation=enable_validation,
            enable_performance_tracking=enable_performance_tracking
        )
        
        # Setup validation framework if validation enabled
        if enable_validation:
            validation_result = validate_algorithm_interface(algorithm_class, strict_validation=True)
            if not validation_result.is_valid:
                raise ValidationError(
                    message=f"Algorithm interface validation failed for {algorithm_name}",
                    validation_type="interface_validation",
                    validation_context={'algorithm_name': algorithm_name},
                    failed_parameters=['algorithm_interface']
                )
        
        # Initialize performance tracking if performance tracking enabled
        if enable_performance_tracking:
            interface_instance.performance_tracking_enabled = True
            interface_instance.performance_metrics = {}
        
        # Configure scientific context and audit trail integration
        set_scientific_context(
            simulation_id=interface_config.get('simulation_id', 'interface_creation'),
            algorithm_name=algorithm_name,
            processing_stage='INTERFACE_CREATION',
            additional_context={'interface_config': interface_config}
        )
        
        # Register interface instance in cache for reuse
        with _interface_lock:
            cache_key = f"{algorithm_name}_{uuid.uuid4().hex[:8]}"
            _algorithm_instance_cache[cache_key] = interface_instance
            
            # Limit cache size for memory management
            if len(_algorithm_instance_cache) > MAX_ALGORITHM_INSTANCES:
                # Remove oldest entries (simplified approach)
                oldest_keys = list(_algorithm_instance_cache.keys())[:10]
                for key in oldest_keys:
                    _algorithm_instance_cache.pop(key, None)
        
        # Log interface creation with configuration details
        logger.info(f"Algorithm interface created successfully: {algorithm_name}")
        
        # Create audit trail entry for interface creation
        create_audit_trail(
            action='ALGORITHM_INTERFACE_CREATED',
            component='ALGORITHM_INTERFACE',
            action_details={
                'algorithm_name': algorithm_name,
                'validation_enabled': enable_validation,
                'performance_tracking_enabled': enable_performance_tracking,
                'interface_config_keys': list(interface_config.keys())
            },
            user_context='SYSTEM'
        )
        
        return interface_instance
        
    except Exception as e:
        logger.error(f"Algorithm interface creation failed for {algorithm_name}: {e}", exc_info=True)
        raise SimulationError(
            message=f"Algorithm interface creation failed: {str(e)}",
            simulation_id="interface_creation",
            algorithm_name=algorithm_name,
            simulation_context={'error': str(e), 'stage': 'interface_creation'}
        )


def validate_interface_compatibility(
    algorithm_names: List[str],
    compatibility_requirements: Dict[str, Any],
    strict_validation: bool = False
) -> ValidationResult:
    """
    Validate algorithm interface compatibility including parameter compatibility, execution context requirements, 
    and performance characteristics for cross-algorithm validation and batch processing coordination.
    
    Args:
        algorithm_names: List of algorithm names to validate for compatibility
        compatibility_requirements: Dictionary of compatibility requirements and constraints
        strict_validation: Whether to enable strict validation mode with enhanced checking
        
    Returns:
        ValidationResult: Interface compatibility validation result with detailed analysis and recommendations
    """
    logger = get_logger('algorithm_interface.validate_compatibility', 'ALGORITHM')
    
    # Create validation result container for compatibility assessment
    validation_result = ValidationResult(
        validation_type="interface_compatibility_validation",
        is_valid=True,
        validation_context=f"algorithms={algorithm_names}, strict={strict_validation}"
    )
    
    try:
        # Validate input parameters
        if not algorithm_names or not isinstance(algorithm_names, list):
            validation_result.add_error("Algorithm names must be a non-empty list", severity="HIGH")
            validation_result.is_valid = False
            return validation_result
        
        if len(algorithm_names) < 2:
            validation_result.add_warning("Compatibility validation requires at least 2 algorithms")
            return validation_result
        
        # Retrieve algorithm interfaces for all specified algorithms
        algorithm_interfaces = {}
        for algorithm_name in algorithm_names:
            try:
                interface = create_algorithm_interface(
                    algorithm_name=algorithm_name,
                    interface_config=compatibility_requirements.get(algorithm_name, {}),
                    enable_validation=True,
                    enable_performance_tracking=False
                )
                algorithm_interfaces[algorithm_name] = interface
            except Exception as e:
                validation_result.add_error(
                    f"Failed to create interface for {algorithm_name}: {str(e)}",
                    severity="HIGH"
                )
                validation_result.is_valid = False
        
        if not algorithm_interfaces:
            validation_result.add_error("No algorithm interfaces could be created", severity="CRITICAL")
            validation_result.is_valid = False
            return validation_result
        
        # Validate parameter compatibility across algorithms
        parameter_compatibility = _validate_parameter_compatibility(
            algorithm_interfaces, compatibility_requirements, validation_result
        )
        
        # Check execution context requirements and constraints
        context_compatibility = _validate_execution_context_compatibility(
            algorithm_interfaces, compatibility_requirements, validation_result
        )
        
        # Validate performance characteristics and resource requirements
        performance_compatibility = _validate_performance_compatibility(
            algorithm_interfaces, compatibility_requirements, validation_result
        )
        
        # Apply strict validation criteria if strict validation enabled
        if strict_validation:
            _apply_strict_compatibility_validation(
                algorithm_interfaces, compatibility_requirements, validation_result
            )
        
        # Generate compatibility analysis and recommendations
        compatibility_score = _calculate_compatibility_score(
            parameter_compatibility, context_compatibility, performance_compatibility
        )
        
        validation_result.add_metric("compatibility_score", compatibility_score)
        validation_result.add_metric("algorithms_validated", len(algorithm_interfaces))
        validation_result.add_metric("parameter_compatibility", parameter_compatibility)
        validation_result.add_metric("context_compatibility", context_compatibility)
        validation_result.add_metric("performance_compatibility", performance_compatibility)
        
        # Generate validation recommendations based on compatibility analysis
        if compatibility_score >= 0.8:
            validation_result.add_recommendation(
                "Algorithm interfaces are highly compatible for batch processing",
                priority="INFO"
            )
        elif compatibility_score >= 0.6:
            validation_result.add_recommendation(
                "Algorithm interfaces are moderately compatible - review parameter differences",
                priority="MEDIUM"
            )
        else:
            validation_result.add_recommendation(
                "Algorithm interfaces have significant compatibility issues - address before batch processing",
                priority="HIGH"
            )
            validation_result.is_valid = False
        
        # Log compatibility validation results for audit trail
        logger.info(f"Interface compatibility validation completed: score={compatibility_score:.3f}, valid={validation_result.is_valid}")
        
        # Create audit trail entry for compatibility validation
        create_audit_trail(
            action='INTERFACE_COMPATIBILITY_VALIDATED',
            component='ALGORITHM_INTERFACE',
            action_details={
                'algorithm_names': algorithm_names,
                'compatibility_score': compatibility_score,
                'validation_result': validation_result.is_valid,
                'strict_validation': strict_validation
            },
            user_context='SYSTEM'
        )
        
    except Exception as e:
        validation_result.add_error(f"Compatibility validation failed: {str(e)}", severity="CRITICAL")
        validation_result.is_valid = False
        logger.error(f"Interface compatibility validation error: {e}", exc_info=True)
    
    validation_result.finalize_validation()
    return validation_result


def execute_algorithm_with_interface(
    algorithm_name: str,
    plume_data: np.ndarray,
    plume_metadata: Dict[str, Any],
    algorithm_parameters: AlgorithmParameters,
    simulation_id: str,
    execution_config: Dict[str, Any]
) -> AlgorithmResult:
    """
    Execute algorithm through interface with comprehensive parameter validation, performance tracking, error 
    handling, and result standardization for reliable algorithm execution and scientific computing compliance.
    
    Args:
        algorithm_name: Name of the navigation algorithm to execute
        plume_data: Plume data array for algorithm processing
        plume_metadata: Metadata containing format and calibration information
        algorithm_parameters: Algorithm parameters for execution
        simulation_id: Unique identifier for the simulation run
        execution_config: Configuration for algorithm execution environment
        
    Returns:
        AlgorithmResult: Comprehensive algorithm execution result with performance metrics and validation
    """
    logger = get_logger('algorithm_interface.execute_algorithm', 'ALGORITHM')
    
    # Update global execution statistics
    with _interface_lock:
        _interface_statistics['total_executions'] += 1
    
    try:
        # Validate algorithm name and retrieve interface from cache or registry
        if not algorithm_name or not isinstance(algorithm_name, str):
            raise ValueError("Algorithm name must be a non-empty string")
        
        # Validate plume data format and compatibility with algorithm requirements
        if not isinstance(plume_data, np.ndarray):
            raise TypeError("Plume data must be a numpy array")
        
        plume_validation = validate_plume_data(
            plume_data=plume_data,
            plume_metadata=plume_metadata,
            strict_validation=False
        )
        
        if not plume_validation.is_valid:
            raise ValidationError(
                message="Plume data validation failed",
                validation_type="plume_data_validation",
                validation_context={'simulation_id': simulation_id},
                failed_parameters=['plume_data']
            )
        
        # Validate algorithm parameters using interface validation framework
        parameter_validation = algorithm_parameters.validate(strict_validation=True)
        if not parameter_validation.is_valid:
            raise ValidationError(
                message=f"Algorithm parameters validation failed: {algorithm_name}",
                validation_type="parameter_validation",
                validation_context={'simulation_id': simulation_id},
                failed_parameters=['algorithm_parameters']
            )
        
        # Create algorithm execution context with scientific context and performance tracking
        execution_context = create_algorithm_context(
            algorithm_name=algorithm_name,
            simulation_id=simulation_id,
            algorithm_parameters=algorithm_parameters.parameters,
            execution_config=execution_config
        )
        
        # Setup algorithm instance with validated parameters and configuration
        algorithm_instance = create_algorithm_instance(
            algorithm_name=algorithm_name,
            parameters=algorithm_parameters,
            execution_config=execution_config,
            enable_validation=True
        )
        
        # Execute algorithm with comprehensive error handling and timeout management
        execution_start_time = time.time()
        
        with execution_context:
            # Execute algorithm with timeout management
            algorithm_result = algorithm_instance.execute(
                plume_data=plume_data,
                plume_metadata=plume_metadata,
                simulation_id=simulation_id
            )
        
        execution_end_time = time.time()
        execution_duration = execution_end_time - execution_start_time
        
        # Track performance metrics and validate execution results
        algorithm_result.execution_time = execution_duration
        algorithm_result.add_performance_metric('interface_execution_time', execution_duration, 'seconds')
        algorithm_result.add_performance_metric('plume_validation_passed', 1.0 if plume_validation.is_valid else 0.0)
        algorithm_result.add_performance_metric('parameter_validation_passed', 1.0 if parameter_validation.is_valid else 0.0)
        
        # Validate execution results against scientific computing standards
        result_validation = algorithm_instance.validate_execution_result(
            result=algorithm_result,
            reference_metrics=execution_config.get('reference_metrics', {})
        )
        
        if not result_validation.is_valid:
            algorithm_result.add_warning(
                f"Result validation issues detected: {len(result_validation.errors)} errors",
                warning_category="result_validation"
            )
        
        # Generate standardized algorithm result with performance analysis
        performance_metrics = calculate_performance_metrics(
            algorithm_result=algorithm_result,
            reference_metrics=execution_config.get('reference_metrics', {}),
            include_correlation_analysis=True
        )
        
        # Update algorithm result with calculated performance metrics
        for metric_name, metric_value in performance_metrics.items():
            algorithm_result.add_performance_metric(metric_name, metric_value)
        
        # Update interface statistics and cache execution results
        with _interface_lock:
            if algorithm_result.success:
                _interface_statistics['successful_executions'] += 1
            else:
                _interface_statistics['failed_executions'] += 1
        
        # Log algorithm execution completion with performance metrics
        logger.info(
            f"Algorithm execution completed: {algorithm_name} [{simulation_id}] - "
            f"success={algorithm_result.success}, time={execution_duration:.3f}s"
        )
        
        # Create audit trail entry for algorithm execution
        create_audit_trail(
            action='ALGORITHM_EXECUTED_WITH_INTERFACE',
            component='ALGORITHM_INTERFACE',
            action_details={
                'algorithm_name': algorithm_name,
                'simulation_id': simulation_id,
                'execution_duration': execution_duration,
                'execution_success': algorithm_result.success,
                'performance_metrics': performance_metrics
            },
            user_context='SYSTEM'
        )
        
        return algorithm_result
        
    except Exception as e:
        # Update failed execution statistics
        with _interface_lock:
            _interface_statistics['failed_executions'] += 1
        
        logger.error(f"Algorithm execution with interface failed: {algorithm_name} [{simulation_id}] - {e}", exc_info=True)
        
        # Generate comprehensive algorithm result even on failure
        if isinstance(e, (ValidationError, SimulationError)):
            # Re-raise known exceptions with context
            raise
        else:
            # Wrap unknown exceptions in SimulationError
            raise SimulationError(
                message=f"Algorithm execution with interface failed: {str(e)}",
                simulation_id=simulation_id,
                algorithm_name=algorithm_name,
                simulation_context={
                    'error': str(e),
                    'stage': 'interface_execution',
                    'execution_config': execution_config
                }
            )


def get_interface_statistics(
    algorithm_name: Optional[str] = None,
    include_performance_trends: bool = False,
    time_period: str = 'all'
) -> Dict[str, Any]:
    """
    Get comprehensive algorithm interface statistics including execution counts, performance metrics, error 
    rates, and usage patterns for system monitoring and optimization analysis.
    
    Args:
        algorithm_name: Specific algorithm name to filter statistics (None for all)
        include_performance_trends: Whether to include performance trends analysis
        time_period: Time period for statistics ('all', 'recent', 'daily')
        
    Returns:
        Dict[str, Any]: Comprehensive interface statistics with performance analysis and usage patterns
    """
    logger = get_logger('algorithm_interface.get_statistics', 'ALGORITHM')
    
    try:
        # Filter statistics by algorithm name if specified
        with _interface_lock:
            base_statistics = _interface_statistics.copy()
        
        # Compile execution counts and success rates
        statistics = {
            'interface_version': ALGORITHM_INTERFACE_VERSION,
            'collection_timestamp': datetime.datetime.now().isoformat(),
            'time_period': time_period,
            'algorithm_filter': algorithm_name,
            'execution_statistics': base_statistics,
            'cache_statistics': {
                'cached_instances': len(_algorithm_instance_cache),
                'max_instances': MAX_ALGORITHM_INSTANCES,
                'cache_ttl_seconds': ALGORITHM_CACHE_TTL
            }
        }
        
        # Calculate success and performance rates
        total_executions = base_statistics.get('total_executions', 0)
        successful_executions = base_statistics.get('successful_executions', 0)
        failed_executions = base_statistics.get('failed_executions', 0)
        
        if total_executions > 0:
            statistics['performance_analysis'] = {
                'success_rate': successful_executions / total_executions,
                'failure_rate': failed_executions / total_executions,
                'execution_efficiency': successful_executions / total_executions if total_executions > 0 else 0.0
            }
        else:
            statistics['performance_analysis'] = {
                'success_rate': 0.0,
                'failure_rate': 0.0,
                'execution_efficiency': 0.0
            }
        
        # Include performance trends if requested
        if include_performance_trends:
            # Get algorithm listing with performance metrics
            algorithm_listing = list_algorithms(
                include_performance_metrics=True,
                only_available=True
            )
            
            performance_trends = {}
            for algo_name, algo_info in algorithm_listing.items():
                if algorithm_name is None or algo_name == algorithm_name:
                    performance_metrics = algo_info.get('performance_metrics', {})
                    if performance_metrics:
                        performance_trends[algo_name] = {
                            'avg_execution_time': performance_metrics.get('execution_time_seconds', 0.0),
                            'success_rate': performance_metrics.get('success_rate', 0.0),
                            'correlation_score': performance_metrics.get('overall_correlation', 0.0),
                            'last_execution': algo_info.get('last_accessed', 'unknown')
                        }
            
            statistics['performance_trends'] = performance_trends
        
        # Include usage patterns and resource utilization
        statistics['system_metrics'] = {
            'interface_tracking_enabled': PERFORMANCE_TRACKING_ENABLED,
            'validation_enabled': VALIDATION_ENABLED,
            'scientific_context_enabled': SCIENTIFIC_CONTEXT_ENABLED,
            'correlation_threshold': DEFAULT_CORRELATION_THRESHOLD,
            'reproducibility_threshold': REPRODUCIBILITY_THRESHOLD
        }
        
        # Generate optimization recommendations based on statistics
        recommendations = []
        if statistics['performance_analysis']['success_rate'] < 0.8:
            recommendations.append("Success rate below 80% - review algorithm configurations and validation")
        
        if statistics['cache_statistics']['cached_instances'] > MAX_ALGORITHM_INSTANCES * 0.8:
            recommendations.append("Cache utilization high - consider increasing cache size or TTL")
        
        if not recommendations:
            recommendations.append("Interface performance within acceptable ranges")
        
        statistics['optimization_recommendations'] = recommendations
        
        # Log statistics request for audit trail
        logger.debug(f"Interface statistics retrieved: algorithm={algorithm_name}, trends={include_performance_trends}")
        
        return statistics
        
    except Exception as e:
        logger.error(f"Failed to retrieve interface statistics: {e}", exc_info=True)
        return {
            'error': f"Statistics retrieval failed: {str(e)}",
            'timestamp': datetime.datetime.now().isoformat(),
            'interface_version': ALGORITHM_INTERFACE_VERSION
        }


def clear_interface_cache(
    algorithm_name: Optional[str] = None,
    preserve_statistics: bool = True,
    force_cleanup: bool = False
) -> Dict[str, int]:
    """
    Clear algorithm interface cache with selective clearing options and statistics preservation for cache 
    management and memory optimization with comprehensive cleanup reporting.
    
    Args:
        algorithm_name: Specific algorithm to clear from cache (None for all)
        preserve_statistics: Whether to preserve interface statistics during clearing
        force_cleanup: Whether to force cleanup of weak references
        
    Returns:
        Dict[str, int]: Cache clearing statistics with cleared entries count and preserved data summary
    """
    logger = get_logger('algorithm_interface.clear_cache', 'ALGORITHM')
    
    try:
        # Identify cache entries to clear based on algorithm name filter
        with _interface_lock:
            initial_cache_size = len(_algorithm_instance_cache)
            cleared_entries = 0
            preserved_entries = 0
            
            if algorithm_name is None:
                # Clear all cache entries
                cleared_entries = len(_algorithm_instance_cache)
                _algorithm_instance_cache.clear()
            else:
                # Clear specific algorithm instances
                keys_to_remove = []
                for cache_key, instance in _algorithm_instance_cache.items():
                    if hasattr(instance, 'algorithm_name') and instance.algorithm_name == algorithm_name:
                        keys_to_remove.append(cache_key)
                
                for key in keys_to_remove:
                    if key in _algorithm_instance_cache:
                        del _algorithm_instance_cache[key]
                        cleared_entries += 1
                
                preserved_entries = len(_algorithm_instance_cache)
            
            # Preserve interface statistics if preserve_statistics enabled
            statistics_preserved = preserve_statistics
            if not preserve_statistics:
                global _interface_statistics
                _interface_statistics = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'failed_executions': 0
                }
        
        # Force cleanup of weak references if force_cleanup enabled
        if force_cleanup:
            import gc
            gc.collect()  # Force garbage collection to clean up weak references
        
        # Generate cache clearing summary
        clearing_stats = {
            'initial_cache_size': initial_cache_size,
            'cleared_entries': cleared_entries,
            'preserved_entries': preserved_entries,
            'final_cache_size': len(_algorithm_instance_cache),
            'algorithm_filter': algorithm_name or 'all',
            'statistics_preserved': statistics_preserved,
            'force_cleanup_applied': force_cleanup,
            'cleared_at': datetime.datetime.now().isoformat()
        }
        
        # Log cache clearing operation with statistics
        logger.info(
            f"Interface cache cleared: {cleared_entries} entries removed, "
            f"{preserved_entries} preserved, algorithm_filter={algorithm_name or 'all'}"
        )
        
        # Create audit trail entry for cache clearing
        create_audit_trail(
            action='INTERFACE_CACHE_CLEARED',
            component='ALGORITHM_INTERFACE',
            action_details=clearing_stats,
            user_context='SYSTEM'
        )
        
        return clearing_stats
        
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}", exc_info=True)
        return {
            'error': f"Cache clearing failed: {str(e)}",
            'timestamp': datetime.datetime.now().isoformat()
        }


class AlgorithmInterface:
    """
    Comprehensive algorithm interface class providing standardized algorithm lifecycle management, parameter 
    validation, execution coordination, performance tracking, and result standardization for navigation 
    algorithms in the plume simulation system with scientific computing compliance and reproducible research support.
    
    This class provides complete algorithm interface functionality with comprehensive validation, performance 
    tracking, and scientific context management for reproducible algorithm execution and research outcomes.
    """
    
    def __init__(
        self,
        algorithm_name: str,
        interface_config: Dict[str, Any],
        enable_validation: bool = True,
        enable_performance_tracking: bool = True
    ):
        """
        Initialize algorithm interface with configuration, validation setup, and performance tracking for 
        comprehensive algorithm management and scientific computing compliance.
        
        Args:
            algorithm_name: Name of the navigation algorithm
            interface_config: Configuration dictionary for interface setup
            enable_validation: Whether to enable comprehensive validation
            enable_performance_tracking: Whether to enable performance tracking
        """
        # Set algorithm name and interface version
        self.algorithm_name = algorithm_name
        self.interface_version = ALGORITHM_INTERFACE_VERSION
        
        # Store interface configuration and validation settings
        self.interface_config = interface_config.copy()
        self.validation_enabled = enable_validation
        self.performance_tracking_enabled = enable_performance_tracking
        
        # Retrieve algorithm class from registry with metadata
        try:
            self.algorithm_class = get_algorithm(
                algorithm_name=algorithm_name,
                validate_availability=True,
                enable_dynamic_loading=True
            )
        except Exception as e:
            raise SimulationError(
                message=f"Failed to retrieve algorithm class: {str(e)}",
                simulation_id="interface_initialization",
                algorithm_name=algorithm_name,
                simulation_context={'error': str(e), 'stage': 'algorithm_retrieval'}
            )
        
        # Initialize algorithm instance (lazy loading)
        self.algorithm_instance: Optional[BaseAlgorithm] = None
        
        # Extract algorithm metadata from configuration
        self.algorithm_metadata = interface_config.get('metadata', {})
        
        # Setup logger with algorithm-specific context
        self.logger = get_logger(f'algorithm_interface.{algorithm_name}', 'ALGORITHM')
        
        # Initialize execution history and performance metrics tracking
        self.execution_history: List[AlgorithmResult] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Set creation and tracking timestamps
        self.creation_timestamp = datetime.datetime.now()
        self.last_execution_timestamp: Optional[datetime.datetime] = None
        
        # Initialize execution counters
        self.execution_count = 0
        
        # Configure validation framework if validation enabled
        if self.validation_enabled:
            # Validate algorithm interface compliance
            validation_result = validate_algorithm_interface(self.algorithm_class, strict_validation=True)
            if not validation_result.is_valid:
                raise ValidationError(
                    message=f"Algorithm interface validation failed: {algorithm_name}",
                    validation_type="interface_compliance",
                    validation_context={'algorithm_name': algorithm_name},
                    failed_parameters=['algorithm_interface']
                )
        
        # Setup performance tracking if performance tracking enabled
        if self.performance_tracking_enabled:
            self.performance_metrics = {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'average_execution_time': 0.0,
                'last_execution_time': 0.0
            }
        
        # Mark interface as initialized and ready for use
        self.is_initialized = True
        
        # Log interface initialization
        self.logger.info(f"Algorithm interface initialized: {algorithm_name}")
    
    def validate_parameters(
        self,
        parameters: AlgorithmParameters,
        strict_validation: bool = False,
        validation_context: Dict[str, Any] = None
    ) -> ValidationResult:
        """
        Validate algorithm parameters using comprehensive validation framework with scientific computing 
        constraints and cross-parameter dependency checking.
        
        Args:
            parameters: Algorithm parameters to validate
            strict_validation: Whether to enable strict validation mode
            validation_context: Additional context for validation
            
        Returns:
            ValidationResult: Comprehensive parameter validation result with constraint compliance assessment
        """
        # Create validation result container for parameter assessment
        validation_result = ValidationResult(
            validation_type="interface_parameter_validation",
            is_valid=True,
            validation_context=f"algorithm={self.algorithm_name}, strict={strict_validation}"
        )
        
        try:
            # Validate parameter types and value ranges using validation utilities
            parameter_validation = validate_algorithm_parameters(
                algorithm_params=parameters.parameters,
                algorithm_type=self.algorithm_name,
                validate_convergence_criteria=strict_validation,
                algorithm_constraints=parameters.constraints
            )
            
            # Merge validation results
            validation_result.errors.extend(parameter_validation.errors)
            validation_result.warnings.extend(parameter_validation.warnings)
            validation_result.is_valid = validation_result.is_valid and parameter_validation.is_valid
            
            # Check algorithm-specific parameter constraints and bounds
            for param_name, param_value in parameters.parameters.items():
                if param_name in parameters.constraints:
                    constraint = parameters.constraints[param_name]
                    
                    # Validate against constraint bounds
                    if 'min' in constraint and param_value < constraint['min']:
                        validation_result.add_error(
                            f"Parameter {param_name} below minimum: {param_value} < {constraint['min']}",
                            severity="HIGH"
                        )
                        validation_result.is_valid = False
                    
                    if 'max' in constraint and param_value > constraint['max']:
                        validation_result.add_error(
                            f"Parameter {param_name} above maximum: {param_value} > {constraint['max']}",
                            severity="HIGH"
                        )
                        validation_result.is_valid = False
            
            # Validate cross-parameter dependencies and relationships
            if strict_validation:
                self._validate_parameter_dependencies(parameters, validation_result)
            
            # Include validation context in parameter assessment
            if validation_context:
                validation_result.set_metadata('validation_context', validation_context)
            
            # Generate parameter validation recommendations
            if not validation_result.is_valid:
                validation_result.add_recommendation(
                    "Correct parameter constraint violations before execution",
                    priority="HIGH"
                )
            else:
                validation_result.add_recommendation(
                    "Parameters passed validation - ready for execution",
                    priority="INFO"
                )
            
            # Log parameter validation results for audit trail
            self.logger.debug(f"Parameter validation: valid={validation_result.is_valid}, errors={len(validation_result.errors)}")
            
        except Exception as e:
            validation_result.add_error(f"Parameter validation failed: {str(e)}", severity="CRITICAL")
            validation_result.is_valid = False
            self.logger.error(f"Parameter validation error: {e}", exc_info=True)
        
        validation_result.finalize_validation()
        return validation_result
    
    def create_algorithm_instance(
        self,
        parameters: AlgorithmParameters,
        execution_config: Dict[str, Any],
        instance_id: str = None
    ) -> BaseAlgorithm:
        """
        Create algorithm instance with validated parameters, performance tracking setup, and scientific 
        context configuration for isolated algorithm execution.
        
        Args:
            parameters: Algorithm parameters for instance creation
            execution_config: Configuration for algorithm execution
            instance_id: Unique identifier for the instance
            
        Returns:
            BaseAlgorithm: Configured algorithm instance with validation and performance tracking
        """
        try:
            # Validate algorithm parameters using interface validation framework
            if self.validation_enabled:
                parameter_validation = self.validate_parameters(parameters, strict_validation=True)
                if not parameter_validation.is_valid:
                    raise ValidationError(
                        message=f"Parameter validation failed for {self.algorithm_name}",
                        validation_type="instance_creation_validation",
                        validation_context={'algorithm_name': self.algorithm_name},
                        failed_parameters=['algorithm_parameters']
                    )
            
            # Generate unique instance identifier if not provided
            if not instance_id:
                instance_id = f"{self.algorithm_name}_{uuid.uuid4().hex[:8]}"
            
            # Create algorithm instance using algorithm class constructor
            algorithm_instance = self.algorithm_class(
                parameters=parameters,
                execution_config=execution_config
            )
            
            # Configure algorithm instance with validated parameters
            if hasattr(algorithm_instance, 'interface_id'):
                algorithm_instance.interface_id = instance_id
            
            # Setup performance tracking for algorithm instance
            if self.performance_tracking_enabled:
                algorithm_instance.performance_tracking_enabled = True
            
            # Configure scientific context for algorithm execution
            if SCIENTIFIC_CONTEXT_ENABLED:
                set_scientific_context(
                    simulation_id=execution_config.get('simulation_id', 'instance_creation'),
                    algorithm_name=self.algorithm_name,
                    processing_stage='INSTANCE_CREATION',
                    additional_context={'instance_id': instance_id}
                )
            
            # Store algorithm instance reference for management
            self.algorithm_instance = algorithm_instance
            
            # Log algorithm instance creation with configuration details
            self.logger.info(f"Algorithm instance created: {self.algorithm_name} [{instance_id}]")
            
            return algorithm_instance
            
        except Exception as e:
            self.logger.error(f"Algorithm instance creation failed: {e}", exc_info=True)
            raise SimulationError(
                message=f"Algorithm instance creation failed: {str(e)}",
                simulation_id=execution_config.get('simulation_id', 'instance_creation'),
                algorithm_name=self.algorithm_name,
                simulation_context={'error': str(e), 'stage': 'instance_creation'}
            )
    
    def execute_algorithm(
        self,
        plume_data: np.ndarray,
        plume_metadata: Dict[str, Any],
        parameters: AlgorithmParameters,
        simulation_id: str,
        execution_config: Dict[str, Any]
    ) -> AlgorithmResult:
        """
        Execute algorithm with comprehensive error handling, performance tracking, timeout management, and 
        result validation for reliable scientific computing execution.
        
        Args:
            plume_data: Plume data array for algorithm processing
            plume_metadata: Metadata containing format and calibration information
            parameters: Algorithm parameters for execution
            simulation_id: Unique identifier for the simulation run
            execution_config: Configuration for algorithm execution
            
        Returns:
            AlgorithmResult: Comprehensive algorithm execution result with performance metrics and validation
        """
        # Update execution count
        self.execution_count += 1
        self.last_execution_timestamp = datetime.datetime.now()
        
        try:
            # Validate plume data format and compatibility with algorithm requirements
            if self.validation_enabled:
                plume_validation = validate_plume_data(
                    plume_data=plume_data,
                    plume_metadata=plume_metadata,
                    strict_validation=False
                )
                
                if not plume_validation.is_valid:
                    raise ValidationError(
                        message="Plume data validation failed",
                        validation_type="plume_data_validation",
                        validation_context={'simulation_id': simulation_id},
                        failed_parameters=['plume_data']
                    )
            
            # Create or retrieve algorithm instance with validated parameters
            if not self.algorithm_instance:
                self.algorithm_instance = self.create_algorithm_instance(
                    parameters=parameters,
                    execution_config=execution_config
                )
            
            # Setup algorithm execution context with scientific context and performance tracking
            execution_context = create_algorithm_context(
                algorithm_name=self.algorithm_name,
                simulation_id=simulation_id,
                algorithm_parameters=parameters.parameters,
                execution_config=execution_config
            )
            
            # Execute algorithm with timeout management and error handling
            execution_start_time = time.time()
            
            with execution_context:
                # Execute algorithm with comprehensive error handling
                algorithm_result = self.algorithm_instance.execute(
                    plume_data=plume_data,
                    plume_metadata=plume_metadata,
                    simulation_id=simulation_id
                )
                
                # Add execution checkpoint
                execution_context.add_checkpoint('algorithm_execution_complete', {
                    'success': algorithm_result.success,
                    'converged': algorithm_result.converged
                })
            
            execution_end_time = time.time()
            execution_duration = execution_end_time - execution_start_time
            
            # Track performance metrics during algorithm execution
            if self.performance_tracking_enabled:
                self.performance_metrics['total_executions'] += 1
                self.performance_metrics['last_execution_time'] = execution_duration
                
                # Update average execution time
                total_execs = self.performance_metrics['total_executions']
                current_avg = self.performance_metrics['average_execution_time']
                self.performance_metrics['average_execution_time'] = (
                    (current_avg * (total_execs - 1) + execution_duration) / total_execs
                )
                
                if algorithm_result.success:
                    self.performance_metrics['successful_executions'] += 1
                else:
                    self.performance_metrics['failed_executions'] += 1
            
            # Validate algorithm execution results against scientific computing standards
            if self.validation_enabled:
                result_validation = self.validate_execution_result(
                    result=algorithm_result,
                    reference_metrics=execution_config.get('reference_metrics', {})
                )
                
                if not result_validation.is_valid:
                    algorithm_result.add_warning(
                        f"Result validation issues: {len(result_validation.errors)} errors",
                        warning_category="result_validation"
                    )
            
            # Generate standardized algorithm result with performance analysis
            algorithm_result.execution_time = execution_duration
            algorithm_result.add_performance_metric('interface_execution_time', execution_duration, 'seconds')
            
            # Update execution history and performance metrics
            self.execution_history.append(algorithm_result)
            
            # Limit execution history size for memory management
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-50:]  # Keep last 50 executions
            
            # Log algorithm execution completion with performance metrics
            self.logger.info(
                f"Algorithm execution completed: {simulation_id}, "
                f"success={algorithm_result.success}, time={execution_duration:.3f}s"
            )
            
            return algorithm_result
            
        except Exception as e:
            # Update failed execution statistics
            if self.performance_tracking_enabled:
                self.performance_metrics['total_executions'] += 1
                self.performance_metrics['failed_executions'] += 1
            
            self.logger.error(f"Algorithm execution failed: {simulation_id} - {e}", exc_info=True)
            
            # Generate comprehensive algorithm result even on failure
            if isinstance(e, (ValidationError, SimulationError)):
                # Re-raise known exceptions with context
                raise
            else:
                # Wrap unknown exceptions in SimulationError
                raise SimulationError(
                    message=f"Algorithm execution failed: {str(e)}",
                    simulation_id=simulation_id,
                    algorithm_name=self.algorithm_name,
                    simulation_context={
                        'error': str(e),
                        'stage': 'interface_execution',
                        'execution_config': execution_config
                    }
                )
    
    def validate_execution_result(
        self,
        result: AlgorithmResult,
        reference_metrics: Dict[str, float] = None,
        validate_correlation: bool = True
    ) -> ValidationResult:
        """
        Validate algorithm execution result against scientific computing standards, correlation thresholds, 
        and reproducibility requirements for quality assurance.
        
        Args:
            result: Algorithm execution result to validate
            reference_metrics: Reference metrics for correlation analysis
            validate_correlation: Whether to include correlation analysis
            
        Returns:
            ValidationResult: Result validation with correlation analysis and compliance assessment
        """
        # Initialize result validation
        validation_result = ValidationResult(
            validation_type="execution_result_validation",
            is_valid=True,
            validation_context=f"algorithm={self.algorithm_name}, simulation={result.simulation_id}"
        )
        
        try:
            # Validate result format and completeness
            if not isinstance(result, AlgorithmResult):
                validation_result.add_error("Result must be an AlgorithmResult instance", severity="CRITICAL")
                validation_result.is_valid = False
                return validation_result
            
            # Check basic result integrity
            if not result.algorithm_name:
                validation_result.add_error("Algorithm name missing from result", severity="HIGH")
                validation_result.is_valid = False
            
            if not result.simulation_id:
                validation_result.add_error("Simulation ID missing from result", severity="HIGH")
                validation_result.is_valid = False
            
            # Check performance metrics against thresholds
            if result.execution_time > DEFAULT_EXECUTION_TIMEOUT:
                validation_result.add_warning(
                    f"Execution time {result.execution_time:.3f}s exceeds timeout {DEFAULT_EXECUTION_TIMEOUT}s"
                )
            
            # Validate trajectory if present
            if result.trajectory is not None:
                if isinstance(result.trajectory, np.ndarray):
                    if result.trajectory.size == 0:
                        validation_result.add_warning("Trajectory is empty")
                    elif np.any(np.isnan(result.trajectory)):
                        validation_result.add_error("Trajectory contains NaN values", severity="HIGH")
                        validation_result.is_valid = False
                    elif np.any(np.isinf(result.trajectory)):
                        validation_result.add_error("Trajectory contains infinite values", severity="HIGH")
                        validation_result.is_valid = False
                else:
                    validation_result.add_warning("Trajectory is not a numpy array")
            
            # Perform correlation analysis against reference if provided
            if validate_correlation and reference_metrics:
                correlation_metrics = calculate_performance_metrics(
                    algorithm_result=result,
                    reference_metrics=reference_metrics,
                    include_correlation_analysis=True
                )
                
                overall_correlation = correlation_metrics.get('overall_correlation', 0.0)
                if overall_correlation < DEFAULT_CORRELATION_THRESHOLD:
                    validation_result.add_error(
                        f"Correlation {overall_correlation:.3f} below threshold {DEFAULT_CORRELATION_THRESHOLD}",
                        severity="HIGH"
                    )
                    validation_result.is_valid = False
                
                # Add correlation metrics to validation
                for metric_name, metric_value in correlation_metrics.items():
                    if 'correlation' in metric_name:
                        validation_result.add_metric(metric_name, metric_value)
            
            # Validate convergence and accuracy criteria
            if not result.success:
                validation_result.add_warning("Algorithm execution was not successful")
            
            if not result.converged and hasattr(result, 'iterations_completed'):
                if result.iterations_completed >= 10000:  # Default max iterations
                    validation_result.add_warning("Algorithm did not converge within iteration limits")
            
            # Check for excessive warnings
            if len(result.warnings) > 5:
                validation_result.add_warning(f"High number of warnings ({len(result.warnings)}) in result")
            
            # Add result validation metrics
            validation_result.add_metric("execution_time", result.execution_time)
            validation_result.add_metric("success_rate", float(result.success))
            validation_result.add_metric("convergence_rate", float(result.converged))
            validation_result.add_metric("warning_count", float(len(result.warnings)))
            
            # Generate result validation with recommendations
            if validation_result.is_valid:
                validation_result.add_recommendation("Algorithm result passed validation", priority="INFO")
            else:
                validation_result.add_recommendation(
                    "Address result validation issues for scientific computing compliance",
                    priority="HIGH"
                )
            
        except Exception as e:
            validation_result.add_error(f"Result validation failed: {str(e)}", severity="CRITICAL")
            validation_result.is_valid = False
            self.logger.error(f"Result validation error: {e}", exc_info=True)
        
        validation_result.finalize_validation()
        return validation_result
    
    def get_performance_summary(
        self,
        history_window: int = 10,
        include_trends: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for algorithm interface including execution statistics, 
        performance trends, and optimization recommendations.
        
        Args:
            history_window: Number of recent executions to include in analysis
            include_trends: Whether to include performance trends analysis
            
        Returns:
            Dict[str, Any]: Performance summary with statistics, trends, and optimization recommendations
        """
        try:
            # Extract performance metrics from execution history within window
            recent_executions = self.execution_history[-history_window:] if self.execution_history else []
            
            if not recent_executions:
                return {
                    'algorithm_name': self.algorithm_name,
                    'total_executions': 0,
                    'recent_executions': 0,
                    'performance_summary': 'No execution history available'
                }
            
            # Calculate execution statistics and success rates
            execution_times = [result.execution_time for result in recent_executions]
            success_rates = [1.0 if result.success else 0.0 for result in recent_executions]
            convergence_rates = [1.0 if result.converged else 0.0 for result in recent_executions]
            
            avg_execution_time = sum(execution_times) / len(execution_times)
            success_rate = sum(success_rates) / len(success_rates)
            convergence_rate = sum(convergence_rates) / len(convergence_rates)
            
            # Generate optimization recommendations based on performance data
            recommendations = []
            if avg_execution_time > 7.2:  # Target processing time
                recommendations.append("Average execution time exceeds target - consider parameter optimization")
            
            if success_rate < 0.8:
                recommendations.append("Success rate below 80% - review algorithm parameters and constraints")
            
            if convergence_rate < 0.7:
                recommendations.append("Convergence rate below 70% - consider adjusting convergence criteria")
            
            if not recommendations:
                recommendations.append("Algorithm performance is within acceptable ranges")
            
            # Compile performance summary
            summary = {
                'algorithm_name': self.algorithm_name,
                'interface_version': self.interface_version,
                'total_executions': self.execution_count,
                'recent_executions': len(recent_executions),
                'history_window': history_window,
                'creation_timestamp': self.creation_timestamp.isoformat(),
                'last_execution_timestamp': self.last_execution_timestamp.isoformat() if self.last_execution_timestamp else None,
                'performance_statistics': {
                    'average_execution_time': avg_execution_time,
                    'success_rate': success_rate,
                    'convergence_rate': convergence_rate,
                    'performance_compliance': avg_execution_time <= 7.2 and success_rate >= 0.8
                },
                'optimization_recommendations': recommendations
            }
            
            # Include performance trends if requested
            if include_trends and len(recent_executions) >= 3:
                # Calculate trends for execution time and success rate
                first_half = recent_executions[:len(recent_executions)//2]
                second_half = recent_executions[len(recent_executions)//2:]
                
                first_avg_time = sum(r.execution_time for r in first_half) / len(first_half)
                second_avg_time = sum(r.execution_time for r in second_half) / len(second_half)
                
                time_trend = 'improving' if second_avg_time < first_avg_time else 'degrading' if second_avg_time > first_avg_time else 'stable'
                
                summary['performance_trends'] = {
                    'execution_time_trend': time_trend,
                    'overall_trend': 'good' if success_rate > 0.8 and avg_execution_time <= 7.2 else 'needs_improvement'
                }
            
            # Include interface-specific metrics if performance tracking enabled
            if self.performance_tracking_enabled:
                summary['interface_metrics'] = self.performance_metrics.copy()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Performance summary generation failed: {e}", exc_info=True)
            return {
                'error': f"Performance summary failed: {str(e)}",
                'algorithm_name': self.algorithm_name,
                'timestamp': datetime.datetime.now().isoformat()
            }
    
    def reset_interface(
        self,
        preserve_configuration: bool = True,
        clear_performance_history: bool = False
    ) -> None:
        """
        Reset algorithm interface to initial state including clearing execution history, resetting performance 
        metrics, and reinitializing algorithm instance.
        
        Args:
            preserve_configuration: Whether to preserve interface configuration
            clear_performance_history: Whether to clear performance history
        """
        try:
            # Clear algorithm instance and execution history
            self.algorithm_instance = None
            self.execution_history.clear()
            
            # Reset performance metrics and execution counters
            self.execution_count = 0
            self.last_execution_timestamp = None
            
            if self.performance_tracking_enabled:
                if clear_performance_history:
                    self.performance_metrics = {
                        'total_executions': 0,
                        'successful_executions': 0,
                        'failed_executions': 0,
                        'average_execution_time': 0.0,
                        'last_execution_time': 0.0
                    }
            
            # Preserve interface configuration if preserve_configuration enabled
            if not preserve_configuration:
                self.interface_config = {}
                self.algorithm_metadata = {}
            
            # Reinitialize interface state and tracking
            self.creation_timestamp = datetime.datetime.now()
            
            # Log interface reset operation with preserved data information
            self.logger.info(f"Algorithm interface reset: preserve_config={preserve_configuration}, clear_history={clear_performance_history}")
            
        except Exception as e:
            self.logger.error(f"Interface reset failed: {e}", exc_info=True)
            raise SimulationError(
                message=f"Interface reset failed: {str(e)}",
                simulation_id="interface_reset",
                algorithm_name=self.algorithm_name,
                simulation_context={'error': str(e), 'stage': 'interface_reset'}
            )
    
    def export_interface_configuration(
        self,
        include_execution_history: bool = False,
        include_performance_data: bool = True,
        export_format: str = 'dict'
    ) -> Dict[str, Any]:
        """
        Export algorithm interface configuration including parameters, performance data, and execution history 
        for reproducibility and documentation.
        
        Args:
            include_execution_history: Whether to include execution history in export
            include_performance_data: Whether to include performance data
            export_format: Format for export (currently only 'dict' supported)
            
        Returns:
            Dict[str, Any]: Complete interface configuration with parameters, history, and performance data
        """
        try:
            # Export interface configuration and algorithm metadata
            configuration = {
                'algorithm_name': self.algorithm_name,
                'interface_version': self.interface_version,
                'creation_timestamp': self.creation_timestamp.isoformat(),
                'last_execution_timestamp': self.last_execution_timestamp.isoformat() if self.last_execution_timestamp else None,
                'execution_count': self.execution_count,
                'validation_enabled': self.validation_enabled,
                'performance_tracking_enabled': self.performance_tracking_enabled,
                'interface_config': self.interface_config.copy(),
                'algorithm_metadata': self.algorithm_metadata.copy(),
                'is_initialized': self.is_initialized,
                'export_metadata': {
                    'exported_at': datetime.datetime.now().isoformat(),
                    'export_format': export_format,
                    'export_version': ALGORITHM_INTERFACE_VERSION
                }
            }
            
            # Include execution history if requested
            if include_execution_history:
                execution_history_export = []
                for result in self.execution_history:
                    # Export result summary to avoid large trajectory data
                    history_entry = result.get_summary()
                    execution_history_export.append(history_entry)
                
                configuration['execution_history'] = execution_history_export
                configuration['export_metadata']['history_included'] = True
                configuration['export_metadata']['history_entries'] = len(execution_history_export)
            else:
                configuration['export_metadata']['history_included'] = False
            
            # Include performance data if requested
            if include_performance_data and self.performance_tracking_enabled:
                configuration['performance_data'] = self.performance_metrics.copy()
                configuration['export_metadata']['performance_data_included'] = True
            else:
                configuration['export_metadata']['performance_data_included'] = False
            
            # Add algorithm class information
            configuration['algorithm_class_info'] = {
                'class_name': self.algorithm_class.__name__,
                'base_class': 'BaseAlgorithm',
                'module': self.algorithm_class.__module__
            }
            
            return configuration
            
        except Exception as e:
            self.logger.error(f"Configuration export failed: {e}", exc_info=True)
            return {
                'error': f"Configuration export failed: {str(e)}",
                'algorithm_name': self.algorithm_name,
                'export_timestamp': datetime.datetime.now().isoformat()
            }
    
    def get_algorithm_metadata(
        self,
        include_performance_history: bool = False,
        include_validation_status: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive algorithm metadata including capabilities, performance characteristics, validation 
        requirements, and compatibility information.
        
        Args:
            include_performance_history: Whether to include performance history
            include_validation_status: Whether to include validation status
            
        Returns:
            Dict[str, Any]: Comprehensive algorithm metadata with performance and validation information
        """
        try:
            # Extract algorithm metadata from registry and interface configuration
            metadata = {
                'algorithm_name': self.algorithm_name,
                'interface_version': self.interface_version,
                'algorithm_metadata': self.algorithm_metadata.copy(),
                'creation_timestamp': self.creation_timestamp.isoformat(),
                'execution_count': self.execution_count,
                'is_initialized': self.is_initialized,
                'validation_enabled': self.validation_enabled,
                'performance_tracking_enabled': self.performance_tracking_enabled
            }
            
            # Include performance history if requested
            if include_performance_history and self.performance_tracking_enabled:
                metadata['performance_history'] = self.performance_metrics.copy()
                metadata['recent_performance'] = self.get_performance_summary(history_window=5, include_trends=False)
            
            # Include validation status if requested
            if include_validation_status and self.validation_enabled:
                try:
                    validation_result = validate_algorithm_interface(self.algorithm_class, strict_validation=False)
                    metadata['validation_status'] = {
                        'is_valid': validation_result.is_valid,
                        'validation_errors': validation_result.errors,
                        'validation_warnings': validation_result.warnings,
                        'last_validated': datetime.datetime.now().isoformat()
                    }
                except Exception as validation_error:
                    metadata['validation_status'] = {
                        'validation_error': str(validation_error),
                        'last_validated': datetime.datetime.now().isoformat()
                    }
            
            # Add algorithm capabilities and constraints
            metadata['algorithm_capabilities'] = {
                'supports_batch_processing': True,
                'supports_performance_tracking': self.performance_tracking_enabled,
                'supports_validation': self.validation_enabled,
                'execution_timeout': DEFAULT_EXECUTION_TIMEOUT,
                'correlation_threshold': DEFAULT_CORRELATION_THRESHOLD
            }
            
            # Include algorithm class information
            metadata['algorithm_class_metadata'] = {
                'class_name': self.algorithm_class.__name__,
                'module_name': self.algorithm_class.__module__,
                'base_classes': [cls.__name__ for cls in self.algorithm_class.__bases__]
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Metadata retrieval failed: {e}", exc_info=True)
            return {
                'error': f"Metadata retrieval failed: {str(e)}",
                'algorithm_name': self.algorithm_name,
                'timestamp': datetime.datetime.now().isoformat()
            }
    
    def _validate_parameter_dependencies(
        self,
        parameters: AlgorithmParameters,
        validation_result: ValidationResult
    ) -> None:
        """Validate cross-parameter dependencies and relationships."""
        # Algorithm-specific parameter dependency validation
        if self.algorithm_name.lower() == 'infotaxis':
            # Check infotaxis-specific dependencies
            if 'information_gain_threshold' in parameters.parameters and 'sensor_noise' in parameters.parameters:
                info_threshold = parameters.parameters['information_gain_threshold']
                sensor_noise = parameters.parameters['sensor_noise']
                
                if info_threshold < sensor_noise:
                    validation_result.add_warning(
                        "Information gain threshold should be greater than sensor noise for optimal performance"
                    )
        
        elif self.algorithm_name.lower() == 'casting':
            # Check casting-specific dependencies
            if 'casting_radius' in parameters.parameters and 'step_size' in parameters.parameters:
                casting_radius = parameters.parameters['casting_radius']
                step_size = parameters.parameters['step_size']
                
                if casting_radius < step_size * 2:
                    validation_result.add_warning(
                        "Casting radius should be at least twice the step size for effective search patterns"
                    )


class InterfaceManager:
    """
    Interface manager class providing centralized management of algorithm interfaces, lifecycle coordination, 
    resource optimization, and performance monitoring for the plume simulation system with thread-safe operations 
    and comprehensive statistics tracking.
    
    This class serves as the central coordinator for all algorithm interfaces with resource management, 
    performance monitoring, and optimization capabilities for scientific computing workflows.
    """
    
    def __init__(
        self,
        manager_config: Dict[str, Any],
        enable_caching: bool = True,
        max_interfaces: int = MAX_ALGORITHM_INSTANCES
    ):
        """
        Initialize interface manager with configuration, caching settings, and resource limits for centralized 
        algorithm interface management.
        
        Args:
            manager_config: Configuration dictionary for manager behavior
            enable_caching: Whether to enable interface caching for performance
            max_interfaces: Maximum number of interfaces to manage simultaneously
        """
        # Set manager configuration and resource limits
        self.manager_config = manager_config.copy()
        self.caching_enabled = enable_caching
        self.max_interfaces = max_interfaces
        
        # Initialize interface registry and cache
        self.interface_registry: Dict[str, AlgorithmInterface] = {}
        self.interface_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        
        # Create thread lock for manager safety
        self.manager_lock = threading.RLock()
        
        # Setup usage and performance statistics tracking
        self.usage_statistics: Dict[str, int] = {
            'interfaces_created': 0,
            'interfaces_accessed': 0,
            'interfaces_removed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.performance_statistics: Dict[str, float] = {
            'average_creation_time': 0.0,
            'average_access_time': 0.0,
            'total_execution_time': 0.0
        }
        
        # Configure logger for interface management operations
        self.logger = get_logger('interface_manager', 'ALGORITHM')
        
        # Set creation timestamp and initialize manager state
        self.creation_timestamp = datetime.datetime.now()
        
        self.logger.info(f"Interface manager initialized: max_interfaces={max_interfaces}, caching={enable_caching}")
    
    def create_interface(
        self,
        algorithm_name: str,
        interface_config: Dict[str, Any],
        cache_interface: bool = True
    ) -> AlgorithmInterface:
        """
        Create algorithm interface with configuration validation, resource management, and registration in 
        manager registry.
        
        Args:
            algorithm_name: Name of the algorithm to create interface for
            interface_config: Configuration for interface creation
            cache_interface: Whether to cache the interface for reuse
            
        Returns:
            AlgorithmInterface: Created and configured algorithm interface
        """
        creation_start_time = time.time()
        
        # Acquire manager lock for thread-safe interface creation
        with self.manager_lock:
            try:
                # Validate algorithm name and interface configuration
                if not algorithm_name or not isinstance(algorithm_name, str):
                    raise ValueError("Algorithm name must be a non-empty string")
                
                if not isinstance(interface_config, dict):
                    raise TypeError("Interface configuration must be a dictionary")
                
                # Check resource limits and interface count
                if len(self.interface_registry) >= self.max_interfaces:
                    # Clean up unused interfaces to make space
                    self._cleanup_unused_interfaces()
                    
                    if len(self.interface_registry) >= self.max_interfaces:
                        raise SimulationError(
                            message=f"Maximum interface limit reached: {self.max_interfaces}",
                            simulation_id="interface_creation",
                            algorithm_name=algorithm_name,
                            simulation_context={'max_interfaces': self.max_interfaces}
                        )
                
                # Create algorithm interface with validated configuration
                interface = create_algorithm_interface(
                    algorithm_name=algorithm_name,
                    interface_config=interface_config,
                    enable_validation=interface_config.get('enable_validation', True),
                    enable_performance_tracking=interface_config.get('enable_performance_tracking', True)
                )
                
                # Register interface in manager registry
                interface_key = f"{algorithm_name}_{len(self.interface_registry)}"
                self.interface_registry[interface_key] = interface
                
                # Cache interface if caching enabled and cache_interface is True
                if self.caching_enabled and cache_interface:
                    cache_key = f"{algorithm_name}_{uuid.uuid4().hex[:8]}"
                    self.interface_cache[cache_key] = interface
                
                # Update usage statistics and performance tracking
                self.usage_statistics['interfaces_created'] += 1
                
                creation_duration = time.time() - creation_start_time
                total_created = self.usage_statistics['interfaces_created']
                current_avg = self.performance_statistics['average_creation_time']
                self.performance_statistics['average_creation_time'] = (
                    (current_avg * (total_created - 1) + creation_duration) / total_created
                )
                
                # Log interface creation with configuration details
                self.logger.info(f"Algorithm interface created and registered: {algorithm_name} [{interface_key}]")
                
                # Create audit trail entry for interface creation
                create_audit_trail(
                    action='INTERFACE_CREATED_BY_MANAGER',
                    component='INTERFACE_MANAGER',
                    action_details={
                        'algorithm_name': algorithm_name,
                        'interface_key': interface_key,
                        'cached': cache_interface,
                        'creation_duration': creation_duration
                    },
                    user_context='SYSTEM'
                )
                
                return interface
                
            except Exception as e:
                self.logger.error(f"Interface creation failed for {algorithm_name}: {e}", exc_info=True)
                raise SimulationError(
                    message=f"Interface creation failed: {str(e)}",
                    simulation_id="interface_creation",
                    algorithm_name=algorithm_name,
                    simulation_context={'error': str(e), 'stage': 'manager_creation'}
                )
    
    def get_interface(
        self,
        algorithm_name: str,
        create_if_missing: bool = True,
        default_config: Dict[str, Any] = None
    ) -> Optional[AlgorithmInterface]:
        """
        Get algorithm interface from registry or cache with automatic creation if not found and dynamic 
        loading support.
        
        Args:
            algorithm_name: Name of the algorithm to retrieve interface for
            create_if_missing: Whether to create interface if not found
            default_config: Default configuration for interface creation
            
        Returns:
            Optional[AlgorithmInterface]: Algorithm interface or None if not found and creation disabled
        """
        access_start_time = time.time()
        
        with self.manager_lock:
            try:
                # Check interface cache for existing interface
                for cache_key, cached_interface in self.interface_cache.items():
                    if cached_interface.algorithm_name == algorithm_name:
                        self.usage_statistics['cache_hits'] += 1
                        self.usage_statistics['interfaces_accessed'] += 1
                        
                        # Update access time statistics
                        access_duration = time.time() - access_start_time
                        total_accessed = self.usage_statistics['interfaces_accessed']
                        current_avg = self.performance_statistics['average_access_time']
                        self.performance_statistics['average_access_time'] = (
                            (current_avg * (total_accessed - 1) + access_duration) / total_accessed
                        )
                        
                        return cached_interface
                
                # Search interface registry if not found in cache
                for interface_key, interface in self.interface_registry.items():
                    if interface.algorithm_name == algorithm_name:
                        self.usage_statistics['interfaces_accessed'] += 1
                        
                        # Cache interface if caching enabled
                        if self.caching_enabled:
                            cache_key = f"{algorithm_name}_{uuid.uuid4().hex[:8]}"
                            self.interface_cache[cache_key] = interface
                        
                        return interface
                
                # Record cache miss
                self.usage_statistics['cache_misses'] += 1
                
                # Create interface if not found and create_if_missing enabled
                if create_if_missing:
                    interface_config = default_config or {}
                    interface = self.create_interface(
                        algorithm_name=algorithm_name,
                        interface_config=interface_config,
                        cache_interface=True
                    )
                    
                    return interface
                
                # Log interface access for tracking
                self.logger.debug(f"Interface not found: {algorithm_name}")
                
                return None
                
            except Exception as e:
                self.logger.error(f"Interface retrieval failed for {algorithm_name}: {e}", exc_info=True)
                return None
    
    def remove_interface(
        self,
        algorithm_name: str,
        force_removal: bool = False
    ) -> bool:
        """
        Remove algorithm interface from registry and cache with cleanup and resource deallocation.
        
        Args:
            algorithm_name: Name of the algorithm to remove interface for
            force_removal: Whether to force removal without checking active executions
            
        Returns:
            bool: Success status of interface removal
        """
        # Acquire manager lock for thread-safe removal
        with self.manager_lock:
            try:
                # Validate interface exists in registry
                interfaces_to_remove = []
                for interface_key, interface in self.interface_registry.items():
                    if interface.algorithm_name == algorithm_name:
                        interfaces_to_remove.append((interface_key, interface))
                
                if not interfaces_to_remove:
                    self.logger.warning(f"Interface not found for removal: {algorithm_name}")
                    return False
                
                # Check for active executions unless force_removal enabled
                if not force_removal:
                    for interface_key, interface in interfaces_to_remove:
                        if interface.algorithm_instance is not None:
                            self.logger.warning(f"Interface has active instance: {algorithm_name}")
                            return False
                
                # Remove interface from registry and cache
                removed_count = 0
                for interface_key, interface in interfaces_to_remove:
                    # Remove from registry
                    if interface_key in self.interface_registry:
                        del self.interface_registry[interface_key]
                        removed_count += 1
                    
                    # Remove from cache
                    cache_keys_to_remove = []
                    for cache_key, cached_interface in self.interface_cache.items():
                        if cached_interface.algorithm_name == algorithm_name:
                            cache_keys_to_remove.append(cache_key)
                    
                    for cache_key in cache_keys_to_remove:
                        if cache_key in self.interface_cache:
                            del self.interface_cache[cache_key]
                
                # Update usage statistics and performance tracking
                self.usage_statistics['interfaces_removed'] += removed_count
                
                # Log interface removal with cleanup details
                self.logger.info(f"Interface removed: {algorithm_name} ({removed_count} instances)")
                
                # Create audit trail entry for interface removal
                create_audit_trail(
                    action='INTERFACE_REMOVED_BY_MANAGER',
                    component='INTERFACE_MANAGER',
                    action_details={
                        'algorithm_name': algorithm_name,
                        'instances_removed': removed_count,
                        'force_removal': force_removal
                    },
                    user_context='SYSTEM'
                )
                
                return removed_count > 0
                
            except Exception as e:
                self.logger.error(f"Interface removal failed for {algorithm_name}: {e}", exc_info=True)
                return False
    
    def get_manager_statistics(
        self,
        include_interface_details: bool = False,
        time_period: str = 'all'
    ) -> Dict[str, Any]:
        """
        Get comprehensive manager statistics including interface usage, performance metrics, and resource 
        utilization.
        
        Args:
            include_interface_details: Whether to include detailed interface information
            time_period: Time period for statistics analysis
            
        Returns:
            Dict[str, Any]: Manager statistics with usage patterns and performance analysis
        """
        with self.manager_lock:
            try:
                # Compile interface usage and access statistics
                statistics = {
                    'manager_creation_timestamp': self.creation_timestamp.isoformat(),
                    'statistics_timestamp': datetime.datetime.now().isoformat(),
                    'time_period': time_period,
                    'resource_limits': {
                        'max_interfaces': self.max_interfaces,
                        'caching_enabled': self.caching_enabled
                    },
                    'current_state': {
                        'registered_interfaces': len(self.interface_registry),
                        'cached_interfaces': len(self.interface_cache),
                        'resource_utilization': len(self.interface_registry) / self.max_interfaces
                    },
                    'usage_statistics': self.usage_statistics.copy(),
                    'performance_statistics': self.performance_statistics.copy()
                }
                
                # Calculate performance metrics and trends
                total_created = self.usage_statistics['interfaces_created']
                total_accessed = self.usage_statistics['interfaces_accessed']
                cache_hits = self.usage_statistics['cache_hits']
                cache_misses = self.usage_statistics['cache_misses']
                
                if total_accessed > 0:
                    statistics['derived_metrics'] = {
                        'cache_hit_rate': cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0,
                        'average_interfaces_per_algorithm': total_created / len(set(
                            interface.algorithm_name for interface in self.interface_registry.values()
                        )) if self.interface_registry else 0.0,
                        'resource_efficiency': statistics['current_state']['resource_utilization']
                    }
                else:
                    statistics['derived_metrics'] = {
                        'cache_hit_rate': 0.0,
                        'average_interfaces_per_algorithm': 0.0,
                        'resource_efficiency': 0.0
                    }
                
                # Include interface details if requested
                if include_interface_details:
                    interface_details = {}
                    for interface_key, interface in self.interface_registry.items():
                        interface_details[interface_key] = {
                            'algorithm_name': interface.algorithm_name,
                            'creation_timestamp': interface.creation_timestamp.isoformat(),
                            'execution_count': interface.execution_count,
                            'last_execution': interface.last_execution_timestamp.isoformat() if interface.last_execution_timestamp else None,
                            'is_initialized': interface.is_initialized,
                            'has_active_instance': interface.algorithm_instance is not None
                        }
                    
                    statistics['interface_details'] = interface_details
                
                # Generate optimization recommendations
                recommendations = []
                if statistics['derived_metrics']['cache_hit_rate'] < 0.5:
                    recommendations.append("Low cache hit rate - consider adjusting caching strategy")
                
                if statistics['current_state']['resource_utilization'] > 0.9:
                    recommendations.append("High resource utilization - consider increasing max_interfaces limit")
                
                if statistics['performance_statistics']['average_creation_time'] > 1.0:
                    recommendations.append("High interface creation time - review configuration complexity")
                
                if not recommendations:
                    recommendations.append("Manager performance is within acceptable ranges")
                
                statistics['optimization_recommendations'] = recommendations
                
                return statistics
                
            except Exception as e:
                self.logger.error(f"Manager statistics retrieval failed: {e}", exc_info=True)
                return {
                    'error': f"Statistics retrieval failed: {str(e)}",
                    'timestamp': datetime.datetime.now().isoformat()
                }
    
    def cleanup_interfaces(
        self,
        idle_threshold_hours: float = 1.0,
        preserve_statistics: bool = True
    ) -> Dict[str, int]:
        """
        Cleanup unused interfaces and optimize resource usage with selective cleanup and statistics preservation.
        
        Args:
            idle_threshold_hours: Hours of inactivity before interface is considered for cleanup
            preserve_statistics: Whether to preserve manager statistics
            
        Returns:
            Dict[str, int]: Cleanup statistics with removed interfaces and preserved data
        """
        with self.manager_lock:
            try:
                # Identify idle interfaces based on idle_threshold_hours
                current_time = datetime.datetime.now()
                idle_threshold = datetime.timedelta(hours=idle_threshold_hours)
                
                interfaces_to_cleanup = []
                for interface_key, interface in self.interface_registry.items():
                    last_activity = interface.last_execution_timestamp or interface.creation_timestamp
                    time_since_activity = current_time - last_activity
                    
                    if time_since_activity > idle_threshold and interface.algorithm_instance is None:
                        interfaces_to_cleanup.append((interface_key, interface))
                
                # Preserve statistics if preserve_statistics enabled
                preserved_data = {}
                if preserve_statistics:
                    preserved_data = {
                        'usage_statistics': self.usage_statistics.copy(),
                        'performance_statistics': self.performance_statistics.copy()
                    }
                
                # Remove idle interfaces from registry and cache
                removed_interfaces = 0
                for interface_key, interface in interfaces_to_cleanup:
                    if interface_key in self.interface_registry:
                        del self.interface_registry[interface_key]
                        removed_interfaces += 1
                    
                    # Remove from cache
                    cache_keys_to_remove = []
                    for cache_key, cached_interface in self.interface_cache.items():
                        if cached_interface.algorithm_name == interface.algorithm_name:
                            cache_keys_to_remove.append(cache_key)
                    
                    for cache_key in cache_keys_to_remove:
                        if cache_key in self.interface_cache:
                            del self.interface_cache[cache_key]
                
                # Update manager statistics and resource tracking
                self.usage_statistics['interfaces_removed'] += removed_interfaces
                
                # Generate cleanup summary
                cleanup_stats = {
                    'interfaces_evaluated': len(self.interface_registry) + removed_interfaces,
                    'interfaces_removed': removed_interfaces,
                    'interfaces_preserved': len(self.interface_registry),
                    'idle_threshold_hours': idle_threshold_hours,
                    'statistics_preserved': preserve_statistics,
                    'cleanup_timestamp': current_time.isoformat(),
                    'resource_utilization_after': len(self.interface_registry) / self.max_interfaces
                }
                
                if preserve_statistics:
                    cleanup_stats['preserved_data'] = preserved_data
                
                # Log cleanup operation with statistics
                self.logger.info(f"Interface cleanup completed: {removed_interfaces} interfaces removed")
                
                # Create audit trail entry for cleanup operation
                create_audit_trail(
                    action='INTERFACES_CLEANED_UP',
                    component='INTERFACE_MANAGER',
                    action_details=cleanup_stats,
                    user_context='SYSTEM'
                )
                
                return cleanup_stats
                
            except Exception as e:
                self.logger.error(f"Interface cleanup failed: {e}", exc_info=True)
                return {
                    'error': f"Cleanup failed: {str(e)}",
                    'timestamp': datetime.datetime.now().isoformat()
                }
    
    def _cleanup_unused_interfaces(self) -> None:
        """Internal method to cleanup unused interfaces when resource limits are reached."""
        # Find interfaces without active instances
        unused_interfaces = []
        for interface_key, interface in self.interface_registry.items():
            if interface.algorithm_instance is None:
                unused_interfaces.append(interface_key)
        
        # Remove oldest unused interfaces
        if unused_interfaces:
            oldest_interface_key = min(
                unused_interfaces,
                key=lambda k: self.interface_registry[k].creation_timestamp
            )
            
            interface = self.interface_registry[oldest_interface_key]
            del self.interface_registry[oldest_interface_key]
            
            self.logger.debug(f"Cleaned up unused interface: {interface.algorithm_name}")


# Helper functions for algorithm interface implementation

def _convert_parameter_type(param_name: str, param_value: Any, algorithm_name: str) -> Any:
    """Convert parameter value to appropriate type based on parameter name and algorithm."""
    # Type conversion mappings for common algorithm parameters
    type_mappings = {
        'convergence_tolerance': float,
        'max_iterations': int,
        'step_size': float,
        'learning_rate': float,
        'sensor_noise': float,
        'information_gain_threshold': float,
        'casting_radius': float
    }
    
    if param_name in type_mappings:
        try:
            return type_mappings[param_name](param_value)
        except (ValueError, TypeError):
            # Return original value if conversion fails
            return param_value
    
    return param_value


def _get_default_algorithm_constraints(algorithm_name: str) -> Dict[str, Dict[str, Any]]:
    """Get default parameter constraints for specific algorithm types."""
    default_constraints = {
        'convergence_tolerance': {'min': 1e-12, 'max': 1e-3},
        'max_iterations': {'min': 1, 'max': 100000},
        'step_size': {'min': 1e-6, 'max': 1.0},
        'learning_rate': {'min': 1e-6, 'max': 1.0}
    }
    
    # Algorithm-specific constraints
    if algorithm_name.lower() == 'infotaxis':
        default_constraints.update({
            'information_gain_threshold': {'min': 0.0, 'max': 10.0},
            'sensor_noise': {'min': 0.0, 'max': 1.0}
        })
    elif algorithm_name.lower() == 'casting':
        default_constraints.update({
            'casting_radius': {'min': 1.0, 'max': 100.0},
            'cast_angle': {'min': 0.0, 'max': 360.0}
        })
    
    return default_constraints


def _validate_parameter_compatibility(
    interfaces: Dict[str, AlgorithmInterface],
    requirements: Dict[str, Any],
    validation_result: ValidationResult
) -> float:
    """Validate parameter compatibility across algorithm interfaces."""
    compatibility_score = 1.0
    
    # Check for compatible parameter ranges across algorithms
    all_parameters = {}
    for algorithm_name, interface in interfaces.items():
        if hasattr(interface, 'algorithm_metadata'):
            metadata = interface.algorithm_metadata
            if 'parameters' in metadata:
                all_parameters[algorithm_name] = metadata['parameters']
    
    # Simple compatibility check - algorithms with similar parameter sets are more compatible
    if len(all_parameters) > 1:
        parameter_sets = list(all_parameters.values())
        common_params = set(parameter_sets[0].keys()) if parameter_sets else set()
        
        for param_set in parameter_sets[1:]:
            common_params &= set(param_set.keys())
        
        total_unique_params = set()
        for param_set in parameter_sets:
            total_unique_params.update(param_set.keys())
        
        if total_unique_params:
            compatibility_score = len(common_params) / len(total_unique_params)
    
    return compatibility_score


def _validate_execution_context_compatibility(
    interfaces: Dict[str, AlgorithmInterface],
    requirements: Dict[str, Any],
    validation_result: ValidationResult
) -> float:
    """Validate execution context compatibility across algorithm interfaces."""
    # All interfaces should support similar execution contexts
    compatibility_score = 1.0
    
    # Check validation and performance tracking consistency
    validation_states = [interface.validation_enabled for interface in interfaces.values()]
    tracking_states = [interface.performance_tracking_enabled for interface in interfaces.values()]
    
    # Penalize inconsistent states
    if not all(validation_states) and any(validation_states):
        compatibility_score -= 0.2
        validation_result.add_warning("Inconsistent validation states across interfaces")
    
    if not all(tracking_states) and any(tracking_states):
        compatibility_score -= 0.1
        validation_result.add_warning("Inconsistent performance tracking states across interfaces")
    
    return max(0.0, compatibility_score)


def _validate_performance_compatibility(
    interfaces: Dict[str, AlgorithmInterface],
    requirements: Dict[str, Any],
    validation_result: ValidationResult
) -> float:
    """Validate performance characteristics compatibility across algorithm interfaces."""
    # Check if all interfaces meet performance requirements
    compatibility_score = 1.0
    
    performance_issues = 0
    for algorithm_name, interface in interfaces.items():
        if interface.performance_tracking_enabled and interface.performance_metrics:
            avg_time = interface.performance_metrics.get('average_execution_time', 0.0)
            if avg_time > 7.2:  # Target processing time
                performance_issues += 1
                validation_result.add_warning(f"Algorithm {algorithm_name} exceeds target execution time")
    
    if performance_issues > 0:
        compatibility_score -= (performance_issues / len(interfaces)) * 0.5
    
    return max(0.0, compatibility_score)


def _apply_strict_compatibility_validation(
    interfaces: Dict[str, AlgorithmInterface],
    requirements: Dict[str, Any],
    validation_result: ValidationResult
) -> None:
    """Apply strict compatibility validation criteria."""
    # Check for identical interface versions
    versions = [interface.interface_version for interface in interfaces.values()]
    if len(set(versions)) > 1:
        validation_result.add_warning("Mixed interface versions detected")
    
    # Check for algorithm class compatibility
    algorithm_classes = [interface.algorithm_class for interface in interfaces.values()]
    base_classes = [set(cls.__bases__) for cls in algorithm_classes]
    
    # All should inherit from BaseAlgorithm
    for i, base_set in enumerate(base_classes):
        if not any(issubclass(base, BaseAlgorithm) for base in base_set):
            algorithm_name = list(interfaces.keys())[i]
            validation_result.add_error(f"Algorithm {algorithm_name} does not inherit from BaseAlgorithm")


def _calculate_compatibility_score(
    parameter_compatibility: float,
    context_compatibility: float,
    performance_compatibility: float
) -> float:
    """Calculate overall compatibility score from individual compatibility metrics."""
    # Weighted average of compatibility scores
    weights = {
        'parameter': 0.4,
        'context': 0.3,
        'performance': 0.3
    }
    
    total_score = (
        parameter_compatibility * weights['parameter'] +
        context_compatibility * weights['context'] +
        performance_compatibility * weights['performance']
    )
    
    return min(1.0, max(0.0, total_score))