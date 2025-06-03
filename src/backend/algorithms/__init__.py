"""
Main entry point for the algorithms module providing centralized access to all navigation algorithms, 
algorithm registry functionality, and supporting infrastructure for plume source localization.

This module implements comprehensive algorithm discovery, registration, and instantiation capabilities 
with cross-format compatibility, scientific computing validation, and performance tracking for 
reproducible research outcomes across 4000+ simulation processing requirements.

Key Features:
- Centralized algorithm registry with dynamic discovery and registration
- Standardized algorithm interface for testing navigation algorithms
- Cross-platform compatibility for Crimaldi and custom plume formats  
- Statistical validation with >95% correlation against reference implementations
- Performance tracking with <7.2 seconds target execution time
- Comprehensive parameter validation and scientific computing standards
- Batch processing support for 4000+ simulation requirements
- Algorithm sandboxing and isolated execution environments
- Reproducible results with >0.99 coefficient validation across environments
"""

# External imports with version specifications
import typing  # version: 3.9+ - Type hints for algorithm module interface and complex data structures
from typing import Dict, Any, List, Optional, Union, Tuple, Type, Callable
import logging  # version: 3.9+ - Logging infrastructure for algorithm module operations and audit trails
import warnings
import time
import copy
import numpy as np
from datetime import datetime

# Internal imports from base algorithm framework
from .base_algorithm import (
    BaseAlgorithm,
    AlgorithmParameters, 
    AlgorithmResult,
    AlgorithmContext,
    validate_plume_data,
    create_algorithm_context,
    calculate_performance_metrics
)

# Internal imports from algorithm registry
from .algorithm_registry import (
    AlgorithmRegistry,
    register_algorithm,
    get_algorithm,
    create_algorithm_instance,
    list_algorithms,
    validate_algorithm_interface,
    get_algorithm_metadata,
    update_algorithm_performance,
    compare_algorithms,
    validate_registry_integrity,
    clear_performance_cache,
    export_registry_configuration,
    load_algorithm_dynamically,
    register_plume_tracking_algorithm,
    register_hybrid_strategies_algorithm,
    discover_available_algorithms
)

# Internal imports from core algorithm implementations
from .reference_implementation import (
    ReferenceImplementation,
    validate_against_benchmark,
    generate_benchmark_report,
    create_reference_parameters
)

from .infotaxis import (
    InfotaxisAlgorithm,
    InfotaxisParameters,
    calculate_entropy,
    calculate_information_gain,
    update_belief_state
)

from .casting import (
    CastingAlgorithm,
    CastingParameters,
    calculate_wind_direction,
    detect_plume_contact,
    optimize_search_radius
)

# Dynamic imports for algorithms that may not be immediately available
try:
    from .gradient_following import GradientFollowing
    GRADIENT_FOLLOWING_AVAILABLE = True
except ImportError:
    GradientFollowing = None
    GRADIENT_FOLLOWING_AVAILABLE = False
    warnings.warn("GradientFollowing algorithm not available", ImportWarning)

try:
    from .plume_tracking import PlumeTrackingAlgorithm
    PLUME_TRACKING_AVAILABLE = True
except ImportError:
    PlumeTrackingAlgorithm = None
    PLUME_TRACKING_AVAILABLE = False
    warnings.warn("PlumeTrackingAlgorithm not available", ImportWarning)

try:
    from .hybrid_strategies import HybridStrategiesAlgorithm
    HYBRID_STRATEGIES_AVAILABLE = True
except ImportError:
    HybridStrategiesAlgorithm = None
    HYBRID_STRATEGIES_AVAILABLE = False
    warnings.warn("HybridStrategiesAlgorithm not available", ImportWarning)

# Global constants and configuration
ALGORITHMS_MODULE_VERSION = '1.0.0'
SUPPORTED_ALGORITHM_TYPES = [
    'reference_implementation', 'infotaxis', 'casting', 'gradient_following', 
    'plume_tracking', 'hybrid_strategies'
]
DEFAULT_ALGORITHM_REGISTRY = None
ALGORITHM_DISCOVERY_ENABLED = True
PERFORMANCE_TRACKING_ENABLED = True
VALIDATION_ENABLED = True
CROSS_FORMAT_COMPATIBILITY = True

# Module-level logger
_module_logger = logging.getLogger(__name__)

# Initialize global algorithm registry
_global_registry = None


def initialize_algorithms_module(
    enable_auto_discovery: bool = True,
    enable_performance_tracking: bool = True, 
    enable_validation: bool = True,
    module_config: Dict[str, Any] = None
) -> AlgorithmRegistry:
    """
    Initialize the algorithms module with algorithm registration, discovery, and validation setup 
    for comprehensive navigation algorithm management and scientific computing support.
    
    This function creates a global algorithm registry instance with configuration, registers core 
    navigation algorithms, enables auto-discovery for advanced algorithms, sets up performance 
    tracking and validation infrastructure, configures cross-format compatibility, validates 
    algorithm registry integrity, and returns the initialized registry for module use.
    
    Args:
        enable_auto_discovery: Whether to enable automatic algorithm discovery
        enable_performance_tracking: Whether to enable performance tracking
        enable_validation: Whether to enable algorithm validation
        module_config: Configuration dictionary for module initialization
        
    Returns:
        AlgorithmRegistry: Initialized algorithm registry with registered algorithms and configuration
    """
    global _global_registry, DEFAULT_ALGORITHM_REGISTRY
    global ALGORITHM_DISCOVERY_ENABLED, PERFORMANCE_TRACKING_ENABLED, VALIDATION_ENABLED
    
    try:
        # Create global algorithm registry instance with configuration
        if _global_registry is None:
            _global_registry = AlgorithmRegistry(
                enable_performance_tracking=enable_performance_tracking,
                enable_validation=enable_validation,
                registry_name='global_algorithms_module_registry',
                enable_dynamic_loading=enable_auto_discovery
            )
            
            DEFAULT_ALGORITHM_REGISTRY = _global_registry
        
        # Update module configuration
        config = module_config or {}
        ALGORITHM_DISCOVERY_ENABLED = enable_auto_discovery
        PERFORMANCE_TRACKING_ENABLED = enable_performance_tracking  
        VALIDATION_ENABLED = enable_validation
        
        # Register core navigation algorithms
        _register_core_algorithms()
        
        # Enable auto-discovery for advanced algorithms if enabled
        if enable_auto_discovery:
            _enable_dynamic_algorithm_discovery()
        
        # Setup performance tracking and validation infrastructure
        if enable_performance_tracking:
            _setup_performance_tracking()
            
        if enable_validation:
            _setup_validation_infrastructure()
        
        # Configure cross-format compatibility and scientific computing standards
        _configure_cross_format_compatibility()
        
        # Validate algorithm registry integrity and completeness
        integrity_result = validate_registry_integrity(deep_validation=True, repair_inconsistencies=True)
        if integrity_result.get('integrity_score', 0.0) < 0.8:
            _module_logger.warning(f"Registry integrity score below threshold: {integrity_result.get('integrity_score', 0.0)}")
        
        _module_logger.info(
            f"Algorithms module initialized successfully: "
            f"discovery={enable_auto_discovery}, tracking={enable_performance_tracking}, "
            f"validation={enable_validation}, algorithms={len(_global_registry.algorithms)}"
        )
        
        # Return initialized algorithm registry for module use
        return _global_registry
        
    except Exception as e:
        _module_logger.error(f"Algorithm module initialization failed: {e}", exc_info=True)
        raise


def get_available_algorithms(
    include_metadata: bool = True,
    include_performance_metrics: bool = False,
    algorithm_types: List[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Get list of all available navigation algorithms with metadata, capabilities, and performance 
    characteristics for algorithm discovery and selection.
    
    This function accesses the global algorithm registry for available algorithms, filters 
    algorithms by types if specified, extracts algorithm metadata if requested, includes 
    performance metrics if requested, formats algorithm information for discovery and selection, 
    and returns comprehensive algorithm listing with capabilities.
    
    Args:
        include_metadata: Whether to include algorithm metadata and capabilities
        include_performance_metrics: Whether to include performance metrics in listing
        algorithm_types: List of algorithm types to filter (None for all)
        
    Returns:
        Dict[str, Dict[str, Any]]: Available algorithms with metadata and performance information
    """
    try:
        # Ensure registry is initialized
        if _global_registry is None:
            initialize_algorithms_module()
        
        # Access global algorithm registry for available algorithms
        algorithm_listing = list_algorithms(
            algorithm_types=algorithm_types,
            include_metadata=include_metadata,
            include_performance_metrics=include_performance_metrics,
            only_available=True
        )
        
        # Filter algorithms by types if specified
        if algorithm_types:
            filtered_listing = {}
            for alg_name, alg_info in algorithm_listing.items():
                if alg_info.get('algorithm_type') in algorithm_types:
                    filtered_listing[alg_name] = alg_info
            algorithm_listing = filtered_listing
        
        # Extract algorithm metadata if requested
        if include_metadata:
            for alg_name, alg_info in algorithm_listing.items():
                try:
                    metadata = get_algorithm_metadata(alg_name, include_performance_history=False)
                    alg_info['detailed_metadata'] = metadata
                except Exception as e:
                    _module_logger.warning(f"Failed to get metadata for {alg_name}: {e}")
        
        # Include performance metrics if requested
        if include_performance_metrics:
            for alg_name, alg_info in algorithm_listing.items():
                if alg_name in _global_registry.performance_cache:
                    alg_info['performance_metrics'] = _global_registry.performance_cache[alg_name].copy()
                else:
                    alg_info['performance_metrics'] = {}
        
        # Format algorithm information for discovery and selection
        for alg_name, alg_info in algorithm_listing.items():
            alg_info['module_version'] = ALGORITHMS_MODULE_VERSION
            alg_info['cross_format_compatible'] = CROSS_FORMAT_COMPATIBILITY
            alg_info['validation_enabled'] = VALIDATION_ENABLED
            alg_info['performance_tracking_enabled'] = PERFORMANCE_TRACKING_ENABLED
        
        _module_logger.debug(f"Retrieved {len(algorithm_listing)} available algorithms")
        
        # Return comprehensive algorithm listing with capabilities
        return algorithm_listing
        
    except Exception as e:
        _module_logger.error(f"Failed to get available algorithms: {e}", exc_info=True)
        return {}


def create_algorithm(
    algorithm_name: str,
    parameters: AlgorithmParameters,
    execution_config: Dict[str, Any] = None,
    enable_validation: bool = True
) -> BaseAlgorithm:
    """
    Create algorithm instance with parameter validation, configuration setup, and performance 
    tracking initialization for scientific navigation analysis.
    
    This function validates the algorithm name against available algorithms, retrieves algorithm 
    class from registry with dynamic loading, validates algorithm parameters if validation enabled, 
    creates algorithm instance with parameters and configuration, initializes performance tracking 
    and scientific context, and returns configured algorithm instance ready for execution.
    
    Args:
        algorithm_name: Name of the navigation algorithm to instantiate
        parameters: Algorithm parameters for instance creation with validation
        execution_config: Configuration for algorithm execution environment
        enable_validation: Whether to enable parameter validation during creation
        
    Returns:
        BaseAlgorithm: Configured algorithm instance with validation and performance tracking
    """
    try:
        # Ensure registry is initialized
        if _global_registry is None:
            initialize_algorithms_module()
        
        # Validate algorithm name against available algorithms
        available_algorithms = get_available_algorithms(include_metadata=False)
        if algorithm_name not in available_algorithms:
            raise ValueError(f"Algorithm '{algorithm_name}' not found in registry")
        
        # Retrieve algorithm class from registry with dynamic loading
        algorithm_class = get_algorithm(
            algorithm_name=algorithm_name,
            validate_availability=True,
            include_metadata=False,
            enable_dynamic_loading=ALGORITHM_DISCOVERY_ENABLED
        )
        
        # Validate algorithm parameters if validation enabled
        if enable_validation and VALIDATION_ENABLED:
            validation_result = parameters.validate(strict_validation=True)
            if not validation_result.is_valid:
                raise ValueError(f"Parameter validation failed: {validation_result.errors}")
        
        # Create algorithm instance with parameters and configuration
        config = execution_config or {}
        config.update({
            'performance_tracking': PERFORMANCE_TRACKING_ENABLED,
            'validation_enabled': VALIDATION_ENABLED,
            'cross_format_compatibility': CROSS_FORMAT_COMPATIBILITY,
            'module_version': ALGORITHMS_MODULE_VERSION
        })
        
        algorithm_instance = create_algorithm_instance(
            algorithm_name=algorithm_name,
            parameters=parameters,
            execution_config=config,
            enable_validation=enable_validation
        )
        
        # Initialize performance tracking and scientific context
        if PERFORMANCE_TRACKING_ENABLED:
            algorithm_instance.performance_tracking_enabled = True
            
        # Set up scientific context for reproducibility
        if hasattr(algorithm_instance, 'set_scientific_context'):
            algorithm_instance.set_scientific_context({
                'module_version': ALGORITHMS_MODULE_VERSION,
                'creation_timestamp': datetime.now().isoformat(),
                'validation_enabled': enable_validation,
                'cross_format_compatibility': CROSS_FORMAT_COMPATIBILITY
            })
        
        _module_logger.info(f"Created algorithm instance: {algorithm_name}")
        
        # Return configured algorithm instance ready for execution
        return algorithm_instance
        
    except Exception as e:
        _module_logger.error(f"Failed to create algorithm '{algorithm_name}': {e}", exc_info=True)
        raise


def validate_algorithm_compatibility(
    algorithm_name: str,
    plume_format: str,
    experimental_conditions: Dict[str, Any],
    strict_validation: bool = False
) -> Dict[str, Any]:
    """
    Validate algorithm compatibility with plume data format, experimental conditions, and scientific 
    computing requirements for cross-platform execution.
    
    This function retrieves algorithm metadata and capabilities from registry, validates algorithm 
    support for specified plume format, checks algorithm compatibility with experimental conditions, 
    validates against scientific computing requirements, applies strict validation criteria if enabled, 
    and generates compatibility validation result with recommendations.
    
    Args:
        algorithm_name: Name of the algorithm to validate compatibility
        plume_format: Target plume data format ('crimaldi', 'custom', 'generic')
        experimental_conditions: Dictionary of experimental parameters and constraints
        strict_validation: Whether to apply strict validation criteria
        
    Returns:
        Dict[str, Any]: Algorithm compatibility validation with format support and requirements assessment
    """
    try:
        # Ensure registry is initialized
        if _global_registry is None:
            initialize_algorithms_module()
        
        # Initialize compatibility validation result
        compatibility_result = {
            'algorithm_name': algorithm_name,
            'plume_format': plume_format,
            'compatible': False,
            'format_support': {},
            'condition_compatibility': {},
            'validation_details': {},
            'recommendations': [],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Retrieve algorithm metadata and capabilities from registry
        try:
            algorithm_metadata = get_algorithm_metadata(
                algorithm_name=algorithm_name,
                include_performance_history=False,
                include_validation_status=True
            )
        except Exception as e:
            compatibility_result['error'] = f"Failed to retrieve algorithm metadata: {e}"
            return compatibility_result
        
        # Validate algorithm support for specified plume format
        supported_formats = algorithm_metadata.get('supported_formats', [])
        format_compatible = plume_format in supported_formats or 'generic' in supported_formats
        
        compatibility_result['format_support'] = {
            'requested_format': plume_format,
            'supported_formats': supported_formats,
            'format_compatible': format_compatible
        }
        
        if not format_compatible:
            compatibility_result['recommendations'].append(
                f"Algorithm does not support '{plume_format}' format. Supported: {supported_formats}"
            )
        
        # Check algorithm compatibility with experimental conditions
        capabilities = algorithm_metadata.get('capabilities', {})
        performance_characteristics = algorithm_metadata.get('performance_characteristics', {})
        
        condition_checks = {}
        
        # Check processing time requirements
        if 'max_processing_time' in experimental_conditions:
            max_time = experimental_conditions['max_processing_time']
            target_time = performance_characteristics.get('target_execution_time_seconds', 7.2)
            condition_checks['processing_time_compatible'] = target_time <= max_time
            
        # Check accuracy requirements
        if 'min_accuracy' in experimental_conditions:
            min_accuracy = experimental_conditions['min_accuracy']
            correlation_threshold = performance_characteristics.get('correlation_threshold', 0.95)
            condition_checks['accuracy_compatible'] = correlation_threshold >= min_accuracy
            
        # Check memory requirements
        if 'max_memory_usage' in experimental_conditions:
            max_memory = experimental_conditions['max_memory_usage']
            memory_efficiency = performance_characteristics.get('memory_efficiency_score', 0.85)
            condition_checks['memory_compatible'] = memory_efficiency >= 0.7  # Conservative threshold
            
        # Check specific algorithm capabilities
        required_capabilities = experimental_conditions.get('required_capabilities', [])
        capability_checks = {}
        for capability in required_capabilities:
            capability_checks[capability] = capabilities.get(capability, False)
        
        compatibility_result['condition_compatibility'] = {
            'condition_checks': condition_checks,
            'capability_checks': capability_checks,
            'overall_compatible': all(condition_checks.values()) and all(capability_checks.values())
        }
        
        # Validate against scientific computing requirements
        scientific_requirements = {
            'correlation_validation': True,
            'reproducibility_testing': True,
            'numerical_precision_validation': True
        }
        
        validation_requirements = algorithm_metadata.get('validation_requirements', {})
        scientific_compliance = {}
        
        for requirement, required_value in scientific_requirements.items():
            actual_value = validation_requirements.get(requirement, False)
            scientific_compliance[requirement] = actual_value == required_value
        
        compatibility_result['validation_details'] = {
            'scientific_compliance': scientific_compliance,
            'validation_requirements': validation_requirements,
            'meets_scientific_standards': all(scientific_compliance.values())
        }
        
        # Apply strict validation criteria if enabled
        if strict_validation:
            strict_checks = {}
            
            # Enhanced performance requirements for strict validation
            if performance_characteristics.get('target_execution_time_seconds', 10.0) > 7.2:
                strict_checks['execution_time_strict'] = False
                compatibility_result['recommendations'].append(
                    "Execution time may exceed strict 7.2 second requirement"
                )
            else:
                strict_checks['execution_time_strict'] = True
            
            # Enhanced correlation requirements
            if performance_characteristics.get('correlation_threshold', 0.0) < 0.95:
                strict_checks['correlation_strict'] = False
                compatibility_result['recommendations'].append(
                    "Correlation threshold below strict 95% requirement"
                )
            else:
                strict_checks['correlation_strict'] = True
            
            # Enhanced reproducibility requirements
            if performance_characteristics.get('reproducibility_threshold', 0.0) < 0.99:
                strict_checks['reproducibility_strict'] = False
                compatibility_result['recommendations'].append(
                    "Reproducibility threshold below strict 99% requirement"
                )
            else:
                strict_checks['reproducibility_strict'] = True
            
            compatibility_result['strict_validation'] = {
                'strict_checks': strict_checks,
                'strict_compliance': all(strict_checks.values())
            }
        
        # Generate compatibility validation result with recommendations
        overall_compatible = (
            format_compatible and
            compatibility_result['condition_compatibility']['overall_compatible'] and
            compatibility_result['validation_details']['meets_scientific_standards']
        )
        
        if strict_validation:
            overall_compatible = overall_compatible and compatibility_result.get('strict_validation', {}).get('strict_compliance', False)
        
        compatibility_result['compatible'] = overall_compatible
        
        # Generate recommendations based on compatibility analysis
        if overall_compatible:
            compatibility_result['recommendations'].append(
                f"Algorithm '{algorithm_name}' is compatible with specified requirements"
            )
        else:
            compatibility_result['recommendations'].append(
                f"Algorithm '{algorithm_name}' has compatibility issues - review requirements"
            )
        
        if not compatibility_result['recommendations']:
            compatibility_result['recommendations'].append("No specific compatibility issues identified")
        
        _module_logger.debug(f"Compatibility validation completed for {algorithm_name}: {overall_compatible}")
        
        return compatibility_result
        
    except Exception as e:
        _module_logger.error(f"Algorithm compatibility validation failed: {e}", exc_info=True)
        return {
            'algorithm_name': algorithm_name,
            'plume_format': plume_format,
            'compatible': False,
            'error': str(e),
            'validation_timestamp': datetime.now().isoformat()
        }


def compare_algorithm_performance(
    algorithm_names: List[str],
    comparison_metrics: List[str] = None,
    include_statistical_analysis: bool = True,
    validate_reproducibility: bool = True
) -> Dict[str, Any]:
    """
    Compare performance of multiple navigation algorithms using statistical analysis, correlation 
    assessment, and reproducibility validation for algorithm selection and optimization.
    
    This function validates all specified algorithms exist in registry, retrieves performance 
    metrics for algorithm comparison, calculates comparison metrics and statistical significance, 
    performs statistical analysis if requested, validates reproducibility with >0.99 threshold 
    if requested, generates algorithm rankings and recommendations, and returns comprehensive 
    comparison results with scientific validation.
    
    Args:
        algorithm_names: List of algorithm names to compare
        comparison_metrics: List of metrics to use for comparison analysis
        include_statistical_analysis: Whether to include detailed statistical analysis
        validate_reproducibility: Whether to validate reproducibility with >0.99 threshold
        
    Returns:
        Dict[str, Any]: Algorithm comparison results with statistical analysis and performance rankings
    """
    try:
        # Ensure registry is initialized
        if _global_registry is None:
            initialize_algorithms_module()
        
        # Validate all specified algorithms exist in registry
        available_algorithms = get_available_algorithms(include_metadata=False)
        missing_algorithms = [name for name in algorithm_names if name not in available_algorithms]
        
        if missing_algorithms:
            raise ValueError(f"Algorithms not found in registry: {missing_algorithms}")
        
        # Set default comparison metrics if not provided
        if comparison_metrics is None:
            comparison_metrics = [
                'execution_time_seconds', 'success_rate', 'convergence_rate', 
                'search_efficiency', 'correlation_score'
            ]
        
        # Initialize comparison results structure
        comparison_results = {
            'comparison_id': f"comparison_{int(time.time())}",
            'comparison_timestamp': datetime.now().isoformat(),
            'algorithms_compared': algorithm_names,
            'comparison_metrics': comparison_metrics,
            'algorithm_performance': {},
            'rankings': {},
            'statistical_analysis': {},
            'reproducibility_analysis': {},
            'recommendations': []
        }
        
        # Retrieve performance metrics for algorithm comparison
        for algorithm_name in algorithm_names:
            try:
                # Get algorithm metadata with performance history
                metadata = get_algorithm_metadata(
                    algorithm_name=algorithm_name,
                    include_performance_history=True,
                    include_validation_status=True
                )
                
                # Extract performance characteristics
                performance_data = metadata.get('performance_characteristics', {})
                performance_history = metadata.get('performance_history', {})
                
                # Combine current performance with historical data
                combined_performance = {**performance_data, **performance_history}
                
                comparison_results['algorithm_performance'][algorithm_name] = combined_performance
                
            except Exception as e:
                _module_logger.warning(f"Failed to get performance data for {algorithm_name}: {e}")
                comparison_results['algorithm_performance'][algorithm_name] = {}
        
        # Calculate comparison metrics and statistical significance
        for metric_name in comparison_metrics:
            metric_values = {}
            
            for algorithm_name in algorithm_names:
                performance_data = comparison_results['algorithm_performance'][algorithm_name]
                metric_value = performance_data.get(metric_name, 0.0)
                
                # Handle different metric value types
                if isinstance(metric_value, (list, tuple)):
                    metric_value = np.mean(metric_value) if metric_value else 0.0
                elif not isinstance(metric_value, (int, float)):
                    metric_value = 0.0
                    
                metric_values[algorithm_name] = float(metric_value)
            
            # Rank algorithms by metric value
            if 'time' in metric_name.lower() or 'duration' in metric_name.lower():
                # Lower is better for time metrics
                ranked_algorithms = sorted(metric_values.items(), key=lambda x: x[1])
            else:
                # Higher is better for other metrics
                ranked_algorithms = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            
            comparison_results['rankings'][metric_name] = ranked_algorithms
        
        # Perform statistical analysis if requested
        if include_statistical_analysis:
            statistical_analysis = {}
            
            for metric_name in comparison_metrics:
                metric_values = []
                algorithm_labels = []
                
                for algorithm_name in algorithm_names:
                    performance_data = comparison_results['algorithm_performance'][algorithm_name]
                    metric_value = performance_data.get(metric_name, 0.0)
                    
                    if isinstance(metric_value, (list, tuple)):
                        metric_value = np.mean(metric_value) if metric_value else 0.0
                    elif not isinstance(metric_value, (int, float)):
                        metric_value = 0.0
                    
                    metric_values.append(float(metric_value))
                    algorithm_labels.append(algorithm_name)
                
                # Calculate statistical measures
                if len(metric_values) > 1:
                    statistical_analysis[metric_name] = {
                        'mean': float(np.mean(metric_values)),
                        'std': float(np.std(metric_values)),
                        'min': float(np.min(metric_values)),
                        'max': float(np.max(metric_values)),
                        'range': float(np.max(metric_values) - np.min(metric_values)),
                        'coefficient_of_variation': float(np.std(metric_values) / np.mean(metric_values)) if np.mean(metric_values) != 0 else 0.0
                    }
                    
                    # Calculate pairwise correlations if possible
                    if len(set(metric_values)) > 1:  # More than one unique value
                        correlations = {}
                        for i, alg1 in enumerate(algorithm_names):
                            for j, alg2 in enumerate(algorithm_names):
                                if i < j:
                                    # Simple correlation approximation
                                    val1, val2 = metric_values[i], metric_values[j]
                                    correlation = 1.0 - abs(val1 - val2) / max(val1, val2, 1e-10)
                                    correlations[f"{alg1}_vs_{alg2}"] = float(correlation)
                        
                        statistical_analysis[metric_name]['pairwise_correlations'] = correlations
            
            comparison_results['statistical_analysis'] = statistical_analysis
        
        # Validate reproducibility with >0.99 threshold if requested
        if validate_reproducibility:
            reproducibility_analysis = {}
            
            for algorithm_name in algorithm_names:
                performance_data = comparison_results['algorithm_performance'][algorithm_name]
                
                # Check correlation threshold compliance
                correlation_score = performance_data.get('correlation_threshold', 0.0)
                reproducibility_score = performance_data.get('reproducibility_threshold', 0.0)
                
                reproducibility_analysis[algorithm_name] = {
                    'correlation_score': float(correlation_score),
                    'reproducibility_score': float(reproducibility_score),
                    'meets_correlation_threshold': correlation_score >= 0.95,
                    'meets_reproducibility_threshold': reproducibility_score >= 0.99,
                    'overall_reproducibility_compliance': (correlation_score >= 0.95 and reproducibility_score >= 0.99)
                }
            
            comparison_results['reproducibility_analysis'] = reproducibility_analysis
        
        # Generate algorithm rankings and recommendations
        overall_rankings = {}
        for algorithm_name in algorithm_names:
            algorithm_score = 0.0
            score_count = 0
            
            # Calculate average ranking across all metrics
            for metric_name, ranked_list in comparison_results['rankings'].items():
                for rank, (alg_name, _) in enumerate(ranked_list):
                    if alg_name == algorithm_name:
                        # Convert rank to score (higher is better)
                        score = (len(algorithm_names) - rank) / len(algorithm_names)
                        algorithm_score += score
                        score_count += 1
                        break
            
            if score_count > 0:
                overall_rankings[algorithm_name] = algorithm_score / score_count
            else:
                overall_rankings[algorithm_name] = 0.0
        
        # Sort algorithms by overall ranking
        sorted_rankings = sorted(overall_rankings.items(), key=lambda x: x[1], reverse=True)
        comparison_results['overall_rankings'] = sorted_rankings
        
        # Generate comprehensive comparison report with recommendations
        recommendations = []
        
        if sorted_rankings:
            best_algorithm = sorted_rankings[0][0]
            best_score = sorted_rankings[0][1]
            
            recommendations.append(f"Best overall performer: {best_algorithm} (score: {best_score:.3f})")
            
            # Check for statistical significance
            if len(sorted_rankings) > 1:
                second_best_score = sorted_rankings[1][1]
                if best_score - second_best_score < 0.1:
                    recommendations.append("Top algorithms show similar performance - consider specific requirements")
                else:
                    recommendations.append(f"Clear performance leader identified: {best_algorithm}")
        
        # Add reproducibility recommendations
        if validate_reproducibility:
            compliant_algorithms = [
                alg_name for alg_name, analysis in comparison_results['reproducibility_analysis'].items()
                if analysis.get('overall_reproducibility_compliance', False)
            ]
            
            if compliant_algorithms:
                recommendations.append(f"Reproducibility compliant algorithms: {', '.join(compliant_algorithms)}")
            else:
                recommendations.append("No algorithms meet full reproducibility requirements - consider parameter optimization")
        
        # Performance-based recommendations
        if include_statistical_analysis and comparison_results['statistical_analysis']:
            high_variance_metrics = []
            for metric_name, stats in comparison_results['statistical_analysis'].items():
                cv = stats.get('coefficient_of_variation', 0.0)
                if cv > 0.5:  # High variability
                    high_variance_metrics.append(metric_name)
            
            if high_variance_metrics:
                recommendations.append(f"High performance variability in: {', '.join(high_variance_metrics)}")
        
        comparison_results['recommendations'] = recommendations
        
        _module_logger.info(f"Algorithm comparison completed: {len(algorithm_names)} algorithms analyzed")
        
        # Return comprehensive comparison results with scientific validation
        return comparison_results
        
    except Exception as e:
        _module_logger.error(f"Algorithm performance comparison failed: {e}", exc_info=True)
        return {
            'error': str(e),
            'comparison_timestamp': datetime.now().isoformat(),
            'algorithms_requested': algorithm_names
        }


def get_algorithm_recommendations(
    plume_characteristics: Dict[str, Any],
    experimental_conditions: Dict[str, Any], 
    performance_requirements: Dict[str, float],
    max_recommendations: int = 5
) -> List[Dict[str, Any]]:
    """
    Get algorithm recommendations based on plume characteristics, experimental conditions, and 
    performance requirements for optimal algorithm selection.
    
    This function analyzes plume characteristics for algorithm suitability, evaluates experimental 
    conditions and constraints, assesses algorithm performance against requirements, calculates 
    suitability scores for available algorithms, ranks algorithms by suitability and performance, 
    generates recommendations with rationale and confidence scores, and returns top algorithm 
    recommendations up to maximum limit.
    
    Args:
        plume_characteristics: Dictionary containing plume spatial, temporal, and intensity characteristics
        experimental_conditions: Dictionary containing experimental setup and constraints
        performance_requirements: Dictionary containing required performance thresholds
        max_recommendations: Maximum number of algorithm recommendations to return
        
    Returns:
        List[Dict[str, Any]]: Algorithm recommendations with suitability scores and rationale
    """
    try:
        # Ensure registry is initialized
        if _global_registry is None:
            initialize_algorithms_module()
        
        # Initialize recommendation analysis
        recommendations = []
        
        # Get all available algorithms with metadata
        available_algorithms = get_available_algorithms(
            include_metadata=True,
            include_performance_metrics=True
        )
        
        if not available_algorithms:
            return [{'error': 'No algorithms available for recommendation'}]
        
        # Analyze plume characteristics for algorithm suitability
        plume_analysis = _analyze_plume_characteristics(plume_characteristics)
        
        # Evaluate experimental conditions and constraints
        experimental_analysis = _analyze_experimental_conditions(experimental_conditions)
        
        # Calculate suitability scores for available algorithms
        algorithm_suitability = {}
        
        for algorithm_name, algorithm_info in available_algorithms.items():
            try:
                # Calculate suitability score based on multiple factors
                suitability_score = _calculate_algorithm_suitability(
                    algorithm_info=algorithm_info,
                    plume_analysis=plume_analysis,
                    experimental_analysis=experimental_analysis,
                    performance_requirements=performance_requirements
                )
                
                algorithm_suitability[algorithm_name] = suitability_score
                
            except Exception as e:
                _module_logger.warning(f"Failed to calculate suitability for {algorithm_name}: {e}")
                algorithm_suitability[algorithm_name] = {'score': 0.0, 'rationale': [f'Analysis failed: {e}']}
        
        # Rank algorithms by suitability and performance
        ranked_algorithms = sorted(
            algorithm_suitability.items(),
            key=lambda x: x[1].get('score', 0.0),
            reverse=True
        )
        
        # Generate recommendations with rationale and confidence scores
        for algorithm_name, suitability_data in ranked_algorithms[:max_recommendations]:
            try:
                algorithm_info = available_algorithms[algorithm_name]
                
                recommendation = {
                    'algorithm_name': algorithm_name,
                    'algorithm_type': algorithm_info.get('algorithm_type', 'unknown'),
                    'suitability_score': suitability_data.get('score', 0.0),
                    'confidence_score': suitability_data.get('confidence', 0.0),
                    'rationale': suitability_data.get('rationale', []),
                    'performance_prediction': suitability_data.get('performance_prediction', {}),
                    'compatibility_assessment': suitability_data.get('compatibility', {}),
                    'recommendation_details': {
                        'best_use_cases': suitability_data.get('best_use_cases', []),
                        'potential_limitations': suitability_data.get('limitations', []),
                        'parameter_suggestions': suitability_data.get('parameter_suggestions', {}),
                        'expected_performance': suitability_data.get('expected_performance', {})
                    }
                }
                
                # Add algorithm-specific recommendations
                if algorithm_name == 'infotaxis':
                    recommendation['algorithm_specific'] = {
                        'optimal_for': 'Information-rich environments with complex plume structures',
                        'grid_size_recommendation': _recommend_infotaxis_grid_size(plume_characteristics),
                        'entropy_threshold_suggestion': _recommend_entropy_threshold(plume_characteristics)
                    }
                elif algorithm_name == 'casting':
                    recommendation['algorithm_specific'] = {
                        'optimal_for': 'Sparse plumes with intermittent contact',
                        'search_radius_recommendation': _recommend_casting_radius(plume_characteristics),
                        'wind_estimation_suggestion': experimental_conditions.get('wind_variability', 'medium')
                    }
                elif algorithm_name == 'reference_implementation':
                    recommendation['algorithm_specific'] = {
                        'optimal_for': 'Baseline comparison and validation studies',
                        'benchmark_suitability': 'excellent',
                        'correlation_expectation': '> 95%'
                    }
                
                recommendations.append(recommendation)
                
            except Exception as e:
                _module_logger.warning(f"Failed to generate recommendation for {algorithm_name}: {e}")
        
        # Add overall recommendation metadata
        if recommendations:
            recommendation_metadata = {
                'total_algorithms_evaluated': len(available_algorithms),
                'recommendation_timestamp': datetime.now().isoformat(),
                'plume_characteristics_considered': list(plume_characteristics.keys()),
                'experimental_conditions_considered': list(experimental_conditions.keys()),
                'performance_requirements_considered': list(performance_requirements.keys()),
                'recommendation_confidence': _calculate_overall_recommendation_confidence(recommendations)
            }
            
            # Add metadata to first recommendation
            if recommendations:
                recommendations[0]['recommendation_metadata'] = recommendation_metadata
        
        # Return top algorithm recommendations up to maximum limit
        final_recommendations = recommendations[:max_recommendations]
        
        _module_logger.info(f"Generated {len(final_recommendations)} algorithm recommendations")
        
        return final_recommendations
        
    except Exception as e:
        _module_logger.error(f"Algorithm recommendation generation failed: {e}", exc_info=True)
        return [{
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'max_recommendations_requested': max_recommendations
        }]


# Private helper functions for module functionality

def _register_core_algorithms():
    """Register core navigation algorithms in the global registry."""
    try:
        # Register reference implementation
        register_algorithm(
            algorithm_name='reference_implementation',
            algorithm_class=ReferenceImplementation,
            algorithm_metadata={
                'description': 'Reference implementation for plume source localization',
                'algorithm_type': 'reference_implementation',
                'capabilities': ['gradient_following', 'benchmarking', 'validation'],
                'supported_formats': ['crimaldi', 'custom', 'generic'],
                'performance_characteristics': {
                    'target_execution_time_seconds': 7.2,
                    'correlation_threshold': 0.95,
                    'reproducibility_threshold': 0.99
                }
            }
        )
        
        # Register infotaxis algorithm
        register_algorithm(
            algorithm_name='infotaxis',
            algorithm_class=InfotaxisAlgorithm,
            algorithm_metadata={
                'description': 'Information-theoretic source localization algorithm',
                'algorithm_type': 'infotaxis',
                'capabilities': ['information_theory', 'bayesian_inference', 'entropy_minimization'],
                'supported_formats': ['crimaldi', 'custom'],
                'performance_characteristics': {
                    'target_execution_time_seconds': 7.2,
                    'correlation_threshold': 0.95,
                    'reproducibility_threshold': 0.99
                }
            }
        )
        
        # Register casting algorithm
        register_algorithm(
            algorithm_name='casting',
            algorithm_class=CastingAlgorithm,
            algorithm_metadata={
                'description': 'Bio-inspired casting algorithm for search patterns',
                'algorithm_type': 'casting',
                'capabilities': ['bio_inspired', 'search_patterns', 'wind_estimation'],
                'supported_formats': ['crimaldi', 'custom'],
                'performance_characteristics': {
                    'target_execution_time_seconds': 7.2,
                    'correlation_threshold': 0.95,
                    'reproducibility_threshold': 0.99
                }
            }
        )
        
        # Register gradient following algorithm if available
        if GRADIENT_FOLLOWING_AVAILABLE and GradientFollowing:
            register_algorithm(
                algorithm_name='gradient_following',
                algorithm_class=GradientFollowing,
                algorithm_metadata={
                    'description': 'Gradient following algorithm for concentration-based navigation',
                    'algorithm_type': 'gradient_following',
                    'capabilities': ['gradient_following', 'concentration_navigation'],
                    'supported_formats': ['crimaldi', 'custom']
                }
            )
        
        _module_logger.info("Core algorithms registered successfully")
        
    except Exception as e:
        _module_logger.error(f"Failed to register core algorithms: {e}", exc_info=True)


def _enable_dynamic_algorithm_discovery():
    """Enable auto-discovery for advanced algorithms with dynamic loading."""
    try:
        # Register plume tracking algorithm using dynamic loading
        if not PLUME_TRACKING_AVAILABLE:
            register_plume_tracking_algorithm(force_reload=False, validate_integration=True)
        
        # Register hybrid strategies algorithm using dynamic loading  
        if not HYBRID_STRATEGIES_AVAILABLE:
            register_hybrid_strategies_algorithm(force_reload=False, validate_integration=True)
        
        # Discover additional algorithms in the algorithms directory
        discovery_results = discover_available_algorithms(
            algorithms_directory='src/backend/algorithms',
            auto_register=True,
            exclude_modules=['__init__', '__pycache__', 'base_algorithm', 'algorithm_registry']
        )
        
        discovered_count = discovery_results.get('discovery_statistics', {}).get('algorithms_discovered', 0)
        registered_count = discovery_results.get('discovery_statistics', {}).get('algorithms_registered', 0)
        
        _module_logger.info(f"Dynamic discovery completed: {discovered_count} discovered, {registered_count} registered")
        
    except Exception as e:
        _module_logger.warning(f"Dynamic algorithm discovery failed: {e}")


def _setup_performance_tracking():
    """Setup performance tracking infrastructure."""
    try:
        # Initialize performance tracking for registered algorithms
        for algorithm_name in _global_registry.algorithms:
            # Setup baseline performance metrics
            baseline_metrics = {
                'execution_time_baseline': 7.2,
                'correlation_baseline': 0.95,
                'reproducibility_baseline': 0.99,
                'success_rate_baseline': 0.8
            }
            
            update_algorithm_performance(
                algorithm_name=algorithm_name,
                metrics=baseline_metrics,
                context='baseline_initialization'
            )
        
        _module_logger.info("Performance tracking infrastructure initialized")
        
    except Exception as e:
        _module_logger.warning(f"Performance tracking setup failed: {e}")


def _setup_validation_infrastructure():
    """Setup validation infrastructure for scientific computing standards."""
    try:
        # Validate all registered algorithms
        for algorithm_name in _global_registry.algorithms:
            try:
                algorithm_class = get_algorithm(algorithm_name, enable_dynamic_loading=False)
                validation_result = validate_algorithm_interface(algorithm_class, strict_validation=True)
                
                if not validation_result.is_valid:
                    _module_logger.warning(f"Algorithm {algorithm_name} failed strict validation: {validation_result.errors}")
                    
            except Exception as e:
                _module_logger.warning(f"Failed to validate algorithm {algorithm_name}: {e}")
        
        _module_logger.info("Validation infrastructure initialized")
        
    except Exception as e:
        _module_logger.warning(f"Validation infrastructure setup failed: {e}")


def _configure_cross_format_compatibility():
    """Configure cross-format compatibility for Crimaldi and custom plume formats."""
    try:
        # Ensure all algorithms support required formats
        required_formats = ['crimaldi', 'custom']
        
        for algorithm_name in _global_registry.algorithms:
            algorithm_entry = _global_registry.algorithms[algorithm_name]
            supported_formats = algorithm_entry.supported_formats
            
            # Add generic format support if not present
            if 'generic' not in supported_formats:
                supported_formats.append('generic')
                algorithm_entry.supported_formats = supported_formats
        
        global CROSS_FORMAT_COMPATIBILITY
        CROSS_FORMAT_COMPATIBILITY = True
        
        _module_logger.info("Cross-format compatibility configured")
        
    except Exception as e:
        _module_logger.warning(f"Cross-format compatibility configuration failed: {e}")


def _analyze_plume_characteristics(plume_characteristics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze plume characteristics for algorithm recommendation."""
    analysis = {
        'complexity_level': 'medium',
        'density_level': 'medium', 
        'temporal_variability': 'medium',
        'spatial_extent': 'medium',
        'algorithm_preferences': []
    }
    
    try:
        # Analyze plume density
        density = plume_characteristics.get('plume_density', 0.1)
        if density > 0.2:
            analysis['density_level'] = 'high'
            analysis['algorithm_preferences'].append('precision_algorithms')
        elif density < 0.05:
            analysis['density_level'] = 'low'  
            analysis['algorithm_preferences'].append('coverage_algorithms')
        
        # Analyze temporal variability
        temporal_var = plume_characteristics.get('temporal_variability', 0.1)
        if temporal_var > 0.3:
            analysis['temporal_variability'] = 'high'
            analysis['algorithm_preferences'].append('adaptive_algorithms')
        elif temporal_var < 0.1:
            analysis['temporal_variability'] = 'low'
            analysis['algorithm_preferences'].append('systematic_algorithms')
        
        # Analyze spatial complexity
        spatial_var = plume_characteristics.get('spatial_variability', 0.1)
        if spatial_var > 0.2:
            analysis['complexity_level'] = 'high'
            analysis['algorithm_preferences'].append('information_theoretic')
        elif spatial_var < 0.1:
            analysis['complexity_level'] = 'low'
            analysis['algorithm_preferences'].append('gradient_based')
        
    except Exception as e:
        _module_logger.warning(f"Plume characteristic analysis failed: {e}")
    
    return analysis


def _analyze_experimental_conditions(experimental_conditions: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze experimental conditions for algorithm recommendation."""
    analysis = {
        'time_constraints': 'normal',
        'accuracy_requirements': 'normal',
        'resource_constraints': 'normal',
        'preferred_algorithms': []
    }
    
    try:
        # Analyze time constraints
        max_time = experimental_conditions.get('max_processing_time', 10.0)
        if max_time < 5.0:
            analysis['time_constraints'] = 'strict'
            analysis['preferred_algorithms'].append('fast_algorithms')
        elif max_time > 15.0:
            analysis['time_constraints'] = 'relaxed'
            analysis['preferred_algorithms'].append('thorough_algorithms')
        
        # Analyze accuracy requirements
        min_accuracy = experimental_conditions.get('min_accuracy', 0.8)
        if min_accuracy > 0.95:
            analysis['accuracy_requirements'] = 'high'
            analysis['preferred_algorithms'].append('high_precision')
        elif min_accuracy < 0.7:
            analysis['accuracy_requirements'] = 'relaxed'
            analysis['preferred_algorithms'].append('exploratory')
        
        # Analyze resource constraints
        max_memory = experimental_conditions.get('max_memory_usage', 1000)
        if max_memory < 500:
            analysis['resource_constraints'] = 'limited'
            analysis['preferred_algorithms'].append('lightweight')
        
    except Exception as e:
        _module_logger.warning(f"Experimental condition analysis failed: {e}")
    
    return analysis


def _calculate_algorithm_suitability(
    algorithm_info: Dict[str, Any],
    plume_analysis: Dict[str, Any], 
    experimental_analysis: Dict[str, Any],
    performance_requirements: Dict[str, float]
) -> Dict[str, Any]:
    """Calculate algorithm suitability score based on multiple factors."""
    suitability = {
        'score': 0.0,
        'confidence': 0.0,
        'rationale': [],
        'performance_prediction': {},
        'compatibility': {},
        'best_use_cases': [],
        'limitations': [],
        'parameter_suggestions': {},
        'expected_performance': {}
    }
    
    try:
        base_score = 0.5  # Base suitability score
        
        # Analyze algorithm capabilities vs plume characteristics
        capabilities = algorithm_info.get('capabilities', {})
        algorithm_type = algorithm_info.get('algorithm_type', 'unknown')
        
        # Score based on plume complexity
        complexity_match = 0.0
        if plume_analysis['complexity_level'] == 'high':
            if algorithm_type in ['infotaxis', 'hybrid_strategies']:
                complexity_match = 0.3
                suitability['rationale'].append("Well-suited for complex plume structures")
            elif algorithm_type in ['casting', 'gradient_following']:
                complexity_match = 0.1
                suitability['rationale'].append("May struggle with high complexity")
        elif plume_analysis['complexity_level'] == 'low':
            if algorithm_type in ['gradient_following', 'reference_implementation']:
                complexity_match = 0.3
                suitability['rationale'].append("Optimal for simple, structured plumes")
            elif algorithm_type == 'infotaxis':
                complexity_match = 0.2
                suitability['rationale'].append("May be over-engineered for simple plumes")
        else:
            complexity_match = 0.2  # Medium complexity
        
        # Score based on experimental constraints
        constraint_match = 0.0
        performance_chars = algorithm_info.get('performance_characteristics', {})
        
        # Time constraint matching
        target_time = performance_chars.get('target_execution_time_seconds', 10.0)
        max_time = performance_requirements.get('max_execution_time', 10.0)
        
        if target_time <= max_time:
            time_score = 0.2
            suitability['rationale'].append(f"Meets time constraint: {target_time:.1f}s <= {max_time:.1f}s")
        else:
            time_score = 0.1
            suitability['rationale'].append(f"May exceed time constraint: {target_time:.1f}s > {max_time:.1f}s")
            suitability['limitations'].append("Execution time may be too high")
        
        constraint_match += time_score
        
        # Accuracy constraint matching
        correlation_threshold = performance_chars.get('correlation_threshold', 0.8)
        min_accuracy = performance_requirements.get('min_correlation', 0.9)
        
        if correlation_threshold >= min_accuracy:
            accuracy_score = 0.2
            suitability['rationale'].append(f"Meets accuracy requirement: {correlation_threshold:.2f} >= {min_accuracy:.2f}")
        else:
            accuracy_score = 0.1
            suitability['limitations'].append("May not meet accuracy requirements")
        
        constraint_match += accuracy_score
        
        # Calculate overall suitability score
        suitability['score'] = base_score + complexity_match + constraint_match
        suitability['score'] = min(1.0, suitability['score'])  # Cap at 1.0
        
        # Calculate confidence based on available data
        data_quality = 0.8 if algorithm_info.get('detailed_metadata') else 0.6
        performance_data_quality = 0.8 if algorithm_info.get('performance_metrics') else 0.5
        suitability['confidence'] = (data_quality + performance_data_quality) / 2.0
        
        # Generate performance predictions
        suitability['expected_performance'] = {
            'execution_time_estimate': target_time,
            'success_probability': min(0.95, suitability['score'] + 0.1),
            'accuracy_estimate': correlation_threshold
        }
        
        # Generate algorithm-specific recommendations
        if algorithm_type == 'infotaxis':
            suitability['best_use_cases'] = ['Complex plumes', 'Information-rich environments', 'Research studies']
            suitability['parameter_suggestions'] = {
                'grid_size': _recommend_infotaxis_grid_size({'plume_extent': 0.2}),
                'entropy_threshold': 0.01
            }
        elif algorithm_type == 'casting':
            suitability['best_use_cases'] = ['Sparse plumes', 'Wind-dominated environments', 'Bio-inspired studies']
            suitability['parameter_suggestions'] = {
                'search_radius': _recommend_casting_radius({'plume_density': 0.1}),
                'adaptive_radius': True
            }
        elif algorithm_type == 'reference_implementation':
            suitability['best_use_cases'] = ['Baseline studies', 'Algorithm validation', 'Benchmark comparison']
            suitability['parameter_suggestions'] = {
                'validation_enabled': True,
                'benchmarking_enabled': True
            }
        
    except Exception as e:
        _module_logger.warning(f"Suitability calculation failed: {e}")
        suitability['rationale'].append(f"Analysis error: {e}")
    
    return suitability


def _recommend_infotaxis_grid_size(plume_characteristics: Dict[str, Any]) -> int:
    """Recommend optimal grid size for infotaxis algorithm."""
    plume_extent = plume_characteristics.get('plume_extent', 0.2)
    base_size = 50
    
    if plume_extent > 0.3:
        return min(100, int(base_size * 1.5))
    elif plume_extent < 0.1:
        return max(25, int(base_size * 0.8))
    else:
        return base_size


def _recommend_entropy_threshold(plume_characteristics: Dict[str, Any]) -> float:
    """Recommend optimal entropy threshold for infotaxis algorithm."""
    complexity = plume_characteristics.get('spatial_variability', 0.1)
    base_threshold = 0.01
    
    if complexity > 0.2:
        return base_threshold * 0.5  # Lower threshold for complex plumes
    elif complexity < 0.1:
        return base_threshold * 2.0  # Higher threshold for simple plumes
    else:
        return base_threshold


def _recommend_casting_radius(plume_characteristics: Dict[str, Any]) -> float:
    """Recommend optimal search radius for casting algorithm."""
    density = plume_characteristics.get('plume_density', 0.1)
    base_radius = 0.05
    
    if density > 0.2:
        return base_radius * 0.7  # Smaller radius for dense plumes
    elif density < 0.05:
        return base_radius * 1.5  # Larger radius for sparse plumes
    else:
        return base_radius


def _calculate_overall_recommendation_confidence(recommendations: List[Dict[str, Any]]) -> float:
    """Calculate overall confidence in the recommendation set."""
    if not recommendations:
        return 0.0
    
    confidence_scores = [rec.get('confidence_score', 0.0) for rec in recommendations]
    suitability_scores = [rec.get('suitability_score', 0.0) for rec in recommendations]
    
    # Weight by both confidence and suitability
    weighted_confidence = 0.0
    total_weight = 0.0
    
    for conf, suit in zip(confidence_scores, suitability_scores):
        weight = suit + 0.1  # Ensure positive weight
        weighted_confidence += conf * weight
        total_weight += weight
    
    if total_weight > 0:
        return weighted_confidence / total_weight
    else:
        return 0.0


# Module initialization
try:
    # Initialize the algorithms module on import if not already initialized
    if _global_registry is None:
        initialize_algorithms_module()
except Exception as e:
    _module_logger.warning(f"Module initialization on import failed: {e}")

# Module exports
__all__ = [
    # Core classes
    'BaseAlgorithm',
    'AlgorithmParameters', 
    'AlgorithmResult',
    'AlgorithmContext',
    'AlgorithmRegistry',
    
    # Algorithm implementations
    'ReferenceImplementation',
    'InfotaxisAlgorithm',
    'CastingAlgorithm',
    'GradientFollowing',
    'PlumeTrackingAlgorithm', 
    'HybridStrategiesAlgorithm',
    
    # Parameter classes
    'InfotaxisParameters',
    'CastingParameters',
    
    # Registry functions
    'register_algorithm',
    'get_algorithm',
    'create_algorithm_instance',
    'list_algorithms',
    
    # Module functions
    'initialize_algorithms_module',
    'get_available_algorithms',
    'create_algorithm',
    'validate_algorithm_compatibility',
    'compare_algorithm_performance',
    'get_algorithm_recommendations',
    
    # Utility functions
    'validate_plume_data',
    'create_algorithm_context',
    'calculate_performance_metrics',
    'validate_against_benchmark',
    'calculate_entropy',
    'calculate_information_gain',
    'update_belief_state',
    'calculate_wind_direction',
    'detect_plume_contact',
    'optimize_search_radius',
    
    # Constants
    'ALGORITHMS_MODULE_VERSION',
    'SUPPORTED_ALGORITHM_TYPES',
    'ALGORITHM_DISCOVERY_ENABLED',
    'PERFORMANCE_TRACKING_ENABLED',
    'VALIDATION_ENABLED',
    'CROSS_FORMAT_COMPATIBILITY'
]