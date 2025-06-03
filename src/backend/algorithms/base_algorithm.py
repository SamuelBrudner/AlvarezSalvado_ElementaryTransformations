"""
Abstract base algorithm class providing the foundational interface and framework for all navigation algorithms 
in the plume simulation system.

This module implements standardized algorithm execution patterns, parameter validation, performance tracking, 
scientific context management, and result standardization to ensure >95% correlation with reference 
implementations and support reproducible research outcomes across different plume recording formats with 
comprehensive error handling and audit trail integration.

Key Features:
- Abstract base class interface for navigation algorithm standardization
- Comprehensive parameter validation with scientific computing constraints
- Performance tracking with <7.2 seconds target execution time
- Scientific context management for reproducible research outcomes
- Audit trail integration for scientific traceability
- Cross-format compatibility for Crimaldi and custom plume data
- Comprehensive error handling with graceful degradation
- Result standardization with statistical validation
- Algorithm state preservation for debugging and recovery
- Batch processing support for 4000+ simulation requirements
"""

# External imports with version specifications
from abc import ABC, abstractmethod  # Python 3.9+ - Abstract base class functionality
import numpy as np  # version: 2.1.3+ - Numerical array operations and scientific computing
from typing import Dict, Any, List, Optional, Union, Callable, Tuple  # Python 3.9+ - Type hints
import dataclasses  # Python 3.9+ - Data classes for parameters and results
import datetime  # Python 3.9+ - Timestamp generation for execution tracking
import uuid  # Python 3.9+ - Unique identifier generation for correlation
import copy  # Python 3.9+ - Deep copying for state preservation
import json  # Python 3.9+ - JSON serialization for results export
import time  # Python 3.9+ - High-precision timing for performance measurement

# Internal imports from utility modules
from ..utils.validation_utils import (
    ValidationResult, validate_algorithm_parameters, validate_numerical_accuracy
)
from ..utils.performance_monitoring import (
    SimulationPerformanceTracker, track_simulation_performance
)
from ..utils.logging_utils import (
    get_logger, set_scientific_context, log_simulation_event
)
from ..error.exceptions import (
    PlumeSimulationException, ValidationError, SimulationError
)

# Global constants for algorithm execution and validation
ALGORITHM_VERSION = '1.0.0'
DEFAULT_TIMEOUT_SECONDS = 300.0
PERFORMANCE_TRACKING_ENABLED = True
VALIDATION_ENABLED = True
SCIENTIFIC_CONTEXT_ENABLED = True
DEFAULT_CONVERGENCE_TOLERANCE = 1e-6
MAX_ITERATIONS = 10000
CORRELATION_THRESHOLD = 0.95
REPRODUCIBILITY_THRESHOLD = 0.99


def validate_plume_data(
    plume_data: np.ndarray,
    plume_metadata: Dict[str, Any],
    strict_validation: bool = False
) -> ValidationResult:
    """
    Validate plume data format and structure for algorithm compatibility including spatial dimensions, 
    temporal resolution, intensity ranges, and data quality assessment.
    
    This function performs comprehensive validation of plume data to ensure compatibility with navigation 
    algorithms and adherence to scientific computing standards for reproducible research outcomes.
    
    Args:
        plume_data: Plume data array to validate (shape: [time, height, width] or [time, height, width, channels])
        plume_metadata: Metadata dictionary containing format and calibration information
        strict_validation: Enable strict validation mode with enhanced checking
        
    Returns:
        ValidationResult: Comprehensive validation result with plume data compatibility assessment
    """
    # Initialize validation result with plume data context
    validation_result = ValidationResult(
        validation_type="plume_data_validation",
        is_valid=True,
        validation_context=f"strict={strict_validation}, shape={plume_data.shape}"
    )
    
    # Get logger for validation operations
    logger = get_logger('plume_validation', 'VALIDATION')
    logger.debug(f"Starting plume data validation: shape={plume_data.shape}, strict={strict_validation}")
    
    try:
        # Validate plume data array structure and dimensions
        if not isinstance(plume_data, np.ndarray):
            validation_result.add_error(
                "Plume data must be a numpy array",
                severity="HIGH",
                error_context={'data_type': str(type(plume_data))}
            )
            validation_result.is_valid = False
            return validation_result
        
        # Check array dimensionality (3D or 4D expected)
        if plume_data.ndim < 3 or plume_data.ndim > 4:
            validation_result.add_error(
                f"Plume data must be 3D or 4D array, got {plume_data.ndim}D",
                severity="HIGH",
                error_context={'dimensions': plume_data.ndim, 'shape': plume_data.shape}
            )
            validation_result.is_valid = False
        
        # Validate minimum array size requirements
        min_temporal_frames = 10
        min_spatial_size = 32
        
        if plume_data.shape[0] < min_temporal_frames:
            validation_result.add_error(
                f"Insufficient temporal frames: {plume_data.shape[0]} < {min_temporal_frames}",
                severity="MEDIUM"
            )
            validation_result.is_valid = False
        
        if plume_data.shape[1] < min_spatial_size or plume_data.shape[2] < min_spatial_size:
            validation_result.add_error(
                f"Insufficient spatial resolution: {plume_data.shape[1]}x{plume_data.shape[2]} < {min_spatial_size}x{min_spatial_size}",
                severity="MEDIUM"
            )
            validation_result.is_valid = False
        
        # Check temporal resolution and sampling consistency
        if 'frame_rate' in plume_metadata:
            frame_rate = plume_metadata['frame_rate']
            if frame_rate <= 0 or frame_rate > 1000:
                validation_result.add_warning(
                    f"Unusual frame rate: {frame_rate} Hz",
                    warning_context={'frame_rate': frame_rate}
                )
        
        # Validate intensity value ranges and data quality
        if np.any(np.isnan(plume_data)):
            validation_result.add_error(
                "Plume data contains NaN values",
                severity="HIGH",
                error_context={'nan_count': np.sum(np.isnan(plume_data))}
            )
            validation_result.is_valid = False
        
        if np.any(np.isinf(plume_data)):
            validation_result.add_error(
                "Plume data contains infinite values",
                severity="HIGH",
                error_context={'inf_count': np.sum(np.isinf(plume_data))}
            )
            validation_result.is_valid = False
        
        # Check intensity range validity
        data_min, data_max = np.min(plume_data), np.max(plume_data)
        if data_min < 0:
            validation_result.add_warning(
                f"Negative intensity values found: min={data_min}",
                warning_context={'data_min': float(data_min)}
            )
        
        if data_max > 1.0 and 'intensity_normalized' not in plume_metadata:
            validation_result.add_warning(
                f"High intensity values suggest non-normalized data: max={data_max}",
                warning_context={'data_max': float(data_max)}
            )
        
        # Verify spatial dimensions and coordinate systems
        if 'pixel_to_meter_ratio' in plume_metadata:
            pixel_ratio = plume_metadata['pixel_to_meter_ratio']
            if pixel_ratio <= 0:
                validation_result.add_error(
                    f"Invalid pixel-to-meter ratio: {pixel_ratio}",
                    severity="HIGH"
                )
                validation_result.is_valid = False
        
        # Check for missing data and interpolation requirements
        zero_frames = np.sum(np.all(plume_data == 0, axis=(1, 2)))
        if zero_frames > plume_data.shape[0] * 0.1:  # More than 10% zero frames
            validation_result.add_warning(
                f"High number of zero frames: {zero_frames}/{plume_data.shape[0]}",
                warning_context={'zero_frames': int(zero_frames), 'total_frames': int(plume_data.shape[0])}
            )
        
        # Validate metadata consistency with data structure
        required_metadata = ['format_type', 'spatial_units', 'temporal_units']
        for key in required_metadata:
            if key not in plume_metadata:
                validation_result.add_warning(
                    f"Missing recommended metadata: {key}",
                    warning_context={'missing_key': key}
                )
        
        # Perform strict validation checks if enabled
        if strict_validation:
            # Enhanced data quality checks
            intensity_variance = np.var(plume_data)
            if intensity_variance < 1e-8:
                validation_result.add_error(
                    f"Extremely low intensity variance: {intensity_variance}",
                    severity="MEDIUM"
                )
                validation_result.is_valid = False
            
            # Check for temporal consistency
            frame_diffs = np.diff(plume_data, axis=0)
            max_frame_diff = np.max(np.abs(frame_diffs))
            if max_frame_diff > 0.5:  # Large frame-to-frame changes
                validation_result.add_warning(
                    f"Large temporal changes detected: max_diff={max_frame_diff}",
                    warning_context={'max_frame_diff': float(max_frame_diff)}
                )
        
        # Add validation metrics
        validation_result.add_metric("temporal_frames", float(plume_data.shape[0]))
        validation_result.add_metric("spatial_height", float(plume_data.shape[1]))
        validation_result.add_metric("spatial_width", float(plume_data.shape[2]))
        validation_result.add_metric("intensity_min", float(data_min))
        validation_result.add_metric("intensity_max", float(data_max))
        validation_result.add_metric("intensity_mean", float(np.mean(plume_data)))
        validation_result.add_metric("intensity_std", float(np.std(plume_data)))
        
        # Generate validation result with compatibility status
        if validation_result.is_valid:
            validation_result.add_recommendation(
                "Plume data passed validation - ready for algorithm processing",
                priority="INFO"
            )
        else:
            validation_result.add_recommendation(
                "Address validation errors before processing",
                priority="HIGH"
            )
        
        logger.info(f"Plume data validation completed: valid={validation_result.is_valid}, errors={len(validation_result.errors)}")
        
    except Exception as e:
        validation_result.add_error(
            f"Plume data validation failed: {str(e)}",
            severity="CRITICAL",
            error_context={'exception': str(e)}
        )
        validation_result.is_valid = False
        logger.error(f"Plume data validation exception: {e}", exc_info=True)
    
    validation_result.finalize_validation()
    return validation_result


def create_algorithm_context(
    algorithm_name: str,
    simulation_id: str,
    algorithm_parameters: Dict[str, Any],
    execution_config: Dict[str, Any]
) -> 'AlgorithmContext':
    """
    Create comprehensive algorithm execution context including scientific parameters, performance tracking 
    setup, validation configuration, and reproducibility settings for standardized algorithm execution.
    
    This function establishes the execution environment for navigation algorithms with comprehensive 
    tracking, validation, and scientific context management for reproducible research outcomes.
    
    Args:
        algorithm_name: Name of the navigation algorithm to execute
        simulation_id: Unique identifier for the simulation run
        algorithm_parameters: Dictionary of algorithm-specific parameters
        execution_config: Configuration for algorithm execution environment
        
    Returns:
        AlgorithmContext: Comprehensive algorithm execution context with tracking and validation setup
    """
    # Get logger for context creation
    logger = get_logger('algorithm_context', 'ALGORITHM')
    logger.debug(f"Creating algorithm context: {algorithm_name} [{simulation_id}]")
    
    try:
        # Generate unique execution identifier for correlation
        execution_id = str(uuid.uuid4())
        
        # Setup scientific context with simulation and algorithm information
        set_scientific_context(
            simulation_id=simulation_id,
            algorithm_name=algorithm_name,
            processing_stage='ALGORITHM_EXECUTION',
            additional_context={
                'execution_id': execution_id,
                'algorithm_version': ALGORITHM_VERSION
            }
        )
        
        # Initialize performance tracking configuration
        performance_config = execution_config.get('performance_tracking', {})
        performance_enabled = performance_config.get('enabled', PERFORMANCE_TRACKING_ENABLED)
        
        # Configure parameter validation settings
        validation_config = execution_config.get('validation', {})
        validation_enabled = validation_config.get('enabled', VALIDATION_ENABLED)
        
        # Setup reproducibility and correlation tracking
        reproducibility_config = execution_config.get('reproducibility', {})
        correlation_threshold = reproducibility_config.get('correlation_threshold', CORRELATION_THRESHOLD)
        
        # Initialize error handling and recovery context
        error_config = execution_config.get('error_handling', {})
        timeout_seconds = error_config.get('timeout_seconds', DEFAULT_TIMEOUT_SECONDS)
        
        # Configure audit trail and logging context
        audit_config = execution_config.get('audit_trail', {})
        audit_enabled = audit_config.get('enabled', True)
        
        # Create comprehensive algorithm execution context
        context = AlgorithmContext(
            algorithm_name=algorithm_name,
            simulation_id=simulation_id,
            execution_config={
                'execution_id': execution_id,
                'performance_tracking': performance_enabled,
                'validation_enabled': validation_enabled,
                'correlation_threshold': correlation_threshold,
                'timeout_seconds': timeout_seconds,
                'audit_enabled': audit_enabled,
                'algorithm_parameters': algorithm_parameters.copy(),
                'created_at': datetime.datetime.now().isoformat()
            }
        )
        
        logger.info(f"Algorithm context created successfully: {execution_id}")
        return context
        
    except Exception as e:
        logger.error(f"Failed to create algorithm context: {e}", exc_info=True)
        raise SimulationError(
            message=f"Algorithm context creation failed: {str(e)}",
            simulation_id=simulation_id,
            algorithm_name=algorithm_name,
            simulation_context={'error': str(e), 'stage': 'context_creation'}
        )


def calculate_performance_metrics(
    algorithm_result: 'AlgorithmResult',
    reference_metrics: Dict[str, float],
    include_correlation_analysis: bool = True
) -> Dict[str, float]:
    """
    Calculate standardized performance metrics for algorithm evaluation including execution time, 
    convergence analysis, accuracy assessment, and resource utilization for scientific computing validation.
    
    This function provides comprehensive performance analysis for navigation algorithms with statistical 
    validation against reference implementations and scientific computing benchmarks.
    
    Args:
        algorithm_result: Algorithm execution result containing performance data
        reference_metrics: Reference performance metrics for comparison
        include_correlation_analysis: Whether to include correlation analysis against reference
        
    Returns:
        Dict[str, float]: Comprehensive performance metrics with correlation analysis and validation
    """
    # Get logger for performance analysis
    logger = get_logger('performance_metrics', 'PERFORMANCE')
    logger.debug(f"Calculating performance metrics for {algorithm_result.algorithm_name}")
    
    try:
        # Initialize performance metrics dictionary
        metrics = {}
        
        # Calculate execution time and convergence metrics
        metrics['execution_time_seconds'] = algorithm_result.execution_time
        metrics['iterations_completed'] = float(algorithm_result.iterations_completed)
        metrics['convergence_achieved'] = float(algorithm_result.converged)
        
        # Calculate processing rate and efficiency
        if algorithm_result.execution_time > 0:
            metrics['processing_rate_fps'] = 1.0 / algorithm_result.execution_time
            metrics['efficiency_score'] = min(1.0, DEFAULT_TIMEOUT_SECONDS / algorithm_result.execution_time)
        else:
            metrics['processing_rate_fps'] = 0.0
            metrics['efficiency_score'] = 0.0
        
        # Assess algorithm accuracy and precision
        if hasattr(algorithm_result, 'trajectory') and algorithm_result.trajectory is not None:
            trajectory = algorithm_result.trajectory
            if isinstance(trajectory, np.ndarray) and trajectory.size > 0:
                # Calculate trajectory-based metrics
                trajectory_length = float(np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))))
                metrics['trajectory_length'] = trajectory_length
                
                # Calculate trajectory smoothness
                if trajectory.shape[0] > 2:
                    second_derivatives = np.diff(trajectory, n=2, axis=0)
                    trajectory_smoothness = 1.0 / (1.0 + np.mean(np.sum(second_derivatives**2, axis=1)))
                    metrics['trajectory_smoothness'] = float(trajectory_smoothness)
                else:
                    metrics['trajectory_smoothness'] = 1.0
        
        # Measure resource utilization efficiency
        if 'memory_usage_mb' in algorithm_result.performance_metrics:
            memory_usage = algorithm_result.performance_metrics['memory_usage_mb']
            memory_efficiency = 1.0 / (1.0 + memory_usage / 1000.0)  # Normalize by 1GB
            metrics['memory_efficiency'] = float(memory_efficiency)
        
        if 'cpu_usage_percent' in algorithm_result.performance_metrics:
            cpu_usage = algorithm_result.performance_metrics['cpu_usage_percent']
            metrics['cpu_efficiency'] = float(min(1.0, cpu_usage / 100.0))
        
        # Perform correlation analysis against reference if provided
        if include_correlation_analysis and reference_metrics:
            correlation_metrics = {}
            
            for metric_name, reference_value in reference_metrics.items():
                if metric_name in metrics:
                    current_value = metrics[metric_name]
                    
                    # Calculate relative correlation
                    if reference_value != 0:
                        relative_correlation = 1.0 - abs(current_value - reference_value) / abs(reference_value)
                        correlation_metrics[f'{metric_name}_correlation'] = max(0.0, relative_correlation)
                    else:
                        correlation_metrics[f'{metric_name}_correlation'] = 1.0 if current_value == 0 else 0.0
            
            # Calculate overall correlation score
            if correlation_metrics:
                overall_correlation = sum(correlation_metrics.values()) / len(correlation_metrics)
                metrics['overall_correlation'] = float(overall_correlation)
                
                # Check correlation threshold compliance
                metrics['correlation_threshold_met'] = float(overall_correlation >= CORRELATION_THRESHOLD)
            
            # Merge correlation metrics
            metrics.update(correlation_metrics)
        
        # Calculate reproducibility coefficients
        metrics['reproducibility_score'] = float(algorithm_result.converged and 
                                                metrics.get('overall_correlation', 1.0) >= REPRODUCIBILITY_THRESHOLD)
        
        # Validate performance against scientific computing thresholds
        performance_compliance = {
            'time_threshold_met': algorithm_result.execution_time <= DEFAULT_TIMEOUT_SECONDS,
            'convergence_achieved': algorithm_result.converged,
            'correlation_met': metrics.get('overall_correlation', 0.0) >= CORRELATION_THRESHOLD
        }
        
        metrics['performance_compliance_score'] = float(sum(performance_compliance.values()) / len(performance_compliance))
        
        # Generate comprehensive performance metrics dictionary
        metrics['total_warnings'] = float(len(algorithm_result.warnings))
        metrics['success_indicator'] = float(algorithm_result.success and 
                                           metrics['performance_compliance_score'] >= 0.8)
        
        # Include statistical validation and confidence intervals
        if 'statistical_confidence' not in metrics:
            # Calculate confidence based on convergence and correlation
            confidence_factors = []
            if algorithm_result.converged:
                confidence_factors.append(0.4)
            if metrics.get('overall_correlation', 0.0) >= CORRELATION_THRESHOLD:
                confidence_factors.append(0.4)
            if algorithm_result.execution_time <= DEFAULT_TIMEOUT_SECONDS:
                confidence_factors.append(0.2)
            
            metrics['statistical_confidence'] = float(sum(confidence_factors))
        
        logger.info(f"Performance metrics calculated: {len(metrics)} metrics, correlation={metrics.get('overall_correlation', 'N/A')}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Performance metrics calculation failed: {e}", exc_info=True)
        # Return minimal metrics in case of error
        return {
            'execution_time_seconds': getattr(algorithm_result, 'execution_time', 0.0),
            'success_indicator': float(getattr(algorithm_result, 'success', False)),
            'error_occurred': 1.0,
            'error_message': str(e)
        }


@dataclasses.dataclass
class AlgorithmParameters:
    """
    Data class for algorithm parameters with validation, serialization, and scientific computing context 
    support providing standardized parameter management across all navigation algorithms.
    
    This class provides comprehensive parameter management with validation, constraints checking, 
    and scientific computing context for reproducible algorithm execution and research outcomes.
    """
    
    # Core algorithm identification
    algorithm_name: str
    version: str = ALGORITHM_VERSION
    
    # Algorithm parameter storage
    parameters: Dict[str, Any] = dataclasses.field(default_factory=dict)
    constraints: Dict[str, Any] = dataclasses.field(default_factory=dict)
    
    # Execution configuration
    convergence_tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE
    max_iterations: int = MAX_ITERATIONS
    enable_performance_tracking: bool = PERFORMANCE_TRACKING_ENABLED
    
    # Metadata and tracking
    created_timestamp: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
    
    def __post_init__(self):
        """Initialize algorithm parameters with validation and scientific computing context."""
        # Validate algorithm name
        if not self.algorithm_name or not isinstance(self.algorithm_name, str):
            raise ValueError("Algorithm name must be a non-empty string")
        
        # Ensure parameters dictionary is properly initialized
        if not isinstance(self.parameters, dict):
            self.parameters = {}
        
        # Set default constraints if not provided
        if not isinstance(self.constraints, dict):
            self.constraints = {}
        
        # Validate convergence tolerance
        if self.convergence_tolerance <= 0:
            raise ValueError("Convergence tolerance must be positive")
        
        # Validate maximum iterations
        if self.max_iterations <= 0:
            raise ValueError("Maximum iterations must be positive")
        
        # Add default constraints for common parameters
        default_constraints = {
            'convergence_tolerance': {'min': 1e-12, 'max': 1e-3},
            'max_iterations': {'min': 1, 'max': 100000},
            'step_size': {'min': 1e-6, 'max': 1.0},
            'learning_rate': {'min': 1e-6, 'max': 1.0}
        }
        
        for param_name, constraint in default_constraints.items():
            if param_name not in self.constraints:
                self.constraints[param_name] = constraint
    
    def validate(self, strict_validation: bool = False) -> ValidationResult:
        """
        Validate algorithm parameters against constraints and scientific computing requirements.
        
        Args:
            strict_validation: Enable strict validation with enhanced constraint checking
            
        Returns:
            ValidationResult: Parameter validation result with constraint compliance assessment
        """
        # Initialize validation result
        validation_result = ValidationResult(
            validation_type="algorithm_parameters_validation",
            is_valid=True,
            validation_context=f"algorithm={self.algorithm_name}, strict={strict_validation}"
        )
        
        try:
            # Validate parameter types and value ranges
            for param_name, param_value in self.parameters.items():
                # Check if parameter has constraints
                if param_name in self.constraints:
                    constraint = self.constraints[param_name]
                    
                    # Validate against minimum constraint
                    if 'min' in constraint and param_value < constraint['min']:
                        validation_result.add_error(
                            f"Parameter {param_name} below minimum: {param_value} < {constraint['min']}",
                            severity="HIGH"
                        )
                        validation_result.is_valid = False
                    
                    # Validate against maximum constraint
                    if 'max' in constraint and param_value > constraint['max']:
                        validation_result.add_error(
                            f"Parameter {param_name} above maximum: {param_value} > {constraint['max']}",
                            severity="HIGH"
                        )
                        validation_result.is_valid = False
                    
                    # Check parameter type if specified
                    if 'type' in constraint:
                        expected_type = constraint['type']
                        if not isinstance(param_value, expected_type):
                            validation_result.add_error(
                                f"Parameter {param_name} wrong type: expected {expected_type}, got {type(param_value)}",
                                severity="MEDIUM"
                            )
                            validation_result.is_valid = False
                
                # Check for NaN or infinite values
                if isinstance(param_value, (int, float)):
                    if np.isnan(param_value) or np.isinf(param_value):
                        validation_result.add_error(
                            f"Parameter {param_name} has invalid numeric value: {param_value}",
                            severity="HIGH"
                        )
                        validation_result.is_valid = False
            
            # Check parameter constraints and bounds
            if self.convergence_tolerance <= 0 or self.convergence_tolerance > 1e-3:
                validation_result.add_warning(
                    f"Convergence tolerance outside recommended range: {self.convergence_tolerance}"
                )
            
            if self.max_iterations < 10 or self.max_iterations > MAX_ITERATIONS:
                validation_result.add_warning(
                    f"Max iterations outside recommended range: {self.max_iterations}"
                )
            
            # Validate convergence criteria and iteration limits
            if strict_validation:
                # Enhanced validation for strict mode
                required_params = ['step_size', 'convergence_criteria']
                for param in required_params:
                    if param not in self.parameters:
                        validation_result.add_warning(
                            f"Recommended parameter missing: {param}"
                        )
                
                # Check parameter consistency
                if 'step_size' in self.parameters and 'learning_rate' in self.parameters:
                    step_size = self.parameters['step_size']
                    learning_rate = self.parameters['learning_rate']
                    if step_size > learning_rate * 10:
                        validation_result.add_warning(
                            "Step size may be too large relative to learning rate"
                        )
            
            # Add validation metrics
            validation_result.add_metric("parameter_count", float(len(self.parameters)))
            validation_result.add_metric("constraint_count", float(len(self.constraints)))
            validation_result.add_metric("convergence_tolerance", float(self.convergence_tolerance))
            validation_result.add_metric("max_iterations", float(self.max_iterations))
            
            # Generate validation result with recommendations
            if validation_result.is_valid:
                validation_result.add_recommendation(
                    "Algorithm parameters passed validation",
                    priority="INFO"
                )
            else:
                validation_result.add_recommendation(
                    "Correct parameter constraint violations before execution",
                    priority="HIGH"
                )
        
        except Exception as e:
            validation_result.add_error(
                f"Parameter validation failed: {str(e)}",
                severity="CRITICAL"
            )
            validation_result.is_valid = False
        
        validation_result.finalize_validation()
        return validation_result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert algorithm parameters to dictionary format for serialization and logging.
        
        Returns:
            Dict[str, Any]: Algorithm parameters as dictionary with metadata
        """
        return {
            'algorithm_name': self.algorithm_name,
            'version': self.version,
            'parameters': self.parameters.copy(),
            'constraints': self.constraints.copy(),
            'convergence_tolerance': self.convergence_tolerance,
            'max_iterations': self.max_iterations,
            'enable_performance_tracking': self.enable_performance_tracking,
            'created_timestamp': self.created_timestamp.isoformat(),
            'parameter_count': len(self.parameters),
            'constraint_count': len(self.constraints)
        }
    
    def copy(self) -> 'AlgorithmParameters':
        """
        Create deep copy of algorithm parameters for isolation and state preservation.
        
        Returns:
            AlgorithmParameters: Deep copy of algorithm parameters
        """
        return AlgorithmParameters(
            algorithm_name=self.algorithm_name,
            version=self.version,
            parameters=copy.deepcopy(self.parameters),
            constraints=copy.deepcopy(self.constraints),
            convergence_tolerance=self.convergence_tolerance,
            max_iterations=self.max_iterations,
            enable_performance_tracking=self.enable_performance_tracking,
            created_timestamp=self.created_timestamp
        )


@dataclasses.dataclass
class AlgorithmResult:
    """
    Data container for algorithm execution results with performance metrics, validation status, and 
    scientific computing context providing standardized result format across all navigation algorithms.
    
    This class provides comprehensive result storage with performance tracking, validation status, 
    and scientific context for reproducible research outcomes and algorithm analysis.
    """
    
    # Core algorithm identification
    algorithm_name: str
    simulation_id: str
    execution_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    
    # Execution status and results
    success: bool = False
    trajectory: Optional[np.ndarray] = None
    
    # Performance tracking
    performance_metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    algorithm_state: Dict[str, Any] = dataclasses.field(default_factory=dict)
    execution_time: float = 0.0
    iterations_completed: int = 0
    converged: bool = False
    
    # Metadata and tracking
    completion_timestamp: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
    warnings: List[str] = dataclasses.field(default_factory=list)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize algorithm result container with execution context and performance tracking."""
        # Validate required fields
        if not self.algorithm_name:
            raise ValueError("Algorithm name is required")
        if not self.simulation_id:
            raise ValueError("Simulation ID is required")
        
        # Initialize performance metrics with basic tracking
        if 'execution_start' not in self.performance_metrics:
            self.performance_metrics['execution_start'] = time.time()
        
        # Set default metadata
        if 'result_version' not in self.metadata:
            self.metadata['result_version'] = ALGORITHM_VERSION
        
        # Initialize algorithm state if empty
        if not self.algorithm_state:
            self.algorithm_state = {
                'initialized': True,
                'stage': 'created',
                'last_update': datetime.datetime.now().isoformat()
            }
    
    def add_performance_metric(
        self,
        metric_name: str,
        metric_value: float,
        metric_unit: str = ""
    ) -> None:
        """
        Add performance metric to result with validation and scientific context.
        
        Args:
            metric_name: Name of the performance metric
            metric_value: Numerical value of the metric
            metric_unit: Unit of measurement for the metric
        """
        # Validate metric parameters
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("Metric name must be a non-empty string")
        
        if not isinstance(metric_value, (int, float)):
            raise TypeError("Metric value must be numeric")
        
        if np.isnan(metric_value) or np.isinf(metric_value):
            raise ValueError(f"Metric value cannot be NaN or infinite: {metric_value}")
        
        # Add metric to performance metrics dictionary
        self.performance_metrics[metric_name] = float(metric_value)
        
        # Store metric metadata
        if metric_unit:
            self.metadata[f"{metric_name}_unit"] = metric_unit
        
        self.metadata[f"{metric_name}_recorded_at"] = datetime.datetime.now().isoformat()
        
        # Log metric addition for audit trail
        logger = get_logger('algorithm_result', 'ALGORITHM')
        logger.debug(f"Performance metric added: {metric_name} = {metric_value} {metric_unit}")
    
    def add_warning(
        self,
        warning_message: str,
        warning_category: str = "general"
    ) -> None:
        """
        Add warning message to result for non-critical issues tracking.
        
        Args:
            warning_message: Warning message describing the issue
            warning_category: Category of the warning for classification
        """
        if not isinstance(warning_message, str) or not warning_message.strip():
            raise ValueError("Warning message must be a non-empty string")
        
        # Format warning with category and timestamp
        formatted_warning = f"[{warning_category.upper()}] {warning_message}"
        self.warnings.append(formatted_warning)
        
        # Store warning metadata
        warning_key = f"warning_{len(self.warnings)}"
        self.metadata[warning_key] = {
            'message': warning_message,
            'category': warning_category,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Log warning for tracking and analysis
        logger = get_logger('algorithm_result', 'ALGORITHM')
        logger.warning(f"Algorithm warning [{warning_category}]: {warning_message}")
    
    def to_dict(
        self,
        include_trajectory: bool = True,
        include_state: bool = True
    ) -> Dict[str, Any]:
        """
        Convert algorithm result to dictionary format for serialization and analysis.
        
        Args:
            include_trajectory: Whether to include trajectory data
            include_state: Whether to include algorithm state data
            
        Returns:
            Dict[str, Any]: Algorithm result as dictionary with optional trajectory and state data
        """
        result_dict = {
            'algorithm_name': self.algorithm_name,
            'simulation_id': self.simulation_id,
            'execution_id': self.execution_id,
            'success': self.success,
            'execution_time': self.execution_time,
            'iterations_completed': self.iterations_completed,
            'converged': self.converged,
            'completion_timestamp': self.completion_timestamp.isoformat(),
            'performance_metrics': self.performance_metrics.copy(),
            'warnings': self.warnings.copy(),
            'metadata': self.metadata.copy()
        }
        
        # Include trajectory data if requested and available
        if include_trajectory and self.trajectory is not None:
            if isinstance(self.trajectory, np.ndarray):
                result_dict['trajectory'] = self.trajectory.tolist()
                result_dict['trajectory_shape'] = list(self.trajectory.shape)
            else:
                result_dict['trajectory'] = self.trajectory
        
        # Include algorithm state if requested
        if include_state:
            result_dict['algorithm_state'] = self.algorithm_state.copy()
        
        return result_dict
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Generate concise result summary with key metrics and status information.
        
        Returns:
            Dict[str, Any]: Concise result summary with key performance indicators
        """
        # Calculate key performance indicators
        key_metrics = {}
        for metric_name, metric_value in self.performance_metrics.items():
            if any(keyword in metric_name.lower() for keyword in ['time', 'accuracy', 'correlation', 'efficiency']):
                key_metrics[metric_name] = metric_value
        
        # Generate summary statistics
        summary = {
            'execution_id': self.execution_id,
            'algorithm_name': self.algorithm_name,
            'success': self.success,
            'converged': self.converged,
            'execution_time': self.execution_time,
            'iterations_completed': self.iterations_completed,
            'warning_count': len(self.warnings),
            'key_metrics': key_metrics,
            'performance_score': self.performance_metrics.get('performance_compliance_score', 0.0),
            'completion_timestamp': self.completion_timestamp.isoformat()
        }
        
        # Add trajectory summary if available
        if self.trajectory is not None and isinstance(self.trajectory, np.ndarray):
            summary['trajectory_summary'] = {
                'shape': list(self.trajectory.shape),
                'length': len(self.trajectory),
                'start_point': self.trajectory[0].tolist() if len(self.trajectory) > 0 else None,
                'end_point': self.trajectory[-1].tolist() if len(self.trajectory) > 0 else None
            }
        
        return summary


class AlgorithmContext:
    """
    Execution context manager for algorithm runs providing scientific context management, performance 
    tracking, error handling, and reproducibility support for standardized algorithm execution.
    
    This class provides comprehensive execution context with automatic resource management, 
    performance tracking, and scientific context for reproducible algorithm execution.
    """
    
    def __init__(
        self,
        algorithm_name: str,
        simulation_id: str,
        execution_config: Dict[str, Any]
    ):
        """
        Initialize algorithm execution context with performance tracking and scientific context setup.
        
        Args:
            algorithm_name: Name of the algorithm being executed
            simulation_id: Unique identifier for the simulation
            execution_config: Configuration for algorithm execution
        """
        # Core context identification
        self.algorithm_name = algorithm_name
        self.simulation_id = simulation_id
        self.execution_id = execution_config.get('execution_id', str(uuid.uuid4()))
        self.execution_config = execution_config.copy()
        
        # Performance tracking setup
        self.performance_tracker = None
        if execution_config.get('performance_tracking', PERFORMANCE_TRACKING_ENABLED):
            try:
                self.performance_tracker = SimulationPerformanceTracker(
                    simulation_id=simulation_id,
                    algorithm_name=algorithm_name
                )
            except ImportError:
                # Handle case where performance monitoring module is not available
                self.performance_tracker = None
        
        # Timing and status tracking
        self.start_time: Optional[datetime.datetime] = None
        self.end_time: Optional[datetime.datetime] = None
        self.is_active = False
        
        # Scientific context storage
        self.scientific_context = {
            'simulation_id': simulation_id,
            'algorithm_name': algorithm_name,
            'execution_id': self.execution_id,
            'context_version': ALGORITHM_VERSION
        }
        
        # Logger setup with algorithm context
        self.logger = get_logger(f'algorithm.{algorithm_name}', 'ALGORITHM')
        
        # Initialize execution metadata
        self.execution_config['context_created_at'] = datetime.datetime.now().isoformat()
    
    def __enter__(self) -> 'AlgorithmContext':
        """
        Enter algorithm execution context and setup tracking, logging, and scientific context.
        
        Returns:
            AlgorithmContext: Self reference for context management
        """
        try:
            # Record algorithm execution start time
            self.start_time = datetime.datetime.now()
            self.is_active = True
            
            # Set scientific context for current thread
            if SCIENTIFIC_CONTEXT_ENABLED:
                set_scientific_context(
                    simulation_id=self.simulation_id,
                    algorithm_name=self.algorithm_name,
                    processing_stage='ALGORITHM_EXECUTION',
                    additional_context=self.scientific_context
                )
            
            # Start performance tracking
            if self.performance_tracker:
                self.performance_tracker.start_tracking()
            
            # Log algorithm execution start event
            log_simulation_event(
                event_type='START',
                simulation_id=self.simulation_id,
                algorithm_name=self.algorithm_name,
                event_data={
                    'execution_id': self.execution_id,
                    'start_time': self.start_time.isoformat(),
                    'execution_config': self.execution_config
                }
            )
            
            self.logger.info(f"Algorithm execution context started: {self.execution_id}")
            return self
            
        except Exception as e:
            self.logger.error(f"Failed to enter algorithm context: {e}", exc_info=True)
            self.is_active = False
            raise SimulationError(
                message=f"Algorithm context initialization failed: {str(e)}",
                simulation_id=self.simulation_id,
                algorithm_name=self.algorithm_name,
                simulation_context={'error': str(e), 'stage': 'context_enter'}
            )
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit algorithm execution context and finalize tracking, logging, and performance analysis.
        
        Args:
            exc_type: Exception type if exception occurred
            exc_val: Exception value if exception occurred
            exc_tb: Exception traceback if exception occurred
            
        Returns:
            bool: False to propagate exceptions
        """
        try:
            # Record algorithm execution end time
            self.end_time = datetime.datetime.now()
            self.is_active = False
            
            # Calculate execution duration
            execution_duration = 0.0
            if self.start_time:
                execution_duration = (self.end_time - self.start_time).total_seconds()
            
            # Stop performance tracking and collect metrics
            performance_metrics = {}
            if self.performance_tracker:
                try:
                    performance_metrics = self.performance_tracker.stop_tracking()
                except Exception as perf_error:
                    self.logger.warning(f"Performance tracking stop failed: {perf_error}")
            
            # Log algorithm execution completion or failure
            event_type = 'ERROR' if exc_type else 'END'
            event_data = {
                'execution_id': self.execution_id,
                'end_time': self.end_time.isoformat(),
                'execution_duration': execution_duration,
                'performance_metrics': performance_metrics
            }
            
            # Include exception information if an error occurred
            if exc_type:
                event_data['exception'] = {
                    'type': exc_type.__name__,
                    'message': str(exc_val),
                    'traceback_summary': str(exc_tb)
                }
                self.logger.error(f"Algorithm execution failed: {exc_type.__name__}: {exc_val}")
            else:
                self.logger.info(f"Algorithm execution completed successfully in {execution_duration:.3f}s")
            
            log_simulation_event(
                event_type=event_type,
                simulation_id=self.simulation_id,
                algorithm_name=self.algorithm_name,
                event_data=event_data,
                performance_metrics=performance_metrics
            )
            
            # Clear scientific context (handled by logging utilities)
            # Note: Context clearing is managed by the logging system
            
        except Exception as context_error:
            self.logger.error(f"Error during context exit: {context_error}", exc_info=True)
        
        # Return False to propagate any exceptions that occurred
        return False
    
    def add_checkpoint(
        self,
        checkpoint_name: str,
        checkpoint_data: Dict[str, Any] = None
    ) -> None:
        """
        Add performance checkpoint during algorithm execution for detailed analysis.
        
        Args:
            checkpoint_name: Name of the checkpoint for identification
            checkpoint_data: Additional data to store with the checkpoint
        """
        try:
            # Validate checkpoint parameters
            if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
                raise ValueError("Checkpoint name must be a non-empty string")
            
            # Add checkpoint to performance tracker
            if self.performance_tracker:
                self.performance_tracker.add_checkpoint(checkpoint_name, checkpoint_data or {})
            
            # Log checkpoint creation with context
            checkpoint_info = {
                'checkpoint_name': checkpoint_name,
                'execution_id': self.execution_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'active': self.is_active
            }
            
            if checkpoint_data:
                checkpoint_info['checkpoint_data'] = checkpoint_data
            
            self.logger.debug(f"Algorithm checkpoint added: {checkpoint_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to add checkpoint {checkpoint_name}: {e}")


class BaseAlgorithm(ABC):
    """
    Abstract base class for all navigation algorithms providing standardized interface, parameter 
    validation, performance tracking, error handling, and scientific computing context management 
    for reproducible research outcomes and cross-platform compatibility.
    
    This class establishes the foundational interface for navigation algorithms with comprehensive 
    validation, performance tracking, and scientific context management to ensure >95% correlation 
    with reference implementations and support 4000+ simulation processing requirements.
    """
    
    def __init__(
        self,
        parameters: AlgorithmParameters,
        execution_config: Dict[str, Any] = None
    ):
        """
        Initialize base algorithm with parameters, validation, and performance tracking setup.
        
        Args:
            parameters: Algorithm parameters with validation and constraints
            execution_config: Configuration for algorithm execution environment
        """
        # Validate parameters instance
        if not isinstance(parameters, AlgorithmParameters):
            raise TypeError("Parameters must be an AlgorithmParameters instance")
        
        # Store algorithm parameters and execution configuration
        self.parameters = parameters
        self.execution_config = execution_config or {}
        
        # Set algorithm identification
        self.algorithm_name = parameters.algorithm_name
        self.version = parameters.version
        
        # Initialize algorithm state and tracking
        self.is_initialized = False
        self.algorithm_state: Dict[str, Any] = {
            'created_at': datetime.datetime.now().isoformat(),
            'state': 'initialized',
            'version': self.version
        }
        
        # Setup logger with algorithm-specific context
        self.logger = get_logger(f'algorithm.{self.algorithm_name}', 'ALGORITHM')
        
        # Initialize execution history and performance tracking
        self.execution_history: List[AlgorithmResult] = []
        self.performance_baselines: Dict[str, float] = {
            'target_execution_time': DEFAULT_TIMEOUT_SECONDS,
            'target_correlation': CORRELATION_THRESHOLD,
            'target_reproducibility': REPRODUCIBILITY_THRESHOLD
        }
        
        # Enable validation by default
        self.validation_enabled = self.execution_config.get('validation_enabled', VALIDATION_ENABLED)
        
        # Validate parameters during initialization
        if self.validation_enabled:
            validation_result = self.validate_parameters(strict_validation=False)
            if not validation_result.is_valid:
                raise ValidationError(
                    message=f"Algorithm parameter validation failed: {len(validation_result.errors)} errors",
                    validation_type="initialization_validation",
                    validation_context={'algorithm': self.algorithm_name},
                    failed_parameters=[error.split(':')[0] for error in validation_result.errors]
                )
        
        # Mark algorithm as initialized
        self.is_initialized = True
        self.algorithm_state['state'] = 'ready'
        
        self.logger.info(f"Algorithm {self.algorithm_name} initialized successfully")
    
    def validate_parameters(self, strict_validation: bool = False) -> ValidationResult:
        """
        Validate algorithm parameters against constraints and scientific computing requirements 
        with comprehensive error reporting.
        
        Args:
            strict_validation: Enable strict validation with enhanced constraint checking
            
        Returns:
            ValidationResult: Comprehensive parameter validation result with constraint compliance assessment
        """
        try:
            # Validate algorithm parameters using built-in validation
            parameter_validation = self.parameters.validate(strict_validation=strict_validation)
            
            # Use validation utilities for enhanced validation
            enhanced_validation = validate_algorithm_parameters(
                algorithm_params=self.parameters.parameters,
                algorithm_type=self.algorithm_name,
                validate_convergence_criteria=strict_validation,
                algorithm_constraints=self.parameters.constraints
            )
            
            # Merge validation results
            merged_validation = ValidationResult(
                validation_type="comprehensive_parameter_validation",
                is_valid=parameter_validation.is_valid and enhanced_validation.is_valid,
                validation_context=f"algorithm={self.algorithm_name}, strict={strict_validation}"
            )
            
            # Combine errors and warnings
            merged_validation.errors.extend(parameter_validation.errors)
            merged_validation.errors.extend(enhanced_validation.errors)
            merged_validation.warnings.extend(parameter_validation.warnings)
            merged_validation.warnings.extend(enhanced_validation.warnings)
            
            # Combine recommendations
            merged_validation.recommendations.extend(parameter_validation.recommendations)
            merged_validation.recommendations.extend(enhanced_validation.recommendations)
            
            # Merge metrics
            merged_validation.metrics.update(parameter_validation.metrics)
            merged_validation.metrics.update(enhanced_validation.metrics)
            
            # Apply strict validation if enabled
            if strict_validation:
                # Additional algorithm-specific validation
                if self.algorithm_name.lower() == 'infotaxis':
                    if 'information_gain_threshold' not in self.parameters.parameters:
                        merged_validation.add_warning(
                            "Infotaxis algorithm missing information_gain_threshold parameter"
                        )
                elif self.algorithm_name.lower() == 'casting':
                    if 'casting_radius' not in self.parameters.parameters:
                        merged_validation.add_warning(
                            "Casting algorithm missing casting_radius parameter"
                        )
            
            # Log validation results for audit trail
            self.logger.debug(f"Parameter validation completed: valid={merged_validation.is_valid}, errors={len(merged_validation.errors)}")
            
            merged_validation.finalize_validation()
            return merged_validation
            
        except Exception as e:
            self.logger.error(f"Parameter validation failed: {e}", exc_info=True)
            error_validation = ValidationResult(
                validation_type="parameter_validation_error",
                is_valid=False,
                validation_context=f"algorithm={self.algorithm_name}"
            )
            error_validation.add_error(f"Validation process failed: {str(e)}", severity="CRITICAL")
            error_validation.finalize_validation()
            return error_validation
    
    def execute(
        self,
        plume_data: np.ndarray,
        plume_metadata: Dict[str, Any],
        simulation_id: str
    ) -> AlgorithmResult:
        """
        Execute algorithm with plume data, performance tracking, and comprehensive error handling.
        
        Args:
            plume_data: Plume data array for navigation algorithm processing
            plume_metadata: Metadata containing format and calibration information
            simulation_id: Unique identifier for the simulation run
            
        Returns:
            AlgorithmResult: Comprehensive algorithm execution result with performance metrics and validation
        """
        # Validate inputs
        if not isinstance(plume_data, np.ndarray):
            raise TypeError("Plume data must be a numpy array")
        if not isinstance(plume_metadata, dict):
            raise TypeError("Plume metadata must be a dictionary")
        if not isinstance(simulation_id, str) or not simulation_id.strip():
            raise ValueError("Simulation ID must be a non-empty string")
        
        # Initialize algorithm result
        result = AlgorithmResult(
            algorithm_name=self.algorithm_name,
            simulation_id=simulation_id,
            success=False
        )
        
        try:
            # Validate plume data format and compatibility
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
                
                # Add validation metrics to result
                for metric_name, metric_value in plume_validation.metrics.items():
                    result.add_performance_metric(f"validation_{metric_name}", metric_value)
            
            # Create algorithm execution context
            context = create_algorithm_context(
                algorithm_name=self.algorithm_name,
                simulation_id=simulation_id,
                algorithm_parameters=self.parameters.parameters,
                execution_config=self.execution_config
            )
            
            # Execute algorithm with context management and performance tracking
            with context:
                # Record execution start time
                execution_start = time.time()
                
                # Execute algorithm-specific implementation
                algorithm_result = self._execute_algorithm(plume_data, plume_metadata, context)
                
                # Record execution end time
                execution_end = time.time()
                execution_time = execution_end - execution_start
                
                # Update result with algorithm-specific results
                if isinstance(algorithm_result, AlgorithmResult):
                    # Merge algorithm-specific result
                    result.success = algorithm_result.success
                    result.trajectory = algorithm_result.trajectory
                    result.converged = algorithm_result.converged
                    result.iterations_completed = algorithm_result.iterations_completed
                    result.performance_metrics.update(algorithm_result.performance_metrics)
                    result.algorithm_state.update(algorithm_result.algorithm_state)
                    result.warnings.extend(algorithm_result.warnings)
                    result.metadata.update(algorithm_result.metadata)
                else:
                    # Handle non-standard result format
                    result.success = True  # Assume success if no errors occurred
                    self.logger.warning("Algorithm returned non-standard result format")
                
                # Set execution time and completion timestamp
                result.execution_time = execution_time
                result.completion_timestamp = datetime.datetime.now()
                
                # Add execution checkpoint
                context.add_checkpoint('algorithm_completion', {
                    'execution_time': execution_time,
                    'success': result.success,
                    'converged': result.converged
                })
            
            # Track performance metrics and convergence
            result.add_performance_metric('execution_time_seconds', execution_time)
            result.add_performance_metric('iterations_completed', float(result.iterations_completed))
            result.add_performance_metric('convergence_achieved', float(result.converged))
            
            # Check performance against targets
            if execution_time > self.performance_baselines['target_execution_time']:
                result.add_warning(
                    f"Execution time {execution_time:.3f}s exceeded target {self.performance_baselines['target_execution_time']}s",
                    warning_category="performance"
                )
            
            # Add algorithm to execution history
            self.execution_history.append(result)
            
            # Limit execution history size for memory management
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-50:]  # Keep last 50 executions
            
            # Log execution completion with metrics
            self.logger.info(
                f"Algorithm execution completed: {simulation_id}, "
                f"success={result.success}, time={execution_time:.3f}s, "
                f"iterations={result.iterations_completed}"
            )
            
            return result
            
        except Exception as e:
            # Handle errors and exceptions gracefully
            result.success = False
            result.completion_timestamp = datetime.datetime.now()
            
            # Add error information to result
            result.add_warning(f"Algorithm execution failed: {str(e)}", warning_category="error")
            result.metadata['execution_error'] = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'occurred_at': datetime.datetime.now().isoformat()
            }
            
            # Log execution error
            self.logger.error(f"Algorithm execution failed: {e}", exc_info=True)
            
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
                        'stage': 'algorithm_execution',
                        'result_state': result.to_dict(include_trajectory=False, include_state=False)
                    }
                )
    
    @abstractmethod
    def _execute_algorithm(
        self,
        plume_data: np.ndarray,
        plume_metadata: Dict[str, Any],
        context: AlgorithmContext
    ) -> AlgorithmResult:
        """
        Abstract method for algorithm-specific implementation that must be overridden by concrete algorithm classes.
        
        This method contains the core algorithm logic and must be implemented by all concrete navigation 
        algorithm classes. It should utilize the provided context for performance tracking and scientific 
        context management.
        
        Args:
            plume_data: Validated plume data array for algorithm processing
            plume_metadata: Plume metadata with format and calibration information
            context: Algorithm execution context with performance tracking and scientific context
            
        Returns:
            AlgorithmResult: Algorithm-specific execution result with trajectory and performance data
        """
        pass
    
    def reset(self) -> None:
        """
        Reset algorithm state to initial conditions for fresh execution.
        
        This method clears execution history, resets algorithm state, and prepares the algorithm 
        for fresh execution while preserving configuration and parameters.
        """
        try:
            # Clear algorithm state and execution history
            self.algorithm_state = {
                'created_at': datetime.datetime.now().isoformat(),
                'state': 'reset',
                'version': self.version,
                'reset_count': self.algorithm_state.get('reset_count', 0) + 1
            }
            
            # Clear execution history
            self.execution_history.clear()
            
            # Reset performance baselines to defaults
            self.performance_baselines = {
                'target_execution_time': DEFAULT_TIMEOUT_SECONDS,
                'target_correlation': CORRELATION_THRESHOLD,
                'target_reproducibility': REPRODUCIBILITY_THRESHOLD
            }
            
            # Reinitialize algorithm-specific state (to be overridden by subclasses)
            self._reset_algorithm_state()
            
            # Mark algorithm as ready for execution
            self.algorithm_state['state'] = 'ready'
            
            # Log algorithm reset operation
            self.logger.info(f"Algorithm {self.algorithm_name} reset successfully")
            
        except Exception as e:
            self.logger.error(f"Algorithm reset failed: {e}", exc_info=True)
            raise SimulationError(
                message=f"Algorithm reset failed: {str(e)}",
                simulation_id="reset_operation",
                algorithm_name=self.algorithm_name,
                simulation_context={'error': str(e), 'stage': 'reset'}
            )
    
    def _reset_algorithm_state(self) -> None:
        """
        Reset algorithm-specific state. This method can be overridden by subclasses 
        to implement algorithm-specific reset logic.
        """
        # Default implementation - no algorithm-specific state to reset
        pass
    
    def get_performance_summary(self, history_window: int = 10) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for algorithm execution history.
        
        Args:
            history_window: Number of recent executions to include in analysis
            
        Returns:
            Dict[str, Any]: Performance summary with statistics and trends
        """
        try:
            # Get recent execution history
            recent_executions = self.execution_history[-history_window:] if self.execution_history else []
            
            if not recent_executions:
                return {
                    'total_executions': 0,
                    'recent_executions': 0,
                    'performance_summary': 'No execution history available'
                }
            
            # Calculate performance statistics
            execution_times = [result.execution_time for result in recent_executions]
            success_rates = [1.0 if result.success else 0.0 for result in recent_executions]
            convergence_rates = [1.0 if result.converged else 0.0 for result in recent_executions]
            
            # Calculate statistics
            avg_execution_time = sum(execution_times) / len(execution_times)
            success_rate = sum(success_rates) / len(success_rates)
            convergence_rate = sum(convergence_rates) / len(convergence_rates)
            
            # Extract performance metrics from recent executions
            all_metrics = {}
            for result in recent_executions:
                for metric_name, metric_value in result.performance_metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)
            
            # Calculate metric averages
            avg_metrics = {}
            for metric_name, values in all_metrics.items():
                if values:
                    avg_metrics[f"avg_{metric_name}"] = sum(values) / len(values)
            
            # Generate performance summary
            performance_summary = {
                'total_executions': len(self.execution_history),
                'recent_executions': len(recent_executions),
                'history_window': history_window,
                'performance_statistics': {
                    'average_execution_time': avg_execution_time,
                    'success_rate': success_rate,
                    'convergence_rate': convergence_rate,
                    'performance_baseline_compliance': avg_execution_time <= self.performance_baselines['target_execution_time']
                },
                'performance_metrics': avg_metrics,
                'performance_trends': {
                    'execution_time_trend': 'stable' if len(execution_times) < 3 else self._calculate_trend(execution_times),
                    'success_rate_trend': 'stable' if len(success_rates) < 3 else self._calculate_trend(success_rates),
                    'overall_performance': 'good' if success_rate > 0.8 and avg_execution_time <= self.performance_baselines['target_execution_time'] else 'needs_improvement'
                },
                'recommendations': self._generate_performance_recommendations(recent_executions)
            }
            
            return performance_summary
            
        except Exception as e:
            self.logger.error(f"Performance summary generation failed: {e}", exc_info=True)
            return {
                'error': f"Performance summary failed: {str(e)}",
                'total_executions': len(self.execution_history),
                'timestamp': datetime.datetime.now().isoformat()
            }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from a list of values."""
        if len(values) < 3:
            return 'stable'
        
        # Simple trend calculation using first and last values
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        change_ratio = (second_half - first_half) / first_half if first_half != 0 else 0
        
        if change_ratio > 0.1:
            return 'improving'
        elif change_ratio < -0.1:
            return 'degrading'
        else:
            return 'stable'
    
    def _generate_performance_recommendations(self, recent_executions: List[AlgorithmResult]) -> List[str]:
        """Generate performance recommendations based on recent execution history."""
        recommendations = []
        
        if not recent_executions:
            return ["No recent executions available for analysis"]
        
        # Analyze execution times
        execution_times = [result.execution_time for result in recent_executions]
        avg_time = sum(execution_times) / len(execution_times)
        
        if avg_time > self.performance_baselines['target_execution_time']:
            recommendations.append(
                f"Average execution time ({avg_time:.3f}s) exceeds target - consider parameter optimization"
            )
        
        # Analyze success rates
        success_rate = sum(1.0 if result.success else 0.0 for result in recent_executions) / len(recent_executions)
        if success_rate < 0.8:
            recommendations.append(
                f"Success rate ({success_rate:.1%}) below 80% - review algorithm parameters and constraints"
            )
        
        # Analyze convergence rates
        convergence_rate = sum(1.0 if result.converged else 0.0 for result in recent_executions) / len(recent_executions)
        if convergence_rate < 0.7:
            recommendations.append(
                f"Convergence rate ({convergence_rate:.1%}) below 70% - consider adjusting convergence criteria"
            )
        
        # Check for warnings
        total_warnings = sum(len(result.warnings) for result in recent_executions)
        if total_warnings > len(recent_executions):
            recommendations.append(
                f"High warning rate ({total_warnings} warnings in {len(recent_executions)} executions) - review warning messages"
            )
        
        if not recommendations:
            recommendations.append("Algorithm performance is within acceptable ranges")
        
        return recommendations
    
    def validate_execution_result(
        self,
        result: AlgorithmResult,
        reference_metrics: Dict[str, float] = None
    ) -> ValidationResult:
        """
        Validate algorithm execution result against scientific computing standards and correlation thresholds.
        
        Args:
            result: Algorithm execution result to validate
            reference_metrics: Reference metrics for correlation analysis
            
        Returns:
            ValidationResult: Result validation with correlation analysis and compliance assessment
        """
        try:
            # Initialize result validation
            validation_result = ValidationResult(
                validation_type="execution_result_validation",
                is_valid=True,
                validation_context=f"algorithm={self.algorithm_name}, simulation={result.simulation_id}"
            )
            
            # Validate result format and completeness
            if not isinstance(result, AlgorithmResult):
                validation_result.add_error(
                    "Result must be an AlgorithmResult instance",
                    severity="CRITICAL"
                )
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
            if result.execution_time > self.performance_baselines['target_execution_time']:
                validation_result.add_warning(
                    f"Execution time {result.execution_time:.3f}s exceeds target {self.performance_baselines['target_execution_time']}s"
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
            if reference_metrics:
                correlation_metrics = calculate_performance_metrics(
                    algorithm_result=result,
                    reference_metrics=reference_metrics,
                    include_correlation_analysis=True
                )
                
                overall_correlation = correlation_metrics.get('overall_correlation', 0.0)
                if overall_correlation < self.performance_baselines['target_correlation']:
                    validation_result.add_error(
                        f"Correlation {overall_correlation:.3f} below threshold {self.performance_baselines['target_correlation']}",
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
            
            if not result.converged and result.iterations_completed >= self.parameters.max_iterations:
                validation_result.add_warning(
                    f"Algorithm did not converge within {self.parameters.max_iterations} iterations"
                )
            
            # Check for excessive warnings
            if len(result.warnings) > 5:
                validation_result.add_warning(
                    f"High number of warnings ({len(result.warnings)}) in result"
                )
            
            # Add result validation metrics
            validation_result.add_metric("execution_time", result.execution_time)
            validation_result.add_metric("success_rate", float(result.success))
            validation_result.add_metric("convergence_rate", float(result.converged))
            validation_result.add_metric("warning_count", float(len(result.warnings)))
            
            # Generate result validation with recommendations
            if validation_result.is_valid:
                validation_result.add_recommendation(
                    "Algorithm result passed validation",
                    priority="INFO"
                )
            else:
                validation_result.add_recommendation(
                    "Address result validation issues for scientific computing compliance",
                    priority="HIGH"
                )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Result validation failed: {e}", exc_info=True)
            error_validation = ValidationResult(
                validation_type="result_validation_error",
                is_valid=False,
                validation_context=f"algorithm={self.algorithm_name}"
            )
            error_validation.add_error(f"Validation process failed: {str(e)}", severity="CRITICAL")
            error_validation.finalize_validation()
            return error_validation
    
    def export_configuration(self, include_execution_history: bool = False) -> Dict[str, Any]:
        """
        Export algorithm configuration and parameters for reproducibility and documentation.
        
        Args:
            include_execution_history: Whether to include execution history in export
            
        Returns:
            Dict[str, Any]: Complete algorithm configuration with parameters and metadata
        """
        try:
            # Export core algorithm configuration
            configuration = {
                'algorithm_name': self.algorithm_name,
                'version': self.version,
                'algorithm_parameters': self.parameters.to_dict(),
                'execution_config': self.execution_config.copy(),
                'performance_baselines': self.performance_baselines.copy(),
                'algorithm_state': self.algorithm_state.copy(),
                'validation_enabled': self.validation_enabled,
                'is_initialized': self.is_initialized,
                'export_metadata': {
                    'exported_at': datetime.datetime.now().isoformat(),
                    'export_version': ALGORITHM_VERSION,
                    'total_executions': len(self.execution_history)
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
            else:
                configuration['export_metadata']['history_included'] = False
            
            # Add algorithm class information
            configuration['algorithm_class'] = {
                'class_name': self.__class__.__name__,
                'base_class': 'BaseAlgorithm',
                'abstract_methods': ['_execute_algorithm']
            }
            
            return configuration
            
        except Exception as e:
            self.logger.error(f"Configuration export failed: {e}", exc_info=True)
            return {
                'error': f"Configuration export failed: {str(e)}",
                'algorithm_name': self.algorithm_name,
                'export_timestamp': datetime.datetime.now().isoformat()
            }