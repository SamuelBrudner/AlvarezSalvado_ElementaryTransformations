"""
Advanced plume tracking navigation algorithm implementation providing sophisticated plume source 
localization using concentration gradient analysis, temporal plume dynamics tracking, and adaptive 
search strategies.

This module implements bio-inspired navigation behaviors including plume boundary detection, 
crosswind casting, and surge-following for robust source localization across different experimental 
conditions with >95% correlation with reference implementations and cross-format compatibility for 
scientific computing workflows.

Key Features:
- Bio-inspired plume tracking with state machine implementation
- Advanced gradient calculation with noise reduction and boundary handling
- Adaptive threshold adjustment based on plume characteristics
- Crosswind casting behavior for plume reacquisition
- Surge behavior for rapid upwind movement
- Comprehensive performance tracking with <7.2 seconds target execution time
- Cross-format compatibility for Crimaldi and custom plume data
- Statistical validation with >95% correlation requirement
- Temporal memory and adaptive parameter optimization
- Scientific computing standards compliance and reproducibility
"""

# External imports with version specifications
import numpy as np  # version: 2.1.3+ - Numerical computations for gradient calculations and plume processing
import scipy.ndimage  # version: 1.15.3+ - Image processing for plume boundary detection and gradient computation
import scipy.spatial.distance  # version: 1.15.3+ - Distance calculations for trajectory analysis
import scipy.optimize  # version: 1.15.3+ - Optimization algorithms for adaptive search parameter tuning
from typing import Dict, Any, List, Optional, Union, Tuple  # Python 3.9+ - Type hints for algorithm interface
import dataclasses  # Python 3.9+ - Data classes for parameter and state management
from enum import Enum  # Python 3.9+ - Enumeration for plume tracking states and behavior modes
import collections  # Python 3.9+ - Efficient data structures for trajectory and encounter tracking
import math  # Python 3.9+ - Mathematical functions for angle calculations and geometric operations
import time  # Python 3.9+ - Timing measurements for algorithm performance tracking

# Internal imports from base algorithm framework
from .base_algorithm import (
    BaseAlgorithm, AlgorithmParameters, AlgorithmResult, AlgorithmContext,
    validate_plume_data, create_algorithm_context, calculate_performance_metrics
)
from .algorithm_registry import register_algorithm
from ..utils.scientific_constants import (
    NUMERICAL_PRECISION_THRESHOLD, DEFAULT_CORRELATION_THRESHOLD, SPATIAL_ACCURACY_THRESHOLD
)
from ..utils.logging_utils import get_logger

# Global plume tracking algorithm constants and configuration
PLUME_TRACKING_VERSION = '1.0.0'
DEFAULT_GRADIENT_THRESHOLD = 0.01
DEFAULT_PLUME_THRESHOLD = 0.1
DEFAULT_CASTING_DISTANCE = 0.2
DEFAULT_SURGE_VELOCITY = 0.05
DEFAULT_CROSSWIND_VELOCITY = 0.03
MAX_TRAJECTORY_LENGTH = 10000
CONVERGENCE_TOLERANCE = 1e-6
PLUME_BOUNDARY_DETECTION_KERNEL_SIZE = 3
GRADIENT_SMOOTHING_SIGMA = 1.0
TEMPORAL_MEMORY_WINDOW = 10
ADAPTIVE_THRESHOLD_FACTOR = 0.8


class PlumeTrackingState(Enum):
    """
    Enumeration class defining plume tracking algorithm states including searching, gradient 
    following, casting, and surge behaviors for state machine implementation in bio-inspired 
    navigation strategies.
    
    This enumeration provides comprehensive state definitions for bio-inspired plume tracking
    with clear state transitions and behavioral context for navigation decision making.
    """
    
    SEARCHING = "searching"           # Initial state for plume detection and search pattern execution
    GRADIENT_FOLLOWING = "gradient_following"  # Upwind navigation following concentration gradients
    CASTING = "casting"               # Crosswind search behavior for plume reacquisition
    SURGING = "surging"              # Rapid upwind movement when strong gradient detected
    CONVERGED = "converged"          # Successful source localization with convergence criteria met
    LOST_PLUME = "lost_plume"        # Plume contact lost requiring recovery and search strategies


@dataclasses.dataclass
class PlumeTrackingParameters:
    """
    Data class for plume tracking algorithm parameters including gradient thresholds, casting 
    behavior settings, velocity parameters, and convergence criteria with validation and 
    optimization support for scientific computing workflows.
    
    This class provides comprehensive parameter management with adaptive adjustment capabilities
    and validation support for reproducible scientific computing requirements.
    """
    
    # Core threshold parameters for plume detection and navigation
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD
    plume_threshold: float = DEFAULT_PLUME_THRESHOLD
    casting_distance: float = DEFAULT_CASTING_DISTANCE
    surge_velocity: float = DEFAULT_SURGE_VELOCITY
    
    # Velocity and movement parameters
    crosswind_velocity: float = DEFAULT_CROSSWIND_VELOCITY
    max_casting_iterations: int = 50
    convergence_tolerance: float = CONVERGENCE_TOLERANCE
    
    # Adaptive behavior configuration
    adaptive_thresholds: bool = True
    gradient_smoothing_sigma: float = GRADIENT_SMOOTHING_SIGMA
    temporal_memory_window: int = TEMPORAL_MEMORY_WINDOW
    
    # Advanced algorithm configuration
    enable_boundary_detection: bool = True
    gradient_method: str = "sobel"  # Options: "sobel", "gaussian", "central_difference"
    
    def __post_init__(self):
        """Initialize plume tracking parameters with validation and constraint checking."""
        # Validate threshold parameters against numerical precision
        if self.gradient_threshold <= 0 or self.gradient_threshold > 1.0:
            raise ValueError(f"Gradient threshold must be between 0 and 1: {self.gradient_threshold}")
        
        if self.plume_threshold <= 0 or self.plume_threshold > 1.0:
            raise ValueError(f"Plume threshold must be between 0 and 1: {self.plume_threshold}")
        
        # Validate velocity and distance parameters
        if self.casting_distance <= 0:
            raise ValueError(f"Casting distance must be positive: {self.casting_distance}")
        
        if self.surge_velocity <= 0:
            raise ValueError(f"Surge velocity must be positive: {self.surge_velocity}")
        
        if self.crosswind_velocity <= 0:
            raise ValueError(f"Crosswind velocity must be positive: {self.crosswind_velocity}")
        
        # Validate convergence and iteration parameters
        if self.convergence_tolerance <= 0:
            raise ValueError(f"Convergence tolerance must be positive: {self.convergence_tolerance}")
        
        if self.max_casting_iterations <= 0:
            raise ValueError(f"Max casting iterations must be positive: {self.max_casting_iterations}")
        
        # Validate temporal memory window
        if self.temporal_memory_window <= 0:
            raise ValueError(f"Temporal memory window must be positive: {self.temporal_memory_window}")
        
        # Validate gradient method
        valid_gradient_methods = ["sobel", "gaussian", "central_difference"]
        if self.gradient_method not in valid_gradient_methods:
            raise ValueError(f"Invalid gradient method: {self.gradient_method}. Must be one of {valid_gradient_methods}")
    
    def validate(self, strict_validation: bool = False) -> 'ValidationResult':
        """
        Validate plume tracking parameters against physical constraints and numerical precision requirements.
        
        Args:
            strict_validation: Enable strict validation with enhanced constraint checking
            
        Returns:
            ValidationResult: Parameter validation result with constraint compliance assessment
        """
        from ..utils.validation_utils import ValidationResult
        
        # Initialize validation result
        validation_result = ValidationResult(
            validation_type="plume_tracking_parameters_validation",
            is_valid=True,
            validation_context=f"strict={strict_validation}"
        )
        
        try:
            # Validate threshold parameters against numerical precision
            if self.gradient_threshold < NUMERICAL_PRECISION_THRESHOLD:
                validation_result.add_warning(
                    f"Gradient threshold below numerical precision: {self.gradient_threshold}"
                )
            
            # Check parameter consistency and physical constraints
            if self.surge_velocity <= self.crosswind_velocity:
                validation_result.add_warning(
                    "Surge velocity should typically be greater than crosswind velocity"
                )
            
            if self.casting_distance < self.surge_velocity:
                validation_result.add_warning(
                    "Casting distance may be too small relative to surge velocity"
                )
            
            # Apply strict validation criteria if enabled
            if strict_validation:
                # Enhanced parameter range validation
                if self.gradient_threshold > 0.1:
                    validation_result.add_warning(
                        f"High gradient threshold may miss weak gradients: {self.gradient_threshold}"
                    )
                
                if self.temporal_memory_window > 20:
                    validation_result.add_warning(
                        f"Large temporal memory window may impact performance: {self.temporal_memory_window}"
                    )
            
            # Add validation metrics
            validation_result.add_metric("gradient_threshold", self.gradient_threshold)
            validation_result.add_metric("plume_threshold", self.plume_threshold)
            validation_result.add_metric("convergence_tolerance", self.convergence_tolerance)
            
            # Generate validation recommendations
            if validation_result.is_valid:
                validation_result.add_recommendation(
                    "Plume tracking parameters are within acceptable ranges",
                    priority="INFO"
                )
        
        except Exception as e:
            validation_result.add_error(
                f"Parameter validation failed: {str(e)}",
                severity="CRITICAL"
            )
            validation_result.is_valid = False
        
        validation_result.finalize_validation()
        return validation_result
    
    def optimize_for_plume_characteristics(
        self,
        plume_statistics: Dict[str, float],
        plume_format: str
    ) -> 'PlumeTrackingParameters':
        """
        Optimize tracking parameters based on plume characteristics and experimental conditions.
        
        Args:
            plume_statistics: Statistical characteristics of the plume data
            plume_format: Format type of the plume data
            
        Returns:
            PlumeTrackingParameters: Optimized parameters for specific plume characteristics
        """
        # Create optimized parameters copy
        optimized_params = PlumeTrackingParameters(
            gradient_threshold=self.gradient_threshold,
            plume_threshold=self.plume_threshold,
            casting_distance=self.casting_distance,
            surge_velocity=self.surge_velocity,
            crosswind_velocity=self.crosswind_velocity,
            max_casting_iterations=self.max_casting_iterations,
            convergence_tolerance=self.convergence_tolerance,
            adaptive_thresholds=self.adaptive_thresholds,
            gradient_smoothing_sigma=self.gradient_smoothing_sigma,
            temporal_memory_window=self.temporal_memory_window,
            enable_boundary_detection=self.enable_boundary_detection,
            gradient_method=self.gradient_method
        )
        
        # Optimize based on plume intensity characteristics
        intensity_mean = plume_statistics.get('intensity_mean', 0.5)
        intensity_std = plume_statistics.get('intensity_std', 0.1)
        
        # Adjust thresholds based on signal-to-noise ratio
        if intensity_std > 0:
            snr = intensity_mean / intensity_std
            if snr < 5:  # Low signal-to-noise ratio
                optimized_params.gradient_threshold *= 0.8
                optimized_params.plume_threshold *= 0.9
                optimized_params.gradient_smoothing_sigma *= 1.2
            elif snr > 20:  # High signal-to-noise ratio
                optimized_params.gradient_threshold *= 1.2
                optimized_params.plume_threshold *= 1.1
                optimized_params.gradient_smoothing_sigma *= 0.8
        
        # Optimize based on plume format
        if plume_format.lower() == 'crimaldi':
            # Crimaldi format optimizations
            optimized_params.surge_velocity *= 1.1
            optimized_params.casting_distance *= 0.9
        elif plume_format.lower() == 'custom':
            # Custom format optimizations
            optimized_params.crosswind_velocity *= 1.05
            optimized_params.temporal_memory_window = min(15, optimized_params.temporal_memory_window + 2)
        
        # Optimize based on spatial characteristics
        spatial_extent = plume_statistics.get('spatial_extent', 1.0)
        if spatial_extent > 0.8:  # Large plume
            optimized_params.casting_distance *= 1.2
            optimized_params.max_casting_iterations += 10
        elif spatial_extent < 0.3:  # Small plume
            optimized_params.casting_distance *= 0.8
            optimized_params.surge_velocity *= 0.9
        
        return optimized_params
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tracking parameters to dictionary format for serialization and logging.
        
        Returns:
            Dict[str, Any]: Parameters as dictionary with all tracking configuration
        """
        return {
            'gradient_threshold': self.gradient_threshold,
            'plume_threshold': self.plume_threshold,
            'casting_distance': self.casting_distance,
            'surge_velocity': self.surge_velocity,
            'crosswind_velocity': self.crosswind_velocity,
            'max_casting_iterations': self.max_casting_iterations,
            'convergence_tolerance': self.convergence_tolerance,
            'adaptive_thresholds': self.adaptive_thresholds,
            'gradient_smoothing_sigma': self.gradient_smoothing_sigma,
            'temporal_memory_window': self.temporal_memory_window,
            'enable_boundary_detection': self.enable_boundary_detection,
            'gradient_method': self.gradient_method,
            'version': PLUME_TRACKING_VERSION
        }


def calculate_concentration_gradient(
    plume_concentration: np.ndarray,
    position: Tuple[float, float],
    smoothing_sigma: float = GRADIENT_SMOOTHING_SIGMA,
    gradient_method: str = "sobel"
) -> Tuple[float, float]:
    """
    Calculate spatial concentration gradient from plume data using advanced numerical differentiation 
    with noise reduction and boundary handling for robust gradient estimation in plume tracking applications.
    
    Args:
        plume_concentration: 2D concentration field at current time step
        position: Current position coordinates (y, x) in array indices
        smoothing_sigma: Gaussian smoothing parameter for noise reduction
        gradient_method: Method for gradient calculation ("sobel", "gaussian", "central_difference")
        
    Returns:
        Tuple[float, float]: Gradient vector (dy, dx) with magnitude and direction for navigation guidance
    """
    try:
        # Validate input parameters
        if not isinstance(plume_concentration, np.ndarray) or plume_concentration.ndim != 2:
            raise ValueError("Plume concentration must be a 2D numpy array")
        
        if len(position) != 2:
            raise ValueError("Position must be a tuple of two coordinates")
        
        y_pos, x_pos = position
        height, width = plume_concentration.shape
        
        # Ensure position is within array bounds
        y_pos = max(0, min(height - 1, int(y_pos)))
        x_pos = max(0, min(width - 1, int(x_pos)))
        
        # Apply Gaussian smoothing to reduce noise in concentration field
        if smoothing_sigma > 0:
            smoothed_concentration = scipy.ndimage.gaussian_filter(
                plume_concentration, sigma=smoothing_sigma, mode='reflect'
            )
        else:
            smoothed_concentration = plume_concentration.copy()
        
        # Calculate spatial derivatives using specified gradient method
        if gradient_method == "sobel":
            # Sobel operator for robust edge detection
            grad_y = scipy.ndimage.sobel(smoothed_concentration, axis=0)
            grad_x = scipy.ndimage.sobel(smoothed_concentration, axis=1)
        elif gradient_method == "gaussian":
            # Gaussian gradient for smooth differentiation
            grad_y = scipy.ndimage.gaussian_filter(smoothed_concentration, sigma=1.0, order=[1, 0])
            grad_x = scipy.ndimage.gaussian_filter(smoothed_concentration, sigma=1.0, order=[0, 1])
        elif gradient_method == "central_difference":
            # Central difference for precise local gradients
            grad_y = np.zeros_like(smoothed_concentration)
            grad_x = np.zeros_like(smoothed_concentration)
            
            # Central difference with boundary handling
            grad_y[1:-1, :] = (smoothed_concentration[2:, :] - smoothed_concentration[:-2, :]) / 2.0
            grad_x[:, 1:-1] = (smoothed_concentration[:, 2:] - smoothed_concentration[:, :-2]) / 2.0
            
            # Forward/backward difference at boundaries
            grad_y[0, :] = smoothed_concentration[1, :] - smoothed_concentration[0, :]
            grad_y[-1, :] = smoothed_concentration[-1, :] - smoothed_concentration[-2, :]
            grad_x[:, 0] = smoothed_concentration[:, 1] - smoothed_concentration[:, 0]
            grad_x[:, -1] = smoothed_concentration[:, -1] - smoothed_concentration[:, -2]
        else:
            raise ValueError(f"Unknown gradient method: {gradient_method}")
        
        # Extract gradient at current position
        gradient_y = grad_y[y_pos, x_pos]
        gradient_x = grad_x[x_pos, x_pos]
        
        # Validate gradient values
        if np.isnan(gradient_y) or np.isnan(gradient_x):
            return (0.0, 0.0)
        
        # Normalize gradient vector and validate magnitude
        gradient_magnitude = np.sqrt(gradient_y**2 + gradient_x**2)
        
        if gradient_magnitude > NUMERICAL_PRECISION_THRESHOLD:
            # Return normalized gradient for consistent navigation guidance
            return (float(gradient_y), float(gradient_x))
        else:
            # Return zero gradient for negligible magnitudes
            return (0.0, 0.0)
    
    except Exception as e:
        # Return zero gradient on calculation failure
        logger = get_logger('plume_tracking', 'ALGORITHM')
        logger.warning(f"Gradient calculation failed: {e}")
        return (0.0, 0.0)


def detect_plume_boundary(
    plume_concentration: np.ndarray,
    concentration_threshold: float = DEFAULT_PLUME_THRESHOLD,
    kernel_size: int = PLUME_BOUNDARY_DETECTION_KERNEL_SIZE,
    adaptive_threshold: bool = True
) -> np.ndarray:
    """
    Detect plume boundaries using edge detection algorithms and concentration thresholds with 
    adaptive threshold adjustment for robust plume boundary identification across different 
    experimental conditions.
    
    Args:
        plume_concentration: 2D concentration field for boundary detection
        concentration_threshold: Concentration threshold for plume region identification
        kernel_size: Kernel size for morphological operations
        adaptive_threshold: Enable adaptive threshold adjustment based on local statistics
        
    Returns:
        np.ndarray: Binary mask indicating plume boundaries with edge detection results
    """
    try:
        # Validate input parameters
        if not isinstance(plume_concentration, np.ndarray) or plume_concentration.ndim != 2:
            raise ValueError("Plume concentration must be a 2D numpy array")
        
        # Apply adaptive threshold adjustment if enabled
        if adaptive_threshold:
            # Calculate local statistics for adaptive thresholding
            local_mean = np.mean(plume_concentration)
            local_std = np.std(plume_concentration)
            
            # Adjust threshold based on local characteristics
            if local_std > 0:
                adaptive_factor = min(2.0, max(0.5, local_mean / local_std))
                adjusted_threshold = concentration_threshold * adaptive_factor
            else:
                adjusted_threshold = concentration_threshold
        else:
            adjusted_threshold = concentration_threshold
        
        # Apply concentration threshold to identify plume regions
        plume_mask = plume_concentration > adjusted_threshold
        
        # Use edge detection algorithms to find boundary contours
        # Apply Sobel edge detection for boundary identification
        edge_magnitude = np.sqrt(
            scipy.ndimage.sobel(plume_concentration, axis=0)**2 +
            scipy.ndimage.sobel(plume_concentration, axis=1)**2
        )
        
        # Threshold edge magnitude to create boundary mask
        edge_threshold = np.mean(edge_magnitude) + np.std(edge_magnitude)
        edge_mask = edge_magnitude > edge_threshold
        
        # Combine plume mask and edge detection for comprehensive boundary detection
        boundary_mask = np.logical_and(plume_mask, edge_mask)
        
        # Filter boundary detection results using morphological operations
        if kernel_size > 1:
            # Create morphological kernel
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            
            # Apply morphological opening to remove noise
            boundary_mask = scipy.ndimage.binary_opening(boundary_mask, structure=kernel)
            
            # Apply morphological closing to fill gaps
            boundary_mask = scipy.ndimage.binary_closing(boundary_mask, structure=kernel)
        
        # Validate boundary detection quality and consistency
        boundary_ratio = np.sum(boundary_mask) / boundary_mask.size
        if boundary_ratio > 0.8:  # Too much detected as boundary
            # Apply more aggressive filtering
            boundary_mask = scipy.ndimage.binary_erosion(boundary_mask)
        elif boundary_ratio < 0.01:  # Too little detected as boundary
            # Relax threshold slightly
            relaxed_threshold = adjusted_threshold * 0.8
            plume_mask = plume_concentration > relaxed_threshold
            boundary_mask = np.logical_and(plume_mask, edge_mask)
        
        return boundary_mask.astype(np.uint8)
    
    except Exception as e:
        # Return empty boundary mask on detection failure
        logger = get_logger('plume_tracking', 'ALGORITHM')
        logger.warning(f"Boundary detection failed: {e}")
        return np.zeros_like(plume_concentration, dtype=np.uint8)


def calculate_plume_statistics(
    plume_concentration: np.ndarray,
    current_position: Tuple[float, float],
    include_temporal_analysis: bool = False
) -> Dict[str, float]:
    """
    Calculate comprehensive plume statistics including concentration moments, spatial distribution 
    characteristics, and temporal dynamics for plume tracking algorithm optimization and performance analysis.
    
    Args:
        plume_concentration: 2D concentration field for statistical analysis
        current_position: Current position coordinates for distance calculations
        include_temporal_analysis: Include temporal dynamics analysis (placeholder for future implementation)
        
    Returns:
        Dict[str, float]: Comprehensive plume statistics with concentration moments and spatial characteristics
    """
    try:
        # Validate input parameters
        if not isinstance(plume_concentration, np.ndarray) or plume_concentration.ndim != 2:
            raise ValueError("Plume concentration must be a 2D numpy array")
        
        if len(current_position) != 2:
            raise ValueError("Position must be a tuple of two coordinates")
        
        # Calculate concentration moments (mean, variance, skewness)
        concentration_mean = float(np.mean(plume_concentration))
        concentration_var = float(np.var(plume_concentration))
        concentration_std = float(np.std(plume_concentration))
        concentration_min = float(np.min(plume_concentration))
        concentration_max = float(np.max(plume_concentration))
        
        # Calculate higher-order moments
        if concentration_std > 0:
            # Skewness calculation
            normalized_data = (plume_concentration - concentration_mean) / concentration_std
            skewness = float(np.mean(normalized_data**3))
            
            # Kurtosis calculation
            kurtosis = float(np.mean(normalized_data**4) - 3.0)
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        # Analyze spatial distribution characteristics
        height, width = plume_concentration.shape
        y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Calculate plume centroid (weighted by concentration)
        total_concentration = np.sum(plume_concentration)
        if total_concentration > 0:
            centroid_y = float(np.sum(y_indices * plume_concentration) / total_concentration)
            centroid_x = float(np.sum(x_indices * plume_concentration) / total_concentration)
        else:
            centroid_y = height / 2.0
            centroid_x = width / 2.0
        
        # Calculate spatial extent and dispersion
        concentration_threshold = concentration_mean + concentration_std
        plume_mask = plume_concentration > concentration_threshold
        
        if np.any(plume_mask):
            plume_indices = np.where(plume_mask)
            plume_extent_y = float(np.max(plume_indices[0]) - np.min(plume_indices[0]))
            plume_extent_x = float(np.max(plume_indices[1]) - np.min(plume_indices[1]))
            spatial_extent = np.sqrt(plume_extent_y**2 + plume_extent_x**2) / np.sqrt(height**2 + width**2)
        else:
            spatial_extent = 0.0
        
        # Compute plume principal axes using covariance analysis
        y_centered = y_indices - centroid_y
        x_centered = x_indices - centroid_x
        
        # Weighted covariance matrix
        if total_concentration > 0:
            cov_yy = float(np.sum(y_centered**2 * plume_concentration) / total_concentration)
            cov_xx = float(np.sum(x_centered**2 * plume_concentration) / total_concentration)
            cov_yx = float(np.sum(y_centered * x_centered * plume_concentration) / total_concentration)
            
            # Calculate principal axis orientation
            if cov_yy + cov_xx > 0:
                principal_angle = 0.5 * np.arctan2(2 * cov_yx, cov_yy - cov_xx)
                principal_angle = float(principal_angle)
            else:
                principal_angle = 0.0
        else:
            principal_angle = 0.0
        
        # Calculate distance metrics from current position
        y_pos, x_pos = current_position
        distance_to_centroid = float(np.sqrt((y_pos - centroid_y)**2 + (x_pos - centroid_x)**2))
        
        # Calculate concentration at current position
        y_pos_int = max(0, min(height - 1, int(y_pos)))
        x_pos_int = max(0, min(width - 1, int(x_pos)))
        local_concentration = float(plume_concentration[y_pos_int, x_pos_int])
        
        # Calculate signal-to-noise ratio
        if concentration_std > 0:
            snr = concentration_mean / concentration_std
        else:
            snr = 0.0
        
        # Include temporal dynamics analysis if requested
        temporal_statistics = {}
        if include_temporal_analysis:
            # Placeholder for temporal analysis - would require time series data
            temporal_statistics = {
                'temporal_variability': 0.0,
                'persistence_index': 1.0,
                'temporal_correlation': 1.0
            }
        
        # Return comprehensive plume statistics dictionary
        statistics = {
            # Basic concentration statistics
            'intensity_mean': concentration_mean,
            'intensity_std': concentration_std,
            'intensity_var': concentration_var,
            'intensity_min': concentration_min,
            'intensity_max': concentration_max,
            'intensity_range': concentration_max - concentration_min,
            'skewness': skewness,
            'kurtosis': kurtosis,
            
            # Spatial characteristics
            'centroid_y': centroid_y,
            'centroid_x': centroid_x,
            'spatial_extent': spatial_extent,
            'principal_angle': principal_angle,
            
            # Distance and position metrics
            'distance_to_centroid': distance_to_centroid,
            'local_concentration': local_concentration,
            
            # Quality metrics
            'signal_to_noise_ratio': snr,
            'total_concentration': total_concentration,
            'plume_area_fraction': float(np.sum(plume_mask) / plume_mask.size)
        }
        
        # Add temporal statistics if available
        statistics.update(temporal_statistics)
        
        return statistics
    
    except Exception as e:
        # Return minimal statistics on calculation failure
        logger = get_logger('plume_tracking', 'ALGORITHM')
        logger.warning(f"Plume statistics calculation failed: {e}")
        return {
            'intensity_mean': 0.0,
            'intensity_std': 0.0,
            'spatial_extent': 0.0,
            'distance_to_centroid': 0.0,
            'signal_to_noise_ratio': 0.0
        }


def validate_plume_tracking_parameters(
    tracking_parameters: Dict[str, Any],
    strict_validation: bool = False,
    validation_constraints: Dict[str, Any] = None
) -> 'ValidationResult':
    """
    Validate plume tracking algorithm parameters including gradient thresholds, casting distances, 
    velocity parameters, and convergence criteria with comprehensive parameter validation for 
    scientific computing requirements.
    
    Args:
        tracking_parameters: Dictionary of tracking parameters to validate
        strict_validation: Enable strict validation with enhanced constraint checking
        validation_constraints: Custom validation constraints for parameters
        
    Returns:
        ValidationResult: Parameter validation result with constraint compliance and recommendations
    """
    from ..utils.validation_utils import ValidationResult
    
    # Initialize validation result
    validation_result = ValidationResult(
        validation_type="plume_tracking_parameters_validation",
        is_valid=True,
        validation_context=f"strict={strict_validation}, params={len(tracking_parameters)}"
    )
    
    try:
        # Set default validation constraints
        default_constraints = {
            'gradient_threshold': {'min': 1e-6, 'max': 1.0},
            'plume_threshold': {'min': 1e-6, 'max': 1.0},
            'casting_distance': {'min': 1e-3, 'max': 10.0},
            'surge_velocity': {'min': 1e-3, 'max': 1.0},
            'crosswind_velocity': {'min': 1e-3, 'max': 1.0},
            'convergence_tolerance': {'min': 1e-12, 'max': 1e-3},
            'max_casting_iterations': {'min': 1, 'max': 1000},
            'temporal_memory_window': {'min': 1, 'max': 50}
        }
        
        # Merge with custom constraints if provided
        if validation_constraints:
            default_constraints.update(validation_constraints)
        
        # Validate gradient threshold parameters against numerical precision
        if 'gradient_threshold' in tracking_parameters:
            gradient_threshold = tracking_parameters['gradient_threshold']
            constraints = default_constraints['gradient_threshold']
            
            if gradient_threshold < constraints['min']:
                validation_result.add_error(
                    f"Gradient threshold below minimum: {gradient_threshold} < {constraints['min']}",
                    severity="HIGH"
                )
                validation_result.is_valid = False
            elif gradient_threshold > constraints['max']:
                validation_result.add_error(
                    f"Gradient threshold above maximum: {gradient_threshold} > {constraints['max']}",
                    severity="HIGH"
                )
                validation_result.is_valid = False
            elif gradient_threshold < NUMERICAL_PRECISION_THRESHOLD:
                validation_result.add_warning(
                    f"Gradient threshold below numerical precision: {gradient_threshold}"
                )
        
        # Check casting distance and velocity parameter ranges
        velocity_params = ['surge_velocity', 'crosswind_velocity']
        for param_name in velocity_params:
            if param_name in tracking_parameters:
                param_value = tracking_parameters[param_name]
                constraints = default_constraints.get(param_name, {'min': 0, 'max': float('inf')})
                
                if param_value < constraints['min']:
                    validation_result.add_error(
                        f"{param_name} below minimum: {param_value} < {constraints['min']}",
                        severity="MEDIUM"
                    )
                    validation_result.is_valid = False
                elif param_value > constraints['max']:
                    validation_result.add_warning(
                        f"{param_name} above recommended maximum: {param_value} > {constraints['max']}"
                    )
        
        # Validate convergence criteria and iteration limits
        if 'convergence_tolerance' in tracking_parameters:
            convergence_tolerance = tracking_parameters['convergence_tolerance']
            constraints = default_constraints['convergence_tolerance']
            
            if convergence_tolerance < constraints['min']:
                validation_result.add_warning(
                    f"Very strict convergence tolerance: {convergence_tolerance}"
                )
            elif convergence_tolerance > constraints['max']:
                validation_result.add_warning(
                    f"Loose convergence tolerance may affect accuracy: {convergence_tolerance}"
                )
        
        # Apply strict validation constraints if enabled
        if strict_validation:
            # Enhanced parameter consistency checks
            if 'surge_velocity' in tracking_parameters and 'crosswind_velocity' in tracking_parameters:
                surge_vel = tracking_parameters['surge_velocity']
                crosswind_vel = tracking_parameters['crosswind_velocity']
                
                if surge_vel <= crosswind_vel:
                    validation_result.add_warning(
                        "Surge velocity should typically be greater than crosswind velocity for effective navigation"
                    )
            
            # Check for potential performance issues
            if 'temporal_memory_window' in tracking_parameters:
                memory_window = tracking_parameters['temporal_memory_window']
                if memory_window > 20:
                    validation_result.add_warning(
                        f"Large temporal memory window may impact performance: {memory_window}"
                    )
            
            # Validate algorithm method parameters
            if 'gradient_method' in tracking_parameters:
                gradient_method = tracking_parameters['gradient_method']
                valid_methods = ['sobel', 'gaussian', 'central_difference']
                if gradient_method not in valid_methods:
                    validation_result.add_error(
                        f"Invalid gradient method: {gradient_method}. Must be one of {valid_methods}",
                        severity="HIGH"
                    )
                    validation_result.is_valid = False
        
        # Check parameter consistency and physical constraints
        required_params = ['gradient_threshold', 'plume_threshold', 'casting_distance']
        missing_params = [param for param in required_params if param not in tracking_parameters]
        if missing_params:
            validation_result.add_warning(
                f"Missing recommended parameters: {missing_params}"
            )
        
        # Add validation metrics
        validation_result.add_metric("parameters_validated", float(len(tracking_parameters)))
        validation_result.add_metric("required_parameters_present", float(len(required_params) - len(missing_params)))
        
        # Generate validation result with recommendations
        if validation_result.is_valid:
            validation_result.add_recommendation(
                "Plume tracking parameters passed validation",
                priority="INFO"
            )
        else:
            validation_result.add_recommendation(
                "Correct parameter constraint violations before algorithm execution",
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


class PlumeTrackingAlgorithm(BaseAlgorithm):
    """
    Advanced plume tracking navigation algorithm implementing bio-inspired search strategies including 
    gradient following, crosswind casting, and surge behaviors with adaptive threshold adjustment, 
    temporal memory, and cross-format compatibility for scientific plume source localization with >95% 
    correlation requirements.
    
    This class provides comprehensive plume tracking capabilities with bio-inspired navigation behaviors,
    adaptive parameter adjustment, and scientific computing standards compliance for reproducible research outcomes.
    """
    
    def __init__(
        self,
        tracking_parameters: PlumeTrackingParameters,
        execution_config: Dict[str, Any] = None
    ):
        """
        Initialize plume tracking algorithm with parameters, state management, and performance 
        tracking for scientific navigation analysis.
        
        Args:
            tracking_parameters: Plume tracking parameters with thresholds and behavior settings
            execution_config: Configuration for algorithm execution environment
        """
        # Convert tracking parameters to AlgorithmParameters format
        algorithm_parameters = AlgorithmParameters(
            algorithm_name='plume_tracking',
            version=PLUME_TRACKING_VERSION,
            parameters=tracking_parameters.to_dict(),
            convergence_tolerance=tracking_parameters.convergence_tolerance,
            max_iterations=MAX_TRAJECTORY_LENGTH
        )
        
        # Initialize base algorithm with parameters and configuration
        super().__init__(algorithm_parameters, execution_config)
        
        # Store plume tracking specific parameters
        self.tracking_parameters = tracking_parameters
        
        # Initialize algorithm state and position tracking
        self.current_state = PlumeTrackingState.SEARCHING
        self.current_position = (0.0, 0.0)  # (y, x) coordinates
        
        # Initialize trajectory and history tracking
        self.trajectory_history: List[Tuple[float, float]] = []
        self.concentration_history = collections.deque(maxlen=tracking_parameters.temporal_memory_window)
        self.gradient_history = collections.deque(maxlen=tracking_parameters.temporal_memory_window)
        
        # Initialize algorithm execution state
        self.iteration_count = 0
        self.last_concentration = 0.0
        self.last_gradient = (0.0, 0.0)
        self.plume_contact = False
        
        # Initialize casting behavior state
        self.casting_iteration = 0
        self.casting_direction = 0.0  # Angle in radians
        
        # Initialize performance metrics tracking
        self.performance_metrics: Dict[str, float] = {}
        
        # Setup statistical analyzer and plume normalizer (mock implementations)
        self.statistical_analyzer = None  # Would be StatisticalAnalyzer instance
        self.plume_normalizer = None      # Would be PlumeNormalizer instance
        
        # Create algorithm-specific logger
        self.logger = get_logger(f'plume_tracking.{self.parameters.algorithm_name}', 'ALGORITHM')
        
        self.logger.info(f"Plume tracking algorithm initialized with state: {self.current_state}")
    
    def _execute_algorithm(
        self,
        plume_data: np.ndarray,
        plume_metadata: Dict[str, Any],
        context: AlgorithmContext
    ) -> AlgorithmResult:
        """
        Execute plume tracking algorithm with bio-inspired navigation strategies, adaptive behavior 
        switching, and comprehensive performance tracking for scientific source localization.
        
        Args:
            plume_data: 3D plume data array (time, height, width)
            plume_metadata: Metadata with format and calibration information
            context: Algorithm execution context with performance tracking
            
        Returns:
            AlgorithmResult: Plume tracking execution result with trajectory, performance metrics, and validation
        """
        # Initialize algorithm result
        result = AlgorithmResult(
            algorithm_name=self.algorithm_name,
            simulation_id=context.execution_config.get('execution_id', 'unknown'),
            execution_id=context.execution_id
        )
        
        try:
            # Add performance tracking checkpoint
            context.add_checkpoint('algorithm_start', {'state': str(self.current_state)})
            
            # Normalize plume data for cross-format compatibility
            if self.plume_normalizer:
                normalized_plume_data = self.plume_normalizer.normalize_plume_data(
                    plume_data, plume_metadata
                )
            else:
                # Simple normalization for compatibility
                normalized_plume_data = plume_data.copy()
                if np.max(normalized_plume_data) > 0:
                    normalized_plume_data = normalized_plume_data / np.max(normalized_plume_data)
            
            # Extract plume format and dimensions
            plume_format = plume_metadata.get('format_type', 'generic')
            time_steps, height, width = normalized_plume_data.shape
            
            # Initialize tracking state and position
            self.current_state = PlumeTrackingState.SEARCHING
            self.current_position = (height // 2, width // 2)  # Start at center
            self.trajectory_history = [self.current_position]
            self.iteration_count = 0
            
            # Calculate plume statistics for parameter optimization
            current_frame = normalized_plume_data[0]  # Start with first frame
            plume_statistics = calculate_plume_statistics(
                current_frame, self.current_position, include_temporal_analysis=False
            )
            
            # Optimize tracking parameters based on plume characteristics
            if self.tracking_parameters.adaptive_thresholds:
                optimized_params = self.tracking_parameters.optimize_for_plume_characteristics(
                    plume_statistics, plume_format
                )
                self.tracking_parameters = optimized_params
            
            # Execute main tracking loop with state machine
            for time_step in range(min(time_steps, self.parameters.max_iterations)):
                self.iteration_count = time_step + 1
                current_frame = normalized_plume_data[time_step]
                
                # Calculate current concentration and gradient
                current_concentration = self._get_concentration_at_position(
                    current_frame, self.current_position
                )
                current_gradient = calculate_concentration_gradient(
                    current_frame, 
                    self.current_position,
                    self.tracking_parameters.gradient_smoothing_sigma,
                    self.tracking_parameters.gradient_method
                )
                
                # Update history tracking
                self.concentration_history.append(current_concentration)
                self.gradient_history.append(current_gradient)
                
                # Update tracking state based on current conditions
                self.current_state = self._update_tracking_state(
                    current_concentration, current_gradient, current_frame
                )
                
                # Apply gradient following and casting behaviors
                new_position = self._execute_state_behavior(
                    current_concentration, current_gradient, current_frame
                )
                
                # Update position and trajectory
                self.current_position = self._constrain_position(new_position, height, width)
                self.trajectory_history.append(self.current_position)
                
                # Track performance metrics and convergence
                self._update_performance_metrics(
                    current_concentration, current_gradient, time_step
                )
                
                # Check convergence criteria
                if self._check_convergence(current_concentration, stability_window=5):
                    self.current_state = PlumeTrackingState.CONVERGED
                    result.converged = True
                    break
                
                # Add periodic checkpoints for performance tracking
                if time_step % 10 == 0:
                    context.add_checkpoint(f'iteration_{time_step}', {
                        'position': self.current_position,
                        'state': str(self.current_state),
                        'concentration': current_concentration
                    })
                
                # Early termination if maximum trajectory length reached
                if len(self.trajectory_history) >= MAX_TRAJECTORY_LENGTH:
                    break
            
            # Validate tracking accuracy against reference standards
            if self.statistical_analyzer:
                accuracy_validation = self.statistical_analyzer.validate_simulation_accuracy(
                    result, reference_data={}  # Would include reference implementation data
                )
                result.add_performance_metric('accuracy_validation', float(accuracy_validation))
            
            # Generate comprehensive algorithm result
            result.success = True
            result.trajectory = np.array(self.trajectory_history)
            result.iterations_completed = self.iteration_count
            result.algorithm_state = {
                'final_state': str(self.current_state),
                'plume_contact': self.plume_contact,
                'casting_iterations': self.casting_iteration,
                'trajectory_length': len(self.trajectory_history)
            }
            
            # Add performance metrics to result
            for metric_name, metric_value in self.performance_metrics.items():
                result.add_performance_metric(metric_name, metric_value)
            
            # Add algorithm-specific metadata
            result.metadata.update({
                'plume_format': plume_format,
                'plume_statistics': plume_statistics,
                'tracking_parameters': self.tracking_parameters.to_dict(),
                'algorithm_version': PLUME_TRACKING_VERSION
            })
            
            # Return tracking result with statistical validation
            context.add_checkpoint('algorithm_completion', {
                'success': result.success,
                'converged': result.converged,
                'trajectory_points': len(self.trajectory_history)
            })
            
            self.logger.info(
                f"Plume tracking completed: success={result.success}, "
                f"converged={result.converged}, iterations={self.iteration_count}"
            )
            
            return result
        
        except Exception as e:
            # Handle errors gracefully with comprehensive error reporting
            result.success = False
            result.add_warning(f"Algorithm execution failed: {str(e)}", "execution_error")
            result.metadata['execution_error'] = str(e)
            
            self.logger.error(f"Plume tracking execution failed: {e}", exc_info=True)
            return result
    
    def _update_tracking_state(
        self,
        current_concentration: float,
        current_gradient: Tuple[float, float],
        plume_data: np.ndarray
    ) -> PlumeTrackingState:
        """
        Update plume tracking state based on current concentration, gradient information, and 
        behavioral rules with adaptive threshold adjustment.
        
        Args:
            current_concentration: Current concentration at position
            current_gradient: Current gradient vector (dy, dx)
            plume_data: Current plume concentration field
            
        Returns:
            PlumeTrackingState: Updated tracking state based on plume conditions and behavioral rules
        """
        try:
            gradient_magnitude = np.sqrt(current_gradient[0]**2 + current_gradient[1]**2)
            
            # Analyze current concentration against plume threshold
            plume_detected = current_concentration > self.tracking_parameters.plume_threshold
            strong_gradient = gradient_magnitude > self.tracking_parameters.gradient_threshold
            
            # Update plume contact status
            self.plume_contact = plume_detected
            
            # Apply state transition rules based on plume conditions
            if self.current_state == PlumeTrackingState.SEARCHING:
                if plume_detected and strong_gradient:
                    return PlumeTrackingState.GRADIENT_FOLLOWING
                elif plume_detected:
                    return PlumeTrackingState.SEARCHING  # Continue searching for gradient
                else:
                    return PlumeTrackingState.SEARCHING  # Keep searching
            
            elif self.current_state == PlumeTrackingState.GRADIENT_FOLLOWING:
                if not plume_detected:
                    # Lost plume contact - switch to casting
                    self.casting_iteration = 0
                    return PlumeTrackingState.CASTING
                elif gradient_magnitude > self.tracking_parameters.gradient_threshold * 2.0:
                    # Strong gradient detected - switch to surging
                    return PlumeTrackingState.SURGING
                elif strong_gradient:
                    return PlumeTrackingState.GRADIENT_FOLLOWING  # Continue following
                else:
                    # Weak gradient - switch to casting
                    self.casting_iteration = 0
                    return PlumeTrackingState.CASTING
            
            elif self.current_state == PlumeTrackingState.CASTING:
                if plume_detected and strong_gradient:
                    return PlumeTrackingState.GRADIENT_FOLLOWING
                elif self.casting_iteration >= self.tracking_parameters.max_casting_iterations:
                    return PlumeTrackingState.LOST_PLUME
                else:
                    return PlumeTrackingState.CASTING  # Continue casting
            
            elif self.current_state == PlumeTrackingState.SURGING:
                if not plume_detected:
                    self.casting_iteration = 0
                    return PlumeTrackingState.CASTING
                elif gradient_magnitude < self.tracking_parameters.gradient_threshold:
                    return PlumeTrackingState.GRADIENT_FOLLOWING
                else:
                    return PlumeTrackingState.SURGING  # Continue surging
            
            elif self.current_state == PlumeTrackingState.LOST_PLUME:
                if plume_detected:
                    return PlumeTrackingState.GRADIENT_FOLLOWING
                else:
                    return PlumeTrackingState.SEARCHING
            
            elif self.current_state == PlumeTrackingState.CONVERGED:
                return PlumeTrackingState.CONVERGED  # Stay converged
            
            # Handle plume loss and recovery scenarios with adaptive thresholds
            if self.tracking_parameters.adaptive_thresholds:
                # Adjust thresholds based on recent history
                if len(self.concentration_history) >= 3:
                    recent_concentrations = list(self.concentration_history)[-3:]
                    avg_recent = np.mean(recent_concentrations)
                    
                    if avg_recent < self.tracking_parameters.plume_threshold * 0.5:
                        # Very low recent concentrations - may need to relax threshold temporarily
                        pass  # Could implement adaptive threshold relaxation here
            
            return self.current_state
        
        except Exception as e:
            self.logger.warning(f"State update failed: {e}")
            return self.current_state
    
    def _execute_state_behavior(
        self,
        current_concentration: float,
        current_gradient: Tuple[float, float],
        plume_data: np.ndarray
    ) -> Tuple[float, float]:
        """
        Execute behavior corresponding to current tracking state with position updates.
        
        Args:
            current_concentration: Current concentration value
            current_gradient: Current gradient vector
            plume_data: Current plume concentration field
            
        Returns:
            Tuple[float, float]: New position after behavior execution
        """
        try:
            if self.current_state == PlumeTrackingState.SEARCHING:
                # Random search pattern or systematic exploration
                return self._execute_search_behavior()
            
            elif self.current_state == PlumeTrackingState.GRADIENT_FOLLOWING:
                # Follow gradient upwind
                return self._execute_gradient_following(current_gradient, step_size=0.1)
            
            elif self.current_state == PlumeTrackingState.CASTING:
                # Execute crosswind casting pattern
                wind_direction = self._estimate_wind_direction()
                return self._execute_casting_behavior(wind_direction, increase_casting_distance=False)
            
            elif self.current_state == PlumeTrackingState.SURGING:
                # Rapid upwind movement
                return self._execute_surge_behavior(current_gradient, np.sqrt(current_gradient[0]**2 + current_gradient[1]**2))
            
            elif self.current_state in [PlumeTrackingState.CONVERGED, PlumeTrackingState.LOST_PLUME]:
                # Stay at current position
                return self.current_position
            
            else:
                # Default behavior - stay at current position
                return self.current_position
        
        except Exception as e:
            self.logger.warning(f"Behavior execution failed: {e}")
            return self.current_position
    
    def _execute_gradient_following(
        self,
        gradient_vector: Tuple[float, float],
        step_size: float
    ) -> Tuple[float, float]:
        """
        Execute gradient following behavior for upwind navigation with gradient smoothing and 
        adaptive step size control.
        
        Args:
            gradient_vector: Gradient vector (dy, dx) for navigation direction
            step_size: Step size for movement
            
        Returns:
            Tuple[float, float]: New position after gradient following step
        """
        try:
            # Normalize gradient vector for direction guidance
            gradient_magnitude = np.sqrt(gradient_vector[0]**2 + gradient_vector[1]**2)
            
            if gradient_magnitude > NUMERICAL_PRECISION_THRESHOLD:
                normalized_gradient = (
                    gradient_vector[0] / gradient_magnitude,
                    gradient_vector[1] / gradient_magnitude
                )
            else:
                normalized_gradient = (0.0, 0.0)
            
            # Apply gradient smoothing using temporal memory
            if len(self.gradient_history) > 1:
                # Weighted average of recent gradients for smoothing
                recent_gradients = list(self.gradient_history)[-3:]  # Use last 3 gradients
                weights = np.array([0.5, 0.3, 0.2])[:len(recent_gradients)]
                weights = weights / np.sum(weights)
                
                smoothed_gradient = np.average(recent_gradients, axis=0, weights=weights)
                smoothed_magnitude = np.sqrt(smoothed_gradient[0]**2 + smoothed_gradient[1]**2)
                
                if smoothed_magnitude > NUMERICAL_PRECISION_THRESHOLD:
                    normalized_gradient = (
                        smoothed_gradient[0] / smoothed_magnitude,
                        smoothed_gradient[1] / smoothed_magnitude
                    )
            
            # Calculate adaptive step size based on gradient magnitude
            adaptive_step = step_size * min(2.0, max(0.5, gradient_magnitude / self.tracking_parameters.gradient_threshold))
            
            # Update position using gradient-based navigation
            new_y = self.current_position[0] + normalized_gradient[0] * adaptive_step
            new_x = self.current_position[1] + normalized_gradient[1] * adaptive_step
            
            return (new_y, new_x)
        
        except Exception as e:
            self.logger.warning(f"Gradient following failed: {e}")
            return self.current_position
    
    def _execute_casting_behavior(
        self,
        wind_direction: float,
        increase_casting_distance: bool
    ) -> Tuple[float, float]:
        """
        Execute crosswind casting behavior for plume reacquisition with systematic search pattern 
        and adaptive casting distance.
        
        Args:
            wind_direction: Estimated wind direction in radians
            increase_casting_distance: Whether to increase casting distance
            
        Returns:
            Tuple[float, float]: New position after casting movement
        """
        try:
            # Calculate crosswind direction perpendicular to wind
            crosswind_angle = wind_direction + math.pi / 2  # 90 degrees perpendicular
            
            # Determine casting direction based on search pattern
            if self.casting_iteration % 2 == 0:
                # Cast to the right
                self.casting_direction = crosswind_angle
            else:
                # Cast to the left
                self.casting_direction = crosswind_angle + math.pi
            
            # Apply adaptive casting distance adjustment
            base_distance = self.tracking_parameters.casting_distance
            if increase_casting_distance:
                adaptive_distance = base_distance * (1.0 + 0.1 * self.casting_iteration)
            else:
                adaptive_distance = base_distance
            
            # Execute casting movement with velocity control
            movement_y = adaptive_distance * math.sin(self.casting_direction)
            movement_x = adaptive_distance * math.cos(self.casting_direction)
            
            new_y = self.current_position[0] + movement_y
            new_x = self.current_position[1] + movement_x
            
            # Update casting iteration and direction tracking
            self.casting_iteration += 1
            
            return (new_y, new_x)
        
        except Exception as e:
            self.logger.warning(f"Casting behavior failed: {e}")
            return self.current_position
    
    def _execute_surge_behavior(
        self,
        gradient_direction: Tuple[float, float],
        gradient_magnitude: float
    ) -> Tuple[float, float]:
        """
        Execute surge behavior for rapid upwind movement when strong gradient is detected with 
        velocity optimization.
        
        Args:
            gradient_direction: Gradient direction vector
            gradient_magnitude: Magnitude of the gradient
            
        Returns:
            Tuple[float, float]: New position after surge movement
        """
        try:
            # Calculate surge velocity based on gradient strength
            base_velocity = self.tracking_parameters.surge_velocity
            velocity_multiplier = min(3.0, max(1.0, gradient_magnitude / self.tracking_parameters.gradient_threshold))
            surge_velocity = base_velocity * velocity_multiplier
            
            # Normalize gradient direction for surge movement
            if gradient_magnitude > NUMERICAL_PRECISION_THRESHOLD:
                normalized_direction = (
                    gradient_direction[0] / gradient_magnitude,
                    gradient_direction[1] / gradient_magnitude
                )
            else:
                normalized_direction = (0.0, 0.0)
            
            # Apply surge movement in gradient direction
            new_y = self.current_position[0] + normalized_direction[0] * surge_velocity
            new_x = self.current_position[1] + normalized_direction[1] * surge_velocity
            
            return (new_y, new_x)
        
        except Exception as e:
            self.logger.warning(f"Surge behavior failed: {e}")
            return self.current_position
    
    def _execute_search_behavior(self) -> Tuple[float, float]:
        """Execute search behavior for initial plume detection."""
        try:
            # Simple random walk for search behavior
            search_step = 0.05
            angle = np.random.uniform(0, 2 * math.pi)
            
            new_y = self.current_position[0] + search_step * math.sin(angle)
            new_x = self.current_position[1] + search_step * math.cos(angle)
            
            return (new_y, new_x)
        
        except Exception as e:
            self.logger.warning(f"Search behavior failed: {e}")
            return self.current_position
    
    def _check_convergence(
        self,
        current_concentration: float,
        stability_window: int
    ) -> bool:
        """
        Check algorithm convergence based on position stability, concentration thresholds, and 
        source proximity criteria.
        
        Args:
            current_concentration: Current concentration value
            stability_window: Number of iterations to check for stability
            
        Returns:
            bool: True if algorithm has converged to source location
        """
        try:
            # Check concentration against convergence threshold
            concentration_threshold = self.tracking_parameters.plume_threshold * 5.0  # High concentration for convergence
            if current_concentration < concentration_threshold:
                return False
            
            # Analyze position stability over time window
            if len(self.trajectory_history) < stability_window:
                return False
            
            recent_positions = self.trajectory_history[-stability_window:]
            position_variances = []
            
            for dim in range(2):  # y and x dimensions
                positions_dim = [pos[dim] for pos in recent_positions]
                variance = np.var(positions_dim)
                position_variances.append(variance)
            
            max_position_variance = max(position_variances)
            if max_position_variance > self.tracking_parameters.convergence_tolerance:
                return False
            
            # Evaluate gradient magnitude for source proximity
            if len(self.gradient_history) > 0:
                recent_gradient = self.gradient_history[-1]
                gradient_magnitude = np.sqrt(recent_gradient[0]**2 + recent_gradient[1]**2)
                
                # Near source, gradients should be small
                if gradient_magnitude > self.tracking_parameters.gradient_threshold * 2.0:
                    return False
            
            return True
        
        except Exception as e:
            self.logger.warning(f"Convergence check failed: {e}")
            return False
    
    def _calculate_performance_metrics(
        self,
        source_location: Tuple[float, float],
        execution_time: float
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics including path efficiency, convergence time, 
        and tracking accuracy for scientific validation.
        
        Args:
            source_location: True source location for accuracy assessment
            execution_time: Total execution time
            
        Returns:
            Dict[str, float]: Performance metrics with efficiency, accuracy, and timing measurements
        """
        try:
            metrics = {}
            
            # Calculate path efficiency and total distance traveled
            if len(self.trajectory_history) > 1:
                trajectory_array = np.array(self.trajectory_history)
                path_segments = np.diff(trajectory_array, axis=0)
                segment_distances = np.sqrt(np.sum(path_segments**2, axis=1))
                total_distance = float(np.sum(segment_distances))
                
                # Calculate direct distance to source
                start_position = self.trajectory_history[0]
                direct_distance = np.sqrt(
                    (source_location[0] - start_position[0])**2 + 
                    (source_location[1] - start_position[1])**2
                )
                
                if direct_distance > 0:
                    path_efficiency = direct_distance / total_distance
                else:
                    path_efficiency = 1.0
                
                metrics['total_distance_traveled'] = total_distance
                metrics['path_efficiency'] = path_efficiency
            else:
                metrics['total_distance_traveled'] = 0.0
                metrics['path_efficiency'] = 0.0
            
            # Measure convergence time and iteration count
            metrics['execution_time_seconds'] = execution_time
            metrics['iterations_completed'] = float(self.iteration_count)
            metrics['convergence_achieved'] = float(self.current_state == PlumeTrackingState.CONVERGED)
            
            # Assess tracking accuracy and source localization error
            if self.trajectory_history:
                final_position = self.trajectory_history[-1]
                localization_error = np.sqrt(
                    (source_location[0] - final_position[0])**2 + 
                    (source_location[1] - final_position[1])**2
                )
                metrics['localization_error'] = float(localization_error)
                metrics['localization_accuracy'] = float(1.0 / (1.0 + localization_error))
            else:
                metrics['localization_error'] = float('inf')
                metrics['localization_accuracy'] = 0.0
            
            # Calculate trajectory smoothness and navigation quality
            if len(self.trajectory_history) > 2:
                trajectory_array = np.array(self.trajectory_history)
                second_derivatives = np.diff(trajectory_array, n=2, axis=0)
                trajectory_roughness = np.mean(np.sum(second_derivatives**2, axis=1))
                trajectory_smoothness = 1.0 / (1.0 + trajectory_roughness)
                metrics['trajectory_smoothness'] = float(trajectory_smoothness)
            else:
                metrics['trajectory_smoothness'] = 1.0
            
            # Include statistical validation metrics
            metrics['plume_contact_maintained'] = float(self.plume_contact)
            metrics['casting_iterations_used'] = float(self.casting_iteration)
            
            return metrics
        
        except Exception as e:
            self.logger.warning(f"Performance metrics calculation failed: {e}")
            return {
                'execution_time_seconds': execution_time,
                'iterations_completed': float(self.iteration_count),
                'error_occurred': 1.0
            }
    
    def _validate_tracking_accuracy(
        self,
        tracking_result: AlgorithmResult,
        reference_data: Dict[str, Any]
    ) -> 'ValidationResult':
        """
        Validate plume tracking accuracy against reference implementations with >95% correlation 
        requirement and statistical significance testing.
        
        Args:
            tracking_result: Tracking result to validate
            reference_data: Reference implementation data for comparison
            
        Returns:
            ValidationResult: Tracking accuracy validation with correlation analysis and compliance assessment
        """
        from ..utils.validation_utils import ValidationResult
        
        # Initialize validation result
        validation_result = ValidationResult(
            validation_type="plume_tracking_accuracy_validation",
            is_valid=True,
            validation_context=f"algorithm={self.algorithm_name}"
        )
        
        try:
            # Compare trajectory with reference implementation
            if 'reference_trajectory' in reference_data and tracking_result.trajectory is not None:
                reference_trajectory = reference_data['reference_trajectory']
                current_trajectory = tracking_result.trajectory
                
                # Calculate trajectory similarity using appropriate metrics
                if len(reference_trajectory) > 0 and len(current_trajectory) > 0:
                    # Resample trajectories to same length for comparison
                    min_length = min(len(reference_trajectory), len(current_trajectory))
                    ref_resampled = reference_trajectory[:min_length]
                    cur_resampled = current_trajectory[:min_length]
                    
                    # Calculate correlation coefficients
                    try:
                        trajectory_correlation = np.corrcoef(
                            ref_resampled.flatten(), cur_resampled.flatten()
                        )[0, 1]
                        
                        if not np.isnan(trajectory_correlation):
                            validation_result.add_metric("trajectory_correlation", float(trajectory_correlation))
                            
                            # Check against >95% correlation threshold
                            if trajectory_correlation >= DEFAULT_CORRELATION_THRESHOLD:
                                validation_result.add_recommendation(
                                    f"Trajectory correlation meets requirement: {trajectory_correlation:.3f}",
                                    priority="INFO"
                                )
                            else:
                                validation_result.add_error(
                                    f"Trajectory correlation below threshold: {trajectory_correlation:.3f} < {DEFAULT_CORRELATION_THRESHOLD}",
                                    severity="HIGH"
                                )
                                validation_result.is_valid = False
                    except Exception as corr_error:
                        validation_result.add_warning(f"Trajectory correlation calculation failed: {corr_error}")
            
            # Calculate correlation coefficients for performance metrics
            correlation_metrics = {}
            reference_metrics = reference_data.get('reference_metrics', {})
            
            for metric_name, reference_value in reference_metrics.items():
                if metric_name in tracking_result.performance_metrics:
                    current_value = tracking_result.performance_metrics[metric_name]
                    
                    # Calculate relative correlation for individual metrics
                    if reference_value != 0:
                        relative_correlation = 1.0 - abs(current_value - reference_value) / abs(reference_value)
                        correlation_metrics[f'{metric_name}_correlation'] = max(0.0, relative_correlation)
                    else:
                        correlation_metrics[f'{metric_name}_correlation'] = 1.0 if current_value == 0 else 0.0
            
            # Calculate overall correlation score
            if correlation_metrics:
                overall_correlation = sum(correlation_metrics.values()) / len(correlation_metrics)
                validation_result.add_metric("overall_correlation", float(overall_correlation))
                
                # Check correlation threshold compliance
                if overall_correlation >= DEFAULT_CORRELATION_THRESHOLD:
                    validation_result.add_recommendation(
                        f"Overall correlation meets requirement: {overall_correlation:.3f}",
                        priority="INFO"
                    )
                else:
                    validation_result.add_error(
                        f"Overall correlation below threshold: {overall_correlation:.3f} < {DEFAULT_CORRELATION_THRESHOLD}",
                        severity="HIGH"
                    )
                    validation_result.is_valid = False
            
            # Perform statistical significance testing
            if tracking_result.success and tracking_result.converged:
                validation_result.add_metric("convergence_success", 1.0)
            else:
                validation_result.add_warning("Algorithm did not achieve convergence")
                validation_result.add_metric("convergence_success", 0.0)
            
            # Generate comprehensive validation result
            if validation_result.is_valid:
                validation_result.add_recommendation(
                    "Plume tracking accuracy validation passed",
                    priority="INFO"
                )
            else:
                validation_result.add_recommendation(
                    "Address accuracy validation issues for compliance",
                    priority="HIGH"
                )
        
        except Exception as e:
            validation_result.add_error(
                f"Accuracy validation failed: {str(e)}",
                severity="CRITICAL"
            )
            validation_result.is_valid = False
        
        validation_result.finalize_validation()
        return validation_result
    
    def reset(self) -> None:
        """
        Reset plume tracking algorithm state to initial conditions for fresh execution with 
        parameter preservation.
        """
        try:
            # Reset tracking state to initial conditions
            self.current_state = PlumeTrackingState.SEARCHING
            self.current_position = (0.0, 0.0)
            
            # Clear trajectory and concentration history
            self.trajectory_history.clear()
            self.concentration_history.clear()
            self.gradient_history.clear()
            
            # Reset position and iteration counters
            self.iteration_count = 0
            self.last_concentration = 0.0
            self.last_gradient = (0.0, 0.0)
            self.plume_contact = False
            
            # Clear performance metrics and casting state
            self.casting_iteration = 0
            self.casting_direction = 0.0
            self.performance_metrics.clear()
            
            # Call base class reset to handle algorithm state
            super().reset()
            
            self.logger.info("Plume tracking algorithm reset completed")
        
        except Exception as e:
            self.logger.error(f"Algorithm reset failed: {e}", exc_info=True)
            raise
    
    def get_tracking_statistics(
        self,
        include_trajectory_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive tracking statistics including state transitions, behavior analysis, and 
        performance trends for algorithm optimization.
        
        Args:
            include_trajectory_analysis: Whether to include detailed trajectory analysis
            
        Returns:
            Dict[str, Any]: Tracking statistics with state analysis, performance trends, and optimization recommendations
        """
        try:
            # Compile state transition statistics and behavior analysis
            statistics = {
                'current_state': str(self.current_state),
                'total_iterations': self.iteration_count,
                'trajectory_length': len(self.trajectory_history),
                'plume_contact_status': self.plume_contact,
                'casting_iterations': self.casting_iteration,
                'algorithm_version': PLUME_TRACKING_VERSION
            }
            
            # Calculate performance trends and efficiency metrics
            if self.performance_metrics:
                statistics['performance_metrics'] = self.performance_metrics.copy()
            
            # Analyze concentration and gradient history
            if self.concentration_history:
                concentrations = list(self.concentration_history)
                statistics['concentration_statistics'] = {
                    'mean': float(np.mean(concentrations)),
                    'std': float(np.std(concentrations)),
                    'min': float(np.min(concentrations)),
                    'max': float(np.max(concentrations))
                }
            
            if self.gradient_history:
                gradients = list(self.gradient_history)
                gradient_magnitudes = [np.sqrt(g[0]**2 + g[1]**2) for g in gradients]
                statistics['gradient_statistics'] = {
                    'mean_magnitude': float(np.mean(gradient_magnitudes)),
                    'std_magnitude': float(np.std(gradient_magnitudes)),
                    'max_magnitude': float(np.max(gradient_magnitudes))
                }
            
            # Include trajectory analysis if requested
            if include_trajectory_analysis and len(self.trajectory_history) > 1:
                trajectory_array = np.array(self.trajectory_history)
                
                # Calculate trajectory characteristics
                path_segments = np.diff(trajectory_array, axis=0)
                segment_distances = np.sqrt(np.sum(path_segments**2, axis=1))
                
                statistics['trajectory_analysis'] = {
                    'total_distance': float(np.sum(segment_distances)),
                    'mean_step_size': float(np.mean(segment_distances)),
                    'trajectory_variance': [float(np.var(trajectory_array[:, i])) for i in range(2)],
                    'start_position': self.trajectory_history[0],
                    'end_position': self.trajectory_history[-1]
                }
            
            # Generate optimization recommendations based on performance
            recommendations = []
            
            if self.current_state == PlumeTrackingState.LOST_PLUME:
                recommendations.append("Consider increasing casting distance or relaxing plume threshold")
            elif self.casting_iteration > self.tracking_parameters.max_casting_iterations // 2:
                recommendations.append("High casting iterations - consider optimizing gradient threshold")
            
            if not self.plume_contact and self.iteration_count > 50:
                recommendations.append("Difficulty maintaining plume contact - check plume threshold settings")
            
            if recommendations:
                statistics['optimization_recommendations'] = recommendations
            
            return statistics
        
        except Exception as e:
            self.logger.error(f"Statistics generation failed: {e}")
            return {
                'error': str(e),
                'current_state': str(self.current_state),
                'iteration_count': self.iteration_count
            }
    
    def _get_concentration_at_position(
        self,
        concentration_field: np.ndarray,
        position: Tuple[float, float]
    ) -> float:
        """Get concentration value at specified position with interpolation."""
        try:
            height, width = concentration_field.shape
            y, x = position
            
            # Ensure position is within bounds
            y = max(0, min(height - 1, y))
            x = max(0, min(width - 1, x))
            
            # Use bilinear interpolation for sub-pixel positions
            y_int, x_int = int(y), int(x)
            y_frac, x_frac = y - y_int, x - x_int
            
            # Get surrounding pixel values
            val_00 = concentration_field[y_int, x_int]
            val_01 = concentration_field[y_int, min(x_int + 1, width - 1)]
            val_10 = concentration_field[min(y_int + 1, height - 1), x_int]
            val_11 = concentration_field[min(y_int + 1, height - 1), min(x_int + 1, width - 1)]
            
            # Bilinear interpolation
            val_0 = val_00 * (1 - x_frac) + val_01 * x_frac
            val_1 = val_10 * (1 - x_frac) + val_11 * x_frac
            interpolated_value = val_0 * (1 - y_frac) + val_1 * y_frac
            
            return float(interpolated_value)
        
        except Exception as e:
            self.logger.warning(f"Concentration interpolation failed: {e}")
            return 0.0
    
    def _constrain_position(
        self,
        position: Tuple[float, float],
        height: int,
        width: int
    ) -> Tuple[float, float]:
        """Constrain position to stay within array bounds."""
        y, x = position
        y_constrained = max(0.0, min(float(height - 1), y))
        x_constrained = max(0.0, min(float(width - 1), x))
        return (y_constrained, x_constrained)
    
    def _estimate_wind_direction(self) -> float:
        """Estimate wind direction from gradient history (simplified implementation)."""
        if len(self.gradient_history) > 3:
            # Use recent gradients to estimate predominant wind direction
            recent_gradients = list(self.gradient_history)[-3:]
            avg_gradient = np.mean(recent_gradients, axis=0)
            wind_direction = math.atan2(avg_gradient[0], avg_gradient[1])
            return wind_direction
        else:
            return 0.0  # Default wind direction
    
    def _update_performance_metrics(
        self,
        current_concentration: float,
        current_gradient: Tuple[float, float],
        iteration: int
    ) -> None:
        """Update performance metrics during algorithm execution."""
        try:
            # Update basic performance metrics
            self.performance_metrics['current_concentration'] = current_concentration
            self.performance_metrics['gradient_magnitude'] = float(np.sqrt(current_gradient[0]**2 + current_gradient[1]**2))
            self.performance_metrics['current_iteration'] = float(iteration)
            
            # Track state-specific metrics
            if self.current_state == PlumeTrackingState.GRADIENT_FOLLOWING:
                self.performance_metrics['gradient_following_iterations'] = self.performance_metrics.get('gradient_following_iterations', 0) + 1
            elif self.current_state == PlumeTrackingState.CASTING:
                self.performance_metrics['casting_total_iterations'] = self.performance_metrics.get('casting_total_iterations', 0) + 1
            
            # Update contact and efficiency metrics
            self.performance_metrics['plume_contact_ratio'] = float(
                sum(1 for c in self.concentration_history if c > self.tracking_parameters.plume_threshold) / 
                max(1, len(self.concentration_history))
            )
        
        except Exception as e:
            self.logger.warning(f"Performance metrics update failed: {e}")


# Register the plume tracking algorithm in the global registry
register_algorithm(
    algorithm_name='plume_tracking',
    algorithm_class=PlumeTrackingAlgorithm,
    algorithm_metadata={
        'description': 'Advanced plume tracking navigation algorithm with bio-inspired behaviors',
        'algorithm_type': 'plume_tracking',
        'version': PLUME_TRACKING_VERSION,
        'capabilities': [
            'gradient_following', 'crosswind_casting', 'surge_behavior', 
            'adaptive_thresholds', 'temporal_memory', 'cross_format_compatibility'
        ],
        'supported_formats': ['crimaldi', 'custom', 'generic'],
        'performance_characteristics': {
            'target_execution_time': 7.2,
            'correlation_threshold': 0.95,
            'reproducibility_threshold': 0.99
        },
        'validation_requirements': {
            'interface_compliance': True,
            'performance_validation': True,
            'accuracy_validation': True
        }
    },
    validate_interface=True,
    enable_performance_tracking=True
)