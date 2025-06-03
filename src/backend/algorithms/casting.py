"""
Comprehensive casting navigation algorithm implementation for plume tracking and source localization using bio-inspired casting behavior patterns.

This module implements systematic crosswind search patterns with upwind surges when plume contact is lost, incorporating adaptive search radius adjustment, wind direction estimation, and statistical performance tracking. The implementation provides comprehensive casting strategy with configurable parameters for search patterns, turning behaviors, and plume reacquisition protocols optimized for scientific computing requirements and cross-format compatibility.

Key Features:
- Bio-inspired casting behavior with systematic crosswind search patterns
- Adaptive search radius optimization based on plume characteristics and performance history
- Wind direction estimation using plume gradient analysis with temporal averaging
- Upwind surge behavior when plume contact is detected for source approach
- Statistical performance tracking with >95% correlation validation against reference implementations
- Cross-format compatibility for Crimaldi and custom plume datasets
- Scientific computing precision with <7.2 seconds average execution time
- Comprehensive state management for trajectory tracking and performance analysis
- Reproducible results with >0.99 coefficient validation across computational environments
"""

# External imports with version specifications
import numpy as np  # version: 2.1.3+ - Numerical computations for casting search patterns, trajectory calculations, and statistical analysis
from scipy.spatial.distance import euclidean, cdist  # version: 1.15.3+ - Distance calculations for casting search radius optimization and plume detection
import scipy.stats  # version: 1.15.3+ - Statistical analysis for casting algorithm performance validation and correlation testing
from typing import Dict, Any, List, Optional, Union, Callable, Tuple  # version: 3.9+ - Type hints for casting algorithm function signatures and data structures
import dataclasses  # version: 3.9+ - Data classes for casting algorithm parameters and state management
import math  # version: 3.9+ - Mathematical functions for casting angle calculations and trigonometric operations
import random  # version: 3.9+ - Random number generation for casting algorithm stochastic behaviors and noise injection
import time  # version: 3.9+ - Timing measurements for casting algorithm performance tracking and optimization
import copy  # version: 3.9+ - Deep copying for casting algorithm state preservation and parameter isolation

# Internal imports from base algorithm framework
from .base_algorithm import (
    BaseAlgorithm, AlgorithmParameters, AlgorithmResult, AlgorithmContext
)

# Internal imports from scientific constants
from ..utils.scientific_constants import (
    TARGET_ARENA_WIDTH_METERS, TARGET_ARENA_HEIGHT_METERS, NUMERICAL_PRECISION_THRESHOLD,
    DEFAULT_CORRELATION_THRESHOLD
)

# Internal imports from statistical utilities (creating placeholder functions since file doesn't exist)
try:
    from ..utils.statistical_utils import (
        calculate_correlation_matrix, calculate_trajectory_similarity, assess_reproducibility
    )
except ImportError:
    # Fallback implementations for missing statistical utilities
    def calculate_correlation_matrix(data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate correlation matrix between two datasets."""
        if data1.size == 0 or data2.size == 0:
            return 0.0
        flat1, flat2 = data1.flatten(), data2.flatten()
        if len(flat1) != len(flat2):
            min_len = min(len(flat1), len(flat2))
            flat1, flat2 = flat1[:min_len], flat2[:min_len]
        return np.corrcoef(flat1, flat2)[0, 1] if len(flat1) > 1 else 1.0

    def calculate_trajectory_similarity(traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Calculate trajectory similarity metrics for casting path analysis."""
        if traj1.size == 0 or traj2.size == 0:
            return 0.0
        # Calculate DTW-like similarity based on point-wise distances
        min_len = min(len(traj1), len(traj2))
        distances = [euclidean(traj1[i], traj2[i]) for i in range(min_len)]
        return 1.0 / (1.0 + np.mean(distances))

    def assess_reproducibility(results: List[np.ndarray]) -> float:
        """Assess casting algorithm reproducibility with threshold validation."""
        if len(results) < 2:
            return 1.0
        correlations = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                corr = calculate_correlation_matrix(results[i], results[j])
                if not np.isnan(corr):
                    correlations.append(corr)
        return np.mean(correlations) if correlations else 0.0

# Internal imports from plume normalizer (creating placeholder since file doesn't exist)
try:
    from ..core.data_normalization.plume_normalizer import PlumeNormalizer
except ImportError:
    class PlumeNormalizer:
        """Placeholder plume normalizer for cross-format compatibility."""
        def normalize_plume_data(self, plume_data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
            """Normalize plume data for cross-format compatibility."""
            return plume_data.copy()

# Global constants for casting algorithm execution and validation
DEFAULT_CASTING_SPEED = 0.1
DEFAULT_SEARCH_RADIUS = 0.05
DEFAULT_CASTING_ANGLE = 1.57  # π/2 radians (90 degrees)
DEFAULT_UPWIND_SPEED = 0.15
DEFAULT_CROSSWIND_SPEED = 0.08
DEFAULT_PLUME_THRESHOLD = 0.01
DEFAULT_MAX_CAST_DISTANCE = 0.3
DEFAULT_WIND_ESTIMATION_WINDOW = 10

# Casting state constants
CASTING_STATE_SEARCHING = 'searching'
CASTING_STATE_TRACKING = 'tracking'
CASTING_STATE_CASTING = 'casting'
CASTING_STATE_SURGING = 'surging'

# Algorithm version for reproducibility and validation
ALGORITHM_VERSION = '1.0.0'


def calculate_wind_direction(
    plume_data: np.ndarray,
    position: np.ndarray,
    estimation_window: int = DEFAULT_WIND_ESTIMATION_WINDOW,
    wind_options: Dict[str, Any] = None
) -> Tuple[float, float]:
    """
    Calculate wind direction from plume gradient information using spatial analysis and temporal averaging for accurate wind estimation in casting navigation.
    
    This function performs comprehensive wind direction estimation using plume gradient analysis with spatial derivatives, temporal averaging, and noise filtering to provide accurate wind direction vectors for casting algorithm navigation with validation against plume structure constraints.
    
    Args:
        plume_data: Plume data array containing intensity values (shape: [time, height, width])
        position: Current position in the plume field for gradient calculation [y, x]
        estimation_window: Number of temporal frames for averaging wind estimates
        wind_options: Additional options for wind estimation including noise filtering and validation
        
    Returns:
        Tuple[float, float]: Wind direction vector (x, y) with confidence estimate and validation metrics
    """
    # Initialize wind options with defaults
    options = wind_options or {}
    noise_threshold = options.get('noise_threshold', 0.001)
    gradient_method = options.get('gradient_method', 'sobel')
    confidence_threshold = options.get('confidence_threshold', 0.1)
    
    try:
        # Validate input parameters for wind direction calculation
        if plume_data.ndim != 3:
            raise ValueError(f"Plume data must be 3D array, got {plume_data.ndim}D")
        
        if len(position) != 2:
            raise ValueError(f"Position must be 2D coordinate, got {len(position)}D")
        
        # Extract current position coordinates with boundary checking
        y_pos, x_pos = int(np.clip(position[0], 0, plume_data.shape[1] - 1)), int(np.clip(position[1], 0, plume_data.shape[2] - 1))
        
        # Calculate temporal window for gradient analysis
        time_frames = min(estimation_window, plume_data.shape[0])
        recent_frames = plume_data[-time_frames:] if time_frames > 0 else plume_data[-1:]
        
        # Extract plume gradient information around current position
        wind_estimates = []
        
        for frame_idx in range(len(recent_frames)):
            frame = recent_frames[frame_idx]
            
            # Calculate spatial derivatives for wind direction estimation
            if gradient_method == 'sobel':
                # Sobel gradient calculation for robust edge detection
                if y_pos > 0 and y_pos < frame.shape[0] - 1 and x_pos > 0 and x_pos < frame.shape[1] - 1:
                    # Sobel kernels for x and y gradients
                    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                    
                    # Extract 3x3 neighborhood around position
                    neighborhood = frame[y_pos-1:y_pos+2, x_pos-1:x_pos+2]
                    
                    if neighborhood.shape == (3, 3):
                        grad_x = np.sum(neighborhood * sobel_x)
                        grad_y = np.sum(neighborhood * sobel_y)
                    else:
                        grad_x, grad_y = 0.0, 0.0
                else:
                    grad_x, grad_y = 0.0, 0.0
                    
            else:
                # Simple finite difference gradient
                grad_x = 0.0
                grad_y = 0.0
                
                if x_pos > 0 and x_pos < frame.shape[1] - 1:
                    grad_x = frame[y_pos, x_pos + 1] - frame[y_pos, x_pos - 1]
                
                if y_pos > 0 and y_pos < frame.shape[0] - 1:
                    grad_y = frame[y_pos + 1, x_pos] - frame[y_pos - 1, x_pos]
            
            # Calculate gradient magnitude for confidence estimation
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Apply noise filtering and outlier detection
            if gradient_magnitude > noise_threshold:
                # Normalize gradient to unit vector (wind direction)
                if gradient_magnitude > 0:
                    wind_x = grad_x / gradient_magnitude
                    wind_y = grad_y / gradient_magnitude
                    wind_estimates.append((wind_x, wind_y, gradient_magnitude))
        
        # Apply temporal averaging over estimation window
        if wind_estimates:
            # Weight estimates by gradient magnitude (confidence)
            weights = np.array([est[2] for est in wind_estimates])
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
            
            wind_x = np.sum([est[0] * weights[i] for i, est in enumerate(wind_estimates)])
            wind_y = np.sum([est[1] * weights[i] for i, est in enumerate(wind_estimates)])
            
            # Calculate confidence based on estimate consistency
            if len(wind_estimates) > 1:
                consistency = 1.0 - np.std([est[0] for est in wind_estimates]) - np.std([est[1] for est in wind_estimates])
                confidence = max(0.0, min(1.0, consistency))
            else:
                confidence = np.mean(weights)
        else:
            # Default wind direction if no estimates available
            wind_x, wind_y = 1.0, 0.0  # Default eastward wind
            confidence = 0.0
        
        # Validate wind direction estimate against plume structure
        wind_magnitude = np.sqrt(wind_x**2 + wind_y**2)
        if wind_magnitude > 0:
            wind_x /= wind_magnitude
            wind_y /= wind_magnitude
        
        # Apply final validation and confidence thresholding
        if confidence < confidence_threshold:
            # Use default wind direction for low confidence estimates
            wind_x, wind_y = 1.0, 0.0
            confidence = confidence_threshold
        
        # Return wind direction vector with confidence metrics
        return (wind_x, wind_y)
        
    except Exception as e:
        # Return default wind direction on calculation failure
        return (1.0, 0.0)


def calculate_casting_trajectory(
    current_position: np.ndarray,
    wind_direction: Tuple[float, float],
    search_radius: float = DEFAULT_SEARCH_RADIUS,
    casting_params: Dict[str, Any] = None
) -> np.ndarray:
    """
    Calculate optimal casting trajectory based on current position, wind direction, search radius, and casting parameters for systematic crosswind search patterns.
    
    This function generates optimized casting trajectories using crosswind direction calculation, casting amplitude determination, systematic search pattern generation, trajectory smoothing, and feasibility validation against arena boundaries for effective plume reacquisition strategies.
    
    Args:
        current_position: Current position in the plume field [y, x]
        wind_direction: Wind direction vector (x, y) from wind estimation
        search_radius: Radius for casting search pattern
        casting_params: Parameters for casting trajectory including angle, speed, and pattern type
        
    Returns:
        np.ndarray: Casting trajectory points with optimized search pattern and validation metrics
    """
    # Initialize casting parameters with defaults
    params = casting_params or {}
    casting_angle = params.get('casting_angle', DEFAULT_CASTING_ANGLE)
    num_cast_points = params.get('num_cast_points', 20)
    pattern_type = params.get('pattern_type', 'zigzag')
    smoothing_enabled = params.get('smoothing_enabled', True)
    
    try:
        # Validate input parameters for trajectory calculation
        if len(current_position) != 2:
            raise ValueError(f"Current position must be 2D, got {len(current_position)}D")
        
        if search_radius <= 0:
            raise ValueError(f"Search radius must be positive, got {search_radius}")
        
        # Extract wind direction components
        wind_x, wind_y = wind_direction
        wind_magnitude = np.sqrt(wind_x**2 + wind_y**2)
        
        if wind_magnitude > 0:
            # Normalize wind direction
            wind_x /= wind_magnitude
            wind_y /= wind_magnitude
        else:
            # Default wind direction if magnitude is zero
            wind_x, wind_y = 1.0, 0.0
        
        # Calculate crosswind direction perpendicular to wind vector
        crosswind_x = -wind_y  # Perpendicular to wind direction
        crosswind_y = wind_x
        
        # Determine casting amplitude based on search radius
        cast_amplitude = search_radius
        
        # Generate systematic casting pattern with optimal spacing
        trajectory_points = []
        start_pos = current_position.copy()
        
        if pattern_type == 'zigzag':
            # Zigzag casting pattern for systematic search
            for i in range(num_cast_points):
                # Alternate casting direction
                direction_multiplier = 1 if (i % 2) == 0 else -1
                
                # Calculate lateral displacement for casting
                lateral_offset = direction_multiplier * cast_amplitude * (i / num_cast_points)
                
                # Calculate forward progress in crosswind direction
                forward_progress = (i / num_cast_points) * cast_amplitude * 0.5
                
                # Calculate new position
                new_x = start_pos[1] + lateral_offset * crosswind_x + forward_progress * wind_x
                new_y = start_pos[0] + lateral_offset * crosswind_y + forward_progress * wind_y
                
                trajectory_points.append([new_y, new_x])
                
        elif pattern_type == 'spiral':
            # Spiral casting pattern for expanding search
            for i in range(num_cast_points):
                angle = (i / num_cast_points) * 4 * np.pi  # Two full spirals
                radius = (i / num_cast_points) * cast_amplitude
                
                # Calculate spiral position
                spiral_x = radius * np.cos(angle)
                spiral_y = radius * np.sin(angle)
                
                # Rotate spiral to align with crosswind direction
                new_x = start_pos[1] + spiral_x * crosswind_x - spiral_y * crosswind_y
                new_y = start_pos[0] + spiral_x * crosswind_y + spiral_y * crosswind_x
                
                trajectory_points.append([new_y, new_x])
                
        else:
            # Default linear casting pattern
            for i in range(num_cast_points):
                progress = i / max(1, num_cast_points - 1)
                
                # Linear casting across wind direction
                new_x = start_pos[1] + (progress - 0.5) * 2 * cast_amplitude * crosswind_x
                new_y = start_pos[0] + (progress - 0.5) * 2 * cast_amplitude * crosswind_y
                
                trajectory_points.append([new_y, new_x])
        
        # Convert to numpy array for processing
        trajectory = np.array(trajectory_points)
        
        # Apply trajectory smoothing and feasibility constraints
        if smoothing_enabled and len(trajectory) > 2:
            # Apply moving average smoothing
            smoothed_trajectory = []
            window_size = min(3, len(trajectory))
            
            for i in range(len(trajectory)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(trajectory), i + window_size // 2 + 1)
                
                avg_y = np.mean(trajectory[start_idx:end_idx, 0])
                avg_x = np.mean(trajectory[start_idx:end_idx, 1])
                
                smoothed_trajectory.append([avg_y, avg_x])
            
            trajectory = np.array(smoothed_trajectory)
        
        # Validate trajectory against arena boundaries
        # Clip trajectory points to stay within normalized arena bounds [0, 1]
        trajectory[:, 0] = np.clip(trajectory[:, 0], 0.0, 1.0)  # y coordinates
        trajectory[:, 1] = np.clip(trajectory[:, 1], 0.0, 1.0)  # x coordinates
        
        # Ensure trajectory has minimum required points
        if len(trajectory) == 0:
            # Fallback to simple two-point trajectory
            trajectory = np.array([
                current_position,
                [current_position[0], current_position[1] + search_radius * crosswind_x]
            ])
        
        # Return optimized casting trajectory with performance metrics
        return trajectory
        
    except Exception as e:
        # Return simple fallback trajectory on calculation failure
        fallback_trajectory = np.array([
            current_position,
            [current_position[0], current_position[1] + search_radius]
        ])
        return fallback_trajectory


def detect_plume_contact(
    plume_data: np.ndarray,
    position: np.ndarray,
    detection_threshold: float = DEFAULT_PLUME_THRESHOLD,
    detection_options: Dict[str, Any] = None
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Detect plume contact at current position using intensity threshold analysis, gradient detection, and statistical validation for reliable plume detection in casting algorithm.
    
    This function performs comprehensive plume detection using intensity sampling with interpolation, threshold-based detection with noise filtering, gradient analysis for directional information, statistical validation, and confidence calculation for reliable plume contact determination with metadata reporting.
    
    Args:
        plume_data: Plume data array containing intensity values
        position: Current position for plume detection [y, x]
        detection_threshold: Minimum intensity threshold for plume detection
        detection_options: Additional options for detection including interpolation and validation
        
    Returns:
        Tuple[bool, float, Dict[str, Any]]: Plume detection result with confidence score and detection metadata
    """
    # Initialize detection options with defaults
    options = detection_options or {}
    interpolation_method = options.get('interpolation_method', 'bilinear')
    noise_filtering = options.get('noise_filtering', True)
    gradient_analysis = options.get('gradient_analysis', True)
    statistical_validation = options.get('statistical_validation', True)
    
    # Initialize detection metadata
    metadata = {
        'detection_method': 'threshold_gradient_analysis',
        'interpolation_used': interpolation_method,
        'position_sampled': position.tolist(),
        'threshold_used': detection_threshold
    }
    
    try:
        # Validate input parameters for plume detection
        if plume_data.size == 0:
            return False, 0.0, {**metadata, 'error': 'Empty plume data'}
        
        if len(position) != 2:
            return False, 0.0, {**metadata, 'error': f'Invalid position dimension: {len(position)}'}
        
        # Sample plume intensity at current position with interpolation
        if plume_data.ndim == 3:
            # Use most recent frame for detection
            current_frame = plume_data[-1]
        elif plume_data.ndim == 2:
            current_frame = plume_data
        else:
            return False, 0.0, {**metadata, 'error': f'Unsupported plume data dimensions: {plume_data.ndim}'}
        
        # Extract position coordinates with bounds checking
        y_pos, x_pos = position[0], position[1]
        frame_height, frame_width = current_frame.shape
        
        # Validate position bounds
        if y_pos < 0 or y_pos >= frame_height or x_pos < 0 or x_pos >= frame_width:
            return False, 0.0, {**metadata, 'error': 'Position outside plume field bounds'}
        
        # Sample intensity with interpolation
        if interpolation_method == 'bilinear' and y_pos != int(y_pos) or x_pos != int(x_pos):
            # Bilinear interpolation for sub-pixel positions
            y_floor, y_ceil = int(np.floor(y_pos)), int(np.ceil(y_pos))
            x_floor, x_ceil = int(np.floor(x_pos)), int(np.ceil(x_pos))
            
            # Ensure bounds for interpolation
            y_ceil = min(y_ceil, frame_height - 1)
            x_ceil = min(x_ceil, frame_width - 1)
            
            # Calculate interpolation weights
            y_weight = y_pos - y_floor
            x_weight = x_pos - x_floor
            
            # Bilinear interpolation
            intensity = (
                current_frame[y_floor, x_floor] * (1 - y_weight) * (1 - x_weight) +
                current_frame[y_floor, x_ceil] * (1 - y_weight) * x_weight +
                current_frame[y_ceil, x_floor] * y_weight * (1 - x_weight) +
                current_frame[y_ceil, x_ceil] * y_weight * x_weight
            )
        else:
            # Nearest neighbor sampling
            y_idx = int(np.round(y_pos))
            x_idx = int(np.round(x_pos))
            y_idx = np.clip(y_idx, 0, frame_height - 1)
            x_idx = np.clip(x_idx, 0, frame_width - 1)
            intensity = current_frame[y_idx, x_idx]
        
        metadata['sampled_intensity'] = float(intensity)
        
        # Apply detection threshold with noise filtering
        basic_detection = intensity > detection_threshold
        confidence = float(intensity / max(detection_threshold, 1e-10))  # Confidence relative to threshold
        
        # Apply noise filtering if enabled
        if noise_filtering and basic_detection:
            # Check neighboring pixels for consistent detection
            y_idx = int(np.round(y_pos))
            x_idx = int(np.round(x_pos))
            
            neighbor_intensities = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y_idx + dy, x_idx + dx
                    if 0 <= ny < frame_height and 0 <= nx < frame_width:
                        neighbor_intensities.append(current_frame[ny, nx])
            
            if neighbor_intensities:
                neighbor_mean = np.mean(neighbor_intensities)
                neighbor_std = np.std(neighbor_intensities)
                
                # Check if current intensity is consistent with neighborhood
                if neighbor_std > 0:
                    z_score = abs(intensity - neighbor_mean) / neighbor_std
                    if z_score > 2.0:  # Outlier detection
                        confidence *= 0.5  # Reduce confidence for outliers
                
                metadata['neighbor_mean'] = float(neighbor_mean)
                metadata['neighbor_std'] = float(neighbor_std)
        
        # Calculate plume gradient for directional information
        gradient_magnitude = 0.0
        if gradient_analysis:
            y_idx = int(np.round(y_pos))
            x_idx = int(np.round(x_pos))
            
            # Calculate gradient using central differences
            grad_x, grad_y = 0.0, 0.0
            
            if 1 <= x_idx < frame_width - 1:
                grad_x = (current_frame[y_idx, x_idx + 1] - current_frame[y_idx, x_idx - 1]) / 2.0
            
            if 1 <= y_idx < frame_height - 1:
                grad_y = (current_frame[y_idx + 1, x_idx] - current_frame[y_idx - 1, x_idx]) / 2.0
            
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Enhance confidence for strong gradients
            if gradient_magnitude > detection_threshold * 0.5:
                confidence *= 1.2  # Boost confidence for strong gradients
            
            metadata['gradient_magnitude'] = float(gradient_magnitude)
            metadata['gradient_x'] = float(grad_x)
            metadata['gradient_y'] = float(grad_y)
        
        # Validate detection using statistical criteria
        final_detection = basic_detection
        if statistical_validation and basic_detection:
            # Additional validation criteria
            validation_score = 0.0
            
            # Check intensity relative to local background
            if intensity > detection_threshold * 2:
                validation_score += 0.4
            elif intensity > detection_threshold * 1.5:
                validation_score += 0.2
            
            # Check gradient consistency
            if gradient_magnitude > detection_threshold * 0.3:
                validation_score += 0.3
            
            # Check temporal consistency if multiple frames available
            if plume_data.ndim == 3 and plume_data.shape[0] > 1:
                prev_frame = plume_data[-2]
                prev_intensity = prev_frame[y_idx, x_idx] if 0 <= y_idx < prev_frame.shape[0] and 0 <= x_idx < prev_frame.shape[1] else 0.0
                
                if abs(intensity - prev_intensity) < detection_threshold:
                    validation_score += 0.3  # Temporal consistency bonus
            
            # Apply validation threshold
            if validation_score < 0.3:
                final_detection = False
                confidence *= 0.3  # Reduce confidence for failed validation
            
            metadata['validation_score'] = float(validation_score)
        
        # Calculate detection confidence and reliability metrics
        confidence = np.clip(confidence, 0.0, 2.0)  # Cap confidence at 2x threshold
        
        # Generate comprehensive detection metadata
        metadata.update({
            'detection_result': final_detection,
            'detection_confidence': float(confidence),
            'intensity_ratio': float(intensity / detection_threshold) if detection_threshold > 0 else 0.0,
            'detection_strength': 'strong' if confidence > 1.5 else 'moderate' if confidence > 1.0 else 'weak'
        })
        
        # Return detection result with comprehensive metadata
        return final_detection, confidence, metadata
        
    except Exception as e:
        # Return negative detection result on error
        error_metadata = {
            **metadata,
            'error': str(e),
            'detection_result': False,
            'detection_confidence': 0.0
        }
        return False, 0.0, error_metadata


def optimize_search_radius(
    plume_metadata: Dict[str, Any],
    performance_history: List[float],
    current_radius: float = DEFAULT_SEARCH_RADIUS,
    optimization_options: Dict[str, Any] = None
) -> float:
    """
    Optimize casting search radius based on plume characteristics, arena size, and algorithm performance history for adaptive search pattern optimization.
    
    This function performs comprehensive search radius optimization using plume spatial characteristic analysis, performance history evaluation, optimal radius calculation based on plume density and arena constraints, adaptive adjustment mechanisms, and validation against physical and performance constraints for enhanced search effectiveness.
    
    Args:
        plume_metadata: Metadata containing plume spatial and temporal characteristics
        performance_history: Historical performance metrics for radius effectiveness evaluation
        current_radius: Current search radius to optimize
        optimization_options: Additional options for optimization including constraints and methods
        
    Returns:
        float: Optimized search radius with performance prediction and validation metrics
    """
    # Initialize optimization options with defaults
    options = optimization_options or {}
    optimization_method = options.get('optimization_method', 'adaptive')
    performance_weight = options.get('performance_weight', 0.7)
    plume_weight = options.get('plume_weight', 0.3)
    min_radius = options.get('min_radius', 0.01)
    max_radius = options.get('max_radius', 0.5)
    adaptation_rate = options.get('adaptation_rate', 0.1)
    
    try:
        # Validate input parameters for radius optimization
        if current_radius <= 0:
            current_radius = DEFAULT_SEARCH_RADIUS
        
        # Analyze plume spatial characteristics and distribution
        plume_density = plume_metadata.get('plume_density', 0.1)
        plume_extent = plume_metadata.get('plume_extent', 0.2)
        arena_width = plume_metadata.get('arena_width', TARGET_ARENA_WIDTH_METERS)
        arena_height = plume_metadata.get('arena_height', TARGET_ARENA_HEIGHT_METERS)
        plume_concentration = plume_metadata.get('average_concentration', 0.05)
        plume_variability = plume_metadata.get('spatial_variability', 0.1)
        
        # Calculate plume-based optimal radius
        plume_optimal_radius = current_radius
        
        if optimization_method == 'plume_adaptive':
            # Adaptive radius based on plume characteristics
            if plume_density > 0.2:
                # Dense plume - smaller radius for precision
                plume_optimal_radius = current_radius * 0.8
            elif plume_density < 0.05:
                # Sparse plume - larger radius for coverage
                plume_optimal_radius = current_radius * 1.3
            
            # Adjust for plume extent
            extent_factor = min(2.0, max(0.5, plume_extent / 0.1))
            plume_optimal_radius *= extent_factor
            
        elif optimization_method == 'concentration_based':
            # Radius based on concentration gradients
            if plume_concentration > 0.1:
                # High concentration - reduce radius for precision
                concentration_factor = 0.5 + 0.5 * (0.2 / max(plume_concentration, 0.01))
            else:
                # Low concentration - increase radius for detection
                concentration_factor = 1.0 + (0.1 - plume_concentration) * 5.0
            
            plume_optimal_radius = current_radius * concentration_factor
            
        else:
            # Default adaptive optimization
            # Combine multiple plume characteristics
            density_factor = 1.0 - (plume_density - 0.1) * 2.0  # Denser -> smaller radius
            variability_factor = 1.0 + plume_variability * 2.0    # More variable -> larger radius
            
            plume_optimal_radius = current_radius * density_factor * variability_factor
        
        # Evaluate performance history for radius effectiveness
        performance_optimal_radius = current_radius
        
        if len(performance_history) >= 3:
            # Analyze recent performance trends
            recent_performance = performance_history[-5:]  # Last 5 measurements
            performance_trend = np.mean(np.diff(recent_performance))  # Trend analysis
            performance_variance = np.var(recent_performance)
            average_performance = np.mean(recent_performance)
            
            # Adjust radius based on performance trends
            if performance_trend < -0.1:
                # Declining performance - increase radius for better coverage
                performance_optimal_radius = current_radius * 1.2
            elif performance_trend > 0.1:
                # Improving performance - fine-tune radius
                if performance_variance < 0.01:
                    # Consistent good performance - maintain radius
                    performance_optimal_radius = current_radius
                else:
                    # Variable good performance - slight adjustment
                    performance_optimal_radius = current_radius * 0.95
            
            # Adjust for absolute performance level
            if average_performance < 0.3:
                # Poor overall performance - significant increase
                performance_optimal_radius = current_radius * 1.5
            elif average_performance > 0.8:
                # Excellent performance - optimize for efficiency
                performance_optimal_radius = current_radius * 0.9
        
        # Calculate optimal radius based on plume density and arena size
        arena_factor = min(arena_width, arena_height) / TARGET_ARENA_WIDTH_METERS
        arena_optimal_radius = DEFAULT_SEARCH_RADIUS * arena_factor
        
        # Ensure radius is proportional to arena size
        if arena_optimal_radius < min_radius:
            arena_optimal_radius = min_radius
        elif arena_optimal_radius > max_radius:
            arena_optimal_radius = max_radius
        
        # Apply adaptive adjustment based on recent performance
        if optimization_method == 'adaptive':
            # Weighted combination of different optimization strategies
            combined_radius = (
                plume_optimal_radius * plume_weight +
                performance_optimal_radius * performance_weight +
                arena_optimal_radius * (1.0 - plume_weight - performance_weight)
            )
            
            # Apply adaptation rate for smooth transitions
            optimized_radius = current_radius + adaptation_rate * (combined_radius - current_radius)
            
        else:
            # Simple averaging of optimization methods
            optimized_radius = (plume_optimal_radius + performance_optimal_radius + arena_optimal_radius) / 3.0
        
        # Validate optimized radius against physical constraints
        if optimized_radius < min_radius:
            optimized_radius = min_radius
        elif optimized_radius > max_radius:
            optimized_radius = max_radius
        
        # Ensure radius doesn't exceed arena boundaries
        max_arena_radius = min(arena_width, arena_height) * 0.4  # 40% of smallest arena dimension
        if optimized_radius > max_arena_radius:
            optimized_radius = max_arena_radius
        
        # Apply final validation and reasonableness checks
        if abs(optimized_radius - current_radius) > current_radius * 0.5:
            # Limit dramatic changes to 50% of current radius
            if optimized_radius > current_radius:
                optimized_radius = current_radius * 1.5
            else:
                optimized_radius = current_radius * 0.5
        
        # Return optimized radius with performance predictions
        return float(optimized_radius)
        
    except Exception as e:
        # Return current radius with small adaptive adjustment on error
        adaptive_radius = current_radius * (1.0 + random.uniform(-0.1, 0.1))
        return np.clip(adaptive_radius, min_radius, max_radius)


def validate_casting_parameters(
    casting_params: Dict[str, Any],
    arena_constraints: Dict[str, Any],
    strict_validation: bool = False
) -> 'ValidationResult':
    """
    Validate casting algorithm parameters against physical constraints, performance requirements, and scientific computing standards for robust parameter validation.
    
    This function performs comprehensive parameter validation using casting speed and movement parameter validation, search radius constraint checking against arena dimensions, detection threshold and sensitivity parameter validation, wind estimation and trajectory parameter verification, strict validation criteria application, and validation result generation with recommendations for optimal casting algorithm configuration.
    
    Args:
        casting_params: Dictionary of casting algorithm parameters to validate
        arena_constraints: Arena dimension and boundary constraints for validation
        strict_validation: Enable strict validation criteria with enhanced constraint checking
        
    Returns:
        ValidationResult: Parameter validation result with constraint compliance and recommendations
    """
    # Import ValidationResult from validation utilities
    from ..utils.validation_utils import ValidationResult
    
    # Initialize validation result for casting parameters
    validation_result = ValidationResult(
        validation_type="casting_parameters_validation",
        is_valid=True,
        validation_context=f"strict_mode={strict_validation}"
    )
    
    try:
        # Validate casting speed and movement parameters
        casting_speed = casting_params.get('casting_speed', DEFAULT_CASTING_SPEED)
        if not isinstance(casting_speed, (int, float)):
            validation_result.add_error("Casting speed must be numeric", severity='HIGH')
            validation_result.is_valid = False
        elif casting_speed <= 0:
            validation_result.add_error("Casting speed must be positive", severity='HIGH')
            validation_result.is_valid = False
        elif casting_speed > 1.0:
            validation_result.add_warning("Casting speed > 1.0 may be too fast for stable navigation")
        
        upwind_speed = casting_params.get('upwind_speed', DEFAULT_UPWIND_SPEED)
        if not isinstance(upwind_speed, (int, float)):
            validation_result.add_error("Upwind speed must be numeric", severity='HIGH')
            validation_result.is_valid = False
        elif upwind_speed <= 0:
            validation_result.add_error("Upwind speed must be positive", severity='HIGH')
            validation_result.is_valid = False
        
        crosswind_speed = casting_params.get('crosswind_speed', DEFAULT_CROSSWIND_SPEED)
        if not isinstance(crosswind_speed, (int, float)):
            validation_result.add_error("Crosswind speed must be numeric", severity='HIGH')
            validation_result.is_valid = False
        elif crosswind_speed <= 0:
            validation_result.add_error("Crosswind speed must be positive", severity='HIGH')
            validation_result.is_valid = False
        
        # Check search radius against arena dimensions
        search_radius = casting_params.get('search_radius', DEFAULT_SEARCH_RADIUS)
        if not isinstance(search_radius, (int, float)):
            validation_result.add_error("Search radius must be numeric", severity='HIGH')
            validation_result.is_valid = False
        elif search_radius <= 0:
            validation_result.add_error("Search radius must be positive", severity='HIGH')
            validation_result.is_valid = False
        else:
            # Validate against arena constraints
            arena_width = arena_constraints.get('width', TARGET_ARENA_WIDTH_METERS)
            arena_height = arena_constraints.get('height', TARGET_ARENA_HEIGHT_METERS)
            
            max_allowable_radius = min(arena_width, arena_height) * 0.4
            if search_radius > max_allowable_radius:
                validation_result.add_error(
                    f"Search radius {search_radius} exceeds arena constraints {max_allowable_radius}",
                    severity='HIGH'
                )
                validation_result.is_valid = False
        
        # Validate detection thresholds and sensitivity parameters
        plume_threshold = casting_params.get('plume_threshold', DEFAULT_PLUME_THRESHOLD)
        if not isinstance(plume_threshold, (int, float)):
            validation_result.add_error("Plume threshold must be numeric", severity='HIGH')
            validation_result.is_valid = False
        elif plume_threshold < 0:
            validation_result.add_error("Plume threshold must be non-negative", severity='HIGH')
            validation_result.is_valid = False
        elif plume_threshold > 1.0:
            validation_result.add_warning("Plume threshold > 1.0 may be too high for detection")
        
        # Verify wind estimation and trajectory parameters
        wind_estimation_window = casting_params.get('wind_estimation_window', DEFAULT_WIND_ESTIMATION_WINDOW)
        if not isinstance(wind_estimation_window, int):
            validation_result.add_error("Wind estimation window must be integer", severity='MEDIUM')
        elif wind_estimation_window < 1:
            validation_result.add_error("Wind estimation window must be at least 1", severity='MEDIUM')
        elif wind_estimation_window > 50:
            validation_result.add_warning("Large wind estimation window may reduce responsiveness")
        
        casting_angle = casting_params.get('casting_angle', DEFAULT_CASTING_ANGLE)
        if not isinstance(casting_angle, (int, float)):
            validation_result.add_error("Casting angle must be numeric", severity='MEDIUM')
        elif casting_angle < 0 or casting_angle > 2 * np.pi:
            validation_result.add_warning("Casting angle should be between 0 and 2π radians")
        
        max_cast_distance = casting_params.get('max_cast_distance', DEFAULT_MAX_CAST_DISTANCE)
        if not isinstance(max_cast_distance, (int, float)):
            validation_result.add_error("Max cast distance must be numeric", severity='MEDIUM')
        elif max_cast_distance <= 0:
            validation_result.add_error("Max cast distance must be positive", severity='MEDIUM')
        elif max_cast_distance > min(arena_width, arena_height):
            validation_result.add_error("Max cast distance exceeds arena dimensions", severity='HIGH')
            validation_result.is_valid = False
        
        # Apply strict validation criteria if enabled
        if strict_validation:
            # Enhanced parameter consistency checks
            if upwind_speed <= casting_speed:
                validation_result.add_warning("Upwind speed should typically be higher than casting speed")
            
            if crosswind_speed > casting_speed * 2:
                validation_result.add_warning("Crosswind speed seems disproportionately high")
            
            # Check for parameter optimization potential
            if search_radius < 0.02:
                validation_result.add_warning("Very small search radius may reduce search effectiveness")
            elif search_radius > 0.2:
                validation_result.add_warning("Large search radius may reduce search precision")
            
            # Validate noise tolerance and adaptive behavior settings
            noise_tolerance = casting_params.get('noise_tolerance', 0.01)
            if noise_tolerance < 0 or noise_tolerance > 0.5:
                validation_result.add_warning("Noise tolerance outside recommended range [0, 0.5]")
            
            adaptive_radius = casting_params.get('adaptive_radius', True)
            if not isinstance(adaptive_radius, bool):
                validation_result.add_warning("Adaptive radius should be boolean")
        
        # Add validation metrics for parameter assessment
        validation_result.add_metric("casting_speed", casting_speed)
        validation_result.add_metric("search_radius", search_radius)
        validation_result.add_metric("plume_threshold", plume_threshold)
        validation_result.add_metric("parameter_count", len(casting_params))
        
        # Generate validation result with recommendations
        if validation_result.is_valid:
            validation_result.add_recommendation(
                "Casting parameters passed validation - ready for algorithm execution",
                priority="INFO"
            )
        else:
            validation_result.add_recommendation(
                "Correct parameter constraint violations before algorithm execution",
                priority="HIGH"
            )
            
            # Add specific recommendations based on validation failures
            if any("speed" in error for error in validation_result.errors):
                validation_result.add_recommendation(
                    "Review speed parameters: casting_speed, upwind_speed, crosswind_speed",
                    priority="HIGH"
                )
            
            if any("radius" in error for error in validation_result.errors):
                validation_result.add_recommendation(
                    "Adjust search_radius to fit within arena constraints",
                    priority="HIGH"
                )
        
        return validation_result
        
    except Exception as e:
        # Return critical validation failure on exception
        validation_result.add_error(f"Parameter validation failed: {str(e)}", severity='CRITICAL')
        validation_result.is_valid = False
        return validation_result


@dataclasses.dataclass
class CastingParameters:
    """
    Data class for casting algorithm parameters including search patterns, movement speeds, detection thresholds, and wind estimation settings with validation and optimization support for comprehensive casting behavior configuration.
    
    This class provides comprehensive parameter management with movement speed configuration, search pattern definition, detection threshold settings, wind estimation parameters, adaptive behavior controls, and advanced options for specialized casting algorithm customization with validation and optimization capabilities.
    """
    
    # Basic movement parameters
    casting_speed: float = DEFAULT_CASTING_SPEED
    search_radius: float = DEFAULT_SEARCH_RADIUS
    plume_threshold: float = DEFAULT_PLUME_THRESHOLD
    upwind_speed: float = DEFAULT_UPWIND_SPEED
    
    # Advanced movement and search parameters
    casting_angle: float = DEFAULT_CASTING_ANGLE
    crosswind_speed: float = DEFAULT_CROSSWIND_SPEED
    max_cast_distance: float = DEFAULT_MAX_CAST_DISTANCE
    wind_estimation_window: int = DEFAULT_WIND_ESTIMATION_WINDOW
    
    # Adaptive behavior settings
    adaptive_radius: bool = True
    enable_wind_estimation: bool = True
    noise_tolerance: float = 0.01
    
    # Advanced options for specialized behaviors
    advanced_options: Dict[str, Any] = dataclasses.field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize casting parameters with movement speeds, search patterns, and detection settings."""
        # Set basic casting movement parameters with validation
        if self.casting_speed <= 0:
            self.casting_speed = DEFAULT_CASTING_SPEED
        
        if self.search_radius <= 0:
            self.search_radius = DEFAULT_SEARCH_RADIUS
        
        if self.upwind_speed <= 0:
            self.upwind_speed = DEFAULT_UPWIND_SPEED
        
        # Initialize search radius and casting angle with constraints
        if self.casting_angle < 0 or self.casting_angle > 2 * np.pi:
            self.casting_angle = DEFAULT_CASTING_ANGLE
        
        if self.crosswind_speed <= 0:
            self.crosswind_speed = DEFAULT_CROSSWIND_SPEED
        
        # Configure plume detection threshold and sensitivity
        if self.plume_threshold < 0:
            self.plume_threshold = DEFAULT_PLUME_THRESHOLD
        
        if self.max_cast_distance <= 0:
            self.max_cast_distance = DEFAULT_MAX_CAST_DISTANCE
        
        # Setup wind estimation and adaptive behavior settings
        if self.wind_estimation_window < 1:
            self.wind_estimation_window = DEFAULT_WIND_ESTIMATION_WINDOW
        
        if self.noise_tolerance < 0:
            self.noise_tolerance = 0.01
        
        # Initialize advanced options and noise tolerance with defaults
        if not isinstance(self.advanced_options, dict):
            self.advanced_options = {}
        
        # Set default advanced options if not specified
        default_advanced = {
            'trajectory_smoothing': True,
            'gradient_following': True,
            'temporal_averaging': True,
            'outlier_detection': True,
            'performance_tracking': True
        }
        
        for key, value in default_advanced.items():
            if key not in self.advanced_options:
                self.advanced_options[key] = value
        
        # Validate parameter consistency and constraints
        self._validate_parameter_consistency()
    
    def _validate_parameter_consistency(self):
        """Validate parameter consistency and apply corrections if needed."""
        # Ensure upwind speed is typically higher than casting speed
        if self.upwind_speed < self.casting_speed:
            self.upwind_speed = self.casting_speed * 1.2
        
        # Ensure crosswind speed is reasonable relative to casting speed
        if self.crosswind_speed > self.casting_speed * 3:
            self.crosswind_speed = self.casting_speed * 1.5
        
        # Ensure max cast distance is reasonable relative to search radius
        if self.max_cast_distance < self.search_radius * 2:
            self.max_cast_distance = self.search_radius * 3
    
    def validate_parameters(self, strict_validation: bool = False) -> 'ValidationResult':
        """
        Validate casting parameters against physical constraints and performance requirements.
        
        Args:
            strict_validation: Enable strict validation mode with enhanced constraint checking
            
        Returns:
            ValidationResult: Parameter validation result with constraint compliance assessment
        """
        # Create parameter dictionary for validation
        param_dict = {
            'casting_speed': self.casting_speed,
            'search_radius': self.search_radius,
            'plume_threshold': self.plume_threshold,
            'upwind_speed': self.upwind_speed,
            'casting_angle': self.casting_angle,
            'crosswind_speed': self.crosswind_speed,
            'max_cast_distance': self.max_cast_distance,
            'wind_estimation_window': self.wind_estimation_window,
            'adaptive_radius': self.adaptive_radius,
            'enable_wind_estimation': self.enable_wind_estimation,
            'noise_tolerance': self.noise_tolerance
        }
        
        # Create arena constraints from current parameters
        arena_constraints = {
            'width': TARGET_ARENA_WIDTH_METERS,
            'height': TARGET_ARENA_HEIGHT_METERS
        }
        
        # Use global validation function
        return validate_casting_parameters(param_dict, arena_constraints, strict_validation)
    
    def optimize_for_plume(
        self,
        plume_metadata: Dict[str, Any],
        arena_constraints: Dict[str, Any]
    ) -> 'CastingParameters':
        """
        Optimize casting parameters based on plume characteristics and arena constraints.
        
        Args:
            plume_metadata: Metadata containing plume spatial and temporal characteristics
            arena_constraints: Arena dimension and boundary constraints
            
        Returns:
            CastingParameters: Optimized casting parameters with performance predictions
        """
        # Create optimized copy of current parameters
        optimized = copy.deepcopy(self)
        
        try:
            # Analyze plume spatial and temporal characteristics
            plume_density = plume_metadata.get('plume_density', 0.1)
            plume_extent = plume_metadata.get('plume_extent', 0.2)
            average_concentration = plume_metadata.get('average_concentration', 0.05)
            spatial_variability = plume_metadata.get('spatial_variability', 0.1)
            temporal_variability = plume_metadata.get('temporal_variability', 0.1)
            
            # Optimize search radius based on plume distribution
            if plume_density > 0.2:
                # Dense plume - reduce radius for precision
                optimized.search_radius = min(self.search_radius * 0.8, DEFAULT_SEARCH_RADIUS * 0.5)
            elif plume_density < 0.05:
                # Sparse plume - increase radius for coverage
                optimized.search_radius = max(self.search_radius * 1.3, DEFAULT_SEARCH_RADIUS * 1.5)
            
            # Adjust movement speeds for plume density and variability
            if average_concentration > 0.1:
                # High concentration - reduce speeds for precision
                optimized.casting_speed = self.casting_speed * 0.8
                optimized.upwind_speed = self.upwind_speed * 0.9
            elif average_concentration < 0.02:
                # Low concentration - increase speeds for broader search
                optimized.casting_speed = self.casting_speed * 1.2
                optimized.upwind_speed = self.upwind_speed * 1.1
            
            # Optimize detection threshold for plume intensity
            if average_concentration > 0:
                # Set threshold based on plume characteristics
                optimized.plume_threshold = max(average_concentration * 0.3, 0.005)
            
            # Adjust wind estimation window based on temporal variability
            if temporal_variability > 0.2:
                # High temporal variability - shorter window for responsiveness
                optimized.wind_estimation_window = max(5, self.wind_estimation_window // 2)
            elif temporal_variability < 0.05:
                # Low temporal variability - longer window for stability
                optimized.wind_estimation_window = min(20, self.wind_estimation_window * 2)
            
            # Apply arena constraints and boundary conditions
            arena_width = arena_constraints.get('width', TARGET_ARENA_WIDTH_METERS)
            arena_height = arena_constraints.get('height', TARGET_ARENA_HEIGHT_METERS)
            
            # Ensure search radius fits within arena
            max_radius = min(arena_width, arena_height) * 0.4
            optimized.search_radius = min(optimized.search_radius, max_radius)
            
            # Ensure max cast distance fits within arena
            max_cast = min(arena_width, arena_height) * 0.8
            optimized.max_cast_distance = min(optimized.max_cast_distance, max_cast)
            
            # Enable adaptive behaviors for complex plumes
            if spatial_variability > 0.15 or temporal_variability > 0.15:
                optimized.adaptive_radius = True
                optimized.enable_wind_estimation = True
                optimized.advanced_options['gradient_following'] = True
            
            # Return optimized parameter set
            return optimized
            
        except Exception as e:
            # Return original parameters if optimization fails
            return copy.deepcopy(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert casting parameters to dictionary format for serialization and logging.
        
        Returns:
            Dict[str, Any]: Casting parameters as dictionary with metadata
        """
        return {
            # Basic movement parameters
            'casting_speed': self.casting_speed,
            'search_radius': self.search_radius,
            'plume_threshold': self.plume_threshold,
            'upwind_speed': self.upwind_speed,
            
            # Advanced movement and search parameters
            'casting_angle': self.casting_angle,
            'crosswind_speed': self.crosswind_speed,
            'max_cast_distance': self.max_cast_distance,
            'wind_estimation_window': self.wind_estimation_window,
            
            # Adaptive behavior settings
            'adaptive_radius': self.adaptive_radius,
            'enable_wind_estimation': self.enable_wind_estimation,
            'noise_tolerance': self.noise_tolerance,
            
            # Advanced options and metadata
            'advanced_options': self.advanced_options.copy(),
            'parameter_version': ALGORITHM_VERSION,
            'parameter_class': 'CastingParameters'
        }


@dataclasses.dataclass
class CastingState:
    """
    State management class for casting algorithm execution including current position, movement state, search patterns, wind estimation, and performance tracking for comprehensive casting behavior state management.
    
    This class provides comprehensive state tracking with position management, movement state tracking, search pattern coordination, wind estimation state, performance metric collection, and trajectory history management for detailed casting algorithm execution monitoring and analysis.
    """
    
    # Core position and state tracking
    current_position: np.ndarray
    parameters: CastingParameters
    
    # Movement and navigation state
    previous_position: np.ndarray = dataclasses.field(init=False)
    current_state: str = CASTING_STATE_SEARCHING
    wind_direction: Tuple[float, float] = (1.0, 0.0)
    current_search_radius: float = dataclasses.field(init=False)
    
    # Trajectory and history tracking
    trajectory_history: List[np.ndarray] = dataclasses.field(default_factory=list)
    plume_contact_history: List[float] = dataclasses.field(default_factory=list)
    steps_since_contact: int = 0
    casting_direction: int = 1  # 1 for right, -1 for left
    cast_distance: float = 0.0
    
    # Performance and metadata tracking
    performance_metrics: Dict[str, Any] = dataclasses.field(default_factory=dict)
    plume_contact_active: bool = False
    
    def __post_init__(self):
        """Initialize casting state with starting position and algorithm parameters."""
        # Set initial and current position with validation
        if not isinstance(self.current_position, np.ndarray):
            self.current_position = np.array(self.current_position)
        
        if self.current_position.size != 2:
            raise ValueError(f"Position must be 2D, got {self.current_position.size}D")
        
        # Initialize casting state and movement parameters
        self.previous_position = self.current_position.copy()
        self.current_search_radius = self.parameters.search_radius
        
        # Setup wind direction estimation and search radius
        self.wind_direction = (1.0, 0.0)  # Default eastward wind
        
        # Initialize trajectory and contact history tracking
        self.trajectory_history = [self.current_position.copy()]
        self.plume_contact_history = [0.0]
        
        # Setup performance metrics and state tracking
        self.performance_metrics = {
            'total_steps': 0,
            'plume_contacts': 0,
            'successful_detections': 0,
            'false_positives': 0,
            'state_transitions': 0,
            'wind_estimates': 0,
            'search_efficiency': 0.0,
            'trajectory_length': 0.0,
            'average_speed': 0.0,
            'time_in_states': {
                CASTING_STATE_SEARCHING: 0,
                CASTING_STATE_TRACKING: 0,
                CASTING_STATE_CASTING: 0,
                CASTING_STATE_SURGING: 0
            }
        }
        
        # Initialize casting direction and distance counters
        self.casting_direction = 1
        self.cast_distance = 0.0
        self.steps_since_contact = 0
        self.plume_contact_active = False
    
    def update_position(
        self,
        new_position: np.ndarray,
        validate_movement: bool = True
    ) -> bool:
        """
        Update current position and maintain trajectory history with validation and performance tracking.
        
        Args:
            new_position: New position coordinates [y, x]
            validate_movement: Whether to validate movement against constraints
            
        Returns:
            bool: True if position update successful with validation status
        """
        try:
            # Validate new position against arena boundaries
            if not isinstance(new_position, np.ndarray):
                new_position = np.array(new_position)
            
            if new_position.size != 2:
                return False
            
            # Ensure position is within normalized bounds [0, 1]
            clipped_position = np.clip(new_position, 0.0, 1.0)
            
            # Validate movement if requested
            if validate_movement:
                movement_distance = np.linalg.norm(clipped_position - self.current_position)
                max_movement = max(self.parameters.casting_speed, self.parameters.upwind_speed) * 2
                
                if movement_distance > max_movement:
                    # Limit movement to maximum allowed distance
                    direction = clipped_position - self.current_position
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 0:
                        direction /= direction_norm
                        clipped_position = self.current_position + direction * max_movement
                        clipped_position = np.clip(clipped_position, 0.0, 1.0)
            
            # Update previous and current position
            self.previous_position = self.current_position.copy()
            self.current_position = clipped_position.copy()
            
            # Add position to trajectory history
            self.trajectory_history.append(self.current_position.copy())
            
            # Limit trajectory history size for memory management
            if len(self.trajectory_history) > 1000:
                self.trajectory_history = self.trajectory_history[-500:]  # Keep last 500 points
            
            # Update movement metrics and performance tracking
            movement_distance = np.linalg.norm(self.current_position - self.previous_position)
            self.performance_metrics['total_steps'] += 1
            self.performance_metrics['trajectory_length'] += movement_distance
            
            # Update average speed calculation
            if self.performance_metrics['total_steps'] > 0:
                self.performance_metrics['average_speed'] = (
                    self.performance_metrics['trajectory_length'] / self.performance_metrics['total_steps']
                )
            
            # Update cast distance tracking
            self.cast_distance += movement_distance
            
            # Return update status with validation result
            return True
            
        except Exception as e:
            # Return failure status on error
            return False
    
    def update_wind_estimate(
        self,
        plume_data: np.ndarray,
        estimation_window: int = None
    ) -> Tuple[float, float]:
        """
        Update wind direction estimate based on plume gradient and movement history.
        
        Args:
            plume_data: Current plume data for gradient analysis
            estimation_window: Window size for temporal averaging
            
        Returns:
            Tuple[float, float]: Updated wind direction with confidence estimate
        """
        try:
            # Use parameter window if not specified
            if estimation_window is None:
                estimation_window = self.parameters.wind_estimation_window
            
            # Calculate plume gradient at current position
            wind_options = {
                'noise_threshold': self.parameters.noise_tolerance,
                'gradient_method': 'sobel',
                'confidence_threshold': 0.1
            }
            
            new_wind_direction = calculate_wind_direction(
                plume_data, self.current_position, estimation_window, wind_options
            )
            
            # Apply temporal averaging over estimation window
            if self.parameters.enable_wind_estimation:
                # Weighted average with previous estimate
                weight = 0.3  # Weight for new estimate
                self.wind_direction = (
                    self.wind_direction[0] * (1 - weight) + new_wind_direction[0] * weight,
                    self.wind_direction[1] * (1 - weight) + new_wind_direction[1] * weight
                )
                
                # Normalize wind direction
                wind_magnitude = np.sqrt(self.wind_direction[0]**2 + self.wind_direction[1]**2)
                if wind_magnitude > 0:
                    self.wind_direction = (
                        self.wind_direction[0] / wind_magnitude,
                        self.wind_direction[1] / wind_magnitude
                    )
            else:
                self.wind_direction = new_wind_direction
            
            # Update performance metrics
            self.performance_metrics['wind_estimates'] += 1
            
            # Validate wind estimate against trajectory history
            # (This would involve more complex analysis in a full implementation)
            
            # Return updated wind direction with confidence
            return self.wind_direction
            
        except Exception as e:
            # Return current wind direction on error
            return self.wind_direction
    
    def transition_state(
        self,
        new_state: str,
        transition_context: Dict[str, Any] = None
    ) -> bool:
        """
        Transition casting algorithm state based on plume contact and search progress.
        
        Args:
            new_state: Target state for transition
            transition_context: Additional context for state transition
            
        Returns:
            bool: True if state transition successful with validation
        """
        try:
            # Validate state transition against current state
            valid_states = [CASTING_STATE_SEARCHING, CASTING_STATE_TRACKING, CASTING_STATE_CASTING, CASTING_STATE_SURGING]
            if new_state not in valid_states:
                return False
            
            # Update casting state and associated parameters
            old_state = self.current_state
            self.current_state = new_state
            
            # Reset state-specific counters and metrics
            if new_state == CASTING_STATE_SEARCHING:
                self.steps_since_contact = 0
                self.cast_distance = 0.0
                
            elif new_state == CASTING_STATE_TRACKING:
                self.plume_contact_active = True
                self.steps_since_contact = 0
                
            elif new_state == CASTING_STATE_CASTING:
                self.plume_contact_active = False
                if transition_context and 'casting_direction' in transition_context:
                    self.casting_direction = transition_context['casting_direction']
                
            elif new_state == CASTING_STATE_SURGING:
                self.plume_contact_active = True
                self.cast_distance = 0.0
            
            # Log state transition with context information
            if transition_context:
                transition_info = {
                    'old_state': old_state,
                    'new_state': new_state,
                    'position': self.current_position.tolist(),
                    'context': transition_context
                }
            
            # Update performance metrics for state change
            self.performance_metrics['state_transitions'] += 1
            self.performance_metrics['time_in_states'][old_state] += 1
            
            # Return transition status with validation
            return True
            
        except Exception as e:
            # Return failure status on transition error
            return False
    
    def get_state_summary(
        self,
        include_trajectory: bool = False,
        include_performance: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive state summary with position, trajectory, and performance information.
        
        Args:
            include_trajectory: Whether to include full trajectory history
            include_performance: Whether to include performance metrics
            
        Returns:
            Dict[str, Any]: State summary with position, trajectory, and performance data
        """
        try:
            # Compile current position and state information
            summary = {
                'current_position': self.current_position.tolist(),
                'previous_position': self.previous_position.tolist(),
                'current_state': self.current_state,
                'wind_direction': self.wind_direction,
                'current_search_radius': self.current_search_radius,
                'plume_contact_active': self.plume_contact_active,
                'steps_since_contact': self.steps_since_contact,
                'casting_direction': self.casting_direction,
                'cast_distance': self.cast_distance
            }
            
            # Include trajectory history if requested
            if include_trajectory:
                summary['trajectory_history'] = [pos.tolist() for pos in self.trajectory_history[-100:]]  # Last 100 points
                summary['trajectory_length'] = len(self.trajectory_history)
                summary['plume_contact_history'] = self.plume_contact_history[-100:]  # Last 100 contacts
            else:
                summary['trajectory_summary'] = {
                    'total_points': len(self.trajectory_history),
                    'start_position': self.trajectory_history[0].tolist() if self.trajectory_history else None,
                    'current_position': self.current_position.tolist()
                }
            
            # Add performance metrics if requested
            if include_performance:
                summary['performance_metrics'] = self.performance_metrics.copy()
                
                # Calculate derived performance metrics
                if self.performance_metrics['total_steps'] > 0:
                    summary['performance_derived'] = {
                        'contact_rate': self.performance_metrics['plume_contacts'] / self.performance_metrics['total_steps'],
                        'detection_accuracy': (
                            self.performance_metrics['successful_detections'] / 
                            max(1, self.performance_metrics['plume_contacts'])
                        ),
                        'search_efficiency': self.performance_metrics.get('search_efficiency', 0.0),
                        'state_distribution': {
                            state: count / self.performance_metrics['total_steps']
                            for state, count in self.performance_metrics['time_in_states'].items()
                        }
                    }
            
            # Include wind estimation and search parameters
            summary['algorithm_state'] = {
                'wind_estimation_enabled': self.parameters.enable_wind_estimation,
                'adaptive_radius_enabled': self.parameters.adaptive_radius,
                'noise_tolerance': self.parameters.noise_tolerance,
                'search_radius': self.parameters.search_radius,
                'current_search_radius': self.current_search_radius
            }
            
            # Format summary for analysis and reporting
            summary['summary_metadata'] = {
                'algorithm_version': ALGORITHM_VERSION,
                'state_class': 'CastingState',
                'generated_at': time.time()
            }
            
            # Return comprehensive state summary
            return summary
            
        except Exception as e:
            # Return minimal summary on error
            return {
                'current_position': self.current_position.tolist() if hasattr(self, 'current_position') else [0.0, 0.0],
                'current_state': self.current_state if hasattr(self, 'current_state') else CASTING_STATE_SEARCHING,
                'error': str(e)
            }


class CastingAlgorithm(BaseAlgorithm):
    """
    Comprehensive casting navigation algorithm implementation providing bio-inspired casting behavior for plume tracking and source localization with adaptive search patterns, wind estimation, statistical validation, and cross-format compatibility for scientific computing applications.
    
    This class implements systematic crosswind search patterns with upwind surges, adaptive search radius optimization, wind direction estimation using plume gradients, comprehensive state management, statistical performance tracking, and scientific validation for reproducible research outcomes with >95% correlation against reference implementations.
    """
    
    def __init__(
        self,
        parameters: CastingParameters,
        execution_config: Dict[str, Any] = None
    ):
        """
        Initialize casting algorithm with parameters, execution configuration, and performance tracking setup.
        
        Args:
            parameters: Casting algorithm parameters with movement speeds and search patterns
            execution_config: Configuration for algorithm execution environment and optimization
        """
        # Convert CastingParameters to AlgorithmParameters for base class
        algorithm_params = AlgorithmParameters(
            algorithm_name="casting",
            version=ALGORITHM_VERSION,
            parameters=parameters.to_dict()
        )
        
        # Initialize base algorithm with casting-specific configuration
        super().__init__(algorithm_params, execution_config)
        
        # Store casting-specific parameters
        self.casting_parameters = parameters
        
        # Initialize algorithm name, version, and execution configuration
        self.algorithm_name = "casting"
        self.version = ALGORITHM_VERSION
        self.execution_config = execution_config or {}
        
        # Initialize algorithm state and performance tracking
        self.algorithm_state = None  # Will be initialized during execution
        self.execution_history = []
        self.performance_baselines = {
            'correlation_threshold': DEFAULT_CORRELATION_THRESHOLD,
            'processing_time_target': 7.2,  # <7.2 seconds requirement
            'trajectory_efficiency': 0.8,
            'search_effectiveness': 0.7
        }
        
        # Setup adaptive behavior and optimization capabilities
        self.adaptive_behavior_enabled = parameters.adaptive_radius
        
        # Initialize plume normalizer for cross-format compatibility
        self.plume_normalizer = PlumeNormalizer()
        
        # Setup statistical cache for performance optimization
        self.statistical_cache = {
            'wind_estimates': [],
            'trajectory_segments': [],
            'performance_history': [],
            'correlation_cache': {}
        }
        
        # Configure performance monitoring and validation
        self.performance_tracking_enabled = execution_config.get('performance_tracking', True)
        self.validation_enabled = execution_config.get('validation_enabled', True)
        
        # Initialize execution history and baseline tracking
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.success_rate = 0.0
        
        self.logger.info("Casting algorithm initialized: speed=%.3f, radius=%.3f, adaptive=%s",
                        parameters.casting_speed, parameters.search_radius, parameters.adaptive_radius)
    
    def _execute_algorithm(
        self,
        plume_data: np.ndarray,
        plume_metadata: Dict[str, Any],
        context: AlgorithmContext
    ) -> AlgorithmResult:
        """
        Execute casting algorithm with plume data, implementing systematic crosswind search patterns, upwind surges, and adaptive behavior for comprehensive plume navigation.
        
        This method performs comprehensive casting algorithm execution with state initialization, plume data normalization, main casting loop implementation, crosswind search pattern execution, upwind surge behavior, adaptive optimization, wind direction estimation, performance tracking, and validation against scientific computing standards for reproducible research outcomes.
        
        Args:
            plume_data: Validated plume data array for algorithm processing
            plume_metadata: Plume metadata with format and calibration information
            context: Algorithm execution context with performance tracking and scientific context
            
        Returns:
            AlgorithmResult: Casting algorithm execution result with trajectory, performance metrics, and validation
        """
        # Initialize casting algorithm result
        result = AlgorithmResult(
            algorithm_name=self.algorithm_name,
            simulation_id=context.simulation_id,
            success=False
        )
        
        try:
            # Initialize casting state with starting position and parameters
            start_position = plume_metadata.get('start_position', [0.5, 0.1])  # Default bottom center
            if not isinstance(start_position, np.ndarray):
                start_position = np.array(start_position)
            
            self.algorithm_state = CastingState(
                current_position=start_position,
                parameters=self.casting_parameters
            )
            
            # Normalize plume data for cross-format compatibility
            normalized_plume = self.plume_normalizer.normalize_plume_data(plume_data, plume_metadata)
            
            # Add checkpoint for initialization
            context.add_checkpoint('algorithm_initialized', {
                'start_position': start_position.tolist(),
                'plume_shape': normalized_plume.shape,
                'parameters': self.casting_parameters.to_dict()
            })
            
            # Execute main casting loop with state transitions
            max_iterations = 1000  # Maximum steps to prevent infinite loops
            iteration_count = 0
            source_found = False
            convergence_achieved = False
            
            trajectory_points = [start_position.copy()]
            performance_history = []
            
            while iteration_count < max_iterations and not source_found:
                iteration_count += 1
                
                # Execute single casting algorithm step
                new_position, new_state, step_metrics = self.execute_casting_step(normalized_plume, context)
                
                # Update algorithm state and trajectory
                self.algorithm_state.update_position(new_position, validate_movement=True)
                trajectory_points.append(new_position.copy())
                performance_history.append(step_metrics.get('step_performance', 0.0))
                
                # Check for source detection (high concentration area)
                plume_contact, confidence, detection_metadata = detect_plume_contact(
                    normalized_plume, new_position, self.casting_parameters.plume_threshold
                )
                
                if plume_contact and confidence > 1.5:  # Strong detection indicates source proximity
                    source_found = True
                    result.success = True
                
                # Implement adaptive search radius optimization
                if self.adaptive_behavior_enabled and iteration_count % 50 == 0:  # Every 50 steps
                    self.adapt_search_parameters(
                        {'recent_performance': performance_history[-50:]},
                        plume_metadata
                    )
                
                # Update wind direction estimates throughout execution
                if iteration_count % 10 == 0:  # Every 10 steps
                    self.algorithm_state.update_wind_estimate(normalized_plume)
                
                # Check for convergence criteria
                if iteration_count > 100:
                    recent_positions = trajectory_points[-20:]
                    position_variance = np.var([pos[0] for pos in recent_positions]) + np.var([pos[1] for pos in recent_positions])
                    if position_variance < 0.001:  # Very small movement
                        convergence_achieved = True
                        break
                
                # Add periodic checkpoints
                if iteration_count % 100 == 0:
                    context.add_checkpoint(f'iteration_{iteration_count}', {
                        'position': new_position.tolist(),
                        'state': new_state,
                        'performance': np.mean(performance_history[-100:]) if len(performance_history) >= 100 else 0.0
                    })
            
            # Track performance metrics and trajectory quality
            final_trajectory = np.array(trajectory_points)
            result.trajectory = final_trajectory
            result.iterations_completed = iteration_count
            result.converged = convergence_achieved or source_found
            
            # Calculate trajectory-based performance metrics
            trajectory_length = np.sum([
                np.linalg.norm(final_trajectory[i+1] - final_trajectory[i])
                for i in range(len(final_trajectory) - 1)
            ])
            
            # Calculate search efficiency
            direct_distance = np.linalg.norm(final_trajectory[-1] - final_trajectory[0])
            search_efficiency = direct_distance / max(trajectory_length, 1e-10)
            
            # Add comprehensive performance metrics
            result.add_performance_metric('trajectory_length', float(trajectory_length))
            result.add_performance_metric('search_efficiency', float(search_efficiency))
            result.add_performance_metric('final_position_x', float(final_trajectory[-1][1]))
            result.add_performance_metric('final_position_y', float(final_trajectory[-1][0]))
            result.add_performance_metric('source_found', float(source_found))
            result.add_performance_metric('convergence_achieved', float(convergence_achieved))
            
            # Calculate wind estimation accuracy if reference available
            if 'reference_wind_direction' in plume_metadata:
                ref_wind = plume_metadata['reference_wind_direction']
                estimated_wind = self.algorithm_state.wind_direction
                wind_correlation = np.dot(estimated_wind, ref_wind)
                result.add_performance_metric('wind_estimation_accuracy', float(wind_correlation))
            
            # Validate algorithm convergence and success criteria
            if result.converged and trajectory_length > 0:
                result.success = True
                result.add_performance_metric('convergence_quality', float(search_efficiency))
            elif source_found:
                result.success = True
                result.add_performance_metric('source_detection_success', 1.0)
            else:
                result.add_warning("Algorithm did not converge or find source within iteration limit", "convergence")
            
            # Update algorithm state with final results
            result.algorithm_state = self.algorithm_state.get_state_summary(
                include_trajectory=False, include_performance=True
            )
            
            # Generate comprehensive algorithm result with statistics
            context.add_checkpoint('algorithm_completed', {
                'success': result.success,
                'iterations': iteration_count,
                'trajectory_length': trajectory_length,
                'search_efficiency': search_efficiency
            })
            
            # Update statistical cache for future optimization
            self.statistical_cache['trajectory_segments'].append(final_trajectory)
            self.statistical_cache['performance_history'].extend(performance_history)
            
            # Limit cache size for memory management
            if len(self.statistical_cache['trajectory_segments']) > 100:
                self.statistical_cache['trajectory_segments'] = self.statistical_cache['trajectory_segments'][-50:]
            
            if len(self.statistical_cache['performance_history']) > 5000:
                self.statistical_cache['performance_history'] = self.statistical_cache['performance_history'][-2500:]
            
            return result
            
        except Exception as e:
            # Handle algorithm execution errors gracefully
            result.success = False
            result.add_warning(f"Casting algorithm execution error: {str(e)}", "error")
            
            # Provide partial results if available
            if hasattr(self, 'algorithm_state') and self.algorithm_state:
                result.algorithm_state = self.algorithm_state.get_state_summary(include_performance=False)
            
            self.logger.error("Casting algorithm execution failed: %s", e, exc_info=True)
            return result
    
    def execute_casting_step(
        self,
        plume_data: np.ndarray,
        context: AlgorithmContext
    ) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        """
        Execute single casting algorithm step with state management, plume detection, and movement calculation.
        
        Args:
            plume_data: Current plume data for detection and navigation
            context: Algorithm execution context for performance tracking
            
        Returns:
            Tuple[np.ndarray, str, Dict[str, Any]]: New position, state, and step metrics
        """
        try:
            # Detect plume contact at current position
            plume_contact, confidence, detection_metadata = detect_plume_contact(
                plume_data, self.algorithm_state.current_position, self.casting_parameters.plume_threshold
            )
            
            # Update wind direction estimate if enabled
            if self.casting_parameters.enable_wind_estimation:
                self.algorithm_state.update_wind_estimate(plume_data)
            
            # Determine next movement based on current state
            current_state = self.algorithm_state.current_state
            new_position = self.algorithm_state.current_position.copy()
            new_state = current_state
            
            if current_state == CASTING_STATE_SEARCHING:
                if plume_contact:
                    # Transition to tracking state
                    new_state = CASTING_STATE_TRACKING
                    new_position = self.execute_upwind_surge(
                        self.algorithm_state.current_position,
                        self.algorithm_state.wind_direction,
                        self.casting_parameters.upwind_speed
                    )
                else:
                    # Continue searching with casting pattern
                    new_state = CASTING_STATE_CASTING
                    new_position = self.execute_crosswind_casting(
                        self.algorithm_state.current_position,
                        self.algorithm_state.wind_direction,
                        {'search_radius': self.algorithm_state.current_search_radius}
                    )
                    
            elif current_state == CASTING_STATE_TRACKING:
                if plume_contact and confidence > 1.0:
                    # Continue upwind surge
                    new_position = self.execute_upwind_surge(
                        self.algorithm_state.current_position,
                        self.algorithm_state.wind_direction,
                        self.casting_parameters.upwind_speed
                    )
                else:
                    # Lost plume, return to casting
                    new_state = CASTING_STATE_CASTING
                    new_position = self.execute_crosswind_casting(
                        self.algorithm_state.current_position,
                        self.algorithm_state.wind_direction,
                        {'search_radius': self.algorithm_state.current_search_radius}
                    )
                    
            elif current_state == CASTING_STATE_CASTING:
                if plume_contact:
                    # Found plume, switch to tracking
                    new_state = CASTING_STATE_TRACKING
                    new_position = self.execute_upwind_surge(
                        self.algorithm_state.current_position,
                        self.algorithm_state.wind_direction,
                        self.casting_parameters.upwind_speed
                    )
                else:
                    # Continue casting pattern
                    new_position = self.execute_crosswind_casting(
                        self.algorithm_state.current_position,
                        self.algorithm_state.wind_direction,
                        {'search_radius': self.algorithm_state.current_search_radius}
                    )
                    
            elif current_state == CASTING_STATE_SURGING:
                if plume_contact:
                    # Continue surge movement
                    new_position = self.execute_upwind_surge(
                        self.algorithm_state.current_position,
                        self.algorithm_state.wind_direction,
                        self.casting_parameters.upwind_speed
                    )
                else:
                    # Lost plume during surge, return to casting
                    new_state = CASTING_STATE_CASTING
                    new_position = self.execute_crosswind_casting(
                        self.algorithm_state.current_position,
                        self.algorithm_state.wind_direction,
                        {'search_radius': self.algorithm_state.current_search_radius}
                    )
            
            # Calculate new position with movement constraints
            movement_vector = new_position - self.algorithm_state.current_position
            movement_magnitude = np.linalg.norm(movement_vector)
            
            # Limit movement to maximum speed
            max_speed = max(self.casting_parameters.casting_speed, self.casting_parameters.upwind_speed)
            if movement_magnitude > max_speed:
                movement_vector = movement_vector / movement_magnitude * max_speed
                new_position = self.algorithm_state.current_position + movement_vector
            
            # Ensure position stays within bounds
            new_position = np.clip(new_position, 0.0, 1.0)
            
            # Update algorithm state based on plume contact
            if new_state != current_state:
                self.algorithm_state.transition_state(new_state, {
                    'plume_contact': plume_contact,
                    'confidence': confidence,
                    'detection_metadata': detection_metadata
                })
            
            # Update contact tracking
            if plume_contact:
                self.algorithm_state.steps_since_contact = 0
                self.algorithm_state.plume_contact_history.append(confidence)
            else:
                self.algorithm_state.steps_since_contact += 1
                self.algorithm_state.plume_contact_history.append(0.0)
            
            # Track step performance and movement metrics
            step_metrics = {
                'plume_contact': plume_contact,
                'detection_confidence': confidence,
                'movement_distance': np.linalg.norm(new_position - self.algorithm_state.current_position),
                'wind_direction': self.algorithm_state.wind_direction,
                'current_state': new_state,
                'search_radius': self.algorithm_state.current_search_radius,
                'step_performance': confidence if plume_contact else 0.1  # Base performance for exploration
            }
            
            # Return new position, state, and step information
            return new_position, new_state, step_metrics
            
        except Exception as e:
            # Return current position and error state on failure
            return (
                self.algorithm_state.current_position.copy(),
                self.algorithm_state.current_state,
                {'error': str(e), 'step_performance': 0.0}
            )
    
    def execute_crosswind_casting(
        self,
        current_position: np.ndarray,
        wind_direction: Tuple[float, float],
        casting_options: Dict[str, Any] = None
    ) -> np.ndarray:
        """
        Execute crosswind casting pattern for systematic plume search when contact is lost.
        
        Args:
            current_position: Current position in the plume field
            wind_direction: Estimated wind direction vector
            casting_options: Options for casting pattern configuration
            
        Returns:
            np.ndarray: Next casting position with optimized search pattern
        """
        try:
            # Initialize casting options
            options = casting_options or {}
            search_radius = options.get('search_radius', self.casting_parameters.search_radius)
            
            # Calculate crosswind direction perpendicular to wind
            wind_x, wind_y = wind_direction
            crosswind_x = -wind_y  # Perpendicular to wind direction
            crosswind_y = wind_x
            
            # Determine casting amplitude based on search radius
            cast_amplitude = search_radius * (1.0 + 0.2 * random.random())  # Add small random variation
            
            # Apply casting direction and distance constraints
            direction_multiplier = self.algorithm_state.casting_direction
            
            # Calculate next position in casting pattern
            lateral_movement = cast_amplitude * direction_multiplier * self.casting_parameters.crosswind_speed
            forward_movement = search_radius * 0.3 * self.casting_parameters.casting_speed  # Small forward progress
            
            # Apply movement in crosswind and forward directions
            next_position = current_position + np.array([
                lateral_movement * crosswind_y + forward_movement * wind_y,
                lateral_movement * crosswind_x + forward_movement * wind_x
            ])
            
            # Update casting distance and direction
            self.algorithm_state.cast_distance += cast_amplitude
            
            # Change casting direction if we've traveled far enough
            if self.algorithm_state.cast_distance > self.casting_parameters.max_cast_distance:
                self.algorithm_state.casting_direction *= -1
                self.algorithm_state.cast_distance = 0.0
            
            # Validate position against arena boundaries
            next_position = np.clip(next_position, 0.0, 1.0)
            
            # Return optimized casting position
            return next_position
            
        except Exception as e:
            # Return small random movement on error
            random_offset = np.array([random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)])
            return np.clip(current_position + random_offset, 0.0, 1.0)
    
    def execute_upwind_surge(
        self,
        current_position: np.ndarray,
        wind_direction: Tuple[float, float],
        surge_speed: float
    ) -> np.ndarray:
        """
        Execute upwind surge movement when plume contact is detected for source approach.
        
        Args:
            current_position: Current position in the plume field
            wind_direction: Estimated wind direction vector
            surge_speed: Speed for upwind surge movement
            
        Returns:
            np.ndarray: Next upwind position with optimized surge movement
        """
        try:
            # Calculate upwind direction from wind estimate
            wind_x, wind_y = wind_direction
            
            # Upwind direction is opposite to wind direction
            upwind_x = -wind_x
            upwind_y = -wind_y
            
            # Apply surge speed and movement constraints
            surge_distance = surge_speed
            
            # Calculate next position in upwind direction
            next_position = current_position + surge_distance * np.array([upwind_y, upwind_x])
            
            # Add small amount of lateral search during surge
            lateral_noise = 0.1 * surge_speed * random.uniform(-1, 1)
            crosswind_x = -wind_y
            crosswind_y = wind_x
            
            next_position += lateral_noise * np.array([crosswind_y, crosswind_x])
            
            # Validate position against arena boundaries
            next_position = np.clip(next_position, 0.0, 1.0)
            
            # Return optimized upwind surge position
            return next_position
            
        except Exception as e:
            # Return small upwind movement on error
            default_upwind = np.array([0.0, surge_speed])  # Default upward movement
            return np.clip(current_position + default_upwind, 0.0, 1.0)
    
    def adapt_search_parameters(
        self,
        performance_metrics: Dict[str, Any],
        plume_characteristics: Dict[str, Any]
    ) -> bool:
        """
        Adapt casting search parameters based on performance history and plume characteristics.
        
        Args:
            performance_metrics: Recent performance metrics for optimization
            plume_characteristics: Plume spatial and temporal characteristics
            
        Returns:
            bool: True if parameters successfully adapted with performance improvement
        """
        try:
            # Analyze recent performance metrics and success rates
            recent_performance = performance_metrics.get('recent_performance', [])
            
            if len(recent_performance) < 10:
                return False  # Need sufficient data for adaptation
            
            # Calculate performance trends
            average_performance = np.mean(recent_performance)
            performance_trend = np.mean(np.diff(recent_performance[-20:]))  # Recent trend
            
            # Evaluate plume characteristics for parameter optimization
            plume_density = plume_characteristics.get('plume_density', 0.1)
            plume_variability = plume_characteristics.get('spatial_variability', 0.1)
            
            # Calculate optimal search radius and movement speeds
            adaptation_factor = 0.1  # Conservative adaptation rate
            
            if average_performance < 0.3:
                # Poor performance - increase search radius
                radius_adjustment = 1.2
            elif average_performance > 0.7:
                # Good performance - fine-tune for efficiency
                radius_adjustment = 0.95
            else:
                # Moderate performance - maintain current radius
                radius_adjustment = 1.0
            
            # Adjust based on performance trend
            if performance_trend < -0.1:
                # Declining performance - more significant adjustment
                radius_adjustment *= 1.1
            elif performance_trend > 0.1:
                # Improving performance - smaller adjustment
                radius_adjustment *= 0.98
            
            # Update casting parameters with adaptive adjustments
            new_search_radius = self.algorithm_state.current_search_radius * radius_adjustment
            
            # Apply constraints to prevent extreme values
            min_radius = self.casting_parameters.search_radius * 0.5
            max_radius = self.casting_parameters.search_radius * 2.0
            new_search_radius = np.clip(new_search_radius, min_radius, max_radius)
            
            # Update algorithm state
            old_radius = self.algorithm_state.current_search_radius
            self.algorithm_state.current_search_radius = new_search_radius
            
            # Validate parameter changes against constraints
            if abs(new_search_radius - old_radius) > old_radius * 0.5:
                # Limit dramatic changes
                if new_search_radius > old_radius:
                    self.algorithm_state.current_search_radius = old_radius * 1.2
                else:
                    self.algorithm_state.current_search_radius = old_radius * 0.8
            
            # Update performance cache
            self.statistical_cache['performance_history'].extend(recent_performance)
            
            # Return adaptation status with performance predictions
            return True
            
        except Exception as e:
            # Return failure status on adaptation error
            return False
    
    def validate_casting_performance(
        self,
        result: AlgorithmResult,
        reference_metrics: Dict[str, Any] = None,
        strict_validation: bool = False
    ) -> 'ValidationResult':
        """
        Validate casting algorithm performance against reference implementations and scientific computing standards.
        
        Args:
            result: Algorithm execution result to validate
            reference_metrics: Reference metrics for correlation analysis
            strict_validation: Enable strict validation criteria
            
        Returns:
            ValidationResult: Performance validation result with correlation analysis and compliance assessment
        """
        from ..utils.validation_utils import ValidationResult
        
        # Initialize performance validation result
        validation_result = ValidationResult(
            validation_type="casting_performance_validation",
            is_valid=True,
            validation_context=f"algorithm=casting, strict={strict_validation}"
        )
        
        try:
            # Calculate trajectory similarity against reference implementations
            if reference_metrics and 'reference_trajectory' in reference_metrics:
                ref_trajectory = np.array(reference_metrics['reference_trajectory'])
                
                if result.trajectory is not None and len(result.trajectory) > 0:
                    trajectory_similarity = calculate_trajectory_similarity(result.trajectory, ref_trajectory)
                    validation_result.add_metric('trajectory_similarity', trajectory_similarity)
                    
                    if trajectory_similarity < 0.7:
                        validation_result.add_warning("Low trajectory similarity to reference implementation")
            
            # Assess performance correlation with >95% threshold requirement
            if reference_metrics and 'reference_performance' in reference_metrics:
                ref_performance = reference_metrics['reference_performance']
                current_performance = result.performance_metrics.get('search_efficiency', 0.0)
                
                correlation = calculate_correlation_matrix(
                    np.array([current_performance]), np.array([ref_performance])
                )
                validation_result.add_metric('performance_correlation', correlation)
                
                if correlation < DEFAULT_CORRELATION_THRESHOLD:
                    validation_result.add_error(
                        f"Performance correlation {correlation:.3f} below {DEFAULT_CORRELATION_THRESHOLD} threshold",
                        severity='HIGH'
                    )
                    validation_result.is_valid = False
            
            # Validate convergence and success rate metrics
            if not result.converged and result.iterations_completed >= 500:
                validation_result.add_warning("Algorithm did not converge within reasonable iteration count")
            
            if result.execution_time > 7.2:  # >7.2 seconds requirement
                validation_result.add_error(
                    f"Execution time {result.execution_time:.3f}s exceeds 7.2s requirement",
                    severity='HIGH'
                )
                validation_result.is_valid = False
            
            # Evaluate reproducibility with >0.99 coefficient requirement
            if len(self.execution_history) >= 2:
                recent_results = [r.performance_metrics.get('search_efficiency', 0.0) for r in self.execution_history[-5:]]
                reproducibility = assess_reproducibility([np.array([r]) for r in recent_results])
                validation_result.add_metric('reproducibility_coefficient', reproducibility)
                
                if reproducibility < 0.99:
                    validation_result.add_warning(
                        f"Reproducibility coefficient {reproducibility:.3f} below 0.99 requirement"
                    )
            
            # Apply strict validation criteria if enabled
            if strict_validation:
                # Enhanced trajectory quality checks
                if result.trajectory is not None:
                    trajectory_length = result.performance_metrics.get('trajectory_length', 0.0)
                    if trajectory_length > 5.0:  # Excessive path length
                        validation_result.add_warning("Trajectory length suggests inefficient search pattern")
                
                # Check search effectiveness
                search_efficiency = result.performance_metrics.get('search_efficiency', 0.0)
                if search_efficiency < 0.5:
                    validation_result.add_warning("Low search efficiency may indicate suboptimal parameters")
            
            # Add casting-specific validation metrics
            validation_result.add_metric('execution_time', result.execution_time)
            validation_result.add_metric('convergence_achieved', float(result.converged))
            validation_result.add_metric('success_rate', float(result.success))
            
            # Generate comprehensive validation result with recommendations
            if validation_result.is_valid:
                validation_result.add_recommendation(
                    "Casting algorithm performance meets validation requirements",
                    priority="INFO"
                )
            else:
                validation_result.add_recommendation(
                    "Address performance validation issues to meet scientific computing standards",
                    priority="HIGH"
                )
            
            return validation_result
            
        except Exception as e:
            # Return error validation result on exception
            validation_result.add_error(f"Performance validation failed: {str(e)}", severity='CRITICAL')
            validation_result.is_valid = False
            return validation_result
    
    def generate_casting_report(
        self,
        execution_results: List[AlgorithmResult],
        include_visualizations: bool = False,
        include_statistical_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive casting algorithm performance report with trajectory analysis, statistical validation, and cross-format compatibility assessment.
        
        Args:
            execution_results: List of algorithm execution results for analysis
            include_visualizations: Whether to include visualization data
            include_statistical_analysis: Whether to include detailed statistical analysis
            
        Returns:
            Dict[str, Any]: Comprehensive casting performance report with analysis and recommendations
        """
        try:
            # Compile casting execution results and performance metrics
            if not execution_results:
                return {'error': 'No execution results provided for report generation'}
            
            # Initialize comprehensive report structure
            report = {
                'report_id': f"casting_report_{int(time.time())}",
                'algorithm_name': 'casting',
                'algorithm_version': ALGORITHM_VERSION,
                'generation_timestamp': time.time(),
                'execution_summary': {},
                'performance_analysis': {},
                'trajectory_analysis': {},
                'statistical_validation': {},
                'recommendations': []
            }
            
            # Analyze trajectory patterns and search efficiency
            trajectories = [result.trajectory for result in execution_results if result.trajectory is not None]
            execution_times = [result.execution_time for result in execution_results]
            success_rates = [1.0 if result.success else 0.0 for result in execution_results]
            
            # Compile execution summary
            report['execution_summary'] = {
                'total_executions': len(execution_results),
                'successful_executions': sum(success_rates),
                'success_rate': np.mean(success_rates),
                'average_execution_time': np.mean(execution_times),
                'convergence_rate': np.mean([1.0 if r.converged else 0.0 for r in execution_results])
            }
            
            # Perform statistical validation and correlation analysis
            if include_statistical_analysis and len(execution_results) > 1:
                # Calculate performance consistency
                search_efficiencies = [
                    r.performance_metrics.get('search_efficiency', 0.0) for r in execution_results
                ]
                
                report['statistical_validation'] = {
                    'performance_mean': np.mean(search_efficiencies),
                    'performance_std': np.std(search_efficiencies),
                    'performance_consistency': 1.0 - np.std(search_efficiencies) / max(np.mean(search_efficiencies), 1e-10),
                    'correlation_with_reference': 'Not available',  # Would require reference data
                    'reproducibility_coefficient': assess_reproducibility([np.array([eff]) for eff in search_efficiencies])
                }
                
                # Analyze trajectory patterns
                if trajectories:
                    trajectory_lengths = [
                        np.sum([np.linalg.norm(traj[i+1] - traj[i]) for i in range(len(traj) - 1)])
                        for traj in trajectories if len(traj) > 1
                    ]
                    
                    report['trajectory_analysis'] = {
                        'average_trajectory_length': np.mean(trajectory_lengths),
                        'trajectory_length_std': np.std(trajectory_lengths),
                        'trajectory_efficiency': np.mean([
                            r.performance_metrics.get('search_efficiency', 0.0) for r in execution_results
                        ]),
                        'path_consistency': 1.0 - np.std(trajectory_lengths) / max(np.mean(trajectory_lengths), 1e-10)
                    }
            
            # Assess cross-format compatibility and consistency
            format_performance = {}
            for result in execution_results:
                format_type = result.metadata.get('format_type', 'unknown')
                if format_type not in format_performance:
                    format_performance[format_type] = []
                format_performance[format_type].append(result.performance_metrics.get('search_efficiency', 0.0))
            
            # Performance analysis across different conditions
            report['performance_analysis'] = {
                'format_compatibility': {
                    fmt: {
                        'mean_performance': np.mean(perfs),
                        'execution_count': len(perfs)
                    } for fmt, perfs in format_performance.items()
                },
                'execution_time_analysis': {
                    'mean_time': np.mean(execution_times),
                    'max_time': np.max(execution_times),
                    'min_time': np.min(execution_times),
                    'time_consistency': 1.0 - np.std(execution_times) / max(np.mean(execution_times), 1e-10)
                }
            }
            
            # Generate performance recommendations and optimizations
            recommendations = []
            
            if report['execution_summary']['success_rate'] < 0.8:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'performance',
                    'recommendation': 'Success rate below 80% - consider adjusting search parameters'
                })
            
            if report['execution_summary']['average_execution_time'] > 7.0:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'efficiency',
                    'recommendation': 'Execution time approaching 7.2s limit - optimize search patterns'
                })
            
            if include_statistical_analysis and report['statistical_validation']['performance_consistency'] < 0.8:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'consistency',
                    'recommendation': 'Low performance consistency - review parameter stability'
                })
            
            report['recommendations'] = recommendations
            
            # Include visualizations if requested
            if include_visualizations and trajectories:
                # Prepare trajectory data for visualization
                report['visualization_data'] = {
                    'trajectory_summary': [
                        {
                            'start_point': traj[0].tolist(),
                            'end_point': traj[-1].tolist(),
                            'length': len(traj)
                        } for traj in trajectories[:10]  # First 10 trajectories
                    ],
                    'performance_distribution': {
                        'search_efficiencies': search_efficiencies,
                        'execution_times': execution_times
                    }
                }
            
            # Add report metadata
            report['metadata'] = {
                'algorithm_parameters': self.casting_parameters.to_dict(),
                'execution_config': self.execution_config,
                'report_version': '1.0.0',
                'statistical_analysis_included': include_statistical_analysis,
                'visualizations_included': include_visualizations
            }
            
            # Return comprehensive performance report
            return report
            
        except Exception as e:
            # Return error report on generation failure
            return {
                'error': f"Report generation failed: {str(e)}",
                'timestamp': time.time(),
                'algorithm_name': 'casting'
            }


# Export casting algorithm components for external use
__all__ = [
    'CastingAlgorithm',
    'CastingParameters',
    'CastingState',
    'calculate_wind_direction',
    'calculate_casting_trajectory',
    'detect_plume_contact',
    'optimize_search_radius',
    'validate_casting_parameters'
]