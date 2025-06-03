"""
Comprehensive trajectory analysis module providing advanced trajectory comparison, similarity metrics calculation, 
movement pattern classification, and navigation strategy evaluation for plume navigation simulation systems.

This module implements sophisticated trajectory analysis including dynamic time warping, Hausdorff distance, 
Fréchet distance, path efficiency assessment, movement phase detection, and cross-algorithm trajectory comparison 
with statistical validation and scientific reproducibility requirements. Features intelligent caching, real-time 
analysis capabilities, and integration with performance metrics systems for scientific computing excellence.

Key Features:
- Advanced similarity metrics (DTW, Hausdorff, Fréchet, LCSS)
- Movement pattern classification using machine learning
- Path efficiency and optimization analysis
- Statistical validation and reproducibility assessment
- Intelligent caching and performance optimization
- Real-time analysis capabilities for batch processing
- Cross-algorithm trajectory comparison and ranking
- Scientific context integration and audit trails

Technical Requirements:
- >95% correlation with reference implementations
- >0.99 reproducibility coefficient validation
- <7.2 seconds average processing time per simulation
- Support for 4000+ simulation batch processing
- Comprehensive error handling and recovery mechanisms
"""

# External library imports with version specifications
import numpy as np  # numpy==2.1.3+ - Numerical array operations for trajectory data processing and mathematical calculations
import pandas as pd  # pandas==2.2.0+ - Data manipulation and analysis for trajectory datasets and statistical processing
from scipy.spatial.distance import pdist, cdist, squareform  # scipy==1.15.3+ - Distance metrics and similarity measures for trajectory comparison analysis
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, ttest_ind, bootstrap  # scipy==1.15.3+ - Statistical analysis and hypothesis testing for trajectory validation
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering  # scikit-learn==1.5.0+ - Clustering algorithms for movement pattern classification and trajectory grouping
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score  # scikit-learn==1.5.0+ - Machine learning metrics for trajectory classification and pattern recognition
from typing import Dict, List, Tuple, Optional, Any, Union, Callable  # typing==3.9+ - Type hints for trajectory analysis function signatures and data structures
from dataclasses import dataclass, field  # dataclasses==3.9+ - Data class decorators for trajectory analysis configuration and result structures
import datetime  # datetime==3.9+ - Timestamp generation and temporal analysis for trajectory processing
import time  # time==3.9+ - High-precision timing for trajectory analysis performance measurement
import threading  # threading==3.9+ - Thread-safe trajectory analysis operations and concurrent processing
import concurrent.futures  # concurrent.futures==3.9+ - Parallel trajectory analysis and batch processing optimization
from collections import defaultdict, deque  # collections==3.9+ - Efficient data structures for trajectory storage and analysis
import functools  # functools==3.9+ - Function decoration and caching for trajectory analysis optimization
import warnings  # warnings==3.9+ - Warning management for trajectory analysis edge cases and validation issues
import copy  # copy==3.9+ - Deep copying of trajectory data for isolation and thread safety
import json  # json==3.9+ - JSON serialization for trajectory analysis results export and reporting

# Internal imports for logging, statistics, simulation results, performance metrics, and scientific constants
from ...utils.logging_utils import (
    get_logger, log_performance_metrics, create_audit_trail, set_scientific_context
)
from ...utils.statistical_utils import (
    StatisticalAnalyzer, calculate_trajectory_similarity, calculate_correlation_matrix, assess_reproducibility
)
from ..simulation.result_collector import SimulationResult, BatchSimulationResult
from .performance_metrics import PerformanceMetricsCalculator, PathEfficiencyAnalyzer
from ...utils.scientific_constants import (
    DEFAULT_CORRELATION_THRESHOLD, REPRODUCIBILITY_THRESHOLD, SPATIAL_ACCURACY_THRESHOLD, 
    TEMPORAL_ACCURACY_THRESHOLD, get_performance_thresholds, get_statistical_constants
)

# =============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# =============================================================================

# Version identifier for trajectory analysis module
TRAJECTORY_ANALYSIS_VERSION: str = '1.0.0'

# Default similarity metrics for comprehensive trajectory comparison
DEFAULT_SIMILARITY_METRICS: List[str] = ['euclidean', 'hausdorff', 'frechet', 'dtw', 'lcss']

# Movement pattern types for navigation strategy classification
MOVEMENT_PATTERN_TYPES: List[str] = ['exploration', 'exploitation', 'casting', 'surge', 'spiral', 'random_walk']

# Trajectory features for comprehensive characterization and analysis
TRAJECTORY_FEATURES: List[str] = ['path_length', 'directness_index', 'sinuosity', 'velocity_profile', 'acceleration_profile', 'turning_angles']

# Feature flags for trajectory analysis capabilities
SIMILARITY_VALIDATION_ENABLED: bool = True
PATTERN_CLASSIFICATION_ENABLED: bool = True
STATISTICAL_ANALYSIS_ENABLED: bool = True
CACHING_ENABLED: bool = True
REAL_TIME_ANALYSIS: bool = True

# Performance configuration parameters
TRAJECTORY_CACHE_TTL_SECONDS: int = 3600
SIMILARITY_CALCULATION_TIMEOUT: float = 300.0
PATTERN_CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.8
TRAJECTORY_SMOOTHING_WINDOW: int = 5
MINIMUM_TRAJECTORY_LENGTH: int = 10
MAXIMUM_TRAJECTORY_LENGTH: int = 10000
SPATIAL_RESOLUTION_THRESHOLD: float = 0.01
TEMPORAL_RESOLUTION_THRESHOLD: float = 0.001

# Global thread-safe caches and data structures for performance optimization
_global_trajectory_cache: Dict[str, Any] = {}
_analysis_locks: Dict[str, threading.RLock] = {}
_pattern_classifiers: Dict[str, Any] = {}
_similarity_calculators: Dict[str, Any] = {}

# =============================================================================
# UTILITY FUNCTIONS FOR TRAJECTORY PROCESSING AND ANALYSIS
# =============================================================================

def _calculate_dynamic_time_warping(trajectory1: np.ndarray, trajectory2: np.ndarray, window_size: Optional[int] = None) -> float:
    """
    Calculate Dynamic Time Warping distance between two trajectories with optimal warping path computation.
    
    This function implements the dynamic time warping algorithm with optional Sakoe-Chiba band constraint
    for efficient computation of trajectory similarity with temporal alignment flexibility.
    
    Args:
        trajectory1: First trajectory as numpy array of shape (n_points, n_dimensions)
        trajectory2: Second trajectory as numpy array of shape (m_points, n_dimensions)
        window_size: Optional constraint window size for Sakoe-Chiba band
        
    Returns:
        float: DTW distance between trajectories with normalized alignment cost
    """
    n, m = len(trajectory1), len(trajectory2)
    
    # Initialize DTW matrix with infinity values
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Apply Sakoe-Chiba band constraint if window_size is specified
    if window_size is not None:
        window_size = max(window_size, abs(n - m))  # Ensure warping path exists
    
    # Fill DTW matrix with optimal warping costs
    for i in range(1, n + 1):
        start_j = max(1, i - window_size) if window_size else 1
        end_j = min(m + 1, i + window_size + 1) if window_size else m + 1
        
        for j in range(start_j, end_j):
            # Calculate Euclidean distance between trajectory points
            cost = np.linalg.norm(trajectory1[i-1] - trajectory2[j-1])
            
            # Find optimal warping path with minimum accumulated cost
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],     # Insertion
                dtw_matrix[i, j-1],     # Deletion
                dtw_matrix[i-1, j-1]    # Match
            )
    
    # Return normalized DTW distance
    return dtw_matrix[n, m] / max(n, m)


def _calculate_hausdorff_distance(trajectory1: np.ndarray, trajectory2: np.ndarray, directed: bool = False) -> float:
    """
    Calculate Hausdorff distance between trajectories with directional and bidirectional options.
    
    Args:
        trajectory1: First trajectory as numpy array
        trajectory2: Second trajectory as numpy array
        directed: Whether to calculate directed or bidirectional Hausdorff distance
        
    Returns:
        float: Hausdorff distance between trajectories
    """
    # Calculate distance matrix between all point pairs
    distances = cdist(trajectory1, trajectory2, metric='euclidean')
    
    # Calculate directed Hausdorff distances
    h1 = np.max(np.min(distances, axis=1))  # Max distance from trajectory1 to trajectory2
    
    if directed:
        return h1
    
    h2 = np.max(np.min(distances, axis=0))  # Max distance from trajectory2 to trajectory1
    
    # Return bidirectional Hausdorff distance
    return max(h1, h2)


def _calculate_frechet_distance(trajectory1: np.ndarray, trajectory2: np.ndarray) -> float:
    """
    Calculate discrete Fréchet distance between trajectories using dynamic programming.
    
    Args:
        trajectory1: First trajectory as numpy array
        trajectory2: Second trajectory as numpy array
        
    Returns:
        float: Fréchet distance between trajectories
    """
    n, m = len(trajectory1), len(trajectory2)
    
    # Initialize distance matrix
    distance_matrix = np.zeros((n, m))
    
    # Fill distance matrix with Euclidean distances
    for i in range(n):
        for j in range(m):
            distance_matrix[i, j] = np.linalg.norm(trajectory1[i] - trajectory2[j])
    
    # Initialize Fréchet distance matrix
    frechet_matrix = np.full((n, m), np.inf)
    frechet_matrix[0, 0] = distance_matrix[0, 0]
    
    # Fill first row and column
    for i in range(1, n):
        frechet_matrix[i, 0] = max(frechet_matrix[i-1, 0], distance_matrix[i, 0])
    for j in range(1, m):
        frechet_matrix[0, j] = max(frechet_matrix[0, j-1], distance_matrix[0, j])
    
    # Fill remaining matrix
    for i in range(1, n):
        for j in range(1, m):
            frechet_matrix[i, j] = max(
                distance_matrix[i, j],
                min(frechet_matrix[i-1, j], frechet_matrix[i, j-1], frechet_matrix[i-1, j-1])
            )
    
    return frechet_matrix[n-1, m-1]


def _calculate_lcss_similarity(trajectory1: np.ndarray, trajectory2: np.ndarray, epsilon: float = 0.1) -> float:
    """
    Calculate Longest Common Subsequence (LCSS) similarity between trajectories.
    
    Args:
        trajectory1: First trajectory as numpy array
        trajectory2: Second trajectory as numpy array
        epsilon: Distance threshold for considering points as matching
        
    Returns:
        float: LCSS similarity score between 0 and 1
    """
    n, m = len(trajectory1), len(trajectory2)
    
    # Initialize LCSS matrix
    lcss_matrix = np.zeros((n + 1, m + 1))
    
    # Fill LCSS matrix using dynamic programming
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if np.linalg.norm(trajectory1[i-1] - trajectory2[j-1]) <= epsilon:
                lcss_matrix[i, j] = lcss_matrix[i-1, j-1] + 1
            else:
                lcss_matrix[i, j] = max(lcss_matrix[i-1, j], lcss_matrix[i, j-1])
    
    # Return normalized LCSS similarity
    return lcss_matrix[n, m] / min(n, m)


def _smooth_trajectory(trajectory: np.ndarray, window_size: int = TRAJECTORY_SMOOTHING_WINDOW) -> np.ndarray:
    """
    Apply smoothing to trajectory data using moving average filter for noise reduction.
    
    Args:
        trajectory: Input trajectory as numpy array
        window_size: Size of smoothing window
        
    Returns:
        np.ndarray: Smoothed trajectory with reduced noise
    """
    if len(trajectory) < window_size:
        return trajectory.copy()
    
    smoothed = np.zeros_like(trajectory)
    half_window = window_size // 2
    
    # Apply moving average smoothing
    for i in range(len(trajectory)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(trajectory), i + half_window + 1)
        smoothed[i] = np.mean(trajectory[start_idx:end_idx], axis=0)
    
    return smoothed


def _normalize_trajectory(trajectory: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize trajectory coordinates for consistent comparison across different scales.
    
    Args:
        trajectory: Input trajectory as numpy array
        method: Normalization method ('minmax', 'zscore', 'unit')
        
    Returns:
        np.ndarray: Normalized trajectory
    """
    if method == 'minmax':
        # Min-max normalization to [0, 1] range
        min_vals = np.min(trajectory, axis=0)
        max_vals = np.max(trajectory, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        return (trajectory - min_vals) / range_vals
    
    elif method == 'zscore':
        # Z-score normalization to zero mean and unit variance
        mean_vals = np.mean(trajectory, axis=0)
        std_vals = np.std(trajectory, axis=0)
        std_vals[std_vals == 0] = 1  # Avoid division by zero
        return (trajectory - mean_vals) / std_vals
    
    elif method == 'unit':
        # Unit vector normalization
        norms = np.linalg.norm(trajectory, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return trajectory / norms
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# =============================================================================
# MAIN TRAJECTORY ANALYSIS FUNCTIONS
# =============================================================================

@log_performance_metrics('trajectory_similarity_calculation')
def calculate_trajectory_similarity_matrix(
    trajectories: List[np.ndarray],
    similarity_metrics: List[str] = DEFAULT_SIMILARITY_METRICS,
    normalize_trajectories: bool = True,
    validate_results: bool = SIMILARITY_VALIDATION_ENABLED,
    metric_parameters: Dict[str, Any] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate comprehensive trajectory similarity matrix using multiple distance metrics including Euclidean, 
    Hausdorff, Fréchet, dynamic time warping, and longest common subsequence for cross-algorithm trajectory 
    comparison and validation.
    
    This function computes similarity matrices using specified metrics with optimized algorithms and 
    comprehensive validation to ensure scientific accuracy and reproducibility in trajectory analysis.
    
    Args:
        trajectories: List of trajectory arrays for similarity analysis
        similarity_metrics: List of similarity metrics to compute
        normalize_trajectories: Whether to normalize trajectories for consistent comparison
        validate_results: Whether to validate similarity calculation results
        metric_parameters: Optional parameters for specific similarity metrics
        
    Returns:
        Dict[str, np.ndarray]: Similarity matrices for each metric with statistical validation and confidence intervals
    """
    # Initialize logger and set scientific context for trajectory similarity calculation
    logger = get_logger('trajectory.similarity', 'ANALYSIS')
    start_time = time.time()
    
    # Validate input parameters and trajectory data structure
    if not trajectories or len(trajectories) < 2:
        raise ValueError("At least two trajectories required for similarity calculation")
    
    if not all(isinstance(traj, np.ndarray) for traj in trajectories):
        raise TypeError("All trajectories must be numpy arrays")
    
    # Validate trajectory dimensions and length constraints
    for i, traj in enumerate(trajectories):
        if len(traj) < MINIMUM_TRAJECTORY_LENGTH:
            raise ValueError(f"Trajectory {i} below minimum length: {len(traj)} < {MINIMUM_TRAJECTORY_LENGTH}")
        if len(traj) > MAXIMUM_TRAJECTORY_LENGTH:
            raise ValueError(f"Trajectory {i} exceeds maximum length: {len(traj)} > {MAXIMUM_TRAJECTORY_LENGTH}")
    
    # Apply trajectory normalization if normalize_trajectories is enabled
    processed_trajectories = []
    if normalize_trajectories:
        logger.debug("Normalizing trajectories for consistent comparison")
        for traj in trajectories:
            normalized_traj = _normalize_trajectory(traj, method='minmax')
            processed_trajectories.append(normalized_traj)
    else:
        processed_trajectories = [traj.copy() for traj in trajectories]
    
    # Initialize metric parameters with defaults if not provided
    if metric_parameters is None:
        metric_parameters = {
            'dtw_window_size': None,
            'hausdorff_directed': False,
            'lcss_epsilon': 0.1,
            'euclidean_normalize': True
        }
    
    # Initialize similarity matrices dictionary for each metric
    similarity_matrices = {}
    n_trajectories = len(processed_trajectories)
    
    # Calculate similarity matrices using specified metrics with optimized algorithms
    for metric in similarity_metrics:
        logger.debug(f"Calculating {metric} similarity matrix")
        similarity_matrix = np.zeros((n_trajectories, n_trajectories))
        
        # Compute pairwise similarities for current metric
        for i in range(n_trajectories):
            for j in range(i, n_trajectories):
                if i == j:
                    # Self-similarity is always 1.0 for normalized metrics
                    similarity_value = 1.0 if metric in ['dtw', 'hausdorff', 'frechet'] else 0.0
                else:
                    # Calculate similarity based on specified metric
                    if metric == 'euclidean':
                        # Euclidean distance between trajectory centroids
                        centroid1 = np.mean(processed_trajectories[i], axis=0)
                        centroid2 = np.mean(processed_trajectories[j], axis=0)
                        distance = np.linalg.norm(centroid1 - centroid2)
                        similarity_value = 1.0 / (1.0 + distance) if metric_parameters.get('euclidean_normalize', True) else distance
                    
                    elif metric == 'dtw':
                        # Dynamic Time Warping with optional window constraint
                        distance = _calculate_dynamic_time_warping(
                            processed_trajectories[i], 
                            processed_trajectories[j],
                            window_size=metric_parameters.get('dtw_window_size')
                        )
                        similarity_value = 1.0 / (1.0 + distance)
                    
                    elif metric == 'hausdorff':
                        # Hausdorff distance with directional option
                        distance = _calculate_hausdorff_distance(
                            processed_trajectories[i],
                            processed_trajectories[j],
                            directed=metric_parameters.get('hausdorff_directed', False)
                        )
                        similarity_value = 1.0 / (1.0 + distance)
                    
                    elif metric == 'frechet':
                        # Discrete Fréchet distance
                        distance = _calculate_frechet_distance(
                            processed_trajectories[i],
                            processed_trajectories[j]
                        )
                        similarity_value = 1.0 / (1.0 + distance)
                    
                    elif metric == 'lcss':
                        # Longest Common Subsequence similarity
                        similarity_value = _calculate_lcss_similarity(
                            processed_trajectories[i],
                            processed_trajectories[j],
                            epsilon=metric_parameters.get('lcss_epsilon', 0.1)
                        )
                    
                    else:
                        logger.warning(f"Unknown similarity metric: {metric}")
                        similarity_value = 0.0
                
                # Fill symmetric matrix
                similarity_matrix[i, j] = similarity_value
                similarity_matrix[j, i] = similarity_value
        
        # Store similarity matrix for current metric
        similarity_matrices[metric] = similarity_matrix
        
        # Log metric completion with performance statistics
        logger.debug(f"Completed {metric} similarity matrix calculation")
    
    # Perform statistical validation if validate_results is enabled
    if validate_results and STATISTICAL_ANALYSIS_ENABLED:
        logger.debug("Performing statistical validation of similarity results")
        validation_results = {}
        
        for metric, matrix in similarity_matrices.items():
            # Validate matrix properties
            is_symmetric = np.allclose(matrix, matrix.T, atol=NUMERICAL_PRECISION_THRESHOLD)
            diagonal_correct = np.allclose(np.diag(matrix), 1.0 if metric != 'euclidean' else 0.0, atol=NUMERICAL_PRECISION_THRESHOLD)
            
            # Check for reasonable similarity ranges
            min_similarity = np.min(matrix)
            max_similarity = np.max(matrix)
            
            validation_results[metric] = {
                'is_symmetric': is_symmetric,
                'diagonal_correct': diagonal_correct,
                'min_similarity': min_similarity,
                'max_similarity': max_similarity,
                'valid_range': 0.0 <= min_similarity <= max_similarity <= 1.0
            }
            
            # Log validation warnings if necessary
            if not all(validation_results[metric].values()):
                logger.warning(f"Validation issues detected for {metric} similarity matrix")
    
    # Generate confidence intervals and significance testing for similarity measures
    if STATISTICAL_ANALYSIS_ENABLED:
        for metric, matrix in similarity_matrices.items():
            # Calculate statistical properties of similarity distribution
            upper_triangle_indices = np.triu_indices_from(matrix, k=1)
            similarity_values = matrix[upper_triangle_indices]
            
            # Add statistical metadata to similarity matrices
            similarity_matrices[f"{metric}_stats"] = {
                'mean_similarity': np.mean(similarity_values),
                'std_similarity': np.std(similarity_values),
                'median_similarity': np.median(similarity_values),
                'min_similarity': np.min(similarity_values),
                'max_similarity': np.max(similarity_values)
            }
    
    # Cache similarity matrices for performance optimization and reuse
    if CACHING_ENABLED:
        cache_key = f"similarity_matrices_{hash(str(trajectories))}"
        _global_trajectory_cache[cache_key] = {
            'matrices': similarity_matrices,
            'timestamp': datetime.datetime.now(),
            'parameters': metric_parameters
        }
    
    # Create comprehensive similarity analysis report with metadata
    calculation_time = time.time() - start_time
    analysis_metadata = {
        'calculation_time': calculation_time,
        'n_trajectories': n_trajectories,
        'metrics_computed': similarity_metrics,
        'normalization_applied': normalize_trajectories,
        'validation_performed': validate_results,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Log similarity calculation performance metrics for monitoring
    log_performance_metrics(
        metric_name='trajectory_similarity_calculation_time',
        metric_value=calculation_time,
        metric_unit='seconds',
        component='TRAJECTORY_ANALYSIS',
        metric_context={'n_trajectories': n_trajectories, 'metrics': similarity_metrics}
    )
    
    # Add metadata to results
    similarity_matrices['_metadata'] = analysis_metadata
    
    logger.info(f"Completed trajectory similarity matrix calculation for {n_trajectories} trajectories using {len(similarity_metrics)} metrics in {calculation_time:.3f}s")
    
    # Return similarity matrices with validation and statistical analysis
    return similarity_matrices


@log_performance_metrics('trajectory_feature_extraction')
def extract_trajectory_features(
    trajectory: np.ndarray,
    feature_types: List[str] = TRAJECTORY_FEATURES,
    include_temporal_features: bool = True,
    smooth_trajectory: bool = True,
    extraction_config: Dict[str, Any] = None
) -> Dict[str, float]:
    """
    Extract comprehensive trajectory features including path length, directness index, sinuosity, velocity 
    profiles, acceleration patterns, and turning angle distributions for movement pattern analysis and 
    algorithm characterization.
    
    This function computes a comprehensive set of trajectory features with statistical properties and 
    validation metadata for scientific analysis and algorithm comparison.
    
    Args:
        trajectory: Input trajectory as numpy array of shape (n_points, n_dimensions)
        feature_types: List of feature types to extract
        include_temporal_features: Whether to include temporal features in extraction
        smooth_trajectory: Whether to apply smoothing before feature extraction
        extraction_config: Optional configuration parameters for feature extraction
        
    Returns:
        Dict[str, float]: Extracted trajectory features with statistical properties and validation metadata
    """
    # Initialize logger and validate trajectory data
    logger = get_logger('trajectory.features', 'ANALYSIS')
    start_time = time.time()
    
    # Validate trajectory data and feature type specifications
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("Trajectory must be a numpy array")
    
    if len(trajectory) < MINIMUM_TRAJECTORY_LENGTH:
        raise ValueError(f"Trajectory too short for feature extraction: {len(trajectory)} < {MINIMUM_TRAJECTORY_LENGTH}")
    
    if trajectory.ndim != 2:
        raise ValueError(f"Trajectory must be 2D array, got {trajectory.ndim}D")
    
    # Initialize extraction configuration with defaults
    if extraction_config is None:
        extraction_config = {
            'smoothing_window': TRAJECTORY_SMOOTHING_WINDOW,
            'velocity_window': 3,
            'acceleration_window': 5,
            'turning_angle_window': 3,
            'sinuosity_window': 10
        }
    
    # Apply trajectory smoothing if smooth_trajectory is enabled
    processed_trajectory = trajectory.copy()
    if smooth_trajectory:
        logger.debug("Applying trajectory smoothing for feature extraction")
        processed_trajectory = _smooth_trajectory(
            trajectory, 
            window_size=extraction_config.get('smoothing_window', TRAJECTORY_SMOOTHING_WINDOW)
        )
    
    # Initialize features dictionary for extracted features
    features = {}
    
    # Calculate spatial features including path length and directness index
    if 'path_length' in feature_types:
        # Total path length as sum of segment distances
        segment_distances = np.linalg.norm(np.diff(processed_trajectory, axis=0), axis=1)
        features['path_length'] = np.sum(segment_distances)
        
        # Path length statistics
        features['path_length_mean_segment'] = np.mean(segment_distances)
        features['path_length_std_segment'] = np.std(segment_distances)
    
    if 'directness_index' in feature_types:
        # Directness index as ratio of straight-line distance to path length
        start_point = processed_trajectory[0]
        end_point = processed_trajectory[-1]
        straight_line_distance = np.linalg.norm(end_point - start_point)
        
        if features.get('path_length', 0) > 0:
            features['directness_index'] = straight_line_distance / features['path_length']
        else:
            features['directness_index'] = 0.0
        
        features['displacement'] = straight_line_distance
    
    # Compute sinuosity and curvature measures for path characterization
    if 'sinuosity' in feature_types:
        # Sinuosity as measure of path complexity and tortuosity
        window_size = extraction_config.get('sinuosity_window', 10)
        sinuosity_values = []
        
        for i in range(window_size, len(processed_trajectory) - window_size):
            segment = processed_trajectory[i-window_size:i+window_size+1]
            segment_path_length = np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1))
            segment_displacement = np.linalg.norm(segment[-1] - segment[0])
            
            if segment_displacement > 0:
                sinuosity_values.append(segment_path_length / segment_displacement)
            else:
                sinuosity_values.append(1.0)  # Straight line default
        
        if sinuosity_values:
            features['sinuosity'] = np.mean(sinuosity_values)
            features['sinuosity_std'] = np.std(sinuosity_values)
            features['sinuosity_max'] = np.max(sinuosity_values)
        else:
            features['sinuosity'] = 1.0
            features['sinuosity_std'] = 0.0
            features['sinuosity_max'] = 1.0
    
    # Extract velocity and acceleration profiles if include_temporal_features is enabled
    if include_temporal_features:
        if 'velocity_profile' in feature_types:
            # Calculate velocity profile from position differences
            velocity_window = extraction_config.get('velocity_window', 3)
            velocities = []
            
            for i in range(velocity_window, len(processed_trajectory) - velocity_window):
                position_diff = processed_trajectory[i+velocity_window] - processed_trajectory[i-velocity_window]
                time_diff = 2 * velocity_window  # Assuming unit time steps
                velocity_magnitude = np.linalg.norm(position_diff) / time_diff
                velocities.append(velocity_magnitude)
            
            if velocities:
                features['velocity_mean'] = np.mean(velocities)
                features['velocity_std'] = np.std(velocities)
                features['velocity_max'] = np.max(velocities)
                features['velocity_min'] = np.min(velocities)
            else:
                features['velocity_mean'] = 0.0
                features['velocity_std'] = 0.0
                features['velocity_max'] = 0.0
                features['velocity_min'] = 0.0
        
        if 'acceleration_profile' in feature_types:
            # Calculate acceleration profile from velocity changes
            acceleration_window = extraction_config.get('acceleration_window', 5)
            accelerations = []
            
            # First calculate velocities
            velocity_vectors = np.diff(processed_trajectory, axis=0)
            
            for i in range(acceleration_window, len(velocity_vectors) - acceleration_window):
                velocity_change = velocity_vectors[i+acceleration_window] - velocity_vectors[i-acceleration_window]
                time_diff = 2 * acceleration_window
                acceleration_magnitude = np.linalg.norm(velocity_change) / time_diff
                accelerations.append(acceleration_magnitude)
            
            if accelerations:
                features['acceleration_mean'] = np.mean(accelerations)
                features['acceleration_std'] = np.std(accelerations)
                features['acceleration_max'] = np.max(accelerations)
            else:
                features['acceleration_mean'] = 0.0
                features['acceleration_std'] = 0.0
                features['acceleration_max'] = 0.0
    
    # Calculate turning angle distributions and movement statistics
    if 'turning_angles' in feature_types:
        # Calculate turning angles between consecutive trajectory segments
        turning_angle_window = extraction_config.get('turning_angle_window', 3)
        turning_angles = []
        
        velocity_vectors = np.diff(processed_trajectory, axis=0)
        
        for i in range(turning_angle_window, len(velocity_vectors) - turning_angle_window):
            vec1 = velocity_vectors[i-turning_angle_window:i]
            vec2 = velocity_vectors[i:i+turning_angle_window]
            
            # Calculate mean direction vectors
            mean_vec1 = np.mean(vec1, axis=0)
            mean_vec2 = np.mean(vec2, axis=0)
            
            # Calculate turning angle
            norm1 = np.linalg.norm(mean_vec1)
            norm2 = np.linalg.norm(mean_vec2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(mean_vec1, mean_vec2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
                turning_angle = np.arccos(cos_angle)
                turning_angles.append(turning_angle)
        
        if turning_angles:
            features['turning_angle_mean'] = np.mean(turning_angles)
            features['turning_angle_std'] = np.std(turning_angles)
            features['turning_angle_max'] = np.max(turning_angles)
            features['total_turning'] = np.sum(turning_angles)
        else:
            features['turning_angle_mean'] = 0.0
            features['turning_angle_std'] = 0.0
            features['turning_angle_max'] = 0.0
            features['total_turning'] = 0.0
    
    # Apply feature-specific configurations and parameters
    for feature_name, feature_value in features.items():
        # Validate extracted features against scientific standards
        if not np.isfinite(feature_value):
            logger.warning(f"Non-finite feature value detected: {feature_name} = {feature_value}")
            features[feature_name] = 0.0
    
    # Add trajectory characterization metadata
    features['trajectory_length'] = len(trajectory)
    features['trajectory_dimensionality'] = trajectory.shape[1]
    features['extraction_time'] = time.time() - start_time
    
    # Generate feature extraction report with statistical properties
    logger.debug(f"Extracted {len(features)} features from trajectory of length {len(trajectory)}")
    
    # Validate extracted features against scientific standards
    for feature_name, feature_value in features.items():
        if feature_name.startswith('trajectory_') or feature_name == 'extraction_time':
            continue  # Skip metadata features
            
        # Check for reasonable feature ranges
        if feature_name in ['directness_index'] and not (0.0 <= feature_value <= 1.0):
            logger.warning(f"Feature {feature_name} outside expected range [0,1]: {feature_value}")
        
        if 'velocity' in feature_name and feature_value < 0:
            logger.warning(f"Negative velocity feature detected: {feature_name} = {feature_value}")
    
    # Log feature extraction performance and results
    extraction_time = time.time() - start_time
    log_performance_metrics(
        metric_name='trajectory_feature_extraction_time',
        metric_value=extraction_time,
        metric_unit='seconds',
        component='TRAJECTORY_ANALYSIS',
        metric_context={'n_features': len(features), 'trajectory_length': len(trajectory)}
    )
    
    logger.info(f"Extracted {len(features)} trajectory features in {extraction_time:.3f}s")
    
    # Return comprehensive feature dictionary with validation metadata
    return features


@log_performance_metrics('movement_pattern_classification')
def classify_movement_patterns(
    trajectories: List[np.ndarray],
    classification_method: str = 'kmeans',
    confidence_threshold: float = PATTERN_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    validate_classifications: bool = True,
    classifier_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Classify movement patterns in trajectories using machine learning algorithms and statistical pattern 
    recognition to identify exploration, exploitation, casting, surge, spiral, and random walk behaviors 
    with confidence assessment.
    
    This function applies specified classification methods with optimized parameters to identify movement 
    patterns and assess classification confidence for reliable pattern identification.
    
    Args:
        trajectories: List of trajectory arrays for pattern classification
        classification_method: Classification algorithm to use ('kmeans', 'dbscan', 'hierarchical')
        confidence_threshold: Minimum confidence score for pattern classifications
        validate_classifications: Whether to validate classification results
        classifier_config: Optional configuration parameters for classification algorithm
        
    Returns:
        Dict[str, Any]: Movement pattern classifications with confidence scores, validation results, and pattern transition analysis
    """
    # Initialize logger and set scientific context for pattern classification
    logger = get_logger('trajectory.patterns', 'ANALYSIS')
    start_time = time.time()
    
    # Extract trajectory features for pattern classification analysis
    logger.debug("Extracting features for movement pattern classification")
    feature_matrices = []
    trajectory_indices = []
    
    for i, trajectory in enumerate(trajectories):
        try:
            # Extract comprehensive features for each trajectory
            features = extract_trajectory_features(
                trajectory,
                feature_types=TRAJECTORY_FEATURES,
                include_temporal_features=True,
                smooth_trajectory=True
            )
            
            # Create feature vector excluding metadata
            feature_vector = []
            for feature_name in ['path_length', 'directness_index', 'sinuosity', 'velocity_mean', 'acceleration_mean', 'turning_angle_mean']:
                if feature_name in features:
                    feature_vector.append(features[feature_name])
                else:
                    feature_vector.append(0.0)
            
            feature_matrices.append(feature_vector)
            trajectory_indices.append(i)
            
        except Exception as e:
            logger.warning(f"Failed to extract features for trajectory {i}: {e}")
            continue
    
    if not feature_matrices:
        raise ValueError("No valid features extracted from trajectories for classification")
    
    # Convert to numpy array for machine learning algorithms
    feature_matrix = np.array(feature_matrices)
    
    # Normalize features for consistent classification
    feature_matrix = _normalize_trajectory(feature_matrix, method='zscore')
    
    # Initialize classifier configuration with defaults
    if classifier_config is None:
        classifier_config = {
            'kmeans_n_clusters': len(MOVEMENT_PATTERN_TYPES),
            'kmeans_random_state': 42,
            'dbscan_eps': 0.5,
            'dbscan_min_samples': 3,
            'hierarchical_n_clusters': len(MOVEMENT_PATTERN_TYPES),
            'hierarchical_linkage': 'ward'
        }
    
    # Apply specified classification method with optimized parameters
    logger.debug(f"Applying {classification_method} classification method")
    
    if classification_method == 'kmeans':
        # K-Means clustering for movement pattern identification
        classifier = KMeans(
            n_clusters=classifier_config.get('kmeans_n_clusters', len(MOVEMENT_PATTERN_TYPES)),
            random_state=classifier_config.get('kmeans_random_state', 42),
            n_init=10,
            max_iter=300
        )
        cluster_labels = classifier.fit_predict(feature_matrix)
        cluster_centers = classifier.cluster_centers_
        
        # Calculate confidence scores as distance to cluster centers
        confidence_scores = []
        for i, features in enumerate(feature_matrix):
            cluster_id = cluster_labels[i]
            distance_to_center = np.linalg.norm(features - cluster_centers[cluster_id])
            # Convert distance to confidence score (lower distance = higher confidence)
            confidence = 1.0 / (1.0 + distance_to_center)
            confidence_scores.append(confidence)
    
    elif classification_method == 'dbscan':
        # DBSCAN clustering for density-based pattern identification
        classifier = DBSCAN(
            eps=classifier_config.get('dbscan_eps', 0.5),
            min_samples=classifier_config.get('dbscan_min_samples', 3)
        )
        cluster_labels = classifier.fit_predict(feature_matrix)
        
        # Calculate confidence scores based on local density
        confidence_scores = []
        for i, features in enumerate(feature_matrix):
            # Calculate local density as inverse of average distance to neighbors
            distances = np.linalg.norm(feature_matrix - features, axis=1)
            k_nearest = np.partition(distances, min(5, len(distances)-1))[:5]
            avg_distance = np.mean(k_nearest[1:])  # Exclude self-distance
            confidence = 1.0 / (1.0 + avg_distance)
            confidence_scores.append(confidence)
    
    elif classification_method == 'hierarchical':
        # Hierarchical clustering for pattern identification
        classifier = AgglomerativeClustering(
            n_clusters=classifier_config.get('hierarchical_n_clusters', len(MOVEMENT_PATTERN_TYPES)),
            linkage=classifier_config.get('hierarchical_linkage', 'ward')
        )
        cluster_labels = classifier.fit_predict(feature_matrix)
        
        # Calculate confidence scores based on cluster cohesion
        confidence_scores = []
        for i, features in enumerate(feature_matrix):
            cluster_id = cluster_labels[i]
            # Find all points in same cluster
            cluster_points = feature_matrix[cluster_labels == cluster_id]
            # Calculate average distance within cluster
            if len(cluster_points) > 1:
                distances = np.linalg.norm(cluster_points - features, axis=1)
                avg_distance = np.mean(distances[distances > 0])  # Exclude self
                confidence = 1.0 / (1.0 + avg_distance)
            else:
                confidence = 1.0  # Single point cluster
            confidence_scores.append(confidence)
    
    else:
        raise ValueError(f"Unknown classification method: {classification_method}")
    
    # Map cluster labels to movement pattern types
    pattern_mapping = {}
    unique_clusters = np.unique(cluster_labels)
    
    # Assign pattern types based on cluster characteristics
    for i, cluster_id in enumerate(unique_clusters):
        if cluster_id == -1:  # DBSCAN noise cluster
            pattern_mapping[cluster_id] = 'random_walk'
        else:
            # Assign pattern types in order (could be improved with domain knowledge)
            pattern_types = MOVEMENT_PATTERN_TYPES.copy()
            if i < len(pattern_types):
                pattern_mapping[cluster_id] = pattern_types[i]
            else:
                pattern_mapping[cluster_id] = 'unknown'
    
    # Filter classifications by confidence threshold for reliability
    classifications = {}
    pattern_confidence_scores = {}
    
    for i, (trajectory_idx, cluster_label, confidence) in enumerate(zip(trajectory_indices, cluster_labels, confidence_scores)):
        pattern_type = pattern_mapping[cluster_label]
        
        if confidence >= confidence_threshold:
            classifications[trajectory_idx] = {
                'pattern_type': pattern_type,
                'confidence': confidence,
                'cluster_id': cluster_label,
                'features': feature_matrices[i]
            }
        else:
            classifications[trajectory_idx] = {
                'pattern_type': 'uncertain',
                'confidence': confidence,
                'cluster_id': cluster_label,
                'features': feature_matrices[i]
            }
        
        # Track confidence scores by pattern type
        if pattern_type not in pattern_confidence_scores:
            pattern_confidence_scores[pattern_type] = []
        pattern_confidence_scores[pattern_type].append(confidence)
    
    # Validate classifications if validate_classifications is enabled
    validation_results = {}
    if validate_classifications and STATISTICAL_ANALYSIS_ENABLED:
        logger.debug("Validating movement pattern classifications")
        
        # Calculate clustering quality metrics
        if len(unique_clusters) > 1 and -1 not in unique_clusters:  # Valid for silhouette score
            try:
                silhouette_avg = silhouette_score(feature_matrix, cluster_labels)
                calinski_harabasz = calinski_harabasz_score(feature_matrix, cluster_labels)
                
                validation_results = {
                    'silhouette_score': silhouette_avg,
                    'calinski_harabasz_score': calinski_harabasz,
                    'n_clusters': len(unique_clusters),
                    'classification_quality': 'good' if silhouette_avg > 0.5 else 'moderate' if silhouette_avg > 0.3 else 'poor'
                }
            except Exception as e:
                logger.warning(f"Failed to calculate clustering validation metrics: {e}")
                validation_results = {'validation_error': str(e)}
    
    # Analyze pattern transitions and temporal dynamics
    pattern_transition_analysis = {}
    if len(trajectories) > 1:
        # Analyze transitions between consecutive trajectories (simplified approach)
        pattern_sequence = [classifications[i]['pattern_type'] for i in range(len(classifications))]
        transition_counts = defaultdict(int)
        
        for i in range(len(pattern_sequence) - 1):
            transition = (pattern_sequence[i], pattern_sequence[i + 1])
            transition_counts[transition] += 1
        
        pattern_transition_analysis = {
            'transition_counts': dict(transition_counts),
            'pattern_sequence': pattern_sequence,
            'dominant_pattern': max(set(pattern_sequence), key=pattern_sequence.count) if pattern_sequence else 'none'
        }
    
    # Generate pattern classification report with statistical analysis
    classification_summary = {
        'total_trajectories': len(trajectories),
        'classified_trajectories': len([c for c in classifications.values() if c['pattern_type'] != 'uncertain']),
        'pattern_distribution': {},
        'average_confidence': np.mean(confidence_scores),
        'classification_method': classification_method,
        'confidence_threshold': confidence_threshold
    }
    
    # Calculate pattern distribution statistics
    for pattern_type in MOVEMENT_PATTERN_TYPES + ['uncertain', 'unknown']:
        count = len([c for c in classifications.values() if c['pattern_type'] == pattern_type])
        if count > 0:
            classification_summary['pattern_distribution'][pattern_type] = {
                'count': count,
                'percentage': (count / len(classifications)) * 100,
                'average_confidence': np.mean(pattern_confidence_scores.get(pattern_type, [0.0]))
            }
    
    # Cache classification results for performance optimization
    if CACHING_ENABLED:
        cache_key = f"pattern_classification_{hash(str(trajectories))}_{classification_method}"
        _global_trajectory_cache[cache_key] = {
            'classifications': classifications,
            'validation_results': validation_results,
            'timestamp': datetime.datetime.now()
        }
    
    # Log classification performance metrics and accuracy
    classification_time = time.time() - start_time
    log_performance_metrics(
        metric_name='movement_pattern_classification_time',
        metric_value=classification_time,
        metric_unit='seconds',
        component='TRAJECTORY_ANALYSIS',
        metric_context={
            'n_trajectories': len(trajectories),
            'method': classification_method,
            'average_confidence': classification_summary['average_confidence']
        }
    )
    
    logger.info(f"Classified {len(trajectories)} trajectories into movement patterns using {classification_method} in {classification_time:.3f}s")
    
    # Return comprehensive classification results with validation
    return {
        'classifications': classifications,
        'classification_summary': classification_summary,
        'validation_results': validation_results,
        'pattern_transition_analysis': pattern_transition_analysis,
        'feature_matrix': feature_matrix.tolist(),
        'cluster_labels': cluster_labels.tolist(),
        'confidence_scores': confidence_scores,
        'pattern_mapping': pattern_mapping,
        'processing_time': classification_time,
        'timestamp': datetime.datetime.now().isoformat()
    }


@log_performance_metrics('trajectory_efficiency_analysis')
def analyze_trajectory_efficiency(
    trajectory: np.ndarray,
    optimal_path: Optional[np.ndarray] = None,
    efficiency_metrics: Dict[str, Any] = None,
    include_coverage_analysis: bool = True,
    analysis_context: Dict[str, Any] = None
) -> Dict[str, float]:
    """
    Analyze trajectory efficiency including path optimality, exploration coverage, search effectiveness, 
    and resource utilization with comparison to theoretical optimal paths and benchmark algorithms.
    
    This function provides comprehensive efficiency analysis with optimality scores, coverage metrics, 
    and comparative assessment for trajectory optimization insights.
    
    Args:
        trajectory: Input trajectory for efficiency analysis
        optimal_path: Optional optimal path for comparison analysis
        efficiency_metrics: Dictionary of efficiency metrics to calculate
        include_coverage_analysis: Whether to include coverage analysis in efficiency assessment
        analysis_context: Additional context information for analysis
        
    Returns:
        Dict[str, float]: Trajectory efficiency analysis with optimality scores, coverage metrics, and comparative assessment
    """
    # Initialize logger and validate trajectory data
    logger = get_logger('trajectory.efficiency', 'ANALYSIS')
    start_time = time.time()
    
    # Validate trajectory input and analysis parameters
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("Trajectory must be a numpy array")
    
    if len(trajectory) < MINIMUM_TRAJECTORY_LENGTH:
        raise ValueError(f"Trajectory too short for efficiency analysis: {len(trajectory)}")
    
    # Initialize efficiency metrics configuration
    if efficiency_metrics is None:
        efficiency_metrics = {
            'path_optimality': True,
            'directness_index': True,
            'exploration_coverage': True,
            'search_effectiveness': True,
            'resource_utilization': True,
            'time_efficiency': True
        }
    
    # Initialize analysis context with defaults
    if analysis_context is None:
        analysis_context = {
            'arena_bounds': None,
            'target_locations': None,
            'time_steps': len(trajectory),
            'resource_budget': None
        }
    
    # Initialize efficiency results dictionary
    efficiency_results = {}
    
    # Calculate path length ratio and directness index for efficiency assessment
    if efficiency_metrics.get('path_optimality', True):
        # Calculate actual path length
        segment_distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        actual_path_length = np.sum(segment_distances)
        
        # Calculate straight-line distance (optimal path length)
        start_point = trajectory[0]
        end_point = trajectory[-1]
        optimal_distance = np.linalg.norm(end_point - start_point)
        
        # Path optimality ratio
        if optimal_distance > 0:
            efficiency_results['path_optimality_ratio'] = optimal_distance / actual_path_length
        else:
            efficiency_results['path_optimality_ratio'] = 1.0  # No displacement case
        
        efficiency_results['actual_path_length'] = actual_path_length
        efficiency_results['optimal_path_length'] = optimal_distance
    
    if efficiency_metrics.get('directness_index', True):
        # Directness index as measure of path efficiency
        efficiency_results['directness_index'] = efficiency_results.get('path_optimality_ratio', 0.0)
    
    # Compare trajectory to optimal path if provided for optimality analysis
    if optimal_path is not None and efficiency_metrics.get('path_optimality', True):
        logger.debug("Comparing trajectory to provided optimal path")
        
        # Calculate optimal path length
        optimal_segments = np.linalg.norm(np.diff(optimal_path, axis=0), axis=1)
        optimal_path_length = np.sum(optimal_segments)
        
        # Path length ratio comparison
        if optimal_path_length > 0:
            efficiency_results['optimal_path_comparison'] = optimal_path_length / efficiency_results['actual_path_length']
        else:
            efficiency_results['optimal_path_comparison'] = 1.0
        
        # Calculate similarity to optimal path
        try:
            # Use DTW distance for path similarity
            path_similarity = _calculate_dynamic_time_warping(trajectory, optimal_path)
            efficiency_results['path_similarity_score'] = 1.0 / (1.0 + path_similarity)
        except Exception as e:
            logger.warning(f"Failed to calculate path similarity: {e}")
            efficiency_results['path_similarity_score'] = 0.0
    
    # Analyze exploration coverage and search effectiveness metrics
    if efficiency_metrics.get('exploration_coverage', True) and include_coverage_analysis:
        logger.debug("Analyzing exploration coverage")
        
        # Calculate convex hull area for coverage assessment
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(trajectory)
            explored_area = hull.volume  # In 2D, volume is area
            efficiency_results['explored_area'] = explored_area
        except Exception as e:
            logger.warning(f"Failed to calculate convex hull area: {e}")
            efficiency_results['explored_area'] = 0.0
        
        # Calculate trajectory spread and distribution
        trajectory_center = np.mean(trajectory, axis=0)
        distances_from_center = np.linalg.norm(trajectory - trajectory_center, axis=1)
        efficiency_results['exploration_radius'] = np.max(distances_from_center)
        efficiency_results['exploration_uniformity'] = 1.0 / (1.0 + np.std(distances_from_center))
        
        # Grid-based coverage analysis
        if analysis_context.get('arena_bounds') is not None:
            arena_bounds = analysis_context['arena_bounds']
            grid_size = 20  # 20x20 grid for coverage analysis
            
            # Create grid
            x_grid = np.linspace(arena_bounds[0], arena_bounds[1], grid_size)
            y_grid = np.linspace(arena_bounds[2], arena_bounds[3], grid_size)
            
            # Calculate visited grid cells
            visited_cells = set()
            for point in trajectory:
                x_idx = np.digitize(point[0], x_grid) - 1
                y_idx = np.digitize(point[1], y_grid) - 1
                if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                    visited_cells.add((x_idx, y_idx))
            
            total_cells = grid_size * grid_size
            efficiency_results['grid_coverage_ratio'] = len(visited_cells) / total_cells
        else:
            efficiency_results['grid_coverage_ratio'] = 0.0
    
    if efficiency_metrics.get('search_effectiveness', True):
        # Search effectiveness based on target proximity
        if analysis_context.get('target_locations') is not None:
            target_locations = analysis_context['target_locations']
            min_distances_to_targets = []
            
            for target in target_locations:
                distances_to_target = np.linalg.norm(trajectory - target, axis=1)
                min_distance = np.min(distances_to_target)
                min_distances_to_targets.append(min_distance)
            
            efficiency_results['min_target_distance'] = np.min(min_distances_to_targets)
            efficiency_results['average_target_distance'] = np.mean(min_distances_to_targets)
            efficiency_results['target_proximity_score'] = 1.0 / (1.0 + efficiency_results['min_target_distance'])
        else:
            # Generic search effectiveness based on trajectory characteristics
            # Use trajectory variance as proxy for search coverage
            trajectory_variance = np.var(trajectory, axis=0)
            efficiency_results['search_diversity'] = np.mean(trajectory_variance)
    
    # Calculate resource utilization and time efficiency measures
    if efficiency_metrics.get('resource_utilization', True):
        # Time efficiency based on trajectory length and displacement
        time_steps = analysis_context.get('time_steps', len(trajectory))
        displacement = efficiency_results.get('optimal_path_length', 0.0)
        
        if time_steps > 0:
            efficiency_results['time_efficiency'] = displacement / time_steps
        else:
            efficiency_results['time_efficiency'] = 0.0
        
        # Movement efficiency based on actual vs. theoretical minimum time
        if displacement > 0:
            # Assume maximum movement speed of 1 unit per time step
            theoretical_min_time = displacement
            efficiency_results['movement_efficiency'] = theoretical_min_time / time_steps
        else:
            efficiency_results['movement_efficiency'] = 1.0
        
        # Resource budget utilization if provided
        if analysis_context.get('resource_budget') is not None:
            resource_budget = analysis_context['resource_budget']
            actual_resource_usage = efficiency_results['actual_path_length']
            if resource_budget > 0:
                efficiency_results['resource_utilization_ratio'] = actual_resource_usage / resource_budget
            else:
                efficiency_results['resource_utilization_ratio'] = 0.0
    
    if efficiency_metrics.get('time_efficiency', True):
        # Additional time-based efficiency metrics
        # Calculate time to first target encounter (if targets provided)
        if analysis_context.get('target_locations') is not None:
            target_locations = analysis_context['target_locations']
            encounter_threshold = 0.1  # Define encounter distance
            
            time_to_first_encounter = None
            for i, point in enumerate(trajectory):
                for target in target_locations:
                    distance = np.linalg.norm(point - target)
                    if distance <= encounter_threshold:
                        time_to_first_encounter = i
                        break
                if time_to_first_encounter is not None:
                    break
            
            if time_to_first_encounter is not None:
                efficiency_results['time_to_first_encounter'] = time_to_first_encounter
                efficiency_results['encounter_efficiency'] = 1.0 / (1.0 + time_to_first_encounter)
            else:
                efficiency_results['time_to_first_encounter'] = len(trajectory)
                efficiency_results['encounter_efficiency'] = 0.0
    
    # Generate efficiency optimization recommendations
    optimization_recommendations = []
    
    if efficiency_results.get('path_optimality_ratio', 0.0) < 0.5:
        optimization_recommendations.append("Consider more direct movement patterns to improve path optimality")
    
    if efficiency_results.get('grid_coverage_ratio', 0.0) < 0.3:
        optimization_recommendations.append("Increase exploration coverage to improve search effectiveness")
    
    if efficiency_results.get('movement_efficiency', 0.0) < 0.7:
        optimization_recommendations.append("Optimize movement patterns to reduce time and resource consumption")
    
    efficiency_results['optimization_recommendations'] = optimization_recommendations
    
    # Validate efficiency metrics against performance standards
    validation_status = {}
    performance_thresholds = get_performance_thresholds('efficiency')
    
    for metric_name, metric_value in efficiency_results.items():
        if isinstance(metric_value, (int, float)):
            # Basic range validation
            if 0.0 <= metric_value <= 1.0 and 'ratio' in metric_name:
                validation_status[metric_name] = 'valid'
            elif metric_value >= 0.0:
                validation_status[metric_name] = 'valid'
            else:
                validation_status[metric_name] = 'invalid'
                logger.warning(f"Invalid efficiency metric value: {metric_name} = {metric_value}")
    
    efficiency_results['validation_status'] = validation_status
    
    # Create comprehensive efficiency analysis report
    analysis_time = time.time() - start_time
    efficiency_results['analysis_metadata'] = {
        'analysis_time': analysis_time,
        'trajectory_length': len(trajectory),
        'metrics_calculated': list(efficiency_metrics.keys()),
        'include_coverage_analysis': include_coverage_analysis,
        'optimal_path_provided': optimal_path is not None,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Log efficiency analysis performance and results
    log_performance_metrics(
        metric_name='trajectory_efficiency_analysis_time',
        metric_value=analysis_time,
        metric_unit='seconds',
        component='TRAJECTORY_ANALYSIS',
        metric_context={
            'trajectory_length': len(trajectory),
            'metrics_count': len(efficiency_metrics),
            'include_coverage': include_coverage_analysis
        }
    )
    
    logger.info(f"Completed trajectory efficiency analysis in {analysis_time:.3f}s with {len(efficiency_results)} metrics")
    
    # Return detailed efficiency assessment with optimization insights
    return efficiency_results


@log_performance_metrics('trajectory_phase_detection')
def detect_trajectory_phases(
    trajectory: np.ndarray,
    detection_method: str = 'changepoint',
    phase_parameters: Dict[str, float] = None,
    validate_phases: bool = True,
    include_transition_analysis: bool = True
) -> Dict[str, Any]:
    """
    Detect distinct phases in trajectory movement including search initiation, exploration, exploitation, 
    and target approach phases using change point detection and statistical segmentation algorithms.
    
    This function segments trajectories into distinct movement phases and analyzes phase characteristics 
    and transitions for comprehensive trajectory understanding.
    
    Args:
        trajectory: Input trajectory for phase detection
        detection_method: Phase detection algorithm ('changepoint', 'velocity', 'direction', 'clustering')
        phase_parameters: Parameters for phase detection algorithm
        validate_phases: Whether to validate detected phases
        include_transition_analysis: Whether to analyze phase transitions
        
    Returns:
        Dict[str, Any]: Trajectory phase detection results with phase boundaries, characteristics, and transition analysis
    """
    # Initialize logger and validate input parameters
    logger = get_logger('trajectory.phases', 'ANALYSIS')
    start_time = time.time()
    
    # Validate trajectory data and detection parameters
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("Trajectory must be a numpy array")
    
    if len(trajectory) < MINIMUM_TRAJECTORY_LENGTH:
        raise ValueError(f"Trajectory too short for phase detection: {len(trajectory)}")
    
    # Initialize phase detection parameters with defaults
    if phase_parameters is None:
        phase_parameters = {
            'changepoint_penalty': 2.0,
            'velocity_threshold': 0.5,
            'direction_change_threshold': np.pi/4,
            'min_phase_length': 5,
            'clustering_n_clusters': 4
        }
    
    # Initialize phase detection results
    phase_results = {
        'phase_boundaries': [],
        'phase_labels': [],
        'phase_characteristics': {},
        'detection_method': detection_method
    }
    
    # Apply change point detection algorithm for phase boundary identification
    if detection_method == 'changepoint':
        logger.debug("Applying change point detection for phase boundaries")
        
        # Calculate velocity profile for change point detection
        velocities = []
        for i in range(1, len(trajectory)):
            velocity = np.linalg.norm(trajectory[i] - trajectory[i-1])
            velocities.append(velocity)
        
        velocities = np.array(velocities)
        
        # Simple change point detection based on velocity changes
        penalty = phase_parameters.get('changepoint_penalty', 2.0)
        min_phase_length = int(phase_parameters.get('min_phase_length', 5))
        
        change_points = [0]  # Start with beginning
        
        # Sliding window approach for change point detection
        window_size = min_phase_length * 2
        for i in range(window_size, len(velocities) - window_size, min_phase_length):
            # Compare statistics before and after potential change point
            before_window = velocities[i-window_size:i]
            after_window = velocities[i:i+window_size]
            
            # Calculate statistical difference
            mean_diff = abs(np.mean(after_window) - np.mean(before_window))
            var_diff = abs(np.var(after_window) - np.var(before_window))
            
            # Detect change point if difference exceeds threshold
            if mean_diff > penalty * np.std(velocities) or var_diff > penalty * np.var(velocities):
                change_points.append(i)
        
        change_points.append(len(trajectory) - 1)  # Add end point
        phase_results['phase_boundaries'] = change_points
    
    elif detection_method == 'velocity':
        logger.debug("Using velocity-based phase detection")
        
        # Calculate velocity magnitudes
        velocities = []
        for i in range(1, len(trajectory)):
            velocity = np.linalg.norm(trajectory[i] - trajectory[i-1])
            velocities.append(velocity)
        
        velocities = np.array(velocities)
        velocity_threshold = phase_parameters.get('velocity_threshold', 0.5)
        min_phase_length = int(phase_parameters.get('min_phase_length', 5))
        
        # Detect phases based on velocity thresholds
        high_velocity_phases = velocities > velocity_threshold
        
        # Find phase boundaries based on velocity changes
        change_points = [0]
        current_phase = high_velocity_phases[0]
        phase_start = 0
        
        for i in range(1, len(high_velocity_phases)):
            if high_velocity_phases[i] != current_phase:
                if i - phase_start >= min_phase_length:
                    change_points.append(i)
                    current_phase = high_velocity_phases[i]
                    phase_start = i
        
        change_points.append(len(trajectory) - 1)
        phase_results['phase_boundaries'] = change_points
    
    elif detection_method == 'direction':
        logger.debug("Using direction change-based phase detection")
        
        # Calculate direction changes (turning angles)
        direction_changes = []
        for i in range(2, len(trajectory)):
            vec1 = trajectory[i-1] - trajectory[i-2]
            vec2 = trajectory[i] - trajectory[i-1]
            
            # Calculate angle between vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                direction_changes.append(angle)
            else:
                direction_changes.append(0.0)
        
        direction_changes = np.array(direction_changes)
        direction_threshold = phase_parameters.get('direction_change_threshold', np.pi/4)
        min_phase_length = int(phase_parameters.get('min_phase_length', 5))
        
        # Detect phases based on significant direction changes
        significant_changes = direction_changes > direction_threshold
        
        # Find phase boundaries
        change_points = [0]
        for i in range(len(significant_changes)):
            if significant_changes[i]:
                # Check if enough time has passed since last change point
                if len(change_points) == 1 or i - change_points[-1] >= min_phase_length:
                    change_points.append(i + 2)  # Adjust for offset
        
        change_points.append(len(trajectory) - 1)
        phase_results['phase_boundaries'] = change_points
    
    elif detection_method == 'clustering':
        logger.debug("Using clustering-based phase detection")
        
        # Extract features for each trajectory segment
        window_size = int(phase_parameters.get('min_phase_length', 5))
        features = []
        
        for i in range(window_size, len(trajectory) - window_size):
            segment = trajectory[i-window_size:i+window_size+1]
            
            # Calculate segment features
            segment_features = extract_trajectory_features(
                segment,
                feature_types=['path_length', 'directness_index', 'velocity_mean'],
                include_temporal_features=True,
                smooth_trajectory=False
            )
            
            feature_vector = [
                segment_features.get('path_length', 0.0),
                segment_features.get('directness_index', 0.0),
                segment_features.get('velocity_mean', 0.0)
            ]
            features.append(feature_vector)
        
        if features:
            features = np.array(features)
            
            # Apply clustering to identify phases
            n_clusters = int(phase_parameters.get('clustering_n_clusters', 4))
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(features)
            
            # Find phase boundaries based on cluster changes
            change_points = [0]
            current_cluster = cluster_labels[0]
            
            for i in range(1, len(cluster_labels)):
                if cluster_labels[i] != current_cluster:
                    change_points.append(i + window_size)
                    current_cluster = cluster_labels[i]
            
            change_points.append(len(trajectory) - 1)
            phase_results['phase_boundaries'] = change_points
    
    else:
        raise ValueError(f"Unknown detection method: {detection_method}")
    
    # Segment trajectory into distinct movement phases
    phase_boundaries = phase_results['phase_boundaries']
    phase_segments = []
    phase_labels = []
    
    for i in range(len(phase_boundaries) - 1):
        start_idx = phase_boundaries[i]
        end_idx = phase_boundaries[i + 1]
        phase_segment = trajectory[start_idx:end_idx + 1]
        phase_segments.append(phase_segment)
        
        # Assign phase label based on characteristics
        if i == 0:
            phase_labels.append('initiation')
        elif i == len(phase_boundaries) - 2:
            phase_labels.append('termination')
        else:
            # Classify intermediate phases based on characteristics
            phase_features = extract_trajectory_features(
                phase_segment,
                feature_types=['directness_index', 'velocity_mean', 'sinuosity'],
                include_temporal_features=True
            )
            
            directness = phase_features.get('directness_index', 0.0)
            velocity = phase_features.get('velocity_mean', 0.0)
            sinuosity = phase_features.get('sinuosity', 1.0)
            
            # Simple rule-based phase classification
            if directness > 0.7 and velocity > 0.5:
                phase_labels.append('directed_movement')
            elif sinuosity > 2.0 and velocity < 0.3:
                phase_labels.append('exploration')
            elif directness < 0.3 and velocity > 0.4:
                phase_labels.append('search')
            else:
                phase_labels.append('mixed_behavior')
    
    phase_results['phase_labels'] = phase_labels
    phase_results['phase_segments'] = phase_segments
    
    # Characterize each phase with movement statistics and features
    for i, (segment, label) in enumerate(zip(phase_segments, phase_labels)):
        phase_characteristics = extract_trajectory_features(
            segment,
            feature_types=TRAJECTORY_FEATURES,
            include_temporal_features=True
        )
        
        # Add phase-specific metrics
        phase_characteristics.update({
            'phase_duration': len(segment),
            'phase_start_time': phase_boundaries[i],
            'phase_end_time': phase_boundaries[i + 1],
            'phase_label': label
        })
        
        phase_results['phase_characteristics'][f'phase_{i}'] = phase_characteristics
    
    # Validate phase detection results if validate_phases is enabled
    validation_results = {}
    if validate_phases:
        logger.debug("Validating phase detection results")
        
        # Check phase length constraints
        min_phase_length = phase_parameters.get('min_phase_length', 5)
        valid_phases = 0
        
        for i, (start, end) in enumerate(zip(phase_boundaries[:-1], phase_boundaries[1:])):
            phase_length = end - start
            if phase_length >= min_phase_length:
                valid_phases += 1
        
        validation_results = {
            'total_phases': len(phase_segments),
            'valid_phases': valid_phases,
            'validation_ratio': valid_phases / len(phase_segments) if phase_segments else 0.0,
            'average_phase_length': np.mean([len(seg) for seg in phase_segments]) if phase_segments else 0.0,
            'phase_length_std': np.std([len(seg) for seg in phase_segments]) if phase_segments else 0.0
        }
    
    # Analyze phase transitions if include_transition_analysis is enabled
    transition_analysis = {}
    if include_transition_analysis and len(phase_labels) > 1:
        logger.debug("Analyzing phase transitions")
        
        # Calculate transition probabilities
        transition_counts = defaultdict(int)
        for i in range(len(phase_labels) - 1):
            transition = (phase_labels[i], phase_labels[i + 1])
            transition_counts[transition] += 1
        
        # Convert to probabilities
        total_transitions = sum(transition_counts.values())
        transition_probabilities = {}
        if total_transitions > 0:
            for transition, count in transition_counts.items():
                transition_probabilities[transition] = count / total_transitions
        
        # Analyze transition characteristics
        transition_points = []
        for i in range(len(phase_boundaries) - 1):
            transition_point = phase_boundaries[i]
            if transition_point > 0 and transition_point < len(trajectory) - 1:
                # Calculate velocity and direction change at transition
                before_velocity = np.linalg.norm(trajectory[transition_point] - trajectory[transition_point - 1])
                after_velocity = np.linalg.norm(trajectory[transition_point + 1] - trajectory[transition_point])
                
                transition_points.append({
                    'time': transition_point,
                    'before_velocity': before_velocity,
                    'after_velocity': after_velocity,
                    'velocity_change': after_velocity - before_velocity,
                    'from_phase': phase_labels[i] if i < len(phase_labels) else 'unknown',
                    'to_phase': phase_labels[i + 1] if i + 1 < len(phase_labels) else 'unknown'
                })
        
        transition_analysis = {
            'transition_counts': dict(transition_counts),
            'transition_probabilities': transition_probabilities,
            'transition_points': transition_points,
            'dominant_transition': max(transition_counts.keys(), key=transition_counts.get) if transition_counts else None
        }
    
    # Calculate phase duration and movement characteristics
    phase_statistics = {
        'total_phases': len(phase_segments),
        'average_phase_duration': np.mean([len(seg) for seg in phase_segments]) if phase_segments else 0.0,
        'phase_duration_std': np.std([len(seg) for seg in phase_segments]) if phase_segments else 0.0,
        'longest_phase': max([len(seg) for seg in phase_segments]) if phase_segments else 0,
        'shortest_phase': min([len(seg) for seg in phase_segments]) if phase_segments else 0,
        'phase_distribution': dict(zip(*np.unique(phase_labels, return_counts=True))) if phase_labels else {}
    }
    
    # Generate phase detection confidence scores and validation
    confidence_scores = {}
    for i, segment in enumerate(phase_segments):
        # Calculate confidence based on phase characteristics consistency
        if len(segment) >= phase_parameters.get('min_phase_length', 5):
            confidence_scores[f'phase_{i}'] = 0.8  # High confidence for valid length
        else:
            confidence_scores[f'phase_{i}'] = 0.3  # Low confidence for short phases
    
    phase_results.update({
        'validation_results': validation_results,
        'transition_analysis': transition_analysis,
        'phase_statistics': phase_statistics,
        'confidence_scores': confidence_scores
    })
    
    # Create comprehensive phase analysis report
    analysis_time = time.time() - start_time
    phase_results['analysis_metadata'] = {
        'analysis_time': analysis_time,
        'trajectory_length': len(trajectory),
        'detection_method': detection_method,
        'parameters': phase_parameters,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Log phase detection performance and accuracy
    log_performance_metrics(
        metric_name='trajectory_phase_detection_time',
        metric_value=analysis_time,
        metric_unit='seconds',
        component='TRAJECTORY_ANALYSIS',
        metric_context={
            'trajectory_length': len(trajectory),
            'method': detection_method,
            'phases_detected': len(phase_segments)
        }
    )
    
    logger.info(f"Detected {len(phase_segments)} trajectory phases using {detection_method} method in {analysis_time:.3f}s")
    
    # Return detailed phase detection results with analysis
    return phase_results


@log_performance_metrics('cross_algorithm_trajectory_comparison')
def compare_algorithm_trajectories(
    algorithm_trajectories: Dict[str, List[np.ndarray]],
    comparison_metrics: List[str] = DEFAULT_SIMILARITY_METRICS,
    include_statistical_tests: bool = STATISTICAL_ANALYSIS_ENABLED,
    generate_rankings: bool = True,
    comparison_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Compare trajectories across different navigation algorithms with statistical analysis, performance 
    ranking, and optimization recommendations for comprehensive algorithm evaluation and scientific validation.
    
    This function performs comprehensive cross-algorithm comparison with statistical significance testing 
    and performance ranking for scientific algorithm evaluation.
    
    Args:
        algorithm_trajectories: Dictionary mapping algorithm names to lists of trajectories
        comparison_metrics: List of metrics to use for comparison
        include_statistical_tests: Whether to perform statistical significance testing
        generate_rankings: Whether to generate algorithm performance rankings
        comparison_config: Optional configuration parameters for comparison
        
    Returns:
        Dict[str, Any]: Cross-algorithm trajectory comparison with statistical analysis, rankings, and optimization recommendations
    """
    # Initialize logger and set scientific context for cross-algorithm comparison
    logger = get_logger('trajectory.comparison', 'ANALYSIS')
    start_time = time.time()
    
    # Validate algorithm trajectories and comparison parameters
    if not algorithm_trajectories:
        raise ValueError("No algorithm trajectories provided for comparison")
    
    if len(algorithm_trajectories) < 2:
        raise ValueError("At least two algorithms required for comparison")
    
    # Validate trajectory data for each algorithm
    for algorithm_name, trajectories in algorithm_trajectories.items():
        if not trajectories:
            raise ValueError(f"No trajectories provided for algorithm: {algorithm_name}")
        
        for i, trajectory in enumerate(trajectories):
            if not isinstance(trajectory, np.ndarray):
                raise TypeError(f"Trajectory {i} for algorithm {algorithm_name} must be numpy array")
    
    # Initialize comparison configuration with defaults
    if comparison_config is None:
        comparison_config = {
            'feature_extraction_config': {},
            'similarity_parameters': {},
            'statistical_significance_level': 0.05,
            'min_trajectories_per_algorithm': 3,
            'normalize_features': True
        }
    
    algorithm_names = list(algorithm_trajectories.keys())
    comparison_results = {
        'algorithm_names': algorithm_names,
        'comparison_metrics': comparison_metrics,
        'algorithm_features': {},
        'similarity_matrices': {},
        'statistical_tests': {},
        'performance_rankings': {},
        'optimization_recommendations': {}
    }
    
    # Extract trajectory features for each algorithm
    logger.debug("Extracting features for cross-algorithm comparison")
    
    for algorithm_name, trajectories in algorithm_trajectories.items():
        algorithm_features = []
        
        for trajectory in trajectories:
            try:
                features = extract_trajectory_features(
                    trajectory,
                    feature_types=TRAJECTORY_FEATURES,
                    include_temporal_features=True,
                    extraction_config=comparison_config.get('feature_extraction_config', {})
                )
                
                # Create feature vector for analysis
                feature_vector = []
                for feature_name in ['path_length', 'directness_index', 'sinuosity', 'velocity_mean', 'acceleration_mean', 'turning_angle_mean']:
                    feature_vector.append(features.get(feature_name, 0.0))
                
                algorithm_features.append(feature_vector)
                
            except Exception as e:
                logger.warning(f"Failed to extract features for trajectory in {algorithm_name}: {e}")
                continue
        
        if algorithm_features:
            comparison_results['algorithm_features'][algorithm_name] = {
                'feature_matrix': np.array(algorithm_features),
                'n_trajectories': len(algorithm_features),
                'feature_means': np.mean(algorithm_features, axis=0).tolist(),
                'feature_stds': np.std(algorithm_features, axis=0).tolist()
            }
        else:
            logger.warning(f"No valid features extracted for algorithm: {algorithm_name}")
    
    # Calculate similarity matrices between algorithm trajectories
    logger.debug("Calculating cross-algorithm similarity matrices")
    
    for metric in comparison_metrics:
        metric_similarities = {}
        
        # Calculate pairwise algorithm similarities
        for i, alg1 in enumerate(algorithm_names):
            for j, alg2 in enumerate(algorithm_names):
                if alg1 not in algorithm_trajectories or alg2 not in algorithm_trajectories:
                    continue
                
                # Calculate similarity between algorithm trajectory sets
                alg1_trajectories = algorithm_trajectories[alg1]
                alg2_trajectories = algorithm_trajectories[alg2]
                
                if i == j:
                    # Intra-algorithm similarity (self-similarity)
                    if len(alg1_trajectories) > 1:
                        similarity_matrix = calculate_trajectory_similarity_matrix(
                            alg1_trajectories,
                            similarity_metrics=[metric],
                            **comparison_config.get('similarity_parameters', {})
                        )
                        # Average similarity within algorithm
                        sim_matrix = similarity_matrix[metric]
                        upper_triangle = np.triu(sim_matrix, k=1)
                        avg_similarity = np.mean(upper_triangle[upper_triangle > 0])
                        metric_similarities[(alg1, alg2)] = avg_similarity
                    else:
                        metric_similarities[(alg1, alg2)] = 1.0
                else:
                    # Inter-algorithm similarity
                    # Sample trajectories for computational efficiency
                    max_trajectories = min(10, len(alg1_trajectories), len(alg2_trajectories))
                    sample_alg1 = alg1_trajectories[:max_trajectories]
                    sample_alg2 = alg2_trajectories[:max_trajectories]
                    
                    # Calculate cross-algorithm similarity
                    combined_trajectories = sample_alg1 + sample_alg2
                    similarity_matrix = calculate_trajectory_similarity_matrix(
                        combined_trajectories,
                        similarity_metrics=[metric],
                        **comparison_config.get('similarity_parameters', {})
                    )
                    
                    # Extract inter-algorithm similarities
                    sim_matrix = similarity_matrix[metric]
                    n_alg1 = len(sample_alg1)
                    inter_similarities = sim_matrix[:n_alg1, n_alg1:]
                    avg_similarity = np.mean(inter_similarities)
                    metric_similarities[(alg1, alg2)] = avg_similarity
        
        comparison_results['similarity_matrices'][metric] = metric_similarities
    
    # Perform statistical significance testing if include_statistical_tests is enabled
    if include_statistical_tests and STATISTICAL_ANALYSIS_ENABLED:
        logger.debug("Performing statistical significance testing")
        
        significance_level = comparison_config.get('statistical_significance_level', 0.05)
        
        for metric in comparison_metrics:
            metric_tests = {}
            
            # Compare each pair of algorithms
            for i, alg1 in enumerate(algorithm_names):
                for j, alg2 in enumerate(algorithm_names[i+1:], i+1):
                    if alg1 not in comparison_results['algorithm_features'] or alg2 not in comparison_results['algorithm_features']:
                        continue
                    
                    # Get feature vectors for statistical testing
                    features1 = comparison_results['algorithm_features'][alg1]['feature_matrix']
                    features2 = comparison_results['algorithm_features'][alg2]['feature_matrix']
                    
                    if len(features1) >= 3 and len(features2) >= 3:
                        try:
                            # Perform Mann-Whitney U test for non-parametric comparison
                            test_results = []
                            feature_names = ['path_length', 'directness_index', 'sinuosity', 'velocity_mean', 'acceleration_mean', 'turning_angle_mean']
                            
                            for k, feature_name in enumerate(feature_names):
                                if k < features1.shape[1] and k < features2.shape[1]:
                                    statistic, p_value = mannwhitneyu(features1[:, k], features2[:, k], alternative='two-sided')
                                    test_results.append({
                                        'feature': feature_name,
                                        'statistic': float(statistic),
                                        'p_value': float(p_value),
                                        'significant': p_value < significance_level
                                    })
                            
                            # Overall comparison using multivariate test
                            # Simplified approach: use mean feature differences
                            mean_diff = np.mean(features1, axis=0) - np.mean(features2, axis=0)
                            overall_difference = np.linalg.norm(mean_diff)
                            
                            metric_tests[(alg1, alg2)] = {
                                'feature_tests': test_results,
                                'overall_difference': float(overall_difference),
                                'n_significant_features': sum(1 for test in test_results if test['significant']),
                                'algorithms_significantly_different': sum(1 for test in test_results if test['significant']) > len(test_results) / 2
                            }
                            
                        except Exception as e:
                            logger.warning(f"Statistical test failed for {alg1} vs {alg2}: {e}")
                            metric_tests[(alg1, alg2)] = {'error': str(e)}
            
            comparison_results['statistical_tests'][metric] = metric_tests
    
    # Generate algorithm performance rankings if generate_rankings is enabled
    if generate_rankings:
        logger.debug("Generating algorithm performance rankings")
        
        for metric in comparison_metrics:
            if metric not in comparison_results['similarity_matrices']:
                continue
            
            similarities = comparison_results['similarity_matrices'][metric]
            algorithm_scores = {}
            
            # Calculate average performance score for each algorithm
            for algorithm in algorithm_names:
                scores = []
                for (alg1, alg2), similarity in similarities.items():
                    if algorithm == alg1:
                        scores.append(similarity)
                
                if scores:
                    algorithm_scores[algorithm] = {
                        'average_score': np.mean(scores),
                        'score_std': np.std(scores),
                        'max_score': np.max(scores),
                        'min_score': np.min(scores)
                    }
                else:
                    algorithm_scores[algorithm] = {
                        'average_score': 0.0,
                        'score_std': 0.0,
                        'max_score': 0.0,
                        'min_score': 0.0
                    }
            
            # Create ranking based on average scores
            ranked_algorithms = sorted(algorithm_scores.items(), key=lambda x: x[1]['average_score'], reverse=True)
            
            comparison_results['performance_rankings'][metric] = {
                'algorithm_scores': algorithm_scores,
                'ranking_order': [alg for alg, _ in ranked_algorithms],
                'top_performer': ranked_algorithms[0][0] if ranked_algorithms else None,
                'performance_spread': max(scores['average_score'] for scores in algorithm_scores.values()) - min(scores['average_score'] for scores in algorithm_scores.values()) if algorithm_scores else 0.0
            }
    
    # Analyze trajectory efficiency and optimization potential
    logger.debug("Analyzing algorithm efficiency and optimization potential")
    
    efficiency_analysis = {}
    for algorithm_name, trajectories in algorithm_trajectories.items():
        algorithm_efficiency = []
        
        for trajectory in trajectories[:10]:  # Limit for computational efficiency
            try:
                efficiency_results = analyze_trajectory_efficiency(
                    trajectory,
                    include_coverage_analysis=True
                )
                algorithm_efficiency.append(efficiency_results)
            except Exception as e:
                logger.warning(f"Efficiency analysis failed for trajectory in {algorithm_name}: {e}")
                continue
        
        if algorithm_efficiency:
            # Aggregate efficiency metrics
            efficiency_metrics = {}
            for metric_name in ['path_optimality_ratio', 'directness_index', 'exploration_radius', 'time_efficiency']:
                values = [eff.get(metric_name, 0.0) for eff in algorithm_efficiency if metric_name in eff]
                if values:
                    efficiency_metrics[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            efficiency_analysis[algorithm_name] = efficiency_metrics
    
    comparison_results['efficiency_analysis'] = efficiency_analysis
    
    # Calculate effect sizes and practical significance measures
    effect_sizes = {}
    if include_statistical_tests:
        for metric in comparison_metrics:
            if metric in comparison_results['statistical_tests']:
                metric_effect_sizes = {}
                
                for (alg1, alg2), test_results in comparison_results['statistical_tests'][metric].items():
                    if 'overall_difference' in test_results:
                        # Simple effect size calculation
                        if alg1 in comparison_results['algorithm_features'] and alg2 in comparison_results['algorithm_features']:
                            features1 = comparison_results['algorithm_features'][alg1]['feature_matrix']
                            features2 = comparison_results['algorithm_features'][alg2]['feature_matrix']
                            
                            # Cohen's d approximation
                            mean1 = np.mean(features1, axis=0)
                            mean2 = np.mean(features2, axis=0)
                            pooled_std = np.sqrt((np.var(features1, axis=0) + np.var(features2, axis=0)) / 2)
                            
                            # Avoid division by zero
                            pooled_std[pooled_std == 0] = 1.0
                            
                            effect_size = np.mean(np.abs(mean1 - mean2) / pooled_std)
                            metric_effect_sizes[(alg1, alg2)] = float(effect_size)
                
                effect_sizes[metric] = metric_effect_sizes
    
    comparison_results['effect_sizes'] = effect_sizes
    
    # Generate cross-algorithm comparison report with recommendations
    optimization_recommendations = {}
    for algorithm_name in algorithm_names:
        recommendations = []
        
        # Check performance rankings
        for metric in comparison_metrics:
            if metric in comparison_results['performance_rankings']:
                ranking = comparison_results['performance_rankings'][metric]['ranking_order']
                if algorithm_name in ranking:
                    position = ranking.index(algorithm_name) + 1
                    if position > len(ranking) / 2:
                        recommendations.append(f"Improve {metric} performance (currently ranked {position}/{len(ranking)})")
        
        # Check efficiency metrics
        if algorithm_name in efficiency_analysis:
            for metric_name, metric_data in efficiency_analysis[algorithm_name].items():
                if metric_data['mean'] < 0.5:  # Arbitrary threshold
                    recommendations.append(f"Optimize {metric_name} (current mean: {metric_data['mean']:.3f})")
        
        if not recommendations:
            recommendations.append("Performance is satisfactory across measured metrics")
        
        optimization_recommendations[algorithm_name] = recommendations
    
    comparison_results['optimization_recommendations'] = optimization_recommendations
    
    # Validate comparison results against scientific standards
    validation_results = {
        'algorithms_compared': len(algorithm_names),
        'total_trajectories': sum(len(trajs) for trajs in algorithm_trajectories.values()),
        'metrics_computed': len(comparison_metrics),
        'statistical_tests_performed': include_statistical_tests,
        'rankings_generated': generate_rankings
    }
    
    comparison_results['validation_results'] = validation_results
    
    # Create comprehensive cross-algorithm analysis report
    analysis_time = time.time() - start_time
    comparison_results['analysis_metadata'] = {
        'analysis_time': analysis_time,
        'algorithms_analyzed': algorithm_names,
        'comparison_metrics': comparison_metrics,
        'statistical_analysis': include_statistical_tests,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Log comparison performance metrics and analysis
    log_performance_metrics(
        metric_name='cross_algorithm_comparison_time',
        metric_value=analysis_time,
        metric_unit='seconds',
        component='TRAJECTORY_ANALYSIS',
        metric_context={
            'n_algorithms': len(algorithm_names),
            'total_trajectories': validation_results['total_trajectories'],
            'metrics_count': len(comparison_metrics)
        }
    )
    
    logger.info(f"Completed cross-algorithm trajectory comparison for {len(algorithm_names)} algorithms in {analysis_time:.3f}s")
    
    # Return detailed algorithm trajectory evaluation
    return comparison_results


def validate_trajectory_quality(
    trajectory: np.ndarray,
    quality_thresholds: Dict[str, float] = None,
    strict_validation: bool = False,
    detect_anomalies: bool = True,
    validation_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Validate trajectory data quality including spatial accuracy, temporal consistency, completeness, 
    and scientific validity with comprehensive quality assessment and error detection.
    
    This function performs comprehensive trajectory quality validation with anomaly detection and 
    quality improvement recommendations for scientific data integrity.
    
    Args:
        trajectory: Input trajectory for quality validation
        quality_thresholds: Dictionary of quality threshold values
        strict_validation: Whether to apply strict validation criteria
        detect_anomalies: Whether to detect anomalies and outliers
        validation_context: Additional context for validation process
        
    Returns:
        Dict[str, Any]: Trajectory quality validation results with quality scores, anomaly detection, and improvement recommendations
    """
    # Initialize logger and validation results
    logger = get_logger('trajectory.quality', 'VALIDATION')
    start_time = time.time()
    
    # Initialize quality thresholds with defaults
    if quality_thresholds is None:
        quality_thresholds = {
            'min_length': MINIMUM_TRAJECTORY_LENGTH,
            'max_length': MAXIMUM_TRAJECTORY_LENGTH,
            'spatial_resolution': SPATIAL_RESOLUTION_THRESHOLD,
            'temporal_resolution': TEMPORAL_RESOLUTION_THRESHOLD,
            'max_velocity': 10.0,
            'max_acceleration': 5.0,
            'outlier_threshold': 3.0
        }
    
    # Initialize validation context with defaults
    if validation_context is None:
        validation_context = {
            'expected_dimensions': 2,
            'coordinate_bounds': None,
            'temporal_sampling_rate': 1.0,
            'measurement_units': 'normalized'
        }
    
    validation_results = {
        'overall_quality_score': 0.0,
        'quality_issues': [],
        'validation_passed': True,
        'anomalies_detected': [],
        'quality_metrics': {},
        'improvement_recommendations': []
    }
    
    # Validate trajectory data structure and format compliance
    logger.debug("Validating trajectory data structure")
    
    structure_issues = []
    
    # Check data type and structure
    if not isinstance(trajectory, np.ndarray):
        structure_issues.append("Trajectory must be numpy array")
        validation_results['validation_passed'] = False
    else:
        # Check dimensions
        if trajectory.ndim != 2:
            structure_issues.append(f"Trajectory must be 2D array, got {trajectory.ndim}D")
            validation_results['validation_passed'] = False
        
        # Check expected dimensions
        expected_dims = validation_context.get('expected_dimensions', 2)
        if trajectory.shape[1] != expected_dims:
            structure_issues.append(f"Expected {expected_dims} spatial dimensions, got {trajectory.shape[1]}")
            if strict_validation:
                validation_results['validation_passed'] = False
        
        # Check trajectory length
        if len(trajectory) < quality_thresholds['min_length']:
            structure_issues.append(f"Trajectory too short: {len(trajectory)} < {quality_thresholds['min_length']}")
            validation_results['validation_passed'] = False
        
        if len(trajectory) > quality_thresholds['max_length']:
            structure_issues.append(f"Trajectory too long: {len(trajectory)} > {quality_thresholds['max_length']}")
            if strict_validation:
                validation_results['validation_passed'] = False
    
    validation_results['quality_issues'].extend(structure_issues)
    
    # Only continue with detailed validation if basic structure is valid
    if isinstance(trajectory, np.ndarray) and trajectory.ndim == 2:
        
        # Check spatial accuracy against threshold requirements
        logger.debug("Checking spatial accuracy and resolution")
        
        spatial_issues = []
        
        # Check for invalid coordinates (NaN, infinity)
        if np.any(~np.isfinite(trajectory)):
            spatial_issues.append("Trajectory contains NaN or infinite coordinates")
            validation_results['validation_passed'] = False
        
        # Check coordinate bounds if specified
        if validation_context.get('coordinate_bounds') is not None:
            bounds = validation_context['coordinate_bounds']
            if len(bounds) >= 4:  # [x_min, x_max, y_min, y_max]
                out_of_bounds = np.any((trajectory[:, 0] < bounds[0]) | (trajectory[:, 0] > bounds[1]) |
                                      (trajectory[:, 1] < bounds[2]) | (trajectory[:, 1] > bounds[3]))
                if out_of_bounds:
                    spatial_issues.append("Trajectory coordinates exceed specified bounds")
                    if strict_validation:
                        validation_results['validation_passed'] = False
        
        # Check spatial resolution
        if len(trajectory) > 1:
            distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            min_distance = np.min(distances[distances > 0]) if np.any(distances > 0) else 0.0
            
            if min_distance < quality_thresholds['spatial_resolution']:
                spatial_issues.append(f"Spatial resolution too fine: {min_distance:.6f} < {quality_thresholds['spatial_resolution']}")
                if strict_validation:
                    validation_results['validation_passed'] = False
            
            validation_results['quality_metrics']['min_spatial_distance'] = min_distance
            validation_results['quality_metrics']['mean_spatial_distance'] = np.mean(distances)
            validation_results['quality_metrics']['spatial_distance_std'] = np.std(distances)
        
        validation_results['quality_issues'].extend(spatial_issues)
        
        # Assess temporal consistency and sampling rate validation
        logger.debug("Assessing temporal consistency")
        
        temporal_issues = []
        
        # Check for temporal resolution consistency
        sampling_rate = validation_context.get('temporal_sampling_rate', 1.0)
        expected_temporal_resolution = 1.0 / sampling_rate
        
        if expected_temporal_resolution < quality_thresholds['temporal_resolution']:
            temporal_issues.append(f"Temporal sampling rate too high: {sampling_rate} Hz")
            if strict_validation:
                validation_results['validation_passed'] = False
        
        # Check for duplicate time points (spatial positions)
        if len(trajectory) > 1:
            duplicate_positions = 0
            for i in range(1, len(trajectory)):
                if np.allclose(trajectory[i], trajectory[i-1], atol=quality_thresholds['spatial_resolution']):
                    duplicate_positions += 1
            
            duplicate_ratio = duplicate_positions / len(trajectory)
            validation_results['quality_metrics']['duplicate_position_ratio'] = duplicate_ratio
            
            if duplicate_ratio > 0.1:  # More than 10% duplicates
                temporal_issues.append(f"High ratio of duplicate positions: {duplicate_ratio:.3f}")
                if strict_validation:
                    validation_results['validation_passed'] = False
        
        validation_results['quality_issues'].extend(temporal_issues)
        
        # Detect anomalies and outliers if detect_anomalies is enabled
        if detect_anomalies:
            logger.debug("Detecting anomalies and outliers in trajectory")
            
            anomalies = []
            
            # Velocity-based anomaly detection
            if len(trajectory) > 1:
                velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
                velocity_mean = np.mean(velocities)
                velocity_std = np.std(velocities)
                
                # Z-score based outlier detection
                if velocity_std > 0:
                    velocity_z_scores = np.abs((velocities - velocity_mean) / velocity_std)
                    velocity_outliers = np.where(velocity_z_scores > quality_thresholds['outlier_threshold'])[0]
                    
                    for outlier_idx in velocity_outliers:
                        anomalies.append({
                            'type': 'velocity_outlier',
                            'index': int(outlier_idx + 1),  # +1 because velocities are between points
                            'value': float(velocities[outlier_idx]),
                            'z_score': float(velocity_z_scores[outlier_idx])
                        })
                
                # Check maximum velocity threshold
                max_velocity = np.max(velocities)
                if max_velocity > quality_thresholds['max_velocity']:
                    anomalies.append({
                        'type': 'excessive_velocity',
                        'max_velocity': float(max_velocity),
                        'threshold': quality_thresholds['max_velocity']
                    })
                
                validation_results['quality_metrics']['max_velocity'] = float(max_velocity)
                validation_results['quality_metrics']['mean_velocity'] = float(velocity_mean)
                validation_results['quality_metrics']['velocity_std'] = float(velocity_std)
            
            # Acceleration-based anomaly detection
            if len(trajectory) > 2:
                velocity_vectors = np.diff(trajectory, axis=0)
                accelerations = np.linalg.norm(np.diff(velocity_vectors, axis=0), axis=1)
                
                max_acceleration = np.max(accelerations)
                if max_acceleration > quality_thresholds['max_acceleration']:
                    anomalies.append({
                        'type': 'excessive_acceleration',
                        'max_acceleration': float(max_acceleration),
                        'threshold': quality_thresholds['max_acceleration']
                    })
                
                validation_results['quality_metrics']['max_acceleration'] = float(max_acceleration)
                validation_results['quality_metrics']['mean_acceleration'] = float(np.mean(accelerations))
            
            # Spatial clustering anomaly detection
            if len(trajectory) > 10:
                # Check for spatial outliers using IQR method
                trajectory_center = np.mean(trajectory, axis=0)
                distances_from_center = np.linalg.norm(trajectory - trajectory_center, axis=1)
                
                q1 = np.percentile(distances_from_center, 25)
                q3 = np.percentile(distances_from_center, 75)
                iqr = q3 - q1
                
                outlier_threshold = q3 + 1.5 * iqr
                spatial_outliers = np.where(distances_from_center > outlier_threshold)[0]
                
                for outlier_idx in spatial_outliers:
                    anomalies.append({
                        'type': 'spatial_outlier',
                        'index': int(outlier_idx),
                        'distance_from_center': float(distances_from_center[outlier_idx]),
                        'threshold': float(outlier_threshold)
                    })
            
            validation_results['anomalies_detected'] = anomalies
            
            # Update validation status based on anomalies
            if anomalies and strict_validation:
                critical_anomalies = [a for a in anomalies if a['type'] in ['excessive_velocity', 'excessive_acceleration']]
                if critical_anomalies:
                    validation_results['validation_passed'] = False
        
        # Apply strict validation criteria if strict_validation is enabled
        if strict_validation:
            strict_issues = []
            
            # Additional strict checks
            if len(trajectory) < quality_thresholds['min_length'] * 2:
                strict_issues.append("Trajectory length below strict minimum for robust analysis")
            
            if validation_results['quality_metrics'].get('duplicate_position_ratio', 0.0) > 0.05:
                strict_issues.append("Excessive stationary periods detected")
            
            if len(validation_results['anomalies_detected']) > len(trajectory) * 0.05:
                strict_issues.append("High anomaly rate detected")
            
            validation_results['quality_issues'].extend(strict_issues)
            
            if strict_issues:
                validation_results['validation_passed'] = False
    
    # Calculate trajectory completeness and data integrity scores
    logger.debug("Calculating quality scores")
    
    quality_score = 1.0
    
    # Penalize for structure issues
    if structure_issues:
        quality_score -= 0.3
    
    # Penalize for spatial/temporal issues
    spatial_temporal_issues = len([issue for issue in validation_results['quality_issues'] 
                                  if 'spatial' in issue.lower() or 'temporal' in issue.lower()])
    quality_score -= spatial_temporal_issues * 0.1
    
    # Penalize for anomalies
    anomaly_penalty = min(0.3, len(validation_results['anomalies_detected']) * 0.02)
    quality_score -= anomaly_penalty
    
    # Ensure score is between 0 and 1
    validation_results['overall_quality_score'] = max(0.0, min(1.0, quality_score))
    
    # Generate quality improvement recommendations
    recommendations = []
    
    if structure_issues:
        recommendations.append("Fix data structure and format issues")
    
    if validation_results['quality_metrics'].get('duplicate_position_ratio', 0.0) > 0.1:
        recommendations.append("Remove or interpolate excessive stationary points")
    
    if len(validation_results['anomalies_detected']) > 0:
        recommendations.append("Investigate and handle detected anomalies")
    
    if validation_results['overall_quality_score'] < 0.7:
        recommendations.append("Consider data preprocessing and cleaning")
    
    if not recommendations:
        recommendations.append("Trajectory quality is satisfactory")
    
    validation_results['improvement_recommendations'] = recommendations
    
    # Add validation metadata
    validation_time = time.time() - start_time
    validation_results['validation_metadata'] = {
        'validation_time': validation_time,
        'trajectory_length': len(trajectory) if isinstance(trajectory, np.ndarray) else 0,
        'strict_validation': strict_validation,
        'anomaly_detection': detect_anomalies,
        'thresholds_used': quality_thresholds,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Log validation results for audit trail
    logger.info(f"Trajectory quality validation completed in {validation_time:.3f}s: "
               f"Score={validation_results['overall_quality_score']:.3f}, "
               f"Passed={validation_results['validation_passed']}")
    
    if not validation_results['validation_passed']:
        logger.warning(f"Trajectory validation failed with {len(validation_results['quality_issues'])} issues")
    
    # Return detailed quality assessment with recommendations
    return validation_results


def generate_trajectory_report(
    analysis_results: Dict[str, Any],
    report_type: str = 'comprehensive',
    include_visualizations: bool = False,
    output_format: str = 'json',
    report_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive trajectory analysis report including similarity analysis, pattern classification, 
    efficiency assessment, and statistical validation for scientific documentation and research publication.
    
    This function creates detailed reports with statistical summaries, visualizations, and scientific 
    documentation for comprehensive trajectory analysis results.
    
    Args:
        analysis_results: Dictionary containing trajectory analysis results
        report_type: Type of report to generate ('comprehensive', 'summary', 'statistical')
        include_visualizations: Whether to include visualization specifications
        output_format: Format for report output ('json', 'html', 'markdown')
        report_config: Optional configuration for report generation
        
    Returns:
        Dict[str, Any]: Comprehensive trajectory analysis report with statistical summaries, visualizations, and scientific documentation
    """
    # Initialize logger and report generation
    logger = get_logger('trajectory.report', 'ANALYSIS')
    start_time = time.time()
    
    # Initialize report configuration with defaults
    if report_config is None:
        report_config = {
            'include_metadata': True,
            'include_raw_data': False,
            'statistical_precision': 4,
            'visualization_config': {},
            'export_settings': {}
        }
    
    # Initialize report structure
    report = {
        'report_metadata': {
            'report_id': f"trajectory_report_{int(time.time())}",
            'generation_time': datetime.datetime.now().isoformat(),
            'report_type': report_type,
            'output_format': output_format,
            'include_visualizations': include_visualizations,
            'trajectory_analysis_version': TRAJECTORY_ANALYSIS_VERSION
        },
        'executive_summary': {},
        'detailed_analysis': {},
        'statistical_validation': {},
        'recommendations': {},
        'appendices': {}
    }
    
    # Aggregate trajectory analysis results and generate summaries
    logger.debug("Aggregating analysis results for report generation")
    
    summary_statistics = {}
    
    # Process similarity analysis results
    if 'similarity_matrices' in analysis_results:
        similarity_data = analysis_results['similarity_matrices']
        similarity_summary = {}
        
        for metric, matrix in similarity_data.items():
            if isinstance(matrix, np.ndarray):
                # Calculate summary statistics for similarity matrix
                upper_triangle = np.triu(matrix, k=1)
                similarities = upper_triangle[upper_triangle > 0]
                
                if len(similarities) > 0:
                    similarity_summary[metric] = {
                        'mean_similarity': round(float(np.mean(similarities)), report_config['statistical_precision']),
                        'std_similarity': round(float(np.std(similarities)), report_config['statistical_precision']),
                        'min_similarity': round(float(np.min(similarities)), report_config['statistical_precision']),
                        'max_similarity': round(float(np.max(similarities)), report_config['statistical_precision']),
                        'median_similarity': round(float(np.median(similarities)), report_config['statistical_precision'])
                    }
        
        summary_statistics['similarity_analysis'] = similarity_summary
    
    # Process pattern classification results
    if 'pattern_classifications' in analysis_results:
        classification_data = analysis_results['pattern_classifications']
        
        if 'classification_summary' in classification_data:
            summary_statistics['pattern_classification'] = {
                'total_trajectories': classification_data['classification_summary'].get('total_trajectories', 0),
                'classified_trajectories': classification_data['classification_summary'].get('classified_trajectories', 0),
                'average_confidence': round(classification_data['classification_summary'].get('average_confidence', 0.0), report_config['statistical_precision']),
                'pattern_distribution': classification_data['classification_summary'].get('pattern_distribution', {})
            }
    
    # Process efficiency analysis results
    if 'efficiency_analysis' in analysis_results:
        efficiency_data = analysis_results['efficiency_analysis']
        efficiency_summary = {}
        
        # Aggregate efficiency metrics across algorithms or trajectories
        if isinstance(efficiency_data, dict):
            for key, metrics in efficiency_data.items():
                if isinstance(metrics, dict):
                    algorithm_efficiency = {}
                    for metric_name, metric_data in metrics.items():
                        if isinstance(metric_data, dict) and 'mean' in metric_data:
                            algorithm_efficiency[metric_name] = {
                                'mean': round(metric_data['mean'], report_config['statistical_precision']),
                                'std': round(metric_data.get('std', 0.0), report_config['statistical_precision'])
                            }
                        elif isinstance(metric_data, (int, float)):
                            algorithm_efficiency[metric_name] = round(metric_data, report_config['statistical_precision'])
                    
                    efficiency_summary[key] = algorithm_efficiency
        
        summary_statistics['efficiency_analysis'] = efficiency_summary
    
    # Process cross-algorithm comparison results
    if 'algorithm_comparison' in analysis_results:
        comparison_data = analysis_results['algorithm_comparison']
        comparison_summary = {}
        
        if 'performance_rankings' in comparison_data:
            comparison_summary['performance_rankings'] = comparison_data['performance_rankings']
        
        if 'statistical_tests' in comparison_data:
            # Summarize statistical significance results
            significant_differences = 0
            total_comparisons = 0
            
            for metric, tests in comparison_data['statistical_tests'].items():
                for comparison, test_result in tests.items():
                    if isinstance(test_result, dict) and 'algorithms_significantly_different' in test_result:
                        total_comparisons += 1
                        if test_result['algorithms_significantly_different']:
                            significant_differences += 1
            
            if total_comparisons > 0:
                comparison_summary['statistical_significance'] = {
                    'significant_differences': significant_differences,
                    'total_comparisons': total_comparisons,
                    'significance_ratio': round(significant_differences / total_comparisons, report_config['statistical_precision'])
                }
        
        summary_statistics['algorithm_comparison'] = comparison_summary
    
    # Generate executive summary based on report type
    if report_type in ['comprehensive', 'summary']:
        executive_summary = {
            'analysis_overview': "Comprehensive trajectory analysis performed using advanced similarity metrics, pattern classification, and efficiency assessment.",
            'key_findings': [],
            'performance_highlights': [],
            'recommendations_summary': []
        }
        
        # Extract key findings from analysis results
        if 'similarity_analysis' in summary_statistics:
            sim_data = summary_statistics['similarity_analysis']
            if sim_data:
                avg_similarities = [metrics.get('mean_similarity', 0.0) for metrics in sim_data.values()]
                if avg_similarities:
                    executive_summary['key_findings'].append(
                        f"Average trajectory similarity across metrics: {np.mean(avg_similarities):.3f}"
                    )
        
        if 'pattern_classification' in summary_statistics:
            pattern_data = summary_statistics['pattern_classification']
            classified_ratio = 0.0
            if pattern_data.get('total_trajectories', 0) > 0:
                classified_ratio = pattern_data.get('classified_trajectories', 0) / pattern_data['total_trajectories']
            
            executive_summary['key_findings'].append(
                f"Successfully classified {classified_ratio:.1%} of trajectories with average confidence {pattern_data.get('average_confidence', 0.0):.3f}"
            )
        
        if 'algorithm_comparison' in summary_statistics:
            comp_data = summary_statistics['algorithm_comparison']
            if 'statistical_significance' in comp_data:
                sig_ratio = comp_data['statistical_significance'].get('significance_ratio', 0.0)
                executive_summary['key_findings'].append(
                    f"Statistically significant differences found in {sig_ratio:.1%} of algorithm comparisons"
                )
        
        report['executive_summary'] = executive_summary
    
    # Include statistical validation and significance testing results
    if report_type in ['comprehensive', 'statistical'] and STATISTICAL_ANALYSIS_ENABLED:
        statistical_validation = {}
        
        # Compile validation results from different analysis components
        validation_sources = [
            'similarity_validation',
            'classification_validation', 
            'efficiency_validation',
            'quality_validation'
        ]
        
        for source in validation_sources:
            if source in analysis_results:
                statistical_validation[source] = analysis_results[source]
        
        # Calculate overall validation scores
        validation_scores = []
        for source, validation_data in statistical_validation.items():
            if isinstance(validation_data, dict):
                if 'overall_quality_score' in validation_data:
                    validation_scores.append(validation_data['overall_quality_score'])
                elif 'validation_passed' in validation_data:
                    validation_scores.append(1.0 if validation_data['validation_passed'] else 0.0)
        
        if validation_scores:
            statistical_validation['overall_validation_score'] = round(np.mean(validation_scores), report_config['statistical_precision'])
        
        report['statistical_validation'] = statistical_validation
    
    # Add trajectory visualization specifications if include_visualizations is enabled
    if include_visualizations:
        visualization_specs = {
            'similarity_heatmaps': {
                'description': "Heatmap visualizations of trajectory similarity matrices",
                'recommended_library': "matplotlib/seaborn",
                'data_requirements': "Similarity matrices from analysis results"
            },
            'pattern_distribution_charts': {
                'description': "Distribution charts showing movement pattern classifications",
                'recommended_library': "plotly/matplotlib",
                'data_requirements': "Pattern classification results with confidence scores"
            },
            'efficiency_comparison_plots': {
                'description': "Comparative plots of trajectory efficiency metrics",
                'recommended_library': "matplotlib/plotly",
                'data_requirements': "Efficiency analysis results across algorithms"
            },
            'trajectory_overlays': {
                'description': "Spatial overlay plots of trajectory paths",
                'recommended_library': "matplotlib/plotly",
                'data_requirements': "Raw trajectory coordinate data"
            }
        }
        
        # Add visualization configuration if provided
        if report_config.get('visualization_config'):
            for viz_type, config in report_config['visualization_config'].items():
                if viz_type in visualization_specs:
                    visualization_specs[viz_type].update(config)
        
        report['visualization_specifications'] = visualization_specs
    
    # Generate cross-algorithm comparison and ranking analysis
    if report_type == 'comprehensive' and 'algorithm_comparison' in analysis_results:
        comparison_analysis = analysis_results['algorithm_comparison']
        
        detailed_comparison = {
            'methodology': "Cross-algorithm trajectory comparison using multiple similarity metrics and statistical validation",
            'algorithms_analyzed': comparison_analysis.get('algorithm_names', []),
            'comparison_metrics': comparison_analysis.get('comparison_metrics', []),
            'performance_rankings': comparison_analysis.get('performance_rankings', {}),
            'statistical_significance': comparison_analysis.get('statistical_tests', {}),
            'effect_sizes': comparison_analysis.get('effect_sizes', {}),
            'optimization_recommendations': comparison_analysis.get('optimization_recommendations', {})
        }
        
        report['detailed_analysis']['algorithm_comparison'] = detailed_comparison
    
    # Include pattern classification and efficiency assessment details
    if report_type == 'comprehensive':
        if 'pattern_classifications' in analysis_results:
            pattern_analysis = {
                'classification_method': analysis_results['pattern_classifications'].get('classification_summary', {}).get('classification_method', 'unknown'),
                'pattern_types_identified': list(MOVEMENT_PATTERN_TYPES),
                'classification_confidence': analysis_results['pattern_classifications'].get('classification_summary', {}).get('average_confidence', 0.0),
                'pattern_transitions': analysis_results['pattern_classifications'].get('pattern_transition_analysis', {}),
                'validation_metrics': analysis_results['pattern_classifications'].get('validation_results', {})
            }
            report['detailed_analysis']['pattern_classification'] = pattern_analysis
        
        if 'efficiency_analysis' in analysis_results:
            efficiency_analysis = {
                'metrics_calculated': list(analysis_results['efficiency_analysis'].keys()) if isinstance(analysis_results['efficiency_analysis'], dict) else [],
                'optimization_potential': {},
                'benchmark_comparisons': {},
                'improvement_areas': []
            }
            
            # Extract optimization recommendations
            for key, data in analysis_results.get('efficiency_analysis', {}).items():
                if isinstance(data, dict) and 'optimization_recommendations' in data:
                    efficiency_analysis['optimization_potential'][key] = data['optimization_recommendations']
            
            report['detailed_analysis']['efficiency_assessment'] = efficiency_analysis
    
    # Format report according to specified output format
    if output_format == 'json':
        # JSON format is already structured appropriately
        pass
    elif output_format == 'html':
        # Add HTML formatting metadata
        report['html_formatting'] = {
            'css_classes': {
                'header': 'trajectory-report-header',
                'summary': 'trajectory-summary',
                'data_table': 'trajectory-data-table',
                'chart': 'trajectory-chart'
            },
            'template_requirements': "Bootstrap 4+ for responsive layout"
        }
    elif output_format == 'markdown':
        # Add markdown formatting guidelines
        report['markdown_formatting'] = {
            'section_headers': "Use ## for main sections, ### for subsections",
            'tables': "Use GitHub-flavored markdown table format",
            'code_blocks': "Use