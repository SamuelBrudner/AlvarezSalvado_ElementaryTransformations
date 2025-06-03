"""
Core test helper utilities module providing comprehensive testing infrastructure for scientific plume navigation simulation validation.

This module implements standardized test fixtures, mock data generation, numerical accuracy validation, performance measurement, 
cross-format compatibility testing, and scientific computing validation utilities with >95% correlation requirements and 
<7.2 seconds per simulation performance targets. Supports automated test environment setup, batch processing validation, 
and reproducible scientific testing workflows.

Key Features:
- Comprehensive test fixture management with cross-platform path handling
- Mock video data generation supporting Crimaldi and custom plume formats
- Numerical accuracy validation with scientific precision thresholds
- Performance profiling and measurement utilities for simulation validation
- Cross-format compatibility testing and validation frameworks
- Batch processing validation for 4000+ simulation requirements
- Automated test environment setup with temporary directory management
- Scientific computing validation with statistical analysis capabilities
"""

# External library imports with version specifications
import numpy as np  # numpy 2.1.3+ - Numerical array operations and scientific computing for test data validation
import pytest  # pytest 8.3.5+ - Testing framework integration and fixture management
import pathlib  # Python 3.9+ - Cross-platform path handling for test fixtures and temporary files
import tempfile  # Python 3.9+ - Temporary file and directory management for test isolation
import time  # Python 3.9+ - Performance timing and measurement for test validation
import functools  # Python 3.9+ - Decorator utilities for performance measurement and test helpers
import contextlib  # Python 3.9+ - Context manager utilities for test environment setup
import warnings  # Python 3.9+ - Warning management for test validation and error reporting
import json  # Python 3.9+ - JSON configuration file handling for test scenarios
import cv2  # opencv-python 4.11.0+ - Video processing for mock data generation and format validation
from typing import Dict, Any, List, Optional, Union, Tuple, Callable  # Python 3.9+ - Type hints for test utility functions
import datetime  # Python 3.9+ - Timestamp handling for test metadata and audit trails
import random  # Python 3.9+ - Random number generation for test data creation
import math  # Python 3.9+ - Mathematical operations for scientific computing validation
import threading  # Python 3.9+ - Thread-safe operations for performance profiling
import uuid  # Python 3.9+ - Unique identifier generation for test correlation
import os  # Python 3.9+ - Operating system interface for environment and file operations
import sys  # Python 3.9+ - System interface for platform detection and output management
import shutil  # Python 3.9+ - High-level file operations for test cleanup
import hashlib  # Python 3.9+ - Cryptographic hash functions for data integrity verification

# Internal imports from backend utilities
from backend.utils.file_utils import (
    ensure_directory_exists,
    validate_file_exists, 
    load_json_config,
    create_temporary_directory
)
from backend.utils.validation_utils import (
    validate_numerical_accuracy,
    validate_cross_format_compatibility,
    ValidationResult
)
from backend.utils.scientific_constants import (
    NUMERICAL_PRECISION_THRESHOLD,
    DEFAULT_CORRELATION_THRESHOLD,
    PhysicalConstants
)

# Global configuration constants for test fixture paths and validation settings
TEST_FIXTURES_BASE_PATH = pathlib.Path(__file__).parent.parent / 'test_fixtures'
REFERENCE_RESULTS_PATH = TEST_FIXTURES_BASE_PATH / 'reference_results'
CONFIG_FIXTURES_PATH = TEST_FIXTURES_BASE_PATH / 'config'
DEFAULT_TOLERANCE = 1e-6
CORRELATION_THRESHOLD = 0.95
PERFORMANCE_TIMEOUT_SECONDS = 7.2
BATCH_TARGET_SIMULATIONS = 4000
REPRODUCIBILITY_THRESHOLD = 0.99

# Test data generation constants for mock video creation
DEFAULT_VIDEO_DIMENSIONS = (640, 480)
DEFAULT_FRAME_COUNT = 100
DEFAULT_FRAME_RATE = 30.0
CRIMALDI_FORMAT_SPECS = {
    'bit_depth': 8,
    'color_space': 'grayscale',
    'pixel_to_meter_ratio': 100.0,
    'temporal_resolution': 50.0
}
CUSTOM_FORMAT_SPECS = {
    'bit_depth': 16,
    'color_space': 'rgb',
    'pixel_to_meter_ratio': 150.0,
    'temporal_resolution': 30.0
}

# Performance testing constants and thresholds
MEMORY_LIMIT_MB = 8192  # 8GB memory limit for performance validation
CPU_USAGE_THRESHOLD = 85.0  # Maximum CPU usage percentage
PROCESSING_RATE_THRESHOLD = 500.0  # Minimum simulations per hour
ERROR_RATE_THRESHOLD = 0.01  # Maximum 1% error rate

# Validation cache for improved test performance
_validation_cache: Dict[str, Any] = {}
_test_environment_registry: Dict[str, Any] = {}
_performance_metrics_buffer: List[Dict[str, Any]] = []


def create_test_fixture_path(
    fixture_name: str,
    fixture_category: str
) -> pathlib.Path:
    """
    Create standardized path to test fixture files with validation and cross-platform compatibility.
    
    This function provides centralized fixture path management with automatic directory creation
    and cross-platform path handling for consistent test data access.
    
    Args:
        fixture_name: Name of the test fixture file or directory
        fixture_category: Category of fixture (video, config, reference, mock)
        
    Returns:
        pathlib.Path: Validated path to test fixture file with cross-platform compatibility
    """
    # Validate fixture name and category parameters
    if not fixture_name or not isinstance(fixture_name, str):
        raise ValueError("Fixture name must be a non-empty string")
    
    if not fixture_category or not isinstance(fixture_category, str):
        raise ValueError("Fixture category must be a non-empty string")
    
    # Construct path using TEST_FIXTURES_BASE_PATH and category
    category_path = TEST_FIXTURES_BASE_PATH / fixture_category
    fixture_path = category_path / fixture_name
    
    # Validate that fixture file exists and is accessible
    if not fixture_path.exists():
        # Create category directory if it doesn't exist
        ensure_directory_exists(str(category_path))
        
        # Log fixture path creation for debugging
        warnings.warn(
            f"Test fixture not found, creating path: {fixture_path}",
            UserWarning,
            stacklevel=2
        )
    
    # Return pathlib.Path object for cross-platform compatibility
    return fixture_path.resolve()


def load_test_config(
    config_name: str,
    validate_schema: bool = True
) -> dict:
    """
    Load and validate test configuration files with schema validation and parameter checking.
    
    This function provides robust configuration loading with comprehensive validation
    for test scenario setup and parameter management.
    
    Args:
        config_name: Name of the configuration file (without extension)
        validate_schema: Enable schema validation for configuration structure
        
    Returns:
        dict: Loaded and validated configuration dictionary with test parameters
    """
    # Construct path to configuration file in CONFIG_FIXTURES_PATH
    config_path = CONFIG_FIXTURES_PATH / f"{config_name}.json"
    
    # Validate configuration file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Test configuration not found: {config_path}")
    
    # Load JSON configuration using load_json_config utility
    try:
        config_data = load_json_config(
            config_path=str(config_path),
            validate_schema=validate_schema
        )
    except Exception as e:
        raise ValueError(f"Failed to load test configuration {config_name}: {e}")
    
    # Validate configuration schema if validate_schema is True
    if validate_schema:
        required_fields = ['test_type', 'parameters']
        for field in required_fields:
            if field not in config_data:
                raise ValueError(f"Missing required configuration field: {field}")
    
    # Check for required test configuration fields
    if 'test_type' in config_data:
        valid_test_types = ['unit', 'integration', 'performance', 'validation']
        if config_data['test_type'] not in valid_test_types:
            raise ValueError(f"Invalid test type: {config_data['test_type']}")
    
    # Validate physical parameters using PhysicalConstants
    if 'physical_parameters' in config_data:
        physical_constants = PhysicalConstants()
        for param_name, param_value in config_data['physical_parameters'].items():
            if not physical_constants.validate_unit(param_name):
                warnings.warn(f"Unknown physical parameter: {param_name}")
    
    # Return validated configuration dictionary
    return config_data


def assert_arrays_almost_equal(
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
    error_message: str = "Arrays are not equal within tolerance"
) -> None:
    """
    Assert that two NumPy arrays are almost equal within scientific precision tolerance for numerical validation.
    
    This function provides comprehensive array comparison with detailed error reporting
    and scientific precision handling for numerical test validation.
    
    Args:
        actual: Actual array values from test execution
        expected: Expected array values for comparison
        tolerance: Numerical tolerance for floating-point comparison
        error_message: Custom error message for assertion failures
        
    Raises:
        AssertionError: If arrays are not equal within tolerance with detailed error information
    """
    # Validate that both inputs are NumPy arrays
    if not isinstance(actual, np.ndarray):
        raise TypeError(f"Actual value must be numpy array, got {type(actual)}")
    
    if not isinstance(expected, np.ndarray):
        raise TypeError(f"Expected value must be numpy array, got {type(expected)}")
    
    # Check array shapes are compatible
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Array shapes do not match: actual {actual.shape} vs expected {expected.shape}"
        )
    
    # Check for NaN or infinite values
    if np.any(np.isnan(actual)) or np.any(np.isinf(actual)):
        raise AssertionError("Actual array contains NaN or infinite values")
    
    if np.any(np.isnan(expected)) or np.any(np.isinf(expected)):
        raise AssertionError("Expected array contains NaN or infinite values")
    
    # Use numpy.allclose with specified tolerance for comparison
    if not np.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        # Calculate relative and absolute differences
        abs_diff = np.abs(actual - expected)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        
        # Calculate relative differences where expected is non-zero
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.abs((actual - expected) / expected)
            rel_diff = rel_diff[np.isfinite(rel_diff)]
            max_rel_diff = np.max(rel_diff) if len(rel_diff) > 0 else 0.0
        
        # Include array statistics in error message
        detailed_message = (
            f"{error_message}\n"
            f"Tolerance: {tolerance}\n"
            f"Maximum absolute difference: {max_abs_diff}\n"
            f"Mean absolute difference: {mean_abs_diff}\n"
            f"Maximum relative difference: {max_rel_diff}\n"
            f"Array shape: {actual.shape}\n"
            f"Array dtype: {actual.dtype}"
        )
        
        raise AssertionError(detailed_message)


def assert_simulation_accuracy(
    simulation_results: np.ndarray,
    reference_results: np.ndarray,
    correlation_threshold: float = CORRELATION_THRESHOLD
) -> None:
    """
    Assert that simulation results meet accuracy requirements with >95% correlation against reference implementations.
    
    This function provides comprehensive simulation accuracy validation with statistical
    analysis and detailed error reporting for scientific computing validation.
    
    Args:
        simulation_results: Results from simulation execution
        reference_results: Reference implementation results for comparison
        correlation_threshold: Minimum correlation coefficient required (default 0.95)
        
    Raises:
        AssertionError: If accuracy requirements not met with detailed accuracy analysis
    """
    # Validate input arrays and check for NaN or infinite values
    if not isinstance(simulation_results, np.ndarray):
        raise TypeError("Simulation results must be numpy array")
    
    if not isinstance(reference_results, np.ndarray):
        raise TypeError("Reference results must be numpy array")
    
    if simulation_results.shape != reference_results.shape:
        raise AssertionError(
            f"Result shapes do not match: simulation {simulation_results.shape} "
            f"vs reference {reference_results.shape}"
        )
    
    # Check for invalid values
    if np.any(np.isnan(simulation_results)) or np.any(np.isinf(simulation_results)):
        raise AssertionError("Simulation results contain NaN or infinite values")
    
    if np.any(np.isnan(reference_results)) or np.any(np.isinf(reference_results)):
        raise AssertionError("Reference results contain NaN or infinite values")
    
    # Calculate correlation coefficient using numpy.corrcoef
    try:
        correlation_matrix = np.corrcoef(simulation_results.flatten(), reference_results.flatten())
        correlation_coefficient = correlation_matrix[0, 1]
        
        # Handle case where correlation cannot be calculated
        if np.isnan(correlation_coefficient):
            raise AssertionError("Cannot calculate correlation coefficient (insufficient variance)")
        
    except Exception as e:
        raise AssertionError(f"Failed to calculate correlation coefficient: {e}")
    
    # Perform statistical significance testing
    from scipy import stats
    statistic, p_value = stats.pearsonr(simulation_results.flatten(), reference_results.flatten())
    
    # Check correlation against threshold (default 0.95)
    if correlation_coefficient < correlation_threshold:
        # Calculate additional accuracy metrics (RMSE, MAE)
        rmse = np.sqrt(np.mean((simulation_results - reference_results) ** 2))
        mae = np.mean(np.abs(simulation_results - reference_results))
        
        # Calculate relative error
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_error = np.mean(np.abs((simulation_results - reference_results) / reference_results))
            relative_error = relative_error if np.isfinite(relative_error) else float('inf')
        
        # Raise AssertionError with detailed accuracy report if failed
        accuracy_report = (
            f"Simulation accuracy requirements not met:\n"
            f"Correlation coefficient: {correlation_coefficient:.6f} (threshold: {correlation_threshold})\n"
            f"P-value: {p_value:.6f}\n"
            f"RMSE: {rmse:.6f}\n"
            f"MAE: {mae:.6f}\n"
            f"Relative error: {relative_error:.6f}\n"
            f"Result array shape: {simulation_results.shape}\n"
            f"Recommendation: Review algorithm parameters or reference implementation"
        )
        
        raise AssertionError(accuracy_report)


def measure_performance(
    time_limit_seconds: float = PERFORMANCE_TIMEOUT_SECONDS,
    memory_limit_mb: int = MEMORY_LIMIT_MB
) -> Callable:
    """
    Decorator function to measure and validate performance of test functions against time and memory thresholds.
    
    This decorator provides comprehensive performance monitoring with time and memory
    validation for ensuring performance requirements are met.
    
    Args:
        time_limit_seconds: Maximum execution time allowed (default 7.2 seconds)
        memory_limit_mb: Maximum memory usage allowed in MB
        
    Returns:
        callable: Decorated function with performance monitoring and validation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import psutil
            import gc
            
            # Record start time and memory usage before function execution
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
            
            # Force garbage collection for accurate memory measurement
            gc.collect()
            
            try:
                # Execute wrapped function and capture results
                result = func(*args, **kwargs)
                
                # Record end time and peak memory usage
                end_time = time.time()
                end_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
                peak_memory = max(start_memory, end_memory)
                
                # Calculate execution time and memory consumption
                execution_time = end_time - start_time
                memory_increase = end_memory - start_memory
                
                # Validate against specified limits
                performance_violations = []
                
                if execution_time > time_limit_seconds:
                    performance_violations.append(
                        f"Execution time {execution_time:.3f}s exceeds limit {time_limit_seconds}s"
                    )
                
                if peak_memory > memory_limit_mb:
                    performance_violations.append(
                        f"Memory usage {peak_memory:.1f}MB exceeds limit {memory_limit_mb}MB"
                    )
                
                # Log performance metrics and raise warnings if limits exceeded
                performance_data = {
                    'function_name': func.__name__,
                    'execution_time': execution_time,
                    'memory_start_mb': start_memory,
                    'memory_end_mb': end_memory,
                    'memory_peak_mb': peak_memory,
                    'memory_increase_mb': memory_increase,
                    'time_limit_seconds': time_limit_seconds,
                    'memory_limit_mb': memory_limit_mb,
                    'violations': performance_violations,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                # Store performance metrics in global buffer
                global _performance_metrics_buffer
                _performance_metrics_buffer.append(performance_data)
                
                # Raise warnings for performance violations
                if performance_violations:
                    warnings.warn(
                        f"Performance limits exceeded in {func.__name__}: " + 
                        "; ".join(performance_violations),
                        UserWarning,
                        stacklevel=2
                    )
                
                # Return function results with performance metadata
                if hasattr(result, '__dict__'):
                    result.__dict__['_performance_data'] = performance_data
                
                return result
                
            except Exception as e:
                # Record performance data even for failed executions
                end_time = time.time()
                execution_time = end_time - start_time
                
                performance_data = {
                    'function_name': func.__name__,
                    'execution_time': execution_time,
                    'error': str(e),
                    'failed': True,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                _performance_metrics_buffer.append(performance_data)
                raise
        
        # Add performance metadata to wrapper function
        wrapper._performance_monitored = True
        wrapper._time_limit = time_limit_seconds
        wrapper._memory_limit = memory_limit_mb
        
        return wrapper
    
    return decorator


def create_mock_video_data(
    dimensions: Tuple[int, int] = DEFAULT_VIDEO_DIMENSIONS,
    frame_count: int = DEFAULT_FRAME_COUNT,
    frame_rate: float = DEFAULT_FRAME_RATE,
    format_type: str = 'custom'
) -> np.ndarray:
    """
    Generate synthetic video data for testing with configurable parameters matching Crimaldi and custom formats.
    
    This function creates realistic synthetic plume video data with configurable parameters
    for comprehensive testing without requiring large test data files.
    
    Args:
        dimensions: Video dimensions as (width, height) tuple
        frame_count: Number of frames to generate
        frame_rate: Frame rate for temporal characteristics
        format_type: Video format type ('crimaldi', 'custom', 'generic')
        
    Returns:
        np.ndarray: Generated video data array with proper dtype and shape
    """
    # Validate input parameters for dimensions, frame count, and format
    width, height = dimensions
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions: {dimensions}. Must be positive integers.")
    
    if frame_count <= 0:
        raise ValueError(f"Invalid frame count: {frame_count}. Must be positive integer.")
    
    if frame_rate <= 0:
        raise ValueError(f"Invalid frame rate: {frame_rate}. Must be positive number.")
    
    valid_formats = ['crimaldi', 'custom', 'generic']
    if format_type not in valid_formats:
        raise ValueError(f"Invalid format type: {format_type}. Must be one of {valid_formats}")
    
    # Generate synthetic plume field using diffusion-advection physics
    np.random.seed(42)  # For reproducible test data
    
    # Create coordinate grids for spatial calculations
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Initialize video data array
    if format_type == 'crimaldi':
        # Grayscale format for Crimaldi data
        video_data = np.zeros((frame_count, height, width), dtype=np.uint8)
        format_specs = CRIMALDI_FORMAT_SPECS
    else:
        # RGB format for custom data
        video_data = np.zeros((frame_count, height, width, 3), dtype=np.uint16 if format_type == 'custom' else np.uint8)
        format_specs = CUSTOM_FORMAT_SPECS if format_type == 'custom' else CRIMALDI_FORMAT_SPECS
    
    # Generate plume source location
    source_x, source_y = 0.2, 0.5  # Fixed source location for reproducibility
    
    # Generate frames with temporal dynamics
    for frame_idx in range(frame_count):
        # Calculate time-dependent parameters
        t = frame_idx / frame_rate
        
        # Generate plume concentration field using Gaussian model
        sigma_x = 0.1 + 0.02 * t  # Growing plume width
        sigma_y = 0.05 + 0.01 * t  # Growing plume height
        
        # Create Gaussian plume distribution
        concentration = np.exp(
            -((X - source_x - 0.1 * t) ** 2) / (2 * sigma_x ** 2) -
            ((Y - source_y) ** 2) / (2 * sigma_y ** 2)
        )
        
        # Add realistic noise and temporal dynamics
        noise_level = 0.1 * format_specs.get('bit_depth', 8) / 8
        noise = np.random.normal(0, noise_level, concentration.shape)
        concentration += noise
        
        # Apply turbulence effects
        turbulence_x = 0.05 * np.sin(2 * np.pi * frame_idx / 30) * np.sin(10 * np.pi * Y)
        turbulence_y = 0.02 * np.cos(2 * np.pi * frame_idx / 20) * np.cos(8 * np.pi * X)
        
        # Apply turbulence to concentration field
        concentration = np.roll(concentration, int(turbulence_x.mean() * width), axis=1)
        concentration = np.roll(concentration, int(turbulence_y.mean() * height), axis=0)
        
        # Apply format-specific characteristics (bit depth, color space)
        if format_type == 'crimaldi':
            # Convert to 8-bit grayscale
            frame_data = np.clip(concentration * 255, 0, 255).astype(np.uint8)
            video_data[frame_idx] = frame_data
        else:
            # Convert to RGB format
            if format_specs['bit_depth'] == 16:
                max_val = 65535
                dtype = np.uint16
            else:
                max_val = 255
                dtype = np.uint8
            
            # Create RGB channels with slight variations
            r_channel = np.clip(concentration * max_val, 0, max_val).astype(dtype)
            g_channel = np.clip(concentration * max_val * 0.8, 0, max_val).astype(dtype)
            b_channel = np.clip(concentration * max_val * 0.6, 0, max_val).astype(dtype)
            
            if len(video_data.shape) == 4:  # RGB format
                video_data[frame_idx, :, :, 0] = r_channel
                video_data[frame_idx, :, :, 1] = g_channel
                video_data[frame_idx, :, :, 2] = b_channel
    
    # Add format-specific metadata and calibration parameters
    video_metadata = {
        'format_type': format_type,
        'dimensions': dimensions,
        'frame_count': frame_count,
        'frame_rate': frame_rate,
        'bit_depth': format_specs.get('bit_depth', 8),
        'color_space': format_specs.get('color_space', 'rgb'),
        'pixel_to_meter_ratio': format_specs.get('pixel_to_meter_ratio', 100.0),
        'temporal_resolution': format_specs.get('temporal_resolution', 30.0),
        'generation_timestamp': datetime.datetime.now().isoformat(),
        'source_location': (source_x, source_y)
    }
    
    # Validate generated data meets format requirements
    if video_data.size == 0:
        raise RuntimeError("Failed to generate video data")
    
    if np.any(np.isnan(video_data)) or np.any(np.isinf(video_data)):
        raise RuntimeError("Generated video data contains invalid values")
    
    # Return video data array with proper dtype and shape
    return video_data


def validate_cross_format_compatibility(
    crimaldi_results: dict,
    custom_results: dict,
    compatibility_threshold: float = 0.9
) -> ValidationResult:
    """
    Validate compatibility between different plume recording formats ensuring consistent processing results.
    
    This function provides comprehensive cross-format validation to ensure consistent
    processing results across Crimaldi and custom plume data formats.
    
    Args:
        crimaldi_results: Processing results from Crimaldi format data
        custom_results: Processing results from custom format data
        compatibility_threshold: Minimum compatibility score required
        
    Returns:
        ValidationResult: Cross-format compatibility validation result with detailed analysis
    """
    # Create ValidationResult container for compatibility assessment
    validation_result = ValidationResult(
        validation_type='cross_format_compatibility',
        is_valid=True,
        validation_context=f"compatibility_threshold={compatibility_threshold}"
    )
    
    try:
        # Compare spatial calibration consistency between formats
        if 'spatial_calibration' in crimaldi_results and 'spatial_calibration' in custom_results:
            crimaldi_cal = crimaldi_results['spatial_calibration']
            custom_cal = custom_results['spatial_calibration']
            
            # Compare pixel-to-meter ratios
            if 'pixel_to_meter_ratio' in crimaldi_cal and 'pixel_to_meter_ratio' in custom_cal:
                ratio_diff = abs(crimaldi_cal['pixel_to_meter_ratio'] - custom_cal['pixel_to_meter_ratio'])
                relative_diff = ratio_diff / max(crimaldi_cal['pixel_to_meter_ratio'], custom_cal['pixel_to_meter_ratio'])
                
                if relative_diff > 0.2:  # Allow 20% difference in calibration
                    validation_result.add_warning(
                        f"Large difference in pixel-to-meter ratios: {relative_diff:.3f}"
                    )
        
        # Validate temporal alignment and synchronization accuracy
        if 'temporal_data' in crimaldi_results and 'temporal_data' in custom_results:
            crimaldi_temporal = crimaldi_results['temporal_data']
            custom_temporal = custom_results['temporal_data']
            
            # Compare frame rates
            if 'frame_rate' in crimaldi_temporal and 'frame_rate' in custom_temporal:
                fps_diff = abs(crimaldi_temporal['frame_rate'] - custom_temporal['frame_rate'])
                if fps_diff > 5.0:  # Allow 5 FPS difference
                    validation_result.add_warning(
                        f"Significant frame rate difference: {fps_diff:.1f} FPS"
                    )
        
        # Check intensity unit conversion consistency
        if 'intensity_data' in crimaldi_results and 'intensity_data' in custom_results:
            crimaldi_intensity = crimaldi_results['intensity_data']
            custom_intensity = custom_results['intensity_data']
            
            # Compare intensity ranges and distributions
            if 'intensity_range' in crimaldi_intensity and 'intensity_range' in custom_intensity:
                crimaldi_range = crimaldi_intensity['intensity_range']
                custom_range = custom_intensity['intensity_range']
                
                # Calculate range overlap
                overlap_min = max(crimaldi_range[0], custom_range[0])
                overlap_max = min(crimaldi_range[1], custom_range[1])
                overlap_ratio = (overlap_max - overlap_min) / (custom_range[1] - custom_range[0])
                
                if overlap_ratio < 0.8:  # Require 80% overlap
                    validation_result.add_error(
                        f"Insufficient intensity range overlap: {overlap_ratio:.3f}"
                    )
                    validation_result.is_valid = False
        
        # Assess coordinate system transformation accuracy
        if 'coordinate_transform' in crimaldi_results and 'coordinate_transform' in custom_results:
            crimaldi_transform = crimaldi_results['coordinate_transform']
            custom_transform = custom_results['coordinate_transform']
            
            # Compare transformation matrices if available
            if 'transform_matrix' in crimaldi_transform and 'transform_matrix' in custom_transform:
                crimaldi_matrix = np.array(crimaldi_transform['transform_matrix'])
                custom_matrix = np.array(custom_transform['transform_matrix'])
                
                # Calculate matrix similarity
                matrix_diff = np.linalg.norm(crimaldi_matrix - custom_matrix)
                if matrix_diff > 0.1:  # Threshold for matrix similarity
                    validation_result.add_warning(
                        f"Coordinate transformation matrices differ significantly: {matrix_diff:.6f}"
                    )
        
        # Calculate cross-format correlation coefficients
        if 'trajectory_data' in crimaldi_results and 'trajectory_data' in custom_results:
            crimaldi_trajectories = np.array(crimaldi_results['trajectory_data'])
            custom_trajectories = np.array(custom_results['trajectory_data'])
            
            if crimaldi_trajectories.shape == custom_trajectories.shape:
                correlation_matrix = np.corrcoef(
                    crimaldi_trajectories.flatten(),
                    custom_trajectories.flatten()
                )
                cross_correlation = correlation_matrix[0, 1]
                
                validation_result.add_metric('cross_format_correlation', cross_correlation)
                
                if cross_correlation < compatibility_threshold:
                    validation_result.add_error(
                        f"Cross-format correlation {cross_correlation:.6f} below threshold {compatibility_threshold}"
                    )
                    validation_result.is_valid = False
        
        # Generate compatibility report with detailed metrics
        compatibility_score = 1.0
        error_count = len(validation_result.errors)
        warning_count = len(validation_result.warnings)
        
        # Reduce score based on errors and warnings
        compatibility_score -= error_count * 0.2
        compatibility_score -= warning_count * 0.1
        compatibility_score = max(0.0, compatibility_score)
        
        validation_result.add_metric('compatibility_score', compatibility_score)
        validation_result.add_metric('error_count', error_count)
        validation_result.add_metric('warning_count', warning_count)
        
        # Return validation result with pass/fail status
        if compatibility_score < compatibility_threshold:
            validation_result.is_valid = False
            validation_result.add_recommendation(
                "Review format-specific processing parameters and calibration methods"
            )
    
    except Exception as e:
        validation_result.add_error(f"Cross-format compatibility validation failed: {e}")
        validation_result.is_valid = False
    
    return validation_result


@contextlib.contextmanager
def setup_test_environment(
    test_name: str,
    cleanup_on_exit: bool = True
):
    """
    Context manager for setting up isolated test environment with temporary directories and resource management.
    
    This context manager provides comprehensive test environment setup with automatic cleanup
    and resource management for isolated testing scenarios.
    
    Args:
        test_name: Unique identifier for the test environment
        cleanup_on_exit: Whether to clean up temporary files on context exit
        
    Yields:
        dict: Test environment context with temporary directories and configuration
    """
    import tempfile
    import atexit
    
    # Create temporary directory for test isolation
    temp_dir = tempfile.mkdtemp(prefix=f"test_{test_name}_")
    temp_path = pathlib.Path(temp_dir)
    
    # Set up test-specific configuration and environment variables
    test_environment = {
        'test_name': test_name,
        'temp_directory': temp_path,
        'fixtures_directory': temp_path / 'fixtures',
        'output_directory': temp_path / 'output',
        'config_directory': temp_path / 'config',
        'log_directory': temp_path / 'logs',
        'start_time': datetime.datetime.now(),
        'cleanup_on_exit': cleanup_on_exit,
        'environment_id': str(uuid.uuid4())
    }
    
    # Create subdirectories for test organization
    for directory in ['fixtures', 'output', 'config', 'logs']:
        (temp_path / directory).mkdir(parents=True, exist_ok=True)
    
    # Register test environment in global registry
    global _test_environment_registry
    _test_environment_registry[test_environment['environment_id']] = test_environment
    
    # Initialize logging for test execution
    test_log_file = test_environment['log_directory'] / f"{test_name}.log"
    
    try:
        # Yield test environment context to calling code
        yield test_environment
        
    except Exception as e:
        # Log test environment exceptions
        test_environment['exception'] = {
            'type': type(e).__name__,
            'message': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        }
        raise
        
    finally:
        # Calculate test environment duration
        test_environment['end_time'] = datetime.datetime.now()
        test_environment['duration_seconds'] = (
            test_environment['end_time'] - test_environment['start_time']
        ).total_seconds()
        
        # Cleanup temporary files and directories if cleanup_on_exit is True
        if cleanup_on_exit:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                test_environment['cleanup_completed'] = True
            except Exception as cleanup_error:
                warnings.warn(
                    f"Failed to cleanup test environment {test_name}: {cleanup_error}",
                    UserWarning
                )
                test_environment['cleanup_completed'] = False
        
        # Remove from test environment registry
        _test_environment_registry.pop(test_environment['environment_id'], None)


def validate_batch_processing_results(
    batch_results: List[dict],
    expected_count: int = BATCH_TARGET_SIMULATIONS,
    completion_threshold: float = 0.95
) -> ValidationResult:
    """
    Validate batch processing results for 4000+ simulations with completion rate and consistency checking.
    
    This function provides comprehensive validation of batch processing results to ensure
    successful completion and consistency across large-scale simulation batches.
    
    Args:
        batch_results: List of individual simulation results from batch processing
        expected_count: Expected number of simulation results
        completion_threshold: Minimum completion rate required (0.95 = 95%)
        
    Returns:
        ValidationResult: Batch processing validation result with detailed statistics
    """
    # Create ValidationResult container for batch validation
    validation_result = ValidationResult(
        validation_type='batch_processing_validation',
        is_valid=True,
        validation_context=f"expected_count={expected_count}, threshold={completion_threshold}"
    )
    
    try:
        # Count successful and failed simulations
        total_results = len(batch_results)
        successful_results = []
        failed_results = []
        
        for result in batch_results:
            if isinstance(result, dict):
                if result.get('success', False):
                    successful_results.append(result)
                else:
                    failed_results.append(result)
            else:
                failed_results.append({'error': 'Invalid result format', 'result': result})
        
        # Calculate completion rate and compare against threshold
        completion_rate = len(successful_results) / max(1, expected_count)
        actual_completion_rate = len(successful_results) / max(1, total_results)
        
        validation_result.add_metric('total_results', total_results)
        validation_result.add_metric('successful_results', len(successful_results))
        validation_result.add_metric('failed_results', len(failed_results))
        validation_result.add_metric('completion_rate', completion_rate)
        validation_result.add_metric('actual_completion_rate', actual_completion_rate)
        
        # Validate completion rate against threshold
        if completion_rate < completion_threshold:
            validation_result.add_error(
                f"Batch completion rate {completion_rate:.3f} below threshold {completion_threshold}"
            )
            validation_result.is_valid = False
        
        # Validate result consistency across batch
        if successful_results:
            # Check for consistent result structure
            first_result_keys = set(successful_results[0].keys())
            inconsistent_results = 0
            
            for result in successful_results[1:]:
                if set(result.keys()) != first_result_keys:
                    inconsistent_results += 1
            
            if inconsistent_results > 0:
                validation_result.add_warning(
                    f"Found {inconsistent_results} results with inconsistent structure"
                )
            
            # Check for outliers and anomalous results
            if 'execution_time' in successful_results[0]:
                execution_times = [r.get('execution_time', 0) for r in successful_results]
                execution_times = [t for t in execution_times if t > 0]
                
                if execution_times:
                    mean_time = np.mean(execution_times)
                    std_time = np.std(execution_times)
                    outlier_threshold = mean_time + 3 * std_time
                    
                    outliers = [t for t in execution_times if t > outlier_threshold]
                    
                    validation_result.add_metric('mean_execution_time', mean_time)
                    validation_result.add_metric('std_execution_time', std_time)
                    validation_result.add_metric('outlier_count', len(outliers))
                    
                    if len(outliers) > 0.05 * len(execution_times):  # More than 5% outliers
                        validation_result.add_warning(
                            f"High number of execution time outliers: {len(outliers)}"
                        )
        
        # Assess processing time distribution
        if successful_results and 'processing_time' in successful_results[0]:
            processing_times = [r.get('processing_time', 0) for r in successful_results]
            processing_times = [t for t in processing_times if t > 0]
            
            if processing_times:
                avg_processing_time = np.mean(processing_times)
                max_processing_time = np.max(processing_times)
                
                validation_result.add_metric('avg_processing_time', avg_processing_time)
                validation_result.add_metric('max_processing_time', max_processing_time)
                
                # Check against performance requirements
                if avg_processing_time > PERFORMANCE_TIMEOUT_SECONDS:
                    validation_result.add_error(
                        f"Average processing time {avg_processing_time:.3f}s exceeds limit {PERFORMANCE_TIMEOUT_SECONDS}s"
                    )
                    validation_result.is_valid = False
        
        # Generate batch processing report with statistics
        batch_summary = {
            'batch_size': total_results,
            'success_count': len(successful_results),
            'failure_count': len(failed_results),
            'completion_rate': completion_rate,
            'meets_threshold': completion_rate >= completion_threshold,
            'validation_timestamp': datetime.datetime.now().isoformat()
        }
        
        validation_result.set_metadata('batch_summary', batch_summary)
        
        # Return validation result with detailed metrics
        if total_results == 0:
            validation_result.add_error("No batch results provided for validation")
            validation_result.is_valid = False
        
    except Exception as e:
        validation_result.add_error(f"Batch validation failed: {e}")
        validation_result.is_valid = False
    
    return validation_result


def compare_algorithm_performance(
    algorithm_results: dict,
    reference_algorithm: str = 'infotaxis',
    significance_level: float = 0.05
) -> dict:
    """
    Compare performance between different navigation algorithms with statistical significance testing.
    
    This function provides comprehensive algorithm performance comparison with statistical
    analysis and significance testing for scientific validation.
    
    Args:
        algorithm_results: Dictionary mapping algorithm names to performance results
        reference_algorithm: Name of the reference algorithm for comparison
        significance_level: Statistical significance level for hypothesis testing
        
    Returns:
        dict: Algorithm performance comparison results with statistical analysis
    """
    from scipy import stats
    
    # Validate algorithm results data structure
    if not isinstance(algorithm_results, dict):
        raise TypeError("Algorithm results must be a dictionary")
    
    if reference_algorithm not in algorithm_results:
        raise ValueError(f"Reference algorithm '{reference_algorithm}' not found in results")
    
    comparison_results = {
        'reference_algorithm': reference_algorithm,
        'significance_level': significance_level,
        'comparison_timestamp': datetime.datetime.now().isoformat(),
        'algorithm_comparisons': {},
        'performance_ranking': [],
        'statistical_summary': {}
    }
    
    try:
        # Calculate performance metrics for each algorithm
        algorithm_metrics = {}
        
        for algo_name, results in algorithm_results.items():
            if not isinstance(results, dict):
                continue
            
            # Extract performance metrics from results
            metrics = {
                'success_rate': results.get('success_rate', 0.0),
                'average_time': results.get('average_time', float('inf')),
                'path_efficiency': results.get('path_efficiency', 0.0),
                'convergence_rate': results.get('convergence_rate', 0.0),
                'error_rate': results.get('error_rate', 1.0)
            }
            
            algorithm_metrics[algo_name] = metrics
        
        # Get reference algorithm metrics
        reference_metrics = algorithm_metrics[reference_algorithm]
        
        # Perform statistical significance testing between algorithms
        for algo_name, metrics in algorithm_metrics.items():
            if algo_name == reference_algorithm:
                continue
            
            algorithm_comparison = {
                'algorithm': algo_name,
                'reference': reference_algorithm,
                'metrics_comparison': {},
                'statistical_tests': {},
                'performance_improvement': {},
                'significance_results': {}
            }
            
            # Compare each metric
            for metric_name in metrics.keys():
                ref_value = reference_metrics.get(metric_name, 0)
                test_value = metrics.get(metric_name, 0)
                
                # Calculate relative improvement
                if ref_value != 0:
                    if metric_name in ['success_rate', 'path_efficiency', 'convergence_rate']:
                        # Higher is better
                        improvement = (test_value - ref_value) / ref_value
                    else:
                        # Lower is better (time, error_rate)
                        improvement = (ref_value - test_value) / ref_value
                else:
                    improvement = 0.0
                
                algorithm_comparison['metrics_comparison'][metric_name] = {
                    'reference_value': ref_value,
                    'test_value': test_value,
                    'improvement': improvement,
                    'is_better': improvement > 0
                }
                
                # Perform t-test if sample data is available
                if f'{metric_name}_samples' in results and f'{metric_name}_samples' in algorithm_results[reference_algorithm]:
                    ref_samples = algorithm_results[reference_algorithm][f'{metric_name}_samples']
                    test_samples = results[f'{metric_name}_samples']
                    
                    try:
                        t_statistic, p_value = stats.ttest_ind(test_samples, ref_samples)
                        
                        algorithm_comparison['statistical_tests'][metric_name] = {
                            't_statistic': t_statistic,
                            'p_value': p_value,
                            'is_significant': p_value < significance_level,
                            'degrees_of_freedom': len(test_samples) + len(ref_samples) - 2
                        }
                    except Exception as e:
                        algorithm_comparison['statistical_tests'][metric_name] = {
                            'error': str(e)
                        }
            
            comparison_results['algorithm_comparisons'][algo_name] = algorithm_comparison
        
        # Generate algorithm ranking based on performance criteria
        ranking_scores = {}
        
        for algo_name, metrics in algorithm_metrics.items():
            # Calculate composite performance score
            score = 0.0
            score += metrics.get('success_rate', 0.0) * 0.3
            score += (1.0 / max(metrics.get('average_time', 1.0), 0.1)) * 0.2
            score += metrics.get('path_efficiency', 0.0) * 0.2
            score += metrics.get('convergence_rate', 0.0) * 0.2
            score -= metrics.get('error_rate', 0.0) * 0.1
            
            ranking_scores[algo_name] = max(0.0, score)
        
        # Sort algorithms by performance score
        sorted_algorithms = sorted(ranking_scores.items(), key=lambda x: x[1], reverse=True)
        comparison_results['performance_ranking'] = [
            {'algorithm': algo, 'score': score, 'rank': rank + 1}
            for rank, (algo, score) in enumerate(sorted_algorithms)
        ]
        
        # Calculate effect sizes and confidence intervals
        for algo_name in algorithm_comparisons.keys():
            if algo_name in comparison_results['algorithm_comparisons']:
                comparison = comparison_results['algorithm_comparisons'][algo_name]
                
                # Calculate Cohen's d effect size for key metrics
                for metric_name in ['success_rate', 'average_time']:
                    if f'{metric_name}_samples' in algorithm_results.get(algo_name, {}):
                        ref_samples = algorithm_results[reference_algorithm].get(f'{metric_name}_samples', [])
                        test_samples = algorithm_results[algo_name].get(f'{metric_name}_samples', [])
                        
                        if len(ref_samples) > 1 and len(test_samples) > 1:
                            ref_mean = np.mean(ref_samples)
                            test_mean = np.mean(test_samples)
                            pooled_std = np.sqrt(((len(ref_samples) - 1) * np.var(ref_samples) + 
                                                (len(test_samples) - 1) * np.var(test_samples)) / 
                                               (len(ref_samples) + len(test_samples) - 2))
                            
                            cohens_d = (test_mean - ref_mean) / pooled_std if pooled_std > 0 else 0
                            
                            comparison['effect_sizes'] = comparison.get('effect_sizes', {})
                            comparison['effect_sizes'][metric_name] = {
                                'cohens_d': cohens_d,
                                'magnitude': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
                            }
        
        # Create performance comparison visualization data
        comparison_results['visualization_data'] = {
            'metric_names': list(reference_metrics.keys()),
            'algorithm_names': list(algorithm_metrics.keys()),
            'performance_matrix': [[algorithm_metrics[algo][metric] for metric in reference_metrics.keys()] 
                                 for algo in algorithm_metrics.keys()]
        }
        
        # Generate detailed comparison report
        comparison_results['statistical_summary'] = {
            'total_algorithms': len(algorithm_metrics),
            'significant_differences': sum(
                1 for comp in comparison_results['algorithm_comparisons'].values()
                for test in comp.get('statistical_tests', {}).values()
                if test.get('is_significant', False)
            ),
            'best_performing_algorithm': sorted_algorithms[0][0] if sorted_algorithms else None,
            'performance_spread': max(ranking_scores.values()) - min(ranking_scores.values()) if ranking_scores else 0
        }
        
        # Return comparison results with statistical analysis
        return comparison_results
        
    except Exception as e:
        comparison_results['error'] = str(e)
        comparison_results['validation_failed'] = True
        return comparison_results


def generate_test_report(
    test_results: dict,
    report_format: str = 'json',
    output_path: str = None
) -> str:
    """
    Generate comprehensive test report with validation results, performance metrics, and recommendations.
    
    This function creates detailed test reports with comprehensive analysis and recommendations
    for test results and performance metrics.
    
    Args:
        test_results: Dictionary containing test execution results and metrics
        report_format: Format for the generated report ('json', 'html', 'text')
        output_path: Path to save the generated report file
        
    Returns:
        str: Path to generated test report file or report content
    """
    # Validate test results data structure and completeness
    if not isinstance(test_results, dict):
        raise TypeError("Test results must be a dictionary")
    
    # Aggregate validation results and performance metrics
    report_data = {
        'report_metadata': {
            'generation_timestamp': datetime.datetime.now().isoformat(),
            'report_format': report_format,
            'output_path': output_path,
            'generator_version': '1.0.0'
        },
        'test_summary': {},
        'performance_analysis': {},
        'validation_results': {},
        'recommendations': [],
        'detailed_results': test_results
    }
    
    try:
        # Generate summary statistics and trend analysis
        total_tests = test_results.get('total_tests', 0)
        passed_tests = test_results.get('passed_tests', 0)
        failed_tests = test_results.get('failed_tests', 0)
        skipped_tests = test_results.get('skipped_tests', 0)
        
        report_data['test_summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'skipped_tests': skipped_tests,
            'pass_rate': passed_tests / max(1, total_tests),
            'failure_rate': failed_tests / max(1, total_tests),
            'execution_time': test_results.get('total_execution_time', 0)
        }
        
        # Analyze performance metrics
        if 'performance_metrics' in test_results:
            perf_metrics = test_results['performance_metrics']
            
            report_data['performance_analysis'] = {
                'average_test_time': perf_metrics.get('average_test_time', 0),
                'slowest_test_time': perf_metrics.get('slowest_test_time', 0),
                'memory_usage_peak': perf_metrics.get('peak_memory_mb', 0),
                'cpu_usage_average': perf_metrics.get('average_cpu_percent', 0),
                'meets_performance_targets': perf_metrics.get('average_test_time', float('inf')) <= PERFORMANCE_TIMEOUT_SECONDS
            }
        
        # Analyze validation results
        if 'validation_results' in test_results:
            validation_data = test_results['validation_results']
            
            report_data['validation_results'] = {
                'correlation_scores': validation_data.get('correlation_scores', []),
                'accuracy_metrics': validation_data.get('accuracy_metrics', {}),
                'cross_format_compatibility': validation_data.get('cross_format_compatibility', {}),
                'numerical_precision_tests': validation_data.get('numerical_precision', {})
            }
        
        # Create visualizations for key metrics and comparisons
        if report_format == 'html':
            report_data['visualizations'] = {
                'test_summary_chart': _generate_test_summary_chart(report_data['test_summary']),
                'performance_trend_chart': _generate_performance_trend_chart(test_results.get('performance_history', [])),
                'correlation_distribution': _generate_correlation_distribution(test_results.get('correlation_scores', []))
            }
        
        # Include recommendations and action items
        recommendations = []
        
        # Performance recommendations
        if report_data['test_summary']['pass_rate'] < 0.95:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Test Quality',
                'description': f"Test pass rate {report_data['test_summary']['pass_rate']:.2%} below target 95%",
                'action': 'Review failing tests and improve test stability'
            })
        
        if 'performance_analysis' in report_data and not report_data['performance_analysis']['meets_performance_targets']:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Performance',
                'description': 'Some tests exceed performance time targets',
                'action': 'Optimize slow tests or increase timeout thresholds'
            })
        
        # Validation recommendations
        if 'validation_results' in report_data:
            validation_results = report_data['validation_results']
            if 'correlation_scores' in validation_results:
                avg_correlation = np.mean(validation_results['correlation_scores']) if validation_results['correlation_scores'] else 0
                if avg_correlation < CORRELATION_THRESHOLD:
                    recommendations.append({
                        'priority': 'HIGH',
                        'category': 'Validation',
                        'description': f'Average correlation {avg_correlation:.3f} below threshold {CORRELATION_THRESHOLD}',
                        'action': 'Review algorithm implementations and reference data'
                    })
        
        report_data['recommendations'] = recommendations
        
        # Format report according to specified format (HTML, PDF, JSON)
        if report_format.lower() == 'json':
            report_content = json.dumps(report_data, indent=2, default=str)
        elif report_format.lower() == 'html':
            report_content = _generate_html_report(report_data)
        elif report_format.lower() == 'text':
            report_content = _generate_text_report(report_data)
        else:
            raise ValueError(f"Unsupported report format: {report_format}")
        
        # Save report to specified output path
        if output_path:
            output_file = pathlib.Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return str(output_file)
        else:
            return report_content
            
    except Exception as e:
        error_report = {
            'error': f"Report generation failed: {e}",
            'timestamp': datetime.datetime.now().isoformat(),
            'test_results_summary': {
                'total_tests': test_results.get('total_tests', 0),
                'passed_tests': test_results.get('passed_tests', 0),
                'failed_tests': test_results.get('failed_tests', 0)
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(error_report, f, indent=2)
            return output_path
        else:
            return json.dumps(error_report, indent=2)


def cache_test_data(
    cache_key: str,
    data: object,
    ttl_hours: int = 24
) -> bool:
    """
    Cache test data and results for improved test execution performance with TTL and invalidation.
    
    This function provides test data caching with time-to-live management for improved
    test execution performance and reduced resource usage.
    
    Args:
        cache_key: Unique identifier for the cached data
        data: Data object to cache (must be serializable)
        ttl_hours: Time-to-live in hours for cache expiration
        
    Returns:
        bool: True if data was cached successfully
    """
    global _validation_cache
    
    try:
        # Generate unique cache key based on input parameters
        normalized_key = f"test_data_{hashlib.md5(cache_key.encode()).hexdigest()}"
        
        # Validate data is serializable for caching
        try:
            json.dumps(data, default=str)
        except (TypeError, ValueError) as e:
            # Try to convert complex objects to serializable format
            if hasattr(data, '__dict__'):
                data = data.__dict__
            elif isinstance(data, np.ndarray):
                data = data.tolist()
            else:
                raise ValueError(f"Data is not serializable: {e}")
        
        # Store data in cache with TTL expiration
        cache_entry = {
            'data': data,
            'cached_at': datetime.datetime.now(),
            'ttl_hours': ttl_hours,
            'expires_at': datetime.datetime.now() + datetime.timedelta(hours=ttl_hours),
            'cache_key': cache_key,
            'access_count': 0,
            'last_accessed': datetime.datetime.now()
        }
        
        _validation_cache[normalized_key] = cache_entry
        
        # Update cache metadata and access statistics
        cache_stats = _validation_cache.get('_cache_stats', {
            'total_entries': 0,
            'total_size_bytes': 0,
            'hit_count': 0,
            'miss_count': 0,
            'last_cleanup': datetime.datetime.now()
        })
        
        cache_stats['total_entries'] = len(_validation_cache) - 1  # Exclude stats entry
        _validation_cache['_cache_stats'] = cache_stats
        
        # Handle cache storage errors gracefully
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to cache test data for key '{cache_key}': {e}", UserWarning)
        return False


def get_cached_test_data(cache_key: str) -> Optional[object]:
    """
    Retrieve cached test data with TTL validation and access tracking.
    
    Args:
        cache_key: Unique identifier for the cached data
        
    Returns:
        object: Cached data if found and valid, None otherwise
    """
    global _validation_cache
    
    try:
        normalized_key = f"test_data_{hashlib.md5(cache_key.encode()).hexdigest()}"
        
        if normalized_key in _validation_cache:
            cache_entry = _validation_cache[normalized_key]
            
            # Check TTL expiration
            if datetime.datetime.now() > cache_entry['expires_at']:
                del _validation_cache[normalized_key]
                return None
            
            # Update access statistics
            cache_entry['access_count'] += 1
            cache_entry['last_accessed'] = datetime.datetime.now()
            
            # Update cache statistics
            if '_cache_stats' in _validation_cache:
                _validation_cache['_cache_stats']['hit_count'] += 1
            
            return cache_entry['data']
        else:
            # Update miss statistics
            if '_cache_stats' in _validation_cache:
                _validation_cache['_cache_stats']['miss_count'] += 1
            
            return None
            
    except Exception as e:
        warnings.warn(f"Failed to retrieve cached data for key '{cache_key}': {e}", UserWarning)
        return None


class TestDataValidator:
    """
    Comprehensive test data validation class providing validation for video data, normalization results, 
    and simulation outputs with scientific accuracy requirements.
    
    This class provides specialized validation methods for different types of test data
    with scientific accuracy requirements and comprehensive error reporting.
    """
    
    def __init__(
        self,
        tolerance: float = DEFAULT_TOLERANCE,
        strict_validation: bool = True
    ):
        """
        Initialize test data validator with tolerance and validation settings.
        
        Args:
            tolerance: Numerical tolerance for validation comparisons
            strict_validation: Enable strict validation mode with comprehensive checks
        """
        # Set numerical tolerance for validation comparisons
        self.tolerance = tolerance
        
        # Configure strict validation mode
        self.strict_validation = strict_validation
        
        # Initialize PhysicalConstants for parameter validation
        self.physical_constants = PhysicalConstants()
        
        # Setup validation cache for performance optimization
        self.validation_cache = {}
        
        # Initialize logging for validation operations
        self.validation_history = []
    
    def validate_video_data(
        self,
        video_data: np.ndarray,
        expected_properties: dict
    ) -> ValidationResult:
        """
        Validate video data format, dimensions, and content for test compatibility.
        
        This method provides comprehensive video data validation including format checking,
        dimensional analysis, and content validation for test scenarios.
        
        Args:
            video_data: Video data array to validate
            expected_properties: Expected video properties for validation
            
        Returns:
            ValidationResult: Video data validation result with detailed analysis
        """
        validation_result = ValidationResult(
            validation_type='video_data_validation',
            is_valid=True,
            validation_context=f"tolerance={self.tolerance}"
        )
        
        try:
            # Check video data array shape and dtype
            if not isinstance(video_data, np.ndarray):
                validation_result.add_error("Video data must be numpy array")
                validation_result.is_valid = False
                return validation_result
            
            # Validate frame count and dimensions
            expected_shape = expected_properties.get('shape')
            if expected_shape and video_data.shape != tuple(expected_shape):
                validation_result.add_error(
                    f"Video shape {video_data.shape} does not match expected {expected_shape}"
                )
                validation_result.is_valid = False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(video_data)) or np.any(np.isinf(video_data)):
                validation_result.add_error("Video data contains NaN or infinite values")
                validation_result.is_valid = False
            
            # Validate intensity range and distribution
            expected_dtype = expected_properties.get('dtype')
            if expected_dtype and video_data.dtype != expected_dtype:
                validation_result.add_warning(
                    f"Video dtype {video_data.dtype} differs from expected {expected_dtype}"
                )
            
            # Check intensity value ranges
            if video_data.dtype == np.uint8:
                valid_range = (0, 255)
            elif video_data.dtype == np.uint16:
                valid_range = (0, 65535)
            else:
                valid_range = (0, 1)
            
            if np.any(video_data < valid_range[0]) or np.any(video_data > valid_range[1]):
                validation_result.add_error(
                    f"Video values outside valid range {valid_range}"
                )
                validation_result.is_valid = False
            
            # Compare against expected properties
            if 'frame_count' in expected_properties:
                expected_frames = expected_properties['frame_count']
                actual_frames = video_data.shape[0] if len(video_data.shape) > 2 else 1
                
                if actual_frames != expected_frames:
                    validation_result.add_error(
                        f"Frame count {actual_frames} does not match expected {expected_frames}"
                    )
                    validation_result.is_valid = False
            
            # Validate video metadata if provided
            if 'frame_rate' in expected_properties:
                # Frame rate validation would require temporal analysis
                validation_result.add_metric('expected_frame_rate', expected_properties['frame_rate'])
            
            # Generate validation report
            validation_result.add_metric('video_shape', video_data.shape)
            validation_result.add_metric('video_dtype', str(video_data.dtype))
            validation_result.add_metric('intensity_min', float(np.min(video_data)))
            validation_result.add_metric('intensity_max', float(np.max(video_data)))
            validation_result.add_metric('intensity_mean', float(np.mean(video_data)))
            validation_result.add_metric('intensity_std', float(np.std(video_data)))
            
            # Return validation result
            return validation_result
            
        except Exception as e:
            validation_result.add_error(f"Video validation failed: {e}")
            validation_result.is_valid = False
            return validation_result
    
    def validate_normalization_results(
        self,
        normalized_data: np.ndarray,
        reference_data: np.ndarray
    ) -> ValidationResult:
        """
        Validate normalization pipeline results against reference benchmarks.
        
        This method validates normalization results by comparing against reference
        benchmarks and checking for proper normalization characteristics.
        
        Args:
            normalized_data: Normalized data array from processing pipeline
            reference_data: Reference benchmark data for comparison
            
        Returns:
            ValidationResult: Normalization validation result with detailed metrics
        """
        validation_result = ValidationResult(
            validation_type='normalization_validation',
            is_valid=True,
            validation_context=f"tolerance={self.tolerance}"
        )
        
        try:
            # Load reference benchmark data (already provided)
            if not isinstance(normalized_data, np.ndarray) or not isinstance(reference_data, np.ndarray):
                validation_result.add_error("Both normalized and reference data must be numpy arrays")
                validation_result.is_valid = False
                return validation_result
            
            # Compare normalized data against reference
            if normalized_data.shape != reference_data.shape:
                validation_result.add_error(
                    f"Shape mismatch: normalized {normalized_data.shape} vs reference {reference_data.shape}"
                )
                validation_result.is_valid = False
                return validation_result
            
            # Calculate correlation and accuracy metrics
            try:
                correlation_matrix = np.corrcoef(normalized_data.flatten(), reference_data.flatten())
                correlation = correlation_matrix[0, 1]
                
                validation_result.add_metric('correlation', correlation)
                
                if correlation < CORRELATION_THRESHOLD:
                    validation_result.add_error(
                        f"Correlation {correlation:.6f} below threshold {CORRELATION_THRESHOLD}"
                    )
                    validation_result.is_valid = False
                    
            except Exception as e:
                validation_result.add_error(f"Correlation calculation failed: {e}")
                validation_result.is_valid = False
            
            # Validate numerical precision within tolerance
            max_abs_diff = np.max(np.abs(normalized_data - reference_data))
            validation_result.add_metric('max_absolute_difference', max_abs_diff)
            
            if max_abs_diff > self.tolerance:
                validation_result.add_warning(
                    f"Maximum difference {max_abs_diff:.2e} exceeds tolerance {self.tolerance:.2e}"
                )
            
            # Check for systematic biases or errors
            mean_diff = np.mean(normalized_data - reference_data)
            validation_result.add_metric('mean_difference', mean_diff)
            
            if abs(mean_diff) > self.tolerance:
                validation_result.add_warning(
                    f"Systematic bias detected: mean difference {mean_diff:.6f}"
                )
            
            # Check normalization properties
            if np.min(normalized_data) < 0 or np.max(normalized_data) > 1:
                validation_result.add_warning(
                    "Normalized data values outside expected range [0, 1]"
                )
            
            # Generate detailed validation report
            validation_result.add_metric('rmse', np.sqrt(np.mean((normalized_data - reference_data) ** 2)))
            validation_result.add_metric('mae', np.mean(np.abs(normalized_data - reference_data)))
            
            # Return validation result with metrics
            return validation_result
            
        except Exception as e:
            validation_result.add_error(f"Normalization validation failed: {e}")
            validation_result.is_valid = False
            return validation_result
    
    def validate_simulation_outputs(
        self,
        simulation_results: dict,
        validation_criteria: dict
    ) -> ValidationResult:
        """
        Validate simulation outputs for accuracy, consistency, and performance requirements.
        
        This method provides comprehensive validation of simulation outputs including
        accuracy checking, consistency verification, and performance validation.
        
        Args:
            simulation_results: Dictionary containing simulation output data
            validation_criteria: Criteria and thresholds for validation
            
        Returns:
            ValidationResult: Simulation output validation result with recommendations
        """
        validation_result = ValidationResult(
            validation_type='simulation_output_validation',
            is_valid=True,
            validation_context=f"criteria_count={len(validation_criteria)}"
        )
        
        try:
            # Validate simulation result data structure
            required_fields = ['trajectory', 'performance_metrics', 'execution_time']
            for field in required_fields:
                if field not in simulation_results:
                    validation_result.add_error(f"Missing required field: {field}")
                    validation_result.is_valid = False
            
            if not validation_result.is_valid:
                return validation_result
            
            # Check trajectory accuracy and path efficiency
            trajectory = simulation_results.get('trajectory', [])
            if isinstance(trajectory, (list, np.ndarray)):
                trajectory_array = np.array(trajectory)
                
                # Validate trajectory properties
                if len(trajectory_array) == 0:
                    validation_result.add_error("Empty trajectory data")
                    validation_result.is_valid = False
                else:
                    # Check for valid trajectory points
                    if np.any(np.isnan(trajectory_array)) or np.any(np.isinf(trajectory_array)):
                        validation_result.add_error("Trajectory contains invalid values")
                        validation_result.is_valid = False
                    
                    # Calculate path efficiency metrics
                    if len(trajectory_array) > 1:
                        path_length = np.sum(np.linalg.norm(np.diff(trajectory_array, axis=0), axis=1))
                        direct_distance = np.linalg.norm(trajectory_array[-1] - trajectory_array[0])
                        path_efficiency = direct_distance / path_length if path_length > 0 else 0
                        
                        validation_result.add_metric('path_efficiency', path_efficiency)
                        validation_result.add_metric('path_length', path_length)
                        validation_result.add_metric('direct_distance', direct_distance)
                        
                        # Validate against efficiency criteria
                        min_efficiency = validation_criteria.get('min_path_efficiency', 0.5)
                        if path_efficiency < min_efficiency:
                            validation_result.add_warning(
                                f"Path efficiency {path_efficiency:.3f} below minimum {min_efficiency}"
                            )
            
            # Validate performance metrics against thresholds
            performance_metrics = simulation_results.get('performance_metrics', {})
            
            # Check execution time
            execution_time = simulation_results.get('execution_time', float('inf'))
            max_execution_time = validation_criteria.get('max_execution_time', PERFORMANCE_TIMEOUT_SECONDS)
            
            validation_result.add_metric('execution_time', execution_time)
            
            if execution_time > max_execution_time:
                validation_result.add_error(
                    f"Execution time {execution_time:.3f}s exceeds maximum {max_execution_time}s"
                )
                validation_result.is_valid = False
            
            # Check for convergence and stability
            if 'convergence_metrics' in performance_metrics:
                convergence = performance_metrics['convergence_metrics']
                
                if 'converged' in convergence and not convergence['converged']:
                    validation_result.add_warning("Simulation did not converge")
                
                if 'stability_metric' in convergence:
                    stability = convergence['stability_metric']
                    min_stability = validation_criteria.get('min_stability', 0.8)
                    
                    if stability < min_stability:
                        validation_result.add_warning(
                            f"Stability metric {stability:.3f} below minimum {min_stability}"
                        )
            
            # Compare against validation criteria
            for criterion_name, criterion_value in validation_criteria.items():
                if criterion_name in performance_metrics:
                    actual_value = performance_metrics[criterion_name]
                    
                    if isinstance(criterion_value, dict) and 'min' in criterion_value:
                        min_value = criterion_value['min']
                        if actual_value < min_value:
                            validation_result.add_error(
                                f"{criterion_name} {actual_value} below minimum {min_value}"
                            )
                            validation_result.is_valid = False
                    
                    if isinstance(criterion_value, dict) and 'max' in criterion_value:
                        max_value = criterion_value['max']
                        if actual_value > max_value:
                            validation_result.add_error(
                                f"{criterion_name} {actual_value} exceeds maximum {max_value}"
                            )
                            validation_result.is_valid = False
            
            # Generate comprehensive validation report
            validation_result.add_metric('total_criteria_checked', len(validation_criteria))
            validation_result.add_metric('simulation_success', validation_result.is_valid)
            
            # Return validation result with recommendations
            if not validation_result.is_valid:
                validation_result.add_recommendation(
                    "Review simulation parameters and algorithm configuration"
                )
            
            return validation_result
            
        except Exception as e:
            validation_result.add_error(f"Simulation validation failed: {e}")
            validation_result.is_valid = False
            return validation_result


class PerformanceProfiler:
    """
    Performance profiling class for monitoring test execution time, memory usage, and resource utilization 
    with threshold validation.
    
    This class provides comprehensive performance profiling with resource monitoring
    and threshold validation for test execution analysis.
    """
    
    def __init__(
        self,
        time_threshold_seconds: float = PERFORMANCE_TIMEOUT_SECONDS,
        memory_threshold_mb: int = MEMORY_LIMIT_MB
    ):
        """
        Initialize performance profiler with threshold settings.
        
        Args:
            time_threshold_seconds: Maximum execution time threshold in seconds
            memory_threshold_mb: Maximum memory usage threshold in MB
        """
        # Set performance thresholds for validation
        self.time_threshold = time_threshold_seconds
        self.memory_threshold = memory_threshold_mb
        
        # Initialize performance data collection
        self.performance_data = {}
        
        # Setup resource monitoring capabilities
        self.profiling_active = False
        
        # Configure profiling state management
        self.current_session = None
        self.session_history = []
    
    def start_profiling(self, session_name: str) -> None:
        """
        Start performance profiling session with resource monitoring.
        
        This method initializes a new profiling session with comprehensive resource
        monitoring and baseline measurements.
        
        Args:
            session_name: Unique identifier for the profiling session
        """
        import psutil
        import gc
        
        # Force garbage collection for accurate baseline
        gc.collect()
        
        # Initialize profiling session with name
        self.current_session = {
            'session_name': session_name,
            'start_time': time.time(),
            'start_timestamp': datetime.datetime.now(),
            'process': psutil.Process(),
            'initial_memory_mb': 0,
            'peak_memory_mb': 0,
            'cpu_samples': [],
            'memory_samples': [],
            'profiling_overhead': 0
        }
        
        # Record start time and memory baseline
        process = self.current_session['process']
        self.current_session['initial_memory_mb'] = process.memory_info().rss / 1024 / 1024
        self.current_session['peak_memory_mb'] = self.current_session['initial_memory_mb']
        
        # Begin resource monitoring
        self._start_resource_monitoring()
        
        # Set profiling_active flag to True
        self.profiling_active = True
    
    def stop_profiling(self) -> dict:
        """
        Stop profiling session and collect performance metrics.
        
        This method finalizes the profiling session and generates comprehensive
        performance metrics and analysis.
        
        Returns:
            dict: Performance metrics and statistics from the profiling session
        """
        if not self.profiling_active or not self.current_session:
            return {'error': 'No active profiling session'}
        
        # Record end time and peak memory usage
        end_time = time.time()
        process = self.current_session['process']
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Calculate execution time and resource consumption
        execution_time = end_time - self.current_session['start_time']
        memory_increase = final_memory_mb - self.current_session['initial_memory_mb']
        
        # Stop resource monitoring
        self._stop_resource_monitoring()
        
        # Set profiling_active flag to False
        self.profiling_active = False
        
        # Calculate performance statistics
        performance_metrics = {
            'session_name': self.current_session['session_name'],
            'execution_time_seconds': execution_time,
            'initial_memory_mb': self.current_session['initial_memory_mb'],
            'final_memory_mb': final_memory_mb,
            'peak_memory_mb': self.current_session['peak_memory_mb'],
            'memory_increase_mb': memory_increase,
            'cpu_usage_samples': self.current_session['cpu_samples'],
            'memory_usage_samples': self.current_session['memory_samples'],
            'average_cpu_percent': np.mean(self.current_session['cpu_samples']) if self.current_session['cpu_samples'] else 0,
            'end_timestamp': datetime.datetime.now(),
            'meets_time_threshold': execution_time <= self.time_threshold,
            'meets_memory_threshold': self.current_session['peak_memory_mb'] <= self.memory_threshold
        }
        
        # Add to session history
        self.session_history.append(performance_metrics)
        
        # Generate performance report
        performance_report = {
            'session_metrics': performance_metrics,
            'threshold_validation': {
                'time_threshold_seconds': self.time_threshold,
                'memory_threshold_mb': self.memory_threshold,
                'time_threshold_met': performance_metrics['meets_time_threshold'],
                'memory_threshold_met': performance_metrics['meets_memory_threshold'],
                'overall_performance_acceptable': (
                    performance_metrics['meets_time_threshold'] and 
                    performance_metrics['meets_memory_threshold']
                )
            },
            'resource_efficiency': {
                'memory_efficiency': 1.0 - (memory_increase / max(self.memory_threshold, 1)),
                'time_efficiency': 1.0 - (execution_time / max(self.time_threshold, 1))
            }
        }
        
        # Clear current session
        self.current_session = None
        
        # Return performance metrics dictionary
        return performance_report
    
    def validate_performance_thresholds(
        self,
        performance_metrics: dict
    ) -> ValidationResult:
        """
        Validate performance metrics against configured thresholds.
        
        This method validates performance metrics against predefined thresholds
        and generates recommendations for performance optimization.
        
        Args:
            performance_metrics: Dictionary containing performance measurements
            
        Returns:
            ValidationResult: Performance threshold validation result with recommendations
        """
        validation_result = ValidationResult(
            validation_type='performance_threshold_validation',
            is_valid=True,
            validation_context=f"time_threshold={self.time_threshold}, memory_threshold={self.memory_threshold}"
        )
        
        try:
            # Compare execution time against time threshold
            execution_time = performance_metrics.get('execution_time_seconds', float('inf'))
            
            validation_result.add_metric('execution_time', execution_time)
            validation_result.add_metric('time_threshold', self.time_threshold)
            
            if execution_time > self.time_threshold:
                validation_result.add_error(
                    f"Execution time {execution_time:.3f}s exceeds threshold {self.time_threshold}s"
                )
                validation_result.is_valid = False
                validation_result.add_recommendation(
                    "Optimize algorithm performance or increase time threshold"
                )
            
            # Check memory usage against memory threshold
            peak_memory = performance_metrics.get('peak_memory_mb', 0)
            
            validation_result.add_metric('peak_memory_mb', peak_memory)
            validation_result.add_metric('memory_threshold_mb', self.memory_threshold)
            
            if peak_memory > self.memory_threshold:
                validation_result.add_error(
                    f"Peak memory usage {peak_memory:.1f}MB exceeds threshold {self.memory_threshold}MB"
                )
                validation_result.is_valid = False
                validation_result.add_recommendation(
                    "Reduce memory usage or increase memory threshold"
                )
            
            # Validate resource utilization efficiency
            memory_efficiency = performance_metrics.get('memory_efficiency', 0)
            time_efficiency = performance_metrics.get('time_efficiency', 0)
            
            if memory_efficiency < 0.5:  # Less than 50% efficiency
                validation_result.add_warning(
                    f"Low memory efficiency: {memory_efficiency:.2%}"
                )
            
            if time_efficiency < 0.5:  # Less than 50% efficiency
                validation_result.add_warning(
                    f"Low time efficiency: {time_efficiency:.2%}"
                )
            
            # Calculate overall performance score
            overall_score = (
                (1.0 if execution_time <= self.time_threshold else 0.0) * 0.5 +
                (1.0 if peak_memory <= self.memory_threshold else 0.0) * 0.3 +
                memory_efficiency * 0.1 +
                time_efficiency * 0.1
            )
            
            validation_result.add_metric('overall_performance_score', overall_score)
            
            # Generate threshold validation report
            if overall_score < 0.7:  # Less than 70% overall performance
                validation_result.add_recommendation(
                    "Review performance optimization opportunities"
                )
            
            # Return validation result with recommendations
            return validation_result
            
        except Exception as e:
            validation_result.add_error(f"Performance validation failed: {e}")
            validation_result.is_valid = False
            return validation_result
    
    def _start_resource_monitoring(self) -> None:
        """Start background resource monitoring for the current session."""
        # This would typically start a background thread for monitoring
        # For simplicity, we'll collect initial samples
        if self.current_session:
            process = self.current_session['process']
            try:
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                self.current_session['cpu_samples'].append(cpu_percent)
                self.current_session['memory_samples'].append(memory_mb)
                self.current_session['peak_memory_mb'] = max(
                    self.current_session['peak_memory_mb'], memory_mb
                )
            except:
                pass  # Handle process monitoring errors gracefully
    
    def _stop_resource_monitoring(self) -> None:
        """Stop background resource monitoring and collect final samples."""
        if self.current_session:
            process = self.current_session['process']
            try:
                # Collect final resource usage samples
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                self.current_session['cpu_samples'].append(cpu_percent)
                self.current_session['memory_samples'].append(memory_mb)
                self.current_session['peak_memory_mb'] = max(
                    self.current_session['peak_memory_mb'], memory_mb
                )
            except:
                pass  # Handle process monitoring errors gracefully


# Helper functions for report generation

def _generate_test_summary_chart(summary_data: dict) -> dict:
    """Generate test summary chart data for HTML reports."""
    return {
        'chart_type': 'pie',
        'data': {
            'labels': ['Passed', 'Failed', 'Skipped'],
            'values': [
                summary_data.get('passed_tests', 0),
                summary_data.get('failed_tests', 0),
                summary_data.get('skipped_tests', 0)
            ]
        }
    }


def _generate_performance_trend_chart(performance_history: List[dict]) -> dict:
    """Generate performance trend chart data for HTML reports."""
    if not performance_history:
        return {'chart_type': 'line', 'data': {'x': [], 'y': []}}
    
    timestamps = [item.get('timestamp', '') for item in performance_history]
    execution_times = [item.get('execution_time', 0) for item in performance_history]
    
    return {
        'chart_type': 'line',
        'data': {
            'x': timestamps,
            'y': execution_times
        }
    }


def _generate_correlation_distribution(correlation_scores: List[float]) -> dict:
    """Generate correlation distribution chart data for HTML reports."""
    if not correlation_scores:
        return {'chart_type': 'histogram', 'data': {'bins': [], 'counts': []}}
    
    # Create histogram bins for correlation scores
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    counts, _ = np.histogram(correlation_scores, bins=bins)
    
    return {
        'chart_type': 'histogram',
        'data': {
            'bins': bins.tolist(),
            'counts': counts.tolist()
        }
    }


def _generate_html_report(report_data: dict) -> str:
    """Generate HTML format test report."""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Report - {report_data['report_metadata']['generation_timestamp']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
            .pass {{ color: green; }}
            .fail {{ color: red; }}
            .warning {{ color: orange; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Test Execution Report</h1>
            <p>Generated: {report_data['report_metadata']['generation_timestamp']}</p>
        </div>
        
        <div class="section">
            <h2>Test Summary</h2>
            <div class="metric">Total Tests: {report_data['test_summary'].get('total_tests', 0)}</div>
            <div class="metric pass">Passed: {report_data['test_summary'].get('passed_tests', 0)}</div>
            <div class="metric fail">Failed: {report_data['test_summary'].get('failed_tests', 0)}</div>
            <div class="metric">Pass Rate: {report_data['test_summary'].get('pass_rate', 0):.2%}</div>
        </div>
        
        <div class="section">
            <h2>Performance Analysis</h2>
            <div class="metric">Average Test Time: {report_data.get('performance_analysis', {}).get('average_test_time', 0):.3f}s</div>
            <div class="metric">Peak Memory: {report_data.get('performance_analysis', {}).get('memory_usage_peak', 0):.1f}MB</div>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
    """
    
    for rec in report_data.get('recommendations', []):
        priority_class = rec.get('priority', 'MEDIUM').lower()
        html_template += f'<li class="{priority_class}">[{rec.get("priority", "MEDIUM")}] {rec.get("description", "")}</li>'
    
    html_template += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html_template


def _generate_text_report(report_data: dict) -> str:
    """Generate plain text format test report."""
    text_report = f"""
TEST EXECUTION REPORT
=====================
Generated: {report_data['report_metadata']['generation_timestamp']}

TEST SUMMARY
------------
Total Tests: {report_data['test_summary'].get('total_tests', 0)}
Passed Tests: {report_data['test_summary'].get('passed_tests', 0)}
Failed Tests: {report_data['test_summary'].get('failed_tests', 0)}
Pass Rate: {report_data['test_summary'].get('pass_rate', 0):.2%}
Execution Time: {report_data['test_summary'].get('execution_time', 0):.3f}s

PERFORMANCE ANALYSIS
-------------------
Average Test Time: {report_data.get('performance_analysis', {}).get('average_test_time', 0):.3f}s
Peak Memory Usage: {report_data.get('performance_analysis', {}).get('memory_usage_peak', 0):.1f}MB
Performance Target Met: {report_data.get('performance_analysis', {}).get('meets_performance_targets', 'Unknown')}

RECOMMENDATIONS
---------------
"""
    
    for i, rec in enumerate(report_data.get('recommendations', []), 1):
        text_report += f"{i}. [{rec.get('priority', 'MEDIUM')}] {rec.get('description', '')}\n"
        text_report += f"   Action: {rec.get('action', 'No action specified')}\n\n"
    
    return text_report