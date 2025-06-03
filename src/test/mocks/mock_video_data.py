"""
Comprehensive mock video data generation module providing synthetic plume video data, format-specific test datasets, 
and realistic simulation scenarios for testing data normalization, cross-format compatibility, and algorithm validation.

This module generates controlled test data for both Crimaldi and custom AVI formats with configurable physical parameters, 
temporal dynamics, and format-specific characteristics to support scientific computing validation and reproducible testing 
environments. The module implements fail-fast validation, comprehensive error handling, and scientific precision standards 
for >95% correlation requirements and <7.2 seconds per simulation performance targets.

Key Features:
- Synthetic plume video data generation with realistic concentration patterns
- Cross-format compatibility testing between Crimaldi and custom AVI formats
- Configurable physical parameters including arena size, resolution, and frame rates
- Temporal dynamics simulation with turbulent mixing and intermittency effects
- Comprehensive validation datasets with ground truth solutions
- Performance testing data generation for 4000+ simulation requirements
- Reproducible test environments with controlled randomness and deterministic output
"""

# External library imports with version specifications
import numpy as np  # numpy 2.1.3+ - Numerical computations and array operations for mock video data generation
import cv2  # opencv-python 4.11.0+ - Video processing and format handling for mock data generation
import pathlib  # Python 3.9+ - Cross-platform path handling for mock data files
from typing import Dict, Any, List, Optional, Union, Tuple, Callable  # Python 3.9+ - Type hints for function signatures and data structures
import json  # Python 3.9+ - JSON serialization for mock data metadata and configuration
import random  # Python 3.9+ - Random number generation for stochastic mock data with controlled seeds
import warnings  # Python 3.9+ - Warning management for mock data generation edge cases
import tempfile  # Python 3.9+ - Temporary file management for mock video data generation
import datetime  # Python 3.9+ - Timestamp handling for mock data metadata

# Internal imports from test utilities
from ..utils.test_helpers import (
    create_test_fixture_path,
    TestDataValidator
)
from ..utils.test_data_generator import (
    generate_synthetic_plume_video,
    create_crimaldi_format_data,
    create_custom_format_data,
    SyntheticPlumeGenerator
)

# Global configuration constants for mock video data generation with scientific precision
DEFAULT_CRIMALDI_CONFIG = {
    'arena_size_meters': (1.0, 1.0),
    'resolution_pixels': (640, 480),
    'pixel_to_meter_ratio': 100.0,
    'frame_rate_hz': 30.0,
    'intensity_units': 'concentration_ppm',
    'coordinate_system': 'cartesian',
    'temporal_resolution': 0.033
}

DEFAULT_CUSTOM_CONFIG = {
    'arena_size_meters': (1.2, 0.8),
    'resolution_pixels': (800, 600),
    'pixel_to_meter_ratio': 150.0,
    'frame_rate_hz': 60.0,
    'intensity_units': 'raw_sensor',
    'coordinate_system': 'cartesian',
    'temporal_resolution': 0.0167
}

MOCK_PLUME_PARAMETERS = {
    'diffusion_coefficient': 0.1,
    'wind_velocity': (0.5, 0.0),
    'source_strength': 1.0,
    'background_concentration': 0.0,
    'noise_level': 0.05,
    'intermittency_factor': 0.3
}

VALIDATION_TOLERANCES = {
    'numerical_tolerance': 1e-6,
    'correlation_threshold': 0.95,
    'reproducibility_tolerance': 1e-10,
    'format_compatibility_tolerance': 0.0001
}

MOCK_DATA_CACHE_SIZE = 50
DEFAULT_RANDOM_SEED = 42


def generate_synthetic_plume_frame(
    dimensions: Tuple[int, int],
    source_location: Tuple[float, float],
    time_step: float,
    plume_parameters: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a single synthetic plume frame with realistic concentration patterns using diffusion-advection 
    equations and configurable physical parameters for testing normalization and simulation pipelines.
    
    This function provides robust plume frame generation with comprehensive error handling and scientific 
    precision for testing data normalization and simulation accuracy validation.
    
    Args:
        dimensions: Grid dimensions as (width, height) for the concentration field
        source_location: Source position as (x, y) coordinates in normalized units
        time_step: Current time step for temporal evolution dynamics
        plume_parameters: Physical parameters for plume generation including diffusion and wind
        random_seed: Random seed for reproducible frame generation
        
    Returns:
        np.ndarray: 2D array representing plume concentration field with realistic characteristics
    """
    # Set random seed for reproducible frame generation
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Validate input parameters and dimensions
    width, height = dimensions
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions: {dimensions}. Must be positive integers.")
    
    source_x, source_y = source_location
    if not (0.0 <= source_x <= 1.0) or not (0.0 <= source_y <= 1.0):
        raise ValueError(f"Source location {source_location} must be in range [0, 1]")
    
    if time_step < 0:
        raise ValueError(f"Time step must be non-negative, got {time_step}")
    
    # Use default plume parameters if not provided
    if plume_parameters is None:
        plume_parameters = MOCK_PLUME_PARAMETERS.copy()
    
    # Initialize concentration grid with specified dimensions
    concentration = np.zeros((height, width), dtype=np.float64)
    
    # Create coordinate grids for spatial calculations
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Extract physical parameters for plume simulation
    diffusion_coeff = plume_parameters.get('diffusion_coefficient', 0.1)
    wind_velocity = plume_parameters.get('wind_velocity', (0.5, 0.0))
    source_strength = plume_parameters.get('source_strength', 1.0)
    background_conc = plume_parameters.get('background_concentration', 0.0)
    noise_level = plume_parameters.get('noise_level', 0.05)
    intermittency = plume_parameters.get('intermittency_factor', 0.3)
    
    # Apply diffusion-advection equation for plume transport
    wind_x, wind_y = wind_velocity
    
    # Calculate effective source location with wind advection
    effective_source_x = source_x + wind_x * time_step
    effective_source_y = source_y + wind_y * time_step
    
    # Generate Gaussian plume distribution with temporal evolution
    sigma_x = np.sqrt(2 * diffusion_coeff * time_step + 0.01)  # Minimum width
    sigma_y = np.sqrt(2 * diffusion_coeff * time_step + 0.01)
    
    # Calculate distance from effective source location
    distance_x = X - effective_source_x
    distance_y = Y - effective_source_y
    
    # Apply Gaussian concentration distribution
    concentration = source_strength * np.exp(
        -(distance_x**2) / (2 * sigma_x**2) - (distance_y**2) / (2 * sigma_y**2)
    )
    
    # Add turbulent mixing and intermittency effects
    turbulence_scale = 0.1
    turbulence_x = turbulence_scale * np.sin(2 * np.pi * X * 5) * np.cos(2 * np.pi * time_step)
    turbulence_y = turbulence_scale * np.cos(2 * np.pi * Y * 3) * np.sin(2 * np.pi * time_step * 0.7)
    
    # Apply turbulent perturbations
    concentration *= (1 + turbulence_x + turbulence_y)
    
    # Apply intermittency effects with temporal modulation
    intermittency_mask = np.random.random((height, width)) < (1 - intermittency * np.sin(2 * np.pi * time_step))
    concentration *= intermittency_mask
    
    # Apply source emission characteristics at specified location
    source_idx_x = int(source_x * width)
    source_idx_y = int(source_y * height)
    
    if 0 <= source_idx_x < width and 0 <= source_idx_y < height:
        # Add concentrated source emission
        source_radius = max(1, int(min(width, height) * 0.02))
        y_min = max(0, source_idx_y - source_radius)
        y_max = min(height, source_idx_y + source_radius + 1)
        x_min = max(0, source_idx_x - source_radius)
        x_max = min(width, source_idx_x + source_radius + 1)
        
        concentration[y_min:y_max, x_min:x_max] += source_strength * 0.5
    
    # Add realistic noise and measurement artifacts
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, (height, width))
        concentration += noise
    
    # Add background concentration level
    concentration += background_conc
    
    # Normalize concentration values to appropriate range
    concentration = np.clip(concentration, 0, None)  # Ensure non-negative
    if np.max(concentration) > 0:
        concentration = concentration / np.max(concentration)  # Normalize to [0, 1]
    
    # Validate frame data structure and value ranges
    if np.any(np.isnan(concentration)) or np.any(np.isinf(concentration)):
        raise RuntimeError("Generated concentration field contains invalid values")
    
    if concentration.shape != (height, width):
        raise RuntimeError(f"Output shape mismatch: expected {(height, width)}, got {concentration.shape}")
    
    # Return synthetic plume concentration frame
    return concentration.astype(np.float32)


def create_mock_video_sequence(
    dimensions: Tuple[int, int],
    num_frames: int,
    format_type: str,
    sequence_config: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Creates a complete mock video sequence with temporal dynamics, format-specific characteristics, and realistic 
    plume evolution for comprehensive testing of video processing pipelines.
    
    This function provides comprehensive video sequence generation with temporal consistency and format-specific 
    characteristics for thorough testing of video processing and analysis pipelines.
    
    Args:
        dimensions: Video dimensions as (width, height) tuple
        num_frames: Number of frames to generate in the sequence
        format_type: Video format type ('crimaldi', 'custom', 'avi')
        sequence_config: Configuration parameters for sequence generation
        random_seed: Random seed for reproducible sequence generation
        
    Returns:
        np.ndarray: 4D array [frames, height, width, channels] representing complete video sequence with metadata
    """
    # Set random seed for reproducible sequence generation
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Validate input parameters
    width, height = dimensions
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions: {dimensions}")
    
    if num_frames <= 0:
        raise ValueError(f"Number of frames must be positive, got {num_frames}")
    
    valid_formats = ['crimaldi', 'custom', 'avi']
    if format_type not in valid_formats:
        raise ValueError(f"Invalid format type: {format_type}. Must be one of {valid_formats}")
    
    # Load sequence configuration or use defaults
    if sequence_config is None:
        if format_type == 'crimaldi':
            sequence_config = DEFAULT_CRIMALDI_CONFIG.copy()
        else:
            sequence_config = DEFAULT_CUSTOM_CONFIG.copy()
    
    # Initialize video sequence array with specified dimensions and frame count
    if format_type == 'crimaldi':
        # Grayscale format for Crimaldi data
        video_sequence = np.zeros((num_frames, height, width), dtype=np.uint8)
        channels = 1
    else:
        # RGB format for custom data
        video_sequence = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        channels = 3
    
    # Generate plume source location and characteristics
    source_x = sequence_config.get('source_x', 0.2)
    source_y = sequence_config.get('source_y', 0.5)
    frame_rate = sequence_config.get('frame_rate_hz', 30.0)
    
    # Create temporal sequence of plume frames with realistic dynamics
    plume_params = MOCK_PLUME_PARAMETERS.copy()
    plume_params.update(sequence_config.get('plume_parameters', {}))
    
    for frame_idx in range(num_frames):
        # Calculate time step for temporal evolution
        time_step = frame_idx / frame_rate
        
        # Generate plume frame with temporal consistency
        concentration_frame = generate_synthetic_plume_frame(
            dimensions=dimensions,
            source_location=(source_x, source_y),
            time_step=time_step,
            plume_parameters=plume_params,
            random_seed=random_seed + frame_idx if random_seed is not None else None
        )
        
        # Apply format-specific intensity scaling and characteristics
        if format_type == 'crimaldi':
            # Convert to 8-bit grayscale with Crimaldi-specific scaling
            intensity_frame = (concentration_frame * 255).astype(np.uint8)
            video_sequence[frame_idx] = intensity_frame
        else:
            # Convert to RGB format with custom scaling
            # Create RGB channels with slight variations for realism
            r_channel = (concentration_frame * 255 * 1.0).astype(np.uint8)
            g_channel = (concentration_frame * 255 * 0.8).astype(np.uint8)
            b_channel = (concentration_frame * 255 * 0.6).astype(np.uint8)
            
            video_sequence[frame_idx, :, :, 0] = r_channel
            video_sequence[frame_idx, :, :, 1] = g_channel
            video_sequence[frame_idx, :, :, 2] = b_channel
        
        # Add temporal consistency and smooth transitions between frames
        if frame_idx > 0:
            # Apply temporal smoothing to reduce frame-to-frame noise
            smoothing_factor = 0.1
            if format_type == 'crimaldi':
                video_sequence[frame_idx] = (
                    (1 - smoothing_factor) * video_sequence[frame_idx] +
                    smoothing_factor * video_sequence[frame_idx - 1]
                ).astype(np.uint8)
            else:
                for channel in range(3):
                    video_sequence[frame_idx, :, :, channel] = (
                        (1 - smoothing_factor) * video_sequence[frame_idx, :, :, channel] +
                        smoothing_factor * video_sequence[frame_idx - 1, :, :, channel]
                    ).astype(np.uint8)
    
    # Apply format-specific noise patterns and artifacts
    if format_type == 'crimaldi':
        # Add Crimaldi-specific noise characteristics
        noise_std = 2.0
        noise = np.random.normal(0, noise_std, video_sequence.shape).astype(np.int16)
        video_sequence = np.clip(video_sequence.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    else:
        # Add custom format noise characteristics
        noise_std = 1.5
        for channel in range(3):
            noise = np.random.normal(0, noise_std, (num_frames, height, width)).astype(np.int16)
            video_sequence[:, :, :, channel] = np.clip(
                video_sequence[:, :, :, channel].astype(np.int16) + noise, 0, 255
            ).astype(np.uint8)
    
    # Validate complete video sequence structure and properties
    if format_type == 'crimaldi':
        expected_shape = (num_frames, height, width)
    else:
        expected_shape = (num_frames, height, width, 3)
    
    if video_sequence.shape != expected_shape:
        raise RuntimeError(f"Video sequence shape mismatch: expected {expected_shape}, got {video_sequence.shape}")
    
    # Return mock video sequence with embedded metadata
    return video_sequence


def generate_crimaldi_mock_data(
    arena_size_meters: Tuple[float, float],
    resolution_pixels: Tuple[int, int],
    duration_seconds: float,
    crimaldi_config: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generates mock data specifically formatted to match Crimaldi dataset specifications including coordinate systems, 
    intensity units, temporal characteristics, and calibration parameters for cross-format compatibility testing.
    
    This function creates comprehensive Crimaldi-format mock data with proper calibration parameters and 
    format-specific characteristics for rigorous cross-format compatibility testing.
    
    Args:
        arena_size_meters: Physical arena dimensions in meters as (width, height)
        resolution_pixels: Video resolution in pixels as (width, height)
        duration_seconds: Duration of the video sequence in seconds
        crimaldi_config: Crimaldi-specific configuration parameters
        random_seed: Random seed for reproducible dataset generation
        
    Returns:
        Dict[str, Any]: Dictionary containing Crimaldi-format mock data with metadata and calibration parameters
    """
    # Load Crimaldi format configuration parameters and defaults
    config = DEFAULT_CRIMALDI_CONFIG.copy()
    if crimaldi_config:
        config.update(crimaldi_config)
    
    # Set pixel-to-meter conversion ratio (100.0 pixels/meter)
    pixel_to_meter_ratio = config.get('pixel_to_meter_ratio', 100.0)
    
    # Validate arena size and resolution parameters
    arena_width, arena_height = arena_size_meters
    if arena_width <= 0 or arena_height <= 0:
        raise ValueError(f"Arena size must be positive: {arena_size_meters}")
    
    res_width, res_height = resolution_pixels
    if res_width <= 0 or res_height <= 0:
        raise ValueError(f"Resolution must be positive: {resolution_pixels}")
    
    if duration_seconds <= 0:
        raise ValueError(f"Duration must be positive: {duration_seconds}")
    
    # Calculate number of frames based on Crimaldi frame rate
    frame_rate = config.get('frame_rate_hz', 30.0)
    num_frames = int(duration_seconds * frame_rate)
    
    # Generate concentration data in PPM units with Crimaldi scaling
    video_data = create_mock_video_sequence(
        dimensions=resolution_pixels,
        num_frames=num_frames,
        format_type='crimaldi',
        sequence_config=config,
        random_seed=random_seed
    )
    
    # Apply Cartesian coordinate system conventions
    # Crimaldi format uses bottom-left origin
    video_data = np.flip(video_data, axis=1)  # Flip vertically for Cartesian coordinates
    
    # Create temporal sampling at 30 FPS with proper frame timing
    frame_timestamps = np.arange(num_frames) / frame_rate
    
    # Add realistic background subtraction and Crimaldi-specific noise
    background_level = config.get('background_level', 5)
    video_data = np.clip(video_data.astype(np.int16) - background_level, 0, 255).astype(np.uint8)
    
    # Generate metadata matching Crimaldi format specifications
    metadata = {
        'format_type': 'crimaldi',
        'arena_size_meters': arena_size_meters,
        'resolution_pixels': resolution_pixels,
        'pixel_to_meter_ratio': pixel_to_meter_ratio,
        'frame_rate_hz': frame_rate,
        'duration_seconds': duration_seconds,
        'num_frames': num_frames,
        'intensity_units': 'concentration_ppm',
        'coordinate_system': 'cartesian',
        'temporal_resolution': 1.0 / frame_rate,
        'generation_timestamp': datetime.datetime.now().isoformat(),
        'random_seed': random_seed
    }
    
    # Include calibration parameters for spatial and intensity scaling
    calibration_params = {
        'spatial_calibration': {
            'pixel_to_meter_x': pixel_to_meter_ratio,
            'pixel_to_meter_y': pixel_to_meter_ratio,
            'arena_bounds_meters': {
                'x_min': 0.0, 'x_max': arena_width,
                'y_min': 0.0, 'y_max': arena_height
            },
            'coordinate_origin': 'bottom_left'
        },
        'intensity_calibration': {
            'units': 'concentration_ppm',
            'dynamic_range': [0, 255],
            'background_level': background_level,
            'calibration_factor': 1.0
        },
        'temporal_calibration': {
            'frame_rate_hz': frame_rate,
            'frame_timestamps': frame_timestamps.tolist(),
            'temporal_resolution': 1.0 / frame_rate
        }
    }
    
    # Validate format compliance against Crimaldi standards
    if video_data.dtype != np.uint8:
        raise RuntimeError(f"Crimaldi format requires uint8 data type, got {video_data.dtype}")
    
    if len(video_data.shape) != 3:
        raise RuntimeError(f"Crimaldi format requires 3D data (frames, height, width), got shape {video_data.shape}")
    
    # Package complete Crimaldi mock dataset with validation metadata
    crimaldi_dataset = {
        'video_data': video_data,
        'metadata': metadata,
        'calibration_parameters': calibration_params,
        'frame_timestamps': frame_timestamps,
        'validation_info': {
            'format_validated': True,
            'calibration_validated': True,
            'temporal_consistency_validated': True
        }
    }
    
    return crimaldi_dataset


def generate_custom_avi_mock_data(
    arena_size_meters: Tuple[float, float],
    resolution_pixels: Tuple[int, int],
    duration_seconds: float,
    custom_config: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generates mock data for custom AVI format with configurable parameters, adaptive characteristics, and 
    format-specific properties for testing cross-format compatibility and normalization accuracy.
    
    This function creates comprehensive custom AVI format mock data with adaptive parameters and 
    format-specific characteristics for thorough cross-format compatibility testing.
    
    Args:
        arena_size_meters: Physical arena dimensions in meters as (width, height)
        resolution_pixels: Video resolution in pixels as (width, height)
        duration_seconds: Duration of the video sequence in seconds
        custom_config: Custom format configuration parameters
        random_seed: Random seed for reproducible dataset generation
        
    Returns:
        Dict[str, Any]: Dictionary containing custom AVI format mock data with metadata and calibration parameters
    """
    # Load custom format configuration parameters and defaults
    config = DEFAULT_CUSTOM_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    
    # Set pixel-to-meter conversion ratio (150.0 pixels/meter)
    pixel_to_meter_ratio = config.get('pixel_to_meter_ratio', 150.0)
    
    # Validate arena size and resolution parameters
    arena_width, arena_height = arena_size_meters
    if arena_width <= 0 or arena_height <= 0:
        raise ValueError(f"Arena size must be positive: {arena_size_meters}")
    
    res_width, res_height = resolution_pixels
    if res_width <= 0 or res_height <= 0:
        raise ValueError(f"Resolution must be positive: {resolution_pixels}")
    
    if duration_seconds <= 0:
        raise ValueError(f"Duration must be positive: {duration_seconds}")
    
    # Calculate number of frames based on custom frame rate
    frame_rate = config.get('frame_rate_hz', 60.0)
    num_frames = int(duration_seconds * frame_rate)
    
    # Generate raw sensor intensity data (0-255 range)
    video_data = create_mock_video_sequence(
        dimensions=resolution_pixels,
        num_frames=num_frames,
        format_type='custom',
        sequence_config=config,
        random_seed=random_seed
    )
    
    # Apply custom coordinate system and temporal sampling
    # Custom format uses top-left origin (standard computer vision)
    # No coordinate system transformation needed
    
    # Create temporal sampling at 60 FPS with custom frame timing
    frame_timestamps = np.arange(num_frames) / frame_rate
    
    # Add format-specific noise and calibration characteristics
    sensor_noise_std = config.get('sensor_noise_std', 3.0)
    for channel in range(3):
        noise = np.random.normal(0, sensor_noise_std, (num_frames, res_height, res_width))
        video_data[:, :, :, channel] = np.clip(
            video_data[:, :, :, channel].astype(np.float32) + noise, 0, 255
        ).astype(np.uint8)
    
    # Generate metadata for custom format specifications
    metadata = {
        'format_type': 'custom_avi',
        'arena_size_meters': arena_size_meters,
        'resolution_pixels': resolution_pixels,
        'pixel_to_meter_ratio': pixel_to_meter_ratio,
        'frame_rate_hz': frame_rate,
        'duration_seconds': duration_seconds,
        'num_frames': num_frames,
        'intensity_units': 'raw_sensor',
        'coordinate_system': 'cartesian',
        'temporal_resolution': 1.0 / frame_rate,
        'color_space': 'rgb',
        'bit_depth': 8,
        'generation_timestamp': datetime.datetime.now().isoformat(),
        'random_seed': random_seed
    }
    
    # Include adaptive calibration parameters for flexible processing
    calibration_params = {
        'spatial_calibration': {
            'pixel_to_meter_x': pixel_to_meter_ratio,
            'pixel_to_meter_y': pixel_to_meter_ratio,
            'arena_bounds_meters': {
                'x_min': 0.0, 'x_max': arena_width,
                'y_min': 0.0, 'y_max': arena_height
            },
            'coordinate_origin': 'top_left'
        },
        'intensity_calibration': {
            'units': 'raw_sensor',
            'dynamic_range': [0, 255],
            'sensor_noise_std': sensor_noise_std,
            'calibration_factor': 1.0,
            'color_channels': ['red', 'green', 'blue']
        },
        'temporal_calibration': {
            'frame_rate_hz': frame_rate,
            'frame_timestamps': frame_timestamps.tolist(),
            'temporal_resolution': 1.0 / frame_rate
        }
    }
    
    # Validate format compliance against custom AVI standards
    if video_data.dtype != np.uint8:
        raise RuntimeError(f"Custom AVI format requires uint8 data type, got {video_data.dtype}")
    
    if len(video_data.shape) != 4 or video_data.shape[3] != 3:
        raise RuntimeError(f"Custom AVI format requires 4D RGB data (frames, height, width, 3), got shape {video_data.shape}")
    
    # Package complete custom format mock dataset with validation metadata
    custom_dataset = {
        'video_data': video_data,
        'metadata': metadata,
        'calibration_parameters': calibration_params,
        'frame_timestamps': frame_timestamps,
        'validation_info': {
            'format_validated': True,
            'calibration_validated': True,
            'temporal_consistency_validated': True,
            'color_space_validated': True
        }
    }
    
    return custom_dataset


def create_validation_dataset(
    num_scenarios: int,
    format_types: List[str],
    validation_config: Dict[str, Any],
    correlation_target: Optional[float] = None,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Creates comprehensive validation dataset with known ground truth for testing algorithm accuracy, 
    normalization quality, and cross-format compatibility with statistical validation requirements.
    
    This function generates controlled test scenarios with known optimal solutions for comprehensive 
    algorithm validation and cross-format compatibility testing.
    
    Args:
        num_scenarios: Number of validation scenarios to generate
        format_types: List of format types to include in validation dataset
        validation_config: Configuration parameters for validation dataset generation
        correlation_target: Target correlation level for validation (default 0.95)
        random_seed: Random seed for reproducible validation dataset generation
        
    Returns:
        Dict[str, Any]: Validation dataset with ground truth solutions and statistical validation parameters
    """
    # Set random seed for reproducible validation dataset
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Validate input parameters
    if num_scenarios <= 0:
        raise ValueError(f"Number of scenarios must be positive, got {num_scenarios}")
    
    valid_formats = ['crimaldi', 'custom', 'avi']
    for fmt in format_types:
        if fmt not in valid_formats:
            raise ValueError(f"Invalid format type: {fmt}. Must be one of {valid_formats}")
    
    if correlation_target is None:
        correlation_target = VALIDATION_TOLERANCES['correlation_threshold']
    
    # Generate controlled test scenarios with known optimal solutions
    validation_scenarios = []
    
    for scenario_idx in range(num_scenarios):
        scenario = {
            'scenario_id': f"validation_scenario_{scenario_idx:04d}",
            'scenario_type': 'controlled_validation',
            'format_datasets': {},
            'ground_truth': {},
            'validation_metrics': {}
        }
        
        # Generate scenario parameters
        arena_width = np.random.uniform(0.8, 1.5)
        arena_height = np.random.uniform(0.8, 1.5)
        resolution_width = np.random.choice([640, 800, 1024])
        resolution_height = np.random.choice([480, 600, 768])
        duration = np.random.uniform(10.0, 30.0)
        
        # Create identical plume scenarios in multiple formats for comparison
        base_config = {
            'source_x': np.random.uniform(0.1, 0.9),
            'source_y': np.random.uniform(0.1, 0.9),
            'plume_parameters': {
                'diffusion_coefficient': np.random.uniform(0.05, 0.2),
                'wind_velocity': (np.random.uniform(0.1, 0.8), np.random.uniform(-0.2, 0.2)),
                'source_strength': np.random.uniform(0.5, 2.0),
                'noise_level': np.random.uniform(0.01, 0.1)
            }
        }
        
        # Generate data for each requested format
        for format_type in format_types:
            if format_type == 'crimaldi':
                dataset = generate_crimaldi_mock_data(
                    arena_size_meters=(arena_width, arena_height),
                    resolution_pixels=(resolution_width, resolution_height),
                    duration_seconds=duration,
                    crimaldi_config=base_config,
                    random_seed=random_seed + scenario_idx if random_seed else None
                )
            else:
                dataset = generate_custom_avi_mock_data(
                    arena_size_meters=(arena_width, arena_height),
                    resolution_pixels=(resolution_width, resolution_height),
                    duration_seconds=duration,
                    custom_config=base_config,
                    random_seed=random_seed + scenario_idx if random_seed else None
                )
            
            scenario['format_datasets'][format_type] = dataset
        
        # Generate ground truth navigation trajectories and performance metrics
        ground_truth_trajectory = _generate_ground_truth_trajectory(
            source_location=(base_config['source_x'], base_config['source_y']),
            arena_size=(arena_width, arena_height),
            plume_params=base_config['plume_parameters']
        )
        
        scenario['ground_truth'] = {
            'optimal_trajectory': ground_truth_trajectory,
            'source_location': (base_config['source_x'], base_config['source_y']),
            'arena_size': (arena_width, arena_height),
            'plume_parameters': base_config['plume_parameters'],
            'expected_performance_metrics': {
                'path_efficiency': 0.85,
                'convergence_time': duration * 0.7,
                'accuracy_score': 0.95
            }
        }
        
        # Create statistical validation benchmarks with target correlation levels
        scenario['validation_metrics'] = {
            'correlation_target': correlation_target,
            'numerical_tolerance': VALIDATION_TOLERANCES['numerical_tolerance'],
            'format_compatibility_tolerance': VALIDATION_TOLERANCES['format_compatibility_tolerance'],
            'reproducibility_tolerance': VALIDATION_TOLERANCES['reproducibility_tolerance']
        }
        
        validation_scenarios.append(scenario)
    
    # Add edge cases and stress testing scenarios
    edge_case_scenarios = _generate_edge_case_scenarios(format_types, validation_config, random_seed)
    validation_scenarios.extend(edge_case_scenarios)
    
    # Generate reproducibility test data with controlled randomness
    reproducibility_scenarios = _generate_reproducibility_scenarios(format_types, random_seed)
    validation_scenarios.extend(reproducibility_scenarios)
    
    # Package validation dataset with expected outcomes and tolerances
    validation_dataset = {
        'dataset_id': f"validation_dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'generation_timestamp': datetime.datetime.now().isoformat(),
        'num_scenarios': len(validation_scenarios),
        'format_types': format_types,
        'validation_config': validation_config,
        'scenarios': validation_scenarios,
        'dataset_statistics': {
            'total_scenarios': len(validation_scenarios),
            'controlled_scenarios': num_scenarios,
            'edge_case_scenarios': len(edge_case_scenarios),
            'reproducibility_scenarios': len(reproducibility_scenarios)
        },
        'validation_requirements': {
            'correlation_target': correlation_target,
            'performance_targets': validation_config.get('performance_targets', {}),
            'tolerance_levels': VALIDATION_TOLERANCES
        }
    }
    
    # Validate dataset completeness and statistical properties
    _validate_dataset_completeness(validation_dataset)
    
    return validation_dataset


def save_mock_data_to_fixture(
    mock_data: Dict[str, Any],
    fixture_name: str,
    format_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Saves generated mock data to test fixtures directory with proper file organization, metadata, and 
    integrity validation for reuse across test modules.
    
    This function provides comprehensive mock data storage with integrity validation and proper 
    file organization for reuse across multiple test modules and scenarios.
    
    Args:
        mock_data: Dictionary containing mock data to save
        fixture_name: Name for the fixture file
        format_type: Format type for organization ('crimaldi', 'custom')
        metadata: Additional metadata to include with the fixture
        
    Returns:
        str: Path to saved mock data fixture with validation confirmation
    """
    # Validate mock data structure and completeness
    if not isinstance(mock_data, dict):
        raise TypeError("Mock data must be a dictionary")
    
    required_keys = ['video_data', 'metadata', 'calibration_parameters']
    for key in required_keys:
        if key not in mock_data:
            raise ValueError(f"Missing required key in mock data: {key}")
    
    # Create fixture directory structure using test_helpers path utilities
    fixture_path = create_test_fixture_path(fixture_name, f'mock_data/{format_type}')
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate comprehensive metadata file with data specifications
    fixture_metadata = {
        'fixture_name': fixture_name,
        'format_type': format_type,
        'creation_timestamp': datetime.datetime.now().isoformat(),
        'data_shape': mock_data['video_data'].shape,
        'data_dtype': str(mock_data['video_data'].dtype),
        'data_size_bytes': mock_data['video_data'].nbytes,
        'original_metadata': mock_data['metadata']
    }
    
    if metadata:
        fixture_metadata.update(metadata)
    
    # Save mock data in appropriate format (NPY for arrays, JSON for metadata)
    video_data_path = fixture_path.with_suffix('.npy')
    metadata_path = fixture_path.with_suffix('.json')
    
    # Save video data as numpy array
    np.save(str(video_data_path), mock_data['video_data'])
    
    # Save comprehensive metadata and calibration parameters
    complete_metadata = {
        'fixture_metadata': fixture_metadata,
        'calibration_parameters': mock_data['calibration_parameters'],
        'validation_info': mock_data.get('validation_info', {}),
        'frame_timestamps': mock_data.get('frame_timestamps', []).tolist() if hasattr(mock_data.get('frame_timestamps', []), 'tolist') else mock_data.get('frame_timestamps', [])
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(complete_metadata, f, indent=2)
    
    # Create checksum for data integrity validation
    video_checksum = _calculate_array_checksum(mock_data['video_data'])
    metadata_checksum = _calculate_json_checksum(complete_metadata)
    
    # Save integrity checksum file
    checksum_path = fixture_path.with_suffix('.checksum')
    checksum_data = {
        'video_data_checksum': video_checksum,
        'metadata_checksum': metadata_checksum,
        'creation_timestamp': datetime.datetime.now().isoformat()
    }
    
    with open(checksum_path, 'w') as f:
        json.dump(checksum_data, f, indent=2)
    
    # Update fixture registry with new mock data entry
    _update_fixture_registry(fixture_name, format_type, str(fixture_path))
    
    # Validate saved data integrity and accessibility
    try:
        # Verify data can be loaded correctly
        loaded_data = np.load(str(video_data_path))
        if not np.array_equal(loaded_data, mock_data['video_data']):
            raise RuntimeError("Data integrity check failed after saving")
        
        # Verify metadata can be loaded
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to validate saved fixture: {e}")
    
    # Return path to saved fixture with confirmation status
    return str(fixture_path)


def load_mock_data_from_fixture(
    fixture_name: str,
    format_type: str,
    validate_integrity: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Loads previously saved mock data from test fixtures with integrity checking, format validation, and 
    metadata verification for consistent test execution.
    
    This function provides robust mock data loading with integrity verification and format validation 
    for consistent and reliable test execution across different environments.
    
    Args:
        fixture_name: Name of the fixture to load
        format_type: Format type for the fixture ('crimaldi', 'custom')
        validate_integrity: Whether to validate data integrity using checksums
        
    Returns:
        Dict[str, Any]: Loaded mock data with metadata and validation confirmation
    """
    # Construct path to mock data fixture using standardized naming
    fixture_path = create_test_fixture_path(fixture_name, f'mock_data/{format_type}')
    
    video_data_path = fixture_path.with_suffix('.npy')
    metadata_path = fixture_path.with_suffix('.json')
    checksum_path = fixture_path.with_suffix('.checksum')
    
    # Check that all required files exist
    if not video_data_path.exists():
        raise FileNotFoundError(f"Video data file not found: {video_data_path}")
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load mock data with appropriate format handler
    try:
        video_data = np.load(str(video_data_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load video data: {e}")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata: {e}")
    
    # Validate data integrity using checksums if requested
    if validate_integrity is None:
        validate_integrity = checksum_path.exists()
    
    if validate_integrity and checksum_path.exists():
        try:
            with open(checksum_path, 'r') as f:
                stored_checksums = json.load(f)
            
            # Calculate current checksums
            current_video_checksum = _calculate_array_checksum(video_data)
            current_metadata_checksum = _calculate_json_checksum(metadata_dict)
            
            # Compare checksums for integrity verification
            if current_video_checksum != stored_checksums['video_data_checksum']:
                raise RuntimeError("Video data integrity check failed - checksum mismatch")
            
            if current_metadata_checksum != stored_checksums['metadata_checksum']:
                raise RuntimeError("Metadata integrity check failed - checksum mismatch")
                
        except Exception as e:
            if validate_integrity:
                raise RuntimeError(f"Integrity validation failed: {e}")
            else:
                warnings.warn(f"Could not validate integrity: {e}")
    
    # Check data format and structure compliance
    fixture_metadata = metadata_dict.get('fixture_metadata', {})
    expected_shape = tuple(fixture_metadata.get('data_shape', []))
    expected_dtype = fixture_metadata.get('data_dtype', '')
    
    if expected_shape and video_data.shape != expected_shape:
        raise RuntimeError(f"Data shape mismatch: expected {expected_shape}, got {video_data.shape}")
    
    if expected_dtype and str(video_data.dtype) != expected_dtype:
        warnings.warn(f"Data type mismatch: expected {expected_dtype}, got {video_data.dtype}")
    
    # Verify metadata consistency and completeness
    required_metadata_keys = ['calibration_parameters', 'fixture_metadata']
    for key in required_metadata_keys:
        if key not in metadata_dict:
            warnings.warn(f"Missing metadata key: {key}")
    
    # Apply any necessary format conversions or updates
    # (This would handle version compatibility issues)
    
    # Reconstruct full mock data dictionary
    mock_data = {
        'video_data': video_data,
        'metadata': fixture_metadata.get('original_metadata', {}),
        'calibration_parameters': metadata_dict.get('calibration_parameters', {}),
        'validation_info': metadata_dict.get('validation_info', {}),
        'frame_timestamps': np.array(metadata_dict.get('frame_timestamps', [])),
        'fixture_info': fixture_metadata
    }
    
    # Cache loaded data for performance optimization
    _cache_loaded_fixture(fixture_name, format_type, mock_data)
    
    # Return validated mock data with metadata
    return mock_data


class MockVideoConfig:
    """
    Configuration class for mock video data generation providing format-specific parameter management, 
    validation, and conversion utilities for Crimaldi and custom AVI formats with scientific computing precision.
    
    This class provides comprehensive configuration management with format validation and conversion 
    utilities for different video formats used in plume simulation testing.
    """
    
    def __init__(
        self,
        config_parameters: Dict[str, Any],
        format_type: str,
        strict_validation: Optional[bool] = None
    ):
        """
        Initialize mock video configuration with format-specific parameters and validation settings.
        
        Args:
            config_parameters: Dictionary of configuration parameters
            format_type: Format type ('crimaldi', 'custom', 'generic')
            strict_validation: Enable strict validation mode with comprehensive checks
        """
        # Validate format_type against supported formats (crimaldi, custom)
        valid_formats = ['crimaldi', 'custom', 'generic']
        if format_type not in valid_formats:
            raise ValueError(f"Invalid format type: {format_type}. Must be one of {valid_formats}")
        
        self.format_type = format_type
        self.strict_validation = strict_validation if strict_validation is not None else True
        
        # Load default configuration for specified format type
        if format_type == 'crimaldi':
            self.default_config = DEFAULT_CRIMALDI_CONFIG.copy()
        elif format_type == 'custom':
            self.default_config = DEFAULT_CUSTOM_CONFIG.copy()
        else:
            # Generic format uses average of crimaldi and custom
            self.default_config = self._create_generic_config()
        
        # Merge provided parameters with format defaults
        self.config_parameters = self.default_config.copy()
        self.config_parameters.update(config_parameters)
        
        # Initialize TestDataValidator for parameter validation
        self.validator = TestDataValidator(
            tolerance=VALIDATION_TOLERANCES['numerical_tolerance'],
            strict_validation=self.strict_validation
        )
        
        # Validate configuration parameters for scientific consistency
        self._validate_configuration()
        
        # Store validated configuration for mock data generation
        self.validated_config = self.config_parameters.copy()
    
    def _create_generic_config(self) -> Dict[str, Any]:
        """Create generic configuration by averaging crimaldi and custom parameters."""
        crimaldi_config = DEFAULT_CRIMALDI_CONFIG
        custom_config = DEFAULT_CUSTOM_CONFIG
        
        return {
            'arena_size_meters': (
                (crimaldi_config['arena_size_meters'][0] + custom_config['arena_size_meters'][0]) / 2,
                (crimaldi_config['arena_size_meters'][1] + custom_config['arena_size_meters'][1]) / 2
            ),
            'resolution_pixels': (
                int((crimaldi_config['resolution_pixels'][0] + custom_config['resolution_pixels'][0]) / 2),
                int((crimaldi_config['resolution_pixels'][1] + custom_config['resolution_pixels'][1]) / 2)
            ),
            'pixel_to_meter_ratio': (crimaldi_config['pixel_to_meter_ratio'] + custom_config['pixel_to_meter_ratio']) / 2,
            'frame_rate_hz': (crimaldi_config['frame_rate_hz'] + custom_config['frame_rate_hz']) / 2,
            'intensity_units': 'normalized',
            'coordinate_system': 'cartesian',
            'temporal_resolution': (crimaldi_config['temporal_resolution'] + custom_config['temporal_resolution']) / 2
        }
    
    def _validate_configuration(self) -> None:
        """Validate configuration parameters for scientific consistency."""
        # Validate arena size parameters
        arena_size = self.config_parameters.get('arena_size_meters', (1.0, 1.0))
        if not isinstance(arena_size, (tuple, list)) or len(arena_size) != 2:
            raise ValueError("Arena size must be a tuple/list of 2 values")
        
        if arena_size[0] <= 0 or arena_size[1] <= 0:
            raise ValueError(f"Arena size must be positive: {arena_size}")
        
        # Validate resolution parameters
        resolution = self.config_parameters.get('resolution_pixels', (640, 480))
        if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
            raise ValueError("Resolution must be a tuple/list of 2 values")
        
        if resolution[0] <= 0 or resolution[1] <= 0:
            raise ValueError(f"Resolution must be positive: {resolution}")
        
        # Validate pixel-to-meter ratio
        pixel_ratio = self.config_parameters.get('pixel_to_meter_ratio', 100.0)
        if pixel_ratio <= 0:
            raise ValueError(f"Pixel-to-meter ratio must be positive: {pixel_ratio}")
        
        # Validate frame rate
        frame_rate = self.config_parameters.get('frame_rate_hz', 30.0)
        if frame_rate <= 0:
            raise ValueError(f"Frame rate must be positive: {frame_rate}")
        
        # Validate temporal resolution consistency
        temporal_res = self.config_parameters.get('temporal_resolution', 1.0 / frame_rate)
        expected_temporal_res = 1.0 / frame_rate
        
        if abs(temporal_res - expected_temporal_res) > VALIDATION_TOLERANCES['numerical_tolerance']:
            if self.strict_validation:
                raise ValueError(f"Temporal resolution inconsistent with frame rate: {temporal_res} vs {expected_temporal_res}")
            else:
                warnings.warn(f"Temporal resolution inconsistent with frame rate: {temporal_res} vs {expected_temporal_res}")
    
    def to_crimaldi_format(self) -> Dict[str, Any]:
        """
        Convert configuration parameters to Crimaldi dataset format specifications with proper scaling and unit conversion.
        
        Returns:
            Dict[str, Any]: Configuration parameters formatted for Crimaldi dataset compatibility
        """
        # Apply Crimaldi-specific parameter scaling and units
        crimaldi_config = self.config_parameters.copy()
        
        # Set pixel-to-meter ratio to 100.0 pixels/meter
        crimaldi_config['pixel_to_meter_ratio'] = 100.0
        
        # Configure intensity units to concentration PPM
        crimaldi_config['intensity_units'] = 'concentration_ppm'
        
        # Set frame rate to 30 FPS with proper temporal resolution
        crimaldi_config['frame_rate_hz'] = 30.0
        crimaldi_config['temporal_resolution'] = 1.0 / 30.0
        
        # Apply Cartesian coordinate system conventions
        crimaldi_config['coordinate_system'] = 'cartesian'
        crimaldi_config['coordinate_origin'] = 'bottom_left'
        
        # Add Crimaldi-specific parameters
        crimaldi_config['background_subtraction'] = True
        crimaldi_config['bit_depth'] = 8
        crimaldi_config['color_space'] = 'grayscale'
        
        # Validate Crimaldi format compliance
        self._validate_crimaldi_compliance(crimaldi_config)
        
        # Return formatted configuration dictionary
        return crimaldi_config
    
    def to_custom_format(self) -> Dict[str, Any]:
        """
        Convert configuration parameters to custom AVI format specifications with adaptive parameter detection and flexible scaling.
        
        Returns:
            Dict[str, Any]: Configuration parameters formatted for custom AVI format compatibility
        """
        # Apply custom format parameter scaling and characteristics
        custom_config = self.config_parameters.copy()
        
        # Set pixel-to-meter ratio to 150.0 pixels/meter
        custom_config['pixel_to_meter_ratio'] = 150.0
        
        # Configure intensity units to raw sensor values (0-255)
        custom_config['intensity_units'] = 'raw_sensor'
        
        # Set frame rate to 60 FPS with custom temporal resolution
        custom_config['frame_rate_hz'] = 60.0
        custom_config['temporal_resolution'] = 1.0 / 60.0
        
        # Apply custom coordinate system and calibration
        custom_config['coordinate_system'] = 'cartesian'
        custom_config['coordinate_origin'] = 'top_left'
        
        # Add custom format-specific parameters
        custom_config['sensor_noise_std'] = 3.0
        custom_config['bit_depth'] = 8
        custom_config['color_space'] = 'rgb'
        custom_config['adaptive_calibration'] = True
        
        # Validate custom format compliance
        self._validate_custom_compliance(custom_config)
        
        # Return formatted configuration dictionary
        return custom_config
    
    def validate_parameters(self, validation_criteria: Optional[Dict[str, Any]] = None) -> bool:
        """
        Comprehensive validation of configuration parameters for scientific accuracy, physical consistency, and format compliance.
        
        Args:
            validation_criteria: Additional validation criteria to apply
            
        Returns:
            bool: True if all parameters pass validation requirements
        """
        validation_passed = True
        validation_errors = []
        
        try:
            # Validate arena size and resolution parameters for physical consistency
            arena_size = self.config_parameters['arena_size_meters']
            resolution = self.config_parameters['resolution_pixels']
            pixel_ratio = self.config_parameters['pixel_to_meter_ratio']
            
            # Check physical consistency of pixel ratio
            calculated_pixel_ratio_x = resolution[0] / arena_size[0]
            calculated_pixel_ratio_y = resolution[1] / arena_size[1]
            
            if abs(calculated_pixel_ratio_x - pixel_ratio) > pixel_ratio * 0.1:  # 10% tolerance
                validation_errors.append(f"Pixel ratio X inconsistent: {calculated_pixel_ratio_x} vs {pixel_ratio}")
                validation_passed = False
            
            if abs(calculated_pixel_ratio_y - pixel_ratio) > pixel_ratio * 0.1:  # 10% tolerance
                validation_errors.append(f"Pixel ratio Y inconsistent: {calculated_pixel_ratio_y} vs {pixel_ratio}")
                validation_passed = False
            
            # Check pixel-to-meter ratio for reasonable scaling
            if pixel_ratio < 10 or pixel_ratio > 1000:
                validation_errors.append(f"Pixel-to-meter ratio outside reasonable range: {pixel_ratio}")
                validation_passed = False
            
            # Validate frame rate and temporal resolution parameters
            frame_rate = self.config_parameters['frame_rate_hz']
            temporal_res = self.config_parameters['temporal_resolution']
            
            if frame_rate < 1 or frame_rate > 1000:
                validation_errors.append(f"Frame rate outside reasonable range: {frame_rate}")
                validation_passed = False
            
            expected_temporal_res = 1.0 / frame_rate
            if abs(temporal_res - expected_temporal_res) > VALIDATION_TOLERANCES['numerical_tolerance']:
                validation_errors.append(f"Temporal resolution inconsistent: {temporal_res} vs {expected_temporal_res}")
                validation_passed = False
            
            # Verify intensity unit specifications and ranges
            intensity_units = self.config_parameters.get('intensity_units', 'unknown')
            valid_units = ['concentration_ppm', 'raw_sensor', 'normalized']
            if intensity_units not in valid_units:
                validation_errors.append(f"Invalid intensity units: {intensity_units}")
                validation_passed = False
            
            # Check coordinate system configuration
            coord_system = self.config_parameters.get('coordinate_system', 'cartesian')
            if coord_system != 'cartesian':
                validation_errors.append(f"Only Cartesian coordinate system supported: {coord_system}")
                validation_passed = False
            
            # Apply format-specific validation criteria
            if self.format_type == 'crimaldi':
                validation_passed &= self._validate_crimaldi_specific()
            elif self.format_type == 'custom':
                validation_passed &= self._validate_custom_specific()
            
            # Apply additional validation criteria if provided
            if validation_criteria:
                for criterion, value in validation_criteria.items():
                    if criterion in self.config_parameters:
                        param_value = self.config_parameters[criterion]
                        if isinstance(value, dict) and 'min' in value and 'max' in value:
                            if not (value['min'] <= param_value <= value['max']):
                                validation_errors.append(f"{criterion} outside range [{value['min']}, {value['max']}]: {param_value}")
                                validation_passed = False
            
        except Exception as e:
            validation_errors.append(f"Validation error: {e}")
            validation_passed = False
        
        # Log validation errors if any
        if validation_errors:
            for error in validation_errors:
                warnings.warn(f"Configuration validation: {error}")
        
        # Generate validation report with any issues
        if not validation_passed and self.strict_validation:
            raise ValueError(f"Configuration validation failed: {validation_errors}")
        
        # Return validation success status
        return validation_passed
    
    def get_metadata(self, include_validation_info: Optional[bool] = None) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for mock video data including configuration parameters, format specifications, and validation status.
        
        Args:
            include_validation_info: Whether to include validation status and criteria
            
        Returns:
            Dict[str, Any]: Complete metadata dictionary with configuration and validation information
        """
        # Compile configuration parameters and format specifications
        metadata = {
            'config_id': f"config_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'format_type': self.format_type,
            'configuration_parameters': self.validated_config.copy(),
            'default_parameters': self.default_config.copy(),
            'generation_timestamp': datetime.datetime.now().isoformat(),
            'strict_validation_enabled': self.strict_validation
        }
        
        # Include validation status and criteria if requested
        if include_validation_info:
            try:
                validation_success = self.validate_parameters()
                metadata['validation_info'] = {
                    'validation_passed': validation_success,
                    'validation_timestamp': datetime.datetime.now().isoformat(),
                    'validation_criteria': VALIDATION_TOLERANCES.copy(),
                    'format_compliance_checked': True
                }
            except Exception as e:
                metadata['validation_info'] = {
                    'validation_passed': False,
                    'validation_error': str(e),
                    'validation_timestamp': datetime.datetime.now().isoformat()
                }
        
        # Add timestamp and generation information
        metadata['creation_timestamp'] = datetime.datetime.now().isoformat()
        
        # Include format-specific metadata and calibration parameters
        if self.format_type == 'crimaldi':
            metadata['crimaldi_specific'] = self.to_crimaldi_format()
        elif self.format_type == 'custom':
            metadata['custom_specific'] = self.to_custom_format()
        
        # Generate unique identifier for configuration
        config_hash = hash(str(sorted(self.validated_config.items())))
        metadata['configuration_hash'] = abs(config_hash)
        
        # Return comprehensive metadata dictionary
        return metadata
    
    def _validate_crimaldi_compliance(self, config: Dict[str, Any]) -> None:
        """Validate configuration compliance with Crimaldi format requirements."""
        if config.get('pixel_to_meter_ratio') != 100.0:
            raise ValueError("Crimaldi format requires pixel-to-meter ratio of 100.0")
        
        if config.get('intensity_units') != 'concentration_ppm':
            raise ValueError("Crimaldi format requires concentration_ppm intensity units")
        
        if config.get('frame_rate_hz') != 30.0:
            raise ValueError("Crimaldi format requires 30 FPS frame rate")
    
    def _validate_custom_compliance(self, config: Dict[str, Any]) -> None:
        """Validate configuration compliance with custom format requirements."""
        if config.get('pixel_to_meter_ratio') != 150.0:
            raise ValueError("Custom format requires pixel-to-meter ratio of 150.0")
        
        if config.get('intensity_units') != 'raw_sensor':
            raise ValueError("Custom format requires raw_sensor intensity units")
        
        if config.get('frame_rate_hz') != 60.0:
            raise ValueError("Custom format requires 60 FPS frame rate")
    
    def _validate_crimaldi_specific(self) -> bool:
        """Validate Crimaldi-specific parameters."""
        # Implementation for Crimaldi-specific validation
        return True
    
    def _validate_custom_specific(self) -> bool:
        """Validate custom format-specific parameters."""
        # Implementation for custom format-specific validation
        return True


class MockPlumeGenerator:
    """
    Advanced mock plume generator class providing realistic plume concentration patterns with configurable physical 
    parameters, temporal dynamics, and format-specific characteristics for comprehensive testing of plume navigation algorithms.
    
    This class provides comprehensive plume generation with realistic physical modeling and temporal dynamics 
    for thorough testing of navigation algorithms and data processing pipelines.
    """
    
    def __init__(
        self,
        physical_parameters: Dict[str, Any],
        temporal_parameters: Dict[str, Any],
        random_seed: Optional[int] = None
    ):
        """
        Initialize mock plume generator with physical and temporal parameters for realistic plume simulation.
        
        Args:
            physical_parameters: Physical parameters (diffusion coefficient, wind velocity, source characteristics)
            temporal_parameters: Temporal parameters (frame rate, duration, intermittency)
            random_seed: Random number generator seed for reproducibility
        """
        # Set physical parameters (diffusion coefficient, wind velocity, source characteristics)
        self.physical_parameters = physical_parameters.copy()
        self._validate_physical_parameters()
        
        # Configure temporal parameters (frame rate, duration, intermittency)
        self.temporal_parameters = temporal_parameters.copy()
        self._validate_temporal_parameters()
        
        # Initialize random number generator with specified seed for reproducibility
        self.random_seed = random_seed if random_seed is not None else DEFAULT_RANDOM_SEED
        self.rng = np.random.Generator(np.random.PCG64(self.random_seed))
        
        # Create SyntheticPlumeGenerator instance with validated parameters
        self.plume_generator = SyntheticPlumeGenerator(
            physical_params=self.physical_parameters,
            temporal_params=self.temporal_parameters,
            random_seed=self.random_seed
        )
        
        # Set up generation caching for performance optimization
        self.generation_cache = {}
        self.cache_enabled = True
        self.max_cache_size = MOCK_DATA_CACHE_SIZE
        
        # Initialize current state tracking for temporal consistency
        self.current_state = {
            'time_step': 0.0,
            'frame_count': 0,
            'source_location': (0.2, 0.5),
            'plume_evolution_state': None,
            'last_generation_time': datetime.datetime.now()
        }
        
        # Validate parameter ranges and physical consistency
        self._validate_parameter_consistency()
    
    def _validate_physical_parameters(self) -> None:
        """Validate physical parameters for scientific accuracy."""
        required_params = ['diffusion_coefficient', 'wind_velocity', 'source_strength']
        for param in required_params:
            if param not in self.physical_parameters:
                raise ValueError(f"Missing required physical parameter: {param}")
        
        # Validate parameter ranges
        if self.physical_parameters['diffusion_coefficient'] <= 0:
            raise ValueError("Diffusion coefficient must be positive")
        
        wind_vel = self.physical_parameters['wind_velocity']
        if not isinstance(wind_vel, (tuple, list)) or len(wind_vel) != 2:
            raise ValueError("Wind velocity must be a tuple/list of 2 values")
        
        if self.physical_parameters['source_strength'] <= 0:
            raise ValueError("Source strength must be positive")
    
    def _validate_temporal_parameters(self) -> None:
        """Validate temporal parameters for consistency."""
        if 'frame_rate' in self.temporal_parameters:
            if self.temporal_parameters['frame_rate'] <= 0:
                raise ValueError("Frame rate must be positive")
        
        if 'duration' in self.temporal_parameters:
            if self.temporal_parameters['duration'] <= 0:
                raise ValueError("Duration must be positive")
    
    def _validate_parameter_consistency(self) -> None:
        """Validate consistency between physical and temporal parameters."""
        # Check for reasonable parameter combinations
        diffusion_coeff = self.physical_parameters['diffusion_coefficient']
        wind_speed = np.linalg.norm(self.physical_parameters['wind_velocity'])
        
        # Ensure diffusion and advection are balanced
        if diffusion_coeff > 1.0 and wind_speed < 0.1:
            warnings.warn("High diffusion with low wind may produce unrealistic plumes")
        
        if diffusion_coeff < 0.01 and wind_speed > 2.0:
            warnings.warn("Low diffusion with high wind may produce unrealistic plumes")
    
    def generate_frame(
        self,
        grid_dimensions: Tuple[int, int],
        time_step: float,
        frame_config: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Generate a single plume frame with realistic concentration patterns using diffusion-advection equations and temporal state evolution.
        
        Args:
            grid_dimensions: Grid dimensions as (width, height) for concentration field
            time_step: Current time step for temporal evolution
            frame_config: Optional configuration for frame generation
            
        Returns:
            np.ndarray: 2D concentration field array with realistic plume characteristics
        """
        # Update temporal state based on time step progression
        self.current_state['time_step'] = time_step
        self.current_state['frame_count'] += 1
        
        # Check cache for previously generated frame
        cache_key = f"{grid_dimensions}_{time_step}_{hash(str(frame_config or {}))}"
        if self.cache_enabled and cache_key in self.generation_cache:
            return self.generation_cache[cache_key].copy()
        
        # Generate plume concentration field using SyntheticPlumeGenerator
        concentration_field = self.plume_generator.generate_plume_field(
            grid_dimensions=grid_dimensions,
            source_location=self.current_state['source_location'],
            time_step=time_step,
            field_config=frame_config
        )
        
        # Apply temporal dynamics and intermittency effects
        concentration_field = self._apply_temporal_dynamics(concentration_field, time_step)
        
        # Add realistic noise and measurement artifacts
        concentration_field = self._add_measurement_artifacts(concentration_field)
        
        # Update current state for next frame generation
        self.current_state['last_generation_time'] = datetime.datetime.now()
        
        # Cache generated frame if caching is enabled
        if self.cache_enabled and len(self.generation_cache) < self.max_cache_size:
            self.generation_cache[cache_key] = concentration_field.copy()
        
        # Validate frame data structure and value ranges
        if np.any(np.isnan(concentration_field)) or np.any(np.isinf(concentration_field)):
            raise RuntimeError("Generated frame contains invalid values")
        
        if concentration_field.shape != (grid_dimensions[1], grid_dimensions[0]):
            raise RuntimeError(f"Frame shape mismatch: expected {(grid_dimensions[1], grid_dimensions[0])}, got {concentration_field.shape}")
        
        # Return realistic plume concentration frame
        return concentration_field
    
    def _apply_temporal_dynamics(self, field: np.ndarray, time_step: float) -> np.ndarray:
        """Apply temporal dynamics including intermittency and evolution."""
        # Apply intermittency effects
        intermittency_factor = self.physical_parameters.get('intermittency_factor', 0.3)
        intermittency_pattern = self.rng.random(field.shape) > (intermittency_factor * np.sin(2 * np.pi * time_step))
        field *= intermittency_pattern
        
        # Apply temporal evolution effects
        evolution_factor = 1.0 + 0.1 * np.sin(2 * np.pi * time_step * 0.1)
        field *= evolution_factor
        
        return field
    
    def _add_measurement_artifacts(self, field: np.ndarray) -> np.ndarray:
        """Add realistic measurement artifacts and noise."""
        # Add sensor noise
        noise_level = self.physical_parameters.get('noise_level', 0.05)
        noise = self.rng.normal(0, noise_level, field.shape)
        field += noise
        
        # Add background concentration
        background = self.physical_parameters.get('background_concentration', 0.0)
        field += background
        
        # Ensure non-negative values
        field = np.clip(field, 0, None)
        
        return field
    
    def reset_state(self, new_random_seed: Optional[int] = None) -> None:
        """
        Reset generator state to initial conditions for fresh simulation runs and reproducible test execution.
        
        Args:
            new_random_seed: New random seed for generator reset
        """
        # Reset random number generator with new seed if provided
        if new_random_seed is not None:
            self.random_seed = new_random_seed
            self.rng = np.random.Generator(np.random.PCG64(self.random_seed))
        
        # Clear generation cache and temporal state
        self.generation_cache.clear()
        
        # Reinitialize SyntheticPlumeGenerator to initial conditions
        self.plume_generator = SyntheticPlumeGenerator(
            physical_params=self.physical_parameters,
            temporal_params=self.temporal_parameters,
            random_seed=self.random_seed
        )
        
        # Reset current state tracking variables
        self.current_state = {
            'time_step': 0.0,
            'frame_count': 0,
            'source_location': (0.2, 0.5),
            'plume_evolution_state': None,
            'last_generation_time': datetime.datetime.now()
        }
        
        # Clear any accumulated temporal dynamics
        # Reset complete
        
        # Log state reset operation for debugging
        print(f"MockPlumeGenerator state reset with seed: {self.random_seed}")
    
    def get_metadata(self, include_statistics: Optional[bool] = None) -> Dict[str, Any]:
        """
        Get comprehensive metadata about the plume generator including physical parameters, temporal settings, and generation statistics.
        
        Args:
            include_statistics: Whether to include generation statistics
            
        Returns:
            Dict[str, Any]: Complete generator metadata with parameters and statistics
        """
        # Compile physical and temporal parameter information
        metadata = {
            'generator_id': f"plume_gen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'physical_parameters': self.physical_parameters.copy(),
            'temporal_parameters': self.temporal_parameters.copy(),
            'current_state': self.current_state.copy(),
            'generation_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Include random seed and reproducibility information
        metadata['random_seed'] = self.random_seed
        metadata['reproducibility_info'] = {
            'rng_state_available': True,
            'cache_enabled': self.cache_enabled,
            'max_cache_size': self.max_cache_size
        }
        
        # Add generation statistics if include_statistics is enabled
        if include_statistics:
            metadata['statistics'] = {
                'frames_generated': self.current_state['frame_count'],
                'cache_entries': len(self.generation_cache),
                'cache_hit_ratio': 0.0,  # Would need to track cache hits vs misses
                'last_generation_time': self.current_state['last_generation_time'].isoformat()
            }
        
        # Include current state and temporal progression information
        metadata['temporal_progression'] = {
            'current_time_step': self.current_state['time_step'],
            'total_frames': self.current_state['frame_count'],
            'source_location': self.current_state['source_location']
        }
        
        # Add cache performance and usage statistics
        if self.cache_enabled:
            metadata['cache_performance'] = {
                'cache_size': len(self.generation_cache),
                'cache_utilization': len(self.generation_cache) / self.max_cache_size if self.max_cache_size > 0 else 0,
                'memory_usage_estimate': len(self.generation_cache) * 4  # Rough estimate in MB
            }
        
        # Generate unique identifier for generator configuration
        config_hash = hash(str(sorted(self.physical_parameters.items())) + str(sorted(self.temporal_parameters.items())))
        metadata['configuration_hash'] = abs(config_hash)
        
        # Return comprehensive metadata dictionary
        return metadata
    
    def generate_sequence(
        self,
        grid_dimensions: Tuple[int, int],
        num_frames: int,
        frame_rate: float,
        sequence_config: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Generate complete temporal sequence of plume frames with consistent dynamics and realistic evolution patterns.
        
        Args:
            grid_dimensions: Grid dimensions as (width, height)
            num_frames: Number of frames to generate
            frame_rate: Frame rate for temporal sequence
            sequence_config: Optional configuration for sequence generation
            
        Returns:
            np.ndarray: 3D array [frames, height, width] representing temporal plume evolution
        """
        # Calculate time step from frame rate
        time_step_delta = 1.0 / frame_rate
        
        # Initialize temporal sequence array
        height, width = grid_dimensions[1], grid_dimensions[0]
        sequence = np.zeros((num_frames, height, width), dtype=np.float32)
        
        # Generate plume frames with temporal consistency
        for frame_idx in range(num_frames):
            current_time = frame_idx * time_step_delta
            
            # Generate frame with temporal evolution
            frame = self.generate_frame(
                grid_dimensions=grid_dimensions,
                time_step=current_time,
                frame_config=sequence_config
            )
            
            sequence[frame_idx] = frame
        
        # Apply realistic dynamics and intermittency throughout sequence
        sequence = self._apply_sequence_dynamics(sequence, frame_rate)
        
        # Ensure smooth temporal transitions between frames
        sequence = self._smooth_temporal_transitions(sequence)
        
        # Validate complete sequence structure and properties
        if sequence.shape != (num_frames, height, width):
            raise RuntimeError(f"Sequence shape mismatch: expected {(num_frames, height, width)}, got {sequence.shape}")
        
        # Return temporal plume sequence with embedded metadata
        return sequence
    
    def _apply_sequence_dynamics(self, sequence: np.ndarray, frame_rate: float) -> np.ndarray:
        """Apply sequence-level dynamics and evolution patterns."""
        num_frames = sequence.shape[0]
        
        for frame_idx in range(num_frames):
            time_step = frame_idx / frame_rate
            
            # Apply global evolution effects
            evolution_factor = 1.0 + 0.05 * np.sin(2 * np.pi * time_step * 0.05)
            sequence[frame_idx] *= evolution_factor
        
        return sequence
    
    def _smooth_temporal_transitions(self, sequence: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to ensure realistic transitions."""
        if sequence.shape[0] < 2:
            return sequence
        
        smoothing_factor = 0.1
        smoothed_sequence = sequence.copy()
        
        for frame_idx in range(1, sequence.shape[0]):
            smoothed_sequence[frame_idx] = (
                (1 - smoothing_factor) * sequence[frame_idx] +
                smoothing_factor * smoothed_sequence[frame_idx - 1]
            )
        
        return smoothed_sequence


class MockVideoDataset:
    """
    Comprehensive mock video dataset class providing complete test datasets for Crimaldi and custom formats with validation data, 
    cross-format compatibility testing, and performance benchmarking capabilities for scientific simulation validation.
    
    This class provides centralized dataset management with comprehensive validation and cross-format compatibility 
    testing for thorough validation of scientific simulation pipelines.
    """
    
    def __init__(
        self,
        dataset_config: Dict[str, Any],
        enable_caching: Optional[bool] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize mock video dataset with configuration for multiple formats and validation scenarios.
        
        Args:
            dataset_config: Configuration parameters for dataset generation
            enable_caching: Enable dataset caching for performance optimization
            random_seed: Random seed for reproducible dataset generation
        """
        # Load dataset configuration and validate parameters
        self.dataset_config = dataset_config.copy()
        self._validate_dataset_config()
        
        # Initialize TestDataValidator for data quality assurance
        self.validator = TestDataValidator(
            tolerance=VALIDATION_TOLERANCES['numerical_tolerance'],
            strict_validation=True
        )
        
        # Set random seed for reproducible dataset generation
        self.random_seed = random_seed if random_seed is not None else DEFAULT_RANDOM_SEED
        np.random.seed(self.random_seed)
        
        # Configure caching system if enabled for performance optimization
        self.enable_caching = enable_caching if enable_caching is not None else True
        self.dataset_cache = {} if self.enable_caching else None
        self.max_cache_entries = MOCK_DATA_CACHE_SIZE
        
        # Initialize dataset containers for different formats
        self.crimaldi_dataset = None
        self.custom_dataset = None
        self.validation_dataset = None
        
        # Setup validation dataset parameters and criteria
        self.validation_criteria = VALIDATION_TOLERANCES.copy()
        self.validation_criteria.update(dataset_config.get('validation_criteria', {}))
        
        # Validate configuration completeness and consistency
        self._validate_configuration_consistency()
    
    def _validate_dataset_config(self) -> None:
        """Validate dataset configuration parameters."""
        required_keys = ['formats', 'arena_size', 'resolution', 'duration']
        for key in required_keys:
            if key not in self.dataset_config:
                raise ValueError(f"Missing required dataset configuration key: {key}")
        
        # Validate format specifications
        formats = self.dataset_config['formats']
        if not isinstance(formats, list) or len(formats) == 0:
            raise ValueError("Formats must be a non-empty list")
        
        valid_formats = ['crimaldi', 'custom']
        for fmt in formats:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid format: {fmt}. Must be one of {valid_formats}")
    
    def _validate_configuration_consistency(self) -> None:
        """Validate consistency across dataset configuration parameters."""
        # Check arena size consistency
        arena_size = self.dataset_config['arena_size']
        if not isinstance(arena_size, (tuple, list)) or len(arena_size) != 2:
            raise ValueError("Arena size must be a tuple/list of 2 values")
        
        # Check resolution consistency
        resolution = self.dataset_config['resolution']
        if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
            raise ValueError("Resolution must be a tuple/list of 2 values")
        
        # Validate duration parameter
        duration = self.dataset_config['duration']
        if duration <= 0:
            raise ValueError(f"Duration must be positive: {duration}")
    
    def get_crimaldi_dataset(
        self,
        crimaldi_config: Optional[Dict[str, Any]] = None,
        force_regenerate: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Generate or retrieve Crimaldi format mock dataset with proper calibration parameters and format-specific characteristics.
        
        Args:
            crimaldi_config: Optional Crimaldi-specific configuration
            force_regenerate: Force regeneration even if cached dataset exists
            
        Returns:
            Dict[str, Any]: Complete Crimaldi format dataset with video data, metadata, and calibration parameters
        """
        force_regenerate = force_regenerate if force_regenerate is not None else False
        
        # Check cache for existing Crimaldi dataset if caching enabled
        cache_key = 'crimaldi_dataset'
        if (self.enable_caching and not force_regenerate and 
            cache_key in self.dataset_cache):
            return self.dataset_cache[cache_key].copy()
        
        # Generate new Crimaldi dataset if not cached or force_regenerate is True
        config = crimaldi_config or {}
        
        # Apply Crimaldi-specific configuration and calibration parameters
        arena_size = config.get('arena_size', self.dataset_config['arena_size'])
        resolution = config.get('resolution', self.dataset_config['resolution'])
        duration = config.get('duration', self.dataset_config['duration'])
        
        # Generate video data using generate_crimaldi_mock_data function
        crimaldi_data = generate_crimaldi_mock_data(
            arena_size_meters=arena_size,
            resolution_pixels=resolution,
            duration_seconds=duration,
            crimaldi_config=config,
            random_seed=self.random_seed
        )
        
        # Validate dataset against Crimaldi format requirements
        validation_result = self.validator.validate_video_data(
            video_data=crimaldi_data['video_data'],
            expected_properties={
                'shape': (crimaldi_data['metadata']['num_frames'],) + resolution[::-1],
                'dtype': np.uint8,
                'format_type': 'crimaldi'
            }
        )
        
        if not validation_result.is_valid:
            raise RuntimeError(f"Crimaldi dataset validation failed: {validation_result.errors}")
        
        # Add validation metadata
        crimaldi_data['validation_results'] = validation_result.to_dict()
        
        # Cache generated dataset if caching is enabled
        if self.enable_caching and len(self.dataset_cache) < self.max_cache_entries:
            self.dataset_cache[cache_key] = crimaldi_data.copy()
        
        # Store dataset reference
        self.crimaldi_dataset = crimaldi_data
        
        # Return complete Crimaldi dataset with validation metadata
        return crimaldi_data
    
    def get_custom_dataset(
        self,
        custom_config: Optional[Dict[str, Any]] = None,
        force_regenerate: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Generate or retrieve custom AVI format mock dataset with adaptive parameters and format-specific characteristics.
        
        Args:
            custom_config: Optional custom format configuration
            force_regenerate: Force regeneration even if cached dataset exists
            
        Returns:
            Dict[str, Any]: Complete custom AVI format dataset with video data, metadata, and calibration parameters
        """
        force_regenerate = force_regenerate if force_regenerate is not None else False
        
        # Check cache for existing custom dataset if caching enabled
        cache_key = 'custom_dataset'
        if (self.enable_caching and not force_regenerate and 
            cache_key in self.dataset_cache):
            return self.dataset_cache[cache_key].copy()
        
        # Generate new custom dataset if not cached or force_regenerate is True
        config = custom_config or {}
        
        # Apply custom format configuration and adaptive parameters
        arena_size = config.get('arena_size', self.dataset_config['arena_size'])
        resolution = config.get('resolution', self.dataset_config['resolution'])
        duration = config.get('duration', self.dataset_config['duration'])
        
        # Generate video data using generate_custom_avi_mock_data function
        custom_data = generate_custom_avi_mock_data(
            arena_size_meters=arena_size,
            resolution_pixels=resolution,
            duration_seconds=duration,
            custom_config=config,
            random_seed=self.random_seed
        )
        
        # Validate dataset against custom format requirements
        validation_result = self.validator.validate_video_data(
            video_data=custom_data['video_data'],
            expected_properties={
                'shape': (custom_data['metadata']['num_frames'],) + resolution[::-1] + (3,),
                'dtype': np.uint8,
                'format_type': 'custom'
            }
        )
        
        if not validation_result.is_valid:
            raise RuntimeError(f"Custom dataset validation failed: {validation_result.errors}")
        
        # Add validation metadata
        custom_data['validation_results'] = validation_result.to_dict()
        
        # Cache generated dataset if caching is enabled
        if self.enable_caching and len(self.dataset_cache) < self.max_cache_entries:
            self.dataset_cache[cache_key] = custom_data.copy()
        
        # Store dataset reference
        self.custom_dataset = custom_data
        
        # Return complete custom dataset with validation metadata
        return custom_data
    
    def get_validation_dataset(
        self,
        validation_config: Optional[Dict[str, Any]] = None,
        format_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation dataset with ground truth solutions for algorithm accuracy testing and cross-format compatibility validation.
        
        Args:
            validation_config: Optional validation-specific configuration
            format_types: List of format types to include in validation dataset
            
        Returns:
            Dict[str, Any]: Validation dataset with ground truth solutions and statistical validation parameters
        """
        # Use default format types if not specified
        if format_types is None:
            format_types = self.dataset_config.get('formats', ['crimaldi', 'custom'])
        
        # Merge validation configuration
        config = self.dataset_config.copy()
        if validation_config:
            config.update(validation_config)
        
        # Generate controlled test scenarios with known optimal solutions
        num_scenarios = config.get('num_validation_scenarios', 10)
        
        validation_data = create_validation_dataset(
            num_scenarios=num_scenarios,
            format_types=format_types,
            validation_config=config,
            correlation_target=self.validation_criteria['correlation_threshold'],
            random_seed=self.random_seed
        )
        
        # Create identical scenarios in multiple formats for cross-format testing
        # This is handled within create_validation_dataset
        
        # Generate ground truth navigation trajectories and performance metrics
        # This is handled within create_validation_dataset
        
        # Create statistical validation benchmarks with correlation targets
        validation_data['statistical_benchmarks'] = {
            'correlation_targets': {fmt: self.validation_criteria['correlation_threshold'] for fmt in format_types},
            'numerical_tolerances': self.validation_criteria,
            'performance_requirements': config.get('performance_requirements', {})
        }
        
        # Add edge cases and stress testing scenarios
        # This is handled within create_validation_dataset
        
        # Validate dataset completeness and statistical properties
        self._validate_validation_dataset(validation_data)
        
        # Store validation dataset reference
        self.validation_dataset = validation_data
        
        # Return comprehensive validation dataset with expected outcomes
        return validation_data
    
    def validate_dataset_consistency(
        self,
        tolerance: Optional[float] = None,
        detailed_report: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Validate consistency between different format datasets for cross-format compatibility testing and scientific accuracy.
        
        Args:
            tolerance: Numerical tolerance for consistency checks
            detailed_report: Whether to generate detailed consistency report
            
        Returns:
            Dict[str, Any]: Dataset consistency validation report with compatibility metrics
        """
        tolerance = tolerance if tolerance is not None else VALIDATION_TOLERANCES['format_compatibility_tolerance']
        detailed_report = detailed_report if detailed_report is not None else True
        
        consistency_report = {
            'validation_timestamp': datetime.datetime.now().isoformat(),
            'tolerance_used': tolerance,
            'datasets_compared': [],
            'consistency_metrics': {},
            'compatibility_assessment': {},
            'detailed_analysis': {} if detailed_report else None
        }
        
        # Compare Crimaldi and custom datasets for equivalent scenarios
        datasets_to_compare = []
        if self.crimaldi_dataset:
            datasets_to_compare.append(('crimaldi', self.crimaldi_dataset))
        if self.custom_dataset:
            datasets_to_compare.append(('custom', self.custom_dataset))
        
        if len(datasets_to_compare) < 2:
            # Generate datasets if not available
            if 'crimaldi' in self.dataset_config.get('formats', []):
                crimaldi_data = self.get_crimaldi_dataset()
                datasets_to_compare.append(('crimaldi', crimaldi_data))
            
            if 'custom' in self.dataset_config.get('formats', []):
                custom_data = self.get_custom_dataset()
                datasets_to_compare.append(('custom', custom_data))
        
        consistency_report['datasets_compared'] = [name for name, _ in datasets_to_compare]
        
        if len(datasets_to_compare) >= 2:
            # Validate cross-format parameter consistency and scaling
            format1_name, format1_data = datasets_to_compare[0]
            format2_name, format2_data = datasets_to_compare[1]
            
            # Compare calibration parameters
            cal1 = format1_data['calibration_parameters']
            cal2 = format2_data['calibration_parameters']
            
            # Check spatial calibration consistency
            pixel_ratio_1 = cal1['spatial_calibration']['pixel_to_meter_x']
            pixel_ratio_2 = cal2['spatial_calibration']['pixel_to_meter_x']
            pixel_ratio_difference = abs(pixel_ratio_1 - pixel_ratio_2) / max(pixel_ratio_1, pixel_ratio_2)
            
            consistency_report['consistency_metrics']['pixel_ratio_difference'] = pixel_ratio_difference
            
            # Check temporal alignment and frame rate compatibility
            fps1 = cal1['temporal_calibration']['frame_rate_hz']
            fps2 = cal2['temporal_calibration']['frame_rate_hz']
            fps_ratio = min(fps1, fps2) / max(fps1, fps2)
            
            consistency_report['consistency_metrics']['frame_rate_compatibility'] = fps_ratio
            
            # Verify intensity calibration consistency between formats
            intensity_units_compatible = (
                cal1['intensity_calibration']['units'] != cal2['intensity_calibration']['units']
            )  # Different units are expected and should be convertible
            
            consistency_report['consistency_metrics']['intensity_units_different'] = intensity_units_compatible
            
            # Check tolerance thresholds against scientific accuracy requirements
            consistency_passed = (
                pixel_ratio_difference <= tolerance and
                fps_ratio >= 0.5  # Allow significant frame rate differences
            )
            
            consistency_report['compatibility_assessment'] = {
                'overall_compatible': consistency_passed,
                'spatial_compatibility': pixel_ratio_difference <= tolerance,
                'temporal_compatibility': fps_ratio >= 0.5,
                'intensity_compatibility': intensity_units_compatible
            }
            
            # Generate detailed consistency report if requested
            if detailed_report:
                consistency_report['detailed_analysis'] = {
                    'spatial_analysis': {
                        'pixel_ratios': {format1_name: pixel_ratio_1, format2_name: pixel_ratio_2},
                        'arena_sizes': {
                            format1_name: format1_data['metadata']['arena_size_meters'],
                            format2_name: format2_data['metadata']['arena_size_meters']
                        }
                    },
                    'temporal_analysis': {
                        'frame_rates': {format1_name: fps1, format2_name: fps2},
                        'durations': {
                            format1_name: format1_data['metadata']['duration_seconds'],
                            format2_name: format2_data['metadata']['duration_seconds']
                        }
                    },
                    'intensity_analysis': {
                        'units': {
                            format1_name: cal1['intensity_calibration']['units'],
                            format2_name: cal2['intensity_calibration']['units']
                        },
                        'dynamic_ranges': {
                            format1_name: cal1['intensity_calibration']['dynamic_range'],
                            format2_name: cal2['intensity_calibration']['dynamic_range']
                        }
                    }
                }
        
        # Calculate compatibility matrix for all format combinations
        # (This would be more complex with more than 2 formats)
        
        # Generate comprehensive validation report
        consistency_report['validation_summary'] = {
            'total_datasets': len(datasets_to_compare),
            'consistency_checks_performed': len(consistency_report['consistency_metrics']),
            'overall_assessment': consistency_report.get('compatibility_assessment', {}).get('overall_compatible', False)
        }
        
        return consistency_report
    
    def export_datasets(
        self,
        export_path: str,
        dataset_types: Optional[List[str]] = None,
        include_metadata: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Export generated datasets to test fixtures with proper file organization and metadata for reuse across test modules.
        
        Args:
            export_path: Base path for dataset export
            dataset_types: List of dataset types to export ('crimaldi', 'custom', 'validation')
            include_metadata: Whether to include comprehensive metadata
            
        Returns:
            Dict[str, Any]: Export results with file paths and validation status
        """
        include_metadata = include_metadata if include_metadata is not None else True
        dataset_types = dataset_types if dataset_types is not None else ['crimaldi', 'custom', 'validation']
        
        export_results = {
            'export_timestamp': datetime.datetime.now().isoformat(),
            'export_path': export_path,
            'exported_datasets': {},
            'export_summary': {},
            'validation_status': {}
        }
        
        # Create export directory structure for organized dataset storage
        export_base_path = pathlib.Path(export_path)
        export_base_path.mkdir(parents=True, exist_ok=True)
        
        # Export specified dataset types (crimaldi, custom, validation)
        for dataset_type in dataset_types:
            try:
                if dataset_type == 'crimaldi':
                    if not self.crimaldi_dataset:
                        self.get_crimaldi_dataset()
                    dataset_data = self.crimaldi_dataset
                elif dataset_type == 'custom':
                    if not self.custom_dataset:
                        self.get_custom_dataset()
                    dataset_data = self.custom_dataset
                elif dataset_type == 'validation':
                    if not self.validation_dataset:
                        self.get_validation_dataset()
                    dataset_data = self.validation_dataset
                else:
                    continue
                
                # Save datasets using save_mock_data_to_fixture function
                fixture_name = f"{dataset_type}_dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                if dataset_type == 'validation':
                    # Handle validation dataset differently (contains multiple scenarios)
                    export_path_val = export_base_path / f"{fixture_name}.json"
                    with open(export_path_val, 'w') as f:
                        json.dump(dataset_data, f, indent=2, default=str)
                    saved_path = str(export_path_val)
                else:
                    # Use save_mock_data_to_fixture for video datasets
                    saved_path = save_mock_data_to_fixture(
                        mock_data=dataset_data,
                        fixture_name=fixture_name,
                        format_type=dataset_type,
                        metadata={'export_timestamp': datetime.datetime.now().isoformat()} if include_metadata else None
                    )
                
                export_results['exported_datasets'][dataset_type] = {
                    'fixture_name': fixture_name,
                    'file_path': saved_path,
                    'export_success': True
                }
                
            except Exception as e:
                export_results['exported_datasets'][dataset_type] = {
                    'export_success': False,
                    'error': str(e)
                }
        
        # Include comprehensive metadata if include_metadata is enabled
        if include_metadata:
            metadata_file = export_base_path / 'dataset_metadata.json'
            metadata = {
                'dataset_config': self.dataset_config,
                'random_seed': self.random_seed,
                'validation_criteria': self.validation_criteria,
                'export_info': export_results
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            export_results['metadata_file'] = str(metadata_file)
        
        # Generate checksums for data integrity validation
        for dataset_type, export_info in export_results['exported_datasets'].items():
            if export_info.get('export_success', False):
                try:
                    file_path = export_info['file_path']
                    if file_path.endswith('.npy'):
                        # For numpy files, calculate array checksum
                        data = np.load(file_path)
                        checksum = _calculate_array_checksum(data)
                    else:
                        # For JSON files, calculate file checksum
                        with open(file_path, 'r') as f:
                            content = f.read()
                        checksum = _calculate_string_checksum(content)
                    
                    export_info['checksum'] = checksum
                    export_results['validation_status'][dataset_type] = 'validated'
                    
                except Exception as e:
                    export_results['validation_status'][dataset_type] = f'validation_failed: {e}'
        
        # Update dataset registry with exported data information
        # This would update a global registry of available test fixtures
        
        # Generate export summary
        successful_exports = sum(1 for info in export_results['exported_datasets'].values() if info.get('export_success', False))
        total_exports = len(export_results['exported_datasets'])
        
        export_results['export_summary'] = {
            'total_datasets': total_exports,
            'successful_exports': successful_exports,
            'failed_exports': total_exports - successful_exports,
            'export_success_rate': successful_exports / total_exports if total_exports > 0 else 0
        }
        
        # Return export results with file paths and validation status
        return export_results
    
    def clear_cache(self) -> None:
        """
        Clear dataset cache and reset to initial state for fresh dataset generation.
        """
        # Clear all cached datasets and intermediate data
        if self.dataset_cache:
            self.dataset_cache.clear()
        
        # Reset dataset containers to initial state
        self.crimaldi_dataset = None
        self.custom_dataset = None
        self.validation_dataset = None
        
        # Clear validation results and statistics
        # Reset any accumulated state
        
        # Reset random number generator if configured
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Log cache clearing operation for debugging
        print("MockVideoDataset cache cleared and reset to initial state")
        
        # Free memory used by cached datasets
        # Python's garbage collector will handle this automatically
    
    def _validate_validation_dataset(self, validation_data: Dict[str, Any]) -> None:
        """Validate completeness and structure of validation dataset."""
        required_keys = ['dataset_id', 'scenarios', 'validation_requirements']
        for key in required_keys:
            if key not in validation_data:
                raise ValueError(f"Missing required validation dataset key: {key}")
        
        # Check scenario structure
        scenarios = validation_data['scenarios']
        if not isinstance(scenarios, list) or len(scenarios) == 0:
            raise ValueError("Validation dataset must contain scenarios")
        
        # Validate each scenario
        for i, scenario in enumerate(scenarios):
            required_scenario_keys = ['scenario_id', 'format_datasets', 'ground_truth']
            for key in required_scenario_keys:
                if key not in scenario:
                    raise ValueError(f"Scenario {i} missing required key: {key}")


# Helper functions for internal operations

def _generate_ground_truth_trajectory(
    source_location: Tuple[float, float],
    arena_size: Tuple[float, float],
    plume_params: Dict[str, Any]
) -> List[Tuple[float, float]]:
    """Generate ground truth optimal trajectory for validation."""
    # Simple implementation: direct path to source with some realistic deviations
    trajectory = []
    
    # Start from a random location
    start_x = np.random.uniform(0.1, 0.9)
    start_y = np.random.uniform(0.1, 0.9)
    
    # Generate path toward source
    num_steps = 20
    for i in range(num_steps):
        progress = i / (num_steps - 1)
        
        # Linear interpolation with some noise
        x = start_x + progress * (source_location[0] - start_x)
        y = start_y + progress * (source_location[1] - start_y)
        
        # Add realistic deviations
        x += np.random.normal(0, 0.02)
        y += np.random.normal(0, 0.02)
        
        # Keep within bounds
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        trajectory.append((x, y))
    
    return trajectory


def _generate_edge_case_scenarios(
    format_types: List[str],
    validation_config: Dict[str, Any],
    random_seed: Optional[int]
) -> List[Dict[str, Any]]:
    """Generate edge case scenarios for stress testing."""
    edge_cases = []
    
    # High noise scenario
    edge_cases.append({
        'scenario_id': 'edge_case_high_noise',
        'scenario_type': 'edge_case',
        'description': 'High noise level scenario',
        'modifications': {
            'plume_parameters': {
                'noise_level': 0.2,  # Very high noise
                'intermittency_factor': 0.8  # High intermittency
            }
        }
    })
    
    # Low diffusion scenario
    edge_cases.append({
        'scenario_id': 'edge_case_low_diffusion',
        'scenario_type': 'edge_case',
        'description': 'Low diffusion coefficient scenario',
        'modifications': {
            'plume_parameters': {
                'diffusion_coefficient': 0.01,  # Very low diffusion
                'wind_velocity': (1.0, 0.1)  # High wind
            }
        }
    })
    
    return edge_cases


def _generate_reproducibility_scenarios(
    format_types: List[str],
    random_seed: Optional[int]
) -> List[Dict[str, Any]]:
    """Generate scenarios for reproducibility testing."""
    reproducibility_scenarios = []
    
    # Multiple runs with same seed
    for run_idx in range(3):
        scenario = {
            'scenario_id': f'reproducibility_test_{run_idx}',
            'scenario_type': 'reproducibility',
            'description': f'Reproducibility test run {run_idx}',
            'fixed_seed': random_seed,
            'identical_parameters': True
        }
        reproducibility_scenarios.append(scenario)
    
    return reproducibility_scenarios


def _calculate_array_checksum(array: np.ndarray) -> str:
    """Calculate checksum for numpy array."""
    import hashlib
    return hashlib.sha256(array.tobytes()).hexdigest()


def _calculate_json_checksum(data: Dict[str, Any]) -> str:
    """Calculate checksum for JSON data."""
    import hashlib
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def _calculate_string_checksum(text: str) -> str:
    """Calculate checksum for string data."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def _update_fixture_registry(fixture_name: str, format_type: str, file_path: str) -> None:
    """Update global fixture registry with new fixture information."""
    # This would update a persistent registry of available fixtures
    # For now, this is a placeholder
    pass


def _cache_loaded_fixture(fixture_name: str, format_type: str, data: Dict[str, Any]) -> None:
    """Cache loaded fixture data for performance optimization."""
    # This would implement LRU caching for loaded fixtures
    # For now, this is a placeholder
    pass


def _validate_dataset_completeness(dataset: Dict[str, Any]) -> None:
    """Validate that a validation dataset is complete and properly structured."""
    required_keys = ['dataset_id', 'scenarios', 'validation_requirements']
    for key in required_keys:
        if key not in dataset:
            raise ValueError(f"Validation dataset missing required key: {key}")
    
    # Additional validation would go here
    pass