"""
Comprehensive scientific constants and physical parameters module for plume navigation simulation systems.

This module provides standardized values, thresholds, and conversion factors for plume navigation
simulation systems. It defines critical constants for data normalization, temporal processing,
intensity calibration, performance validation, and cross-format compatibility to ensure >95%
correlation with reference implementations and reproducible scientific analysis across 4000+
simulation processing requirements.

The module includes:
- Numerical precision standards for scientific computing workflows
- Physical scale normalization constants for cross-format compatibility
- Performance threshold validation for quality assurance
- Statistical validation constants for performance analysis
- Comprehensive unit conversion utilities with scientific precision
"""

import math  # version: 3.9+
import dataclasses  # version: 3.9+
from datetime import datetime  # version: 3.9+
from enum import Enum  # version: 3.9+
from typing import Dict, List, Union, Any, Optional  # version: 3.9+

# =============================================================================
# GLOBAL CONSTANTS - NUMERICAL PRECISION AND VALIDATION STANDARDS
# =============================================================================

# Numerical precision threshold for scientific calculations and floating-point comparisons
NUMERICAL_PRECISION_THRESHOLD: float = 1e-6

# Default correlation threshold for performance validation against reference implementations
DEFAULT_CORRELATION_THRESHOLD: float = 0.95

# Reproducibility threshold for ensuring consistent results across computational environments
REPRODUCIBILITY_THRESHOLD: float = 0.99

# Processing time target for individual simulation execution (seconds)
PROCESSING_TIME_TARGET_SECONDS: float = 7.2

# Batch completion target for 4000+ simulation processing (hours)
BATCH_COMPLETION_TARGET_HOURS: float = 8.0

# Error rate threshold for cross-format data processing validation
ERROR_RATE_THRESHOLD: float = 0.01

# =============================================================================
# SPATIAL NORMALIZATION CONSTANTS
# =============================================================================

# Pixel-to-meter conversion ratio for Crimaldi dataset format
CRIMALDI_PIXEL_TO_METER_RATIO: float = 100.0

# Pixel-to-meter conversion ratio for custom dataset formats
CUSTOM_PIXEL_TO_METER_RATIO: float = 150.0

# Target arena width for spatial normalization (meters)
TARGET_ARENA_WIDTH_METERS: float = 1.0

# Target arena height for spatial normalization (meters)
TARGET_ARENA_HEIGHT_METERS: float = 1.0

# Spatial accuracy threshold for calibration validation
SPATIAL_ACCURACY_THRESHOLD: float = 0.01

# =============================================================================
# TEMPORAL PROCESSING CONSTANTS
# =============================================================================

# Target frame rate for temporal normalization (frames per second)
TARGET_FPS: float = 30.0

# Standard frame rate for Crimaldi dataset format (Hz)
CRIMALDI_FRAME_RATE_HZ: float = 50.0

# Default frame rate for custom dataset formats (Hz)
CUSTOM_FRAME_RATE_HZ: float = 30.0

# Temporal accuracy threshold for normalization validation
TEMPORAL_ACCURACY_THRESHOLD: float = 0.001

# Anti-aliasing cutoff ratio for temporal resampling
ANTI_ALIASING_CUTOFF_RATIO: float = 0.8

# Motion preservation threshold for temporal interpolation quality
MOTION_PRESERVATION_THRESHOLD: float = 0.95

# =============================================================================
# INTENSITY CALIBRATION CONSTANTS
# =============================================================================

# Target minimum intensity value for normalization
TARGET_INTENSITY_MIN: float = 0.0

# Target maximum intensity value for normalization
TARGET_INTENSITY_MAX: float = 1.0

# Intensity calibration accuracy threshold for validation
INTENSITY_CALIBRATION_ACCURACY: float = 0.02

# Default gamma correction value for intensity adjustment
GAMMA_CORRECTION_DEFAULT: float = 1.0

# Number of histogram bins for intensity distribution analysis
HISTOGRAM_BINS: int = 256

# =============================================================================
# STATISTICAL ANALYSIS CONSTANTS
# =============================================================================

# Sigma threshold for outlier detection in intensity data
OUTLIER_DETECTION_SIGMA: float = 3.0

# Statistical significance level for hypothesis testing
STATISTICAL_SIGNIFICANCE_LEVEL: float = 0.05

# Default confidence interval for statistical analysis
CONFIDENCE_INTERVAL_DEFAULT: float = 0.95

# =============================================================================
# SYSTEM PERFORMANCE CONSTANTS
# =============================================================================

# Memory limit for processing operations (GB)
MEMORY_LIMIT_GB: int = 8

# Default maximum number of worker processes for parallel processing
MAX_WORKERS_DEFAULT: int = 8

# Cache size limit for temporary data storage (GB)
CACHE_SIZE_LIMIT_GB: int = 5

# Default timeout for long-running operations (seconds)
TIMEOUT_SECONDS_DEFAULT: int = 1800


# =============================================================================
# ENUMERATION CLASSES FOR CONSTANT CATEGORIES
# =============================================================================

class ConstantCategory(Enum):
    """Enumeration of available constant categories for organized access."""
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    INTENSITY = "intensity"
    STATISTICAL = "statistical"
    PERFORMANCE = "performance"
    VALIDATION = "validation"


class UnitSystem(Enum):
    """Enumeration of supported unit systems for scientific calculations."""
    METRIC = "metric"
    IMPERIAL = "imperial"
    PIXEL = "pixel"
    NORMALIZED = "normalized"


class FormatType(Enum):
    """Enumeration of supported plume data formats."""
    CRIMALDI = "crimaldi"
    CUSTOM = "custom"
    AVI = "avi"
    GENERIC = "generic"


# =============================================================================
# UTILITY FUNCTIONS FOR CONSTANT RETRIEVAL AND VALIDATION
# =============================================================================

def get_performance_thresholds(
    threshold_category: str = "all",
    include_derived_thresholds: bool = False
) -> Dict[str, float]:
    """
    Retrieve comprehensive performance threshold values for monitoring, validation, and quality
    assurance across all system components with configurable threshold categories.

    Args:
        threshold_category: Category of thresholds to retrieve ("all", "processing", "validation", "statistical")
        include_derived_thresholds: Whether to include calculated derived thresholds

    Returns:
        Performance thresholds dictionary with category-specific values and validation criteria
    """
    # Validate threshold category parameter against supported categories
    valid_categories = ["all", "processing", "validation", "statistical", "performance"]
    if threshold_category not in valid_categories:
        raise ValueError(f"Invalid threshold category: {threshold_category}. Must be one of {valid_categories}")

    # Load base performance thresholds from global constants
    base_thresholds = {
        "numerical_precision": NUMERICAL_PRECISION_THRESHOLD,
        "correlation_threshold": DEFAULT_CORRELATION_THRESHOLD,
        "reproducibility_threshold": REPRODUCIBILITY_THRESHOLD,
        "processing_time_target": PROCESSING_TIME_TARGET_SECONDS,
        "batch_completion_target": BATCH_COMPLETION_TARGET_HOURS,
        "error_rate_threshold": ERROR_RATE_THRESHOLD,
        "spatial_accuracy": SPATIAL_ACCURACY_THRESHOLD,
        "temporal_accuracy": TEMPORAL_ACCURACY_THRESHOLD,
        "intensity_calibration_accuracy": INTENSITY_CALIBRATION_ACCURACY,
        "statistical_significance": STATISTICAL_SIGNIFICANCE_LEVEL,
        "confidence_interval": CONFIDENCE_INTERVAL_DEFAULT,
        "outlier_detection_sigma": OUTLIER_DETECTION_SIGMA
    }

    # Apply category-specific threshold adjustments
    if threshold_category == "processing":
        thresholds = {k: v for k, v in base_thresholds.items() 
                     if k in ["processing_time_target", "batch_completion_target", "error_rate_threshold"]}
    elif threshold_category == "validation":
        thresholds = {k: v for k, v in base_thresholds.items() 
                     if k in ["correlation_threshold", "reproducibility_threshold", "numerical_precision"]}
    elif threshold_category == "statistical":
        thresholds = {k: v for k, v in base_thresholds.items() 
                     if k in ["statistical_significance", "confidence_interval", "outlier_detection_sigma"]}
    else:
        thresholds = base_thresholds.copy()

    # Calculate derived thresholds if include_derived_thresholds is enabled
    if include_derived_thresholds:
        # Derive additional performance metrics
        thresholds["max_processing_time"] = PROCESSING_TIME_TARGET_SECONDS * 1.5
        thresholds["min_correlation"] = DEFAULT_CORRELATION_THRESHOLD * 0.95
        thresholds["max_error_rate"] = ERROR_RATE_THRESHOLD * 2.0
        thresholds["simulation_efficiency"] = 4000.0 / (BATCH_COMPLETION_TARGET_HOURS * 3600.0)

    return thresholds


def get_statistical_constants(
    analysis_type: str = "general",
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Retrieve statistical constants for performance analysis, hypothesis testing, and validation
    procedures with configurable precision and confidence levels.

    Args:
        analysis_type: Type of statistical analysis ("general", "hypothesis", "correlation", "performance")
        confidence_level: Confidence level for statistical calculations (0.0 to 1.0)

    Returns:
        Statistical constants dictionary with analysis-specific values and precision parameters
    """
    # Validate analysis type and confidence level parameters
    valid_types = ["general", "hypothesis", "correlation", "performance"]
    if analysis_type not in valid_types:
        raise ValueError(f"Invalid analysis type: {analysis_type}. Must be one of {valid_types}")
    
    if not 0.0 < confidence_level <= 1.0:
        raise ValueError(f"Confidence level must be between 0.0 and 1.0, got: {confidence_level}")

    # Load statistical constants for specified analysis type
    base_constants = {
        "significance_level": 1.0 - confidence_level,
        "confidence_level": confidence_level,
        "outlier_sigma": OUTLIER_DETECTION_SIGMA,
        "numerical_precision": NUMERICAL_PRECISION_THRESHOLD,
        "correlation_threshold": DEFAULT_CORRELATION_THRESHOLD,
        "reproducibility_threshold": REPRODUCIBILITY_THRESHOLD
    }

    # Adjust constants based on confidence level requirements
    if confidence_level != CONFIDENCE_INTERVAL_DEFAULT:
        # Adjust critical values for different confidence levels
        base_constants["z_critical"] = math.sqrt(2) * math.erfinv(confidence_level)
        base_constants["t_critical_approx"] = base_constants["z_critical"] * (1 + 1/(4*100))  # Approximate for large df

    # Include analysis-specific constants
    if analysis_type == "hypothesis":
        base_constants.update({
            "alpha": 1.0 - confidence_level,
            "beta": 0.2,  # Power = 0.8
            "effect_size_small": 0.2,
            "effect_size_medium": 0.5,
            "effect_size_large": 0.8
        })
    elif analysis_type == "correlation":
        base_constants.update({
            "min_correlation": DEFAULT_CORRELATION_THRESHOLD,
            "strong_correlation": 0.8,
            "moderate_correlation": 0.5,
            "weak_correlation": 0.3
        })
    elif analysis_type == "performance":
        base_constants.update({
            "performance_threshold": DEFAULT_CORRELATION_THRESHOLD,
            "reproducibility_min": REPRODUCIBILITY_THRESHOLD,
            "accuracy_threshold": 1.0 - ERROR_RATE_THRESHOLD
        })

    return base_constants


def get_normalization_constants(
    format_type: str = "crimaldi",
    include_conversion_factors: bool = True
) -> Dict[str, Union[float, int]]:
    """
    Retrieve data normalization constants including pixel-to-meter ratios, arena dimensions,
    temporal scaling factors, and intensity calibration parameters for cross-format compatibility.

    Args:
        format_type: Data format type ("crimaldi", "custom", "generic")
        include_conversion_factors: Whether to include cross-format conversion factors

    Returns:
        Normalization constants dictionary with format-specific values and conversion factors
    """
    # Validate format type parameter against supported formats
    valid_formats = ["crimaldi", "custom", "generic"]
    if format_type not in valid_formats:
        raise ValueError(f"Invalid format type: {format_type}. Must be one of {valid_formats}")

    # Load format-specific normalization constants
    base_constants = {
        "target_arena_width": TARGET_ARENA_WIDTH_METERS,
        "target_arena_height": TARGET_ARENA_HEIGHT_METERS,
        "target_fps": TARGET_FPS,
        "target_intensity_min": TARGET_INTENSITY_MIN,
        "target_intensity_max": TARGET_INTENSITY_MAX,
        "histogram_bins": HISTOGRAM_BINS,
        "gamma_correction": GAMMA_CORRECTION_DEFAULT
    }

    # Add format-specific constants
    if format_type == "crimaldi":
        base_constants.update({
            "pixel_to_meter_ratio": CRIMALDI_PIXEL_TO_METER_RATIO,
            "frame_rate_hz": CRIMALDI_FRAME_RATE_HZ,
            "spatial_accuracy": SPATIAL_ACCURACY_THRESHOLD,
            "temporal_accuracy": TEMPORAL_ACCURACY_THRESHOLD
        })
    elif format_type == "custom":
        base_constants.update({
            "pixel_to_meter_ratio": CUSTOM_PIXEL_TO_METER_RATIO,
            "frame_rate_hz": CUSTOM_FRAME_RATE_HZ,
            "spatial_accuracy": SPATIAL_ACCURACY_THRESHOLD,
            "temporal_accuracy": TEMPORAL_ACCURACY_THRESHOLD
        })
    else:  # generic
        base_constants.update({
            "pixel_to_meter_ratio": (CRIMALDI_PIXEL_TO_METER_RATIO + CUSTOM_PIXEL_TO_METER_RATIO) / 2.0,
            "frame_rate_hz": TARGET_FPS,
            "spatial_accuracy": SPATIAL_ACCURACY_THRESHOLD,
            "temporal_accuracy": TEMPORAL_ACCURACY_THRESHOLD
        })

    # Include conversion factors if include_conversion_factors is enabled
    if include_conversion_factors:
        base_constants.update({
            "crimaldi_to_custom_pixel_ratio": CUSTOM_PIXEL_TO_METER_RATIO / CRIMALDI_PIXEL_TO_METER_RATIO,
            "crimaldi_to_target_fps_ratio": TARGET_FPS / CRIMALDI_FRAME_RATE_HZ,
            "custom_to_target_fps_ratio": TARGET_FPS / CUSTOM_FRAME_RATE_HZ,
            "anti_aliasing_cutoff": ANTI_ALIASING_CUTOFF_RATIO,
            "motion_preservation": MOTION_PRESERVATION_THRESHOLD
        })

    return base_constants


def get_physical_constants(
    constant_category: str = "all",
    unit_system: str = "metric"
) -> Dict[str, float]:
    """
    Retrieve physical constants and unit conversion factors for scientific calculations,
    spatial transformations, and temporal processing with validation and precision metadata.

    Args:
        constant_category: Category of constants ("all", "spatial", "temporal", "intensity")
        unit_system: Unit system for constants ("metric", "pixel", "normalized")

    Returns:
        Physical constants dictionary with unit-specific values and conversion metadata
    """
    # Validate constant category and unit system parameters
    valid_categories = ["all", "spatial", "temporal", "intensity"]
    valid_units = ["metric", "pixel", "normalized"]
    
    if constant_category not in valid_categories:
        raise ValueError(f"Invalid constant category: {constant_category}. Must be one of {valid_categories}")
    
    if unit_system not in valid_units:
        raise ValueError(f"Invalid unit system: {unit_system}. Must be one of {valid_units}")

    # Load physical constants for specified category
    constants = {}
    
    if constant_category in ["all", "spatial"]:
        spatial_constants = {
            "arena_width": TARGET_ARENA_WIDTH_METERS,
            "arena_height": TARGET_ARENA_HEIGHT_METERS,
            "spatial_accuracy": SPATIAL_ACCURACY_THRESHOLD,
            "pixel_meter_crimaldi": CRIMALDI_PIXEL_TO_METER_RATIO,
            "pixel_meter_custom": CUSTOM_PIXEL_TO_METER_RATIO
        }
        constants.update(spatial_constants)

    if constant_category in ["all", "temporal"]:
        temporal_constants = {
            "target_fps": TARGET_FPS,
            "crimaldi_fps": CRIMALDI_FRAME_RATE_HZ,
            "custom_fps": CUSTOM_FRAME_RATE_HZ,
            "temporal_accuracy": TEMPORAL_ACCURACY_THRESHOLD,
            "anti_aliasing_cutoff": ANTI_ALIASING_CUTOFF_RATIO,
            "motion_preservation": MOTION_PRESERVATION_THRESHOLD
        }
        constants.update(temporal_constants)

    if constant_category in ["all", "intensity"]:
        intensity_constants = {
            "intensity_min": TARGET_INTENSITY_MIN,
            "intensity_max": TARGET_INTENSITY_MAX,
            "intensity_accuracy": INTENSITY_CALIBRATION_ACCURACY,
            "gamma_default": GAMMA_CORRECTION_DEFAULT,
            "histogram_bins": float(HISTOGRAM_BINS),
            "outlier_sigma": OUTLIER_DETECTION_SIGMA
        }
        constants.update(intensity_constants)

    # Apply unit system conversions if required
    if unit_system == "pixel":
        # Convert metric units to pixel units where applicable
        if "arena_width" in constants:
            constants["arena_width"] *= CRIMALDI_PIXEL_TO_METER_RATIO
        if "arena_height" in constants:
            constants["arena_height"] *= CRIMALDI_PIXEL_TO_METER_RATIO
    elif unit_system == "normalized":
        # Convert to normalized units (0-1 range)
        if "arena_width" in constants:
            constants["arena_width"] = 1.0
        if "arena_height" in constants:
            constants["arena_height"] = 1.0

    return constants


def validate_constant_precision(
    constant_value: Union[float, int],
    constant_name: str,
    required_precision: float = NUMERICAL_PRECISION_THRESHOLD
) -> bool:
    """
    Validate numerical precision of constants against scientific computing requirements
    and reproducibility standards with comprehensive precision analysis.

    Args:
        constant_value: Numerical value to validate
        constant_name: Name of the constant for error reporting
        required_precision: Required precision threshold for validation

    Returns:
        True if constant meets precision requirements with validation metadata
    """
    # Validate constant value and precision requirements
    if not isinstance(constant_value, (int, float)):
        raise TypeError(f"Constant value must be numeric, got {type(constant_value)} for {constant_name}")
    
    if required_precision <= 0:
        raise ValueError(f"Required precision must be positive, got {required_precision}")

    # Check numerical precision against required thresholds
    if isinstance(constant_value, int):
        # Integer constants have perfect precision
        return True

    # Assess floating-point representation accuracy
    # Check if the constant can be represented exactly in floating-point
    rounded_value = round(constant_value, int(-math.log10(required_precision)))
    precision_error = abs(constant_value - rounded_value)
    
    # Validate against scientific computing standards
    meets_precision = precision_error <= required_precision
    
    # Additional validation for special values
    if math.isnan(constant_value) or math.isinf(constant_value):
        return False
    
    # Check for meaningful precision (not too close to machine epsilon)
    if abs(constant_value) > 0 and precision_error / abs(constant_value) > required_precision:
        return False

    return meets_precision


def calculate_derived_constants(
    base_constants: Dict[str, float],
    derivation_rules: List[str],
    validate_derivations: bool = True
) -> Dict[str, float]:
    """
    Calculate derived constants from base physical parameters including scaling factors,
    conversion ratios, and composite thresholds for system optimization.

    Args:
        base_constants: Dictionary of base constants for derivation
        derivation_rules: List of derivation rule names to apply
        validate_derivations: Whether to validate derived constants

    Returns:
        Derived constants dictionary with calculated values and validation status
    """
    # Validate base constants and derivation rules
    if not isinstance(base_constants, dict):
        raise TypeError("Base constants must be a dictionary")
    
    if not isinstance(derivation_rules, list):
        raise TypeError("Derivation rules must be a list")

    derived_constants = {}
    
    # Apply mathematical derivation rules to base constants
    for rule in derivation_rules:
        if rule == "pixel_conversion_ratios":
            if "crimaldi_pixel_ratio" in base_constants and "custom_pixel_ratio" in base_constants:
                derived_constants["pixel_ratio_difference"] = (
                    base_constants["custom_pixel_ratio"] / base_constants["crimaldi_pixel_ratio"]
                )
                
        elif rule == "temporal_scaling_factors":
            if "crimaldi_fps" in base_constants and "target_fps" in base_constants:
                derived_constants["temporal_scaling_crimaldi"] = (
                    base_constants["target_fps"] / base_constants["crimaldi_fps"]
                )
                
        elif rule == "performance_efficiency":
            if "processing_time" in base_constants and "batch_size" in base_constants:
                derived_constants["throughput_rate"] = (
                    base_constants["batch_size"] / base_constants["processing_time"]
                )
                
        elif rule == "accuracy_composite":
            accuracy_components = [k for k in base_constants.keys() if "accuracy" in k]
            if accuracy_components:
                derived_constants["composite_accuracy"] = (
                    sum(base_constants[k] for k in accuracy_components) / len(accuracy_components)
                )

    # Calculate composite constants and scaling factors
    if "arena_width" in base_constants and "arena_height" in base_constants:
        derived_constants["arena_aspect_ratio"] = (
            base_constants["arena_width"] / base_constants["arena_height"]
        )
        derived_constants["arena_area"] = (
            base_constants["arena_width"] * base_constants["arena_height"]
        )

    # Validate derived constants if validate_derivations is enabled
    if validate_derivations:
        for name, value in derived_constants.items():
            if not validate_constant_precision(value, name):
                raise ValueError(f"Derived constant {name} does not meet precision requirements: {value}")

    return derived_constants


# =============================================================================
# COMPREHENSIVE PHYSICAL CONSTANTS CONTAINER CLASS
# =============================================================================

@dataclasses.dataclass
class PhysicalConstants:
    """
    Comprehensive physical constants container class providing standardized access to scientific
    constants, unit conversion utilities, and validation methods for consistent physical parameter
    management across the plume simulation system.
    """
    
    # Core configuration parameters
    unit_system: str = "metric"
    precision_threshold: float = NUMERICAL_PRECISION_THRESHOLD
    
    # Constant categories
    spatial_constants: Dict[str, float] = dataclasses.field(default_factory=dict)
    temporal_constants: Dict[str, float] = dataclasses.field(default_factory=dict)
    intensity_constants: Dict[str, float] = dataclasses.field(default_factory=dict)
    statistical_constants: Dict[str, float] = dataclasses.field(default_factory=dict)
    performance_constants: Dict[str, float] = dataclasses.field(default_factory=dict)
    
    # Conversion and validation metadata
    conversion_matrices: Dict[str, Dict[str, float]] = dataclasses.field(default_factory=dict)
    validation_enabled: bool = True
    last_updated: datetime = dataclasses.field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize physical constants container with comprehensive constant management."""
        # Initialize spatial constants for arena dimensions and scaling
        self.spatial_constants = {
            "arena_width_meters": TARGET_ARENA_WIDTH_METERS,
            "arena_height_meters": TARGET_ARENA_HEIGHT_METERS,
            "crimaldi_pixel_ratio": CRIMALDI_PIXEL_TO_METER_RATIO,
            "custom_pixel_ratio": CUSTOM_PIXEL_TO_METER_RATIO,
            "spatial_accuracy": SPATIAL_ACCURACY_THRESHOLD
        }
        
        # Setup temporal constants for frame rates and timing
        self.temporal_constants = {
            "target_fps": TARGET_FPS,
            "crimaldi_fps": CRIMALDI_FRAME_RATE_HZ,
            "custom_fps": CUSTOM_FRAME_RATE_HZ,
            "temporal_accuracy": TEMPORAL_ACCURACY_THRESHOLD,
            "anti_aliasing_cutoff": ANTI_ALIASING_CUTOFF_RATIO,
            "motion_preservation": MOTION_PRESERVATION_THRESHOLD
        }
        
        # Configure intensity constants for calibration and normalization
        self.intensity_constants = {
            "intensity_min": TARGET_INTENSITY_MIN,
            "intensity_max": TARGET_INTENSITY_MAX,
            "calibration_accuracy": INTENSITY_CALIBRATION_ACCURACY,
            "gamma_correction": GAMMA_CORRECTION_DEFAULT,
            "histogram_bins": float(HISTOGRAM_BINS),
            "outlier_sigma": OUTLIER_DETECTION_SIGMA
        }
        
        # Load statistical constants for analysis and validation
        self.statistical_constants = {
            "significance_level": STATISTICAL_SIGNIFICANCE_LEVEL,
            "confidence_interval": CONFIDENCE_INTERVAL_DEFAULT,
            "correlation_threshold": DEFAULT_CORRELATION_THRESHOLD,
            "reproducibility_threshold": REPRODUCIBILITY_THRESHOLD,
            "numerical_precision": NUMERICAL_PRECISION_THRESHOLD
        }
        
        # Initialize performance constants for monitoring and optimization
        self.performance_constants = {
            "processing_time_target": PROCESSING_TIME_TARGET_SECONDS,
            "batch_completion_target": BATCH_COMPLETION_TARGET_HOURS,
            "error_rate_threshold": ERROR_RATE_THRESHOLD,
            "memory_limit_gb": float(MEMORY_LIMIT_GB),
            "max_workers": float(MAX_WORKERS_DEFAULT),
            "cache_limit_gb": float(CACHE_SIZE_LIMIT_GB),
            "timeout_seconds": float(TIMEOUT_SECONDS_DEFAULT)
        }
        
        # Build conversion matrices for unit transformations
        self._build_conversion_matrices()

    def _build_conversion_matrices(self):
        """Build comprehensive conversion matrices for unit transformations."""
        # Spatial conversion matrices
        self.conversion_matrices["spatial"] = {
            "meter_to_crimaldi_pixel": CRIMALDI_PIXEL_TO_METER_RATIO,
            "meter_to_custom_pixel": CUSTOM_PIXEL_TO_METER_RATIO,
            "crimaldi_to_custom_pixel": CUSTOM_PIXEL_TO_METER_RATIO / CRIMALDI_PIXEL_TO_METER_RATIO,
            "custom_to_crimaldi_pixel": CRIMALDI_PIXEL_TO_METER_RATIO / CUSTOM_PIXEL_TO_METER_RATIO
        }
        
        # Temporal conversion matrices
        self.conversion_matrices["temporal"] = {
            "crimaldi_to_target_fps": TARGET_FPS / CRIMALDI_FRAME_RATE_HZ,
            "custom_to_target_fps": TARGET_FPS / CUSTOM_FRAME_RATE_HZ,
            "target_to_crimaldi_fps": CRIMALDI_FRAME_RATE_HZ / TARGET_FPS,
            "target_to_custom_fps": CUSTOM_FRAME_RATE_HZ / TARGET_FPS
        }

    def get_conversion_factor(
        self,
        source_unit: str,
        target_unit: str,
        quantity_type: str = "spatial"
    ) -> float:
        """
        Calculate conversion factor between different units with precision validation
        and error handling for accurate unit transformations.

        Args:
            source_unit: Source unit for conversion
            target_unit: Target unit for conversion
            quantity_type: Type of quantity being converted ("spatial", "temporal")

        Returns:
            Conversion factor with precision metadata and validation status
        """
        # Validate source and target unit parameters
        if not isinstance(source_unit, str) or not isinstance(target_unit, str):
            raise TypeError("Source and target units must be strings")
        
        if quantity_type not in self.conversion_matrices:
            raise ValueError(f"Unsupported quantity type: {quantity_type}")

        # Look up conversion factor in conversion matrices
        conversion_key = f"{source_unit}_to_{target_unit}"
        if conversion_key in self.conversion_matrices[quantity_type]:
            factor = self.conversion_matrices[quantity_type][conversion_key]
        else:
            # Try reverse conversion
            reverse_key = f"{target_unit}_to_{source_unit}"
            if reverse_key in self.conversion_matrices[quantity_type]:
                factor = 1.0 / self.conversion_matrices[quantity_type][reverse_key]
            else:
                raise ValueError(f"No conversion available from {source_unit} to {target_unit}")

        # Validate conversion factor precision
        if self.validation_enabled:
            if not validate_constant_precision(factor, f"conversion_{source_unit}_to_{target_unit}"):
                raise ValueError(f"Conversion factor precision validation failed: {factor}")

        return factor

    def validate_unit(self, unit_name: str, quantity_type: str = "spatial") -> bool:
        """
        Validate unit specification against supported unit systems and physical constraints
        with comprehensive validation reporting.

        Args:
            unit_name: Name of the unit to validate
            quantity_type: Type of quantity for the unit

        Returns:
            True if unit is valid with validation details and recommendations
        """
        # Check unit name against supported unit registry
        supported_spatial_units = ["meter", "crimaldi_pixel", "custom_pixel", "normalized"]
        supported_temporal_units = ["fps", "hz", "second"]
        
        if quantity_type == "spatial":
            valid_units = supported_spatial_units
        elif quantity_type == "temporal":
            valid_units = supported_temporal_units
        else:
            return False

        # Validate unit compatibility with quantity type
        is_valid = any(unit_name.endswith(unit) or unit_name == unit for unit in valid_units)
        
        return is_valid

    def get_constant_by_category(
        self,
        category: str,
        target_unit_system: str = None,
        validate_precision: bool = True
    ) -> Dict[str, float]:
        """
        Retrieve constants by category with optional unit conversion and precision validation
        for organized constant access.

        Args:
            category: Category of constants to retrieve
            target_unit_system: Target unit system for conversion
            validate_precision: Whether to validate precision of constants

        Returns:
            Category-specific constants with unit conversion and validation metadata
        """
        # Validate category parameter against available categories
        category_map = {
            "spatial": self.spatial_constants,
            "temporal": self.temporal_constants,
            "intensity": self.intensity_constants,
            "statistical": self.statistical_constants,
            "performance": self.performance_constants
        }
        
        if category not in category_map:
            raise ValueError(f"Invalid category: {category}. Must be one of {list(category_map.keys())}")

        # Load constants for specified category
        constants = category_map[category].copy()
        
        # Apply unit system conversion if target_unit_system differs
        if target_unit_system and target_unit_system != self.unit_system:
            # Apply conversions based on category and target unit system
            if category == "spatial" and target_unit_system == "pixel":
                for key, value in constants.items():
                    if "meter" in key:
                        constants[key] = value * CRIMALDI_PIXEL_TO_METER_RATIO

        # Validate precision if validate_precision is enabled
        if validate_precision and self.validation_enabled:
            for name, value in constants.items():
                if not validate_constant_precision(value, name, self.precision_threshold):
                    raise ValueError(f"Constant {name} failed precision validation: {value}")

        return constants

    def update_constants(
        self,
        new_constants: Dict[str, float],
        validate_updates: bool = True,
        preserve_precision: bool = True
    ) -> bool:
        """
        Update physical constants with new values including validation, precision checking,
        and consistency verification for dynamic constant management.

        Args:
            new_constants: Dictionary of new constant values
            validate_updates: Whether to validate updates against physical constraints
            preserve_precision: Whether to preserve precision requirements

        Returns:
            True if update successful with validation report and change summary
        """
        # Validate new constants format and values
        if not isinstance(new_constants, dict):
            raise TypeError("New constants must be provided as a dictionary")

        # Check precision requirements if preserve_precision is enabled
        if preserve_precision:
            for name, value in new_constants.items():
                if not validate_constant_precision(value, name, self.precision_threshold):
                    raise ValueError(f"New constant {name} does not meet precision requirements: {value}")

        # Validate updates against physical constraints if validate_updates is enabled
        if validate_updates:
            for name, value in new_constants.items():
                # Physical constraint validations
                if "accuracy" in name and not (0.0 <= value <= 1.0):
                    raise ValueError(f"Accuracy values must be between 0 and 1: {name} = {value}")
                if "fps" in name and value <= 0:
                    raise ValueError(f"Frame rate values must be positive: {name} = {value}")
                if "threshold" in name and value < 0:
                    raise ValueError(f"Threshold values must be non-negative: {name} = {value}")

        # Update constant values and conversion matrices
        for name, value in new_constants.items():
            # Determine which category the constant belongs to
            for category_name, category_dict in [
                ("spatial", self.spatial_constants),
                ("temporal", self.temporal_constants),
                ("intensity", self.intensity_constants),
                ("statistical", self.statistical_constants),
                ("performance", self.performance_constants)
            ]:
                if name in category_dict:
                    category_dict[name] = value
                    break

        # Rebuild conversion matrices with updated values
        self._build_conversion_matrices()
        
        # Record update timestamp
        self.last_updated = datetime.now()

        return True

    def export_constants(
        self,
        output_path: str,
        export_format: str = "json",
        include_metadata: bool = True
    ) -> bool:
        """
        Export physical constants to file format with metadata, validation status,
        and precision information for reproducibility and documentation.

        Args:
            output_path: Path for output file
            export_format: Format for export ("json", "csv", "yaml")
            include_metadata: Whether to include metadata in export

        Returns:
            True if export successful with file validation and integrity checking
        """
        import json
        
        # Prepare constants data for export with specified format
        export_data = {
            "unit_system": self.unit_system,
            "precision_threshold": self.precision_threshold,
            "spatial_constants": self.spatial_constants,
            "temporal_constants": self.temporal_constants,
            "intensity_constants": self.intensity_constants,
            "statistical_constants": self.statistical_constants,
            "performance_constants": self.performance_constants
        }
        
        # Include metadata and validation status if include_metadata is enabled
        if include_metadata:
            export_data["metadata"] = {
                "export_timestamp": datetime.now().isoformat(),
                "validation_enabled": self.validation_enabled,
                "last_updated": self.last_updated.isoformat(),
                "conversion_matrices": self.conversion_matrices
            }

        # Export constants to specified output path
        try:
            if export_format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            return True
        except Exception as e:
            raise IOError(f"Failed to export constants to {output_path}: {e}")

    def validate_all_constants(
        self,
        strict_validation: bool = False,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation of all physical constants including precision,
        consistency, and physical constraint checking with detailed reporting.

        Args:
            strict_validation: Whether to apply strict validation criteria
            generate_report: Whether to generate detailed validation report

        Returns:
            Comprehensive validation results with detailed analysis and recommendations
        """
        validation_results = {
            "overall_status": True,
            "category_results": {},
            "precision_validation": {},
            "constraint_validation": {},
            "recommendations": []
        }

        # Validate all constant categories against precision thresholds
        for category_name, category_constants in [
            ("spatial", self.spatial_constants),
            ("temporal", self.temporal_constants),
            ("intensity", self.intensity_constants),
            ("statistical", self.statistical_constants),
            ("performance", self.performance_constants)
        ]:
            category_valid = True
            category_issues = []
            
            for const_name, const_value in category_constants.items():
                # Validate precision
                precision_valid = validate_constant_precision(
                    const_value, const_name, self.precision_threshold
                )
                validation_results["precision_validation"][const_name] = precision_valid
                
                if not precision_valid:
                    category_valid = False
                    category_issues.append(f"Precision validation failed for {const_name}")

                # Apply strict validation criteria if strict_validation is enabled
                if strict_validation:
                    # Additional constraint checks
                    if "accuracy" in const_name and not (0.0 <= const_value <= 1.0):
                        category_valid = False
                        category_issues.append(f"Accuracy constraint violated for {const_name}")
                    
                    if "fps" in const_name and const_value <= 0:
                        category_valid = False
                        category_issues.append(f"Frame rate constraint violated for {const_name}")

            validation_results["category_results"][category_name] = {
                "valid": category_valid,
                "issues": category_issues
            }
            
            if not category_valid:
                validation_results["overall_status"] = False

        # Check consistency between related constants
        consistency_checks = [
            ("spatial_constants", "arena_width_meters", "arena_height_meters"),
            ("temporal_constants", "target_fps", "crimaldi_fps"),
            ("intensity_constants", "intensity_min", "intensity_max")
        ]
        
        for category, const1, const2 in consistency_checks:
            category_dict = getattr(self, category)
            if const1 in category_dict and const2 in category_dict:
                if category == "intensity_constants":
                    if category_dict[const1] >= category_dict[const2]:
                        validation_results["overall_status"] = False
                        validation_results["recommendations"].append(
                            f"Intensity min should be less than max: {const1} >= {const2}"
                        )

        return validation_results


# =============================================================================
# PERFORMANCE THRESHOLD MANAGEMENT CLASS
# =============================================================================

@dataclasses.dataclass
class PerformanceThresholds:
    """
    Performance threshold management class providing dynamic threshold configuration,
    validation criteria, and monitoring parameters for system performance optimization
    and quality assurance with adaptive threshold adjustment capabilities.
    """
    
    # Core threshold configuration
    base_thresholds: Dict[str, float] = dataclasses.field(default_factory=dict)
    adaptive_thresholds_enabled: bool = False
    
    # Dynamic threshold management
    current_thresholds: Dict[str, float] = dataclasses.field(default_factory=dict)
    threshold_history: Dict[str, List[float]] = dataclasses.field(default_factory=dict)
    adaptation_factors: Dict[str, float] = dataclasses.field(default_factory=dict)
    adaptation_rate: float = 0.1
    adaptation_window: int = 10
    last_adaptation: datetime = dataclasses.field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize performance thresholds with comprehensive threshold management."""
        # Set default base thresholds if not provided
        if not self.base_thresholds:
            self.base_thresholds = {
                "processing_time": PROCESSING_TIME_TARGET_SECONDS,
                "correlation": DEFAULT_CORRELATION_THRESHOLD,
                "reproducibility": REPRODUCIBILITY_THRESHOLD,
                "error_rate": ERROR_RATE_THRESHOLD,
                "memory_usage": float(MEMORY_LIMIT_GB),
                "batch_completion": BATCH_COMPLETION_TARGET_HOURS
            }
        
        # Initialize current thresholds with base values
        self.current_thresholds = self.base_thresholds.copy()
        
        # Setup threshold history tracking for adaptation
        self.threshold_history = {name: [] for name in self.base_thresholds}
        
        # Configure adaptation factors and rates
        self.adaptation_factors = {name: 1.0 for name in self.base_thresholds}

    def get_threshold(
        self,
        metric_name: str,
        apply_adaptation: bool = True
    ) -> float:
        """
        Retrieve current threshold value for specified metric with adaptive adjustment
        and validation metadata.

        Args:
            metric_name: Name of the performance metric
            apply_adaptation: Whether to apply adaptive adjustment

        Returns:
            Current threshold value with adaptation status and validation metadata
        """
        # Validate metric name against available thresholds
        if metric_name not in self.current_thresholds:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(self.current_thresholds.keys())}")

        # Apply adaptive adjustment if apply_adaptation is enabled
        threshold_value = self.current_thresholds[metric_name]
        
        if apply_adaptation and self.adaptive_thresholds_enabled:
            adaptation_factor = self.adaptation_factors.get(metric_name, 1.0)
            threshold_value *= adaptation_factor

        return threshold_value

    def update_threshold(
        self,
        metric_name: str,
        new_value: float,
        validate_update: bool = True
    ) -> bool:
        """
        Update threshold value with validation and history tracking for dynamic
        threshold management.

        Args:
            metric_name: Name of the metric threshold to update
            new_value: New threshold value
            validate_update: Whether to validate the update

        Returns:
            True if update successful with validation status and change tracking
        """
        # Validate new threshold value if validate_update is enabled
        if validate_update:
            if not isinstance(new_value, (int, float)):
                raise TypeError(f"Threshold value must be numeric, got {type(new_value)}")
            
            if new_value <= 0:
                raise ValueError(f"Threshold values must be positive, got {new_value}")

            # Metric-specific validation
            if "correlation" in metric_name and not (0.0 <= new_value <= 1.0):
                raise ValueError(f"Correlation thresholds must be between 0 and 1, got {new_value}")
            
            if "error_rate" in metric_name and not (0.0 <= new_value <= 1.0):
                raise ValueError(f"Error rate thresholds must be between 0 and 1, got {new_value}")

        # Update threshold value and record in history
        old_value = self.current_thresholds.get(metric_name, 0.0)
        self.current_thresholds[metric_name] = new_value
        
        # Record in threshold history
        if metric_name not in self.threshold_history:
            self.threshold_history[metric_name] = []
        self.threshold_history[metric_name].append(new_value)
        
        # Limit history size to adaptation window
        if len(self.threshold_history[metric_name]) > self.adaptation_window:
            self.threshold_history[metric_name] = self.threshold_history[metric_name][-self.adaptation_window:]

        # Update adaptation factors if adaptive thresholds enabled
        if self.adaptive_thresholds_enabled and old_value > 0:
            change_ratio = new_value / old_value
            self.adaptation_factors[metric_name] = (
                self.adaptation_factors.get(metric_name, 1.0) * 
                (1.0 - self.adaptation_rate) + change_ratio * self.adaptation_rate
            )

        return True

    def adapt_thresholds(
        self,
        performance_history: Dict[str, List[float]],
        validate_adaptations: bool = True
    ) -> Dict[str, float]:
        """
        Perform adaptive threshold adjustment based on performance history and system
        behavior with optimization and validation.

        Args:
            performance_history: Historical performance data for adaptation
            validate_adaptations: Whether to validate adapted thresholds

        Returns:
            Adapted thresholds with adjustment rationale and validation status
        """
        if not self.adaptive_thresholds_enabled:
            return self.current_thresholds.copy()

        adapted_thresholds = {}
        
        # Analyze performance history for adaptation opportunities
        for metric_name, history in performance_history.items():
            if metric_name in self.current_thresholds and len(history) >= 3:
                # Calculate optimal threshold adjustments
                current_threshold = self.current_thresholds[metric_name]
                recent_performance = history[-self.adaptation_window:]
                
                # Statistical analysis of recent performance
                mean_performance = sum(recent_performance) / len(recent_performance)
                performance_variance = sum((x - mean_performance) ** 2 for x in recent_performance) / len(recent_performance)
                performance_std = math.sqrt(performance_variance)
                
                # Adaptive adjustment based on performance characteristics
                if "processing_time" in metric_name:
                    # For processing time, adjust based on recent performance + buffer
                    adapted_threshold = mean_performance + 2 * performance_std
                elif "correlation" in metric_name:
                    # For correlation, maintain conservative threshold
                    adapted_threshold = max(current_threshold, mean_performance * 0.95)
                elif "error_rate" in metric_name:
                    # For error rate, use recent maximum with safety margin
                    adapted_threshold = min(current_threshold, max(recent_performance) * 1.1)
                else:
                    # Default adaptation strategy
                    adapted_threshold = current_threshold
                
                # Apply adaptations with validation if validate_adaptations is enabled
                if validate_adaptations:
                    # Ensure adapted threshold is within reasonable bounds
                    base_threshold = self.base_thresholds.get(metric_name, current_threshold)
                    max_adaptation = base_threshold * 2.0
                    min_adaptation = base_threshold * 0.5
                    adapted_threshold = max(min_adaptation, min(adapted_threshold, max_adaptation))

                adapted_thresholds[metric_name] = adapted_threshold
            else:
                adapted_thresholds[metric_name] = self.current_thresholds[metric_name]

        # Update current thresholds with adaptations
        self.current_thresholds.update(adapted_thresholds)
        self.last_adaptation = datetime.now()
        
        return adapted_thresholds


# =============================================================================
# COMPREHENSIVE UNIT CONVERTER CLASS
# =============================================================================

class UnitConverter:
    """
    Comprehensive unit conversion utility class providing accurate conversions between
    different unit systems, validation of conversion accuracy, and support for scientific
    computing precision requirements with extensive unit registry and conversion validation.
    """

    def __init__(
        self,
        default_unit_system: str = "metric",
        conversion_precision: float = NUMERICAL_PRECISION_THRESHOLD
    ):
        """
        Initialize unit converter with default unit system and precision requirements
        for accurate scientific unit conversions.

        Args:
            default_unit_system: Default unit system for conversions
            conversion_precision: Required precision for conversion accuracy
        """
        # Set default unit system and conversion precision
        self.default_unit_system = default_unit_system
        self.conversion_precision = conversion_precision
        
        # Initialize conversion registry with standard conversions
        self.conversion_registry = self._build_conversion_registry()
        
        # Setup unit categories and supported systems
        self.unit_categories = self._build_unit_categories()
        self.supported_unit_systems = ["metric", "imperial", "pixel", "normalized"]
        
        # Enable conversion validation and accuracy checking
        self.validation_enabled = True

    def _build_conversion_registry(self) -> Dict[str, Dict[str, float]]:
        """Build comprehensive conversion registry with standard scientific conversions."""
        return {
            "length": {
                "meter_to_pixel_crimaldi": CRIMALDI_PIXEL_TO_METER_RATIO,
                "meter_to_pixel_custom": CUSTOM_PIXEL_TO_METER_RATIO,
                "pixel_crimaldi_to_meter": 1.0 / CRIMALDI_PIXEL_TO_METER_RATIO,
                "pixel_custom_to_meter": 1.0 / CUSTOM_PIXEL_TO_METER_RATIO,
                "pixel_crimaldi_to_pixel_custom": CUSTOM_PIXEL_TO_METER_RATIO / CRIMALDI_PIXEL_TO_METER_RATIO,
                "pixel_custom_to_pixel_crimaldi": CRIMALDI_PIXEL_TO_METER_RATIO / CUSTOM_PIXEL_TO_METER_RATIO,
                "meter_to_inch": 39.3701,
                "inch_to_meter": 1.0 / 39.3701,
                "meter_to_foot": 3.28084,
                "foot_to_meter": 1.0 / 3.28084
            },
            "time": {
                "hz_to_fps": 1.0,
                "fps_to_hz": 1.0,
                "second_to_millisecond": 1000.0,
                "millisecond_to_second": 0.001,
                "crimaldi_fps_to_target": TARGET_FPS / CRIMALDI_FRAME_RATE_HZ,
                "custom_fps_to_target": TARGET_FPS / CUSTOM_FRAME_RATE_HZ,
                "target_to_crimaldi_fps": CRIMALDI_FRAME_RATE_HZ / TARGET_FPS,
                "target_to_custom_fps": CUSTOM_FRAME_RATE_HZ / TARGET_FPS
            },
            "intensity": {
                "normalized_to_8bit": 255.0,
                "8bit_to_normalized": 1.0 / 255.0,
                "normalized_to_16bit": 65535.0,
                "16bit_to_normalized": 1.0 / 65535.0
            }
        }

    def _build_unit_categories(self) -> Dict[str, str]:
        """Build unit category mappings for organized conversion access."""
        return {
            "meter": "length",
            "pixel": "length",
            "inch": "length",
            "foot": "length",
            "second": "time",
            "millisecond": "time",
            "fps": "time",
            "hz": "time",
            "normalized": "intensity",
            "8bit": "intensity",
            "16bit": "intensity"
        }

    def convert(
        self,
        value: float,
        source_unit: str,
        target_unit: str,
        validate_accuracy: bool = True
    ) -> float:
        """
        Convert value between units with precision validation and accuracy verification
        for scientific computing applications.

        Args:
            value: Numerical value to convert
            source_unit: Source unit for conversion
            target_unit: Target unit for conversion
            validate_accuracy: Whether to validate conversion accuracy

        Returns:
            Converted value with precision metadata and validation status
        """
        # Validate source and target units
        if not isinstance(value, (int, float)):
            raise TypeError(f"Value must be numeric, got {type(value)}")
        
        if not isinstance(source_unit, str) or not isinstance(target_unit, str):
            raise TypeError("Units must be specified as strings")

        # If units are the same, return original value
        if source_unit == target_unit:
            return float(value)

        # Look up conversion factor in registry
        conversion_factor = self._get_conversion_factor(source_unit, target_unit)
        
        # Apply conversion with precision preservation
        converted_value = value * conversion_factor
        
        # Validate accuracy if validate_accuracy is enabled
        if validate_accuracy and self.validation_enabled:
            # Round-trip conversion check for accuracy validation
            reverse_factor = self._get_conversion_factor(target_unit, source_unit)
            round_trip_value = converted_value * reverse_factor
            conversion_error = abs(value - round_trip_value) / abs(value) if value != 0 else abs(round_trip_value)
            
            if conversion_error > self.conversion_precision:
                raise ValueError(f"Conversion accuracy validation failed: error {conversion_error} > {self.conversion_precision}")

        return converted_value

    def _get_conversion_factor(self, source_unit: str, target_unit: str) -> float:
        """Get conversion factor between two units from the conversion registry."""
        # Determine unit categories
        source_category = self._get_unit_category(source_unit)
        target_category = self._get_unit_category(target_unit)
        
        if source_category != target_category:
            raise ValueError(f"Cannot convert between different unit categories: {source_category} vs {target_category}")

        # Look for direct conversion
        conversion_key = f"{source_unit}_to_{target_unit}"
        if conversion_key in self.conversion_registry[source_category]:
            return self.conversion_registry[source_category][conversion_key]
        
        # Look for reverse conversion
        reverse_key = f"{target_unit}_to_{source_unit}"
        if reverse_key in self.conversion_registry[source_category]:
            return 1.0 / self.conversion_registry[source_category][reverse_key]
        
        # Try to find path through intermediate units (simple case)
        for intermediate_conversion in self.conversion_registry[source_category]:
            if intermediate_conversion.startswith(source_unit + "_to_"):
                intermediate_unit = intermediate_conversion.replace(source_unit + "_to_", "")
                try:
                    factor1 = self.conversion_registry[source_category][intermediate_conversion]
                    factor2 = self._get_conversion_factor(intermediate_unit, target_unit)
                    return factor1 * factor2
                except (KeyError, ValueError):
                    continue
        
        raise ValueError(f"No conversion path found from {source_unit} to {target_unit}")

    def _get_unit_category(self, unit: str) -> str:
        """Determine the category of a given unit."""
        # Check exact matches first
        if unit in self.unit_categories:
            return self.unit_categories[unit]
        
        # Check for partial matches
        for unit_key, category in self.unit_categories.items():
            if unit_key in unit:
                return category
        
        raise ValueError(f"Unknown unit category for: {unit}")

    def register_conversion(
        self,
        source_unit: str,
        target_unit: str,
        conversion_factor: float,
        bidirectional: bool = True
    ) -> bool:
        """
        Register new unit conversion with validation and precision checking
        for extensible unit support.

        Args:
            source_unit: Source unit for the new conversion
            target_unit: Target unit for the new conversion
            conversion_factor: Conversion factor from source to target
            bidirectional: Whether to register reverse conversion automatically

        Returns:
            True if registration successful with validation status and registry update
        """
        # Validate conversion factor and unit specifications
        if not isinstance(conversion_factor, (int, float)):
            raise TypeError("Conversion factor must be numeric")
        
        if conversion_factor <= 0:
            raise ValueError("Conversion factor must be positive")
        
        if not validate_constant_precision(conversion_factor, f"conversion_{source_unit}_to_{target_unit}"):
            raise ValueError(f"Conversion factor does not meet precision requirements: {conversion_factor}")

        # Determine unit category
        try:
            category = self._get_unit_category(source_unit)
        except ValueError:
            # Create new category if needed
            category = "custom"
            if category not in self.conversion_registry:
                self.conversion_registry[category] = {}
            self.unit_categories[source_unit] = category
            self.unit_categories[target_unit] = category

        # Register conversion in conversion registry
        conversion_key = f"{source_unit}_to_{target_unit}"
        self.conversion_registry[category][conversion_key] = conversion_factor
        
        # Add bidirectional conversion if bidirectional is enabled
        if bidirectional:
            reverse_key = f"{target_unit}_to_{source_unit}"
            self.conversion_registry[category][reverse_key] = 1.0 / conversion_factor

        return True


# =============================================================================
# MODULE EXPORTS AND PUBLIC API
# =============================================================================

__all__ = [
    # Global constants
    "NUMERICAL_PRECISION_THRESHOLD",
    "DEFAULT_CORRELATION_THRESHOLD",
    "REPRODUCIBILITY_THRESHOLD",
    "PROCESSING_TIME_TARGET_SECONDS",
    "BATCH_COMPLETION_TARGET_HOURS",
    "ERROR_RATE_THRESHOLD",
    "CRIMALDI_PIXEL_TO_METER_RATIO",
    "CUSTOM_PIXEL_TO_METER_RATIO",
    "TARGET_ARENA_WIDTH_METERS",
    "TARGET_ARENA_HEIGHT_METERS",
    "SPATIAL_ACCURACY_THRESHOLD",
    "TARGET_FPS",
    "CRIMALDI_FRAME_RATE_HZ",
    "CUSTOM_FRAME_RATE_HZ",
    "TEMPORAL_ACCURACY_THRESHOLD",
    "ANTI_ALIASING_CUTOFF_RATIO",
    "MOTION_PRESERVATION_THRESHOLD",
    "TARGET_INTENSITY_MIN",
    "TARGET_INTENSITY_MAX",
    "INTENSITY_CALIBRATION_ACCURACY",
    "GAMMA_CORRECTION_DEFAULT",
    "HISTOGRAM_BINS",
    "OUTLIER_DETECTION_SIGMA",
    "STATISTICAL_SIGNIFICANCE_LEVEL",
    "CONFIDENCE_INTERVAL_DEFAULT",
    "MEMORY_LIMIT_GB",
    "MAX_WORKERS_DEFAULT",
    "CACHE_SIZE_LIMIT_GB",
    "TIMEOUT_SECONDS_DEFAULT",
    
    # Enumeration classes
    "ConstantCategory",
    "UnitSystem",
    "FormatType",
    
    # Utility functions
    "get_performance_thresholds",
    "get_statistical_constants",
    "get_normalization_constants",
    "get_physical_constants",
    "validate_constant_precision",
    "calculate_derived_constants",
    
    # Main classes
    "PhysicalConstants",
    "PerformanceThresholds",
    "UnitConverter"
]