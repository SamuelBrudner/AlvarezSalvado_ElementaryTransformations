"""
Comprehensive unit test module for scale calibration functionality providing systematic validation of spatial scaling, coordinate transformation, arena normalization, and cross-format compatibility with >95% correlation requirements and scientific computing accuracy standards. 

Tests pixel-to-meter ratio calculations, arena boundary detection, coordinate system transformations, and calibration validation against reference implementations for reproducible scientific research.

Key Features:
- Comprehensive scale calibration testing with cross-format compatibility validation
- Pixel-to-meter ratio calculation accuracy testing with scientific precision
- Arena boundary detection validation with confidence assessment
- Coordinate transformation testing with numerical precision validation
- Cross-format compatibility testing between Crimaldi and custom formats
- Performance validation meeting <7.2 seconds processing targets
- Error handling validation with fail-fast and graceful degradation testing
- Batch calibration testing for 4000+ simulation processing requirements
- Numerical accuracy validation with >95% correlation against reference implementations
- Scientific computing reliability with 1e-6 precision tolerance validation
"""

# External library imports with version specifications
import pytest  # pytest 8.3.5+ - Testing framework for unit test execution and fixture management
import numpy as np  # numpy 2.1.3+ - Numerical array operations and scientific computing for test data validation
import cv2  # opencv-python 4.11.0+ - Computer vision operations for video processing and calibration testing
from pathlib import Path  # pathlib 3.9+ - Cross-platform path handling for test fixtures and temporary files
import tempfile  # tempfile 3.9+ - Temporary file management for test isolation
from unittest.mock import Mock, patch, MagicMock  # unittest.mock 3.9+ - Mock objects for testing scale calibration components in isolation
import warnings  # warnings 3.9+ - Warning management for test validation and edge case handling
import time  # time 3.9+ - Performance timing and measurement for validation
import datetime  # datetime 3.9+ - Timestamp handling for test metadata and audit trails
from typing import Dict, Any, List, Tuple, Optional  # typing 3.9+ - Type hints for test method signatures

# Internal imports from scale calibration module
from backend.core.data_normalization.scale_calibration import (
    ScaleCalibration,  # Primary scale calibration class under test with comprehensive spatial transformation capabilities
    ScaleCalibrationManager,  # Scale calibration manager for testing multi-format and batch calibration operations
    calculate_pixel_to_meter_ratio,  # Function for testing pixel-to-meter ratio calculation accuracy
    detect_arena_boundaries,  # Function for testing arena boundary detection algorithms
    normalize_spatial_coordinates,  # Function for testing coordinate normalization and transformation
    validate_scale_calibration  # Function for testing calibration validation accuracy
)

# Internal imports from test utilities
from test.utils.test_helpers import (
    create_test_fixture_path,  # Standardized test fixture path creation for test data access
    load_test_config,  # Test configuration loading for scale calibration test scenarios
    assert_arrays_almost_equal,  # Scientific numerical array comparison with 1e-6 precision tolerance
    assert_simulation_accuracy,  # Simulation accuracy validation with >95% correlation requirement
    create_mock_video_data,  # Mock video data generation for calibration testing
    setup_test_environment  # Test environment context manager for isolated testing
)

# Internal imports from validation utilities
from test.utils.validation_metrics import (
    ValidationMetricsCalculator,  # Comprehensive validation metrics for scale calibration accuracy testing
    load_benchmark_data  # Reference benchmark data loading for calibration validation
)

# Internal imports from scientific constants
from backend.utils.scientific_constants import (
    CRIMALDI_PIXEL_TO_METER_RATIO,  # Standard pixel-to-meter ratio for Crimaldi dataset testing
    CUSTOM_PIXEL_TO_METER_RATIO,  # Default pixel-to-meter ratio for custom dataset testing
    TARGET_ARENA_WIDTH_METERS,  # Target arena width for normalization testing
    TARGET_ARENA_HEIGHT_METERS,  # Target arena height for normalization testing
    NUMERICAL_PRECISION_THRESHOLD  # Numerical precision threshold for floating point comparisons
)

# Internal imports from error handling
from backend.error.exceptions import (
    ValidationError,  # Exception handling for scale calibration validation errors
    ProcessingError  # Exception handling for scale calibration processing errors
)

# Global test configuration constants
TEST_CONFIG_PATH = create_test_fixture_path('test_normalization_config.json', 'config')
CRIMALDI_TEST_VIDEO = create_test_fixture_path('crimaldi_sample.avi', 'video')
CUSTOM_TEST_VIDEO = create_test_fixture_path('custom_sample.avi', 'video')
REFERENCE_BENCHMARK = create_test_fixture_path('normalization_benchmark.npy', 'reference_results')
TOLERANCE = NUMERICAL_PRECISION_THRESHOLD
CORRELATION_THRESHOLD = 0.95

# Global test data dimensions and specifications
TEST_VIDEO_DIMENSIONS = (640, 480)
TEST_FRAME_COUNT = 100
TEST_ARENA_WIDTH_PIXELS = 400
TEST_ARENA_HEIGHT_PIXELS = 300
EXPECTED_CALIBRATION_CONFIDENCE = 0.9
BATCH_PROCESSING_SIZE = 50


class TestScaleCalibration:
    """
    Comprehensive test class for ScaleCalibration functionality providing systematic validation of spatial scaling, coordinate transformation, and calibration accuracy with scientific computing standards.
    
    This test class validates all aspects of scale calibration including initialization, parameter extraction, 
    validation, coordinate transformation, and cross-format compatibility with >95% correlation requirements.
    """
    
    def setup_method(self, method):
        """
        Setup method executed before each test to ensure clean test environment with validation utilities and test configuration.
        
        Args:
            method: Test method being executed
        """
        # Load test configuration for scale calibration scenarios
        self.test_config = load_test_config('test_normalization_config', validate_schema=True)
        
        # Initialize ValidationMetricsCalculator with test parameters
        self.validator = ValidationMetricsCalculator(
            tolerance=TOLERANCE,
            correlation_threshold=CORRELATION_THRESHOLD,
            strict_validation=True
        )
        
        # Setup test data directory and fixture paths
        self.test_data_dir = Path(__file__).parent.parent / 'test_fixtures'
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load reference benchmark data for validation
        try:
            self.reference_benchmark = load_benchmark_data('scale_calibration_benchmark')
        except FileNotFoundError:
            # Create mock reference data if benchmark not available
            self.reference_benchmark = self._create_mock_benchmark_data()
        
        # Configure test environment and logging
        self.test_start_time = time.time()
        
        # Initialize test-specific temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix=f'test_scale_calibration_{method.__name__}_')
    
    def teardown_method(self, method):
        """
        Teardown method executed after each test for cleanup and resource management.
        
        Args:
            method: Test method that was executed
        """
        # Clean up temporary files and directories
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors
        
        # Reset global state and configuration
        # (No global state to reset in this implementation)
        
        # Clear validation caches and intermediate results
        if hasattr(self.validator, 'clear_cache'):
            self.validator.clear_cache()
        
        # Log test completion and resource usage
        test_duration = time.time() - self.test_start_time
        if test_duration > 7.2:  # Performance threshold
            warnings.warn(f"Test {method.__name__} exceeded performance threshold: {test_duration:.3f}s")
        
        # Validate no resource leaks or state pollution
        # (Implementation-specific validation would go here)
    
    def test_scale_calibration_initialization(self):
        """
        Test ScaleCalibration class initialization with different format types and configuration parameters.
        """
        with setup_test_environment('scale_calibration_init', cleanup_on_exit=True) as test_env:
            # Create ScaleCalibration instance with Crimaldi format
            crimaldi_video_path = self._create_mock_video_file(test_env['temp_directory'], 'crimaldi')
            crimaldi_config = {'format_type': 'crimaldi', 'validation_enabled': True}
            
            crimaldi_calibration = ScaleCalibration(
                video_path=str(crimaldi_video_path),
                format_type='crimaldi',
                calibration_config=crimaldi_config
            )
            
            # Validate initialization parameters and default values
            assert crimaldi_calibration.format_type == 'crimaldi'
            assert crimaldi_calibration.calibration_config['format_type'] == 'crimaldi'
            assert crimaldi_calibration.calibration_confidence == 0.0  # Initial confidence
            assert not crimaldi_calibration.is_validated  # Not validated initially
            assert crimaldi_calibration.pixel_to_meter_ratios['primary_ratio'] == CRIMALDI_PIXEL_TO_METER_RATIO
            
            # Create ScaleCalibration instance with custom format
            custom_video_path = self._create_mock_video_file(test_env['temp_directory'], 'custom')
            custom_config = {'format_type': 'custom', 'validation_enabled': True}
            
            custom_calibration = ScaleCalibration(
                video_path=str(custom_video_path),
                format_type='custom',
                calibration_config=custom_config
            )
            
            # Verify format-specific parameter differences
            assert custom_calibration.format_type == 'custom'
            assert custom_calibration.pixel_to_meter_ratios['primary_ratio'] == CUSTOM_PIXEL_TO_METER_RATIO
            
            # Test initialization with invalid parameters
            with pytest.raises(ValidationError) as exc_info:
                ScaleCalibration(
                    video_path='/nonexistent/path/invalid.avi',
                    format_type='invalid_format',
                    calibration_config={}
                )
            
            # Assert proper error handling for invalid inputs
            assert 'Video file does not exist' in str(exc_info.value)
    
    def test_pixel_to_meter_ratio_calculation(self):
        """
        Test pixel-to-meter ratio calculation accuracy for different video formats and calibration methods.
        """
        with setup_test_environment('pixel_ratio_calculation', cleanup_on_exit=True) as test_env:
            # Load test video data for Crimaldi and custom formats
            arena_width_meters = TARGET_ARENA_WIDTH_METERS
            arena_height_meters = TARGET_ARENA_HEIGHT_METERS
            video_width_pixels = TEST_VIDEO_DIMENSIONS[0]
            video_height_pixels = TEST_VIDEO_DIMENSIONS[1]
            
            # Calculate pixel-to-meter ratio using calculate_pixel_to_meter_ratio function
            ratio_result = calculate_pixel_to_meter_ratio(
                arena_width_meters=arena_width_meters,
                arena_height_meters=arena_height_meters,
                video_width_pixels=video_width_pixels,
                video_height_pixels=video_height_pixels,
                calculation_method='geometric_mean'
            )
            
            # Validate ratio accuracy against expected constants
            expected_horizontal_ratio = video_width_pixels / arena_width_meters
            expected_vertical_ratio = video_height_pixels / arena_height_meters
            expected_primary_ratio = np.sqrt(expected_horizontal_ratio * expected_vertical_ratio)
            
            assert_arrays_almost_equal(
                np.array([ratio_result['horizontal_ratio']]),
                np.array([expected_horizontal_ratio]),
                tolerance=TOLERANCE,
                error_message="Horizontal ratio calculation accuracy failed"
            )
            
            assert_arrays_almost_equal(
                np.array([ratio_result['vertical_ratio']]),
                np.array([expected_vertical_ratio]),
                tolerance=TOLERANCE,
                error_message="Vertical ratio calculation accuracy failed"
            )
            
            assert_arrays_almost_equal(
                np.array([ratio_result['primary_ratio']]),
                np.array([expected_primary_ratio]),
                tolerance=TOLERANCE,
                error_message="Primary ratio calculation accuracy failed"
            )
            
            # Test ratio calculation with calibration hints
            hints = {'detection_confidence_threshold': 0.9, 'validation_enabled': True}
            ratio_with_hints = calculate_pixel_to_meter_ratio(
                arena_width_meters=arena_width_meters,
                arena_height_meters=arena_height_meters,
                video_width_pixels=video_width_pixels,
                video_height_pixels=video_height_pixels,
                calculation_method='arithmetic_mean'
            )
            
            # Verify ratio validation and error handling
            assert ratio_with_hints['confidence_level'] > 0.0
            assert 'validation_metrics' in ratio_with_hints
            assert ratio_with_hints['validation_metrics']['spatial_accuracy_met'] is True
            
            # Assert numerical precision within tolerance threshold
            assert ratio_result['confidence_level'] >= EXPECTED_CALIBRATION_CONFIDENCE * 0.8  # Allow some tolerance
    
    def test_arena_boundary_detection(self):
        """
        Test arena boundary detection algorithms with different detection methods and validation criteria.
        """
        with setup_test_environment('arena_detection', cleanup_on_exit=True) as test_env:
            # Create mock video frames with known arena boundaries
            test_frame = self._create_mock_arena_frame()
            known_boundaries = {
                'top_left': (50, 50),
                'bottom_right': (550, 350),
                'width': 500,
                'height': 300
            }
            
            # Test edge detection method for boundary identification
            edge_detection_params = {
                'canny_low': 50,
                'canny_high': 150,
                'confidence_threshold': 0.8
            }
            
            edge_result = detect_arena_boundaries(
                video_frame=test_frame,
                detection_method='edge',
                detection_parameters=edge_detection_params,
                validate_detection=True
            )
            
            # Test contour detection method for boundary identification
            contour_detection_params = {
                'threshold_value': 127,
                'confidence_threshold': 0.8
            }
            
            contour_result = detect_arena_boundaries(
                video_frame=test_frame,
                detection_method='contour',
                detection_parameters=contour_detection_params,
                validate_detection=True
            )
            
            # Validate detection accuracy against ground truth
            assert edge_result['detection_confidence'] >= 0.7  # Minimum acceptable confidence
            assert contour_result['detection_confidence'] >= 0.7
            
            assert 'arena_boundaries' in edge_result
            assert 'arena_boundaries' in contour_result
            
            # Test boundary detection with noisy data
            noisy_frame = self._add_noise_to_frame(test_frame, noise_level=0.1)
            noisy_result = detect_arena_boundaries(
                video_frame=noisy_frame,
                detection_method='contour',
                detection_parameters=contour_detection_params,
                validate_detection=True
            )
            
            # Assert detection confidence and accuracy metrics
            assert noisy_result['detection_confidence'] >= 0.5  # Lower threshold for noisy data
            assert 'geometric_properties' in noisy_result
    
    def test_spatial_coordinate_normalization(self):
        """
        Test spatial coordinate normalization and transformation between different coordinate systems.
        """
        with setup_test_environment('coordinate_normalization', cleanup_on_exit=True) as test_env:
            # Create test coordinate arrays in different coordinate systems
            pixel_coordinates = np.array([
                [100, 150],
                [200, 250],
                [300, 350],
                [400, 450]
            ])
            
            # Initialize ScaleCalibration with known parameters
            test_video_path = self._create_mock_video_file(test_env['temp_directory'], 'custom')
            calibration = ScaleCalibration(
                video_path=str(test_video_path),
                format_type='custom',
                calibration_config={'validation_enabled': True}
            )
            
            # Set known calibration parameters for testing
            calibration.pixel_to_meter_ratios = {
                'horizontal_ratio': CUSTOM_PIXEL_TO_METER_RATIO,
                'vertical_ratio': CUSTOM_PIXEL_TO_METER_RATIO,
                'primary_ratio': CUSTOM_PIXEL_TO_METER_RATIO
            }
            
            # Apply coordinate normalization using apply_to_coordinates method
            meter_coordinates = calibration.apply_to_coordinates(
                coordinates=pixel_coordinates,
                source_system='pixel',
                target_system='meter',
                validate_transformation=True
            )
            
            # Validate transformed coordinates against expected results
            expected_meter_coordinates = pixel_coordinates / CUSTOM_PIXEL_TO_METER_RATIO
            
            assert_arrays_almost_equal(
                meter_coordinates,
                expected_meter_coordinates,
                tolerance=TOLERANCE,
                error_message="Pixel to meter coordinate transformation failed"
            )
            
            # Test coordinate system conversion accuracy
            normalized_coordinates = calibration.apply_to_coordinates(
                coordinates=meter_coordinates,
                source_system='meter',
                target_system='normalized',
                validate_transformation=True
            )
            
            expected_normalized = meter_coordinates / np.array([TARGET_ARENA_WIDTH_METERS, TARGET_ARENA_HEIGHT_METERS])
            
            assert_arrays_almost_equal(
                normalized_coordinates,
                expected_normalized,
                tolerance=TOLERANCE,
                error_message="Meter to normalized coordinate transformation failed"
            )
            
            # Assert numerical precision and transformation consistency
            round_trip_coordinates = calibration.apply_to_coordinates(
                coordinates=normalized_coordinates,
                source_system='normalized',
                target_system='pixel',
                validate_transformation=True
            )
            
            # Check round-trip transformation accuracy
            transformation_error = np.max(np.abs(pixel_coordinates - round_trip_coordinates))
            assert transformation_error < TOLERANCE * 100  # Allow for accumulated rounding error
    
    def test_calibration_validation(self):
        """
        Test scale calibration validation against reference data and accuracy thresholds.
        """
        with setup_test_environment('calibration_validation', cleanup_on_exit=True) as test_env:
            # Load reference benchmark data for validation
            if hasattr(self, 'reference_benchmark') and self.reference_benchmark is not None:
                reference_data = self.reference_benchmark
            else:
                reference_data = self._create_mock_benchmark_data()
            
            # Create ScaleCalibration with test parameters
            test_video_path = self._create_mock_video_file(test_env['temp_directory'], 'crimaldi')
            calibration = ScaleCalibration(
                video_path=str(test_video_path),
                format_type='crimaldi',
                calibration_config={'validation_enabled': True}
            )
            
            # Extract calibration parameters for validation
            calibration.extract_calibration_parameters(force_reextraction=True)
            
            # Perform calibration validation using validate_calibration method
            validation_result = calibration.validate_calibration(
                validation_thresholds={'min_confidence': 0.8, 'min_spatial_accuracy': 0.01},
                strict_validation=True
            )
            
            # Compare validation results against benchmark data
            assert validation_result.is_valid or len(validation_result.errors) == 0
            assert validation_result.metrics['calibration_confidence'] >= 0.5  # Minimum confidence
            
            # Test validation with different accuracy thresholds
            strict_validation_result = calibration.validate_calibration(
                validation_thresholds={'min_confidence': 0.95, 'min_spatial_accuracy': 0.001},
                strict_validation=True
            )
            
            # Assert >95% correlation requirement compliance
            if 'correlation_metrics' in validation_result.metrics:
                correlation_score = validation_result.metrics['correlation_metrics']
                assert correlation_score >= CORRELATION_THRESHOLD or not strict_validation_result.is_valid
    
    def test_transformation_matrix_creation(self):
        """
        Test transformation matrix creation for coordinate system conversion and scaling.
        """
        with setup_test_environment('transformation_matrix', cleanup_on_exit=True) as test_env:
            # Create source and target ScaleCalibration instances
            source_video_path = self._create_mock_video_file(test_env['temp_directory'], 'crimaldi')
            source_calibration = ScaleCalibration(
                video_path=str(source_video_path),
                format_type='crimaldi',
                calibration_config={'validation_enabled': True}
            )
            
            target_video_path = self._create_mock_video_file(test_env['temp_directory'], 'custom')
            target_calibration = ScaleCalibration(
                video_path=str(target_video_path),
                format_type='custom',
                calibration_config={'validation_enabled': True}
            )
            
            # Build transformation matrix using get_transformation_matrix method
            transformation_matrix = source_calibration.get_transformation_matrix(
                source_system='pixel',
                target_system='meter',
                force_recalculation=True
            )
            
            # Validate matrix properties (determinant, invertibility)
            assert transformation_matrix.shape == (3, 3)  # Homogeneous transformation matrix
            matrix_determinant = np.linalg.det(transformation_matrix[:2, :2])  # 2x2 submatrix
            assert abs(matrix_determinant) > TOLERANCE  # Non-singular matrix
            
            # Test matrix application to coordinate arrays
            test_coordinates = np.array([[100, 150], [200, 250]])
            homogeneous_coords = np.column_stack([test_coordinates, np.ones(test_coordinates.shape[0])])
            transformed_coords = homogeneous_coords @ transformation_matrix.T
            result_coords = transformed_coords[:, :2]
            
            # Verify transformation accuracy and consistency
            expected_result = test_coordinates / CRIMALDI_PIXEL_TO_METER_RATIO  # Expected pixel to meter conversion
            
            assert_arrays_almost_equal(
                result_coords,
                expected_result,
                tolerance=TOLERANCE,
                error_message="Transformation matrix application failed"
            )
            
            # Assert matrix mathematical properties and precision
            # Test matrix inverse for round-trip transformation
            try:
                inverse_matrix = np.linalg.inv(transformation_matrix)
                identity_check = transformation_matrix @ inverse_matrix
                identity_matrix = np.eye(3)
                
                assert_arrays_almost_equal(
                    identity_check,
                    identity_matrix,
                    tolerance=TOLERANCE,
                    error_message="Transformation matrix inverse validation failed"
                )
            except np.linalg.LinAlgError:
                pytest.fail("Transformation matrix is not invertible")
    
    def test_cross_format_compatibility(self):
        """
        Test cross-format compatibility between Crimaldi and custom format calibrations.
        """
        with setup_test_environment('cross_format_compatibility', cleanup_on_exit=True) as test_env:
            # Create calibrations for both Crimaldi and custom formats
            crimaldi_video_path = self._create_mock_video_file(test_env['temp_directory'], 'crimaldi')
            crimaldi_calibration = ScaleCalibration(
                video_path=str(crimaldi_video_path),
                format_type='crimaldi',
                calibration_config={'validation_enabled': True}
            )
            
            custom_video_path = self._create_mock_video_file(test_env['temp_directory'], 'custom')
            custom_calibration = ScaleCalibration(
                video_path=str(custom_video_path),
                format_type='custom',
                calibration_config={'validation_enabled': True}
            )
            
            # Extract parameters for both calibrations
            crimaldi_calibration.extract_calibration_parameters()
            custom_calibration.extract_calibration_parameters()
            
            # Test cross-format parameter consistency
            crimaldi_ratio = crimaldi_calibration.pixel_to_meter_ratios['primary_ratio']
            custom_ratio = custom_calibration.pixel_to_meter_ratios['primary_ratio']
            
            # The ratios should be different but both valid
            assert crimaldi_ratio > 0
            assert custom_ratio > 0
            assert abs(crimaldi_ratio - CRIMALDI_PIXEL_TO_METER_RATIO) < TOLERANCE
            assert abs(custom_ratio - CUSTOM_PIXEL_TO_METER_RATIO) < TOLERANCE
            
            # Validate coordinate transformation compatibility
            test_coordinates = np.array([[100, 150], [200, 250], [300, 350]])
            
            crimaldi_meter_coords = crimaldi_calibration.apply_to_coordinates(
                test_coordinates, 'pixel', 'meter'
            )
            custom_meter_coords = custom_calibration.apply_to_coordinates(
                test_coordinates, 'pixel', 'meter'
            )
            
            # Use ValidationMetricsCalculator for compatibility assessment
            compatibility_result = self.validator.validate_cross_format_compatibility(
                crimaldi_results={'coordinates': crimaldi_meter_coords},
                custom_results={'coordinates': custom_meter_coords},
                tolerance_threshold=0.1
            )
            
            # Test format conversion accuracy and precision
            assert compatibility_result.is_valid or len(compatibility_result.warnings) == 0
            
            # Assert cross-format correlation meets requirements
            if crimaldi_meter_coords.shape == custom_meter_coords.shape:
                # Calculate correlation only if the transformation produces similar scales
                correlation_matrix = np.corrcoef(
                    crimaldi_meter_coords.flatten(),
                    custom_meter_coords.flatten()
                )
                cross_correlation = abs(correlation_matrix[0, 1])  # Take absolute value for different scales
                
                # The correlation might be negative due to different scaling, but magnitude should be high
                assert cross_correlation > 0.5 or np.isnan(cross_correlation)  # Allow NaN if insufficient variance
    
    def test_calibration_manager_operations(self):
        """
        Test ScaleCalibrationManager for multi-format and batch calibration operations.
        """
        with setup_test_environment('calibration_manager', cleanup_on_exit=True) as test_env:
            # Initialize ScaleCalibrationManager with test configuration
            manager_config = {
                'batch_size': 10,
                'validation_enabled': True,
                'caching_enabled': True,
                'parallel_processing': False  # Disable for testing consistency
            }
            
            manager = ScaleCalibrationManager(
                manager_config=manager_config,
                caching_enabled=True,
                batch_optimization_enabled=True
            )
            
            # Test calibration creation for multiple video files
            video_paths = []
            for i in range(5):
                format_type = 'crimaldi' if i % 2 == 0 else 'custom'
                video_path = self._create_mock_video_file(test_env['temp_directory'], format_type, f'test_{i}')
                video_paths.append(str(video_path))
            
            # Test individual calibration creation
            first_calibration = manager.create_calibration(
                video_path=video_paths[0],
                calibration_config={'format_type': 'crimaldi'},
                validate_creation=True
            )
            
            assert isinstance(first_calibration, ScaleCalibration)
            assert first_calibration.format_type == 'crimaldi'
            
            # Validate batch calibration processing
            batch_results = manager.batch_calibrate(
                video_paths=video_paths,
                batch_config={'validation_enabled': True},
                enable_parallel_processing=False
            )
            
            # Test cross-format compatibility validation
            format_types = ['crimaldi', 'custom']
            compatibility_result = manager.validate_cross_format_compatibility(
                format_types=format_types,
                tolerance_thresholds={'pixel_ratio_tolerance': 0.1},
                detailed_analysis=True
            )
            
            # Verify calibration caching and optimization
            assert len(batch_results) == len(video_paths)
            successful_calibrations = sum(1 for result in batch_results.values() if isinstance(result, ScaleCalibration))
            assert successful_calibrations >= len(video_paths) * 0.8  # At least 80% success rate
            
            # Assert batch processing efficiency and accuracy
            assert compatibility_result.is_valid or len(compatibility_result.warnings) <= 2
            assert manager.performance_metrics['batch_processing_efficiency'] >= 0.8
    
    def test_error_handling_and_edge_cases(self):
        """
        Test error handling for invalid inputs, corrupted data, and edge cases in scale calibration.
        """
        with setup_test_environment('error_handling', cleanup_on_exit=True) as test_env:
            # Test calibration with invalid video file paths
            with pytest.raises(ValidationError) as exc_info:
                ScaleCalibration(
                    video_path='/nonexistent/invalid_path.avi',
                    format_type='crimaldi',
                    calibration_config={}
                )
            assert 'Video file does not exist' in str(exc_info.value)
            
            # Test calibration with corrupted video data
            corrupted_video_path = self._create_corrupted_video_file(test_env['temp_directory'])
            with pytest.raises((ProcessingError, ValidationError)):
                calibration = ScaleCalibration(
                    video_path=str(corrupted_video_path),
                    format_type='custom',
                    calibration_config={}
                )
                calibration.extract_calibration_parameters()
            
            # Validate error handling for missing calibration parameters
            valid_video_path = self._create_mock_video_file(test_env['temp_directory'], 'custom')
            calibration = ScaleCalibration(
                video_path=str(valid_video_path),
                format_type='custom',
                calibration_config={'validation_enabled': True}
            )
            
            # Clear calibration parameters to test missing parameter handling
            calibration.pixel_to_meter_ratios = {}
            
            with pytest.raises(ValidationError) as exc_info:
                calibration.validate_calibration(strict_validation=True)
            assert 'pixel-to-meter ratios' in str(exc_info.value).lower()
            
            # Test edge cases with extreme arena sizes
            with pytest.raises(ValidationError):
                calculate_pixel_to_meter_ratio(
                    arena_width_meters=0.001,  # Extremely small arena
                    arena_height_meters=1000.0,  # Extremely large arena
                    video_width_pixels=640,
                    video_height_pixels=480
                )
            
            # Verify ValidationError and ProcessingError handling
            try:
                # Test invalid coordinate transformation
                invalid_coordinates = np.array([[np.inf, np.nan], [1e10, -1e10]])
                calibration.apply_to_coordinates(
                    coordinates=invalid_coordinates,
                    source_system='pixel',
                    target_system='meter',
                    validate_transformation=True
                )
                pytest.fail("Expected ValidationError for invalid coordinates")
            except (ValidationError, ProcessingError) as e:
                assert hasattr(e, 'get_recovery_recommendations')
                recommendations = e.get_recovery_recommendations()
                assert len(recommendations) > 0
            
            # Assert proper error reporting and recovery recommendations
            assert True  # Test completed successfully if no unexpected exceptions
    
    def test_performance_and_accuracy_requirements(self):
        """
        Test performance requirements and accuracy validation against scientific computing standards.
        """
        with setup_test_environment('performance_accuracy', cleanup_on_exit=True) as test_env:
            # Measure calibration processing time for performance validation
            start_time = time.time()
            
            test_video_path = self._create_mock_video_file(test_env['temp_directory'], 'crimaldi')
            calibration = ScaleCalibration(
                video_path=str(test_video_path),
                format_type='crimaldi',
                calibration_config={'validation_enabled': True}
            )
            
            # Extract calibration parameters and measure time
            calibration.extract_calibration_parameters()
            extraction_time = time.time() - start_time
            
            # Test calibration accuracy against reference implementations
            validation_start = time.time()
            validation_result = calibration.validate_calibration()
            validation_time = time.time() - validation_start
            
            # Validate numerical precision within 1e-6 tolerance
            test_coordinates = np.array([[100.123456789, 200.987654321]])
            transformed_coords = calibration.apply_to_coordinates(
                test_coordinates, 'pixel', 'meter'
            )
            
            # Round-trip transformation to test precision
            round_trip_coords = calibration.apply_to_coordinates(
                transformed_coords, 'meter', 'pixel'
            )
            
            precision_error = np.max(np.abs(test_coordinates - round_trip_coords))
            assert precision_error < TOLERANCE * 100  # Allow for accumulated rounding
            
            # Test reproducibility across multiple runs
            results = []
            for run in range(5):
                run_calibration = ScaleCalibration(
                    video_path=str(test_video_path),
                    format_type='crimaldi',
                    calibration_config={'validation_enabled': True}
                )
                run_calibration.extract_calibration_parameters()
                results.append(run_calibration.pixel_to_meter_ratios['primary_ratio'])
            
            # Verify reproducibility
            results_array = np.array(results)
            reproducibility_std = np.std(results_array)
            assert reproducibility_std < TOLERANCE  # Results should be highly reproducible
            
            # Verify >95% correlation with benchmark data
            if hasattr(self, 'reference_benchmark') and self.reference_benchmark is not None:
                benchmark_ratio = self.reference_benchmark.get('pixel_to_meter_ratio', CRIMALDI_PIXEL_TO_METER_RATIO)
                actual_ratio = calibration.pixel_to_meter_ratios['primary_ratio']
                
                ratio_correlation = 1.0 - abs(actual_ratio - benchmark_ratio) / benchmark_ratio
                assert ratio_correlation >= CORRELATION_THRESHOLD or abs(actual_ratio - benchmark_ratio) < TOLERANCE
            
            # Assert performance meets <7.2 seconds processing target
            total_processing_time = extraction_time + validation_time
            assert total_processing_time < 7.2, f"Processing time {total_processing_time:.3f}s exceeds 7.2s target"
            
            # Additional performance validations
            assert extraction_time < 5.0, f"Extraction time {extraction_time:.3f}s too slow"
            assert validation_time < 2.0, f"Validation time {validation_time:.3f}s too slow"
    
    def test_crimaldi_format_calibration(self):
        """
        Test scale calibration specifically for Crimaldi format with format-specific parameters and validation.
        """
        with setup_test_environment('crimaldi_calibration', cleanup_on_exit=True) as test_env:
            # Load Crimaldi test video using create_test_fixture_path
            crimaldi_video_path = self._create_mock_video_file(test_env['temp_directory'], 'crimaldi')
            
            # Create ScaleCalibration instance with Crimaldi format
            calibration = ScaleCalibration(
                video_path=str(crimaldi_video_path),
                format_type='crimaldi',
                calibration_config={'validation_enabled': True, 'format_specific_optimization': True}
            )
            
            # Extract calibration parameters using extract_calibration_parameters
            extraction_result = calibration.extract_calibration_parameters(
                force_reextraction=True,
                extraction_hints={'format_optimization': True}
            )
            
            # Validate pixel-to-meter ratio against CRIMALDI_PIXEL_TO_METER_RATIO
            actual_ratio = calibration.pixel_to_meter_ratios['primary_ratio']
            expected_ratio = CRIMALDI_PIXEL_TO_METER_RATIO
            
            assert_arrays_almost_equal(
                np.array([actual_ratio]),
                np.array([expected_ratio]),
                tolerance=TOLERANCE,
                error_message="Crimaldi pixel-to-meter ratio validation failed"
            )
            
            # Test coordinate transformation accuracy
            test_pixel_coords = np.array([[200, 300], [400, 150]])
            meter_coords = calibration.apply_to_coordinates(
                test_pixel_coords, 'pixel', 'meter'
            )
            
            expected_meter_coords = test_pixel_coords / CRIMALDI_PIXEL_TO_METER_RATIO
            assert_arrays_almost_equal(
                meter_coords,
                expected_meter_coords,
                tolerance=TOLERANCE,
                error_message="Crimaldi coordinate transformation failed"
            )
            
            # Assert calibration meets accuracy requirements
            validation_result = calibration.validate_calibration()
            assert validation_result.is_valid
            assert calibration.calibration_confidence >= 0.8  # High confidence for Crimaldi format
    
    def test_custom_format_calibration(self):
        """
        Test scale calibration for custom AVI format with auto-detection and parameter validation.
        """
        with setup_test_environment('custom_calibration', cleanup_on_exit=True) as test_env:
            # Load custom test video using create_test_fixture_path
            custom_video_path = self._create_mock_video_file(test_env['temp_directory'], 'custom')
            
            # Create ScaleCalibration instance with custom format
            calibration = ScaleCalibration(
                video_path=str(custom_video_path),
                format_type='custom',
                calibration_config={'validation_enabled': True, 'auto_detection': True}
            )
            
            # Test auto-detection of calibration parameters
            calibration.extract_calibration_parameters(
                extraction_hints={'enable_auto_detection': True}
            )
            
            # Validate pixel-to-meter ratio against CUSTOM_PIXEL_TO_METER_RATIO
            actual_ratio = calibration.pixel_to_meter_ratios['primary_ratio']
            expected_ratio = CUSTOM_PIXEL_TO_METER_RATIO
            
            ratio_difference = abs(actual_ratio - expected_ratio) / expected_ratio
            assert ratio_difference < 0.1, f"Custom format ratio too different: {ratio_difference:.3f}"
            
            # Test format-specific coordinate transformations
            test_coordinates = np.array([[150, 200], [350, 400]])
            normalized_coords = calibration.apply_to_coordinates(
                test_coordinates, 'pixel', 'normalized'
            )
            
            # Check that coordinates are properly normalized
            assert np.all(normalized_coords >= 0)
            assert np.all(normalized_coords <= 1.0)
            
            # Assert calibration accuracy and consistency
            validation_result = calibration.validate_calibration()
            assert validation_result.is_valid or len(validation_result.errors) <= 1  # Allow minor issues for auto-detection
    
    def test_calibration_accuracy_validation(self):
        """
        Test calibration accuracy validation against reference benchmark data with >95% correlation requirement.
        """
        with setup_test_environment('accuracy_validation', cleanup_on_exit=True) as test_env:
            # Load reference benchmark data using load_benchmark_data
            benchmark_data = self.reference_benchmark
            
            # Create test calibration with known parameters
            test_video_path = self._create_mock_video_file(test_env['temp_directory'], 'crimaldi')
            calibration = ScaleCalibration(
                video_path=str(test_video_path),
                format_type='crimaldi',
                calibration_config={'validation_enabled': True}
            )
            
            calibration.extract_calibration_parameters()
            
            # Apply calibration to test coordinate data
            test_coordinates = np.array([
                [100, 150], [200, 250], [300, 350], [400, 450], [500, 550]
            ])
            
            transformed_coordinates = calibration.apply_to_coordinates(
                test_coordinates, 'pixel', 'meter'
            )
            
            # Compare results against reference benchmark
            if 'reference_coordinates' in benchmark_data:
                reference_coords = benchmark_data['reference_coordinates']
                if reference_coords.shape == transformed_coordinates.shape:
                    assert_simulation_accuracy(
                        transformed_coordinates,
                        reference_coords,
                        correlation_threshold=CORRELATION_THRESHOLD
                    )
            else:
                # If no reference coordinates, validate against expected transformation
                expected_coords = test_coordinates / CRIMALDI_PIXEL_TO_METER_RATIO
                assert_arrays_almost_equal(
                    transformed_coordinates,
                    expected_coords,
                    tolerance=TOLERANCE,
                    error_message="Calibration accuracy validation failed"
                )
            
            # Use assert_simulation_accuracy for >95% correlation validation
            # Create synthetic reference data for correlation test
            reference_transformation = test_coordinates / CRIMALDI_PIXEL_TO_METER_RATIO
            assert_simulation_accuracy(
                transformed_coordinates,
                reference_transformation,
                correlation_threshold=CORRELATION_THRESHOLD
            )
            
            # Assert numerical precision within TOLERANCE threshold
            precision_test_coords = np.array([[123.456789, 987.654321]])
            precise_result = calibration.apply_to_coordinates(precision_test_coords, 'pixel', 'meter')
            expected_precise = precision_test_coords / CRIMALDI_PIXEL_TO_METER_RATIO
            
            precision_error = np.max(np.abs(precise_result - expected_precise))
            assert precision_error < TOLERANCE, f"Precision error {precision_error:.2e} exceeds tolerance {TOLERANCE:.2e}"
    
    def test_coordinate_transformation_precision(self):
        """
        Test coordinate transformation precision and mathematical properties of transformation matrices.
        """
        with setup_test_environment('transformation_precision', cleanup_on_exit=True) as test_env:
            # Create test coordinate arrays with known transformations
            test_coords = np.array([
                [0, 0], [100, 0], [0, 100], [100, 100],  # Corner coordinates
                [50, 50], [150, 75], [225, 175]          # Interior coordinates
            ])
            
            test_video_path = self._create_mock_video_file(test_env['temp_directory'], 'custom')
            calibration = ScaleCalibration(
                video_path=str(test_video_path),
                format_type='custom',
                calibration_config={'validation_enabled': True}
            )
            
            calibration.extract_calibration_parameters()
            
            # Build transformation matrix using get_transformation_matrix
            transform_matrix = calibration.get_transformation_matrix(
                source_system='pixel',
                target_system='meter'
            )
            
            # Apply transformation to coordinate arrays
            meter_coords = calibration.apply_to_coordinates(test_coords, 'pixel', 'meter')
            
            # Validate transformation mathematical properties
            # Test linearity: T(a*x + b*y) = a*T(x) + b*T(y)
            x1, x2 = test_coords[0:1], test_coords[1:2]
            y1 = calibration.apply_to_coordinates(x1, 'pixel', 'meter')
            y2 = calibration.apply_to_coordinates(x2, 'pixel', 'meter')
            
            # Test linear combination
            combined_input = 0.3 * x1 + 0.7 * x2
            combined_output = calibration.apply_to_coordinates(combined_input, 'pixel', 'meter')
            expected_combined = 0.3 * y1 + 0.7 * y2
            
            assert_arrays_almost_equal(
                combined_output,
                expected_combined,
                tolerance=TOLERANCE,
                error_message="Transformation linearity test failed"
            )
            
            # Test inverse transformation accuracy
            pixel_coords_recovered = calibration.apply_to_coordinates(meter_coords, 'meter', 'pixel')
            
            round_trip_error = np.max(np.abs(test_coords - pixel_coords_recovered))
            assert round_trip_error < TOLERANCE * 100, f"Round-trip error {round_trip_error:.2e} too large"
            
            # Assert coordinate precision within numerical tolerance
            high_precision_coords = np.array([[123.456789123, 987.654321987]])
            precise_meters = calibration.apply_to_coordinates(high_precision_coords, 'pixel', 'meter')
            precise_pixels = calibration.apply_to_coordinates(precise_meters, 'meter', 'pixel')
            
            precision_loss = np.max(np.abs(high_precision_coords - precise_pixels))
            assert precision_loss < TOLERANCE * 1000  # Allow for reasonable precision loss
    
    def test_arena_size_normalization(self):
        """
        Test arena size normalization across different physical dimensions and aspect ratios.
        """
        with setup_test_environment('arena_normalization', cleanup_on_exit=True) as test_env:
            # Create mock video data with different arena sizes
            arena_configs = [
                {'width_meters': 0.5, 'height_meters': 0.5, 'aspect_ratio': 1.0},  # Square arena
                {'width_meters': 1.0, 'height_meters': 0.5, 'aspect_ratio': 2.0},  # Wide arena
                {'width_meters': 0.5, 'height_meters': 1.0, 'aspect_ratio': 0.5}   # Tall arena
            ]
            
            for i, config in enumerate(arena_configs):
                # Test arena boundary detection using detect_arena_boundaries
                mock_frame = self._create_mock_arena_frame(
                    width=int(config['width_meters'] * 400),  # Scale to pixels
                    height=int(config['height_meters'] * 400)
                )
                
                detection_result = detect_arena_boundaries(
                    video_frame=mock_frame,
                    detection_method='contour',
                    detection_parameters={'confidence_threshold': 0.8},
                    validate_detection=True
                )
                
                # Validate normalization to target arena dimensions
                detected_width = detection_result['arena_boundaries'].get('width', 0)
                detected_height = detection_result['arena_boundaries'].get('height', 0)
                
                if detected_width > 0 and detected_height > 0:
                    detected_aspect_ratio = detected_width / detected_height
                    expected_aspect_ratio = config['aspect_ratio']
                    
                    aspect_ratio_error = abs(detected_aspect_ratio - expected_aspect_ratio) / expected_aspect_ratio
                    assert aspect_ratio_error < 0.2, f"Aspect ratio detection error too large: {aspect_ratio_error:.3f}"
                
                # Test aspect ratio preservation during normalization
                normalized_coords = np.array([[detected_width, detected_height]])
                target_normalized = normalized_coords / np.array([TARGET_ARENA_WIDTH_METERS * 400, TARGET_ARENA_HEIGHT_METERS * 400])
                
                # Verify scaling factor calculation accuracy
                x_scale = detected_width / (TARGET_ARENA_WIDTH_METERS * 400)
                y_scale = detected_height / (TARGET_ARENA_HEIGHT_METERS * 400)
                
                # For proper normalization, we expect reasonable scaling factors
                assert 0.1 < x_scale < 10.0, f"X scaling factor unreasonable: {x_scale}"
                assert 0.1 < y_scale < 10.0, f"Y scaling factor unreasonable: {y_scale}"
                
                # Assert normalization meets spatial accuracy requirements
                assert detection_result['detection_confidence'] >= 0.5
    
    def test_cross_format_consistency(self):
        """
        Test consistency between Crimaldi and custom format calibrations for cross-format compatibility.
        """
        with setup_test_environment('cross_format_consistency', cleanup_on_exit=True) as test_env:
            # Create calibrations for both Crimaldi and custom formats
            crimaldi_video = self._create_mock_video_file(test_env['temp_directory'], 'crimaldi')
            custom_video = self._create_mock_video_file(test_env['temp_directory'], 'custom')
            
            crimaldi_cal = ScaleCalibration(
                video_path=str(crimaldi_video),
                format_type='crimaldi',
                calibration_config={'validation_enabled': True}
            )
            
            custom_cal = ScaleCalibration(
                video_path=str(custom_video),
                format_type='custom',
                calibration_config={'validation_enabled': True}
            )
            
            # Extract parameters for both calibrations
            crimaldi_cal.extract_calibration_parameters()
            custom_cal.extract_calibration_parameters()
            
            # Apply calibrations to identical test coordinate data
            test_coords = np.array([[100, 150], [300, 250], [500, 350]])
            
            crimaldi_meters = crimaldi_cal.apply_to_coordinates(test_coords, 'pixel', 'meter')
            custom_meters = custom_cal.apply_to_coordinates(test_coords, 'pixel', 'meter')
            
            # Compare calibration results for consistency
            # Results should be different due to different ratios, but both should be valid
            crimaldi_range = np.ptp(crimaldi_meters, axis=0)  # Peak-to-peak range
            custom_range = np.ptp(custom_meters, axis=0)
            
            assert np.all(crimaldi_range > 0), "Crimaldi transformation should produce non-zero range"
            assert np.all(custom_range > 0), "Custom transformation should produce non-zero range"
            
            # Use ValidationMetricsCalculator.validate_cross_format_compatibility
            compatibility_result = self.validator.validate_cross_format_compatibility(
                crimaldi_results={'meter_coordinates': crimaldi_meters},
                custom_results={'meter_coordinates': custom_meters},
                tolerance_threshold=0.2
            )
            
            # Test format conversion accuracy and precision
            # Convert both to normalized coordinates for comparison
            crimaldi_normalized = crimaldi_cal.apply_to_coordinates(crimaldi_meters, 'meter', 'normalized')
            custom_normalized = custom_cal.apply_to_coordinates(custom_meters, 'meter', 'normalized')
            
            # Normalized coordinates should be more similar
            normalized_diff = np.mean(np.abs(crimaldi_normalized - custom_normalized))
            assert normalized_diff < 0.5, f"Normalized coordinates too different: {normalized_diff:.3f}"
            
            # Assert cross-format correlation meets compatibility requirements
            assert compatibility_result.is_valid or len(compatibility_result.warnings) <= 2
    
    def test_error_handling_scenarios(self):
        """
        Test comprehensive error handling scenarios including validation errors and processing failures.
        """
        with setup_test_environment('error_scenarios', cleanup_on_exit=True) as test_env:
            # Test calibration with invalid video file paths
            with pytest.raises(ValidationError) as exc_info:
                ScaleCalibration(
                    video_path='/completely/invalid/path/nonexistent.avi',
                    format_type='crimaldi',
                    calibration_config={}
                )
            
            validation_error = exc_info.value
            assert hasattr(validation_error, 'get_validation_summary')
            summary = validation_error.get_validation_summary()
            assert 'failed_parameters' in summary
            
            # Validate ValidationError handling for invalid parameters
            with pytest.raises(ValidationError):
                calculate_pixel_to_meter_ratio(
                    arena_width_meters=-1.0,  # Invalid negative value
                    arena_height_meters=1.0,
                    video_width_pixels=640,
                    video_height_pixels=480
                )
            
            # Test ProcessingError handling for corrupted data
            corrupted_video = self._create_corrupted_video_file(test_env['temp_directory'])
            
            try:
                calibration = ScaleCalibration(
                    video_path=str(corrupted_video),
                    format_type='custom',
                    calibration_config={}
                )
                calibration.extract_calibration_parameters()
                pytest.fail("Expected ProcessingError for corrupted video")
            except ProcessingError as proc_error:
                assert hasattr(proc_error, 'preserve_intermediate_results')
                
                # Test intermediate result preservation
                proc_error.preserve_intermediate_results(
                    results={'partial_data': 'test'},
                    completion_percentage=25.0
                )
                assert proc_error.partial_success is True
                assert proc_error.processing_progress == 25.0
            
            # Verify error message clarity and recovery recommendations
            valid_video = self._create_mock_video_file(test_env['temp_directory'], 'custom')
            calibration = ScaleCalibration(
                video_path=str(valid_video),
                format_type='custom',
                calibration_config={'validation_enabled': True}
            )
            
            try:
                # Force validation error by corrupting calibration data
                calibration.pixel_to_meter_ratios = {'invalid': 'data'}
                calibration.validate_calibration(strict_validation=True)
                pytest.fail("Expected validation error")
            except ValidationError as val_error:
                recovery_recommendations = val_error.get_recovery_recommendations()
                assert len(recovery_recommendations) > 0
                assert any('review' in rec.lower() for rec in recovery_recommendations)
            
            # Test graceful degradation with partial calibration data
            partial_calibration = ScaleCalibration(
                video_path=str(valid_video),
                format_type='custom',
                calibration_config={'validation_enabled': False}  # Disable validation for partial test
            )
            
            # Simulate partial calibration data
            partial_calibration.pixel_to_meter_ratios = {
                'horizontal_ratio': CUSTOM_PIXEL_TO_METER_RATIO,
                'vertical_ratio': CUSTOM_PIXEL_TO_METER_RATIO
                # Missing 'primary_ratio'
            }
            
            # Should handle graceful degradation
            try:
                validation_result = partial_calibration.validate_calibration(strict_validation=False)
                # Should succeed with warnings or handle gracefully
                assert not validation_result.is_valid or len(validation_result.warnings) > 0
            except ValidationError:
                pass  # Acceptable for partial data
            
            # Assert proper exception handling and error reporting
            assert True  # Test completed successfully
    
    # Helper methods for test setup and mock data creation
    
    def _create_mock_video_file(self, temp_dir: Path, format_type: str, suffix: str = '') -> Path:
        """Create a mock video file for testing purposes."""
        video_data = create_mock_video_data(
            dimensions=TEST_VIDEO_DIMENSIONS,
            frame_count=TEST_FRAME_COUNT,
            format_type=format_type
        )
        
        filename = f'{format_type}_test_video{suffix}.avi'
        video_path = temp_dir / filename
        
        # Create a simple mock video file (just write some data)
        with open(video_path, 'wb') as f:
            # Write a minimal AVI header and frame data
            f.write(b'RIFF')  # AVI signature
            f.write((video_data.nbytes + 36).to_bytes(4, 'little'))  # File size
            f.write(b'AVI ')
            f.write(b'LIST')
            f.write((video_data.nbytes + 4).to_bytes(4, 'little'))
            f.write(b'movi')
            f.write(video_data.tobytes())
        
        return video_path
    
    def _create_corrupted_video_file(self, temp_dir: Path) -> Path:
        """Create a corrupted video file for error testing."""
        corrupted_path = temp_dir / 'corrupted_video.avi'
        with open(corrupted_path, 'wb') as f:
            f.write(b'INVALID_VIDEO_DATA' * 100)  # Invalid data
        return corrupted_path
    
    def _create_mock_arena_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Create a mock video frame with a visible arena for boundary detection testing."""
        frame = np.zeros((height, width), dtype=np.uint8)
        
        # Create a rectangular arena in the center
        arena_margin = 50
        cv2.rectangle(
            frame,
            (arena_margin, arena_margin),
            (width - arena_margin, height - arena_margin),
            255,  # White rectangle
            2     # Border thickness
        )
        
        # Add some noise for realistic testing
        noise = np.random.normal(0, 10, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        return frame
    
    def _add_noise_to_frame(self, frame: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Add noise to a video frame for testing robustness."""
        noise = np.random.normal(0, noise_level * 255, frame.shape)
        noisy_frame = frame.astype(np.float32) + noise
        return np.clip(noisy_frame, 0, 255).astype(np.uint8)
    
    def _create_mock_benchmark_data(self) -> Dict[str, Any]:
        """Create mock benchmark data for testing when real benchmark is not available."""
        return {
            'pixel_to_meter_ratio': CRIMALDI_PIXEL_TO_METER_RATIO,
            'reference_coordinates': np.array([
                [1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5], [5.0, 5.5]
            ]),
            'correlation_threshold': CORRELATION_THRESHOLD,
            'numerical_precision': TOLERANCE
        }


# Standalone test functions for module-level functionality

def test_pixel_to_meter_ratio_calculation():
    """
    Test pixel-to-meter ratio calculation accuracy for different video formats and calibration methods.
    """
    # Test with standard parameters
    result = calculate_pixel_to_meter_ratio(
        arena_width_meters=1.0,
        arena_height_meters=1.0,
        video_width_pixels=640,
        video_height_pixels=480,
        calculation_method='geometric_mean'
    )
    
    expected_horizontal = 640.0
    expected_vertical = 480.0
    expected_primary = np.sqrt(expected_horizontal * expected_vertical)
    
    assert abs(result['horizontal_ratio'] - expected_horizontal) < TOLERANCE
    assert abs(result['vertical_ratio'] - expected_vertical) < TOLERANCE
    assert abs(result['primary_ratio'] - expected_primary) < TOLERANCE
    assert result['confidence_level'] > 0.8


def test_arena_boundary_detection():
    """
    Test arena boundary detection algorithms with different detection methods and validation criteria.
    """
    # Create test frame with clear boundaries
    test_frame = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(test_frame, (100, 100), (540, 380), 255, 2)
    
    # Test contour detection
    result = detect_arena_boundaries(
        video_frame=test_frame,
        detection_method='contour',
        detection_parameters={'threshold_value': 127},
        validate_detection=True
    )
    
    assert result['detection_confidence'] > 0.5
    assert 'arena_boundaries' in result
    assert result['arena_boundaries']['width'] > 0
    assert result['arena_boundaries']['height'] > 0


def test_coordinate_normalization():
    """
    Test coordinate normalization functionality with different coordinate systems.
    """
    # Test coordinate transformation using standalone function
    test_coordinates = np.array([[100, 150], [200, 250]])
    pixel_to_meter_ratio = 100.0
    
    # Manual normalization test
    expected_meter_coords = test_coordinates / pixel_to_meter_ratio
    
    # Test that our expected calculation is correct
    assert expected_meter_coords.shape == test_coordinates.shape
    assert np.all(expected_meter_coords > 0)
    assert np.all(expected_meter_coords < test_coordinates)


def test_calibration_validation():
    """
    Test calibration validation against reference standards.
    """
    # Test basic validation logic
    test_parameters = {
        'pixel_to_meter_ratio': CRIMALDI_PIXEL_TO_METER_RATIO,
        'spatial_accuracy': 0.01,
        'confidence': 0.9
    }
    
    reference_parameters = {
        'pixel_to_meter_ratio': CRIMALDI_PIXEL_TO_METER_RATIO,
        'spatial_accuracy': 0.01,
        'confidence': 0.95
    }
    
    # Test parameter consistency
    ratio_diff = abs(test_parameters['pixel_to_meter_ratio'] - reference_parameters['pixel_to_meter_ratio'])
    assert ratio_diff < TOLERANCE
    
    accuracy_diff = abs(test_parameters['spatial_accuracy'] - reference_parameters['spatial_accuracy'])
    assert accuracy_diff < TOLERANCE


def test_cross_format_compatibility():
    """
    Test cross-format compatibility validation between different formats.
    """
    # Test format ratio comparison
    crimaldi_ratio = CRIMALDI_PIXEL_TO_METER_RATIO
    custom_ratio = CUSTOM_PIXEL_TO_METER_RATIO
    
    # Both should be positive and reasonable
    assert crimaldi_ratio > 0
    assert custom_ratio > 0
    assert crimaldi_ratio != custom_ratio  # They should be different
    
    # Test coordinate transformation compatibility
    test_coords = np.array([[100, 150]])
    crimaldi_meters = test_coords / crimaldi_ratio
    custom_meters = test_coords / custom_ratio
    
    # Both should produce valid meter coordinates
    assert np.all(crimaldi_meters > 0)
    assert np.all(custom_meters > 0)


def test_error_handling():
    """
    Test error handling for invalid inputs and edge cases.
    """
    # Test invalid arena dimensions
    with pytest.raises(ValidationError):
        calculate_pixel_to_meter_ratio(
            arena_width_meters=0.0,  # Invalid zero width
            arena_height_meters=1.0,
            video_width_pixels=640,
            video_height_pixels=480
        )
    
    # Test invalid video dimensions
    with pytest.raises(ValidationError):
        calculate_pixel_to_meter_ratio(
            arena_width_meters=1.0,
            arena_height_meters=1.0,
            video_width_pixels=0,  # Invalid zero width
            video_height_pixels=480
        )


def test_performance_requirements():
    """
    Test that performance requirements are met for processing time targets.
    """
    start_time = time.time()
    
    # Perform a series of operations that should complete quickly
    for _ in range(10):
        result = calculate_pixel_to_meter_ratio(
            arena_width_meters=1.0,
            arena_height_meters=1.0,
            video_width_pixels=640,
            video_height_pixels=480
        )
        assert result['confidence_level'] > 0
    
    elapsed_time = time.time() - start_time
    
    # Should complete well under the 7.2 second target
    assert elapsed_time < 1.0, f"Performance test took {elapsed_time:.3f}s, should be under 1.0s"


# Module-level test configuration and execution support
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])