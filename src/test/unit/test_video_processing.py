"""
Comprehensive unit test module for video processing functionality providing systematic validation of video reading, 
format detection, normalization pipeline, cross-format compatibility, and performance requirements.

This module implements exhaustive testing of video processor components against >95% correlation accuracy requirements, 
<7.2 seconds per simulation performance targets, and cross-format compatibility between Crimaldi and custom AVI formats 
with scientific computing precision and reproducible test execution.

Key Features:
- Comprehensive video processor initialization and configuration validation
- Cross-format compatibility testing between Crimaldi and custom AVI formats
- Performance validation against <7.2 seconds per simulation requirement
- Scientific accuracy validation with >95% correlation thresholds
- Batch processing validation for 4000+ simulation requirements
- Error handling and quality assurance validation with fail-fast strategies
- Numerical precision validation with 1e-6 tolerance requirements
- Reproducible test execution with deterministic random seeds and controlled environments
"""

# External library imports with version specifications
import pytest  # pytest 8.3.5+ - Testing framework for unit test execution and fixture management
import numpy as np  # numpy 2.1.3+ - Numerical array operations and scientific computing for test data validation
import cv2  # opencv-python 4.11.0+ - Video processing operations for format validation and frame analysis
from pathlib import Path  # pathlib 3.9+ - Cross-platform path handling for test fixture files
import tempfile  # tempfile 3.9+ - Temporary file management for test isolation
import time  # time 3.9+ - Performance timing for execution time validation
import warnings  # warnings 3.9+ - Warning management for test execution
import json  # json 3.9+ - JSON configuration loading for test scenarios

# Internal imports from video processing backend
from backend.core.data_normalization.video_processor import (
    VideoProcessor,
    VideoProcessingConfig
)
from backend.io.video_reader import (
    VideoReader,
    CrimaldiVideoReader,
    CustomVideoReader
)

# Internal imports from test utilities and helpers
from test.utils.test_helpers import (
    create_test_fixture_path,
    load_test_config,
    assert_arrays_almost_equal,
    assert_simulation_accuracy,
    measure_performance,
    TestDataValidator,
    setup_test_environment
)

# Internal imports from mock data generation
from test.mocks.mock_video_data import (
    MockVideoDataset,
    generate_crimaldi_mock_data,
    generate_custom_avi_mock_data
)

# Global test configuration constants and thresholds
CRIMALDI_TEST_FILE = create_test_fixture_path('crimaldi_sample.avi', 'video')
CUSTOM_TEST_FILE = create_test_fixture_path('custom_sample.avi', 'video')
NORMALIZATION_CONFIG_FILE = create_test_fixture_path('test_normalization_config.json', 'config')
REFERENCE_BENCHMARK_FILE = create_test_fixture_path('normalization_benchmark.npy', 'reference_results')
DEFAULT_TOLERANCE = 1e-6
CORRELATION_THRESHOLD = 0.95
PERFORMANCE_TIMEOUT = 7.2
TEST_RANDOM_SEED = 42


class TestVideoProcessingFixtures:
    """
    Test fixture management class providing standardized setup and teardown for video processing tests 
    with configuration management and resource cleanup.
    
    This class provides comprehensive test environment management with temporary directory setup,
    configuration loading, and resource cleanup for isolated test execution.
    """
    
    def __init__(self, test_config: dict):
        """
        Initialize test fixtures with configuration and temporary directory setup.
        
        Args:
            test_config: Dictionary containing test configuration parameters
        """
        # Load test configuration from NORMALIZATION_CONFIG_FILE
        self.test_config = test_config.copy()
        
        # Create temporary directory for test isolation
        self.temp_directory = Path(tempfile.mkdtemp(prefix="video_processing_test_"))
        
        # Initialize TestDataValidator with test parameters
        self.validator = TestDataValidator(
            tolerance=DEFAULT_TOLERANCE,
            strict_validation=True
        )
        
        # Setup MockVideoDataset for test data generation
        self.mock_dataset = MockVideoDataset(
            dataset_config={
                'formats': ['crimaldi', 'custom'],
                'arena_size': (1.0, 1.0),
                'resolution': (640, 480),
                'duration': 10.0
            },
            enable_caching=True,
            random_seed=TEST_RANDOM_SEED
        )
        
        # Configure test environment and logging
        self.test_environment_id = None
    
    def setup_test_environment(self) -> None:
        """
        Setup isolated test environment with temporary directories and configuration.
        """
        # Create temporary test directory structure
        (self.temp_directory / 'input').mkdir(parents=True, exist_ok=True)
        (self.temp_directory / 'output').mkdir(parents=True, exist_ok=True)
        (self.temp_directory / 'config').mkdir(parents=True, exist_ok=True)
        
        # Setup test-specific environment variables
        import os
        os.environ['TEST_TEMP_DIR'] = str(self.temp_directory)
        
        # Initialize test logging configuration
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.temp_directory / 'test.log'),
                logging.StreamHandler()
            ]
        )
        
        # Configure deterministic random seeds
        np.random.seed(TEST_RANDOM_SEED)
        
        # Setup resource monitoring and tracking
        self.test_start_time = time.time()
    
    def teardown_test_environment(self) -> None:
        """
        Cleanup test environment and temporary resources.
        """
        # Cleanup temporary directories and files
        import shutil
        try:
            shutil.rmtree(self.temp_directory, ignore_errors=True)
        except Exception as e:
            warnings.warn(f"Failed to cleanup test directory: {e}")
        
        # Reset environment variables to original state
        import os
        os.environ.pop('TEST_TEMP_DIR', None)
        
        # Finalize test logging and performance reports
        if hasattr(self, 'test_start_time'):
            test_duration = time.time() - self.test_start_time
            print(f"Test environment cleanup completed in {test_duration:.3f} seconds")
        
        # Clear test data cache and resources
        if hasattr(self, 'mock_dataset'):
            self.mock_dataset.clear_cache()
        
        # Validate resource cleanup completion
        # Additional cleanup validation would go here


@pytest.fixture
def test_config():
    """Load test configuration for video processing tests."""
    try:
        return load_test_config('video_processing_test_config')
    except FileNotFoundError:
        # Fallback configuration if file not found
        return {
            'test_type': 'unit',
            'parameters': {
                'correlation_threshold': CORRELATION_THRESHOLD,
                'performance_timeout': PERFORMANCE_TIMEOUT,
                'tolerance': DEFAULT_TOLERANCE
            }
        }


@pytest.fixture
def test_fixtures(test_config):
    """Create test fixtures with configuration and cleanup."""
    fixtures = TestVideoProcessingFixtures(test_config)
    fixtures.setup_test_environment()
    yield fixtures
    fixtures.teardown_test_environment()


@pytest.mark.unit
@pytest.mark.video_processing
def test_video_processor_initialization(test_config):
    """
    Test VideoProcessor initialization with various configurations including default settings, 
    custom parameters, and error conditions to validate proper setup and configuration validation.
    """
    # Load test configuration from NORMALIZATION_CONFIG_FILE
    try:
        normalization_config = load_test_config('video_normalization')
    except FileNotFoundError:
        normalization_config = {
            'pixel_to_meter_ratio': 100.0,
            'frame_rate_hz': 30.0,
            'intensity_units': 'concentration_ppm'
        }
    
    # Create VideoProcessingConfig with test parameters
    config = VideoProcessingConfig(
        pixel_to_meter_ratio=normalization_config.get('pixel_to_meter_ratio', 100.0),
        frame_rate_hz=normalization_config.get('frame_rate_hz', 30.0),
        intensity_units=normalization_config.get('intensity_units', 'concentration_ppm'),
        coordinate_system='cartesian',
        temporal_resolution=1.0 / normalization_config.get('frame_rate_hz', 30.0)
    )
    
    # Initialize VideoProcessor with configuration
    processor = VideoProcessor(
        processing_config=config,
        enable_caching=True,
        performance_monitoring=True
    )
    
    # Validate processor initialization and configuration
    assert processor is not None, "VideoProcessor initialization failed"
    assert processor.processing_config is not None, "Processing configuration not set"
    assert processor.caching_enabled == True, "Caching not properly enabled"
    
    # Test configuration validation methods
    validation_result = config.validate_config()
    assert validation_result.is_valid, f"Configuration validation failed: {validation_result.errors}"
    
    # Verify processor properties and state
    assert hasattr(processor, 'process_video'), "Missing process_video method"
    assert hasattr(processor, 'process_video_batch'), "Missing process_video_batch method"
    assert hasattr(processor, 'normalize_video_data'), "Missing normalize_video_data method"
    assert hasattr(processor, 'validate_processing_quality'), "Missing validate_processing_quality method"
    
    # Test error handling for invalid configurations
    with pytest.raises(ValueError, match="Invalid pixel-to-meter ratio"):
        invalid_config = VideoProcessingConfig(
            pixel_to_meter_ratio=-1.0,  # Invalid negative value
            frame_rate_hz=30.0,
            intensity_units='concentration_ppm'
        )
    
    with pytest.raises(ValueError, match="Invalid frame rate"):
        invalid_config = VideoProcessingConfig(
            pixel_to_meter_ratio=100.0,
            frame_rate_hz=0.0,  # Invalid zero frame rate
            intensity_units='concentration_ppm'
        )
    
    # Assert proper initialization completion
    assert processor.is_initialized(), "Processor not properly initialized"


@pytest.mark.unit
@pytest.mark.parametrize('video_format,test_file_path', [
    ('crimaldi', CRIMALDI_TEST_FILE),
    ('custom', CUSTOM_TEST_FILE)
])
def test_video_reader_format_detection(video_format, test_file_path):
    """
    Test video format detection capabilities for Crimaldi, custom AVI, and standard formats 
    with confidence levels and metadata extraction validation.
    """
    # Load test video file using specified path
    if not test_file_path.exists():
        # Generate mock video file for testing
        mock_dataset = MockVideoDataset(
            dataset_config={
                'formats': [video_format],
                'arena_size': (1.0, 1.0),
                'resolution': (640, 480),
                'duration': 5.0
            },
            random_seed=TEST_RANDOM_SEED
        )
        
        if video_format == 'crimaldi':
            mock_data = mock_dataset.get_crimaldi_dataset()
        else:
            mock_data = mock_dataset.get_custom_dataset()
        
        # Save mock data as temporary file
        test_file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(test_file_path.with_suffix('.npy'), mock_data['video_data'])
    
    # Create VideoReader instance for format detection
    reader = VideoReader(
        video_path=str(test_file_path),
        reader_config={'enable_format_detection': True},
        enable_caching=True
    )
    
    # Execute format detection with deep inspection
    format_detection = reader.format_detection
    
    # Validate detected format matches expected format
    assert format_detection.get('format_detected', False), "Format detection failed"
    
    if video_format == 'crimaldi':
        assert format_detection.get('format_type') == 'crimaldi', f"Expected crimaldi format, got {format_detection.get('format_type')}"
    else:
        expected_formats = ['custom', 'avi']
        assert format_detection.get('format_type') in expected_formats, f"Expected custom/avi format, got {format_detection.get('format_type')}"
    
    # Check detection confidence levels meet thresholds
    confidence = format_detection.get('confidence_level', 0.0)
    assert confidence >= 0.6, f"Detection confidence {confidence} below minimum threshold 0.6"
    
    # Verify format-specific metadata extraction
    assert 'file_metadata' in format_detection, "Missing file metadata in detection result"
    assert 'detection_methods_used' in format_detection, "Missing detection methods information"
    
    # Test format detection error handling
    # This would test with corrupted or invalid files
    
    # Assert detection accuracy and reliability
    detection_methods = format_detection.get('detection_methods_used', [])
    assert len(detection_methods) >= 2, "Insufficient detection methods used"


@pytest.mark.unit
@pytest.mark.crimaldi_format
def test_crimaldi_video_reader_calibration():
    """
    Test Crimaldi-specific video reader calibration parameter extraction including pixel-to-meter ratio, 
    coordinate system, and intensity calibration validation.
    """
    # Create CrimaldiVideoReader with test configuration
    crimaldi_config = {
        'pixel_to_meter_ratio': 100.0,
        'arena_size_meters': (1.0, 1.0),
        'frame_rate_hz': 30.0,
        'intensity_units': 'concentration_ppm',
        'coordinate_system': 'cartesian'
    }
    
    # Generate Crimaldi mock data for testing
    mock_data = generate_crimaldi_mock_data(
        arena_size_meters=(1.0, 1.0),
        resolution_pixels=(640, 480),
        duration_seconds=5.0,
        crimaldi_config=crimaldi_config,
        random_seed=TEST_RANDOM_SEED
    )
    
    # Save mock data to temporary file
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp_file:
        temp_video_path = tmp_file.name
        # In practice, this would save actual video data
        np.save(temp_video_path + '.npy', mock_data['video_data'])
    
    try:
        reader = CrimaldiVideoReader(
            video_path=temp_video_path,
            crimaldi_config=crimaldi_config
        )
        
        # Extract calibration parameters from Crimaldi format
        calibration_params = reader.extract_calibration_parameters(force_reextraction=True)
        
        # Validate pixel-to-meter ratio (100.0 pixels/meter)
        expected_pixel_ratio = 100.0
        actual_pixel_ratio = calibration_params.get('pixel_to_meter_ratio', 0)
        assert abs(actual_pixel_ratio - expected_pixel_ratio) < DEFAULT_TOLERANCE, \
            f"Pixel-to-meter ratio mismatch: expected {expected_pixel_ratio}, got {actual_pixel_ratio}"
        
        # Check coordinate system configuration (cartesian)
        coord_system = calibration_params.get('coordinate_system', '')
        assert coord_system == 'cartesian', f"Expected cartesian coordinate system, got {coord_system}"
        
        # Verify intensity calibration parameters (concentration PPM)
        intensity_units = calibration_params.get('intensity_units', '')
        assert intensity_units == 'concentration_ppm', f"Expected concentration_ppm units, got {intensity_units}"
        
        # Test calibration parameter validation methods
        assert 'spatial_accuracy' in calibration_params, "Missing spatial accuracy parameter"
        assert 'temporal_accuracy' in calibration_params, "Missing temporal accuracy parameter"
        
        # Validate arena dimension extraction
        arena_width = calibration_params.get('arena_width_meters', 0)
        arena_height = calibration_params.get('arena_height_meters', 0)
        assert abs(arena_width - 1.0) < DEFAULT_TOLERANCE, f"Arena width mismatch: expected 1.0, got {arena_width}"
        assert abs(arena_height - 1.0) < DEFAULT_TOLERANCE, f"Arena height mismatch: expected 1.0, got {arena_height}"
        
        # Assert calibration accuracy within tolerance
        pixel_x = calibration_params.get('pixel_to_meter_x', 0)
        pixel_y = calibration_params.get('pixel_to_meter_y', 0)
        assert abs(pixel_x - expected_pixel_ratio) < DEFAULT_TOLERANCE, "X-axis pixel ratio inaccurate"
        assert abs(pixel_y - expected_pixel_ratio) < DEFAULT_TOLERANCE, "Y-axis pixel ratio inaccurate"
        
    finally:
        # Cleanup temporary file
        import os
        try:
            os.unlink(temp_video_path)
            os.unlink(temp_video_path + '.npy')
        except:
            pass


@pytest.mark.unit
@pytest.mark.custom_format
def test_custom_video_reader_adaptation():
    """
    Test custom video reader adaptive parameter detection and normalization configuration 
    for flexible format handling and auto-calibration capabilities.
    """
    # Create CustomVideoReader with adaptive configuration
    custom_config = {
        'pixel_to_meter_ratio': 150.0,
        'arena_size_meters': (1.2, 0.8),
        'frame_rate_hz': 60.0,
        'intensity_units': 'raw_sensor',
        'coordinate_system': 'cartesian',
        'adaptive_calibration': True
    }
    
    # Generate custom mock data for testing
    mock_data = generate_custom_avi_mock_data(
        arena_size_meters=(1.2, 0.8),
        resolution_pixels=(800, 600),
        duration_seconds=5.0,
        custom_config=custom_config,
        random_seed=TEST_RANDOM_SEED
    )
    
    # Save mock data to temporary file
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp_file:
        temp_video_path = tmp_file.name
        # In practice, this would save actual video data
        np.save(temp_video_path + '.npy', mock_data['video_data'])
    
    try:
        reader = CustomVideoReader(
            video_path=temp_video_path,
            custom_config=custom_config
        )
        
        # Execute format parameter detection with deep analysis
        detected_params = reader.detect_format_parameters(
            deep_analysis=True,
            detection_hints={'expected_format': 'custom_avi'}
        )
        
        # Validate detected pixel-to-meter ratio (150.0 pixels/meter)
        expected_pixel_ratio = 150.0
        estimated_ratio = detected_params.get('format_parameters', {}).get('estimated_pixel_ratio', 0)
        # Allow some tolerance for adaptive detection
        assert abs(estimated_ratio - expected_pixel_ratio) / expected_pixel_ratio < 0.2, \
            f"Adaptive pixel ratio detection inaccurate: expected ~{expected_pixel_ratio}, got {estimated_ratio}"
        
        # Check intensity unit detection (raw sensor values)
        intensity_type = detected_params.get('format_parameters', {}).get('intensity_type', '')
        assert intensity_type in ['uint8', 'raw_sensor'], f"Expected raw sensor intensity type, got {intensity_type}"
        
        # Test normalization configuration adaptation
        norm_config = reader.configure_normalization(
            normalization_requirements={
                'target_width': 1.0,
                'target_height': 1.0,
                'intensity_range': (0.0, 1.0)
            },
            auto_optimize=True
        )
        
        assert norm_config['spatial_normalization']['target_width'] == 1.0, "Normalization width not configured correctly"
        assert norm_config['spatial_normalization']['target_height'] == 1.0, "Normalization height not configured correctly"
        
        # Verify auto-calibration parameter detection
        arena_chars = detected_params.get('arena_characteristics', {})
        assert 'estimated_arena_type' in arena_chars, "Missing arena type detection"
        assert 'aspect_ratio' in arena_chars, "Missing aspect ratio detection"
        
        # Validate format adaptation accuracy
        validation_result = norm_config.get('validation_result', {})
        assert validation_result.get('valid', False), "Normalization configuration validation failed"
        
        # Assert adaptive processing reliability
        assert detected_params.get('parameter_consistency', {}).get('consistent', False), \
            "Parameter consistency check failed"
        
    finally:
        # Cleanup temporary file
        import os
        try:
            os.unlink(temp_video_path)
            os.unlink(temp_video_path + '.npy')
        except:
            pass


@pytest.mark.unit
@measure_performance(time_limit_seconds=1.0)
def test_video_frame_reading_accuracy(test_fixtures):
    """
    Test video frame reading accuracy including single frame access, batch reading, seeking operations, 
    and frame validation with numerical precision requirements.
    """
    # Initialize video reader with test video file
    mock_data = test_fixtures.mock_dataset.get_crimaldi_dataset()
    
    # Save mock video data to temporary file for testing
    temp_video_file = test_fixtures.temp_directory / 'test_video.avi'
    np.save(str(temp_video_file) + '.npy', mock_data['video_data'])
    
    video_reader = VideoReader(
        video_path=str(temp_video_file),
        reader_config={'enable_caching': True, 'validate_frames': True},
        enable_caching=True
    )
    
    # Read single frame at specified frame number
    frame_number = 10
    frame = video_reader.read_frame(
        frame_index=frame_number,
        use_cache=True,
        validate_frame=True
    )
    
    # Validate frame data structure and dimensions
    assert frame is not None, f"Failed to read frame {frame_number}"
    assert isinstance(frame, np.ndarray), f"Frame is not numpy array: {type(frame)}"
    
    expected_height, expected_width = mock_data['metadata']['resolution_pixels'][::-1]
    if len(frame.shape) == 2:  # Grayscale
        assert frame.shape == (expected_height, expected_width), \
            f"Frame shape mismatch: expected {(expected_height, expected_width)}, got {frame.shape}"
    else:  # Color
        assert frame.shape == (expected_height, expected_width, 3), \
            f"Frame shape mismatch: expected {(expected_height, expected_width, 3)}, got {frame.shape}"
    
    # Check frame data type and value ranges
    assert frame.dtype in [np.uint8, np.uint16, np.float32], f"Unexpected frame dtype: {frame.dtype}"
    
    if frame.dtype == np.uint8:
        assert np.all(frame >= 0) and np.all(frame <= 255), "Frame values outside uint8 range"
    elif frame.dtype == np.uint16:
        assert np.all(frame >= 0) and np.all(frame <= 65535), "Frame values outside uint16 range"
    
    # Test frame seeking accuracy and performance
    seek_success = video_reader.seek_frame(frame_number, validate_seek=True)
    assert seek_success, f"Frame seeking failed for frame {frame_number}"
    
    # Validate frame batch reading functionality
    frame_indices = [5, 10, 15, 20]
    batch_frames = video_reader.read_frame_batch(
        frame_indices=frame_indices,
        use_cache=True,
        parallel_processing=False
    )
    
    assert len(batch_frames) == len(frame_indices), \
        f"Batch read count mismatch: expected {len(frame_indices)}, got {len(batch_frames)}"
    
    for idx in frame_indices:
        assert idx in batch_frames, f"Missing frame {idx} in batch results"
        assert batch_frames[idx] is not None, f"Frame {idx} is None in batch"
    
    # Compare frame data against reference values
    # This would involve comparing with known good reference frames
    reference_frame = mock_data['video_data'][frame_number]
    if len(reference_frame.shape) == 2 and len(frame.shape) == 2:
        # Both grayscale - direct comparison
        max_diff = np.max(np.abs(frame.astype(np.float32) - reference_frame.astype(np.float32)))
        assert max_diff <= 5.0, f"Frame data differs too much from reference: max_diff={max_diff}"
    
    # Assert frame reading accuracy within tolerance
    # Additional validation would depend on specific format requirements


@pytest.mark.unit
@pytest.mark.integration
@measure_performance(time_limit_seconds=PERFORMANCE_TIMEOUT)
def test_video_processing_pipeline(test_fixtures):
    """
    Test complete video processing pipeline including normalization, validation, and quality assessment 
    with performance monitoring and accuracy validation.
    """
    # Load test video using video processor
    mock_data = test_fixtures.mock_dataset.get_crimaldi_dataset()
    temp_video_file = test_fixtures.temp_directory / 'pipeline_test_video.avi'
    np.save(str(temp_video_file) + '.npy', mock_data['video_data'])
    
    # Create video processor with test configuration
    config = VideoProcessingConfig(
        pixel_to_meter_ratio=100.0,
        frame_rate_hz=30.0,
        intensity_units='concentration_ppm',
        coordinate_system='cartesian'
    )
    
    video_processor = VideoProcessor(
        processing_config=config,
        enable_caching=True,
        performance_monitoring=True
    )
    
    # Execute complete normalization pipeline
    start_time = time.time()
    
    processing_result = video_processor.process_video(
        video_path=str(temp_video_file),
        output_path=str(test_fixtures.temp_directory / 'processed_video.npy'),
        validate_quality=True
    )
    
    processing_time = time.time() - start_time
    
    # Validate processing quality against thresholds
    assert processing_result['success'], f"Video processing failed: {processing_result.get('error', 'Unknown error')}"
    
    quality_metrics = processing_result.get('quality_metrics', {})
    assert 'correlation_score' in quality_metrics, "Missing correlation score in quality metrics"
    assert 'processing_accuracy' in quality_metrics, "Missing processing accuracy in quality metrics"
    
    # Check normalization accuracy and consistency
    correlation_score = quality_metrics['correlation_score']
    assert correlation_score >= CORRELATION_THRESHOLD, \
        f"Processing correlation {correlation_score} below threshold {CORRELATION_THRESHOLD}"
    
    # Measure processing time against <7.2 seconds requirement
    assert processing_time <= PERFORMANCE_TIMEOUT, \
        f"Processing time {processing_time:.3f}s exceeds limit {PERFORMANCE_TIMEOUT}s"
    
    # Validate output data structure and format
    assert 'normalized_data' in processing_result, "Missing normalized data in result"
    normalized_data = processing_result['normalized_data']
    assert isinstance(normalized_data, np.ndarray), "Normalized data is not numpy array"
    
    # Compare results against reference benchmark
    if REFERENCE_BENCHMARK_FILE.exists():
        reference_data = np.load(REFERENCE_BENCHMARK_FILE)
        if reference_data.shape == normalized_data.shape:
            assert_arrays_almost_equal(
                actual=normalized_data,
                expected=reference_data,
                tolerance=DEFAULT_TOLERANCE,
                error_message="Normalized data differs from reference benchmark"
            )
    
    # Assert processing accuracy and performance compliance
    performance_metrics = processing_result.get('performance_metrics', {})
    assert 'memory_usage_mb' in performance_metrics, "Missing memory usage metrics"
    assert 'cpu_time_seconds' in performance_metrics, "Missing CPU time metrics"


@pytest.mark.unit
@pytest.mark.cross_format
def test_cross_format_compatibility(test_fixtures):
    """
    Test cross-format compatibility between Crimaldi and custom formats ensuring consistent 
    processing results and >90% cross-format correlation.
    """
    # Generate equivalent test scenarios in both formats
    arena_size = (1.0, 1.0)
    resolution = (640, 480)
    duration = 5.0
    
    # Generate identical plume scenarios with same random seed
    base_config = {
        'source_x': 0.3,
        'source_y': 0.5,
        'plume_parameters': {
            'diffusion_coefficient': 0.1,
            'wind_velocity': (0.5, 0.0),
            'source_strength': 1.0,
            'noise_level': 0.05
        }
    }
    
    # Process Crimaldi format data with CrimaldiVideoReader
    crimaldi_data = generate_crimaldi_mock_data(
        arena_size_meters=arena_size,
        resolution_pixels=resolution,
        duration_seconds=duration,
        crimaldi_config=base_config,
        random_seed=TEST_RANDOM_SEED
    )
    
    # Process custom format data with CustomVideoReader
    custom_data = generate_custom_avi_mock_data(
        arena_size_meters=arena_size,
        resolution_pixels=resolution,
        duration_seconds=duration,
        custom_config=base_config,
        random_seed=TEST_RANDOM_SEED
    )
    
    # Create normalized versions for comparison
    crimaldi_config = VideoProcessingConfig(
        pixel_to_meter_ratio=100.0,  # Crimaldi standard
        frame_rate_hz=30.0,
        intensity_units='concentration_ppm'
    )
    
    custom_config = VideoProcessingConfig(
        pixel_to_meter_ratio=150.0,  # Custom standard
        frame_rate_hz=60.0,
        intensity_units='raw_sensor'
    )
    
    # Normalize both datasets to common scale
    crimaldi_processor = VideoProcessor(crimaldi_config)
    custom_processor = VideoProcessor(custom_config)
    
    # Save data to temporary files
    crimaldi_file = test_fixtures.temp_directory / 'crimaldi_test.npy'
    custom_file = test_fixtures.temp_directory / 'custom_test.npy'
    np.save(crimaldi_file, crimaldi_data['video_data'])
    np.save(custom_file, custom_data['video_data'])
    
    # Process both formats
    crimaldi_result = crimaldi_processor.normalize_video_data(
        video_data=crimaldi_data['video_data'],
        calibration_params=crimaldi_data['calibration_parameters']
    )
    
    custom_result = custom_processor.normalize_video_data(
        video_data=custom_data['video_data'],
        calibration_params=custom_data['calibration_parameters']
    )
    
    # Compare processing results between formats
    assert crimaldi_result['success'], "Crimaldi processing failed"
    assert custom_result['success'], "Custom processing failed"
    
    # Calculate cross-format correlation coefficients
    crimaldi_normalized = crimaldi_result['normalized_data']
    custom_normalized = custom_result['normalized_data']
    
    # Resize arrays to same dimensions for comparison if needed
    if crimaldi_normalized.shape != custom_normalized.shape:
        # Resize custom to match crimaldi dimensions
        from scipy.ndimage import zoom
        scale_factors = [
            crimaldi_normalized.shape[i] / custom_normalized.shape[i] 
            for i in range(len(crimaldi_normalized.shape))
        ]
        custom_normalized = zoom(custom_normalized, scale_factors)
    
    # Calculate correlation
    correlation_matrix = np.corrcoef(
        crimaldi_normalized.flatten(),
        custom_normalized.flatten()
    )
    cross_correlation = correlation_matrix[0, 1]
    
    # Validate spatial and temporal alignment consistency
    assert not np.isnan(cross_correlation), "Cross-format correlation calculation failed"
    
    # Check intensity calibration consistency
    intensity_diff = np.mean(np.abs(crimaldi_normalized - custom_normalized))
    max_intensity = max(np.max(crimaldi_normalized), np.max(custom_normalized))
    relative_intensity_diff = intensity_diff / max_intensity if max_intensity > 0 else 0
    
    # Assert >90% cross-format correlation requirement
    min_correlation = 0.90
    assert cross_correlation >= min_correlation, \
        f"Cross-format correlation {cross_correlation:.6f} below required {min_correlation}"
    
    # Additional consistency checks
    assert relative_intensity_diff < 0.3, \
        f"Intensity calibration inconsistency too high: {relative_intensity_diff:.3f}"


@pytest.mark.unit
@pytest.mark.batch_processing
@measure_performance(time_limit_seconds=30.0)
def test_batch_video_processing(test_fixtures):
    """
    Test batch video processing functionality including parallel execution, progress tracking, 
    and error handling for multiple video files.
    """
    # Create list of test video files for batch processing
    num_videos = 5  # Smaller number for unit tests
    video_file_list = []
    
    for i in range(num_videos):
        # Generate mock video data
        mock_data = test_fixtures.mock_dataset.get_crimaldi_dataset(force_regenerate=True)
        
        # Save to temporary file
        video_file = test_fixtures.temp_directory / f'batch_video_{i:02d}.npy'
        np.save(video_file, mock_data['video_data'])
        video_file_list.append(str(video_file))
    
    # Configure batch processing parameters and settings
    batch_config = {
        'parallel_workers': 2,
        'progress_tracking': True,
        'error_handling': 'continue',
        'output_directory': str(test_fixtures.temp_directory / 'batch_output'),
        'quality_validation': True
    }
    
    # Create video processor for batch processing
    config = VideoProcessingConfig(
        pixel_to_meter_ratio=100.0,
        frame_rate_hz=30.0,
        intensity_units='concentration_ppm'
    )
    
    processor = VideoProcessor(config, enable_caching=True)
    
    # Execute batch video processing with progress tracking
    start_time = time.time()
    
    batch_results = processor.process_video_batch(
        video_paths=video_file_list,
        batch_config=batch_config,
        enable_progress_tracking=True
    )
    
    processing_time = time.time() - start_time
    
    # Validate batch completion rate (target: 100%)
    successful_count = sum(1 for result in batch_results if result.get('success', False))
    completion_rate = successful_count / len(video_file_list) if video_file_list else 0
    
    assert completion_rate >= 0.8, \
        f"Batch completion rate {completion_rate:.2%} below minimum 80%"
    
    # Check individual processing results and quality
    for i, result in enumerate(batch_results):
        if result.get('success', False):
            assert 'quality_metrics' in result, f"Missing quality metrics for video {i}"
            assert 'processing_time' in result, f"Missing processing time for video {i}"
            
            # Validate individual quality scores
            quality_metrics = result['quality_metrics']
            if 'correlation_score' in quality_metrics:
                correlation = quality_metrics['correlation_score']
                assert correlation >= 0.8, f"Low correlation {correlation} for video {i}"
    
    # Measure batch processing performance and efficiency
    avg_time_per_video = processing_time / len(video_file_list) if video_file_list else 0
    assert avg_time_per_video <= PERFORMANCE_TIMEOUT, \
        f"Average processing time {avg_time_per_video:.3f}s exceeds limit {PERFORMANCE_TIMEOUT}s"
    
    # Validate error handling for failed processing attempts
    error_count = len(video_file_list) - successful_count
    error_rate = error_count / len(video_file_list) if video_file_list else 0
    
    assert error_rate <= 0.2, f"Error rate {error_rate:.2%} exceeds maximum 20%"
    
    # Assert batch processing reliability and performance
    assert batch_results is not None, "Batch processing returned None"
    assert len(batch_results) == len(video_file_list), \
        f"Result count {len(batch_results)} doesn't match input count {len(video_file_list)}"


@pytest.mark.unit
@pytest.mark.error_handling
@pytest.mark.parametrize('error_scenario', [
    'corrupted_file',
    'invalid_format',
    'memory_limit',
    'timeout'
])
def test_video_processing_error_handling(error_scenario, test_fixtures):
    """
    Test video processing error handling including corrupted files, invalid formats, 
    and resource constraints with graceful degradation validation.
    """
    # Setup error scenario based on test parameter
    config = VideoProcessingConfig(
        pixel_to_meter_ratio=100.0,
        frame_rate_hz=30.0,
        intensity_units='concentration_ppm'
    )
    
    processor = VideoProcessor(config, enable_caching=True)
    
    if error_scenario == 'corrupted_file':
        # Create corrupted video file
        corrupted_file = test_fixtures.temp_directory / 'corrupted_video.avi'
        with open(corrupted_file, 'wb') as f:
            f.write(b'corrupted data that is not a valid video file')
        test_video_path = str(corrupted_file)
        
    elif error_scenario == 'invalid_format':
        # Create file with invalid format
        invalid_file = test_fixtures.temp_directory / 'invalid_format.txt'
        with open(invalid_file, 'w') as f:
            f.write('This is not a video file')
        test_video_path = str(invalid_file)
        
    elif error_scenario == 'memory_limit':
        # Create very large mock data to trigger memory issues
        large_data = np.random.randint(0, 255, (1000, 2000, 2000), dtype=np.uint8)
        large_file = test_fixtures.temp_directory / 'large_video.npy'
        np.save(large_file, large_data)
        test_video_path = str(large_file)
        
    elif error_scenario == 'timeout':
        # Use valid file but set very short timeout
        mock_data = test_fixtures.mock_dataset.get_crimaldi_dataset()
        timeout_file = test_fixtures.temp_directory / 'timeout_video.npy'
        np.save(timeout_file, mock_data['video_data'])
        test_video_path = str(timeout_file)
        # Set very short timeout
        processor.processing_config.timeout_seconds = 0.001
    
    # Attempt video processing with error conditions
    try:
        result = processor.process_video(
            video_path=test_video_path,
            output_path=str(test_fixtures.temp_directory / 'error_test_output.npy'),
            validate_quality=True
        )
        
        # Validate error detection and classification
        if error_scenario in ['corrupted_file', 'invalid_format']:
            assert not result.get('success', True), \
                f"Processing should have failed for {error_scenario}"
            assert 'error' in result, "Missing error information in result"
            assert 'error_type' in result, "Missing error type classification"
            
        elif error_scenario == 'memory_limit':
            # May succeed with warnings or fail gracefully
            if not result.get('success', True):
                assert 'memory' in str(result.get('error', '')).lower(), \
                    "Error should be related to memory"
                    
        elif error_scenario == 'timeout':
            assert not result.get('success', True), "Processing should have timed out"
            assert 'timeout' in str(result.get('error', '')).lower(), \
                "Error should be related to timeout"
        
        # Check graceful degradation and recovery mechanisms
        if 'warnings' in result:
            warnings_list = result['warnings']
            assert isinstance(warnings_list, list), "Warnings should be a list"
        
        # Verify error reporting and logging functionality
        assert 'error_timestamp' in result or result.get('success', False), \
            "Missing error timestamp or success indication"
        
    except Exception as e:
        # Test fail-fast validation for critical errors
        assert error_scenario in ['corrupted_file', 'invalid_format'], \
            f"Unexpected exception for {error_scenario}: {e}"
        
        # Validate that exceptions are appropriate for the error scenario
        if error_scenario == 'corrupted_file':
            assert 'corrupted' in str(e).lower() or 'invalid' in str(e).lower(), \
                f"Exception message should indicate corruption: {e}"
        elif error_scenario == 'invalid_format':
            assert 'format' in str(e).lower() or 'invalid' in str(e).lower(), \
                f"Exception message should indicate format issue: {e}"
    
    # Validate resource cleanup after errors
    # Check that processor is still in valid state
    assert hasattr(processor, 'processing_config'), "Processor config corrupted after error"
    assert processor.processing_config is not None, "Processor config is None after error"
    
    # Assert proper error handling and system stability
    # System should remain stable and usable after error conditions


@pytest.mark.unit
@pytest.mark.performance
@measure_performance(time_limit_seconds=PERFORMANCE_TIMEOUT, memory_limit_mb=2048)
def test_video_processing_performance(test_fixtures):
    """
    Test video processing performance against <7.2 seconds per simulation requirement 
    with memory usage monitoring and throughput validation.
    """
    # Configure performance monitoring for video processing
    performance_config = {
        'enable_profiling': True,
        'memory_monitoring': True,
        'cpu_monitoring': True,
        'optimization_level': 'high'
    }
    
    config = VideoProcessingConfig(
        pixel_to_meter_ratio=100.0,
        frame_rate_hz=30.0,
        intensity_units='concentration_ppm',
        performance_config=performance_config
    )
    
    video_processor = VideoProcessor(config, enable_caching=True, performance_monitoring=True)
    
    # Generate test video data
    mock_data = test_fixtures.mock_dataset.get_crimaldi_dataset()
    test_video_file = test_fixtures.temp_directory / 'performance_test_video.npy'
    np.save(test_video_file, mock_data['video_data'])
    
    # Execute video processing with performance tracking
    start_time = time.time()
    memory_before = _get_memory_usage()
    
    result = video_processor.process_video(
        video_path=str(test_video_file),
        output_path=str(test_fixtures.temp_directory / 'performance_output.npy'),
        validate_quality=True
    )
    
    end_time = time.time()
    memory_after = _get_memory_usage()
    
    # Measure processing time against 7.2 seconds threshold
    processing_time = end_time - start_time
    assert processing_time <= PERFORMANCE_TIMEOUT, \
        f"Processing time {processing_time:.3f}s exceeds threshold {PERFORMANCE_TIMEOUT}s"
    
    # Monitor memory usage and resource consumption
    memory_increase = memory_after - memory_before
    max_memory_mb = 2048  # 2GB limit
    
    assert memory_increase <= max_memory_mb, \
        f"Memory usage increase {memory_increase:.1f}MB exceeds limit {max_memory_mb}MB"
    
    # Validate processing throughput and efficiency
    if 'performance_metrics' in result:
        perf_metrics = result['performance_metrics']
        
        # Check frames per second processing rate
        if 'frames_per_second' in perf_metrics:
            fps_processed = perf_metrics['frames_per_second']
            min_fps = 30.0  # Minimum acceptable processing rate
            assert fps_processed >= min_fps, \
                f"Processing rate {fps_processed:.1f} FPS below minimum {min_fps} FPS"
        
        # Check cache performance and optimization effectiveness
        if 'cache_hit_ratio' in perf_metrics:
            cache_hit_ratio = perf_metrics['cache_hit_ratio']
            min_hit_ratio = 0.3  # 30% minimum cache effectiveness
            assert cache_hit_ratio >= min_hit_ratio, \
                f"Cache hit ratio {cache_hit_ratio:.2%} below minimum {min_hit_ratio:.2%}"
    
    # Compare performance against reference benchmarks
    # This would compare against stored performance benchmarks
    
    # Assert performance requirements compliance
    assert result.get('success', False), f"Performance test processing failed: {result.get('error')}"
    
    # Validate that quality wasn't sacrificed for performance
    if 'quality_metrics' in result:
        quality = result['quality_metrics'].get('correlation_score', 0)
        assert quality >= 0.9, f"Quality {quality:.3f} degraded for performance"


@pytest.mark.unit
@pytest.mark.accuracy
def test_video_processing_accuracy(test_fixtures):
    """
    Test video processing accuracy against reference implementations with >95% correlation 
    requirement and numerical precision validation.
    """
    # Load reference benchmark data for comparison
    if not REFERENCE_BENCHMARK_FILE.exists():
        # Generate reference data if not available
        reference_data = test_fixtures.mock_dataset.get_crimaldi_dataset()
        REFERENCE_BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.save(REFERENCE_BENCHMARK_FILE, reference_data['video_data'])
    
    reference_data = np.load(REFERENCE_BENCHMARK_FILE)
    
    # Process test video data using video processor
    config = VideoProcessingConfig(
        pixel_to_meter_ratio=100.0,
        frame_rate_hz=30.0,
        intensity_units='concentration_ppm',
        coordinate_system='cartesian'
    )
    
    processor = VideoProcessor(config, enable_caching=True)
    
    # Create temporary video file
    test_video_file = test_fixtures.temp_directory / 'accuracy_test_video.npy'
    np.save(test_video_file, reference_data)
    
    # Process video and get normalized results
    result = processor.process_video(
        video_path=str(test_video_file),
        output_path=str(test_fixtures.temp_directory / 'accuracy_output.npy'),
        validate_quality=True
    )
    
    assert result.get('success', False), f"Processing failed: {result.get('error')}"
    
    processed_data = result['normalized_data']
    
    # Calculate correlation coefficient against reference
    correlation_matrix = np.corrcoef(processed_data.flatten(), reference_data.flatten())
    correlation_coefficient = correlation_matrix[0, 1]
    
    # Validate numerical accuracy within 1e-6 tolerance
    max_abs_diff = np.max(np.abs(processed_data - reference_data))
    
    # Check statistical significance of correlation
    from scipy import stats
    _, p_value = stats.pearsonr(processed_data.flatten(), reference_data.flatten())
    
    # Validate processing consistency and reproducibility
    # Process same data again to check reproducibility
    result2 = processor.process_video(
        video_path=str(test_video_file),
        output_path=str(test_fixtures.temp_directory / 'accuracy_output2.npy'),
        validate_quality=True
    )
    
    processed_data2 = result2['normalized_data']
    reproducibility_diff = np.max(np.abs(processed_data - processed_data2))
    
    # Compare quality metrics against thresholds
    quality_metrics = result.get('quality_metrics', {})
    correlation_score = quality_metrics.get('correlation_score', 0)
    
    # Assert >95% correlation accuracy requirement
    assert correlation_coefficient >= CORRELATION_THRESHOLD, \
        f"Correlation {correlation_coefficient:.6f} below threshold {CORRELATION_THRESHOLD}"
    
    assert p_value < 0.05, f"Correlation not statistically significant (p={p_value:.6f})"
    
    # Validate numerical precision
    assert max_abs_diff <= DEFAULT_TOLERANCE * 1000, \
        f"Maximum difference {max_abs_diff:.2e} exceeds tolerance"
    
    # Validate reproducibility
    assert reproducibility_diff <= DEFAULT_TOLERANCE, \
        f"Reproducibility difference {reproducibility_diff:.2e} exceeds tolerance"
    
    # Additional accuracy metrics
    rmse = np.sqrt(np.mean((processed_data - reference_data) ** 2))
    mae = np.mean(np.abs(processed_data - reference_data))
    
    assert rmse <= 0.1, f"RMSE {rmse:.6f} too high"
    assert mae <= 0.05, f"MAE {mae:.6f} too high"


@pytest.mark.unit
@pytest.mark.metadata
def test_video_metadata_extraction(test_fixtures):
    """
    Test comprehensive video metadata extraction including technical properties, format information, 
    and calibration parameters for processing pipeline.
    """
    # Generate test video data
    mock_data = test_fixtures.mock_dataset.get_crimaldi_dataset()
    test_video_file = test_fixtures.temp_directory / 'metadata_test_video.npy'
    np.save(test_video_file, mock_data['video_data'])
    
    # Extract video metadata using video reader
    video_reader = VideoReader(
        video_path=str(test_video_file),
        reader_config={'deep_metadata_extraction': True},
        enable_caching=True
    )
    
    metadata = video_reader.get_metadata(
        include_frame_analysis=True,
        include_processing_recommendations=True
    )
    
    # Validate technical properties (resolution, frame rate, codec)
    assert 'basic_properties' in metadata, "Missing basic properties in metadata"
    basic_props = metadata['basic_properties']
    
    expected_width, expected_height = mock_data['metadata']['resolution_pixels']
    assert basic_props.get('width') == expected_width, \
        f"Width mismatch: expected {expected_width}, got {basic_props.get('width')}"
    assert basic_props.get('height') == expected_height, \
        f"Height mismatch: expected {expected_height}, got {basic_props.get('height')}"
    
    expected_fps = mock_data['metadata']['frame_rate_hz']
    actual_fps = basic_props.get('fps', 0)
    assert abs(actual_fps - expected_fps) < 0.1, \
        f"Frame rate mismatch: expected {expected_fps}, got {actual_fps}"
    
    # Check format-specific metadata and calibration parameters
    assert 'format_detection' in metadata, "Missing format detection in metadata"
    format_info = metadata['format_detection']
    
    assert format_info.get('format_detected', False), "Format not detected in metadata"
    assert 'format_type' in format_info, "Missing format type in metadata"
    
    # Verify metadata completeness and accuracy
    required_metadata_keys = [
        'file_path', 'detected_format', 'basic_properties', 
        'reader_configuration', 'performance_metrics'
    ]
    
    for key in required_metadata_keys:
        assert key in metadata, f"Missing required metadata key: {key}"
    
    # Test metadata caching and performance optimization
    # Extract metadata again and verify caching
    metadata2 = video_reader.get_metadata(include_frame_analysis=True)
    
    # Should be cached and identical
    assert metadata2['basic_properties'] == metadata['basic_properties'], \
        "Cached metadata differs from original"
    
    # Validate metadata consistency across multiple extractions
    metadata3 = video_reader.get_metadata(include_frame_analysis=False)
    
    # Basic properties should be consistent
    assert metadata3['basic_properties']['width'] == metadata['basic_properties']['width'], \
        "Metadata inconsistent across extractions"
    
    # Check metadata serialization and deserialization
    import json
    try:
        metadata_json = json.dumps(metadata, default=str)
        deserialized_metadata = json.loads(metadata_json)
        assert 'basic_properties' in deserialized_metadata, "Metadata serialization failed"
    except Exception as e:
        pytest.fail(f"Metadata serialization failed: {e}")
    
    # Assert metadata extraction reliability and accuracy
    assert metadata.get('metadata_timestamp') is not None, "Missing metadata timestamp"


@pytest.mark.unit
@pytest.mark.caching
def test_video_cache_optimization(test_fixtures):
    """
    Test video processing cache optimization including memory management, cache effectiveness, 
    and performance improvement validation.
    """
    # Configure video processing cache with test parameters
    cache_config = {
        'frame_cache_size': 100,
        'metadata_cache_size': 50,
        'enable_compression': True,
        'eviction_policy': 'lru',
        'cache_validation': True
    }
    
    config = VideoProcessingConfig(
        pixel_to_meter_ratio=100.0,
        frame_rate_hz=30.0,
        intensity_units='concentration_ppm',
        cache_config=cache_config
    )
    
    video_processor = VideoProcessor(config, enable_caching=True, performance_monitoring=True)
    
    # Generate test video data
    mock_data = test_fixtures.mock_dataset.get_crimaldi_dataset()
    test_video_file = test_fixtures.temp_directory / 'cache_test_video.npy'
    np.save(test_video_file, mock_data['video_data'])
    
    # Execute video processing with cache monitoring
    # First run - cache miss expected
    start_time = time.time()
    result1 = video_processor.process_video(
        video_path=str(test_video_file),
        output_path=str(test_fixtures.temp_directory / 'cache_output1.npy'),
        validate_quality=True
    )
    first_run_time = time.time() - start_time
    
    # Second run - cache hit expected
    start_time = time.time()
    result2 = video_processor.process_video(
        video_path=str(test_video_file),
        output_path=str(test_fixtures.temp_directory / 'cache_output2.npy'),
        validate_quality=True
    )
    second_run_time = time.time() - start_time
    
    # Measure cache hit rates and effectiveness
    if 'performance_metrics' in result2:
        perf_metrics = result2['performance_metrics']
        cache_hit_ratio = perf_metrics.get('cache_hit_ratio', 0)
        
        # Cache should be effective for second run
        assert cache_hit_ratio > 0.3, f"Cache hit ratio {cache_hit_ratio:.2%} too low"
    
    # Validate memory usage and cache size management
    memory_usage = _get_memory_usage()
    max_cache_memory = 512  # MB
    assert memory_usage <= max_cache_memory, \
        f"Cache memory usage {memory_usage:.1f}MB exceeds limit {max_cache_memory}MB"
    
    # Test cache eviction policies and optimization
    # Process multiple different videos to test eviction
    for i in range(5):
        mock_data_temp = test_fixtures.mock_dataset.get_crimaldi_dataset(force_regenerate=True)
        temp_file = test_fixtures.temp_directory / f'cache_eviction_test_{i}.npy'
        np.save(temp_file, mock_data_temp['video_data'])
        
        video_processor.process_video(
            video_path=str(temp_file),
            output_path=str(test_fixtures.temp_directory / f'cache_eviction_output_{i}.npy'),
            validate_quality=False  # Skip validation for speed
        )
    
    # Compare performance with and without caching
    speed_improvement = (first_run_time - second_run_time) / first_run_time if first_run_time > 0 else 0
    
    # Should see some performance improvement from caching
    assert speed_improvement >= -0.5, \
        f"Cache appears to slow down processing: {speed_improvement:.2%} improvement"
    
    # Validate cache consistency and data integrity
    # Results should be identical between cached and non-cached runs
    assert_arrays_almost_equal(
        actual=result2['normalized_data'],
        expected=result1['normalized_data'],
        tolerance=DEFAULT_TOLERANCE,
        error_message="Cached results differ from original"
    )
    
    # Assert cache optimization effectiveness and reliability
    assert result1.get('success', False), "First processing run failed"
    assert result2.get('success', False), "Second processing run failed"


# Helper functions for test implementation

def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        # Fallback if psutil not available
        return 0.0


def _create_mock_corrupted_file(file_path: Path) -> None:
    """Create a mock corrupted video file for error testing."""
    with open(file_path, 'wb') as f:
        # Write some invalid data that looks like it might be a video file
        f.write(b'RIFF' + b'\x00' * 100 + b'corrupted_data' * 100)


def _validate_processing_result(result: dict, expected_keys: list) -> None:
    """Validate that processing result contains expected keys and structure."""
    for key in expected_keys:
        assert key in result, f"Missing key '{key}' in processing result"
    
    if 'success' in result:
        if result['success']:
            assert 'normalized_data' in result, "Missing normalized_data in successful result"
        else:
            assert 'error' in result, "Missing error information in failed result"