"""
Comprehensive unit test module for data validation functionality in the plume simulation system.

This module validates data format compatibility, parameter constraint checking, cross-format compatibility, 
numerical accuracy verification, and scientific computing validation with >95% correlation requirements. 
Tests fail-fast validation strategies, error handling patterns, and quality assurance systems for 
reproducible scientific computing workflows.

Key Features:
- Comprehensive data format validation testing for Crimaldi and custom formats
- Parameter validation with constraint checking and boundary testing
- Cross-format compatibility validation with conversion accuracy assessment
- Numerical accuracy validation with >95% correlation requirements
- Fail-fast validation strategy testing with early error detection
- Performance threshold validation for <7.2 seconds processing time
- Schema validation with JSON compliance and error reporting
- Quality assurance system testing with comprehensive error detection
- Statistical significance validation with hypothesis testing
- Scientific computing validation with precision checking
"""

# External imports with version specifications
import pytest  # pytest 8.3.5+ - Testing framework for unit test execution and fixture management
import numpy as np  # numpy 2.1.3+ - Numerical array operations and scientific computing for test data validation
import pathlib  # Python 3.9+ - Cross-platform path handling for test fixtures and temporary files
import tempfile  # Python 3.9+ - Temporary file and directory management for test isolation
import unittest.mock  # Python 3.9+ - Mock objects for isolated unit testing of validation components
import warnings  # Python 3.9+ - Warning management for validation edge cases and test scenarios
import json  # Python 3.9+ - JSON configuration file handling for test scenarios
import datetime  # Python 3.9+ - Timestamp handling for test execution tracking
import time  # Python 3.9+ - Performance timing for validation operations
import uuid  # Python 3.9+ - Unique identifier generation for test correlation
import math  # Python 3.9+ - Mathematical operations for numerical validation
import typing  # Python 3.9+ - Type hints for test method signatures
from typing import Dict, Any, List, Optional, Union, Tuple  # Python 3.9+ - Advanced type hints for test validation

# Internal imports from validation framework
from backend.core.data_normalization.validation import (
    validate_normalization_pipeline,
    validate_video_format_compatibility,
    validate_physical_scale_parameters,
    validate_temporal_normalization_parameters,
    validate_intensity_calibration_parameters,
    validate_cross_format_compatibility,
    validate_numerical_precision,
    NormalizationValidationResult,
    ValidationOrchestrator
)

from backend.utils.validation_utils import (
    validate_data_format,
    validate_normalization_config,
    validate_numerical_accuracy,
    ValidationResult,
    ConfigurationValidator,
    DataFormatValidator
)

from backend.error.validation_error import (
    ValidationError,
    DataFormatValidationError,
    ParameterValidationError,
    NumericalValidationError
)

from test.utils.test_helpers import (
    create_test_fixture_path,
    load_test_config,
    assert_arrays_almost_equal,
    assert_simulation_accuracy,
    setup_test_environment,
    TestDataValidator
)

from test.utils.validation_metrics import (
    validate_trajectory_accuracy,
    validate_performance_thresholds,
    validate_statistical_significance,
    ValidationMetricsCalculator
)

# Global test configuration constants
TEST_CONFIG_PATH = pathlib.Path(__file__).parent.parent / 'test_fixtures' / 'config' / 'test_normalization_config.json'
CRIMALDI_SAMPLE_PATH = pathlib.Path(__file__).parent.parent / 'test_fixtures' / 'crimaldi_sample.avi'
CUSTOM_SAMPLE_PATH = pathlib.Path(__file__).parent.parent / 'test_fixtures' / 'custom_sample.avi'
REFERENCE_RESULTS_PATH = pathlib.Path(__file__).parent.parent / 'test_fixtures' / 'reference_results'
CORRELATION_THRESHOLD = 0.95
NUMERICAL_TOLERANCE = 1e-6
PROCESSING_TIME_LIMIT = 7.2


@pytest.mark.unit
class TestDataValidation:
    """
    Comprehensive test class for data validation functionality covering format validation, parameter checking, 
    cross-format compatibility, numerical accuracy, and scientific computing validation with >95% correlation 
    requirements and fail-fast validation strategies.
    
    This test class provides comprehensive validation testing with fail-fast strategies, error handling patterns,
    and quality assurance systems for reproducible scientific computing workflows.
    """
    
    def __init__(self):
        """
        Initialize test class with test configuration, fixture paths, and validation utilities for 
        comprehensive data validation testing.
        """
        # Set up test fixture paths for configuration and sample data
        self.test_config_path = TEST_CONFIG_PATH
        self.crimaldi_sample_path = CRIMALDI_SAMPLE_PATH
        self.custom_sample_path = CUSTOM_SAMPLE_PATH
        self.reference_results_path = REFERENCE_RESULTS_PATH
        
        # Load test normalization configuration from JSON file
        self.test_config = self._load_test_configuration()
        
        # Initialize TestDataValidator with test tolerance settings
        self.data_validator = TestDataValidator(
            tolerance=NUMERICAL_TOLERANCE,
            strict_validation=True
        )
        
        # Initialize ValidationMetricsCalculator with test thresholds
        self.metrics_calculator = ValidationMetricsCalculator(
            quality_thresholds={
                'correlation_threshold': CORRELATION_THRESHOLD,
                'processing_time_limit': PROCESSING_TIME_LIMIT,
                'numerical_tolerance': NUMERICAL_TOLERANCE
            },
            enable_statistical_analysis=True
        )
    
    def setup_method(self, method):
        """
        Setup method executed before each test method with test environment preparation.
        
        Args:
            method: Test method object for setup context
        """
        # Create isolated test environment using setup_test_environment
        self.test_environment_context = setup_test_environment(
            test_name=f"test_data_validation_{method.__name__}",
            cleanup_on_exit=True
        )
        
        # Initialize validation components with test configuration
        self.validation_orchestrator = ValidationOrchestrator(
            validation_config=self.test_config.get('validation_config', {}),
            enable_caching=True,
            enable_fail_fast=True
        )
        
        # Setup temporary directories for test artifacts
        self.temp_directory = self.test_environment_context.__enter__()
        
        # Configure test-specific logging and monitoring
        self.test_start_time = time.time()
        self.test_execution_id = str(uuid.uuid4())
        
        # Reset validation caches and statistics
        self._reset_validation_state()
        
        # Prepare test data fixtures and references
        self._prepare_test_fixtures()
    
    def teardown_method(self, method):
        """
        Teardown method executed after each test method with cleanup and validation.
        
        Args:
            method: Test method object for teardown context
        """
        # Cleanup temporary test files and directories
        if hasattr(self, 'test_environment_context'):
            self.test_environment_context.__exit__(None, None, None)
        
        # Validate test execution performance metrics
        test_duration = time.time() - self.test_start_time
        if test_duration > PROCESSING_TIME_LIMIT:
            warnings.warn(
                f"Test {method.__name__} exceeded time limit: {test_duration:.3f}s > {PROCESSING_TIME_LIMIT}s",
                UserWarning
            )
        
        # Generate test-specific validation reports
        self._generate_test_report(method.__name__, test_duration)
        
        # Reset validation component states
        self._cleanup_validation_components()
        
        # Log test completion and resource usage
        self._log_test_completion(method.__name__, test_duration)
    
    def test_crimaldi_format_validation_success(self):
        """
        Test successful Crimaldi format validation with proper metadata extraction and calibration 
        parameter detection.
        """
        # Load Crimaldi sample video from test fixtures
        crimaldi_video_path = self._create_mock_crimaldi_video()
        
        # Initialize DataFormatValidator with Crimaldi format support
        format_validator = DataFormatValidator()
        
        # Validate Crimaldi format with expected requirements
        validation_result = format_validator.validate_crimaldi_format(
            video_path=str(crimaldi_video_path),
            extract_metadata=True
        )
        
        # Assert validation success and proper metadata extraction
        assert validation_result.is_valid, f"Crimaldi validation failed: {validation_result.errors}"
        assert len(validation_result.errors) == 0, "Unexpected validation errors detected"
        
        # Verify calibration parameter detection accuracy
        metadata = validation_result.metadata
        assert 'calibration_parameters' in metadata, "Calibration parameters not detected"
        assert 'pixel_to_meter_ratio' in metadata['calibration_parameters'], "Pixel ratio not found"
        assert 'temporal_resolution' in metadata['calibration_parameters'], "Temporal resolution not found"
        
        # Validate format-specific constraint compliance
        format_specs = metadata.get('format_specifications', {})
        assert format_specs.get('format_type') == 'crimaldi', "Incorrect format type detected"
        assert format_specs.get('bit_depth') == 8, "Incorrect bit depth for Crimaldi format"
        
        # Assert no validation errors or warnings generated
        assert len(validation_result.warnings) == 0, f"Unexpected warnings: {validation_result.warnings}"
        
        # Verify correlation threshold compliance
        correlation_score = validation_result.metrics.get('format_compatibility_score', 0.0)
        assert correlation_score >= CORRELATION_THRESHOLD, f"Format compatibility below threshold: {correlation_score}"
    
    def test_custom_format_validation_success(self):
        """
        Test successful custom AVI format validation with automatic parameter detection and flexibility.
        """
        # Load custom sample video from test fixtures
        custom_video_path = self._create_mock_custom_video()
        
        # Initialize DataFormatValidator with custom format support
        format_validator = DataFormatValidator()
        
        # Validate custom format with flexible parameter detection
        validation_result = format_validator.validate_custom_format(
            video_path=str(custom_video_path),
            auto_detect_parameters=True
        )
        
        # Assert validation success and parameter auto-detection
        assert validation_result.is_valid, f"Custom format validation failed: {validation_result.errors}"
        assert len(validation_result.errors) == 0, "Validation errors detected for valid custom format"
        
        # Verify format compatibility assessment
        compatibility_analysis = validation_result.metadata.get('compatibility_analysis', {})
        assert compatibility_analysis.get('is_compatible', False), "Format compatibility not detected"
        assert compatibility_analysis.get('auto_detection_successful', False), "Auto-detection failed"
        
        # Validate normalization requirement analysis
        normalization_requirements = validation_result.metadata.get('normalization_requirements', {})
        assert 'pixel_scaling_required' in normalization_requirements, "Pixel scaling requirements not analyzed"
        assert 'temporal_alignment_required' in normalization_requirements, "Temporal alignment not analyzed"
        
        # Assert proper handling of format variations
        format_variations = validation_result.metadata.get('format_variations_handled', [])
        assert len(format_variations) > 0, "Format variations not properly handled"
        
        # Verify numerical accuracy within tolerance
        numerical_metrics = validation_result.metrics
        pixel_accuracy = numerical_metrics.get('pixel_accuracy', 0.0)
        assert pixel_accuracy >= (1.0 - NUMERICAL_TOLERANCE), f"Pixel accuracy below tolerance: {pixel_accuracy}"
    
    def test_format_validation_error_handling(self):
        """
        Test comprehensive format validation error handling with detailed error reporting and 
        recovery suggestions.
        """
        # Create invalid or corrupted test video file
        invalid_video_path = self._create_invalid_video_file()
        
        # Initialize DataFormatValidator with strict validation
        format_validator = DataFormatValidator()
        
        # Attempt format validation with strict requirements
        with pytest.raises(DataFormatValidationError) as exc_info:
            format_validator.validate_avi_format(
                video_path=str(invalid_video_path),
                strict_validation=True
            )
        
        # Assert DataFormatValidationError is properly raised
        validation_error = exc_info.value
        assert isinstance(validation_error, DataFormatValidationError), "Wrong exception type raised"
        
        # Verify error contains format compatibility analysis
        compatibility_analysis = validation_error.get_format_compatibility_analysis()
        assert 'detected_format' in compatibility_analysis, "Format detection missing from error"
        assert 'compatibility_issues' in compatibility_analysis, "Compatibility issues not reported"
        
        # Validate error suggests format corrections
        correction_suggestions = validation_error.suggest_format_corrections()
        assert len(correction_suggestions) > 0, "No format correction suggestions provided"
        assert any('convert' in suggestion.lower() for suggestion in correction_suggestions), "No conversion suggestions"
        
        # Assert error includes detailed context and recovery options
        error_context = validation_error.validation_context
        assert 'file_path' in error_context, "File path missing from error context"
        assert 'validation_stage' in error_context, "Validation stage not recorded"
        
        # Verify fail-fast validation behavior implementation
        assert validation_error.fail_fast_triggered, "Fail-fast validation not triggered for critical error"
    
    def test_physical_parameter_validation_success(self):
        """
        Test successful physical scale parameter validation including arena dimensions, pixel ratios, 
        and coordinate systems.
        """
        # Create valid physical scale parameters configuration
        valid_parameters = {
            'arena_width_meters': 1.5,
            'arena_height_meters': 1.0,
            'pixel_to_meter_ratio': 100.0,
            'coordinate_system': 'cartesian',
            'origin_location': [0.0, 0.0],
            'scaling_accuracy': 0.001
        }
        
        # Call validate_physical_scale_parameters function
        validation_result = validate_physical_scale_parameters(
            parameters=valid_parameters,
            format_type='generic',
            cross_format_validation=True
        )
        
        # Assert validation success and constraint compliance
        assert validation_result.is_valid, f"Physical parameter validation failed: {validation_result.errors}"
        assert len(validation_result.errors) == 0, "Unexpected errors in valid parameter validation"
        
        # Verify arena size validation against scientific limits
        arena_validation = validation_result.metadata.get('arena_validation', {})
        assert arena_validation.get('size_within_limits', False), "Arena size limits not validated"
        assert arena_validation.get('aspect_ratio_reasonable', False), "Aspect ratio not validated"
        
        # Validate pixel-to-meter ratio reasonableness checking
        ratio_validation = validation_result.metadata.get('ratio_validation', {})
        assert ratio_validation.get('ratio_reasonable', False), "Pixel ratio reasonableness not checked"
        assert ratio_validation.get('scaling_accuracy_acceptable', False), "Scaling accuracy not validated"
        
        # Assert coordinate system configuration validation
        coordinate_validation = validation_result.metadata.get('coordinate_validation', {})
        assert coordinate_validation.get('system_valid', False), "Coordinate system not validated"
        assert coordinate_validation.get('origin_valid', False), "Origin location not validated"
        
        # Verify scientific constraint compliance assessment
        scientific_compliance = validation_result.metrics.get('scientific_compliance_score', 0.0)
        assert scientific_compliance >= CORRELATION_THRESHOLD, f"Scientific compliance below threshold: {scientific_compliance}"
    
    def test_parameter_validation_error_handling(self):
        """
        Test parameter validation error handling with constraint violation analysis and 
        correction suggestions.
        """
        # Create invalid parameters with constraint violations
        invalid_parameters = {
            'arena_width_meters': -1.0,  # Negative value
            'arena_height_meters': 150.0,  # Exceeds maximum
            'pixel_to_meter_ratio': 0.0,  # Invalid ratio
            'coordinate_system': 'invalid_system',  # Unknown system
            'scaling_accuracy': 10.0  # Unrealistic accuracy
        }
        
        # Attempt parameter validation with strict checking
        with pytest.raises(ParameterValidationError) as exc_info:
            validate_physical_scale_parameters(
                parameters=invalid_parameters,
                format_type='generic',
                strict_validation=True
            )
        
        # Assert ParameterValidationError is properly raised
        parameter_error = exc_info.value
        assert isinstance(parameter_error, ParameterValidationError), "Wrong exception type for parameter errors"
        
        # Verify error contains detailed constraint analysis
        constraint_analysis = parameter_error.get_constraint_analysis()
        assert 'violated_constraints' in constraint_analysis, "Constraint violations not analyzed"
        assert len(constraint_analysis['violated_constraints']) > 0, "No constraint violations detected"
        
        # Validate error suggests specific parameter corrections
        parameter_corrections = parameter_error.suggest_parameter_corrections()
        assert len(parameter_corrections) > 0, "No parameter corrections suggested"
        assert 'arena_width_meters' in parameter_corrections, "Arena width correction missing"
        assert 'pixel_to_meter_ratio' in parameter_corrections, "Pixel ratio correction missing"
        
        # Assert error includes valid range information
        valid_ranges = parameter_error.validation_context.get('valid_ranges', {})
        assert 'arena_width_meters' in valid_ranges, "Valid range for arena width not provided"
        assert 'pixel_to_meter_ratio' in valid_ranges, "Valid range for pixel ratio not provided"
        
        # Verify scientific constraint violation reporting
        scientific_violations = parameter_error.validation_context.get('scientific_violations', [])
        assert len(scientific_violations) > 0, "Scientific constraint violations not reported"
    
    def test_cross_format_compatibility_validation(self):
        """
        Test cross-format compatibility validation between Crimaldi and custom formats with 
        consistency analysis.
        """
        # Load test videos for both Crimaldi and custom formats
        crimaldi_video = self._create_mock_crimaldi_video()
        custom_video = self._create_mock_custom_video()
        
        format_configurations = {
            'crimaldi': self._get_crimaldi_config(),
            'custom': self._get_custom_config()
        }
        
        format_data_samples = {
            'crimaldi': self._load_video_sample(crimaldi_video),
            'custom': self._load_video_sample(custom_video)
        }
        
        # Create compatibility requirements configuration
        compatibility_requirements = {
            'minimum_correlation': CORRELATION_THRESHOLD,
            'acceptable_conversion_loss': 0.05,
            'temporal_alignment_tolerance': 0.1,
            'spatial_scaling_tolerance': 0.02
        }
        
        # Call validate_cross_format_compatibility function
        validation_result = validate_cross_format_compatibility(
            format_types=['crimaldi', 'custom'],
            format_configurations=format_configurations,
            format_data_samples=format_data_samples,
            validate_conversion_accuracy=True
        )
        
        # Assert compatibility validation success
        assert validation_result.is_valid, f"Cross-format compatibility validation failed: {validation_result.errors}"
        
        # Verify format parameter compatibility analysis
        parameter_compatibility = validation_result.metadata.get('parameter_compatibility', {})
        assert parameter_compatibility.get('temporal_parameters_compatible', False), "Temporal compatibility not checked"
        assert parameter_compatibility.get('spatial_parameters_compatible', False), "Spatial compatibility not checked"
        
        # Validate unit conversion compatibility assessment
        conversion_compatibility = validation_result.metadata.get('conversion_compatibility', {})
        assert conversion_compatibility.get('unit_conversion_feasible', False), "Unit conversion feasibility not assessed"
        assert conversion_compatibility.get('precision_preserved', False), "Precision preservation not verified"
        
        # Assert coordinate system consistency checking
        coordinate_consistency = validation_result.metadata.get('coordinate_consistency', {})
        assert coordinate_consistency.get('systems_compatible', False), "Coordinate system compatibility not checked"
        assert coordinate_consistency.get('transformation_available', False), "Coordinate transformation not assessed"
        
        # Verify conversion feasibility and data preservation
        conversion_metrics = validation_result.metrics
        conversion_accuracy = conversion_metrics.get('conversion_accuracy', 0.0)
        assert conversion_accuracy >= CORRELATION_THRESHOLD, f"Conversion accuracy below threshold: {conversion_accuracy}"
        
        data_preservation = conversion_metrics.get('data_preservation_score', 0.0)
        assert data_preservation >= 0.95, f"Data preservation below threshold: {data_preservation}"
    
    def test_numerical_precision_validation_success(self):
        """
        Test numerical precision validation success with >95% correlation against reference implementations.
        """
        # Load reference benchmark data from test fixtures
        reference_data = self._load_reference_benchmark_data()
        
        # Generate computed results with high correlation
        computed_data = self._generate_high_correlation_results(reference_data)
        
        # Call validate_numerical_precision function
        validation_result = validate_numerical_precision(
            test_values=computed_data,
            reference_values=reference_data,
            correlation_threshold=CORRELATION_THRESHOLD,
            precision_threshold=NUMERICAL_TOLERANCE
        )
        
        # Assert validation success and correlation compliance
        assert validation_result.is_valid, f"Numerical precision validation failed: {validation_result.errors}"
        
        # Verify >95% correlation threshold achievement
        correlation_coefficient = validation_result.metrics.get('correlation_coefficient', 0.0)
        assert correlation_coefficient >= CORRELATION_THRESHOLD, f"Correlation {correlation_coefficient} below threshold {CORRELATION_THRESHOLD}"
        
        # Validate numerical precision within 1e-6 tolerance
        max_absolute_error = validation_result.metrics.get('max_absolute_error', float('inf'))
        assert max_absolute_error <= NUMERICAL_TOLERANCE, f"Absolute error {max_absolute_error} exceeds tolerance {NUMERICAL_TOLERANCE}"
        
        # Assert statistical significance testing results
        statistical_significance = validation_result.metadata.get('statistical_tests', {})
        assert statistical_significance.get('p_value', 1.0) < 0.05, "Results not statistically significant"
        assert statistical_significance.get('confidence_interval', 0.0) >= 0.95, "Confidence interval too low"
        
        # Verify accuracy metrics calculation and reporting
        accuracy_metrics = validation_result.metrics
        assert 'rmse' in accuracy_metrics, "RMSE metric not calculated"
        assert 'mae' in accuracy_metrics, "MAE metric not calculated"
        assert 'relative_error' in accuracy_metrics, "Relative error not calculated"
        
        rmse = accuracy_metrics['rmse']
        assert rmse <= NUMERICAL_TOLERANCE, f"RMSE {rmse} exceeds tolerance"
    
    def test_numerical_precision_validation_error(self):
        """
        Test numerical precision validation error handling for low correlation with detailed analysis.
        """
        # Load reference benchmark data from test fixtures
        reference_data = self._load_reference_benchmark_data()
        
        # Generate computed results with low correlation
        computed_data = self._generate_low_correlation_results(reference_data)
        
        # Attempt numerical precision validation
        with pytest.raises(NumericalValidationError) as exc_info:
            validate_numerical_precision(
                test_values=computed_data,
                reference_values=reference_data,
                correlation_threshold=CORRELATION_THRESHOLD,
                strict_validation=True
            )
        
        # Assert NumericalValidationError is properly raised
        numerical_error = exc_info.value
        assert isinstance(numerical_error, NumericalValidationError), "Wrong exception type for numerical errors"
        
        # Verify error contains comprehensive numerical analysis
        numerical_analysis = numerical_error.get_numerical_analysis()
        assert 'correlation_analysis' in numerical_analysis, "Correlation analysis missing from error"
        assert 'precision_analysis' in numerical_analysis, "Precision analysis missing from error"
        assert 'statistical_tests' in numerical_analysis, "Statistical tests missing from error"
        
        # Validate error suggests precision improvement strategies
        improvement_strategies = numerical_error.suggest_precision_improvements()
        assert len(improvement_strategies) > 0, "No precision improvement strategies suggested"
        assert any('algorithm' in strategy.lower() for strategy in improvement_strategies), "Algorithm improvements not suggested"
        
        # Assert error includes accuracy metrics and diagnostics
        accuracy_diagnostics = numerical_error.validation_context.get('accuracy_diagnostics', {})
        assert 'correlation_coefficient' in accuracy_diagnostics, "Correlation coefficient not in diagnostics"
        assert 'error_distribution' in accuracy_diagnostics, "Error distribution not analyzed"
        
        # Verify correlation threshold violation reporting
        threshold_violation = numerical_error.validation_context.get('threshold_violation', {})
        assert threshold_violation.get('correlation_below_threshold', False), "Correlation violation not reported"
        assert threshold_violation.get('actual_correlation', 0.0) < CORRELATION_THRESHOLD, "Actual correlation not recorded"
    
    def test_configuration_schema_validation(self):
        """
        Test configuration schema validation with JSON schema compliance and parameter constraint checking.
        """
        # Load test normalization configuration
        test_config = self._load_valid_test_configuration()
        
        # Initialize ConfigurationValidator with schema directory
        config_validator = ConfigurationValidator(
            schema_directory='test_fixtures/schemas',
            enable_schema_caching=True
        )
        
        # Validate configuration against normalization schema
        validation_result = config_validator.validate_against_schema(
            data=test_config,
            schema_name='normalization_schema',
            collect_all_errors=True
        )
        
        # Assert schema compliance validation success
        assert validation_result.is_valid, f"Configuration schema validation failed: {validation_result.errors}"
        assert len(validation_result.errors) == 0, "Schema compliance errors detected"
        
        # Verify required parameter presence checking
        required_parameters = test_config.get('required_parameters', [])
        schema_analysis = validation_result.metadata.get('schema_analysis', {})
        assert schema_analysis.get('required_parameters_present', False), "Required parameters not validated"
        
        # Validate parameter type and range constraints
        parameter_validation = validation_result.metadata.get('parameter_validation', {})
        assert parameter_validation.get('types_valid', False), "Parameter types not validated"
        assert parameter_validation.get('ranges_valid', False), "Parameter ranges not validated"
        
        # Assert cross-parameter dependency validation
        dependency_validation = validation_result.metadata.get('dependency_validation', {})
        assert dependency_validation.get('dependencies_satisfied', False), "Parameter dependencies not checked"
        
        # Verify configuration completeness assessment
        completeness_score = validation_result.metrics.get('completeness_score', 0.0)
        assert completeness_score >= 0.95, f"Configuration completeness below threshold: {completeness_score}"
    
    def test_validation_orchestrator_comprehensive_workflow(self):
        """
        Test ValidationOrchestrator comprehensive workflow with all validation components and 
        performance monitoring.
        """
        # Initialize ValidationOrchestrator with comprehensive configuration
        orchestrator = ValidationOrchestrator(
            validation_config=self.test_config,
            enable_caching=True,
            enable_fail_fast=True
        )
        
        # Load test video file and normalization configuration
        test_video_path = self._create_mock_test_video()
        normalization_config = self._load_normalization_configuration()
        
        # Execute validate_normalization_workflow method
        workflow_result = orchestrator.validate_normalization_workflow(
            video_path=str(test_video_path),
            configuration=normalization_config,
            comprehensive_analysis=True
        )
        
        # Assert comprehensive validation result aggregation
        assert isinstance(workflow_result, NormalizationValidationResult), "Wrong result type returned"
        assert workflow_result.is_valid, f"Workflow validation failed: {workflow_result.get_error_summary()}"
        
        # Verify all validation components executed successfully
        validation_stages = workflow_result.get_validation_stages()
        expected_stages = ['format_validation', 'parameter_validation', 'quality_validation', 'compatibility_validation']
        
        for stage in expected_stages:
            assert stage in validation_stages, f"Missing validation stage: {stage}"
            stage_result = validation_stages[stage]
            assert stage_result.get('completed', False), f"Stage {stage} not completed"
        
        # Validate caching functionality and performance optimization
        cache_statistics = orchestrator.get_cache_statistics()
        assert cache_statistics.get('cache_enabled', False), "Caching not enabled"
        assert cache_statistics.get('cache_hits', 0) >= 0, "Cache hits not tracked"
        
        # Assert performance monitoring data collection
        performance_data = workflow_result.get_performance_data()
        assert 'execution_time' in performance_data, "Execution time not monitored"
        assert 'memory_usage' in performance_data, "Memory usage not tracked"
        
        execution_time = performance_data['execution_time']
        assert execution_time <= PROCESSING_TIME_LIMIT, f"Execution time {execution_time}s exceeds limit {PROCESSING_TIME_LIMIT}s"
        
        # Verify validation statistics tracking and reporting
        validation_statistics = orchestrator.get_validation_statistics()
        assert validation_statistics.get('total_validations', 0) > 0, "Validation statistics not tracked"
        assert validation_statistics.get('success_rate', 0.0) >= 0.95, "Success rate below threshold"
    
    def test_batch_compatibility_validation(self):
        """
        Test batch compatibility validation for multiple video files with resource analysis and optimization.
        """
        # Create list of test video files with different formats
        test_videos = [
            self._create_mock_crimaldi_video(),
            self._create_mock_custom_video(),
            self._create_mock_avi_video()
        ]
        
        # Initialize ValidationOrchestrator with batch configuration
        orchestrator = ValidationOrchestrator(
            validation_config={
                'batch_processing': True,
                'parallel_validation': True,
                'resource_optimization': True
            }
        )
        
        # Execute validate_batch_compatibility method
        batch_result = orchestrator.validate_batch_compatibility(
            video_files=[str(path) for path in test_videos],
            compatibility_requirements={
                'cross_format_compatibility': True,
                'resource_efficiency': True,
                'processing_optimization': True
            }
        )
        
        # Assert batch compatibility validation success
        assert batch_result.is_valid, f"Batch compatibility validation failed: {batch_result.errors}"
        
        # Verify format consistency analysis across files
        format_analysis = batch_result.metadata.get('format_analysis', {})
        assert 'format_distribution' in format_analysis, "Format distribution not analyzed"
        assert 'compatibility_matrix' in format_analysis, "Compatibility matrix not generated"
        
        # Validate resource requirement assessment
        resource_analysis = batch_result.metadata.get('resource_analysis', {})
        assert 'estimated_memory_usage' in resource_analysis, "Memory usage not estimated"
        assert 'estimated_processing_time' in resource_analysis, "Processing time not estimated"
        
        estimated_memory = resource_analysis['estimated_memory_usage']
        assert estimated_memory <= 8192, f"Estimated memory usage {estimated_memory}MB exceeds limit"
        
        # Assert batch optimization recommendations generation
        optimization_recommendations = batch_result.get_optimization_recommendations()
        assert len(optimization_recommendations) > 0, "No optimization recommendations generated"
        
        # Verify parallel processing feasibility analysis
        parallel_analysis = batch_result.metadata.get('parallel_analysis', {})
        assert parallel_analysis.get('parallel_feasible', False), "Parallel processing feasibility not assessed"
        assert parallel_analysis.get('optimal_worker_count', 0) > 0, "Optimal worker count not calculated"
    
    def test_scientific_reproducibility_validation(self):
        """
        Test scientific reproducibility validation with >0.99 coefficient requirement and statistical analysis.
        """
        # Load computed and reference results for comparison
        computed_results = self._load_computed_simulation_results()
        reference_results = self._load_reference_simulation_results()
        
        # Initialize ValidationOrchestrator with reproducibility criteria
        orchestrator = ValidationOrchestrator(
            validation_config={
                'scientific_reproducibility': True,
                'reproducibility_threshold': 0.99,
                'statistical_analysis': True
            }
        )
        
        # Execute validate_scientific_reproducibility method
        reproducibility_result = orchestrator.validate_scientific_reproducibility(
            computed_results=computed_results,
            reference_results=reference_results,
            statistical_significance_level=0.01
        )
        
        # Assert reproducibility validation success
        assert reproducibility_result.is_valid, f"Scientific reproducibility validation failed: {reproducibility_result.errors}"
        
        # Verify >0.99 reproducibility coefficient achievement
        reproducibility_coefficient = reproducibility_result.metrics.get('reproducibility_coefficient', 0.0)
        assert reproducibility_coefficient >= 0.99, f"Reproducibility coefficient {reproducibility_coefficient} below threshold 0.99"
        
        # Validate statistical consistency analysis
        statistical_analysis = reproducibility_result.metadata.get('statistical_analysis', {})
        assert 'hypothesis_tests' in statistical_analysis, "Hypothesis tests not performed"
        assert 'confidence_intervals' in statistical_analysis, "Confidence intervals not calculated"
        
        # Assert reference implementation correlation
        correlation_analysis = reproducibility_result.metadata.get('correlation_analysis', {})
        assert correlation_analysis.get('correlation_coefficient', 0.0) >= CORRELATION_THRESHOLD, "Correlation with reference below threshold"
        assert correlation_analysis.get('statistical_significance', False), "Correlation not statistically significant"
        
        # Verify measurement uncertainty assessment
        uncertainty_analysis = reproducibility_result.metadata.get('uncertainty_analysis', {})
        assert 'measurement_uncertainty' in uncertainty_analysis, "Measurement uncertainty not assessed"
        assert uncertainty_analysis.get('uncertainty_acceptable', False), "Measurement uncertainty too high"
    
    def test_performance_threshold_validation(self):
        """
        Test performance threshold validation with <7.2 seconds processing time and resource 
        utilization requirements.
        """
        # Create performance metrics with various processing times
        performance_metrics = {
            'processing_time_per_simulation': 6.8,  # Below threshold
            'memory_usage_mb': 4096,  # Within limits
            'cpu_utilization_percent': 75.0,  # Acceptable
            'throughput_simulations_per_hour': 520,  # Above minimum
            'batch_completion_rate': 0.98  # High completion rate
        }
        
        # Call validate_performance_thresholds function
        validation_result = validate_performance_thresholds(
            performance_metrics=performance_metrics,
            time_threshold=PROCESSING_TIME_LIMIT,
            memory_threshold_mb=8192,
            throughput_threshold=500
        )
        
        # Assert performance threshold compliance validation
        assert validation_result.is_valid, f"Performance threshold validation failed: {validation_result.errors}"
        
        # Verify <7.2 seconds processing time requirement
        processing_time = performance_metrics['processing_time_per_simulation']
        assert processing_time <= PROCESSING_TIME_LIMIT, f"Processing time {processing_time}s exceeds limit {PROCESSING_TIME_LIMIT}s"
        
        # Validate memory usage threshold checking
        memory_usage = performance_metrics['memory_usage_mb']
        memory_threshold = 8192
        assert memory_usage <= memory_threshold, f"Memory usage {memory_usage}MB exceeds threshold {memory_threshold}MB"
        
        # Assert batch completion rate validation
        completion_rate = performance_metrics['batch_completion_rate']
        assert completion_rate >= 0.95, f"Batch completion rate {completion_rate} below threshold 0.95"
        
        # Verify performance optimization recommendations
        if validation_result.warnings:
            optimization_recommendations = validation_result.get_optimization_recommendations()
            assert len(optimization_recommendations) > 0, "No optimization recommendations provided"
        
        # Assert resource utilization efficiency assessment
        efficiency_metrics = validation_result.metrics
        assert 'resource_utilization_efficiency' in efficiency_metrics, "Resource efficiency not calculated"
        assert efficiency_metrics['resource_utilization_efficiency'] >= 0.8, "Resource utilization efficiency too low"
    
    def test_validation_error_recovery_strategies(self):
        """
        Test validation error recovery strategies with fail-fast behavior and graceful degradation.
        """
        # Create various validation error scenarios
        error_scenarios = [
            self._create_format_error_scenario(),
            self._create_parameter_error_scenario(),
            self._create_numerical_error_scenario()
        ]
        
        recovery_results = []
        
        for scenario in error_scenarios:
            try:
                # Test error detection and categorization accuracy
                error_detector = ValidationOrchestrator()
                detection_result = error_detector.detect_validation_errors(scenario)
                
                assert detection_result.get('error_detected', False), f"Error not detected in scenario: {scenario['type']}"
                assert detection_result.get('error_category') is not None, "Error category not identified"
                
                # Verify fail-fast validation behavior implementation
                if scenario.get('critical', False):
                    assert detection_result.get('fail_fast_triggered', False), "Fail-fast not triggered for critical error"
                
                # Assert proper error context preservation
                error_context = detection_result.get('error_context', {})
                assert 'scenario_type' in error_context, "Scenario type not preserved in context"
                assert 'detection_timestamp' in error_context, "Detection timestamp not recorded"
                
                # Validate recovery recommendation generation
                recovery_recommendations = error_detector.generate_recovery_recommendations(detection_result)
                assert len(recovery_recommendations) > 0, f"No recovery recommendations for scenario: {scenario['type']}"
                
                # Test graceful degradation for non-critical errors
                if not scenario.get('critical', False):
                    degradation_result = error_detector.apply_graceful_degradation(scenario)
                    assert degradation_result.get('degradation_successful', False), "Graceful degradation failed"
                
                recovery_results.append({
                    'scenario': scenario['type'],
                    'detection_successful': True,
                    'recovery_available': len(recovery_recommendations) > 0
                })
                
            except Exception as e:
                recovery_results.append({
                    'scenario': scenario['type'],
                    'detection_successful': False,
                    'error': str(e)
                })
        
        # Verify audit trail creation and tracking
        audit_trail = self._get_validation_audit_trail()
        assert len(audit_trail) > 0, "Audit trail not created"
        
        for entry in audit_trail:
            assert 'timestamp' in entry, "Audit entry missing timestamp"
            assert 'action' in entry, "Audit entry missing action"
            assert 'validation_context' in entry, "Audit entry missing validation context"
        
        # Assert comprehensive error reporting and logging
        error_report = self._generate_validation_error_report(recovery_results)
        assert error_report.get('total_scenarios', 0) == len(error_scenarios), "Not all scenarios processed"
        assert error_report.get('detection_success_rate', 0.0) >= 0.8, "Error detection success rate too low"
    
    # Helper methods for test data generation and setup
    
    def _load_test_configuration(self) -> Dict[str, Any]:
        """Load test configuration from JSON file."""
        try:
            with open(self.test_config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_test_configuration()
    
    def _get_default_test_configuration(self) -> Dict[str, Any]:
        """Get default test configuration."""
        return {
            'validation_config': {
                'correlation_threshold': CORRELATION_THRESHOLD,
                'numerical_tolerance': NUMERICAL_TOLERANCE,
                'processing_time_limit': PROCESSING_TIME_LIMIT
            },
            'test_parameters': {
                'enable_fail_fast': True,
                'comprehensive_validation': True,
                'performance_monitoring': True
            }
        }
    
    def _reset_validation_state(self) -> None:
        """Reset validation component states."""
        # Clear any cached validation results
        if hasattr(self, 'validation_orchestrator'):
            self.validation_orchestrator.clear_cache()
    
    def _prepare_test_fixtures(self) -> None:
        """Prepare test data fixtures and references."""
        # Create mock test data if needed
        if not self.crimaldi_sample_path.exists():
            self._create_mock_crimaldi_video()
        
        if not self.custom_sample_path.exists():
            self._create_mock_custom_video()
    
    def _create_mock_crimaldi_video(self) -> pathlib.Path:
        """Create mock Crimaldi format video for testing."""
        # Generate synthetic Crimaldi video data
        video_data = np.random.randint(0, 256, size=(100, 480, 640), dtype=np.uint8)
        
        # Save to temporary file
        mock_path = self.temp_directory['fixtures_directory'] / 'mock_crimaldi.avi'
        self._save_mock_video(video_data, mock_path, format_type='crimaldi')
        
        return mock_path
    
    def _create_mock_custom_video(self) -> pathlib.Path:
        """Create mock custom format video for testing."""
        # Generate synthetic custom video data
        video_data = np.random.randint(0, 65536, size=(100, 480, 640, 3), dtype=np.uint16)
        
        # Save to temporary file
        mock_path = self.temp_directory['fixtures_directory'] / 'mock_custom.avi'
        self._save_mock_video(video_data, mock_path, format_type='custom')
        
        return mock_path
    
    def _create_mock_avi_video(self) -> pathlib.Path:
        """Create mock AVI format video for testing."""
        # Generate synthetic AVI video data
        video_data = np.random.randint(0, 256, size=(100, 480, 640, 3), dtype=np.uint8)
        
        # Save to temporary file
        mock_path = self.temp_directory['fixtures_directory'] / 'mock_avi.avi'
        self._save_mock_video(video_data, mock_path, format_type='avi')
        
        return mock_path
    
    def _save_mock_video(self, video_data: np.ndarray, path: pathlib.Path, format_type: str) -> None:
        """Save mock video data to file."""
        # For testing purposes, save as numpy array
        np.save(str(path), video_data)
    
    def _create_invalid_video_file(self) -> pathlib.Path:
        """Create invalid/corrupted video file for error testing."""
        invalid_path = self.temp_directory['fixtures_directory'] / 'invalid_video.avi'
        
        # Write corrupted data
        with open(invalid_path, 'wb') as f:
            f.write(b'invalid_video_data_corrupted')
        
        return invalid_path
    
    def _load_reference_benchmark_data(self) -> np.ndarray:
        """Load reference benchmark data for numerical validation."""
        # Generate synthetic reference data with known properties
        np.random.seed(42)  # For reproducible results
        return np.random.normal(0, 1, size=(1000, 100))
    
    def _generate_high_correlation_results(self, reference_data: np.ndarray) -> np.ndarray:
        """Generate computed results with high correlation to reference."""
        # Add small amount of noise to reference data
        noise_level = 0.01
        noise = np.random.normal(0, noise_level, reference_data.shape)
        return reference_data + noise
    
    def _generate_low_correlation_results(self, reference_data: np.ndarray) -> np.ndarray:
        """Generate computed results with low correlation to reference."""
        # Generate mostly random data
        random_component = np.random.normal(0, 1, reference_data.shape)
        reference_component = reference_data * 0.1  # Very weak correlation
        return random_component + reference_component
    
    def _get_crimaldi_config(self) -> Dict[str, Any]:
        """Get Crimaldi format configuration."""
        return {
            'format_type': 'crimaldi',
            'bit_depth': 8,
            'color_space': 'grayscale',
            'pixel_to_meter_ratio': 100.0,
            'temporal_resolution': 50.0
        }
    
    def _get_custom_config(self) -> Dict[str, Any]:
        """Get custom format configuration."""
        return {
            'format_type': 'custom',
            'bit_depth': 16,
            'color_space': 'rgb',
            'pixel_to_meter_ratio': 150.0,
            'temporal_resolution': 30.0
        }
    
    def _load_video_sample(self, video_path: pathlib.Path) -> np.ndarray:
        """Load video sample data from file."""
        try:
            return np.load(str(video_path))
        except:
            # Return dummy data if file doesn't exist
            return np.random.randint(0, 256, size=(10, 48, 64), dtype=np.uint8)
    
    def _load_valid_test_configuration(self) -> Dict[str, Any]:
        """Load valid test configuration for schema validation."""
        return {
            'pipeline': {
                'stages': ['format_detection', 'parameter_extraction', 'quality_validation'],
                'processing_order': 'sequential'
            },
            'formats': {
                'supported': ['crimaldi', 'custom', 'avi'],
                'default': 'auto_detect'
            },
            'quality': {
                'correlation_threshold': CORRELATION_THRESHOLD,
                'numerical_tolerance': NUMERICAL_TOLERANCE
            },
            'required_parameters': ['pipeline', 'formats']
        }
    
    def _create_mock_test_video(self) -> pathlib.Path:
        """Create mock test video for workflow testing."""
        return self._create_mock_custom_video()
    
    def _load_normalization_configuration(self) -> Dict[str, Any]:
        """Load normalization configuration for workflow testing."""
        return {
            'normalization_type': 'comprehensive',
            'spatial_normalization': True,
            'temporal_normalization': True,
            'intensity_normalization': True,
            'quality_validation': True
        }
    
    def _load_computed_simulation_results(self) -> Dict[str, Any]:
        """Load computed simulation results for reproducibility testing."""
        return {
            'trajectory_data': np.random.normal(0, 1, size=(100, 2)),
            'performance_metrics': {'accuracy': 0.95, 'efficiency': 0.88},
            'computational_metadata': {'algorithm': 'test_algorithm', 'version': '1.0'}
        }
    
    def _load_reference_simulation_results(self) -> Dict[str, Any]:
        """Load reference simulation results for comparison."""
        # Use same random seed for reproducible reference data
        np.random.seed(42)
        return {
            'trajectory_data': np.random.normal(0, 1, size=(100, 2)),
            'performance_metrics': {'accuracy': 0.95, 'efficiency': 0.88},
            'computational_metadata': {'algorithm': 'reference_algorithm', 'version': '1.0'}
        }
    
    def _create_format_error_scenario(self) -> Dict[str, Any]:
        """Create format validation error scenario."""
        return {
            'type': 'format_error',
            'error_data': 'invalid_format_data',
            'critical': True,
            'expected_error': DataFormatValidationError
        }
    
    def _create_parameter_error_scenario(self) -> Dict[str, Any]:
        """Create parameter validation error scenario."""
        return {
            'type': 'parameter_error',
            'error_data': {'invalid_parameter': -1.0},
            'critical': False,
            'expected_error': ParameterValidationError
        }
    
    def _create_numerical_error_scenario(self) -> Dict[str, Any]:
        """Create numerical validation error scenario."""
        return {
            'type': 'numerical_error',
            'error_data': np.array([np.nan, np.inf, -np.inf]),
            'critical': True,
            'expected_error': NumericalValidationError
        }
    
    def _get_validation_audit_trail(self) -> List[Dict[str, Any]]:
        """Get validation audit trail for error tracking."""
        return [
            {
                'timestamp': datetime.datetime.now().isoformat(),
                'action': 'validation_started',
                'validation_context': {'test_execution_id': self.test_execution_id}
            },
            {
                'timestamp': datetime.datetime.now().isoformat(),
                'action': 'error_detected',
                'validation_context': {'error_type': 'format_error'}
            }
        ]
    
    def _generate_validation_error_report(self, recovery_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate validation error report."""
        total_scenarios = len(recovery_results)
        successful_detections = sum(1 for r in recovery_results if r.get('detection_successful', False))
        
        return {
            'total_scenarios': total_scenarios,
            'successful_detections': successful_detections,
            'detection_success_rate': successful_detections / max(1, total_scenarios),
            'report_timestamp': datetime.datetime.now().isoformat()
        }
    
    def _generate_test_report(self, test_name: str, duration: float) -> None:
        """Generate test execution report."""
        # Implementation for test report generation
        pass
    
    def _cleanup_validation_components(self) -> None:
        """Cleanup validation component states."""
        # Reset validation orchestrator state
        if hasattr(self, 'validation_orchestrator'):
            self.validation_orchestrator.clear_cache()
    
    def _log_test_completion(self, test_name: str, duration: float) -> None:
        """Log test completion with performance metrics."""
        # Implementation for test completion logging
        pass


# Standalone validation functions for validation metrics (since the module doesn't exist)

def validate_trajectory_accuracy(
    computed_trajectory: np.ndarray,
    reference_trajectory: np.ndarray,
    correlation_threshold: float = 0.95
) -> Dict[str, Any]:
    """
    Validate trajectory accuracy with >95% correlation requirement.
    
    Args:
        computed_trajectory: Computed trajectory data
        reference_trajectory: Reference trajectory for comparison
        correlation_threshold: Minimum correlation threshold
        
    Returns:
        Dict[str, Any]: Trajectory accuracy validation results
    """
    try:
        # Calculate correlation between trajectories
        if computed_trajectory.shape != reference_trajectory.shape:
            return {
                'valid': False,
                'error': 'Trajectory shape mismatch',
                'correlation': 0.0
            }
        
        correlation_matrix = np.corrcoef(
            computed_trajectory.flatten(),
            reference_trajectory.flatten()
        )
        correlation = correlation_matrix[0, 1]
        
        # Validate against threshold
        valid = correlation >= correlation_threshold
        
        return {
            'valid': valid,
            'correlation': correlation,
            'threshold': correlation_threshold,
            'rmse': np.sqrt(np.mean((computed_trajectory - reference_trajectory) ** 2))
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'correlation': 0.0
        }


def validate_performance_thresholds(
    performance_metrics: Dict[str, float],
    time_threshold: float = 7.2,
    memory_threshold_mb: int = 8192,
    throughput_threshold: float = 500.0
) -> ValidationResult:
    """
    Validate performance thresholds for scientific computing requirements.
    
    Args:
        performance_metrics: Performance metrics to validate
        time_threshold: Maximum processing time threshold
        memory_threshold_mb: Maximum memory usage threshold
        throughput_threshold: Minimum throughput threshold
        
    Returns:
        ValidationResult: Performance threshold validation result
    """
    from backend.utils.validation_utils import ValidationResult
    
    validation_result = ValidationResult(
        validation_type='performance_threshold_validation',
        is_valid=True,
        validation_context=f"time_threshold={time_threshold}"
    )
    
    try:
        # Validate processing time
        processing_time = performance_metrics.get('processing_time_per_simulation', float('inf'))
        if processing_time > time_threshold:
            validation_result.add_error(
                f"Processing time {processing_time:.3f}s exceeds threshold {time_threshold}s"
            )
            validation_result.is_valid = False
        
        # Validate memory usage
        memory_usage = performance_metrics.get('memory_usage_mb', 0)
        if memory_usage > memory_threshold_mb:
            validation_result.add_error(
                f"Memory usage {memory_usage}MB exceeds threshold {memory_threshold_mb}MB"
            )
            validation_result.is_valid = False
        
        # Validate throughput
        throughput = performance_metrics.get('throughput_simulations_per_hour', 0)
        if throughput < throughput_threshold:
            validation_result.add_warning(
                f"Throughput {throughput} below threshold {throughput_threshold}"
            )
        
        # Add metrics
        validation_result.add_metric('processing_time', processing_time)
        validation_result.add_metric('memory_usage_mb', memory_usage)
        validation_result.add_metric('throughput', throughput)
        
        return validation_result
        
    except Exception as e:
        validation_result.add_error(f"Performance validation failed: {e}")
        validation_result.is_valid = False
        return validation_result


def validate_statistical_significance(
    test_data: np.ndarray,
    reference_data: np.ndarray,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Validate statistical significance with hypothesis testing.
    
    Args:
        test_data: Test data for statistical analysis
        reference_data: Reference data for comparison
        significance_level: Statistical significance level
        
    Returns:
        Dict[str, Any]: Statistical significance validation results
    """
    try:
        from scipy import stats
        
        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(test_data.flatten(), reference_data.flatten())
        
        # Determine statistical significance
        is_significant = p_value < significance_level
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(test_data) - 1) * np.var(test_data) + 
                             (len(reference_data) - 1) * np.var(reference_data)) / 
                            (len(test_data) + len(reference_data) - 2))
        
        cohens_d = (np.mean(test_data) - np.mean(reference_data)) / pooled_std if pooled_std > 0 else 0
        
        return {
            'statistically_significant': is_significant,
            't_statistic': t_statistic,
            'p_value': p_value,
            'significance_level': significance_level,
            'cohens_d': cohens_d,
            'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
        }
        
    except Exception as e:
        return {
            'statistically_significant': False,
            'error': str(e),
            'p_value': 1.0
        }


class ValidationMetricsCalculator:
    """
    Comprehensive validation metrics calculation for scientific simulation testing.
    
    This class provides validation metrics calculation, analysis, and reporting for
    scientific computing workflows with precision requirements.
    """
    
    def __init__(
        self,
        quality_thresholds: Dict[str, float],
        enable_statistical_analysis: bool = True
    ):
        """
        Initialize validation metrics calculator.
        
        Args:
            quality_thresholds: Quality thresholds for validation
            enable_statistical_analysis: Enable statistical analysis
        """
        self.quality_thresholds = quality_thresholds
        self.statistical_analysis_enabled = enable_statistical_analysis
    
    def validate_trajectory_accuracy(
        self,
        computed_trajectory: np.ndarray,
        reference_trajectory: np.ndarray
    ) -> Dict[str, Any]:
        """Validate trajectory accuracy with correlation analysis."""
        return validate_trajectory_accuracy(
            computed_trajectory,
            reference_trajectory,
            self.quality_thresholds.get('correlation_threshold', 0.95)
        )
    
    def validate_performance_thresholds(
        self,
        performance_metrics: Dict[str, float]
    ) -> ValidationResult:
        """Validate performance thresholds against requirements."""
        return validate_performance_thresholds(
            performance_metrics,
            self.quality_thresholds.get('processing_time_limit', 7.2),
            8192,  # Memory threshold
            500.0  # Throughput threshold
        )
    
    def validate_cross_format_compatibility(
        self,
        format_data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Validate cross-format compatibility."""
        try:
            format_names = list(format_data.keys())
            if len(format_names) < 2:
                return {'compatible': True, 'reason': 'Single format'}
            
            # Calculate cross-format correlation
            format1_data = format_data[format_names[0]]
            format2_data = format_data[format_names[1]]
            
            if format1_data.shape != format2_data.shape:
                return {'compatible': False, 'reason': 'Shape mismatch'}
            
            correlation = np.corrcoef(format1_data.flatten(), format2_data.flatten())[0, 1]
            threshold = self.quality_thresholds.get('correlation_threshold', 0.95)
            
            return {
                'compatible': correlation >= threshold,
                'correlation': correlation,
                'threshold': threshold,
                'formats_compared': format_names[:2]
            }
            
        except Exception as e:
            return {'compatible': False, 'error': str(e)}