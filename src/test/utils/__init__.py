"""
Comprehensive test utilities package initialization module providing centralized access to 
comprehensive testing infrastructure for scientific plume navigation simulation validation.

This module exposes core test helpers, validation metrics, performance monitoring, test data 
generation, and result comparison utilities with >95% correlation requirements and <7.2 seconds 
per simulation performance targets. Supports automated test environment setup, batch processing 
validation, cross-format compatibility testing, and scientific computing validation standards 
for reproducible research workflows.

Key Features:
- Centralized test utilities access with comprehensive testing infrastructure
- Cross-format compatibility testing for Crimaldi and custom plume formats
- Performance monitoring and validation with scientific computing thresholds
- Statistical validation with >95% correlation requirements and hypothesis testing
- Test data generation with realistic physics modeling and format characteristics
- Result comparison with comprehensive statistical analysis and validation
- Batch processing validation for 4000+ simulation execution scenarios
- Scientific computing test standards with 1e-6 numerical tolerance
- Automated test environment setup with resource management and isolation
- Comprehensive test reporting with metrics, analysis, and recommendations
"""

# Package metadata and version information
__version__ = '1.0.0'
__author__ = 'Plume Navigation Test Framework Team'
__description__ = 'Comprehensive test utilities for scientific plume navigation simulation validation'

# Test utilities configuration constants
TEST_UTILITIES_VERSION = '1.0.0'
SUPPORTED_TEST_CATEGORIES = ['unit', 'integration', 'performance', 'validation']
DEFAULT_TEST_CONFIGURATION = {
    'correlation_threshold': 0.95,
    'numerical_tolerance': 1e-6,
    'performance_timeout': 7.2,
    'batch_target': 4000,
    'reproducibility_threshold': 0.99
}

# Core test helper imports providing standardized test fixtures and validation utilities
from .test_helpers import (
    # Test fixture and configuration management functions
    create_test_fixture_path,
    load_test_config,
    
    # Numerical accuracy and simulation validation functions
    assert_arrays_almost_equal,
    assert_simulation_accuracy,
    
    # Performance measurement and monitoring utilities
    measure_performance,
    
    # Mock data generation and synthetic test data creation
    create_mock_video_data,
    
    # Cross-format compatibility testing and validation
    validate_cross_format_compatibility,
    
    # Test environment setup and resource management
    setup_test_environment,
    
    # Batch processing validation for large-scale simulation testing
    validate_batch_processing_results,
    
    # Algorithm performance comparison with statistical analysis
    compare_algorithm_performance,
    
    # Test reporting and documentation generation
    generate_test_report,
    
    # Test data caching and performance optimization
    cache_test_data,
    
    # Comprehensive test data validation classes
    TestDataValidator,
    PerformanceProfiler
)

# Validation metrics imports for statistical analysis and accuracy assessment
from .validation_metrics import (
    # Trajectory accuracy validation with correlation requirements
    validate_trajectory_accuracy,
    
    # Performance threshold validation for scientific computing
    validate_performance_thresholds,
    
    # Statistical significance validation with hypothesis testing
    validate_statistical_significance,
    
    # Reproducibility validation with coefficient requirements
    validate_reproducibility_metrics,
    
    # Algorithm performance ranking with statistical validation
    calculate_algorithm_rankings,
    
    # Benchmark data loading with caching and validation
    load_benchmark_data,
    
    # Comprehensive validation report generation
    generate_validation_report,
    
    # Comprehensive validation metrics calculation classes
    ValidationMetricsCalculator,
    StatisticalValidator,
    BenchmarkComparator
)

# Performance monitoring imports for test execution analysis and resource tracking
from .performance_monitoring import (
    # Test-specific performance monitor creation
    create_test_performance_monitor,
    
    # Test execution performance monitoring decorator
    monitor_test_execution_performance,
    
    # Test performance threshold validation
    validate_test_performance_thresholds,
    
    # Batch test performance tracking
    track_batch_test_performance,
    
    # Resource utilization metrics collection
    collect_test_resource_metrics,
    
    # Performance trend analysis and regression detection
    analyze_test_performance_trends,
    
    # Comprehensive test performance reporting
    generate_test_performance_report,
    
    # Specialized performance monitoring classes
    TestPerformanceMonitor,
    ResourceTracker,
    TestPerformanceContext
)

# Test data generation imports for synthetic data creation and scenario building
from .test_data_generator import (
    # Synthetic plume video generation with realistic physics
    generate_synthetic_plume_video,
    
    # Crimaldi format test data creation
    create_crimaldi_format_data,
    
    # Custom AVI format test data creation
    create_custom_format_data,
    
    # Batch test scenario generation for large-scale validation
    generate_batch_test_scenarios,
    
    # Normalization pipeline test data creation
    create_normalization_test_data,
    
    # Algorithm validation dataset generation
    generate_algorithm_validation_data,
    
    # Performance benchmark dataset creation
    create_performance_benchmark_data,
    
    # Cross-format compatibility test suite generation
    generate_cross_format_test_suite,
    
    # Comprehensive synthetic data generation classes
    SyntheticPlumeGenerator,
    TestScenarioBuilder
)

# Result comparison imports for statistical analysis and validation
from .result_comparator import (
    # Simulation result comparison with statistical analysis
    compare_simulation_results,
    
    # Benchmark comparison with accuracy validation
    compare_against_benchmark,
    
    # Cross-format result comparison with consistency analysis
    compare_cross_format_results,
    
    # Similarity metrics calculation between result sets
    calculate_result_similarity,
    
    # Result reproducibility validation
    validate_reproducibility,
    
    # Comprehensive comparison report generation
    generate_comparison_report,
    
    # Comprehensive result comparison classes
    ResultComparator,
    AlgorithmPerformanceComparator,
    CrossFormatCompatibilityComparator
)

# Define comprehensive package exports for external access
__all__ = [
    # Package metadata and configuration
    '__version__',
    '__author__',
    '__description__',
    'TEST_UTILITIES_VERSION',
    'SUPPORTED_TEST_CATEGORIES',
    'DEFAULT_TEST_CONFIGURATION',
    
    # Core test helper functions for fixture management and validation
    'create_test_fixture_path',
    'load_test_config',
    'assert_arrays_almost_equal',
    'assert_simulation_accuracy',
    'measure_performance',
    'create_mock_video_data',
    'validate_cross_format_compatibility',
    'setup_test_environment',
    'validate_batch_processing_results',
    'compare_algorithm_performance',
    'generate_test_report',
    'cache_test_data',
    
    # Test data validation and performance profiling classes
    'TestDataValidator',
    'PerformanceProfiler',
    
    # Validation metrics functions for statistical analysis and accuracy assessment
    'validate_trajectory_accuracy',
    'validate_performance_thresholds',
    'validate_statistical_significance',
    'validate_reproducibility_metrics',
    'calculate_algorithm_rankings',
    'load_benchmark_data',
    'generate_validation_report',
    
    # Validation metrics calculation and statistical analysis classes
    'ValidationMetricsCalculator',
    'StatisticalValidator',
    'BenchmarkComparator',
    
    # Performance monitoring functions for test execution analysis
    'create_test_performance_monitor',
    'monitor_test_execution_performance',
    'validate_test_performance_thresholds',
    'track_batch_test_performance',
    'collect_test_resource_metrics',
    'analyze_test_performance_trends',
    'generate_test_performance_report',
    
    # Performance monitoring and resource tracking classes
    'TestPerformanceMonitor',
    'ResourceTracker',
    'TestPerformanceContext',
    
    # Test data generation functions for synthetic data creation
    'generate_synthetic_plume_video',
    'create_crimaldi_format_data',
    'create_custom_format_data',
    'generate_batch_test_scenarios',
    'create_normalization_test_data',
    'generate_algorithm_validation_data',
    'create_performance_benchmark_data',
    'generate_cross_format_test_suite',
    
    # Test data generation and scenario building classes
    'SyntheticPlumeGenerator',
    'TestScenarioBuilder',
    
    # Result comparison functions for statistical analysis and validation
    'compare_simulation_results',
    'compare_against_benchmark',
    'compare_cross_format_results',
    'calculate_result_similarity',
    'validate_reproducibility',
    'generate_comparison_report',
    
    # Result comparison and analysis classes
    'ResultComparator',
    'AlgorithmPerformanceComparator',
    'CrossFormatCompatibilityComparator'
]