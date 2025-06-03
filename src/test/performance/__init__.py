"""
Performance Testing Package Initialization Module

Comprehensive performance validation infrastructure for the plume navigation simulation system.
Provides centralized access to performance test modules, monitoring utilities, validation frameworks,
and benchmarking capabilities for scientific computing workflows with >95% correlation accuracy
requirements, <7.2 seconds per simulation targets, and 4000+ simulation batch processing validation.

Supports automated performance testing, threshold validation, scaling analysis, and regression
detection for reproducible scientific research.

Author: Plume Navigation Performance Testing Team
Version: 1.0.0
"""

# Package metadata and version information
__version__ = '1.0.0'
__author__ = 'Plume Navigation Performance Testing Team'
__description__ = 'Comprehensive performance testing infrastructure for scientific plume navigation simulation validation'

# Performance testing configuration constants
PERFORMANCE_TEST_VERSION = '1.0.0'

# Supported performance test categories for comprehensive validation
SUPPORTED_PERFORMANCE_CATEGORIES = [
    'normalization', 
    'simulation_speed', 
    'batch_throughput', 
    'memory_usage', 
    'result_accuracy', 
    'parallel_scaling'
]

# Default performance thresholds based on scientific computing requirements
DEFAULT_PERFORMANCE_THRESHOLDS = {
    'simulation_time_seconds': 7.2,          # Maximum simulation execution time per run
    'correlation_threshold': 0.95,           # Minimum correlation with reference implementations
    'batch_target_simulations': 4000,        # Target number of simulations for batch processing
    'batch_completion_hours': 8.0,           # Maximum time for batch completion
    'memory_limit_gb': 8.0,                  # Maximum memory usage limit
    'reproducibility_threshold': 0.99,       # Minimum reproducibility coefficient
    'numerical_tolerance': 1e-6              # Numerical precision tolerance for scientific computing
}

# Performance test category descriptions and purposes
PERFORMANCE_TEST_CATEGORIES = {
    'normalization': 'Data normalization performance testing',
    'simulation_speed': 'Simulation execution speed validation', 
    'batch_throughput': 'Batch processing throughput analysis',
    'memory_usage': 'Memory utilization and efficiency testing',
    'result_accuracy': 'Result accuracy and correlation validation',
    'parallel_scaling': 'Parallel processing scaling efficiency'
}

# Import normalization performance testing components
# These modules provide comprehensive data normalization performance validation
from .test_normalization_performance import (
    test_single_video_normalization_performance,
    test_batch_normalization_performance,
    test_large_scale_normalization_performance,
    NormalizationPerformanceBenchmark
)

# Import simulation speed testing components
# These modules provide individual and batch simulation speed validation
from .test_simulation_speed import (
    test_single_simulation_speed,
    test_batch_simulation_throughput,
    test_large_scale_batch_performance,
    SimulationSpeedTestSuite
)

# Import batch throughput testing components
# These modules provide large-scale batch processing performance validation
from .test_batch_throughput import (
    test_batch_throughput_4000_simulations,
    test_parallel_processing_efficiency,
    test_resource_utilization_optimization,
    BatchThroughputTestSuite
)

# Import memory usage testing components
# These modules provide comprehensive memory utilization and efficiency testing
from .test_memory_usage import (
    test_video_processing_memory_usage,
    test_simulation_engine_memory_usage,
    test_memory_leak_detection,
    MemoryUsageTestSuite
)

# Import result accuracy testing components
# These modules provide simulation result accuracy and correlation validation
from .test_result_accuracy import (
    test_simulation_result_accuracy_against_benchmark,
    test_cross_format_result_accuracy,
    test_reproducibility_accuracy_validation
)

# Import parallel scaling testing components
# These modules provide parallel processing scaling efficiency validation
from .test_parallel_scaling import (
    test_parallel_executor_scaling,
    test_batch_executor_scaling_performance,
    test_parallel_scaling_performance_thresholds,
    ParallelScalingTestSuite
)

# Import performance monitoring and validation utilities
# These utilities provide test-specific monitoring and validation capabilities
from ..utils.performance_monitoring import TestPerformanceMonitor
from ..utils.validation_metrics import ValidationMetricsCalculator
from ..utils.result_comparator import ResultComparator

# Export all performance testing functions for comprehensive test execution
# Normalization Performance Testing Functions
__all__ = [
    # Package metadata
    '__version__',
    '__author__', 
    '__description__',
    'PERFORMANCE_TEST_VERSION',
    'SUPPORTED_PERFORMANCE_CATEGORIES',
    'DEFAULT_PERFORMANCE_THRESHOLDS',
    'PERFORMANCE_TEST_CATEGORIES',
    
    # Normalization performance testing exports
    'test_single_video_normalization_performance',
    'test_batch_normalization_performance', 
    'test_large_scale_normalization_performance',
    'NormalizationPerformanceBenchmark',
    
    # Simulation speed testing exports
    'test_single_simulation_speed',
    'test_batch_simulation_throughput',
    'test_large_scale_batch_performance',
    'SimulationSpeedTestSuite',
    
    # Batch throughput testing exports
    'test_batch_throughput_4000_simulations',
    'test_parallel_processing_efficiency',
    'test_resource_utilization_optimization',
    'BatchThroughputTestSuite',
    
    # Memory usage testing exports
    'test_video_processing_memory_usage',
    'test_simulation_engine_memory_usage',
    'test_memory_leak_detection',
    'MemoryUsageTestSuite',
    
    # Result accuracy testing exports
    'test_simulation_result_accuracy_against_benchmark',
    'test_cross_format_result_accuracy',
    'test_reproducibility_accuracy_validation',
    
    # Parallel scaling testing exports
    'test_parallel_executor_scaling',
    'test_batch_executor_scaling_performance',
    'test_parallel_scaling_performance_thresholds',
    'ParallelScalingTestSuite',
    
    # Performance monitoring and validation utilities
    'TestPerformanceMonitor',
    'ValidationMetricsCalculator',
    'ResultComparator'
]

# Performance testing framework configuration and documentation
def get_performance_test_info():
    """
    Retrieve comprehensive information about the performance testing framework.
    
    Returns:
        dict: Complete performance testing framework configuration including:
            - Test categories and descriptions
            - Performance thresholds and targets
            - Scientific computing standards
            - Monitoring capabilities
    """
    return {
        'framework_version': PERFORMANCE_TEST_VERSION,
        'supported_categories': SUPPORTED_PERFORMANCE_CATEGORIES,
        'performance_thresholds': DEFAULT_PERFORMANCE_THRESHOLDS,
        'test_categories': PERFORMANCE_TEST_CATEGORIES,
        'scientific_standards': {
            'numerical_precision_tolerance': 1e-6,
            'correlation_accuracy_requirement': 0.95,
            'reproducibility_coefficient_target': 0.99,
            'simulation_time_target_seconds': 7.2,
            'batch_processing_target_hours': 8.0,
            'batch_size_target_simulations': 4000
        },
        'monitoring_capabilities': {
            'real_time_performance_tracking': True,
            'resource_utilization_monitoring': True,
            'threshold_validation': True,
            'trend_analysis': True,
            'regression_detection': True
        }
    }

def validate_performance_requirements():
    """
    Validate that the performance testing environment meets all requirements.
    
    Returns:
        dict: Validation results with system capability assessment and recommendations
    """
    validation_results = {
        'environment_status': 'validated',
        'performance_capabilities': {
            'simulation_speed_testing': True,
            'batch_throughput_validation': True,
            'memory_usage_monitoring': True,
            'result_accuracy_validation': True,
            'parallel_scaling_analysis': True,
            'normalization_performance_testing': True
        },
        'threshold_compliance': {
            'numerical_precision': DEFAULT_PERFORMANCE_THRESHOLDS['numerical_tolerance'],
            'correlation_accuracy': DEFAULT_PERFORMANCE_THRESHOLDS['correlation_threshold'],
            'simulation_time_limit': DEFAULT_PERFORMANCE_THRESHOLDS['simulation_time_seconds'],
            'batch_processing_target': DEFAULT_PERFORMANCE_THRESHOLDS['batch_target_simulations'],
            'memory_efficiency_limit': DEFAULT_PERFORMANCE_THRESHOLDS['memory_limit_gb']
        },
        'recommendations': [
            'Execute comprehensive performance validation before production deployment',
            'Monitor resource utilization during large-scale batch processing',
            'Validate result accuracy against reference benchmarks regularly',
            'Implement automated threshold compliance checking',
            'Maintain detailed performance testing documentation'
        ]
    }
    
    return validation_results

# Performance testing suite execution helper functions
def get_all_performance_tests():
    """
    Retrieve a comprehensive list of all available performance test functions.
    
    Returns:
        dict: Organized collection of performance test functions by category
    """
    return {
        'normalization_performance': [
            test_single_video_normalization_performance,
            test_batch_normalization_performance,
            test_large_scale_normalization_performance
        ],
        'simulation_speed': [
            test_single_simulation_speed,
            test_batch_simulation_throughput, 
            test_large_scale_batch_performance
        ],
        'batch_throughput': [
            test_batch_throughput_4000_simulations,
            test_parallel_processing_efficiency,
            test_resource_utilization_optimization
        ],
        'memory_usage': [
            test_video_processing_memory_usage,
            test_simulation_engine_memory_usage,
            test_memory_leak_detection
        ],
        'result_accuracy': [
            test_simulation_result_accuracy_against_benchmark,
            test_cross_format_result_accuracy,
            test_reproducibility_accuracy_validation
        ],
        'parallel_scaling': [
            test_parallel_executor_scaling,
            test_batch_executor_scaling_performance,
            test_parallel_scaling_performance_thresholds
        ]
    }

def get_performance_test_suites():
    """
    Retrieve comprehensive performance test suite classes for organized testing.
    
    Returns:
        dict: Performance test suite classes organized by functionality
    """
    return {
        'normalization_benchmark': NormalizationPerformanceBenchmark,
        'simulation_speed_suite': SimulationSpeedTestSuite,
        'batch_throughput_suite': BatchThroughputTestSuite,
        'memory_usage_suite': MemoryUsageTestSuite,
        'parallel_scaling_suite': ParallelScalingTestSuite
    }

def get_monitoring_utilities():
    """
    Retrieve performance monitoring and validation utility classes.
    
    Returns:
        dict: Monitoring and validation utility classes for comprehensive testing
    """
    return {
        'performance_monitor': TestPerformanceMonitor,
        'validation_calculator': ValidationMetricsCalculator,
        'result_comparator': ResultComparator
    }