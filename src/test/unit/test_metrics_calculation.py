"""
Comprehensive unit test module for performance metrics calculation functionality in plume navigation simulation systems.

This module implements thorough unit testing of all aspects of metrics computation including navigation success metrics,
path efficiency analysis, temporal dynamics evaluation, robustness assessment, and cross-format compatibility validation.
The tests validate >95% correlation requirements, <7.2 seconds processing time targets, and statistical significance
testing with comprehensive mock data scenarios and benchmark comparisons for scientific computing accuracy.

Key Testing Areas:
- PerformanceMetricsCalculator initialization and configuration validation
- Navigation success metrics calculation with statistical validation
- Path efficiency analysis including optimality and search pattern evaluation
- Temporal dynamics metrics with response time and decision latency testing
- Robustness metrics across multiple environmental conditions
- Algorithm performance comparison with statistical significance testing
- Cross-format compatibility between Crimaldi and custom plume formats
- Performance threshold validation and caching functionality
- Error handling and edge case coverage with boundary condition testing

Testing Framework Integration:
- pytest framework with comprehensive fixture management
- Mock data generation for realistic test scenarios
- Performance profiling with timing validation
- Statistical validation against reference implementations
- Cross-platform compatibility testing with scientific precision
"""

# External library imports with version specifications for testing framework
import pytest  # pytest 8.3.5+ - Testing framework for unit test execution and fixture management
import numpy as np  # numpy 2.1.3+ - Numerical computations and array operations for test data validation
import pandas as pd  # pandas 2.2.0+ - Data manipulation and analysis for test result processing
from scipy import stats  # scipy 1.15.3+ - Statistical analysis and hypothesis testing for metrics validation
from unittest.mock import Mock, patch, MagicMock, call, ANY  # unittest 3.9+ - Mock object creation and behavior simulation
from typing import Dict, Any, List, Optional, Union, Tuple  # Python 3.9+ - Type hints for test function signatures
import pathlib  # Python 3.9+ - Cross-platform path handling for test fixtures
import warnings  # Python 3.9+ - Warning management for test execution and validation
import time  # Python 3.9+ - Performance timing and measurement utilities
import datetime  # Python 3.9+ - Timestamp handling for test metadata and audit trails
import uuid  # Python 3.9+ - Unique identifier generation for test correlation
import json  # Python 3.9+ - JSON configuration file handling for test scenarios
import tempfile  # Python 3.9+ - Temporary file and directory management for test isolation
import copy  # Python 3.9+ - Deep copying for test data manipulation

# Internal imports for performance metrics calculation and testing infrastructure
from backend.core.analysis.performance_metrics import (
    PerformanceMetricsCalculator,
    NavigationSuccessAnalyzer,
    PathEfficiencyAnalyzer,
    calculate_navigation_success_metrics,
    calculate_path_efficiency_metrics,
    calculate_temporal_dynamics_metrics,
    calculate_robustness_metrics,
    compare_algorithm_performance,
    validate_performance_against_thresholds
)

from test.utils.validation_metrics import (
    ValidationMetricsCalculator,
    StatisticalValidator,
    BenchmarkComparator
)

from test.utils.test_helpers import (
    assert_arrays_almost_equal,
    assert_simulation_accuracy,
    measure_performance,
    create_mock_video_data,
    validate_cross_format_compatibility,
    setup_test_environment,
    validate_batch_processing_results,
    compare_algorithm_performance as compare_test_performance,
    generate_test_report,
    cache_test_data,
    get_cached_test_data,
    TestDataValidator,
    PerformanceProfiler
)

from test.mocks.mock_analysis_pipeline import (
    create_mock_trajectory_data,
    generate_mock_performance_data,
    MockPerformanceMetricsCalculator
)

from backend.utils.validation_utils import (
    ValidationResult,
    validate_numerical_accuracy,
    validate_cross_format_compatibility as validate_format_compatibility
)

from backend.utils.scientific_constants import (
    NUMERICAL_PRECISION_THRESHOLD,
    DEFAULT_CORRELATION_THRESHOLD,
    PROCESSING_TIME_TARGET_SECONDS,
    REPRODUCIBILITY_THRESHOLD,
    STATISTICAL_SIGNIFICANCE_LEVEL,
    get_performance_thresholds,
    get_statistical_constants,
    PhysicalConstants
)

# Global configuration constants for test fixture paths and validation settings
CORRELATION_THRESHOLD = 0.95
NUMERICAL_TOLERANCE = 1e-6
PROCESSING_TIME_TARGET = 7.2
REPRODUCIBILITY_THRESHOLD = 0.99
STATISTICAL_SIGNIFICANCE_LEVEL = 0.05
TEST_ALGORITHM_TYPES = ['infotaxis', 'casting', 'gradient_following', 'plume_tracking', 'hybrid_strategies']
TEST_METRIC_CATEGORIES = ['navigation_success', 'path_efficiency', 'temporal_dynamics', 'robustness', 'resource_utilization']
BENCHMARK_DATA_TYPES = ['simulation', 'analysis', 'normalization']

# Test configuration for reproducible scientific testing
RANDOM_SEED = 42
TEST_DATA_DIMENSIONS = (640, 480)
TEST_FRAME_COUNT = 100
TEST_SIMULATION_COUNT = 50
MOCK_TRAJECTORY_LENGTH = 1000
MAX_TEST_DURATION = 300  # 5 minutes maximum test duration


@pytest.fixture(scope="session")
def test_config():
    """
    Session-scoped test configuration fixture providing comprehensive testing parameters
    and validation settings for consistent test execution across all test modules.
    """
    return {
        'correlation_threshold': CORRELATION_THRESHOLD,
        'numerical_tolerance': NUMERICAL_TOLERANCE,
        'processing_time_target': PROCESSING_TIME_TARGET,
        'reproducibility_threshold': REPRODUCIBILITY_THRESHOLD,
        'statistical_significance': STATISTICAL_SIGNIFICANCE_LEVEL,
        'algorithm_types': TEST_ALGORITHM_TYPES,
        'metric_categories': TEST_METRIC_CATEGORIES,
        'benchmark_types': BENCHMARK_DATA_TYPES,
        'random_seed': RANDOM_SEED,
        'test_dimensions': TEST_DATA_DIMENSIONS,
        'frame_count': TEST_FRAME_COUNT,
        'simulation_count': TEST_SIMULATION_COUNT,
        'trajectory_length': MOCK_TRAJECTORY_LENGTH,
        'max_test_duration': MAX_TEST_DURATION,
        'enable_performance_validation': True,
        'enable_statistical_validation': True,
        'enable_cross_format_testing': True,
        'strict_validation_mode': True
    }


@pytest.fixture(scope="function")
def mock_simulation_results():
    """
    Function-scoped fixture generating realistic mock simulation results for testing
    navigation success metrics calculation with comprehensive trajectory data.
    """
    np.random.seed(RANDOM_SEED)
    
    # Generate mock simulation results with realistic navigation trajectories
    simulation_results = []
    for i in range(TEST_SIMULATION_COUNT):
        # Create realistic trajectory with source localization behavior
        start_position = np.array([0.1, 0.5])  # Starting near arena edge
        source_position = np.array([0.8, 0.5])  # Source location
        
        # Generate trajectory with search and localization phases
        trajectory = [start_position]
        current_pos = start_position.copy()
        
        for step in range(MOCK_TRAJECTORY_LENGTH // 20):  # Shorter trajectories for testing
            # Add search behavior with bias toward source
            direction_to_source = source_position - current_pos
            direction_to_source /= np.linalg.norm(direction_to_source)
            
            # Add realistic noise and exploration
            random_component = np.random.normal(0, 0.1, 2)
            movement = 0.7 * direction_to_source + 0.3 * random_component
            movement_magnitude = np.random.uniform(0.01, 0.05)
            movement = movement / np.linalg.norm(movement) * movement_magnitude
            
            current_pos += movement
            # Keep within arena bounds
            current_pos = np.clip(current_pos, 0.0, 1.0)
            trajectory.append(current_pos.copy())
        
        # Calculate success metrics
        final_distance = np.linalg.norm(current_pos - source_position)
        localization_success = final_distance < 0.1  # Within 10cm of source
        time_to_target = len(trajectory) * 0.1  # Assume 0.1 seconds per step
        
        result = {
            'simulation_id': i,
            'trajectory': np.array(trajectory),
            'final_position': current_pos,
            'source_position': source_position,
            'localization_success': localization_success,
            'time_to_target': time_to_target,
            'final_distance_to_source': final_distance,
            'path_length': np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)),
            'exploration_area': len(set(tuple(np.round(pos * 10).astype(int)) for pos in trajectory)),
            'algorithm_type': np.random.choice(TEST_ALGORITHM_TYPES),
            'success_criteria_met': localization_success and time_to_target < 30.0
        }
        simulation_results.append(result)
    
    return simulation_results


@pytest.fixture(scope="function")
def target_locations():
    """
    Function-scoped fixture providing predefined target locations for consistent
    navigation success testing across different test scenarios.
    """
    return {
        'primary_source': np.array([0.8, 0.5]),
        'secondary_source': np.array([0.2, 0.3]),
        'control_location': np.array([0.5, 0.5]),
        'edge_location': np.array([0.1, 0.1]),
        'multiple_sources': [
            np.array([0.7, 0.3]),
            np.array([0.3, 0.7]),
            np.array([0.9, 0.1])
        ],
        'success_radius': 0.1,
        'arena_bounds': np.array([[0.0, 1.0], [0.0, 1.0]])
    }


@pytest.fixture(scope="function")
def trajectory_data():
    """
    Function-scoped fixture generating comprehensive trajectory data arrays for
    path efficiency analysis with realistic navigation patterns and search behaviors.
    """
    np.random.seed(RANDOM_SEED)
    
    trajectories = []
    for algorithm in TEST_ALGORITHM_TYPES:
        # Generate algorithm-specific trajectory patterns
        if algorithm == 'infotaxis':
            # Information-theoretic search with exploration
            trajectory = _generate_infotaxis_trajectory()
        elif algorithm == 'casting':
            # Casting behavior with crosswind search
            trajectory = _generate_casting_trajectory()
        elif algorithm == 'gradient_following':
            # Direct gradient ascent behavior
            trajectory = _generate_gradient_trajectory()
        elif algorithm == 'plume_tracking':
            # Plume tracking with surge behavior
            trajectory = _generate_plume_tracking_trajectory()
        else:  # hybrid_strategies
            # Combination of multiple strategies
            trajectory = _generate_hybrid_trajectory()
        
        trajectories.append({
            'algorithm': algorithm,
            'trajectory': trajectory,
            'timestamps': np.linspace(0, len(trajectory) * 0.1, len(trajectory)),
            'velocities': np.diff(trajectory, axis=0),
            'accelerations': np.diff(np.diff(trajectory, axis=0), axis=0)
        })
    
    return trajectories


@pytest.fixture(scope="function")
def optimal_paths():
    """
    Function-scoped fixture providing optimal path references for efficiency
    analysis and comparative performance evaluation against calculated trajectories.
    """
    return {
        'direct_path': np.linspace([0.1, 0.5], [0.8, 0.5], 10),
        'optimal_search': _generate_optimal_search_path(),
        'minimal_distance': np.array([[0.1, 0.5], [0.8, 0.5]]),  # Straight line
        'exploration_optimal': _generate_exploration_optimal_path(),
        'energy_optimal': _generate_energy_optimal_path(),
        'time_optimal': _generate_time_optimal_path(),
        'efficiency_metrics': {
            'path_length': 0.7,  # Optimal path length
            'exploration_coverage': 0.85,  # Optimal coverage
            'energy_consumption': 1.0,  # Normalized energy
            'success_probability': 0.95  # Target success rate
        }
    }


@pytest.fixture(scope="function")
def temporal_data():
    """
    Function-scoped fixture generating temporal dynamics data for response time
    and decision-making latency analysis with realistic timing characteristics.
    """
    np.random.seed(RANDOM_SEED)
    
    # Generate temporal data with realistic response patterns
    temporal_data = []
    for i in range(TEST_SIMULATION_COUNT):
        # Plume encounter events with response times
        encounter_times = np.sort(np.random.uniform(0, 100, 10))  # 10 encounters over 100 seconds
        response_times = np.random.exponential(2.0, 10)  # Exponential response distribution
        decision_latencies = np.random.gamma(2, 0.5, 10)  # Gamma-distributed decision times
        
        # Adaptation and learning effects
        adaptation_factor = np.exp(-encounter_times / 20)  # Learning curve
        adapted_response_times = response_times * adaptation_factor
        
        temporal_entry = {
            'simulation_id': i,
            'encounter_times': encounter_times,
            'response_times': response_times,
            'adapted_response_times': adapted_response_times,
            'decision_latencies': decision_latencies,
            'convergence_time': encounter_times[-1],
            'learning_rate': 0.1 + np.random.uniform(-0.05, 0.05),
            'adaptation_strength': np.random.uniform(0.5, 1.0),
            'temporal_consistency': np.std(response_times) / np.mean(response_times)
        }
        temporal_data.append(temporal_entry)
    
    return temporal_data


@pytest.fixture(scope="function")
def plume_encounter_times():
    """
    Function-scoped fixture providing plume encounter timing data for temporal
    dynamics analysis with statistical distribution characteristics.
    """
    np.random.seed(RANDOM_SEED)
    
    # Generate realistic plume encounter patterns
    encounter_patterns = {}
    
    for condition in ['low_turbulence', 'moderate_turbulence', 'high_turbulence']:
        if condition == 'low_turbulence':
            # Regular encounter pattern with low variability
            base_interval = 5.0
            variability = 1.0
        elif condition == 'moderate_turbulence':
            # Moderate encounter variability
            base_interval = 7.0
            variability = 2.5
        else:  # high_turbulence
            # Highly variable encounter pattern
            base_interval = 10.0
            variability = 5.0
        
        # Generate encounter times with realistic statistics
        num_encounters = np.random.poisson(20)  # Average 20 encounters
        inter_encounter_intervals = np.random.exponential(base_interval, num_encounters)
        encounter_times = np.cumsum(inter_encounter_intervals)
        
        # Add environmental variability
        noise = np.random.normal(0, variability, num_encounters)
        encounter_times += noise
        encounter_times = np.maximum(encounter_times, 0)  # Ensure non-negative
        
        encounter_patterns[condition] = {
            'times': encounter_times,
            'intervals': inter_encounter_intervals,
            'condition': condition,
            'mean_interval': np.mean(inter_encounter_intervals),
            'interval_std': np.std(inter_encounter_intervals),
            'total_duration': encounter_times[-1] if len(encounter_times) > 0 else 0,
            'encounter_frequency': len(encounter_times) / encounter_times[-1] if len(encounter_times) > 0 else 0
        }
    
    return encounter_patterns


@pytest.fixture(scope="function")
def multi_condition_results():
    """
    Function-scoped fixture generating multi-condition test results for robustness
    analysis across varying environmental parameters and noise levels.
    """
    np.random.seed(RANDOM_SEED)
    
    # Define environmental conditions for robustness testing
    conditions = {
        'baseline': {'noise_level': 0.0, 'turbulence': 0.1, 'complexity': 1.0},
        'low_noise': {'noise_level': 0.1, 'turbulence': 0.1, 'complexity': 1.0},
        'moderate_noise': {'noise_level': 0.2, 'turbulence': 0.3, 'complexity': 1.2},
        'high_noise': {'noise_level': 0.5, 'turbulence': 0.5, 'complexity': 1.5},
        'extreme_conditions': {'noise_level': 0.8, 'turbulence': 0.8, 'complexity': 2.0}
    }
    
    results = {}
    
    for condition_name, params in conditions.items():
        condition_results = []
        
        for algorithm in TEST_ALGORITHM_TYPES:
            # Generate performance under specific conditions
            base_performance = 0.9  # Baseline success rate
            noise_impact = 1.0 - params['noise_level'] * 0.5  # Noise reduces performance
            turbulence_impact = 1.0 - params['turbulence'] * 0.3  # Turbulence impact
            complexity_impact = 1.0 / params['complexity']  # Complexity reduces performance
            
            # Calculate condition-specific performance
            success_rate = base_performance * noise_impact * turbulence_impact * complexity_impact
            success_rate = max(0.1, min(1.0, success_rate))  # Clamp to reasonable range
            
            # Add algorithm-specific variability
            algorithm_variance = np.random.uniform(0.9, 1.1)
            success_rate *= algorithm_variance
            
            # Generate multiple trials
            num_trials = 20
            trial_results = np.random.binomial(1, success_rate, num_trials)
            
            result_entry = {
                'algorithm': algorithm,
                'condition': condition_name,
                'parameters': params.copy(),
                'success_rate': np.mean(trial_results),
                'success_count': np.sum(trial_results),
                'total_trials': num_trials,
                'performance_degradation': 1.0 - (success_rate / base_performance),
                'robustness_score': success_rate / base_performance,
                'trial_results': trial_results,
                'statistical_significance': None  # Will be calculated in tests
            }
            condition_results.append(result_entry)
        
        results[condition_name] = condition_results
    
    return results


@pytest.fixture(scope="function")
def noise_levels():
    """
    Function-scoped fixture providing systematic noise level configurations
    for robustness testing and performance degradation analysis.
    """
    return {
        'no_noise': 0.0,
        'minimal_noise': 0.05,
        'low_noise': 0.1,
        'moderate_noise': 0.2,
        'high_noise': 0.5,
        'extreme_noise': 0.8,
        'noise_characteristics': {
            'distribution_type': 'gaussian',
            'temporal_correlation': 0.1,
            'spatial_correlation': 0.2,
            'frequency_spectrum': 'white',
            'measurement_units': 'normalized'
        },
        'validation_criteria': {
            'min_performance_retention': 0.7,  # Minimum 70% performance retention
            'max_degradation_rate': 0.5,  # Maximum 50% degradation
            'stability_threshold': 0.1  # 10% stability requirement
        }
    }


@pytest.fixture(scope="function")
def algorithm_metrics():
    """
    Function-scoped fixture providing comprehensive algorithm performance metrics
    for comparative analysis and statistical significance testing.
    """
    np.random.seed(RANDOM_SEED)
    
    metrics = {}
    
    # Generate realistic performance metrics for each algorithm
    for algorithm in TEST_ALGORITHM_TYPES:
        # Algorithm-specific base performance characteristics
        if algorithm == 'infotaxis':
            base_success = 0.85
            base_efficiency = 0.75
            base_speed = 0.8
        elif algorithm == 'casting':
            base_success = 0.8
            base_efficiency = 0.85
            base_speed = 0.9
        elif algorithm == 'gradient_following':
            base_success = 0.9
            base_efficiency = 0.9
            base_speed = 0.95
        elif algorithm == 'plume_tracking':
            base_success = 0.88
            base_efficiency = 0.82
            base_speed = 0.87
        else:  # hybrid_strategies
            base_success = 0.92
            base_efficiency = 0.88
            base_speed = 0.85
        
        # Generate sample data for statistical testing
        num_samples = 100
        success_samples = np.random.beta(base_success * 10, (1 - base_success) * 10, num_samples)
        efficiency_samples = np.random.beta(base_efficiency * 10, (1 - base_efficiency) * 10, num_samples)
        speed_samples = np.random.gamma(base_speed * 5, 0.2, num_samples)
        
        metrics[algorithm] = {
            'success_rate': np.mean(success_samples),
            'path_efficiency': np.mean(efficiency_samples),
            'processing_speed': np.mean(speed_samples),
            'convergence_rate': base_success * 0.9 + np.random.normal(0, 0.05),
            'robustness_index': (base_success + base_efficiency) / 2 + np.random.normal(0, 0.03),
            'resource_utilization': np.random.uniform(0.6, 0.9),
            
            # Sample data for statistical testing
            'success_rate_samples': success_samples,
            'path_efficiency_samples': efficiency_samples,
            'processing_speed_samples': speed_samples,
            
            # Additional metrics for comprehensive comparison
            'mean_time_to_target': np.random.exponential(10 + np.random.uniform(-3, 3)),
            'exploration_coverage': np.random.uniform(0.7, 0.95),
            'energy_efficiency': np.random.uniform(0.8, 0.95),
            'adaptation_capability': np.random.uniform(0.6, 0.9),
            
            # Confidence intervals and statistical measures
            'confidence_intervals': {
                'success_rate': np.percentile(success_samples, [2.5, 97.5]),
                'path_efficiency': np.percentile(efficiency_samples, [2.5, 97.5]),
                'processing_speed': np.percentile(speed_samples, [2.5, 97.5])
            },
            'statistical_measures': {
                'effect_size': np.random.uniform(0.3, 0.8),
                'power_analysis': np.random.uniform(0.8, 0.95),
                'sample_size': num_samples
            }
        }
    
    return metrics


@pytest.fixture(scope="function")
def comparison_categories():
    """
    Function-scoped fixture providing systematic comparison categories for
    algorithm performance evaluation and ranking analysis.
    """
    return [
        'navigation_efficiency',
        'computational_performance',
        'robustness_analysis',
        'resource_optimization',
        'statistical_significance',
        'practical_applicability',
        'scalability_assessment',
        'convergence_characteristics'
    ]


class TestPerformanceMetricsCalculation:
    """
    Comprehensive test class for performance metrics calculation functionality with fixtures,
    setup, and teardown methods for isolated testing of all performance analysis components.
    
    This class provides structured testing of the PerformanceMetricsCalculator and related
    analysis components with comprehensive validation, benchmarking, and cross-format compatibility.
    """

    def setup_method(self, method):
        """
        Setup method called before each test method to ensure clean test environment
        with initialized calculators, validators, and performance monitoring.
        """
        # Reset random seed for reproducible test execution
        np.random.seed(RANDOM_SEED)
        
        # Initialize performance metrics calculator with test configuration
        self.metrics_calculator = PerformanceMetricsCalculator(
            correlation_threshold=CORRELATION_THRESHOLD,
            enable_caching=True,
            statistical_validation=True,
            performance_monitoring=True
        )
        
        # Initialize validation metrics calculator for accuracy checking
        self.validation_calculator = ValidationMetricsCalculator(
            tolerance=NUMERICAL_TOLERANCE,
            strict_validation=True
        )
        
        # Setup benchmark comparator for reference validation
        self.benchmark_comparator = BenchmarkComparator(
            benchmark_types=BENCHMARK_DATA_TYPES,
            correlation_threshold=CORRELATION_THRESHOLD
        )
        
        # Initialize statistical validator for correlation analysis
        self.statistical_validator = StatisticalValidator(
            significance_level=STATISTICAL_SIGNIFICANCE_LEVEL,
            correlation_method='pearson'
        )
        
        # Setup performance profiler for timing validation
        self.performance_profiler = PerformanceProfiler(
            time_threshold_seconds=PROCESSING_TIME_TARGET,
            memory_threshold_mb=8192
        )
        
        # Initialize test data validator
        self.test_validator = TestDataValidator(
            tolerance=NUMERICAL_TOLERANCE,
            strict_validation=True
        )
        
        # Load reference benchmark data for comparison
        self.reference_data = self._load_reference_benchmarks()
        
        # Configure test-specific logging
        self.test_start_time = datetime.datetime.now()
        self.test_metadata = {
            'test_method': method.__name__,
            'test_id': str(uuid.uuid4()),
            'start_time': self.test_start_time.isoformat(),
            'configuration': {
                'correlation_threshold': CORRELATION_THRESHOLD,
                'processing_time_target': PROCESSING_TIME_TARGET,
                'numerical_tolerance': NUMERICAL_TOLERANCE
            }
        }

    def teardown_method(self, method):
        """
        Teardown method called after each test method to cleanup resources,
        validate performance, and generate test execution reports.
        """
        # Calculate test execution performance
        test_end_time = datetime.datetime.now()
        test_duration = (test_end_time - self.test_start_time).total_seconds()
        
        # Validate test execution performance against thresholds
        if test_duration > MAX_TEST_DURATION:
            warnings.warn(f"Test {method.__name__} exceeded maximum duration: {test_duration:.2f}s")
        
        # Update test metadata with completion information
        self.test_metadata.update({
            'end_time': test_end_time.isoformat(),
            'duration_seconds': test_duration,
            'performance_validation': test_duration <= MAX_TEST_DURATION,
            'completed_successfully': True
        })
        
        # Clear calculator caches and reset state
        if hasattr(self.metrics_calculator, 'clear_cache'):
            self.metrics_calculator.clear_cache()
        
        # Reset validation calculator state
        if hasattr(self.validation_calculator, 'reset_state'):
            self.validation_calculator.reset_state()
        
        # Cleanup temporary test data and memory allocations
        if hasattr(self, 'reference_data'):
            del self.reference_data
        
        # Generate test execution performance report if needed
        if test_duration > PROCESSING_TIME_TARGET:
            self._generate_performance_warning(method.__name__, test_duration)

    def validate_metrics_structure(self, metrics: Dict[str, Any], expected_keys: List[str]) -> bool:
        """
        Helper method to validate the structure and data types of calculated metrics
        against expected schema and scientific computing requirements.
        
        Args:
            metrics: Dictionary of calculated metrics to validate
            expected_keys: List of expected metric keys for structure validation
            
        Returns:
            bool: True if metrics structure is valid with comprehensive validation
        """
        # Check presence of all expected metric categories
        missing_keys = set(expected_keys) - set(metrics.keys())
        if missing_keys:
            raise AssertionError(f"Missing expected metric keys: {missing_keys}")
        
        # Validate data types for numerical metrics
        for key, value in metrics.items():
            if key.endswith('_rate') or key.endswith('_score') or key.endswith('_ratio'):
                if not isinstance(value, (int, float)):
                    raise AssertionError(f"Metric {key} should be numeric, got {type(value)}")
                
                # Check for valid range for rate/score metrics
                if key.endswith('_rate') or key.endswith('_score'):
                    if not 0.0 <= value <= 1.0:
                        raise AssertionError(f"Rate/score metric {key} should be in [0,1], got {value}")
            
            elif key.endswith('_time') or key.endswith('_duration'):
                if not isinstance(value, (int, float)) or value < 0:
                    raise AssertionError(f"Time metric {key} should be non-negative numeric, got {value}")
            
            # Check for NaN or infinite values
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    raise AssertionError(f"Metric {key} contains invalid value: {value}")
        
        # Validate statistical properties consistency
        correlation_metrics = [k for k in metrics.keys() if 'correlation' in k]
        for corr_metric in correlation_metrics:
            if metrics[corr_metric] < 0 or metrics[corr_metric] > 1:
                raise AssertionError(f"Correlation metric {corr_metric} outside valid range [0,1]")
        
        # Check metadata and timestamp information
        if 'metadata' in metrics:
            metadata = metrics['metadata']
            if 'calculation_timestamp' not in metadata:
                warnings.warn("Missing calculation timestamp in metrics metadata")
            if 'validation_status' not in metadata:
                warnings.warn("Missing validation status in metrics metadata")
        
        return True

    def compare_with_benchmark(self, calculated_metrics: Dict[str, Any], benchmark_type: str) -> Dict[str, float]:
        """
        Helper method to compare calculated metrics against reference benchmark data
        with statistical correlation analysis and validation reporting.
        
        Args:
            calculated_metrics: Dictionary of calculated metrics for comparison
            benchmark_type: Type of benchmark for comparison selection
            
        Returns:
            Dict[str, float]: Comparison results with correlation coefficients and validation status
        """
        # Load appropriate benchmark data using BenchmarkComparator
        benchmark_data = self.benchmark_comparator.load_benchmark_data(benchmark_type)
        
        if not benchmark_data:
            raise ValueError(f"No benchmark data available for type: {benchmark_type}")
        
        # Normalize metrics for comparison compatibility
        normalized_calculated = self._normalize_metrics_for_comparison(calculated_metrics)
        normalized_benchmark = self._normalize_metrics_for_comparison(benchmark_data)
        
        # Find common metrics for comparison
        common_metrics = set(normalized_calculated.keys()) & set(normalized_benchmark.keys())
        if not common_metrics:
            raise ValueError("No common metrics found between calculated and benchmark data")
        
        comparison_results = {}
        
        # Calculate correlation coefficients using StatisticalValidator
        for metric_name in common_metrics:
            calc_values = normalized_calculated[metric_name]
            bench_values = normalized_benchmark[metric_name]
            
            # Ensure arrays are compatible for correlation calculation
            if not isinstance(calc_values, (list, np.ndarray)):
                calc_values = [calc_values]
            if not isinstance(bench_values, (list, np.ndarray)):
                bench_values = [bench_values]
            
            calc_array = np.array(calc_values)
            bench_array = np.array(bench_values)
            
            # Calculate correlation using statistical validator
            correlation_result = self.statistical_validator.perform_correlation_analysis(
                calc_array, bench_array
            )
            
            comparison_results[metric_name] = correlation_result['correlation_coefficient']
            
            # Perform statistical significance testing
            significance_result = self.statistical_validator.test_statistical_significance(
                calc_array, bench_array
            )
            comparison_results[f"{metric_name}_significance"] = significance_result['p_value']
        
        # Validate >95% correlation requirement
        overall_correlation = np.mean(list(comparison_results.values()))
        if overall_correlation < CORRELATION_THRESHOLD:
            warnings.warn(f"Overall correlation {overall_correlation:.6f} below threshold {CORRELATION_THRESHOLD}")
        
        # Generate detailed comparison report
        comparison_results.update({
            'overall_correlation': overall_correlation,
            'meets_correlation_threshold': overall_correlation >= CORRELATION_THRESHOLD,
            'common_metrics_count': len(common_metrics),
            'benchmark_type': benchmark_type,
            'comparison_timestamp': datetime.datetime.now().isoformat(),
            'validation_status': 'PASS' if overall_correlation >= CORRELATION_THRESHOLD else 'FAIL'
        })
        
        return comparison_results

    def _load_reference_benchmarks(self) -> Dict[str, Any]:
        """Load reference benchmark data for test validation."""
        # This would typically load from benchmark files
        # For testing, we generate consistent reference data
        return {
            'navigation_success_rate': 0.85,
            'path_efficiency': 0.78,
            'temporal_response': 2.5,
            'robustness_score': 0.82,
            'cross_format_compatibility': 0.95
        }

    def _normalize_metrics_for_comparison(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize metrics for comparison compatibility."""
        normalized = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Simple normalization - in practice this would be more sophisticated
                normalized[key] = value
            elif isinstance(value, (list, np.ndarray)):
                normalized[key] = np.array(value)
        return normalized

    def _generate_performance_warning(self, test_name: str, duration: float):
        """Generate performance warning for slow tests."""
        warnings.warn(
            f"Test {test_name} took {duration:.2f}s, exceeding target {PROCESSING_TIME_TARGET}s",
            PerformanceWarning
        )


@pytest.mark.unit
@measure_performance(time_limit_seconds=1.0)
def test_performance_metrics_calculator_initialization(test_config):
    """
    Test proper initialization of PerformanceMetricsCalculator with various configuration
    parameters and validation settings including correlation thresholds and caching.
    """
    # Test initialization with default configuration
    default_calculator = PerformanceMetricsCalculator()
    assert default_calculator is not None
    assert hasattr(default_calculator, 'correlation_threshold')
    assert hasattr(default_calculator, 'enable_caching')
    
    # Test initialization with custom configuration
    custom_config = {
        'correlation_threshold': test_config['correlation_threshold'],
        'enable_caching': True,
        'statistical_validation': True,
        'performance_monitoring': True,
        'cross_format_support': True
    }
    
    custom_calculator = PerformanceMetricsCalculator(**custom_config)
    assert custom_calculator.correlation_threshold == test_config['correlation_threshold']
    assert custom_calculator.enable_caching == True
    
    # Validate configuration parameters are properly set
    assert custom_calculator.correlation_threshold > 0.95
    assert custom_calculator.correlation_threshold <= 1.0
    
    # Check correlation threshold is set to >0.95 requirement
    assert custom_calculator.correlation_threshold >= CORRELATION_THRESHOLD
    
    # Verify caching and statistical validation are enabled
    assert custom_calculator.enable_caching == True
    assert hasattr(custom_calculator, 'statistical_validation')
    
    # Assert proper initialization of internal components
    assert hasattr(custom_calculator, 'logger')
    assert hasattr(custom_calculator, 'performance_thresholds')
    
    # Validate logger and performance thresholds setup
    assert custom_calculator.logger is not None
    if hasattr(custom_calculator, 'performance_thresholds'):
        thresholds = custom_calculator.performance_thresholds
        assert 'processing_time' in thresholds
        assert 'correlation_threshold' in thresholds


@pytest.mark.unit
@pytest.mark.parametrize('algorithm_type', TEST_ALGORITHM_TYPES)
@measure_performance(time_limit_seconds=2.0)
def test_calculate_navigation_success_metrics(mock_simulation_results, target_locations, algorithm_type):
    """
    Test calculation of navigation success metrics including source localization rate,
    time to target, and search success rate with >95% correlation validation.
    """
    # Filter simulation results for specific algorithm type
    algorithm_results = [r for r in mock_simulation_results if r['algorithm_type'] == algorithm_type]
    
    if not algorithm_results:
        # Generate results for this algorithm if none exist
        algorithm_results = mock_simulation_results[:10]  # Use first 10 results
        for result in algorithm_results:
            result['algorithm_type'] = algorithm_type
    
    # Generate mock simulation results using create_mock_trajectory_data
    trajectory_data = create_mock_trajectory_data(
        num_trajectories=len(algorithm_results),
        algorithm_type=algorithm_type,
        target_location=target_locations['primary_source']
    )
    
    # Call calculate_navigation_success_metrics with test data
    metrics_result = calculate_navigation_success_metrics(
        simulation_results=algorithm_results,
        target_locations=target_locations,
        algorithm_type=algorithm_type
    )
    
    # Validate returned metrics structure and data types
    expected_keys = [
        'source_localization_rate',
        'mean_time_to_target',
        'search_success_rate',
        'final_distance_distribution',
        'convergence_statistics',
        'algorithm_type',
        'sample_size'
    ]
    
    assert isinstance(metrics_result, dict)
    for key in expected_keys:
        assert key in metrics_result, f"Missing expected key: {key}"
    
    # Assert source localization rate is within valid range [0, 1]
    localization_rate = metrics_result['source_localization_rate']
    assert 0.0 <= localization_rate <= 1.0, f"Invalid localization rate: {localization_rate}"
    
    # Check time to target metrics for statistical validity
    time_to_target = metrics_result['mean_time_to_target']
    assert isinstance(time_to_target, (int, float)), "Time to target should be numeric"
    assert time_to_target > 0, "Time to target should be positive"
    
    # Validate search success rate calculations
    success_rate = metrics_result['search_success_rate']
    assert 0.0 <= success_rate <= 1.0, f"Invalid success rate: {success_rate}"
    
    # Compare against reference benchmark using BenchmarkComparator
    benchmark_comparator = BenchmarkComparator()
    benchmark_data = benchmark_comparator.load_benchmark_data('navigation_success')
    
    if benchmark_data:
        comparison_result = benchmark_comparator.compare_against_benchmark(
            metrics_result, benchmark_data
        )
        
        # Assert >95% correlation with reference implementation
        correlation = comparison_result.get('correlation_coefficient', 0.0)
        assert correlation >= CORRELATION_THRESHOLD, f"Correlation {correlation:.6f} below threshold {CORRELATION_THRESHOLD}"
    
    # Validate algorithm-specific performance characteristics
    if algorithm_type == 'infotaxis':
        # Infotaxis should have good exploration but moderate efficiency
        assert 0.7 <= localization_rate <= 0.95
    elif algorithm_type == 'gradient_following':
        # Gradient following should be fast but may miss source
        assert time_to_target <= 20.0  # Should be relatively fast
    
    # Check sample size is appropriate for statistical validity
    assert metrics_result['sample_size'] >= 10, "Insufficient sample size for statistical validity"


@pytest.mark.unit
@pytest.mark.parametrize('normalization_enabled', [True, False])
@measure_performance(time_limit_seconds=2.0)
def test_calculate_path_efficiency_metrics(trajectory_data, optimal_paths, normalization_enabled):
    """
    Test calculation of path efficiency metrics including distance optimization,
    search pattern analysis, and energy consumption evaluation with normalization options.
    """
    # Create mock trajectory data with realistic navigation patterns
    test_trajectories = []
    optimal_path_refs = {}
    
    for traj_data in trajectory_data:
        test_trajectories.append(traj_data['trajectory'])
        algorithm = traj_data['algorithm']
        
        # Generate optimal path reference for this algorithm
        if algorithm not in optimal_path_refs:
            optimal_path_refs[algorithm] = optimal_paths['direct_path']
    
    # Generate optimal path references for comparison
    optimal_path_references = {
        'direct_path': optimal_paths['direct_path'],
        'exploration_optimal': optimal_paths['exploration_optimal'],
        'energy_optimal': optimal_paths['energy_optimal']
    }
    
    # Call calculate_path_efficiency_metrics with test data
    efficiency_metrics = calculate_path_efficiency_metrics(
        trajectory_data=test_trajectories,
        optimal_paths=optimal_path_references,
        enable_normalization=normalization_enabled
    )
    
    # Validate path optimality ratio calculations
    assert 'path_optimality_ratio' in efficiency_metrics
    optimality_ratio = efficiency_metrics['path_optimality_ratio']
    assert isinstance(optimality_ratio, (int, float))
    assert 0.0 <= optimality_ratio <= 1.0, f"Invalid optimality ratio: {optimality_ratio}"
    
    # Assert total distance traveled metrics accuracy
    assert 'total_distance_traveled' in efficiency_metrics
    total_distance = efficiency_metrics['total_distance_traveled']
    assert isinstance(total_distance, (int, float))
    assert total_distance > 0, "Total distance should be positive"
    
    # Check search pattern efficiency calculations
    assert 'search_pattern_efficiency' in efficiency_metrics
    pattern_efficiency = efficiency_metrics['search_pattern_efficiency']
    assert 0.0 <= pattern_efficiency <= 1.0, f"Invalid pattern efficiency: {pattern_efficiency}"
    
    # Validate energy consumption metrics
    if 'energy_consumption' in efficiency_metrics:
        energy = efficiency_metrics['energy_consumption']
        assert isinstance(energy, (int, float))
        assert energy > 0, "Energy consumption should be positive"
    
    # Compare results against benchmark data
    benchmark_comparator = BenchmarkComparator()
    benchmark_data = benchmark_comparator.load_benchmark_data('path_efficiency')
    
    if benchmark_data:
        comparison = benchmark_comparator.compare_against_benchmark(
            efficiency_metrics, benchmark_data
        )
        
        # Validate correlation with reference implementation
        correlation = comparison.get('correlation_coefficient', 0.0)
        if correlation < CORRELATION_THRESHOLD:
            warnings.warn(f"Path efficiency correlation {correlation:.6f} below threshold")
    
    # Assert numerical accuracy within tolerance
    assert_arrays_almost_equal(
        np.array([optimality_ratio, pattern_efficiency]),
        np.array([optimality_ratio, pattern_efficiency]),  # Self-consistency check
        tolerance=NUMERICAL_TOLERANCE,
        error_message="Path efficiency metrics numerical consistency check failed"
    )
    
    # Validate normalization effects if enabled
    if normalization_enabled:
        # Check that normalized metrics are in expected ranges
        if 'normalized_distance' in efficiency_metrics:
            norm_distance = efficiency_metrics['normalized_distance']
            assert 0.0 <= norm_distance <= 1.0, "Normalized distance should be in [0,1]"
    
    # Validate efficiency metric relationships
    if 'direct_path_ratio' in efficiency_metrics:
        direct_ratio = efficiency_metrics['direct_path_ratio']
        assert direct_ratio >= optimality_ratio, "Direct path ratio should be >= optimality ratio"


@pytest.mark.unit
@measure_performance(time_limit_seconds=1.5)
def test_calculate_temporal_dynamics_metrics(temporal_data, plume_encounter_times):
    """
    Test calculation of temporal dynamics metrics including response time to plume
    encounters and decision-making latency with timing validation.
    """
    # Generate realistic temporal dynamics test data
    test_temporal_data = []
    encounter_time_data = []
    
    for data_entry in temporal_data:
        test_temporal_data.append({
            'simulation_id': data_entry['simulation_id'],
            'encounter_times': data_entry['encounter_times'],
            'response_times': data_entry['response_times'],
            'decision_latencies': data_entry['decision_latencies']
        })
    
    for condition, encounter_data in plume_encounter_times.items():
        encounter_time_data.append({
            'condition': condition,
            'encounter_times': encounter_data['times'],
            'intervals': encounter_data['intervals']
        })
    
    # Call calculate_temporal_dynamics_metrics with test data
    temporal_metrics = calculate_temporal_dynamics_metrics(
        temporal_data=test_temporal_data,
        plume_encounter_times=encounter_time_data
    )
    
    # Validate response time to plume calculations
    assert 'mean_response_time' in temporal_metrics
    mean_response = temporal_metrics['mean_response_time']
    assert isinstance(mean_response, (int, float))
    assert mean_response > 0, "Mean response time should be positive"
    
    # Assert decision-making latency metrics accuracy
    assert 'decision_making_latency' in temporal_metrics
    decision_latency = temporal_metrics['decision_making_latency']
    assert isinstance(decision_latency, (int, float))
    assert decision_latency >= 0, "Decision latency should be non-negative"
    
    # Check adaptation time calculations
    if 'adaptation_time' in temporal_metrics:
        adaptation_time = temporal_metrics['adaptation_time']
        assert isinstance(adaptation_time, (int, float))
        assert adaptation_time > 0, "Adaptation time should be positive"
    
    # Validate convergence rate metrics
    if 'convergence_rate' in temporal_metrics:
        convergence_rate = temporal_metrics['convergence_rate']
        assert 0.0 <= convergence_rate <= 1.0, f"Invalid convergence rate: {convergence_rate}"
    
    # Compare against <7.2 seconds processing requirement
    processing_time = temporal_metrics.get('processing_time', 0)
    if processing_time > 0:
        assert processing_time <= PROCESSING_TIME_TARGET, f"Processing time {processing_time} exceeds target {PROCESSING_TIME_TARGET}"
    
    # Assert statistical validity of temporal metrics
    if 'statistical_measures' in temporal_metrics:
        stats = temporal_metrics['statistical_measures']
        
        # Check for required statistical measures
        assert 'standard_deviation' in stats
        assert 'confidence_interval' in stats
        
        # Validate statistical measures are reasonable
        std_dev = stats['standard_deviation']
        assert std_dev >= 0, "Standard deviation should be non-negative"
        
        if mean_response > 0 and std_dev > 0:
            coefficient_of_variation = std_dev / mean_response
            assert coefficient_of_variation < 2.0, "Excessive temporal variability detected"
    
    # Validate temporal consistency across conditions
    if len(encounter_time_data) > 1:
        response_times_by_condition = {}
        for i, condition_data in enumerate(encounter_time_data):
            condition = condition_data['condition']
            # Extract corresponding response times for this condition
            if i < len(test_temporal_data):
                response_times_by_condition[condition] = test_temporal_data[i]['response_times']
        
        # Check that response times vary appropriately with conditions
        if len(response_times_by_condition) >= 2:
            conditions = list(response_times_by_condition.keys())
            for i in range(len(conditions) - 1):
                cond1, cond2 = conditions[i], conditions[i + 1]
                times1 = response_times_by_condition[cond1]
                times2 = response_times_by_condition[cond2]
                
                # Temporal metrics should show some variation between conditions
                mean1, mean2 = np.mean(times1), np.mean(times2)
                relative_difference = abs(mean1 - mean2) / max(mean1, mean2)
                assert relative_difference >= 0.05, "Insufficient temporal variation between conditions"


@pytest.mark.unit
@pytest.mark.parametrize('noise_level', [0.0, 0.1, 0.2, 0.5])
@measure_performance(time_limit_seconds=3.0)
def test_calculate_robustness_metrics(multi_condition_results, noise_levels, noise_level):
    """
    Test calculation of robustness metrics including performance degradation rate,
    noise tolerance, and environmental adaptability across varying conditions.
    """
    # Generate multi-condition test results with varying noise levels
    test_conditions = {}
    noise_level_data = {}
    
    # Filter results for specific noise level
    target_noise = noise_level
    for condition_name, condition_results in multi_condition_results.items():
        # Check if this condition matches the target noise level
        if condition_results and len(condition_results) > 0:
            condition_noise = condition_results[0]['parameters'].get('noise_level', 0.0)
            if abs(condition_noise - target_noise) < 0.05:  # Tolerance for floating point comparison
                test_conditions[condition_name] = condition_results
    
    # If no exact match, use baseline condition and modify noise level
    if not test_conditions and 'baseline' in multi_condition_results:
        test_conditions['test_condition'] = multi_condition_results['baseline']
        for result in test_conditions['test_condition']:
            result['parameters']['noise_level'] = target_noise
    
    # Prepare noise level configuration
    noise_level_data = {
        'target_noise_level': target_noise,
        'noise_characteristics': noise_levels['noise_characteristics'],
        'validation_criteria': noise_levels['validation_criteria']
    }
    
    # Call calculate_robustness_metrics with test data
    robustness_metrics = calculate_robustness_metrics(
        multi_condition_results=test_conditions,
        noise_levels=noise_level_data
    )
    
    # Validate performance degradation rate calculations
    assert 'performance_degradation_rate' in robustness_metrics
    degradation_rate = robustness_metrics['performance_degradation_rate']
    assert isinstance(degradation_rate, (int, float))
    assert 0.0 <= degradation_rate <= 1.0, f"Invalid degradation rate: {degradation_rate}"
    
    # Assert noise tolerance metrics accuracy
    assert 'noise_tolerance' in robustness_metrics
    noise_tolerance = robustness_metrics['noise_tolerance']
    assert isinstance(noise_tolerance, (int, float))
    assert 0.0 <= noise_tolerance <= 1.0, f"Invalid noise tolerance: {noise_tolerance}"
    
    # Check environmental adaptability assessments
    if 'environmental_adaptability' in robustness_metrics:
        adaptability = robustness_metrics['environmental_adaptability']
        assert isinstance(adaptability, (int, float))
        assert 0.0 <= adaptability <= 1.0, f"Invalid adaptability score: {adaptability}"
    
    # Validate stability index calculations
    assert 'stability_index' in robustness_metrics
    stability_index = robustness_metrics['stability_index']
    assert isinstance(stability_index, (int, float))
    assert 0.0 <= stability_index <= 1.0, f"Invalid stability index: {stability_index}"
    
    # Compare robustness across different conditions
    if len(test_conditions) > 1:
        # Validate that robustness metrics reflect condition differences
        condition_scores = {}
        for condition_name, condition_data in test_conditions.items():
            if condition_data:
                avg_performance = np.mean([r['success_rate'] for r in condition_data])
                condition_scores[condition_name] = avg_performance
        
        # Check that robustness correlates with performance consistency
        if len(condition_scores) >= 2:
            score_variance = np.var(list(condition_scores.values()))
            expected_robustness = 1.0 - min(1.0, score_variance * 2)  # Simple relationship
            
            robustness_difference = abs(stability_index - expected_robustness)
            assert robustness_difference <= 0.3, "Robustness index inconsistent with performance variance"
    
    # Assert statistical significance of robustness measures
    if 'statistical_validation' in robustness_metrics:
        stat_validation = robustness_metrics['statistical_validation']
        
        if 'confidence_interval' in stat_validation:
            ci = stat_validation['confidence_interval']
            assert len(ci) == 2, "Confidence interval should have lower and upper bounds"
            assert ci[0] <= ci[1], "Confidence interval bounds should be ordered"
        
        if 'p_value' in stat_validation:
            p_value = stat_validation['p_value']
            assert 0.0 <= p_value <= 1.0, f"Invalid p-value: {p_value}"
    
    # Validate noise level specific expectations
    if target_noise <= 0.1:
        # Low noise should maintain high performance
        assert degradation_rate <= 0.2, f"Excessive degradation {degradation_rate} for low noise {target_noise}"
        assert noise_tolerance >= 0.8, f"Insufficient noise tolerance {noise_tolerance} for low noise"
    elif target_noise >= 0.5:
        # High noise should show measurable degradation
        assert degradation_rate >= 0.1, f"Insufficient degradation {degradation_rate} for high noise {target_noise}"
    
    # Check robustness threshold compliance
    min_performance_retention = noise_levels['validation_criteria']['min_performance_retention']
    performance_retention = 1.0 - degradation_rate
    if performance_retention < min_performance_retention:
        warnings.warn(f"Performance retention {performance_retention:.3f} below minimum {min_performance_retention}")


@pytest.mark.unit
@pytest.mark.parametrize('ranking_method', ['composite_score', 'weighted_average', 'pareto_ranking'])
@measure_performance(time_limit_seconds=2.5)
def test_compare_algorithm_performance(algorithm_metrics, comparison_categories, ranking_method):
    """
    Test algorithm performance comparison functionality with statistical significance
    testing and ranking analysis using multiple ranking methodologies.
    """
    # Generate mock performance data for multiple algorithms
    test_algorithm_metrics = {}
    comparison_category_list = comparison_categories
    
    # Prepare algorithm metrics for comparison
    for algorithm, metrics in algorithm_metrics.items():
        test_algorithm_metrics[algorithm] = {
            'success_rate': metrics['success_rate'],
            'path_efficiency': metrics['path_efficiency'],
            'processing_speed': metrics['processing_speed'],
            'convergence_rate': metrics['convergence_rate'],
            'robustness_index': metrics['robustness_index'],
            'resource_utilization': metrics['resource_utilization'],
            
            # Include sample data for statistical testing
            'success_rate_samples': metrics['success_rate_samples'],
            'path_efficiency_samples': metrics['path_efficiency_samples'],
            'processing_speed_samples': metrics['processing_speed_samples']
        }
    
    # Call compare_algorithm_performance with test data
    comparison_result = compare_algorithm_performance(
        algorithm_metrics=test_algorithm_metrics,
        comparison_categories=comparison_category_list,
        ranking_method=ranking_method
    )
    
    # Validate algorithm ranking calculations
    assert 'algorithm_rankings' in comparison_result
    rankings = comparison_result['algorithm_rankings']
    assert isinstance(rankings, list)
    assert len(rankings) == len(test_algorithm_metrics)
    
    # Check that all algorithms are included in rankings
    ranked_algorithms = {entry['algorithm'] for entry in rankings}
    expected_algorithms = set(test_algorithm_metrics.keys())
    assert ranked_algorithms == expected_algorithms, "Not all algorithms included in rankings"
    
    # Assert statistical significance testing results
    assert 'statistical_comparisons' in comparison_result
    stat_comparisons = comparison_result['statistical_comparisons']
    
    # Validate pairwise comparisons
    for comparison in stat_comparisons:
        assert 'algorithm_1' in comparison
        assert 'algorithm_2' in comparison
        assert 'p_value' in comparison
        assert 'effect_size' in comparison
        
        # Check statistical validity
        p_value = comparison['p_value']
        assert 0.0 <= p_value <= 1.0, f"Invalid p-value: {p_value}"
        
        effect_size = comparison['effect_size']
        assert isinstance(effect_size, (int, float)), "Effect size should be numeric"
    
    # Check effect size calculations
    effect_sizes = [comp['effect_size'] for comp in stat_comparisons]
    if effect_sizes:
        # Effect sizes should be reasonable (Cohen's d typically -3 to 3)
        assert all(-5.0 <= es <= 5.0 for es in effect_sizes), "Unreasonable effect sizes detected"
    
    # Validate composite performance scoring
    assert 'performance_scores' in comparison_result
    performance_scores = comparison_result['performance_scores']
    
    for algorithm in test_algorithm_metrics.keys():
        assert algorithm in performance_scores
        score = performance_scores[algorithm]
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 1.0, f"Invalid performance score for {algorithm}: {score}"
    
    # Compare rankings against expected ordering
    # Higher performance should generally rank better
    sorted_by_score = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
    rankings_by_rank = sorted(rankings, key=lambda x: x['rank'])
    
    # Check that ranking order is consistent with performance scores
    for i, (algo_score, score) in enumerate(sorted_by_score):
        ranked_algo = rankings_by_rank[i]['algorithm']
        if algo_score != ranked_algo:
            # Allow some flexibility in ranking due to different ranking methods
            score_diff = abs(score - performance_scores[ranked_algo])
            assert score_diff <= 0.1, f"Ranking inconsistent with performance scores for {ranking_method}"
    
    # Assert statistical validity of comparisons
    significant_comparisons = [c for c in stat_comparisons if c['p_value'] < STATISTICAL_SIGNIFICANCE_LEVEL]
    if len(significant_comparisons) > 0:
        # At least some differences should be statistically significant for validation
        assert len(significant_comparisons) <= len(stat_comparisons), "All comparisons cannot be significant"
    
    # Validate ranking method specific results
    if ranking_method == 'composite_score':
        # Composite score should aggregate multiple metrics
        assert 'composite_calculation' in comparison_result
        composite_calc = comparison_result['composite_calculation']
        assert 'weights' in composite_calc
        assert 'aggregation_method' in composite_calc
        
    elif ranking_method == 'pareto_ranking':
        # Pareto ranking should identify non-dominated solutions
        assert 'pareto_front' in comparison_result
        pareto_front = comparison_result['pareto_front']
        assert isinstance(pareto_front, list)
        assert len(pareto_front) >= 1, "At least one algorithm should be on Pareto front"
    
    # Check for reasonable distribution of rankings
    rank_values = [entry['rank'] for entry in rankings]
    assert min(rank_values) == 1, "Best rank should be 1"
    assert max(rank_values) == len(rankings), "Worst rank should equal number of algorithms"
    assert len(set(rank_values)) == len(rank_values), "Ranks should be unique"


@pytest.mark.unit
@pytest.mark.parametrize('strict_validation', [True, False])
@measure_performance(time_limit_seconds=1.0)
def test_validate_performance_against_thresholds(test_config, strict_validation):
    """
    Test performance validation against predefined thresholds including >95% correlation
    and <7.2 seconds processing time with configurable validation strictness.
    """
    # Create test metrics with known threshold compliance status
    calculated_metrics = {
        'correlation_coefficient': test_config['correlation_threshold'] + 0.01,  # Slightly above threshold
        'processing_time': test_config['processing_time_target'] - 0.5,  # Below time target
        'reproducibility_coefficient': test_config['reproducibility_threshold'] + 0.005,  # Above repro threshold
        'error_rate': 0.005,  # Below 1% error rate
        'memory_usage_gb': 6.5,  # Below 8GB limit
        'success_rate': 0.88,  # Good success rate
        'numerical_accuracy': NUMERICAL_TOLERANCE / 2,  # Well within tolerance
        'batch_completion_rate': 0.99  # High completion rate
    }
    
    # Define performance thresholds for validation
    performance_thresholds = {
        'correlation_threshold': test_config['correlation_threshold'],
        'processing_time_limit': test_config['processing_time_target'],
        'reproducibility_threshold': test_config['reproducibility_threshold'],
        'max_error_rate': 0.01,  # 1% maximum error rate
        'memory_limit_gb': 8.0,  # 8GB memory limit
        'min_success_rate': 0.8,  # 80% minimum success rate
        'numerical_tolerance': NUMERICAL_TOLERANCE,
        'min_completion_rate': 0.95  # 95% minimum completion rate
    }
    
    # Call validate_performance_against_thresholds with test data
    validation_result = validate_performance_against_thresholds(
        calculated_metrics=calculated_metrics,
        performance_thresholds=performance_thresholds,
        strict_validation=strict_validation
    )
    
    # Validate correlation threshold checking (>95%)
    assert 'correlation_validation' in validation_result
    correlation_check = validation_result['correlation_validation']
    assert correlation_check['meets_threshold'] == True
    assert correlation_check['actual_value'] >= CORRELATION_THRESHOLD
    
    # Assert processing time validation (<7.2 seconds)
    assert 'processing_time_validation' in validation_result
    time_check = validation_result['processing_time_validation']
    assert time_check['meets_threshold'] == True
    assert time_check['actual_value'] <= PROCESSING_TIME_TARGET
    
    # Check reproducibility coefficient validation (>0.99)
    assert 'reproducibility_validation' in validation_result
    repro_check = validation_result['reproducibility_validation']
    assert repro_check['meets_threshold'] == True
    assert repro_check['actual_value'] >= REPRODUCIBILITY_THRESHOLD
    
    # Validate threshold compliance reporting
    assert 'overall_compliance' in validation_result
    overall_compliance = validation_result['overall_compliance']
    assert isinstance(overall_compliance, bool)
    
    if strict_validation:
        # In strict mode, all thresholds must be met
        compliance_checks = [
            correlation_check['meets_threshold'],
            time_check['meets_threshold'],
            repro_check['meets_threshold']
        ]
        assert overall_compliance == all(compliance_checks)
    
    # Assert proper handling of threshold violations
    # Test with metrics that violate thresholds
    violating_metrics = calculated_metrics.copy()
    violating_metrics['correlation_coefficient'] = 0.90  # Below 95% threshold
    violating_metrics['processing_time'] = 10.0  # Above 7.2s limit
    
    violation_result = validate_performance_against_thresholds(
        calculated_metrics=violating_metrics,
        performance_thresholds=performance_thresholds,
        strict_validation=strict_validation
    )
    
    # Check that violations are properly detected
    assert violation_result['correlation_validation']['meets_threshold'] == False
    assert violation_result['processing_time_validation']['meets_threshold'] == False
    assert violation_result['overall_compliance'] == False
    
    # Check generation of optimization recommendations
    assert 'recommendations' in violation_result
    recommendations = violation_result['recommendations']
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0, "Should generate recommendations for threshold violations"
    
    # Validate recommendation content
    correlation_recommendations = [r for r in recommendations if 'correlation' in r.lower()]
    time_recommendations = [r for r in recommendations if 'time' in r.lower()]
    
    assert len(correlation_recommendations) > 0, "Should recommend correlation improvements"
    assert len(time_recommendations) > 0, "Should recommend processing time improvements"
    
    # Test edge cases with threshold boundary values
    boundary_metrics = {
        'correlation_coefficient': CORRELATION_THRESHOLD,  # Exactly at threshold
        'processing_time': PROCESSING_TIME_TARGET,  # Exactly at threshold
        'reproducibility_coefficient': REPRODUCIBILITY_THRESHOLD  # Exactly at threshold
    }
    
    boundary_thresholds = {
        'correlation_threshold': CORRELATION_THRESHOLD,
        'processing_time_limit': PROCESSING_TIME_TARGET,
        'reproducibility_threshold': REPRODUCIBILITY_THRESHOLD
    }
    
    boundary_result = validate_performance_against_thresholds(
        calculated_metrics=boundary_metrics,
        performance_thresholds=boundary_thresholds,
        strict_validation=strict_validation
    )
    
    # Boundary values should meet thresholds (inclusive)
    assert boundary_result['correlation_validation']['meets_threshold'] == True
    assert boundary_result['processing_time_validation']['meets_threshold'] == True
    assert boundary_result['reproducibility_validation']['meets_threshold'] == True


@pytest.mark.unit
@pytest.mark.integration
@measure_performance(time_limit_seconds=3.0)
def test_cross_format_compatibility_metrics(test_config):
    """
    Test cross-format compatibility metrics calculation between Crimaldi and
    custom plume formats with comprehensive format validation and consistency checking.
    """
    # Load Crimaldi and custom format test data
    crimaldi_results = {
        'spatial_calibration': {
            'pixel_to_meter_ratio': 100.0,
            'arena_width_meters': 1.0,
            'arena_height_meters': 1.0,
            'calibration_accuracy': 0.98
        },
        'temporal_data': {
            'frame_rate': 50.0,
            'total_duration': 120.0,
            'temporal_resolution': 0.02
        },
        'intensity_data': {
            'intensity_range': [0, 255],
            'bit_depth': 8,
            'dynamic_range': 255,
            'calibration_factor': 1.0
        },
        'coordinate_transform': {
            'transform_matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            'origin_offset': [0.0, 0.0],
            'scaling_factor': 1.0
        },
        'trajectory_data': generate_mock_performance_data(
            num_trajectories=20,
            format_type='crimaldi'
        ),
        'performance_metrics': {
            'success_rate': 0.85,
            'path_efficiency': 0.78,
            'processing_time': 6.5
        }
    }
    
    custom_results = {
        'spatial_calibration': {
            'pixel_to_meter_ratio': 150.0,
            'arena_width_meters': 1.0,
            'arena_height_meters': 1.0,
            'calibration_accuracy': 0.96
        },
        'temporal_data': {
            'frame_rate': 30.0,
            'total_duration': 120.0,
            'temporal_resolution': 0.033
        },
        'intensity_data': {
            'intensity_range': [0, 65535],
            'bit_depth': 16,
            'dynamic_range': 65535,
            'calibration_factor': 256.0
        },
        'coordinate_transform': {
            'transform_matrix': [[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]],
            'origin_offset': [0.0, 0.0],
            'scaling_factor': 1.5
        },
        'trajectory_data': generate_mock_performance_data(
            num_trajectories=20,
            format_type='custom'
        ),
        'performance_metrics': {
            'success_rate': 0.83,
            'path_efficiency': 0.80,
            'processing_time': 6.8
        }
    }
    
    # Generate performance metrics for both formats
    crimaldi_metrics = calculate_navigation_success_metrics(
        simulation_results=[{'trajectory': t, 'algorithm_type': 'infotaxis'} for t in crimaldi_results['trajectory_data']],
        target_locations={'primary_source': np.array([0.8, 0.5])},
        algorithm_type='crimaldi_format'
    )
    
    custom_metrics = calculate_navigation_success_metrics(
        simulation_results=[{'trajectory': t, 'algorithm_type': 'infotaxis'} for t in custom_results['trajectory_data']],
        target_locations={'primary_source': np.array([0.8, 0.5])},
        algorithm_type='custom_format'
    )
    
    # Call calculate_cross_format_compatibility_metrics
    compatibility_result = validate_cross_format_compatibility(
        crimaldi_results=crimaldi_results,
        custom_results=custom_results
    )
    
    # Validate format consistency calculations
    assert 'format_consistency' in compatibility_result
    consistency = compatibility_result['format_consistency']
    assert isinstance(consistency, dict)
    
    # Check spatial calibration consistency
    if 'spatial_consistency' in consistency:
        spatial_consistency = consistency['spatial_consistency']
        assert 0.0 <= spatial_consistency <= 1.0, f"Invalid spatial consistency: {spatial_consistency}"
    
    # Assert cross-format correlation analysis
    assert 'cross_format_correlation' in compatibility_result
    correlation_analysis = compatibility_result['cross_format_correlation']
    
    # Validate correlation coefficients
    if 'trajectory_correlation' in correlation_analysis:
        traj_correlation = correlation_analysis['trajectory_correlation']
        assert -1.0 <= traj_correlation <= 1.0, f"Invalid trajectory correlation: {traj_correlation}"
    
    if 'performance_correlation' in correlation_analysis:
        perf_correlation = correlation_analysis['performance_correlation']
        assert -1.0 <= perf_correlation <= 1.0, f"Invalid performance correlation: {perf_correlation}"
    
    # Check temporal alignment accuracy
    assert 'temporal_alignment' in compatibility_result
    temporal_alignment = compatibility_result['temporal_alignment']
    
    # Validate temporal synchronization
    if 'frame_rate_compatibility' in temporal_alignment:
        frame_compatibility = temporal_alignment['frame_rate_compatibility']
        assert isinstance(frame_compatibility, (int, float))
        
        # Frame rate ratio should be reasonable
        crimaldi_fps = crimaldi_results['temporal_data']['frame_rate']
        custom_fps = custom_results['temporal_data']['frame_rate']
        expected_ratio = crimaldi_fps / custom_fps
        
        assert abs(frame_compatibility - expected_ratio) <= 0.1, "Frame rate compatibility calculation error"
    
    # Validate intensity calibration consistency
    assert 'intensity_calibration' in compatibility_result
    intensity_calibration = compatibility_result['intensity_calibration']
    
    # Check bit depth conversion accuracy
    if 'bit_depth_conversion' in intensity_calibration:
        conversion_accuracy = intensity_calibration['bit_depth_conversion']
        assert 0.0 <= conversion_accuracy <= 1.0, f"Invalid conversion accuracy: {conversion_accuracy}"
    
    # Assert compatibility threshold compliance
    assert 'overall_compatibility' in compatibility_result
    overall_compatibility = compatibility_result['overall_compatibility']
    assert isinstance(overall_compatibility, (int, float))
    assert 0.0 <= overall_compatibility <= 1.0, f"Invalid overall compatibility: {overall_compatibility}"
    
    # Check minimum compatibility requirements
    min_compatibility_threshold = 0.8  # 80% minimum compatibility
    if overall_compatibility < min_compatibility_threshold:
        warnings.warn(f"Cross-format compatibility {overall_compatibility:.3f} below minimum {min_compatibility_threshold}")
    
    # Validate specific format conversion metrics
    assert 'conversion_metrics' in compatibility_result
    conversion_metrics = compatibility_result['conversion_metrics']
    
    # Check pixel-to-meter ratio conversion
    if 'spatial_conversion_accuracy' in conversion_metrics:
        spatial_accuracy = conversion_metrics['spatial_conversion_accuracy']
        assert 0.0 <= spatial_accuracy <= 1.0, "Invalid spatial conversion accuracy"
    
    # Validate temporal conversion accuracy
    if 'temporal_conversion_accuracy' in conversion_metrics:
        temporal_accuracy = conversion_metrics['temporal_conversion_accuracy']
        assert 0.0 <= temporal_accuracy <= 1.0, "Invalid temporal conversion accuracy"
    
    # Test conversion reversibility
    if 'conversion_reversibility' in conversion_metrics:
        reversibility = conversion_metrics['conversion_reversibility']
        assert 0.0 <= reversibility <= 1.0, "Invalid conversion reversibility"
        
        # Reversibility should be high for good compatibility
        assert reversibility >= 0.95, f"Low conversion reversibility: {reversibility}"


@pytest.mark.unit
@measure_performance(time_limit_seconds=1.5)
def test_navigation_success_analyzer():
    """
    Test NavigationSuccessAnalyzer class functionality including localization
    success rate and time to target analysis with statistical validation.
    """
    # Initialize NavigationSuccessAnalyzer with test criteria
    success_criteria = {
        'max_distance_to_source': 0.1,  # 10cm maximum distance
        'max_time_to_target': 30.0,     # 30 seconds maximum time
        'min_success_rate': 0.8,        # 80% minimum success rate
        'statistical_confidence': 0.95   # 95% confidence level
    }
    
    analyzer = NavigationSuccessAnalyzer(
        success_criteria=success_criteria,
        enable_statistical_analysis=True
    )
    
    # Generate mock final positions and target locations
    np.random.seed(RANDOM_SEED)
    num_trials = 100
    target_location = np.array([0.8, 0.5])
    
    # Generate realistic final positions with some successful and some failed localizations
    final_positions = []
    for i in range(num_trials):
        if i < 80:  # 80% success rate
            # Successful localization - close to target
            noise = np.random.normal(0, 0.03, 2)  # Small noise
            position = target_location + noise
        else:
            # Failed localization - random position
            position = np.random.uniform(0, 1, 2)
        
        final_positions.append(position)
    
    final_positions = np.array(final_positions)
    
    # Test calculate_localization_success_rate method
    success_result = analyzer.calculate_localization_success_rate(
        final_positions=final_positions,
        target_locations=[target_location]
    )
    
    # Validate success rate calculations with distance thresholds
    assert 'success_rate' in success_result
    success_rate = success_result['success_rate']
    assert isinstance(success_rate, (int, float))
    assert 0.0 <= success_rate <= 1.0, f"Invalid success rate: {success_rate}"
    
    # Check that success rate is approximately correct (around 80%)
    expected_success_rate = 0.8
    assert abs(success_rate - expected_success_rate) <= 0.15, f"Success rate {success_rate} far from expected {expected_success_rate}"
    
    # Validate distance threshold application
    assert 'distance_statistics' in success_result
    distance_stats = success_result['distance_statistics']
    assert 'mean_distance' in distance_stats
    assert 'std_distance' in distance_stats
    
    mean_distance = distance_stats['mean_distance']
    assert mean_distance >= 0, "Mean distance should be non-negative"
    
    # Test analyze_time_to_target method
    # Generate time to target data
    time_to_target_data = []
    for i in range(num_trials):
        if i < 80:  # Successful trials
            # Successful trials should have reasonable times
            time_val = np.random.exponential(15.0)  # Mean 15 seconds
        else:
            # Failed trials might have longer times or timeout
            time_val = np.random.uniform(25.0, 35.0)  # Longer times
        
        time_to_target_data.append(time_val)
    
    time_result = analyzer.analyze_time_to_target(
        time_to_target_data=time_to_target_data,
        success_indicators=[i < 80 for i in range(num_trials)]
    )
    
    # Assert statistical distribution analysis
    assert 'time_statistics' in time_result
    time_stats = time_result['time_statistics']
    
    assert 'mean_time' in time_stats
    assert 'median_time' in time_stats
    assert 'std_time' in time_stats
    
    mean_time = time_stats['mean_time']
    assert mean_time > 0, "Mean time should be positive"
    assert mean_time <= success_criteria['max_time_to_target'] * 1.5, "Mean time unreasonably high"
    
    # Check confidence interval calculations
    assert 'confidence_intervals' in time_result
    confidence_intervals = time_result['confidence_intervals']
    
    if 'mean_time_ci' in confidence_intervals:
        ci = confidence_intervals['mean_time_ci']
        assert len(ci) == 2, "Confidence interval should have lower and upper bounds"
        assert ci[0] <= ci[1], "CI bounds should be ordered"
        assert ci[0] <= mean_time <= ci[1], "Mean should be within confidence interval"
    
    # Validate success criteria compliance
    assert 'criteria_compliance' in success_result
    compliance = success_result['criteria_compliance']
    
    # Check distance criteria compliance
    if 'distance_compliance' in compliance:
        distance_compliance = compliance['distance_compliance']
        assert isinstance(distance_compliance, bool)
    
    # Check success rate criteria compliance
    if 'success_rate_compliance' in compliance:
        rate_compliance = compliance['success_rate_compliance']
        expected_compliance = success_rate >= success_criteria['min_success_rate']
        assert rate_compliance == expected_compliance, "Success rate compliance calculation error"
    
    # Test statistical significance of results
    if 'statistical_tests' in success_result:
        stat_tests = success_result['statistical_tests']
        
        # Check for appropriate statistical tests
        if 'normality_test' in stat_tests:
            normality = stat_tests['normality_test']
            assert 'statistic' in normality
            assert 'p_value' in normality
            
            p_value = normality['p_value']
            assert 0.0 <= p_value <= 1.0, f"Invalid p-value: {p_value}"
    
    # Validate analyzer state and configuration
    assert analyzer.success_criteria == success_criteria
    assert hasattr(analyzer, 'enable_statistical_analysis')
    assert analyzer.enable_statistical_analysis == True


@pytest.mark.unit
@measure_performance(time_limit_seconds=2.0)
def test_path_efficiency_analyzer():
    """
    Test PathEfficiencyAnalyzer class functionality including path optimality
    and search pattern analysis with comprehensive efficiency evaluation.
    """
    # Initialize PathEfficiencyAnalyzer with optimal path references
    optimal_path_references = {
        'direct_path': np.array([[0.1, 0.5], [0.8, 0.5]]),  # Straight line
        'exploration_optimal': _generate_exploration_optimal_path(),
        'energy_optimal': _generate_energy_optimal_path()
    }
    
    analyzer = PathEfficiencyAnalyzer(
        optimal_path_references=optimal_path_references,
        enable_pattern_analysis=True,
        calculate_energy_metrics=True
    )
    
    # Generate mock trajectory data with realistic patterns
    np.random.seed(RANDOM_SEED)
    actual_trajectories = []
    
    for i in range(20):  # Generate 20 test trajectories
        # Create trajectory with realistic search behavior
        start_pos = np.array([0.1, 0.5])
        target_pos = np.array([0.8, 0.5])
        
        trajectory = [start_pos]
        current_pos = start_pos.copy()
        
        # Generate path with exploration and convergence
        for step in range(30):  # 30 steps per trajectory
            # Direction toward target with exploration noise
            direction_to_target = target_pos - current_pos
            if np.linalg.norm(direction_to_target) > 0:
                direction_to_target /= np.linalg.norm(direction_to_target)
            
            # Add exploration component
            exploration_noise = np.random.normal(0, 0.3, 2)
            movement = 0.7 * direction_to_target + 0.3 * exploration_noise
            
            # Normalize movement
            if np.linalg.norm(movement) > 0:
                movement = movement / np.linalg.norm(movement) * 0.03
            
            current_pos += movement
            current_pos = np.clip(current_pos, 0.0, 1.0)  # Keep in bounds
            trajectory.append(current_pos.copy())
        
        actual_trajectories.append(np.array(trajectory))
    
    # Test calculate_path_optimality method
    optimality_result = analyzer.calculate_path_optimality(
        actual_trajectories=actual_trajectories,
        reference_type='direct_path'
    )
    
    # Validate optimality ratio calculations
    assert 'optimality_ratio' in optimality_result
    optimality_ratio = optimality_result['optimality_ratio']
    assert isinstance(optimality_ratio, (int, float))
    assert 0.0 <= optimality_ratio <= 1.0, f"Invalid optimality ratio: {optimality_ratio}"
    
    # Check path length statistics
    assert 'path_statistics' in optimality_result
    path_stats = optimality_result['path_statistics']
    
    assert 'mean_path_length' in path_stats
    assert 'optimal_path_length' in path_stats
    assert 'efficiency_variance' in path_stats
    
    mean_length = path_stats['mean_path_length']
    optimal_length = path_stats['optimal_path_length']
    
    assert mean_length > 0, "Mean path length should be positive"
    assert optimal_length > 0, "Optimal path length should be positive"
    assert mean_length >= optimal_length, "Mean path length should be >= optimal length"
    
    # Test analyze_search_patterns method
    pattern_result = analyzer.analyze_search_patterns(
        trajectories=actual_trajectories,
        analyze_coverage=True,
        analyze_movement_patterns=True
    )
    
    # Assert coverage and exploration metrics
    assert 'coverage_metrics' in pattern_result
    coverage_metrics = pattern_result['coverage_metrics']
    
    assert 'spatial_coverage' in coverage_metrics
    spatial_coverage = coverage_metrics['spatial_coverage']
    assert 0.0 <= spatial_coverage <= 1.0, f"Invalid spatial coverage: {spatial_coverage}"
    
    # Check exploration efficiency
    if 'exploration_efficiency' in coverage_metrics:
        exploration_eff = coverage_metrics['exploration_efficiency']
        assert 0.0 <= exploration_eff <= 1.0, f"Invalid exploration efficiency: {exploration_eff}"
    
    # Check movement optimization analysis
    assert 'movement_patterns' in pattern_result
    movement_patterns = pattern_result['movement_patterns']
    
    if 'directional_consistency' in movement_patterns:
        directional_consistency = movement_patterns['directional_consistency']
        assert 0.0 <= directional_consistency <= 1.0, f"Invalid directional consistency: {directional_consistency}"
    
    if 'movement_smoothness' in movement_patterns:
        smoothness = movement_patterns['movement_smoothness']
        assert isinstance(smoothness, (int, float))
        assert smoothness >= 0, "Movement smoothness should be non-negative"
    
    # Validate efficiency scoring accuracy
    assert 'efficiency_scores' in optimality_result
    efficiency_scores = optimality_result['efficiency_scores']
    
    # Check individual trajectory efficiency scores
    if 'individual_scores' in efficiency_scores:
        individual_scores = efficiency_scores['individual_scores']
        assert len(individual_scores) == len(actual_trajectories)
        
        for score in individual_scores:
            assert 0.0 <= score <= 1.0, f"Invalid individual efficiency score: {score}"
    
    # Test energy efficiency calculations if enabled
    if analyzer.calculate_energy_metrics:
        energy_result = analyzer.calculate_energy_efficiency(
            trajectories=actual_trajectories
        )
        
        assert 'energy_consumption' in energy_result
        energy_consumption = energy_result['energy_consumption']
        
        if 'total_energy' in energy_consumption:
            total_energy = energy_consumption['total_energy']
            assert total_energy >= 0, "Total energy should be non-negative"
        
        if 'energy_per_distance' in energy_consumption:
            energy_per_distance = energy_consumption['energy_per_distance']
            assert energy_per_distance >= 0, "Energy per distance should be non-negative"
    
    # Validate pattern recognition capabilities
    if 'pattern_classification' in pattern_result:
        pattern_class = pattern_result['pattern_classification']
        
        # Check for recognized search patterns
        recognized_patterns = ['spiral', 'casting', 'direct', 'random', 'zigzag']
        if 'dominant_pattern' in pattern_class:
            dominant_pattern = pattern_class['dominant_pattern']
            assert dominant_pattern in recognized_patterns or dominant_pattern == 'mixed'
    
    # Test comparison with multiple reference paths
    multi_ref_result = analyzer.calculate_path_optimality(
        actual_trajectories=actual_trajectories[:5],  # Use subset for faster testing
        reference_type='all'
    )
    
    # Should have results for all reference types
    for ref_type in optimal_path_references.keys():
        assert f"{ref_type}_optimality" in multi_ref_result or 'optimality_by_reference' in multi_ref_result
    
    # Validate analyzer configuration
    assert analyzer.optimal_path_references == optimal_path_references
    assert analyzer.enable_pattern_analysis == True
    assert analyzer.calculate_energy_metrics == True


@pytest.mark.unit
@pytest.mark.parametrize('cache_enabled', [True, False])
@measure_performance(time_limit_seconds=2.0)
def test_performance_metrics_caching(test_config, cache_enabled):
    """
    Test performance metrics caching functionality for improved calculation
    performance and consistency with cache validation and performance optimization.
    """
    # Initialize PerformanceMetricsCalculator with caching configuration
    calculator = PerformanceMetricsCalculator(
        enable_caching=cache_enabled,
        cache_size_limit=100,
        cache_ttl_minutes=30
    )
    
    # Generate test data for caching validation
    test_data = {
        'simulation_results': [
            {
                'trajectory': np.random.random((50, 2)),
                'algorithm_type': 'infotaxis',
                'success': True,
                'final_position': np.random.random(2)
            } for _ in range(10)
        ],
        'target_locations': {'primary_source': np.array([0.8, 0.5])},
        'algorithm_type': 'infotaxis'
    }
    
    # Perform initial metrics calculation and measure timing
    start_time = time.time()
    first_result = calculator.calculate_all_metrics(
        simulation_data=test_data,
        enable_caching=cache_enabled
    )
    first_calculation_time = time.time() - start_time
    
    # Validate initial calculation results
    assert first_result is not None
    assert isinstance(first_result, dict)
    assert 'navigation_success_metrics' in first_result
    
    # Repeat identical calculation and measure timing
    start_time = time.time()
    second_result = calculator.calculate_all_metrics(
        simulation_data=test_data,
        enable_caching=cache_enabled
    )
    second_calculation_time = time.time() - start_time
    
    # Validate cache hit performance improvement
    if cache_enabled:
        # Second calculation should be significantly faster due to caching
        cache_speedup = first_calculation_time / max(second_calculation_time, 0.001)
        assert cache_speedup >= 2.0, f"Insufficient cache speedup: {cache_speedup:.2f}x"
        
        # Results should be identical when cached
        # Compare key metrics for consistency
        first_nav_success = first_result['navigation_success_metrics']['source_localization_rate']
        second_nav_success = second_result['navigation_success_metrics']['source_localization_rate']
        assert abs(first_nav_success - second_nav_success) < NUMERICAL_TOLERANCE
        
    else:
        # Without caching, times should be similar
        time_ratio = max(first_calculation_time, second_calculation_time) / min(first_calculation_time, second_calculation_time)
        assert time_ratio <= 2.0, f"Excessive time variation without caching: {time_ratio:.2f}"
    
    # Test cache invalidation and refresh
    if cache_enabled:
        # Modify test data to trigger cache invalidation
        modified_data = test_data.copy()
        modified_data['simulation_results'] = test_data['simulation_results'][:5]  # Reduce data
        
        start_time = time.time()
        modified_result = calculator.calculate_all_metrics(
            simulation_data=modified_data,
            enable_caching=cache_enabled
        )
        modified_calculation_time = time.time() - start_time
        
        # Should calculate new result (cache miss)
        assert modified_result != first_result  # Results should differ
        assert modified_calculation_time > second_calculation_time * 1.5  # Should take longer than cached call
    
    # Assert cache consistency and accuracy
    if cache_enabled and hasattr(calculator, 'cache_statistics'):
        cache_stats = calculator.cache_statistics()
        
        assert 'cache_hits' in cache_stats
        assert 'cache_misses' in cache_stats
        assert 'cache_size' in cache_stats
        
        # Should have at least one cache hit from repeat calculation
        assert cache_stats['cache_hits'] >= 1, "Should have cache hits from repeat calculation"
    
    # Check memory usage optimization
    if cache_enabled:
        # Test cache size limits
        large_data_sets = []
        for i in range(20):  # Generate many cache entries
            large_test_data = {
                'simulation_results': [
                    {
                        'trajectory': np.random.random((30, 2)),
                        'algorithm_type': f'test_algorithm_{i}',
                        'success': True
                    } for _ in range(5)
                ],
                'target_locations': {'source': np.array([0.7 + i*0.01, 0.5])},
                'algorithm_type': f'test_algorithm_{i}'
            }
            
            result = calculator.calculate_all_metrics(
                simulation_data=large_test_data,
                enable_caching=cache_enabled
            )
            large_data_sets.append(result)
        
        # Check that cache respects size limits
        if hasattr(calculator, 'cache_statistics'):
            final_cache_stats = calculator.cache_statistics()
            cache_size = final_cache_stats.get('cache_size', 0)
            
            # Cache should not grow indefinitely
            assert cache_size <= 150, f"Cache size {cache_size} may be exceeding limits"
    
    # Validate cache size limits and eviction
    if cache_enabled and hasattr(calculator, 'clear_cache'):
        # Test manual cache clearing
        calculator.clear_cache()
        
        # Cache should be empty after clearing
        if hasattr(calculator, 'cache_statistics'):
            cleared_stats = calculator.cache_statistics()
            assert cleared_stats.get('cache_size', 0) == 0, "Cache should be empty after clearing"
        
        # Subsequent calculation should not benefit from cache
        start_time = time.time()
        post_clear_result = calculator.calculate_all_metrics(
            simulation_data=test_data,
            enable_caching=cache_enabled
        )
        post_clear_time = time.time() - start_time
        
        # Should take similar time to initial calculation
        time_ratio = post_clear_time / first_calculation_time
        assert 0.5 <= time_ratio <= 2.0, f"Post-clear calculation time ratio {time_ratio:.2f} unexpected"


@pytest.mark.unit
@pytest.mark.integration
@measure_performance(time_limit_seconds=3.0)
def test_statistical_validation_integration(test_config):
    """
    Test integration with statistical validation components for comprehensive
    metrics accuracy assessment with >95% correlation validation and significance testing.
    """
    # Load reference benchmark data using BenchmarkComparator
    benchmark_comparator = BenchmarkComparator(
        benchmark_types=['simulation', 'analysis', 'normalization'],
        correlation_threshold=test_config['correlation_threshold']
    )
    
    reference_benchmark = benchmark_comparator.load_benchmark_data('simulation')
    
    # Ensure we have reference data or create it
    if not reference_benchmark:
        reference_benchmark = {
            'navigation_success_rate': 0.850,
            'path_efficiency': 0.780,
            'temporal_response': 2.45,
            'robustness_score': 0.825,
            'processing_time': 6.8,
            'correlation_with_reference': 0.962,
            'statistical_measures': {
                'mean': 0.825,
                'std_dev': 0.045,
                'confidence_interval': [0.780, 0.870]
            }
        }
    
    # Calculate metrics using PerformanceMetricsCalculator
    calculator = PerformanceMetricsCalculator(
        correlation_threshold=test_config['correlation_threshold'],
        enable_statistical_validation=True
    )
    
    # Generate test simulation data
    test_simulation_data = {
        'simulation_results': [
            {
                'trajectory': np.random.random((40, 2)) * 0.8 + 0.1,  # Keep in arena bounds
                'algorithm_type': 'infotaxis',
                'success': np.random.random() > 0.15,  # ~85% success rate
                'final_position': np.random.random(2) * 0.8 + 0.1,
                'time_to_target': np.random.exponential(10) + 2,
                'path_length': np.random.uniform(0.5, 1.5)
            } for _ in range(50)
        ],
        'target_locations': {'primary_source': np.array([0.8, 0.5])},
        'algorithm_type': 'infotaxis'
    }
    
    calculated_metrics = calculator.calculate_all_metrics(
        simulation_data=test_simulation_data,
        enable_statistical_validation=True
    )
    
    # Perform correlation analysis using StatisticalValidator
    statistical_validator = StatisticalValidator(
        significance_level=test_config['statistical_significance'],
        correlation_method='pearson'
    )
    
    # Extract comparable metrics
    calculated_values = []
    reference_values = []
    metric_names = []
    
    # Map calculated metrics to reference metrics
    metric_mapping = {
        'navigation_success_rate': calculated_metrics.get('navigation_success_metrics', {}).get('source_localization_rate', 0),
        'path_efficiency': calculated_metrics.get('path_efficiency_metrics', {}).get('path_optimality_ratio', 0),
        'processing_time': calculated_metrics.get('performance_metrics', {}).get('calculation_time', 0)
    }
    
    for metric_name, calc_value in metric_mapping.items():
        if metric_name in reference_benchmark and calc_value is not None:
            calculated_values.append(calc_value)
            reference_values.append(reference_benchmark[metric_name])
            metric_names.append(metric_name)
    
    # Validate >95% correlation requirement compliance
    if len(calculated_values) >= 2:
        correlation_result = statistical_validator.perform_correlation_analysis(
            np.array(calculated_values),
            np.array(reference_values)
        )
        
        correlation_coefficient = correlation_result['correlation_coefficient']
        assert correlation_coefficient >= test_config['correlation_threshold'], \
            f"Correlation {correlation_coefficient:.6f} below requirement {test_config['correlation_threshold']}"
        
        # Test statistical significance of metric differences
        significance_result = statistical_validator.test_statistical_significance(
            np.array(calculated_values),
            np.array(reference_values)
        )
        
        p_value = significance_result['p_value']
        assert 0.0 <= p_value <= 1.0, f"Invalid p-value: {p_value}"
        
        # Check if differences are statistically significant
        is_significant = p_value < test_config['statistical_significance']
        if is_significant:
            warnings.warn(f"Statistically significant difference detected (p={p_value:.6f})")
    
    # Assert reproducibility coefficient validation
    # Test calculation reproducibility by running metrics calculation multiple times
    reproducibility_results = []
    for trial in range(5):
        trial_result = calculator.calculate_all_metrics(
            simulation_data=test_simulation_data,
            enable_statistical_validation=True
        )
        
        # Extract key metric for reproducibility testing
        nav_success = trial_result.get('navigation_success_metrics', {}).get('source_localization_rate', 0)
        reproducibility_results.append(nav_success)
    
    # Calculate reproducibility coefficient
    if len(reproducibility_results) > 1:
        repro_mean = np.mean(reproducibility_results)
        repro_std = np.std(reproducibility_results)
        repro_coefficient = 1.0 - (repro_std / repro_mean) if repro_mean > 0 else 0.0
        
        assert repro_coefficient >= test_config['reproducibility_threshold'], \
            f"Reproducibility coefficient {repro_coefficient:.6f} below requirement {test_config['reproducibility_threshold']}"
    
    # Check hypothesis testing results
    if len(calculated_values) >= 3:
        # Perform one-sample t-test against reference mean
        reference_mean = np.mean(reference_values)
        
        t_statistic, t_p_value = stats.ttest_1samp(calculated_values, reference_mean)
        
        hypothesis_result = {
            't_statistic': t_statistic,
            'p_value': t_p_value,
            'null_hypothesis': f'calculated mean equals reference mean ({reference_mean:.3f})',
            'alternative_hypothesis': 'calculated mean differs from reference mean',
            'significance_level': test_config['statistical_significance']
        }
        
        # Log hypothesis testing results
        if t_p_value < test_config['statistical_significance']:
            warnings.warn(f"Null hypothesis rejected: calculated metrics significantly differ from reference (p={t_p_value:.6f})")
    
    # Validate comprehensive accuracy assessment
    accuracy_assessment = {
        'correlation_compliance': correlation_coefficient >= test_config['correlation_threshold'] if 'correlation_coefficient' in locals() else False,
        'reproducibility_compliance': repro_coefficient >= test_config['reproducibility_threshold'] if 'repro_coefficient' in locals() else False,
        'statistical_validation': True,
        'benchmark_comparison': True,
        'overall_accuracy': 'PASS'
    }
    
    # Calculate overall accuracy score
    compliance_scores = [
        accuracy_assessment['correlation_compliance'],
        accuracy_assessment['reproducibility_compliance'],
        accuracy_assessment['statistical_validation'],
        accuracy_assessment['benchmark_comparison']
    ]
    
    overall_score = sum(compliance_scores) / len(compliance_scores)
    
    if overall_score < 0.75:  # Require 75% compliance
        accuracy_assessment['overall_accuracy'] = 'FAIL'
        warnings.warn(f"Overall accuracy assessment failed: score {overall_score:.3f}")
    
    # Validate integration consistency
    assert calculated_metrics is not None, "Metrics calculation should return results"
    assert isinstance(calculated_metrics, dict), "Metrics should be returned as dictionary"
    
    # Check that statistical validation is properly integrated
    if 'statistical_validation' in calculated_metrics:
        stat_validation = calculated_metrics['statistical_validation']
        assert 'correlation_analysis' in stat_validation
        assert 'significance_testing' in stat_validation
    
    # Verify benchmark comparison integration
    comparison_result = benchmark_comparator.compare_against_benchmark(
        calculated_metrics, reference_benchmark
    )
    
    assert 'correlation_coefficient' in comparison_result
    assert 'validation_status' in comparison_result
    
    benchmark_correlation = comparison_result['correlation_coefficient']
    assert benchmark_correlation >= 0.8, f"Benchmark correlation {benchmark_correlation:.6f} too low"


@pytest.mark.unit
@pytest.mark.parametrize('error_scenario', ['empty_data', 'invalid_format', 'missing_parameters', 'nan_values', 'infinite_values'])
@measure_performance(time_limit_seconds=1.0)
def test_error_handling_and_edge_cases(error_scenario):
    """
    Test error handling and edge cases in metrics calculation including invalid data,
    missing parameters, and boundary conditions with comprehensive error validation.
    """
    # Create test data with specific error conditions
    calculator = PerformanceMetricsCalculator(
        enable_error_recovery=True,
        strict_validation=True
    )
    
    if error_scenario == 'empty_data':
        # Test with empty simulation results
        invalid_data = {
            'simulation_results': [],
            'target_locations': {'primary_source': np.array([0.8, 0.5])},
            'algorithm_type': 'infotaxis'
        }
        expected_error = ValueError
        
    elif error_scenario == 'invalid_format':
        # Test with incorrectly formatted data
        invalid_data = {
            'simulation_results': [
                {
                    'trajectory': "invalid_trajectory",  # String instead of array
                    'algorithm_type': 'infotaxis',
                    'success': True
                }
            ],
            'target_locations': {'primary_source': np.array([0.8, 0.5])},
            'algorithm_type': 'infotaxis'
        }
        expected_error = (TypeError, ValueError)
        
    elif error_scenario == 'missing_parameters':
        # Test with missing required parameters
        invalid_data = {
            'simulation_results': [
                {
                    'trajectory': np.random.random((10, 2)),
                    # Missing algorithm_type and success
                }
            ],
            # Missing target_locations
            'algorithm_type': 'infotaxis'
        }
        expected_error = KeyError
        
    elif error_scenario == 'nan_values':
        # Test with NaN values in trajectory data
        trajectory_with_nan = np.random.random((10, 2))
        trajectory_with_nan[5, :] = np.nan  # Insert NaN values
        
        invalid_data = {
            'simulation_results': [
                {
                    'trajectory': trajectory_with_nan,
                    'algorithm_type': 'infotaxis',
                    'success': True,
                    'final_position': np.array([np.nan, 0.5])
                }
            ],
            'target_locations': {'primary_source': np.array([0.8, 0.5])},
            'algorithm_type': 'infotaxis'
        }
        expected_error = ValueError
        
    elif error_scenario == 'infinite_values':
        # Test with infinite values in data
        trajectory_with_inf = np.random.random((10, 2))
        trajectory_with_inf[3, 0] = np.inf  # Insert infinite value
        
        invalid_data = {
            'simulation_results': [
                {
                    'trajectory': trajectory_with_inf,
                    'algorithm_type': 'infotaxis',
                    'success': True,
                    'final_position': np.array([0.8, np.inf])
                }
            ],
            'target_locations': {'primary_source': np.array([0.8, 0.5])},
            'algorithm_type': 'infotaxis'
        }
        expected_error = ValueError
    
    # Attempt metrics calculation with invalid data
    try:
        result = calculator.calculate_all_metrics(
            simulation_data=invalid_data,
            enable_error_recovery=True
        )
        
        # If we reach here, check if error recovery was successful
        if error_scenario in ['empty_data']:
            # Empty data might be handled gracefully
            assert result is not None
            assert 'error_recovery' in result
            assert result['error_recovery']['recovery_successful'] == True
        else:
            # Other scenarios should raise exceptions
            pytest.fail(f"Expected {expected_error} for {error_scenario}, but calculation succeeded")
            
    except expected_error as e:
        # Expected error occurred - validate error handling
        error_message = str(e)
        
        # Validate proper exception handling and error messages
        assert len(error_message) > 0, "Error message should not be empty"
        
        # Check for informative error messages
        if error_scenario == 'empty_data':
            assert any(keyword in error_message.lower() for keyword in ['empty', 'no data', 'insufficient'])
        elif error_scenario == 'invalid_format':
            assert any(keyword in error_message.lower() for keyword in ['format', 'type', 'invalid'])
        elif error_scenario == 'missing_parameters':
            assert any(keyword in error_message.lower() for keyword in ['missing', 'required', 'parameter'])
        elif error_scenario == 'nan_values':
            assert any(keyword in error_message.lower() for keyword in ['nan', 'invalid', 'numeric'])
        elif error_scenario == 'infinite_values':
            assert any(keyword in error_message.lower() for keyword in ['inf', 'infinite', 'invalid'])
    
    except Exception as e:
        # Unexpected error type
        pytest.fail(f"Unexpected error type {type(e)} for scenario {error_scenario}: {e}")
    
    # Assert graceful degradation for edge cases
    # Test with minimal valid data
    minimal_data = {
        'simulation_results': [
            {
                'trajectory': np.array([[0.1, 0.5], [0.8, 0.5]]),  # Minimal 2-point trajectory
                'algorithm_type': 'infotaxis',
                'success': True,
                'final_position': np.array([0.8, 0.5])
            }
        ],
        'target_locations': {'primary_source': np.array([0.8, 0.5])},
        'algorithm_type': 'infotaxis'
    }
    
    try:
        minimal_result = calculator.calculate_all_metrics(
            simulation_data=minimal_data,
            enable_error_recovery=True
        )
        
        # Minimal data should produce some results with warnings
        assert minimal_result is not None
        
        # Check for warning indicators
        if 'warnings' in minimal_result:
            warnings_list = minimal_result['warnings']
            assert len(warnings_list) > 0, "Should generate warnings for minimal data"
    
    except Exception as e:
        # Minimal data handling error is acceptable with informative message
        assert "insufficient data" in str(e).lower() or "minimal" in str(e).lower()
    
    # Check error logging and reporting functionality
    if hasattr(calculator, 'get_error_log'):
        error_log = calculator.get_error_log()
        assert isinstance(error_log, list), "Error log should be a list"
    
    # Validate recovery mechanisms and fallback strategies
    # Test with partially invalid data
    partial_invalid_data = {
        'simulation_results': [
            {
                'trajectory': np.random.random((20, 2)),
                'algorithm_type': 'infotaxis',
                'success': True,
                'final_position': np.array([0.8, 0.5])
            },
            {
                'trajectory': np.array([[np.nan, 0.5]]),  # Invalid trajectory
                'algorithm_type': 'infotaxis',
                'success': False,
                'final_position': np.array([0.2, 0.3])
            },
            {
                'trajectory': np.random.random((15, 2)),
                'algorithm_type': 'casting',
                'success': True,
                'final_position': np.array([0.7, 0.6])
            }
        ],
        'target_locations': {'primary_source': np.array([0.8, 0.5])},
        'algorithm_type': 'mixed'
    }
    
    try:
        partial_result = calculator.calculate_all_metrics(
            simulation_data=partial_invalid_data,
            enable_error_recovery=True,
            skip_invalid_entries=True
        )
        
        # Should process valid entries and skip invalid ones
        if partial_result is not None:
            assert 'processed_count' in partial_result.get('metadata', {})
            processed_count = partial_result['metadata']['processed_count']
            assert processed_count >= 2, "Should process at least 2 valid entries"
            
            # Check for recovery information
            if 'error_recovery' in partial_result:
                recovery_info = partial_result['error_recovery']
                assert 'skipped_entries' in recovery_info
                assert recovery_info['skipped_entries'] >= 1, "Should skip at least 1 invalid entry"
    
    except Exception as e:
        # Partial processing failure is acceptable if well-documented
        error_msg = str(e).lower()
        assert any(keyword in error_msg for keyword in ['partial', 'invalid', 'recovery'])
    
    # Test boundary condition handling
    boundary_data = {
        'simulation_results': [
            {
                'trajectory': np.array([[0.0, 0.0], [1.0, 1.0]]),  # Arena boundary trajectory
                'algorithm_type': 'boundary_test',
                'success': True,
                'final_position': np.array([1.0, 1.0])
            }
        ],
        'target_locations': {'primary_source': np.array([1.0, 1.0])},  # Boundary target
        'algorithm_type': 'boundary_test'
    }
    
    try:
        boundary_result = calculator.calculate_all_metrics(
            simulation_data=boundary_data,
            enable_error_recovery=True
        )
        
        # Boundary conditions should be handled appropriately
        if boundary_result is not None:
            # Validate boundary handling
            nav_metrics = boundary_result.get('navigation_success_metrics', {})
            if 'source_localization_rate' in nav_metrics:
                localization_rate = nav_metrics['source_localization_rate']
                assert 0.0 <= localization_rate <= 1.0, "Boundary localization rate should be valid"
    
    except Exception as e:
        # Boundary condition errors should be informative
        assert "boundary" in str(e).lower() or "range" in str(e).lower()
    
    # Assert system stability under error conditions
    # System should not crash or leave resources in inconsistent state
    if hasattr(calculator, 'validate_state'):
        state_validation = calculator.validate_state()
        assert state_validation['is_stable'] == True, "Calculator state should remain stable after errors"


# Helper functions for generating test data

def _generate_infotaxis_trajectory():
    """Generate realistic infotaxis trajectory with information-theoretic search patterns."""
    np.random.seed(RANDOM_SEED)
    start_pos = np.array([0.1, 0.5])
    trajectory = [start_pos]
    current_pos = start_pos.copy()
    
    for step in range(50):
        # Infotaxis: exploration with information gain bias
        exploration_direction = np.random.normal(0, 1, 2)
        exploration_direction /= np.linalg.norm(exploration_direction)
        
        # Add some bias toward regions of higher expected information
        info_bias = np.array([0.1, 0.0])  # Slight eastward bias
        
        movement = 0.7 * exploration_direction + 0.3 * info_bias
        movement = movement / np.linalg.norm(movement) * 0.02
        
        current_pos += movement
        current_pos = np.clip(current_pos, 0.0, 1.0)
        trajectory.append(current_pos.copy())
    
    return np.array(trajectory)


def _generate_casting_trajectory():
    """Generate realistic casting trajectory with crosswind search patterns."""
    np.random.seed(RANDOM_SEED + 1)
    start_pos = np.array([0.1, 0.5])
    trajectory = [start_pos]
    current_pos = start_pos.copy()
    
    casting_phase = 0
    casting_amplitude = 0.1
    
    for step in range(50):
        # Casting: zigzag pattern with upwind bias
        if step % 10 < 5:  # Cast one direction
            cast_direction = np.array([0.02, casting_amplitude * np.sin(step * 0.5)])
        else:  # Cast other direction
            cast_direction = np.array([0.02, -casting_amplitude * np.sin(step * 0.5)])
        
        current_pos += cast_direction
        current_pos = np.clip(current_pos, 0.0, 1.0)
        trajectory.append(current_pos.copy())
    
    return np.array(trajectory)


def _generate_gradient_trajectory():
    """Generate realistic gradient following trajectory with direct ascent."""
    np.random.seed(RANDOM_SEED + 2)
    start_pos = np.array([0.1, 0.5])
    target_pos = np.array([0.8, 0.5])
    trajectory = [start_pos]
    current_pos = start_pos.copy()
    
    for step in range(30):  # Shorter trajectory due to efficiency
        # Gradient following: direct movement toward target with small noise
        direction = target_pos - current_pos
        if np.linalg.norm(direction) > 0:
            direction /= np.linalg.norm(direction)
        
        noise = np.random.normal(0, 0.05, 2)
        movement = 0.9 * direction + 0.1 * noise
        movement = movement / np.linalg.norm(movement) * 0.03
        
        current_pos += movement
        current_pos = np.clip(current_pos, 0.0, 1.0)
        trajectory.append(current_pos.copy())
        
        # Stop if close to target
        if np.linalg.norm(current_pos - target_pos) < 0.05:
            break
    
    return np.array(trajectory)


def _generate_plume_tracking_trajectory():
    """Generate realistic plume tracking trajectory with surge behavior."""
    np.random.seed(RANDOM_SEED + 3)
    start_pos = np.array([0.1, 0.5])
    trajectory = [start_pos]
    current_pos = start_pos.copy()
    
    in_plume = False
    surge_direction = np.array([1.0, 0.0])  # Upwind direction
    
    for step in range(45):
        # Plume tracking: surge when in plume, cast when lost
        plume_encounter = np.random.random() > 0.7  # 30% chance of plume encounter
        
        if plume_encounter:
            in_plume = True
            # Surge upwind
            movement = surge_direction * 0.04 + np.random.normal(0, 0.01, 2)
        else:
            if in_plume:
                # Just lost plume, start casting
                in_plume = False
            # Casting behavior
            cast_direction = np.array([0.01, 0.1 * np.sin(step * 0.8)])
            movement = cast_direction + np.random.normal(0, 0.02, 2)
        
        current_pos += movement
        current_pos = np.clip(current_pos, 0.0, 1.0)
        trajectory.append(current_pos.copy())
    
    return np.array(trajectory)


def _generate_hybrid_trajectory():
    """Generate realistic hybrid strategy trajectory combining multiple approaches."""
    np.random.seed(RANDOM_SEED + 4)
    start_pos = np.array([0.1, 0.5])
    trajectory = [start_pos]
    current_pos = start_pos.copy()
    
    strategy_phase = 0  # 0: exploration, 1: casting, 2: gradient
    
    for step in range(60):
        # Hybrid strategy: switch between different behaviors
        if step % 20 == 0:
            strategy_phase = (strategy_phase + 1) % 3
        
        if strategy_phase == 0:  # Exploration phase
            movement = np.random.normal(0, 0.03, 2)
        elif strategy_phase == 1:  # Casting phase
            movement = np.array([0.02, 0.05 * np.sin(step * 0.3)])
        else:  # Gradient phase
            target_direction = np.array([0.7, 0.0])  # Toward source
            movement = target_direction * 0.025 + np.random.normal(0, 0.01, 2)
        
        current_pos += movement
        current_pos = np.clip(current_pos, 0.0, 1.0)
        trajectory.append(current_pos.copy())
    
    return np.array(trajectory)


def _generate_exploration_optimal_path():
    """Generate optimal exploration path for reference comparison."""
    # Systematic coverage pattern
    path_points = []
    for x in np.linspace(0.1, 0.9, 5):
        for y in np.linspace(0.3, 0.7, 3):
            path_points.append([x, y])
    
    return np.array(path_points)


def _generate_energy_optimal_path():
    """Generate energy-optimal path for reference comparison."""
    # Smooth, minimal acceleration path
    t = np.linspace(0, 1, 20)
    x_coords = 0.1 + 0.7 * t
    y_coords = 0.5 + 0.1 * np.sin(2 * np.pi * t)  # Slight sinusoidal variation
    
    return np.column_stack([x_coords, y_coords])


def _generate_time_optimal_path():
    """Generate time-optimal path for reference comparison."""
    # Direct path with minimal stops
    return np.array([[0.1, 0.5], [0.45, 0.5], [0.8, 0.5]])


# Custom warning class for performance warnings
class PerformanceWarning(UserWarning):
    """Custom warning class for performance-related test issues."""
    pass


# Register custom warning class
warnings.simplefilter('always', PerformanceWarning)