"""
Comprehensive unit test module for statistical validation functionality in the plume navigation simulation system.

This module implements comprehensive testing of statistical comparison methods, correlation analysis, hypothesis testing,
reproducibility assessment, cross-format compatibility validation, and scientific computing standards compliance.
Validates >95% correlation requirements, >0.99 reproducibility coefficients, and statistical significance testing
with comprehensive error handling and performance validation for scientific research reproducibility.

Key Test Coverage:
- Statistical comparison framework validation with multiple algorithm types
- Simulation accuracy validation testing against >95% correlation thresholds
- Result reproducibility assessment testing with >0.99 coefficient targets
- Cross-platform statistical validation testing for Crimaldi and custom formats
- Performance analysis and metrics engine testing within <7.2 second targets
- Error handling and quality assurance testing with comprehensive edge cases
- Bootstrap analysis and statistical power testing for robust inference
- Algorithm ranking and statistical assumption validation testing
- Caching and performance optimization validation
- Integration testing with reference benchmark data
"""

# External library imports with version specifications
import pytest  # pytest 8.3.5+ - Testing framework for unit test execution and fixture management
import numpy as np  # numpy 2.1.3+ - Numerical computations and array operations for statistical test validation
import scipy.stats  # scipy 1.15.3+ - Statistical analysis and hypothesis testing validation
import pandas as pd  # pandas 2.2.0+ - Data manipulation for statistical test data processing
from unittest.mock import Mock, patch, MagicMock, call  # unittest.mock 3.9+ - Mock objects for isolated unit testing
import warnings  # warnings 3.9+ - Warning management for statistical edge case testing
import pathlib  # pathlib 3.9+ - Path handling for test fixture access
import datetime  # datetime 3.9+ - Timestamp handling for test metadata
import time  # time 3.9+ - Performance timing for test validation
import json  # json 3.9+ - JSON data handling for test configuration
import math  # math 3.9+ - Mathematical operations for statistical calculations
import uuid  # uuid 3.9+ - Unique identifier generation for test correlation
import threading  # threading 3.9+ - Thread-safe operations for concurrent testing
import contextlib  # contextlib 3.9+ - Context manager utilities for test resource management
from typing import Dict, List, Any, Tuple, Optional, Union  # typing 3.9+ - Type hints for test function signatures

# Internal imports from backend analysis modules
from src.backend.core.analysis.statistical_comparison import (
    StatisticalComparator,
    compare_algorithm_performance,
    validate_cross_format_consistency,
    assess_simulation_reproducibility
)
from src.backend.utils.statistical_utils import (
    StatisticalAnalyzer,
    calculate_correlation_matrix,
    perform_hypothesis_testing,
    assess_reproducibility
)
from src.test.utils.validation_metrics import (
    ValidationMetricsCalculator,
    StatisticalValidator
)
from src.test.utils.test_helpers import (
    assert_arrays_almost_equal,
    assert_simulation_accuracy,
    measure_performance,
    create_mock_video_data,
    validate_cross_format_compatibility,
    setup_test_environment,
    validate_batch_processing_results,
    compare_algorithm_performance as compare_perf_helper,
    TestDataValidator,
    PerformanceProfiler
)

# Global test configuration constants
CORRELATION_THRESHOLD_95_PERCENT = 0.95
REPRODUCIBILITY_THRESHOLD_99_PERCENT = 0.99
NUMERICAL_TOLERANCE = 1e-6
STATISTICAL_SIGNIFICANCE_LEVEL = 0.05
TEST_RANDOM_SEED = 42

# Test fixture and mock data configuration
TEST_ALGORITHM_TYPES = ['infotaxis', 'casting', 'gradient_following', 'hybrid']
TEST_FORMAT_TYPES = ['crimaldi', 'custom', 'avi']
TEST_CORRELATION_METHODS = ['pearson', 'spearman', 'kendall']
TEST_HYPOTHESIS_TYPES = ['ttest_ind', 'mannwhitneyu', 'kruskal']
TEST_ICC_TYPES = ['ICC(2,1)', 'ICC(3,1)']

# Performance testing constants
MAX_PROCESSING_TIME_SECONDS = 7.2
MAX_MEMORY_USAGE_MB = 1000
BENCHMARK_SIMULATION_COUNT = 100
BATCH_TARGET_SIMULATIONS = 4000


class TestStatisticalComparatorInitialization:
    """Test suite for StatisticalComparator class initialization with various configuration parameters."""
    
    def setup_method(self):
        """Set up test fixtures and configuration for StatisticalComparator initialization tests."""
        np.random.seed(TEST_RANDOM_SEED)
        self.test_config = {
            'correlation_threshold': CORRELATION_THRESHOLD_95_PERCENT,
            'reproducibility_threshold': REPRODUCIBILITY_THRESHOLD_99_PERCENT,
            'significance_level': STATISTICAL_SIGNIFICANCE_LEVEL,
            'numerical_tolerance': NUMERICAL_TOLERANCE
        }
    
    def test_statistical_comparator_default_initialization(self):
        """Test StatisticalComparator initialization with default configuration parameters."""
        # Initialize StatisticalComparator with default configuration
        comparator = StatisticalComparator()
        
        # Validate default configuration setup
        assert hasattr(comparator, 'correlation_threshold')
        assert hasattr(comparator, 'significance_level')
        assert hasattr(comparator, 'numerical_tolerance')
        
        # Verify default threshold values meet scientific requirements
        assert comparator.correlation_threshold >= CORRELATION_THRESHOLD_95_PERCENT
        assert comparator.significance_level == STATISTICAL_SIGNIFICANCE_LEVEL
        assert comparator.numerical_tolerance <= NUMERICAL_TOLERANCE
        
        # Assert proper initialization of internal components
        assert hasattr(comparator, 'statistical_analyzer')
        assert hasattr(comparator, 'validation_metrics')
        assert hasattr(comparator, 'performance_tracker')
    
    def test_statistical_comparator_custom_configuration(self):
        """Test StatisticalComparator initialization with custom significance level and validation settings."""
        custom_significance = 0.01
        custom_config = self.test_config.copy()
        custom_config['significance_level'] = custom_significance
        
        # Initialize with custom configuration
        comparator = StatisticalComparator(config=custom_config)
        
        # Test initialization with custom significance level
        assert comparator.significance_level == custom_significance
        assert comparator.correlation_threshold == CORRELATION_THRESHOLD_95_PERCENT
        
        # Validate configuration parameter validation
        assert comparator.config['correlation_threshold'] == CORRELATION_THRESHOLD_95_PERCENT
        assert comparator.config['reproducibility_threshold'] == REPRODUCIBILITY_THRESHOLD_99_PERCENT
    
    def test_statistical_comparator_caching_configuration(self):
        """Test StatisticalComparator initialization with caching enabled and disabled."""
        # Test initialization with caching enabled
        comparator_cached = StatisticalComparator(enable_caching=True)
        assert comparator_cached.caching_enabled is True
        assert hasattr(comparator_cached, 'result_cache')
        
        # Test initialization with caching disabled
        comparator_no_cache = StatisticalComparator(enable_caching=False)
        assert comparator_no_cache.caching_enabled is False
        
        # Verify logging configuration setup
        assert hasattr(comparator_cached, 'logger')
        assert hasattr(comparator_no_cache, 'logger')
    
    def test_statistical_comparator_validation_edge_cases(self):
        """Test StatisticalComparator initialization with invalid configuration parameters."""
        # Test invalid correlation threshold (> 1.0)
        with pytest.raises(ValueError, match="Correlation threshold must be between 0 and 1"):
            StatisticalComparator(config={'correlation_threshold': 1.5})
        
        # Test invalid significance level (negative)
        with pytest.raises(ValueError, match="Significance level must be positive"):
            StatisticalComparator(config={'significance_level': -0.05})
        
        # Test invalid numerical tolerance (zero)
        with pytest.raises(ValueError, match="Numerical tolerance must be positive"):
            StatisticalComparator(config={'numerical_tolerance': 0.0})


class TestAlgorithmPerformanceComparison:
    """Test suite for comprehensive algorithm performance comparison functionality."""
    
    def setup_method(self):
        """Set up test fixtures and mock algorithm results for performance comparison tests."""
        np.random.seed(TEST_RANDOM_SEED)
        self.comparator = StatisticalComparator()
        
        # Create realistic mock algorithm results with performance metrics
        self.mock_algorithm_results = self._create_mock_algorithm_results()
        self.comparison_metrics = ['success_rate', 'path_efficiency', 'convergence_time', 'error_rate']
    
    def _create_mock_algorithm_results(self) -> Dict[str, Dict[str, Any]]:
        """Create mock algorithm results with realistic performance data and statistical properties."""
        results = {}
        
        for i, algorithm in enumerate(TEST_ALGORITHM_TYPES):
            # Generate performance data with known statistical properties
            base_success_rate = 0.8 + i * 0.05
            base_efficiency = 0.7 + i * 0.03
            base_convergence = 10.0 - i * 1.0
            base_error = 0.05 - i * 0.01
            
            # Create sample data for statistical testing
            n_samples = 50
            results[algorithm] = {
                'success_rate': base_success_rate,
                'path_efficiency': base_efficiency,
                'convergence_time': base_convergence,
                'error_rate': max(0.001, base_error),
                'success_rate_samples': np.random.normal(base_success_rate, 0.1, n_samples),
                'path_efficiency_samples': np.random.normal(base_efficiency, 0.05, n_samples),
                'convergence_time_samples': np.random.gamma(2, base_convergence/2, n_samples),
                'error_rate_samples': np.random.exponential(base_error, n_samples)
            }
        
        return results
    
    @pytest.mark.parametrize('algorithm_count', [2, 3, 5])
    def test_algorithm_performance_comparison_multiple_algorithms(self, algorithm_count):
        """Test comprehensive algorithm performance comparison with varying algorithm counts."""
        # Select subset of algorithms for testing
        selected_algorithms = dict(list(self.mock_algorithm_results.items())[:algorithm_count])
        
        # Execute compare_algorithm_performance function
        comparison_result = compare_algorithm_performance(
            algorithm_results=selected_algorithms,
            comparison_metrics=self.comparison_metrics,
            significance_level=STATISTICAL_SIGNIFICANCE_LEVEL
        )
        
        # Validate statistical test results and p-values
        assert 'statistical_tests' in comparison_result
        assert 'performance_rankings' in comparison_result
        assert 'effect_sizes' in comparison_result
        
        # Verify algorithm ranking generation and confidence intervals
        rankings = comparison_result['performance_rankings']
        assert len(rankings) == algorithm_count
        assert all('rank' in ranking for ranking in rankings)
        assert all('confidence_interval' in ranking for ranking in rankings)
        
        # Test multiple hypothesis correction application
        assert 'multiple_comparison_correction' in comparison_result
        assert comparison_result['multiple_comparison_correction']['method'] in ['bonferroni', 'fdr_bh']
        
        # Validate comprehensive comparison report structure
        assert 'comparison_summary' in comparison_result
        assert 'statistical_significance_tests' in comparison_result
        assert 'recommendation_report' in comparison_result
    
    def test_algorithm_performance_statistical_significance(self):
        """Test statistical significance testing and effect size calculation in algorithm comparison."""
        # Execute performance comparison with statistical analysis
        comparison_result = compare_algorithm_performance(
            algorithm_results=self.mock_algorithm_results,
            comparison_metrics=['success_rate', 'path_efficiency'],
            significance_level=STATISTICAL_SIGNIFICANCE_LEVEL
        )
        
        # Assert effect size calculations are within expected ranges
        effect_sizes = comparison_result['effect_sizes']
        for metric in ['success_rate', 'path_efficiency']:
            assert metric in effect_sizes
            # Cohen's d effect sizes should be reasonable
            for algorithm_pair, effect_size in effect_sizes[metric].items():
                assert -3.0 <= effect_size <= 3.0  # Reasonable effect size range
        
        # Validate statistical test p-values
        statistical_tests = comparison_result['statistical_tests']
        for test_result in statistical_tests.values():
            assert 'p_value' in test_result
            assert 0.0 <= test_result['p_value'] <= 1.0
            assert 'test_statistic' in test_result
            assert 'degrees_of_freedom' in test_result
    
    @measure_performance(time_limit_seconds=MAX_PROCESSING_TIME_SECONDS)
    def test_algorithm_performance_comparison_performance(self):
        """Test algorithm performance comparison within processing time limits."""
        start_time = time.time()
        
        # Execute comprehensive performance comparison
        comparison_result = compare_algorithm_performance(
            algorithm_results=self.mock_algorithm_results,
            comparison_metrics=self.comparison_metrics,
            significance_level=STATISTICAL_SIGNIFICANCE_LEVEL,
            include_bootstrap_analysis=True,
            bootstrap_iterations=1000
        )
        
        execution_time = time.time() - start_time
        
        # Validate execution time meets performance requirements
        assert execution_time <= MAX_PROCESSING_TIME_SECONDS
        
        # Verify comprehensive analysis was completed within time limit
        assert 'bootstrap_confidence_intervals' in comparison_result
        assert 'performance_rankings' in comparison_result
        assert len(comparison_result['performance_rankings']) == len(self.mock_algorithm_results)


class TestCorrelationMatrixCalculation:
    """Test suite for correlation matrix calculation with multiple correlation methods."""
    
    def setup_method(self):
        """Set up test fixtures and synthetic data for correlation matrix testing."""
        np.random.seed(TEST_RANDOM_SEED)
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Generate synthetic test data matrix with known correlations
        self.test_data_matrix = self._generate_correlated_data()
        self.correlation_methods = TEST_CORRELATION_METHODS
    
    def _generate_correlated_data(self) -> np.ndarray:
        """Generate synthetic test data matrix with known correlation structure."""
        n_samples = 100
        n_variables = 5
        
        # Create base data with controlled correlations
        base_data = np.random.randn(n_samples, n_variables)
        
        # Introduce known correlations between variables
        correlation_matrix = np.array([
            [1.0, 0.8, 0.6, 0.3, 0.1],
            [0.8, 1.0, 0.7, 0.4, 0.2],
            [0.6, 0.7, 1.0, 0.5, 0.3],
            [0.3, 0.4, 0.5, 1.0, 0.6],
            [0.1, 0.2, 0.3, 0.6, 1.0]
        ])
        
        # Apply Cholesky decomposition to create correlated data
        L = np.linalg.cholesky(correlation_matrix)
        correlated_data = base_data @ L.T
        
        return correlated_data
    
    @pytest.mark.parametrize('correlation_method', ['pearson', 'spearman', 'kendall'])
    def test_correlation_matrix_calculation_methods(self, correlation_method):
        """Test correlation matrix calculation using different correlation methods."""
        # Calculate correlation matrix using specified method
        correlation_result = calculate_correlation_matrix(
            data_matrix=self.test_data_matrix,
            method=correlation_method,
            significance_testing=True
        )
        
        # Validate correlation coefficients against expected values
        correlation_matrix = correlation_result['correlation_matrix']
        assert correlation_matrix.shape == (5, 5)
        
        # Check diagonal elements are 1.0 (self-correlation)
        np.testing.assert_array_almost_equal(np.diag(correlation_matrix), np.ones(5), decimal=6)
        
        # Verify correlation matrix is symmetric
        np.testing.assert_array_almost_equal(correlation_matrix, correlation_matrix.T, decimal=6)
        
        # Test statistical significance calculation and p-values
        p_values = correlation_result['p_values']
        assert p_values.shape == (5, 5)
        assert np.all(p_values >= 0.0) and np.all(p_values <= 1.0)
        
        # Assert confidence interval generation functionality
        if 'confidence_intervals' in correlation_result:
            ci_lower = correlation_result['confidence_intervals']['lower']
            ci_upper = correlation_result['confidence_intervals']['upper']
            assert np.all(ci_lower <= correlation_matrix)
            assert np.all(correlation_matrix <= ci_upper)
    
    def test_correlation_threshold_validation(self):
        """Test correlation matrix validation against >95% correlation threshold."""
        # Calculate correlation matrix
        correlation_result = calculate_correlation_matrix(
            data_matrix=self.test_data_matrix,
            method='pearson',
            threshold_validation=True,
            correlation_threshold=CORRELATION_THRESHOLD_95_PERCENT
        )
        
        # Verify >95% correlation threshold validation
        threshold_validation = correlation_result['threshold_validation']
        assert 'correlation_threshold' in threshold_validation
        assert threshold_validation['correlation_threshold'] == CORRELATION_THRESHOLD_95_PERCENT
        
        # Check for high correlation pairs
        high_correlations = threshold_validation['high_correlation_pairs']
        correlation_matrix = correlation_result['correlation_matrix']
        
        # Verify identification of correlations above threshold
        for pair_info in high_correlations:
            i, j = pair_info['indices']
            correlation_value = correlation_matrix[i, j]
            if i != j:  # Exclude diagonal elements
                assert abs(correlation_value) >= CORRELATION_THRESHOLD_95_PERCENT
    
    def test_correlation_edge_cases(self):
        """Test correlation matrix calculation with edge cases including perfect and zero correlations."""
        # Test with perfect correlation data
        perfect_corr_data = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]).T
        
        correlation_result = calculate_correlation_matrix(
            data_matrix=perfect_corr_data,
            method='pearson',
            handle_edge_cases=True
        )
        
        # Perfect correlation should be handled gracefully
        assert 'edge_case_handling' in correlation_result
        assert correlation_result['edge_case_handling']['perfect_correlations_detected']
        
        # Test with zero correlation data
        zero_corr_data = np.random.randn(50, 3)
        zero_corr_data[:, 1] = np.random.randn(50)  # Independent variable
        
        correlation_result_zero = calculate_correlation_matrix(
            data_matrix=zero_corr_data,
            method='pearson',
            handle_edge_cases=True
        )
        
        # Verify handling of near-zero correlations
        correlation_matrix = correlation_result_zero['correlation_matrix']
        assert np.all(np.abs(correlation_matrix) <= 1.0)


class TestHypothesisTestingFunctionality:
    """Test suite for comprehensive hypothesis testing with various test types."""
    
    def setup_method(self):
        """Set up test fixtures and group data for hypothesis testing validation."""
        np.random.seed(TEST_RANDOM_SEED)
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Create mock group data with known statistical properties
        self.group_data = self._create_mock_group_data()
    
    def _create_mock_group_data(self) -> Dict[str, np.ndarray]:
        """Create mock group data with known statistical properties for hypothesis testing."""
        # Group 1: Normal distribution with mean=10, std=2
        group1 = np.random.normal(10.0, 2.0, 50)
        
        # Group 2: Normal distribution with mean=12, std=2.5 (different from group1)
        group2 = np.random.normal(12.0, 2.5, 45)
        
        # Group 3: Non-normal distribution (gamma) for non-parametric tests
        group3 = np.random.gamma(2.0, 2.0, 40)
        
        return {
            'group1': group1,
            'group2': group2,
            'group3': group3
        }
    
    @pytest.mark.parametrize('test_type', ['ttest_ind', 'mannwhitneyu', 'kruskal'])
    def test_hypothesis_testing_various_types(self, test_type):
        """Test comprehensive hypothesis testing with different statistical test types."""
        # Execute hypothesis testing with specified test type
        if test_type == 'kruskal':
            # Kruskal-Wallis test for multiple groups
            test_result = perform_hypothesis_testing(
                group_data=self.group_data,
                test_type=test_type,
                significance_level=STATISTICAL_SIGNIFICANCE_LEVEL
            )
        else:
            # Two-sample tests
            test_result = perform_hypothesis_testing(
                group_data={'group1': self.group_data['group1'], 'group2': self.group_data['group2']},
                test_type=test_type,
                significance_level=STATISTICAL_SIGNIFICANCE_LEVEL
            )
        
        # Validate test statistic calculation and p-value accuracy
        assert 'test_statistic' in test_result
        assert 'p_value' in test_result
        assert 'degrees_of_freedom' in test_result or test_type == 'mannwhitneyu'
        
        # Verify p-value is in valid range
        assert 0.0 <= test_result['p_value'] <= 1.0
        
        # Test effect size calculation (Cohen's d, eta-squared)
        assert 'effect_size' in test_result
        effect_size_info = test_result['effect_size']
        
        if test_type in ['ttest_ind']:
            assert 'cohens_d' in effect_size_info
            # Cohen's d should be reasonable
            assert -5.0 <= effect_size_info['cohens_d'] <= 5.0
        
        # Assert multiple comparison correction application
        if 'multiple_comparison_correction' in test_result:
            assert 'corrected_p_value' in test_result['multiple_comparison_correction']
            corrected_p = test_result['multiple_comparison_correction']['corrected_p_value']
            assert corrected_p >= test_result['p_value']  # Corrected p should be >= original
        
        # Test statistical significance interpretation
        assert 'interpretation' in test_result
        interpretation = test_result['interpretation']
        assert 'is_significant' in interpretation
        assert interpretation['is_significant'] == (test_result['p_value'] < STATISTICAL_SIGNIFICANCE_LEVEL)
    
    def test_hypothesis_testing_effect_sizes(self):
        """Test effect size calculation and interpretation in hypothesis testing."""
        # Perform t-test with effect size calculation
        test_result = perform_hypothesis_testing(
            group_data={'group1': self.group_data['group1'], 'group2': self.group_data['group2']},
            test_type='ttest_ind',
            calculate_effect_size=True,
            confidence_level=0.95
        )
        
        # Verify confidence interval generation
        assert 'confidence_interval' in test_result
        ci = test_result['confidence_interval']
        assert 'lower_bound' in ci and 'upper_bound' in ci
        assert ci['lower_bound'] <= ci['upper_bound']
        
        # Test effect size interpretation
        effect_size = test_result['effect_size']
        assert 'magnitude' in effect_size
        magnitude = effect_size['magnitude']
        assert magnitude in ['small', 'medium', 'large', 'very_large']
        
        # Validate Cohen's d thresholds
        cohens_d = effect_size['cohens_d']
        if abs(cohens_d) < 0.2:
            assert magnitude == 'small'
        elif abs(cohens_d) < 0.5:
            assert magnitude == 'small'
        elif abs(cohens_d) < 0.8:
            assert magnitude == 'medium'
        else:
            assert magnitude in ['large', 'very_large']
    
    def test_hypothesis_testing_assumptions(self):
        """Test statistical assumption validation in hypothesis testing."""
        # Test normality assumption checking
        test_result = perform_hypothesis_testing(
            group_data={'group1': self.group_data['group1'], 'group2': self.group_data['group2']},
            test_type='ttest_ind',
            check_assumptions=True,
            assumption_tests=['normality', 'equal_variance']
        )
        
        # Verify assumption testing results
        assert 'assumption_tests' in test_result
        assumptions = test_result['assumption_tests']
        
        # Check normality tests
        if 'normality' in assumptions:
            normality_tests = assumptions['normality']
            for group_name, test_results in normality_tests.items():
                assert 'shapiro_wilk' in test_results
                assert 'p_value' in test_results['shapiro_wilk']
                assert 0.0 <= test_results['shapiro_wilk']['p_value'] <= 1.0
        
        # Check equal variance tests
        if 'equal_variance' in assumptions:
            equal_var_test = assumptions['equal_variance']
            assert 'levenes_test' in equal_var_test
            assert 'p_value' in equal_var_test['levenes_test']
        
        # Verify assumption violation reporting
        assert 'assumption_violations' in test_result
        violations = test_result['assumption_violations']
        assert isinstance(violations, list)


class TestReproducibilityAssessment:
    """Test suite for reproducibility assessment functionality with ICC analysis."""
    
    def setup_method(self):
        """Set up test fixtures and repeated measurements for reproducibility testing."""
        np.random.seed(TEST_RANDOM_SEED)
        self.statistical_validator = StatisticalValidator()
        
        # Generate repeated measurements with controlled variability
        self.repeated_measurements = self._generate_repeated_measurements()
        self.environment_factors = {
            'computational_environment': 'testing',
            'random_seed_control': True,
            'hardware_specification': 'standardized',
            'software_versions': 'controlled'
        }
    
    def _generate_repeated_measurements(self) -> List[np.ndarray]:
        """Generate repeated measurements with controlled variability for reproducibility testing."""
        # True underlying signal
        true_signal = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.5
        
        # Multiple measurement repetitions with different noise levels
        measurements = []
        noise_levels = [0.05, 0.08, 0.06, 0.07, 0.05]  # Controlled variability
        
        for noise_level in noise_levels:
            # Add measurement noise
            measurement = true_signal + np.random.normal(0, noise_level, len(true_signal))
            measurements.append(measurement)
        
        return measurements
    
    @pytest.mark.parametrize('icc_type', ['ICC(2,1)', 'ICC(3,1)'])
    def test_reproducibility_assessment_icc_types(self, icc_type):
        """Test reproducibility assessment with different ICC calculation types."""
        # Execute assess_reproducibility function
        reproducibility_result = assess_reproducibility(
            repeated_measurements=self.repeated_measurements,
            icc_type=icc_type,
            confidence_level=0.95,
            environment_factors=self.environment_factors
        )
        
        # Validate ICC calculation accuracy and confidence intervals
        assert 'icc_coefficient' in reproducibility_result
        icc_value = reproducibility_result['icc_coefficient']
        
        # ICC should be between 0 and 1
        assert 0.0 <= icc_value <= 1.0
        
        # Test confidence intervals
        assert 'confidence_interval' in reproducibility_result
        ci = reproducibility_result['confidence_interval']
        assert 'lower_bound' in ci and 'upper_bound' in ci
        assert ci['lower_bound'] <= icc_value <= ci['upper_bound']
        
        # Assert >0.99 reproducibility threshold validation
        threshold_check = reproducibility_result['threshold_validation']
        assert 'reproducibility_threshold' in threshold_check
        assert threshold_check['reproducibility_threshold'] == REPRODUCIBILITY_THRESHOLD_99_PERCENT
        
        meets_threshold = icc_value >= REPRODUCIBILITY_THRESHOLD_99_PERCENT
        assert threshold_check['meets_threshold'] == meets_threshold
    
    def test_reproducibility_variance_decomposition(self):
        """Test variance component analysis and decomposition in reproducibility assessment."""
        # Execute reproducibility assessment with variance decomposition
        reproducibility_result = assess_reproducibility(
            repeated_measurements=self.repeated_measurements,
            icc_type='ICC(2,1)',
            variance_decomposition=True,
            detailed_analysis=True
        )
        
        # Test variance component analysis and decomposition
        assert 'variance_components' in reproducibility_result
        variance_comp = reproducibility_result['variance_components']
        
        # Check variance components
        assert 'between_subjects' in variance_comp
        assert 'within_subjects' in variance_comp
        assert 'measurement_error' in variance_comp
        
        # Variance components should sum to total variance
        total_variance = sum(variance_comp.values())
        assert total_variance > 0
        
        # Verify environmental factor contribution analysis
        if 'environment_analysis' in reproducibility_result:
            env_analysis = reproducibility_result['environment_analysis']
            assert 'environment_factors' in env_analysis
            assert 'factor_contributions' in env_analysis
    
    def test_reproducibility_compliance_reporting(self):
        """Test reproducibility compliance reporting and threshold validation."""
        # Execute reproducibility assessment with compliance reporting
        reproducibility_result = assess_reproducibility(
            repeated_measurements=self.repeated_measurements,
            icc_type='ICC(3,1)',
            compliance_reporting=True,
            threshold_validation=True
        )
        
        # Test reproducibility compliance reporting
        assert 'compliance_report' in reproducibility_result
        compliance = reproducibility_result['compliance_report']
        
        # Check compliance status
        assert 'overall_compliance' in compliance
        assert 'reproducibility_score' in compliance
        assert 'quality_indicators' in compliance
        
        # Verify reproducibility score calculation
        repro_score = compliance['reproducibility_score']
        assert 0.0 <= repro_score <= 1.0
        
        # Check quality indicators
        quality_indicators = compliance['quality_indicators']
        assert 'measurement_consistency' in quality_indicators
        assert 'environmental_stability' in quality_indicators
        assert 'systematic_bias_detected' in quality_indicators


class TestCrossFormatCompatibilityValidation:
    """Test suite for cross-format consistency validation between different plume formats."""
    
    def setup_method(self):
        """Set up test fixtures and mock results for cross-format compatibility testing."""
        np.random.seed(TEST_RANDOM_SEED)
        self.validator = StatisticalValidator()
        
        # Create mock results for both Crimaldi and custom formats
        self.crimaldi_results, self.custom_results = self._create_cross_format_results()
    
    def _create_cross_format_results(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Create mock results for both Crimaldi and custom formats with realistic differences."""
        # Base trajectory data
        n_points = 200
        time_points = np.linspace(0, 10, n_points)
        base_trajectory = np.column_stack([
            np.cumsum(np.random.randn(n_points) * 0.1),  # x coordinates
            np.cumsum(np.random.randn(n_points) * 0.1)   # y coordinates
        ])
        
        # Crimaldi format results with specific characteristics
        crimaldi_results = {
            'trajectory_data': base_trajectory,
            'performance_metrics': {
                'success_rate': 0.85,
                'path_efficiency': 0.72,
                'processing_time': 6.8
            },
            'format_metadata': {
                'pixel_to_meter_ratio': 100.0,
                'frame_rate': 50.0,
                'intensity_range': (0, 255)
            },
            'calibration_parameters': {
                'spatial_calibration': 0.98,
                'temporal_calibration': 0.96,
                'intensity_calibration': 0.94
            }
        }
        
        # Custom format results with slight differences
        custom_results = {
            'trajectory_data': base_trajectory + np.random.normal(0, 0.01, base_trajectory.shape),  # Small differences
            'performance_metrics': {
                'success_rate': 0.83,  # Slightly different
                'path_efficiency': 0.74,  # Slightly different
                'processing_time': 7.1  # Slightly different
            },
            'format_metadata': {
                'pixel_to_meter_ratio': 150.0,  # Different calibration
                'frame_rate': 30.0,  # Different frame rate
                'intensity_range': (0, 65535)  # Different bit depth
            },
            'calibration_parameters': {
                'spatial_calibration': 0.97,
                'temporal_calibration': 0.95,
                'intensity_calibration': 0.93
            }
        }
        
        return crimaldi_results, custom_results
    
    def test_cross_format_consistency_validation_basic(self):
        """Test basic cross-format consistency validation between Crimaldi and custom formats."""
        # Execute validate_cross_format_consistency function
        consistency_result = validate_cross_format_consistency(
            crimaldi_results=self.crimaldi_results,
            custom_results=self.custom_results,
            tolerance_threshold=0.1
        )
        
        # Validate equivalence testing using TOST procedure
        assert 'equivalence_testing' in consistency_result
        equivalence = consistency_result['equivalence_testing']
        assert 'tost_procedure' in equivalence
        assert 'equivalence_conclusion' in equivalence
        
        # Test cross-format correlation coefficient calculation
        assert 'cross_format_correlation' in consistency_result
        correlation = consistency_result['cross_format_correlation']
        assert 'trajectory_correlation' in correlation
        assert 'performance_correlation' in correlation
        
        # Verify trajectory correlation
        traj_corr = correlation['trajectory_correlation']
        assert -1.0 <= traj_corr <= 1.0
        
        # Assert compatibility metrics within tolerance thresholds
        assert 'compatibility_metrics' in consistency_result
        compatibility = consistency_result['compatibility_metrics']
        assert 'overall_compatibility_score' in compatibility
        assert 0.0 <= compatibility['overall_compatibility_score'] <= 1.0
    
    def test_cross_format_performance_differences(self):
        """Test analysis of format-specific performance differences and compatibility metrics."""
        # Execute detailed cross-format validation
        consistency_result = validate_cross_format_consistency(
            crimaldi_results=self.crimaldi_results,
            custom_results=self.custom_results,
            detailed_analysis=True,
            performance_comparison=True
        )
        
        # Verify format-specific performance difference analysis
        assert 'performance_differences' in consistency_result
        perf_diff = consistency_result['performance_differences']
        
        # Check individual metric differences
        assert 'success_rate_difference' in perf_diff
        assert 'path_efficiency_difference' in perf_diff
        assert 'processing_time_difference' in perf_diff
        
        # Validate difference calculations
        success_diff = perf_diff['success_rate_difference']
        expected_diff = abs(self.crimaldi_results['performance_metrics']['success_rate'] - 
                          self.custom_results['performance_metrics']['success_rate'])
        assert abs(success_diff - expected_diff) < 1e-6
        
        # Test comprehensive consistency validation reporting
        assert 'consistency_report' in consistency_result
        report = consistency_result['consistency_report']
        assert 'format_compatibility_assessment' in report
        assert 'recommended_adjustments' in report
    
    def test_cross_format_calibration_compatibility(self):
        """Test cross-format calibration parameter compatibility and conversion accuracy."""
        # Execute calibration compatibility assessment
        consistency_result = validate_cross_format_consistency(
            crimaldi_results=self.crimaldi_results,
            custom_results=self.custom_results,
            calibration_analysis=True,
            conversion_validation=True
        )
        
        # Verify calibration parameter analysis
        assert 'calibration_compatibility' in consistency_result
        calib_comp = consistency_result['calibration_compatibility']
        
        # Check spatial calibration compatibility
        assert 'spatial_calibration_compatibility' in calib_comp
        spatial_compat = calib_comp['spatial_calibration_compatibility']
        assert 'pixel_ratio_conversion' in spatial_compat
        assert 'spatial_accuracy_assessment' in spatial_compat
        
        # Check temporal calibration compatibility
        assert 'temporal_calibration_compatibility' in calib_comp
        temporal_compat = calib_comp['temporal_calibration_compatibility']
        assert 'frame_rate_conversion' in temporal_compat
        assert 'temporal_accuracy_assessment' in temporal_compat
        
        # Verify conversion accuracy validation
        if 'conversion_validation' in consistency_result:
            conversion_val = consistency_result['conversion_validation']
            assert 'conversion_accuracy_score' in conversion_val
            assert 0.0 <= conversion_val['conversion_accuracy_score'] <= 1.0


class TestTrajectoryAccuracyValidation:
    """Test suite for trajectory accuracy validation against reference implementations."""
    
    def setup_method(self):
        """Set up test fixtures and reference trajectory data for accuracy validation."""
        np.random.seed(TEST_RANDOM_SEED)
        self.validation_calculator = ValidationMetricsCalculator()
        
        # Load reference trajectory data from test fixtures
        self.reference_trajectory = self._create_reference_trajectory()
        self.test_trajectory = self._create_test_trajectory()
    
    def _create_reference_trajectory(self) -> np.ndarray:
        """Create reference trajectory with known accuracy characteristics."""
        # Reference trajectory: smooth path with known mathematical properties
        t = np.linspace(0, 2*np.pi, 100)
        reference_path = np.column_stack([
            np.cos(t) + 0.5 * np.cos(3*t),  # x coordinates
            np.sin(t) + 0.5 * np.sin(3*t)   # y coordinates
        ])
        return reference_path
    
    def _create_test_trajectory(self) -> np.ndarray:
        """Generate test trajectory with controlled accuracy characteristics."""
        # Test trajectory: reference + small deviations
        noise_level = 0.02  # 2% noise level for >95% correlation
        test_path = self.reference_trajectory + np.random.normal(0, noise_level, self.reference_trajectory.shape)
        return test_path
    
    @measure_performance(time_limit_seconds=1.0)
    def test_trajectory_accuracy_validation_correlation(self):
        """Test trajectory accuracy validation with >95% correlation requirement."""
        # Execute validate_trajectory_accuracy function
        accuracy_result = self.validation_calculator.validate_trajectory_accuracy(
            test_trajectory=self.test_trajectory,
            reference_trajectory=self.reference_trajectory,
            correlation_threshold=CORRELATION_THRESHOLD_95_PERCENT
        )
        
        # Validate correlation calculation against >95% threshold
        assert 'correlation_analysis' in accuracy_result
        correlation_analysis = accuracy_result['correlation_analysis']
        
        # Check overall correlation
        assert 'overall_correlation' in correlation_analysis
        overall_corr = correlation_analysis['overall_correlation']
        assert -1.0 <= overall_corr <= 1.0
        
        # Test statistical significance of trajectory similarity
        assert 'statistical_significance' in correlation_analysis
        significance = correlation_analysis['statistical_significance']
        assert 'p_value' in significance
        assert 'is_significant' in significance
        
        # Verify correlation meets threshold requirement
        assert 'threshold_compliance' in accuracy_result
        compliance = accuracy_result['threshold_compliance']
        assert 'correlation_threshold' in compliance
        assert compliance['correlation_threshold'] == CORRELATION_THRESHOLD_95_PERCENT
        
        meets_threshold = overall_corr >= CORRELATION_THRESHOLD_95_PERCENT
        assert compliance['meets_threshold'] == meets_threshold
    
    def test_trajectory_similarity_metrics(self):
        """Test trajectory similarity metrics including DTW and Hausdorff distance."""
        # Execute comprehensive trajectory similarity analysis
        accuracy_result = self.validation_calculator.validate_trajectory_accuracy(
            test_trajectory=self.test_trajectory,
            reference_trajectory=self.reference_trajectory,
            similarity_metrics=['dtw', 'hausdorff', 'frechet'],
            detailed_analysis=True
        )
        
        # Assert trajectory similarity metrics (DTW, Hausdorff)
        assert 'similarity_metrics' in accuracy_result
        similarity = accuracy_result['similarity_metrics']
        
        # Check Dynamic Time Warping distance
        if 'dtw_distance' in similarity:
            dtw_dist = similarity['dtw_distance']
            assert dtw_dist >= 0.0  # DTW distance should be non-negative
            assert 'normalized_dtw' in similarity
        
        # Check Hausdorff distance
        if 'hausdorff_distance' in similarity:
            hausdorff_dist = similarity['hausdorff_distance']
            assert hausdorff_dist >= 0.0  # Hausdorff distance should be non-negative
        
        # Check Fréchet distance
        if 'frechet_distance' in similarity:
            frechet_dist = similarity['frechet_distance']
            assert frechet_dist >= 0.0  # Fréchet distance should be non-negative
        
        # Verify comprehensive accuracy validation reporting
        assert 'accuracy_report' in accuracy_result
        report = accuracy_result['accuracy_report']
        assert 'overall_accuracy_score' in report
        assert 'trajectory_quality_assessment' in report
    
    def test_trajectory_validation_edge_cases(self):
        """Test trajectory accuracy validation with edge cases and error conditions."""
        # Test with identical trajectories (perfect correlation)
        identical_result = self.validation_calculator.validate_trajectory_accuracy(
            test_trajectory=self.reference_trajectory,
            reference_trajectory=self.reference_trajectory,
            correlation_threshold=CORRELATION_THRESHOLD_95_PERCENT
        )
        
        # Perfect correlation should be 1.0
        perfect_corr = identical_result['correlation_analysis']['overall_correlation']
        assert abs(perfect_corr - 1.0) < NUMERICAL_TOLERANCE
        
        # Test with completely different trajectories
        random_trajectory = np.random.randn(*self.reference_trajectory.shape)
        different_result = self.validation_calculator.validate_trajectory_accuracy(
            test_trajectory=random_trajectory,
            reference_trajectory=self.reference_trajectory,
            correlation_threshold=CORRELATION_THRESHOLD_95_PERCENT
        )
        
        # Correlation should be low for random trajectory
        low_corr = different_result['correlation_analysis']['overall_correlation']
        assert abs(low_corr) < 0.5  # Should be much lower than threshold
        
        # Test with mismatched trajectory lengths
        short_trajectory = self.reference_trajectory[:50]
        with pytest.warns(UserWarning, match="Trajectory length mismatch"):
            mismatch_result = self.validation_calculator.validate_trajectory_accuracy(
                test_trajectory=short_trajectory,
                reference_trajectory=self.reference_trajectory,
                handle_length_mismatch=True
            )
        
        # Should handle length mismatch gracefully
        assert 'length_mismatch_handling' in mismatch_result
        assert mismatch_result['length_mismatch_handling']['applied']


class TestPerformanceThresholdValidation:
    """Test suite for performance threshold validation against scientific computing requirements."""
    
    def setup_method(self):
        """Set up test fixtures and performance metrics for threshold validation testing."""
        self.validation_calculator = ValidationMetricsCalculator()
        self.performance_profiler = PerformanceProfiler()
        
        # Create mock performance metrics with realistic values
        self.performance_metrics = self._create_performance_metrics()
        self.threshold_config = self._create_threshold_configuration()
    
    def _create_performance_metrics(self) -> Dict[str, float]:
        """Create mock performance metrics with realistic values for threshold testing."""
        return {
            'processing_time_per_simulation': 6.5,  # Within 7.2 second target
            'memory_usage_mb': 850.0,  # Within reasonable limits
            'cpu_utilization_percent': 75.0,
            'correlation_coefficient': 0.97,  # Above 95% threshold
            'reproducibility_coefficient': 0.995,  # Above 99% threshold
            'error_rate': 0.008,  # Below 1% threshold
            'throughput_simulations_per_hour': 520.0,
            'cache_hit_rate': 0.85,
            'numerical_precision_error': 1e-7
        }
    
    def _create_threshold_configuration(self) -> Dict[str, Any]:
        """Create threshold configuration for performance validation testing."""
        return {
            'processing_time_limit': MAX_PROCESSING_TIME_SECONDS,
            'memory_limit_mb': MAX_MEMORY_USAGE_MB,
            'correlation_threshold': CORRELATION_THRESHOLD_95_PERCENT,
            'reproducibility_threshold': REPRODUCIBILITY_THRESHOLD_99_PERCENT,
            'error_rate_threshold': 0.01,  # 1% error rate threshold
            'throughput_minimum': 500.0,  # Minimum simulations per hour
            'cpu_utilization_limit': 85.0,
            'numerical_precision_limit': NUMERICAL_TOLERANCE
        }
    
    def test_performance_threshold_validation_basic(self):
        """Test basic performance threshold validation against scientific computing requirements."""
        # Execute validate_performance_thresholds function
        threshold_result = self.validation_calculator.validate_performance_thresholds(
            performance_metrics=self.performance_metrics,
            threshold_config=self.threshold_config
        )
        
        # Validate processing time against <7.2 seconds target
        assert 'processing_time_validation' in threshold_result
        proc_time_val = threshold_result['processing_time_validation']
        assert 'meets_threshold' in proc_time_val
        assert 'actual_time' in proc_time_val
        assert 'threshold_time' in proc_time_val
        
        actual_time = self.performance_metrics['processing_time_per_simulation']
        meets_time_threshold = actual_time <= MAX_PROCESSING_TIME_SECONDS
        assert proc_time_val['meets_threshold'] == meets_time_threshold
        
        # Test batch completion rate validation
        assert 'throughput_validation' in threshold_result
        throughput_val = threshold_result['throughput_validation']
        assert 'throughput_rate' in throughput_val
        assert 'minimum_required' in throughput_val
        
        # Assert memory usage and resource efficiency validation
        assert 'memory_validation' in threshold_result
        memory_val = threshold_result['memory_validation']
        assert 'memory_usage_mb' in memory_val
        assert 'memory_limit_mb' in memory_val
        
        memory_usage = self.performance_metrics['memory_usage_mb']
        meets_memory_threshold = memory_usage <= self.threshold_config['memory_limit_mb']
        assert memory_val['meets_threshold'] == meets_memory_threshold
    
    def test_performance_correlation_thresholds(self):
        """Test correlation and reproducibility threshold validation."""
        # Test correlation threshold validation
        threshold_result = self.validation_calculator.validate_performance_thresholds(
            performance_metrics=self.performance_metrics,
            threshold_config=self.threshold_config,
            detailed_analysis=True
        )
        
        # Verify correlation threshold compliance
        assert 'correlation_validation' in threshold_result
        corr_val = threshold_result['correlation_validation']
        assert 'correlation_coefficient' in corr_val
        assert 'correlation_threshold' in corr_val
        
        correlation = self.performance_metrics['correlation_coefficient']
        meets_correlation = correlation >= CORRELATION_THRESHOLD_95_PERCENT
        assert corr_val['meets_threshold'] == meets_correlation
        
        # Test reproducibility threshold validation
        assert 'reproducibility_validation' in threshold_result
        repro_val = threshold_result['reproducibility_validation']
        assert 'reproducibility_coefficient' in repro_val
        assert 'reproducibility_threshold' in repro_val
        
        reproducibility = self.performance_metrics['reproducibility_coefficient']
        meets_reproducibility = reproducibility >= REPRODUCIBILITY_THRESHOLD_99_PERCENT
        assert repro_val['meets_threshold'] == meets_reproducibility
    
    def test_performance_threshold_violations(self):
        """Test threshold violation detection and reporting for performance metrics."""
        # Create performance metrics with threshold violations
        violation_metrics = self.performance_metrics.copy()
        violation_metrics.update({
            'processing_time_per_simulation': 8.5,  # Exceeds 7.2 second limit
            'memory_usage_mb': 1200.0,  # Exceeds memory limit
            'correlation_coefficient': 0.92,  # Below 95% threshold
            'error_rate': 0.015  # Above 1% threshold
        })
        
        # Execute threshold validation with violations
        threshold_result = self.validation_calculator.validate_performance_thresholds(
            performance_metrics=violation_metrics,
            threshold_config=self.threshold_config
        )
        
        # Verify performance compliance reporting
        assert 'compliance_summary' in threshold_result
        compliance = threshold_result['compliance_summary']
        assert 'overall_compliance' in compliance
        assert 'violations_detected' in compliance
        assert 'violation_count' in compliance
        
        # Check that violations are properly detected
        violations = compliance['violations_detected']
        assert len(violations) > 0
        
        # Verify specific violations
        violation_types = [v['metric_name'] for v in violations]
        assert 'processing_time_per_simulation' in violation_types
        assert 'memory_usage_mb' in violation_types
        assert 'correlation_coefficient' in violation_types
        assert 'error_rate' in violation_types
        
        # Test threshold violation detection and reporting
        assert 'performance_recommendations' in threshold_result
        recommendations = threshold_result['performance_recommendations']
        assert len(recommendations) > 0
        assert all('recommendation' in rec for rec in recommendations)


class TestStatisticalPowerAnalysis:
    """Test suite for statistical power analysis and experimental design validation."""
    
    def setup_method(self):
        """Set up test fixtures for statistical power analysis testing."""
        np.random.seed(TEST_RANDOM_SEED)
        self.statistical_analyzer = StatisticalAnalyzer()
    
    @pytest.mark.parametrize('effect_size', [0.2, 0.5, 0.8])
    def test_statistical_power_analysis_effect_sizes(self, effect_size):
        """Test statistical power analysis functionality with different effect sizes."""
        sample_size = 50
        alpha = STATISTICAL_SIGNIFICANCE_LEVEL
        
        # Execute calculate_statistical_power function
        power_result = self.statistical_analyzer.calculate_statistical_power(
            effect_size=effect_size,
            sample_size=sample_size,
            alpha=alpha,
            test_type='two_sample_ttest'
        )
        
        # Validate power calculation accuracy for given parameters
        assert 'statistical_power' in power_result
        power = power_result['statistical_power']
        assert 0.0 <= power <= 1.0
        
        # Verify power increases with effect size
        if effect_size >= 0.8:  # Large effect size
            assert power >= 0.8  # Should have good power
        elif effect_size >= 0.5:  # Medium effect size
            assert power >= 0.5  # Moderate power expected
        
        # Test sample size recommendation generation
        assert 'sample_size_recommendations' in power_result
        recommendations = power_result['sample_size_recommendations']
        assert 'current_sample_size' in recommendations
        assert 'recommended_sample_size' in recommendations
        
        # Verify minimum detectable effect size calculation
        assert 'minimum_detectable_effect' in power_result
        min_effect = power_result['minimum_detectable_effect']
        assert min_effect > 0
        
        # Test experimental design recommendations
        assert 'design_recommendations' in power_result
        design_recs = power_result['design_recommendations']
        assert 'power_adequacy' in design_recs
        assert 'study_feasibility' in design_recs
    
    def test_statistical_power_sample_size_determination(self):
        """Test sample size determination for adequate statistical power."""
        target_power = 0.8
        effect_size = 0.5
        alpha = STATISTICAL_SIGNIFICANCE_LEVEL
        
        # Calculate required sample size for target power
        power_result = self.statistical_analyzer.calculate_required_sample_size(
            target_power=target_power,
            effect_size=effect_size,
            alpha=alpha,
            test_type='two_sample_ttest'
        )
        
        # Validate sample size calculation
        assert 'required_sample_size' in power_result
        required_n = power_result['required_sample_size']
        assert isinstance(required_n, int)
        assert required_n > 0
        
        # Verify power curve generation functionality
        assert 'power_curve' in power_result
        power_curve = power_result['power_curve']
        assert 'sample_sizes' in power_curve
        assert 'power_values' in power_curve
        
        # Power should increase with sample size
        sample_sizes = power_curve['sample_sizes']
        power_values = power_curve['power_values']
        assert len(sample_sizes) == len(power_values)
        assert all(p1 <= p2 for p1, p2 in zip(power_values[:-1], power_values[1:]))
    
    def test_statistical_power_multiple_testing_correction(self):
        """Test statistical power analysis with multiple testing correction."""
        effect_size = 0.6
        sample_size = 40
        alpha = STATISTICAL_SIGNIFICANCE_LEVEL
        num_comparisons = 5
        
        # Calculate power with multiple testing correction
        power_result = self.statistical_analyzer.calculate_statistical_power(
            effect_size=effect_size,
            sample_size=sample_size,
            alpha=alpha,
            multiple_comparisons=num_comparisons,
            correction_method='bonferroni'
        )
        
        # Verify multiple comparison correction impact
        assert 'multiple_testing_correction' in power_result
        correction = power_result['multiple_testing_correction']
        assert 'correction_method' in correction
        assert 'adjusted_alpha' in correction
        assert 'num_comparisons' in correction
        
        # Adjusted alpha should be more stringent
        adjusted_alpha = correction['adjusted_alpha']
        assert adjusted_alpha < alpha
        assert adjusted_alpha == alpha / num_comparisons  # Bonferroni correction
        
        # Power should be reduced due to correction
        corrected_power = power_result['statistical_power']
        
        # Compare with uncorrected power
        uncorrected_result = self.statistical_analyzer.calculate_statistical_power(
            effect_size=effect_size,
            sample_size=sample_size,
            alpha=alpha,
            multiple_comparisons=1
        )
        uncorrected_power = uncorrected_result['statistical_power']
        
        assert corrected_power <= uncorrected_power


class TestBootstrapAnalysis:
    """Test suite for bootstrap resampling analysis and robust statistical inference."""
    
    def setup_method(self):
        """Set up test fixtures for bootstrap analysis testing."""
        np.random.seed(TEST_RANDOM_SEED)
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Generate sample data with known statistical properties
        self.sample_data = self._generate_sample_data()
        self.bootstrap_iterations = 1000
    
    def _generate_sample_data(self) -> np.ndarray:
        """Generate sample data with known statistical properties for bootstrap testing."""
        # Create data with known mean and standard deviation
        true_mean = 10.0
        true_std = 2.5
        sample_size = 80
        
        # Generate data from normal distribution
        sample = np.random.normal(true_mean, true_std, sample_size)
        return sample
    
    def test_bootstrap_analysis_basic(self):
        """Test bootstrap resampling analysis for robust statistical inference."""
        # Execute perform_bootstrap_analysis function
        bootstrap_result = self.statistical_analyzer.perform_bootstrap_analysis(
            sample_data=self.sample_data,
            statistic='mean',
            bootstrap_iterations=self.bootstrap_iterations,
            confidence_level=0.95
        )
        
        # Validate bootstrap distribution generation
        assert 'bootstrap_distribution' in bootstrap_result
        bootstrap_dist = bootstrap_result['bootstrap_distribution']
        assert len(bootstrap_dist) == self.bootstrap_iterations
        
        # Check bootstrap distribution properties
        bootstrap_mean = np.mean(bootstrap_dist)
        sample_mean = np.mean(self.sample_data)
        assert abs(bootstrap_mean - sample_mean) < 0.1  # Should be close to sample mean
        
        # Test confidence interval calculation (percentile, BCa)
        assert 'confidence_intervals' in bootstrap_result
        ci = bootstrap_result['confidence_intervals']
        
        # Percentile confidence interval
        assert 'percentile' in ci
        percentile_ci = ci['percentile']
        assert 'lower_bound' in percentile_ci
        assert 'upper_bound' in percentile_ci
        assert percentile_ci['lower_bound'] <= sample_mean <= percentile_ci['upper_bound']
        
        # BCa confidence interval (if available)
        if 'bca' in ci:
            bca_ci = ci['bca']
            assert 'lower_bound' in bca_ci
            assert 'upper_bound' in bca_ci
            assert bca_ci['lower_bound'] <= bca_ci['upper_bound']
    
    def test_bootstrap_bias_estimation(self):
        """Test bootstrap bias estimation and correction."""
        # Execute bootstrap analysis with bias estimation
        bootstrap_result = self.statistical_analyzer.perform_bootstrap_analysis(
            sample_data=self.sample_data,
            statistic='mean',
            bootstrap_iterations=self.bootstrap_iterations,
            bias_correction=True,
            jackknife_estimation=True
        )
        
        # Assert bias estimation and correction
        assert 'bias_analysis' in bootstrap_result
        bias_analysis = bootstrap_result['bias_analysis']
        
        assert 'estimated_bias' in bias_analysis
        assert 'bias_corrected_estimate' in bias_analysis
        assert 'jackknife_estimate' in bias_analysis
        
        # Bias should be small for unbiased estimator (mean)
        estimated_bias = bias_analysis['estimated_bias']
        assert abs(estimated_bias) < 0.1  # Bias should be small
        
        # Bias-corrected estimate
        original_estimate = np.mean(self.sample_data)
        corrected_estimate = bias_analysis['bias_corrected_estimate']
        expected_corrected = original_estimate - estimated_bias
        assert abs(corrected_estimate - expected_corrected) < NUMERICAL_TOLERANCE
    
    def test_bootstrap_convergence_assessment(self):
        """Test bootstrap convergence assessment and iteration adequacy."""
        # Execute bootstrap with convergence monitoring
        bootstrap_result = self.statistical_analyzer.perform_bootstrap_analysis(
            sample_data=self.sample_data,
            statistic='mean',
            bootstrap_iterations=self.bootstrap_iterations,
            convergence_monitoring=True,
            convergence_threshold=0.01
        )
        
        # Verify bootstrap convergence assessment
        assert 'convergence_analysis' in bootstrap_result
        convergence = bootstrap_result['convergence_analysis']
        
        assert 'convergence_achieved' in convergence
        assert 'iterations_needed' in convergence
        assert 'convergence_criterion' in convergence
        
        # Check convergence monitoring
        iterations_needed = convergence['iterations_needed']
        assert iterations_needed <= self.bootstrap_iterations
        
        if convergence['convergence_achieved']:
            assert 'stability_assessment' in convergence
            stability = convergence['stability_assessment']
            assert 'estimate_stability' in stability
    
    def test_bootstrap_multiple_statistics(self):
        """Test bootstrap analysis with multiple statistics simultaneously."""
        statistics = ['mean', 'median', 'std', 'percentile_95']
        
        # Execute bootstrap for multiple statistics
        bootstrap_result = self.statistical_analyzer.perform_bootstrap_analysis(
            sample_data=self.sample_data,
            statistic=statistics,
            bootstrap_iterations=self.bootstrap_iterations,
            confidence_level=0.95
        )
        
        # Test robust statistical inference reporting
        assert 'multiple_statistics' in bootstrap_result
        multi_stats = bootstrap_result['multiple_statistics']
        
        # Check that all requested statistics are computed
        for stat in statistics:
            assert stat in multi_stats
            stat_result = multi_stats[stat]
            assert 'bootstrap_estimate' in stat_result
            assert 'confidence_interval' in stat_result
            
            # Verify confidence intervals
            ci = stat_result['confidence_interval']
            assert 'lower_bound' in ci
            assert 'upper_bound' in ci
            assert ci['lower_bound'] <= ci['upper_bound']
        
        # Check correlation between bootstrap estimates
        if 'correlation_matrix' in bootstrap_result:
            corr_matrix = bootstrap_result['correlation_matrix']
            assert corr_matrix.shape == (len(statistics), len(statistics))
            # Diagonal should be 1.0
            np.testing.assert_array_almost_equal(np.diag(corr_matrix), np.ones(len(statistics)))


class TestAlgorithmRankingCalculation:
    """Test suite for algorithm performance ranking calculation with statistical significance."""
    
    def setup_method(self):
        """Set up test fixtures for algorithm ranking calculation testing."""
        np.random.seed(TEST_RANDOM_SEED)
        self.validation_calculator = ValidationMetricsCalculator()
        
        # Create mock algorithm performance metrics
        self.algorithm_metrics = self._create_algorithm_metrics()
        self.metric_weights = self._create_metric_weights()
    
    def _create_algorithm_metrics(self) -> Dict[str, Dict[str, float]]:
        """Create mock algorithm performance metrics for ranking testing."""
        algorithms = TEST_ALGORITHM_TYPES
        metrics = {}
        
        for i, algorithm in enumerate(algorithms):
            # Create performance gradients for ranking validation
            base_performance = 0.7 + i * 0.05
            metrics[algorithm] = {
                'success_rate': base_performance + np.random.normal(0, 0.02),
                'path_efficiency': base_performance + np.random.normal(0, 0.03),
                'convergence_speed': 1.0 / (base_performance + 0.3),  # Inverse relationship
                'computational_cost': 0.5 - i * 0.05 + np.random.normal(0, 0.02),
                'robustness_score': base_performance + np.random.normal(0, 0.01),
                'memory_efficiency': 0.8 + i * 0.02 + np.random.normal(0, 0.01)
            }
        
        return metrics
    
    def _create_metric_weights(self) -> Dict[str, float]:
        """Create metric weights for algorithm ranking calculation."""
        return {
            'success_rate': 0.3,
            'path_efficiency': 0.25,
            'convergence_speed': 0.2,
            'computational_cost': 0.1,
            'robustness_score': 0.1,
            'memory_efficiency': 0.05
        }
    
    def test_algorithm_ranking_calculation_basic(self):
        """Test basic algorithm performance ranking calculation with weighted metrics."""
        # Execute calculate_algorithm_rankings function
        ranking_result = self.validation_calculator.calculate_algorithm_rankings(
            algorithm_metrics=self.algorithm_metrics,
            metric_weights=self.metric_weights,
            ranking_method='weighted_sum'
        )
        
        # Validate metric normalization and weighting
        assert 'normalized_metrics' in ranking_result
        normalized = ranking_result['normalized_metrics']
        
        # Check that normalization is applied correctly
        for algorithm in self.algorithm_metrics:
            assert algorithm in normalized
            for metric_name, normalized_value in normalized[algorithm].items():
                assert 0.0 <= normalized_value <= 1.0  # Should be normalized to [0,1]
        
        # Test composite score calculation accuracy
        assert 'composite_scores' in ranking_result
        composite_scores = ranking_result['composite_scores']
        
        # Verify composite scores calculation
        for algorithm in self.algorithm_metrics:
            assert algorithm in composite_scores
            score = composite_scores[algorithm]
            assert 0.0 <= score <= 1.0  # Composite score should be normalized
        
        # Assert ranking generation and statistical significance
        assert 'rankings' in ranking_result
        rankings = ranking_result['rankings']
        
        # Check ranking structure
        assert len(rankings) == len(self.algorithm_metrics)
        for i, ranking_entry in enumerate(rankings):
            assert 'algorithm' in ranking_entry
            assert 'rank' in ranking_entry
            assert 'composite_score' in ranking_entry
            assert ranking_entry['rank'] == i + 1  # Ranks should be sequential
        
        # Rankings should be in descending order of composite score
        scores = [entry['composite_score'] for entry in rankings]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    def test_algorithm_ranking_confidence_intervals(self):
        """Test confidence interval calculation for algorithm rankings."""
        # Execute ranking with confidence interval calculation
        ranking_result = self.validation_calculator.calculate_algorithm_rankings(
            algorithm_metrics=self.algorithm_metrics,
            metric_weights=self.metric_weights,
            confidence_intervals=True,
            bootstrap_iterations=500
        )
        
        # Verify confidence interval calculation for rankings
        assert 'ranking_confidence_intervals' in ranking_result
        ci_results = ranking_result['ranking_confidence_intervals']
        
        for algorithm in self.algorithm_metrics:
            assert algorithm in ci_results
            ci = ci_results[algorithm]
            assert 'composite_score_ci' in ci
            assert 'rank_distribution' in ci
            
            # Check composite score confidence interval
            score_ci = ci['composite_score_ci']
            assert 'lower_bound' in score_ci
            assert 'upper_bound' in score_ci
            assert score_ci['lower_bound'] <= score_ci['upper_bound']
            
            # Check rank distribution
            rank_dist = ci['rank_distribution']
            assert 'mean_rank' in rank_dist
            assert 'rank_variance' in rank_dist
    
    def test_algorithm_ranking_stability_assessment(self):
        """Test ranking stability assessment across different metric weightings."""
        # Test ranking with different weight configurations
        weight_configs = [
            self.metric_weights,  # Original weights
            {k: 1.0/len(self.metric_weights) for k in self.metric_weights},  # Equal weights
            {k: v*1.5 if k == 'success_rate' else v*0.5 for k, v in self.metric_weights.items()}  # Adjusted weights
        ]
        
        rankings_list = []
        for weights in weight_configs:
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
            
            ranking_result = self.validation_calculator.calculate_algorithm_rankings(
                algorithm_metrics=self.algorithm_metrics,
                metric_weights=normalized_weights,
                ranking_method='weighted_sum'
            )
            rankings_list.append(ranking_result['rankings'])
        
        # Test ranking stability assessment
        stability_result = self.validation_calculator.assess_ranking_stability(
            rankings_list=rankings_list,
            stability_metrics=['kendall_tau', 'spearman_rho', 'rank_correlation']
        )
        
        assert 'stability_assessment' in stability_result
        stability = stability_result['stability_assessment']
        
        # Check stability metrics
        assert 'average_kendall_tau' in stability
        assert 'average_spearman_rho' in stability
        assert 'ranking_consistency_score' in stability
        
        # Stability scores should be between -1 and 1
        assert -1.0 <= stability['average_kendall_tau'] <= 1.0
        assert -1.0 <= stability['average_spearman_rho'] <= 1.0
        assert 0.0 <= stability['ranking_consistency_score'] <= 1.0


class TestStatisticalAssumptionValidation:
    """Test suite for statistical assumption validation including normality and variance tests."""
    
    def setup_method(self):
        """Set up test fixtures for statistical assumption validation testing."""
        np.random.seed(TEST_RANDOM_SEED)
        self.statistical_validator = StatisticalValidator()
        
        # Generate test data with known distributional properties
        self.data_groups = self._generate_distributional_test_data()
        self.assumptions_to_test = ['normality', 'equal_variance', 'independence', 'outliers']
    
    def _generate_distributional_test_data(self) -> Dict[str, np.ndarray]:
        """Generate test data with known distributional properties for assumption testing."""
        return {
            'normal_data': np.random.normal(0, 1, 100),
            'skewed_data': np.random.gamma(2, 2, 100),
            'uniform_data': np.random.uniform(-2, 2, 100),
            'outlier_data': np.concatenate([np.random.normal(0, 1, 95), np.array([10, -10, 12, -12, 15])]),
            'heavy_tail_data': np.random.standard_t(3, 100)
        }
    
    def test_normality_assumption_validation(self):
        """Test normality assumption validation using multiple statistical tests."""
        # Execute validate_statistical_assumptions function
        assumption_result = self.statistical_validator.validate_statistical_assumptions(
            data_groups=self.data_groups,
            assumptions_to_test=['normality'],
            normality_tests=['shapiro', 'anderson', 'kstest', 'jarque_bera']
        )
        
        # Test normality assumption validation (Shapiro-Wilk, Anderson-Darling)
        assert 'normality_tests' in assumption_result
        normality_results = assumption_result['normality_tests']
        
        for group_name, group_data in self.data_groups.items():
            assert group_name in normality_results
            group_tests = normality_results[group_name]
            
            # Check Shapiro-Wilk test
            if 'shapiro' in group_tests:
                shapiro_result = group_tests['shapiro']
                assert 'statistic' in shapiro_result
                assert 'p_value' in shapiro_result
                assert 0.0 <= shapiro_result['p_value'] <= 1.0
                
                # Normal data should pass normality test
                if group_name == 'normal_data':
                    assert shapiro_result['p_value'] > STATISTICAL_SIGNIFICANCE_LEVEL
            
            # Check Anderson-Darling test
            if 'anderson' in group_tests:
                anderson_result = group_tests['anderson']
                assert 'statistic' in anderson_result
                assert 'critical_values' in anderson_result
                assert 'significance_levels' in anderson_result
        
        # Verify assumption violation reporting
        assert 'normality_violations' in assumption_result
        violations = assumption_result['normality_violations']
        
        # Skewed data should violate normality
        skewed_violations = [v for v in violations if v['group'] == 'skewed_data']
        assert len(skewed_violations) > 0
    
    def test_variance_homogeneity_testing(self):
        """Test homogeneity of variance testing using Levene's and Bartlett's tests."""
        # Select groups with different variances for testing
        variance_test_groups = {
            'low_variance': np.random.normal(0, 0.5, 50),
            'medium_variance': np.random.normal(0, 1.0, 50),
            'high_variance': np.random.normal(0, 2.0, 50)
        }
        
        # Execute variance homogeneity testing
        assumption_result = self.statistical_validator.validate_statistical_assumptions(
            data_groups=variance_test_groups,
            assumptions_to_test=['equal_variance'],
            variance_tests=['levene', 'bartlett', 'fligner']
        )
        
        # Validate homogeneity of variance testing (Levene's, Bartlett's)
        assert 'variance_tests' in assumption_result
        variance_results = assumption_result['variance_tests']
        
        # Check Levene's test
        if 'levene' in variance_results:
            levene_result = variance_results['levene']
            assert 'statistic' in levene_result
            assert 'p_value' in levene_result
            assert 0.0 <= levene_result['p_value'] <= 1.0
        
        # Check Bartlett's test
        if 'bartlett' in variance_results:
            bartlett_result = variance_results['bartlett']
            assert 'statistic' in bartlett_result
            assert 'p_value' in bartlett_result
            assert 0.0 <= bartlett_result['p_value'] <= 1.0
        
        # Groups with different variances should violate equal variance assumption
        assert 'variance_violations' in assumption_result
        variance_violations = assumption_result['variance_violations']
        assert len(variance_violations) > 0  # Should detect variance differences
    
    def test_outlier_detection_methods(self):
        """Test outlier detection using multiple statistical methods."""
        # Execute outlier detection
        assumption_result = self.statistical_validator.validate_statistical_assumptions(
            data_groups=self.data_groups,
            assumptions_to_test=['outliers'],
            outlier_methods=['zscore', 'iqr', 'isolation_forest', 'local_outlier_factor']
        )
        
        # Assert outlier detection using multiple methods
        assert 'outlier_detection' in assumption_result
        outlier_results = assumption_result['outlier_detection']
        
        for group_name, group_data in self.data_groups.items():
            if group_name in outlier_results:
                group_outliers = outlier_results[group_name]
                
                # Check Z-score method
                if 'zscore' in group_outliers:
                    zscore_outliers = group_outliers['zscore']
                    assert 'outlier_indices' in zscore_outliers
                    assert 'outlier_scores' in zscore_outliers
                    
                    # Outlier data should have detected outliers
                    if group_name == 'outlier_data':
                        assert len(zscore_outliers['outlier_indices']) > 0
                
                # Check IQR method
                if 'iqr' in group_outliers:
                    iqr_outliers = group_outliers['iqr']
                    assert 'outlier_indices' in iqr_outliers
                    assert 'lower_bound' in iqr_outliers
                    assert 'upper_bound' in iqr_outliers
        
        # Test alternative method recommendations
        assert 'alternative_methods' in assumption_result
        alternatives = assumption_result['alternative_methods']
        
        # Should recommend non-parametric methods when assumptions violated
        for group_name in self.data_groups:
            if group_name in alternatives:
                group_alternatives = alternatives[group_name]
                assert 'recommended_tests' in group_alternatives
                assert len(group_alternatives['recommended_tests']) > 0


class TestErrorHandlingAndEdgeCases:
    """Test suite for comprehensive error handling and edge cases in statistical validation."""
    
    def setup_method(self):
        """Set up test fixtures for error handling and edge case testing."""
        self.statistical_comparator = StatisticalComparator()
        self.statistical_analyzer = StatisticalAnalyzer()
    
    @pytest.mark.parametrize('error_scenario', ['empty_data', 'nan_values', 'infinite_values'])
    def test_error_handling_invalid_inputs(self, error_scenario):
        """Test comprehensive error handling for various invalid input scenarios."""
        # Create invalid inputs based on error scenario
        if error_scenario == 'empty_data':
            invalid_data = np.array([])
            reference_data = np.array([])
        elif error_scenario == 'nan_values':
            invalid_data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
            reference_data = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        elif error_scenario == 'infinite_values':
            invalid_data = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
            reference_data = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # Test handling of empty or None input data
        if error_scenario == 'empty_data':
            with pytest.raises(ValueError, match="Empty data arrays not supported"):
                calculate_correlation_matrix(invalid_data)
        
        # Validate error handling for NaN and infinite values
        elif error_scenario in ['nan_values', 'infinite_values']:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                try:
                    result = calculate_correlation_matrix(
                        data_matrix=invalid_data.reshape(-1, 1),
                        handle_invalid_values=True
                    )
                    
                    # Should handle invalid values gracefully
                    assert 'invalid_value_handling' in result
                    assert result['invalid_value_handling']['applied']
                    
                except ValueError as e:
                    # Should raise appropriate error for invalid data
                    assert "invalid values" in str(e).lower()
                
                # Should generate warnings for problematic data
                assert len(w) > 0
                assert any("invalid" in str(warning.message).lower() for warning in w)
    
    def test_dimension_mismatch_error_handling(self):
        """Test error handling for dimension mismatches in statistical operations."""
        # Create arrays with mismatched dimensions
        array1 = np.random.randn(100, 3)
        array2 = np.random.randn(80, 3)  # Different number of rows
        array3 = np.random.randn(100, 5)  # Different number of columns
        
        # Test dimension mismatch error handling
        with pytest.raises(ValueError, match="Shape mismatch"):
            assert_arrays_almost_equal(array1, array2)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            assert_arrays_almost_equal(array1, array3)
        
        # Test handling in correlation calculation
        with pytest.raises(ValueError, match="dimension"):
            self.statistical_analyzer.compare_algorithms(
                algorithm_results={'alg1': array1, 'alg2': array2},
                comparison_method='correlation'
            )
    
    def test_edge_case_statistical_operations(self):
        """Test statistical operations with edge cases like constant data and perfect correlations."""
        # Test with constant data (zero variance)
        constant_data = np.ones(50)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            correlation_result = calculate_correlation_matrix(
                data_matrix=np.column_stack([constant_data, constant_data]),
                handle_edge_cases=True
            )
            
            # Should handle constant data gracefully
            assert 'edge_case_handling' in correlation_result
            assert correlation_result['edge_case_handling']['constant_data_detected']
            
            # Should generate warning about zero variance
            assert len(w) > 0
            assert any("variance" in str(warning.message).lower() for warning in w)
        
        # Test with perfect correlation
        perfect_data1 = np.linspace(0, 10, 50)
        perfect_data2 = perfect_data1 * 2  # Perfect linear relationship
        
        correlation_result = calculate_correlation_matrix(
            data_matrix=np.column_stack([perfect_data1, perfect_data2]),
            handle_edge_cases=True
        )
        
        # Should detect perfect correlation
        correlation_matrix = correlation_result['correlation_matrix']
        assert abs(correlation_matrix[0, 1] - 1.0) < NUMERICAL_TOLERANCE
    
    def test_parameter_validation_edge_cases(self):
        """Test parameter validation for edge cases and boundary conditions."""
        # Test invalid correlation thresholds
        with pytest.raises(ValueError, match="Correlation threshold must be between 0 and 1"):
            StatisticalComparator(config={'correlation_threshold': 1.5})
        
        with pytest.raises(ValueError, match="Correlation threshold must be between 0 and 1"):
            StatisticalComparator(config={'correlation_threshold': -0.1})
        
        # Test invalid significance levels
        with pytest.raises(ValueError, match="Significance level must be positive"):
            perform_hypothesis_testing(
                group_data={'group1': np.random.randn(20), 'group2': np.random.randn(20)},
                test_type='ttest_ind',
                significance_level=-0.05
            )
        
        with pytest.raises(ValueError, match="Significance level must be less than 1"):
            perform_hypothesis_testing(
                group_data={'group1': np.random.randn(20), 'group2': np.random.randn(20)},
                test_type='ttest_ind',
                significance_level=1.5
            )
        
        # Assert proper exception raising for invalid parameters
        with pytest.raises(TypeError, match="Data must be numeric"):
            calculate_correlation_matrix(data_matrix="invalid_data")
    
    def test_graceful_degradation_edge_cases(self):
        """Test graceful degradation for statistical operations under edge conditions."""
        # Test with very small sample sizes
        small_sample = np.random.randn(3)  # Very small sample
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                power_result = self.statistical_analyzer.calculate_statistical_power(
                    effect_size=0.5,
                    sample_size=3,  # Very small
                    alpha=0.05
                )
                
                # Should include warnings about small sample size
                assert 'small_sample_warning' in power_result
                
            except ValueError as e:
                # Or should raise appropriate error
                assert "sample size" in str(e).lower()
            
            # Should generate warnings
            assert len(w) > 0
        
        # Verify graceful degradation for edge cases
        # Test with extreme effect sizes
        extreme_effect_result = self.statistical_analyzer.calculate_statistical_power(
            effect_size=10.0,  # Very large effect size
            sample_size=10,
            alpha=0.05
        )
        
        # Should handle extreme values gracefully
        assert 'extreme_effect_size_warning' in extreme_effect_result
        power = extreme_effect_result['statistical_power']
        assert 0.0 <= power <= 1.0  # Power should still be in valid range


class TestCachingAndPerformanceOptimization:
    """Test suite for caching functionality and performance optimization."""
    
    def setup_method(self):
        """Set up test fixtures for caching and performance optimization testing."""
        self.statistical_comparator = StatisticalComparator(enable_caching=True)
        self.performance_profiler = PerformanceProfiler()
    
    @measure_performance(time_limit_seconds=2.0)
    def test_caching_functionality_basic(self):
        """Test basic caching functionality and performance optimization."""
        # Initialize statistical components with caching enabled
        enable_caching = True
        cache_size_limit = 100
        
        comparator = StatisticalComparator(
            enable_caching=enable_caching,
            cache_size_limit=cache_size_limit
        )
        
        # Create test data for repeated operations
        test_data = np.random.randn(100, 5)
        
        # Execute first operation (should be cached)
        start_time = time.time()
        result1 = comparator.calculate_correlation_matrix(
            data_matrix=test_data,
            method='pearson'
        )
        first_execution_time = time.time() - start_time
        
        # Execute same operation again (should use cache)
        start_time = time.time()
        result2 = comparator.calculate_correlation_matrix(
            data_matrix=test_data,
            method='pearson'
        )
        second_execution_time = time.time() - start_time
        
        # Validate cache hit/miss behavior
        assert 'cache_info' in result2
        cache_info = result2['cache_info']
        assert cache_info['cache_hit'] is True
        assert cache_info['cache_key'] is not None
        
        # Second execution should be faster due to caching
        assert second_execution_time < first_execution_time * 0.5  # At least 50% faster
        
        # Results should be identical
        np.testing.assert_array_equal(result1['correlation_matrix'], result2['correlation_matrix'])
    
    def test_cache_eviction_and_size_management(self):
        """Test cache eviction and size management for memory optimization."""
        # Initialize with small cache size for testing eviction
        small_cache_size = 3
        comparator = StatisticalComparator(
            enable_caching=True,
            cache_size_limit=small_cache_size
        )
        
        # Generate multiple different datasets to fill cache beyond limit
        datasets = []
        for i in range(5):  # More than cache limit
            dataset = np.random.randn(50, 3) + i  # Different data each time
            datasets.append(dataset)
        
        # Execute operations to fill cache beyond limit
        results = []
        for i, dataset in enumerate(datasets):
            result = comparator.calculate_correlation_matrix(
                data_matrix=dataset,
                method='pearson'
            )
            results.append(result)
        
        # Test cache eviction and size management
        cache_stats = comparator.get_cache_statistics()
        assert 'current_cache_size' in cache_stats
        assert 'cache_limit' in cache_stats
        assert 'evictions_performed' in cache_stats
        
        # Cache size should not exceed limit
        assert cache_stats['current_cache_size'] <= small_cache_size
        
        # Should have performed evictions
        assert cache_stats['evictions_performed'] > 0
        
        # Test cache performance metrics
        assert 'hit_rate' in cache_stats
        assert 'miss_rate' in cache_stats
        assert 0.0 <= cache_stats['hit_rate'] <= 1.0
        assert 0.0 <= cache_stats['miss_rate'] <= 1.0
        assert abs(cache_stats['hit_rate'] + cache_stats['miss_rate'] - 1.0) < NUMERICAL_TOLERANCE
    
    def test_cache_invalidation_functionality(self):
        """Test cache invalidation functionality for data consistency."""
        comparator = StatisticalComparator(enable_caching=True)
        
        # Create initial data and cache result
        initial_data = np.random.randn(50, 3)
        initial_result = comparator.calculate_correlation_matrix(
            data_matrix=initial_data,
            method='pearson'
        )
        
        # Verify result is cached
        cached_result = comparator.calculate_correlation_matrix(
            data_matrix=initial_data,
            method='pearson'
        )
        assert cached_result['cache_info']['cache_hit'] is True
        
        # Invalidate cache
        comparator.invalidate_cache()
        
        # Verify cache invalidation functionality
        post_invalidation_result = comparator.calculate_correlation_matrix(
            data_matrix=initial_data,
            method='pearson'
        )
        
        # Should be cache miss after invalidation
        assert post_invalidation_result['cache_info']['cache_hit'] is False
        
        # Results should still be identical (data hasn't changed)
        np.testing.assert_array_equal(
            initial_result['correlation_matrix'],
            post_invalidation_result['correlation_matrix']
        )
    
    def test_performance_optimization_parallel_processing(self):
        """Test performance optimization through parallel processing capabilities."""
        # Test with parallel processing enabled
        start_time = time.time()
        comparator_parallel = StatisticalComparator(
            enable_parallel_processing=True,
            max_workers=4
        )
        
        # Create multiple algorithm datasets for parallel processing
        algorithm_datasets = {}
        for algorithm in TEST_ALGORITHM_TYPES:
            dataset = np.random.randn(100, 10)
            algorithm_datasets[algorithm] = dataset
        
        # Execute parallel comparison
        parallel_result = comparator_parallel.compare_multiple_algorithms(
            algorithm_datasets=algorithm_datasets,
            comparison_metrics=['correlation', 'statistical_tests'],
            parallel_processing=True
        )
        parallel_time = time.time() - start_time
        
        # Test with sequential processing
        start_time = time.time()
        comparator_sequential = StatisticalComparator(
            enable_parallel_processing=False
        )
        
        sequential_result = comparator_sequential.compare_multiple_algorithms(
            algorithm_datasets=algorithm_datasets,
            comparison_metrics=['correlation', 'statistical_tests'],
            parallel_processing=False
        )
        sequential_time = time.time() - start_time
        
        # Assert performance improvement with parallel processing
        assert parallel_time < sequential_time * 0.8  # At least 20% improvement
        
        # Results should be consistent
        assert len(parallel_result['algorithm_comparisons']) == len(sequential_result['algorithm_comparisons'])
        
        # Verify performance optimization metrics
        assert 'performance_metrics' in parallel_result
        perf_metrics = parallel_result['performance_metrics']
        assert 'parallel_processing_enabled' in perf_metrics
        assert 'execution_time_seconds' in perf_metrics
        assert 'speedup_factor' in perf_metrics


class TestIntegrationWithBenchmarkData:
    """Test suite for integration with reference benchmark data."""
    
    def setup_method(self):
        """Set up test fixtures and benchmark data for integration testing."""
        np.random.seed(TEST_RANDOM_SEED)
        self.statistical_comparator = StatisticalComparator()
        self.validation_calculator = ValidationMetricsCalculator()
        
        # Load reference benchmark data from test fixtures
        self.benchmark_data = self._load_benchmark_fixtures()
    
    def _load_benchmark_fixtures(self) -> Dict[str, Any]:
        """Load reference benchmark data from test fixtures for validation testing."""
        # Create synthetic benchmark data representing known algorithms
        benchmark_algorithms = TEST_ALGORITHM_TYPES
        benchmark_data = {}
        
        for i, algorithm in enumerate(benchmark_algorithms):
            # Create benchmark trajectory with known performance characteristics
            trajectory_length = 200
            time_points = np.linspace(0, 10, trajectory_length)
            
            # Algorithm-specific trajectory patterns
            if algorithm == 'infotaxis':
                # Infotaxis: more exploration, some randomness
                trajectory = self._generate_infotaxis_trajectory(time_points)
                expected_success_rate = 0.85
                expected_efficiency = 0.72
            elif algorithm == 'casting':
                # Casting: systematic search pattern
                trajectory = self._generate_casting_trajectory(time_points)
                expected_success_rate = 0.78
                expected_efficiency = 0.68
            elif algorithm == 'gradient_following':
                # Gradient following: direct approach
                trajectory = self._generate_gradient_trajectory(time_points)
                expected_success_rate = 0.82
                expected_efficiency = 0.75
            else:  # hybrid
                # Hybrid: combination of strategies
                trajectory = self._generate_hybrid_trajectory(time_points)
                expected_success_rate = 0.88
                expected_efficiency = 0.78
            
            benchmark_data[algorithm] = {
                'trajectory': trajectory,
                'performance_metrics': {
                    'success_rate': expected_success_rate,
                    'path_efficiency': expected_efficiency,
                    'convergence_time': 8.0 - i * 0.5,
                    'robustness_score': 0.8 + i * 0.02
                },
                'statistical_properties': {
                    'mean_displacement': np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)),
                    'trajectory_variance': np.var(trajectory, axis=0),
                    'correlation_structure': np.corrcoef(trajectory.T)
                }
            }
        
        return benchmark_data
    
    def _generate_infotaxis_trajectory(self, time_points: np.ndarray) -> np.ndarray:
        """Generate synthetic infotaxis trajectory with characteristic exploration pattern."""
        n_points = len(time_points)
        trajectory = np.zeros((n_points, 2))
        
        # Infotaxis characteristics: information-driven exploration
        for i in range(1, n_points):
            # Add exploration component
            exploration = np.random.normal(0, 0.1, 2)
            # Add information-driven bias (simplified)
            info_bias = np.array([0.02, 0.01]) * i / n_points
            trajectory[i] = trajectory[i-1] + exploration + info_bias
        
        return trajectory
    
    def _generate_casting_trajectory(self, time_points: np.ndarray) -> np.ndarray:
        """Generate synthetic casting trajectory with systematic search pattern."""
        n_points = len(time_points)
        trajectory = np.zeros((n_points, 2))
        
        # Casting characteristics: systematic zigzag pattern
        for i in range(1, n_points):
            # Zigzag pattern
            zigzag_x = 0.05 * np.sin(2 * np.pi * i / 20)
            upwind_y = 0.02
            trajectory[i] = trajectory[i-1] + np.array([zigzag_x, upwind_y])
        
        return trajectory
    
    def _generate_gradient_trajectory(self, time_points: np.ndarray) -> np.ndarray:
        """Generate synthetic gradient-following trajectory with direct approach."""
        n_points = len(time_points)
        trajectory = np.zeros((n_points, 2))
        
        # Gradient following: direct path with small deviations
        target = np.array([5.0, 5.0])
        for i in range(1, n_points):
            direction = target - trajectory[i-1]
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            step = direction * 0.03 + np.random.normal(0, 0.01, 2)
            trajectory[i] = trajectory[i-1] + step
        
        return trajectory
    
    def _generate_hybrid_trajectory(self, time_points: np.ndarray) -> np.ndarray:
        """Generate synthetic hybrid trajectory combining multiple strategies."""
        n_points = len(time_points)
        trajectory = np.zeros((n_points, 2))
        
        # Hybrid: combine exploration and exploitation
        for i in range(1, n_points):
            if i < n_points / 3:
                # Early: exploration phase
                step = np.random.normal(0, 0.08, 2)
            elif i < 2 * n_points / 3:
                # Middle: transition phase
                step = np.array([0.03, 0.02]) + np.random.normal(0, 0.03, 2)
            else:
                # Late: exploitation phase
                step = np.array([0.02, 0.025]) + np.random.normal(0, 0.02, 2)
            
            trajectory[i] = trajectory[i-1] + step
        
        return trajectory
    
    def test_benchmark_data_integration_basic(self):
        """Test basic integration with reference benchmark data for validation."""
        # Execute statistical validation against benchmark data
        benchmark_validation_result = self.statistical_comparator.validate_against_benchmarks(
            test_data=self.benchmark_data,
            benchmark_source='synthetic_reference',
            validation_criteria=['correlation', 'performance_metrics', 'statistical_properties']
        )
        
        # Validate correlation against >95% threshold requirement
        assert 'correlation_validation' in benchmark_validation_result
        correlation_val = benchmark_validation_result['correlation_validation']
        
        for algorithm in TEST_ALGORITHM_TYPES:
            assert algorithm in correlation_val
            alg_correlation = correlation_val[algorithm]
            assert 'benchmark_correlation' in alg_correlation
            assert 'meets_95_percent_threshold' in alg_correlation
            
            # Check correlation threshold compliance
            correlation_coeff = alg_correlation['benchmark_correlation']
            meets_threshold = correlation_coeff >= CORRELATION_THRESHOLD_95_PERCENT
            assert alg_correlation['meets_95_percent_threshold'] == meets_threshold
        
        # Test statistical significance of benchmark comparison
        assert 'statistical_significance' in benchmark_validation_result
        significance = benchmark_validation_result['statistical_significance']
        
        for algorithm in TEST_ALGORITHM_TYPES:
            if algorithm in significance:
                alg_significance = significance[algorithm]
                assert 'p_value' in alg_significance
                assert 'is_significant' in alg_significance
                assert 0.0 <= alg_significance['p_value'] <= 1.0
    
    def test_benchmark_performance_metrics_validation(self):
        """Test validation of performance metrics against benchmark standards."""
        # Execute performance metrics validation
        performance_validation = self.validation_calculator.validate_benchmark_performance(
            test_algorithms=self.benchmark_data,
            benchmark_standards={
                'minimum_success_rate': 0.75,
                'minimum_efficiency': 0.65,
                'maximum_convergence_time': 10.0,
                'minimum_robustness': 0.70
            }
        )
        
        # Assert comprehensive validation report generation
        assert 'performance_compliance' in performance_validation
        compliance = performance_validation['performance_compliance']
        
        for algorithm in TEST_ALGORITHM_TYPES:
            if algorithm in compliance:
                alg_compliance = compliance[algorithm]
                assert 'meets_standards' in alg_compliance
                assert 'performance_scores' in alg_compliance
                assert 'compliance_details' in alg_compliance
                
                # Check individual metric compliance
                performance_scores = alg_compliance['performance_scores']
                expected_metrics = ['success_rate', 'path_efficiency', 'convergence_time', 'robustness_score']
                for metric in expected_metrics:
                    if metric in performance_scores:
                        assert 'value' in performance_scores[metric]
                        assert 'meets_standard' in performance_scores[metric]
        
        # Verify benchmark data integrity and consistency
        assert 'data_integrity_check' in performance_validation
        integrity = performance_validation['data_integrity_check']
        assert 'data_completeness' in integrity
        assert 'statistical_consistency' in integrity
        assert 'benchmark_validity' in integrity
    
    def test_benchmark_cross_validation(self):
        """Test cross-validation with benchmark data for statistical robustness."""
        # Perform k-fold cross-validation with benchmark data
        cross_validation_result = self.statistical_comparator.cross_validate_benchmarks(
            benchmark_data=self.benchmark_data,
            k_folds=5,
            validation_metrics=['correlation', 'rmse', 'mae'],
            random_seed=TEST_RANDOM_SEED
        )
        
        # Validate cross-validation results
        assert 'cross_validation_scores' in cross_validation_result
        cv_scores = cross_validation_result['cross_validation_scores']
        
        for algorithm in TEST_ALGORITHM_TYPES:
            if algorithm in cv_scores:
                alg_scores = cv_scores[algorithm]
                assert 'mean_correlation' in alg_scores
                assert 'std_correlation' in alg_scores
                assert 'mean_rmse' in alg_scores
                assert 'std_rmse' in alg_scores
                
                # Cross-validation scores should be reasonable
                mean_corr = alg_scores['mean_correlation']
                assert 0.0 <= mean_corr <= 1.0
                
                # Standard deviation should indicate consistency
                std_corr = alg_scores['std_correlation']
                assert std_corr >= 0.0
                assert std_corr < 0.3  # Should be reasonably consistent
        
        # Test statistical robustness assessment
        assert 'robustness_assessment' in cross_validation_result
        robustness = cross_validation_result['robustness_assessment']
        assert 'overall_stability' in robustness
        assert 'algorithm_rankings_consistency' in robustness
        
        # Check algorithm ranking consistency across folds
        ranking_consistency = robustness['algorithm_rankings_consistency']
        assert 'kendall_tau' in ranking_consistency
        assert 'spearman_rho' in ranking_consistency
        assert -1.0 <= ranking_consistency['kendall_tau'] <= 1.0
        assert -1.0 <= ranking_consistency['spearman_rho'] <= 1.0


# Performance and integration test fixtures
@pytest.fixture
def performance_test_environment():
    """Set up performance testing environment with monitoring and profiling."""
    with setup_test_environment(
        test_name="statistical_validation_performance",
        cleanup_on_exit=True
    ) as test_env:
        # Initialize performance monitoring
        profiler = PerformanceProfiler(
            time_threshold_seconds=MAX_PROCESSING_TIME_SECONDS,
            memory_threshold_mb=MAX_MEMORY_USAGE_MB
        )
        
        test_env['profiler'] = profiler
        test_env['performance_metrics'] = {}
        
        yield test_env


@pytest.fixture
def batch_processing_simulator():
    """Simulate batch processing environment for large-scale testing."""
    def simulate_batch(simulation_count: int = BENCHMARK_SIMULATION_COUNT):
        """Simulate batch processing with specified simulation count."""
        np.random.seed(TEST_RANDOM_SEED)
        
        batch_results = []
        for i in range(simulation_count):
            # Simulate individual simulation result
            success = np.random.random() > 0.1  # 90% success rate
            execution_time = np.random.normal(6.0, 1.0)  # Around 6 seconds
            
            result = {
                'simulation_id': i,
                'success': success,
                'execution_time': max(0.1, execution_time),
                'performance_metrics': {
                    'correlation': np.random.normal(0.96, 0.02),
                    'reproducibility': np.random.normal(0.995, 0.003)
                }
            }
            batch_results.append(result)
        
        return batch_results
    
    return simulate_batch


# Integration test for comprehensive statistical validation workflow
@measure_performance(time_limit_seconds=30.0)
def test_comprehensive_statistical_validation_workflow(performance_test_environment, batch_processing_simulator):
    """Integration test for comprehensive statistical validation workflow."""
    test_env = performance_test_environment
    profiler = test_env['profiler']
    
    # Start performance profiling
    profiler.start_profiling("comprehensive_validation_workflow")
    
    try:
        # Step 1: Initialize validation components
        statistical_comparator = StatisticalComparator(
            config={
                'correlation_threshold': CORRELATION_THRESHOLD_95_PERCENT,
                'reproducibility_threshold': REPRODUCIBILITY_THRESHOLD_99_PERCENT,
                'significance_level': STATISTICAL_SIGNIFICANCE_LEVEL
            },
            enable_caching=True
        )
        
        # Step 2: Generate comprehensive test data
        algorithm_results = {}
        for algorithm in TEST_ALGORITHM_TYPES[:3]:  # Limit for performance
            n_samples = 80
            base_performance = 0.8 + np.random.random() * 0.1
            
            algorithm_results[algorithm] = {
                'trajectory_data': np.random.randn(n_samples, 2),
                'performance_metrics': {
                    'success_rate': base_performance,
                    'path_efficiency': base_performance * 0.9,
                    'processing_time': np.random.normal(6.0, 0.5)
                },
                'statistical_samples': {
                    'success_rate_samples': np.random.normal(base_performance, 0.05, n_samples),
                    'efficiency_samples': np.random.normal(base_performance * 0.9, 0.03, n_samples)
                }
            }
        
        # Step 3: Execute algorithm performance comparison
        comparison_result = compare_algorithm_performance(
            algorithm_results=algorithm_results,
            comparison_metrics=['success_rate', 'path_efficiency'],
            significance_level=STATISTICAL_SIGNIFICANCE_LEVEL
        )
        
        # Validate comparison results
        assert 'statistical_tests' in comparison_result
        assert 'performance_rankings' in comparison_result
        assert 'effect_sizes' in comparison_result
        
        # Step 4: Execute reproducibility assessment
        repeated_measurements = []
        for _ in range(5):
            measurement = np.random.normal(10.0, 0.1, 50)  # High reproducibility
            repeated_measurements.append(measurement)
        
        reproducibility_result = assess_reproducibility(
            repeated_measurements=repeated_measurements,
            icc_type='ICC(2,1)',
            confidence_level=0.95
        )
        
        # Validate reproducibility results
        assert 'icc_coefficient' in reproducibility_result
        icc_value = reproducibility_result['icc_coefficient']
        assert icc_value >= REPRODUCIBILITY_THRESHOLD_99_PERCENT  # Should meet threshold
        
        # Step 5: Execute cross-format compatibility validation
        crimaldi_data = {
            'trajectory_data': np.random.randn(100, 2),
            'performance_metrics': {'success_rate': 0.85, 'efficiency': 0.72}
        }
        
        custom_data = {
            'trajectory_data': crimaldi_data['trajectory_data'] + np.random.normal(0, 0.01, (100, 2)),
            'performance_metrics': {'success_rate': 0.83, 'efficiency': 0.74}
        }
        
        compatibility_result = validate_cross_format_consistency(
            crimaldi_results=crimaldi_data,
            custom_results=custom_data,
            tolerance_threshold=0.1
        )
        
        # Validate compatibility results
        assert 'cross_format_correlation' in compatibility_result
        assert 'compatibility_metrics' in compatibility_result
        
        # Step 6: Execute batch processing validation
        batch_results = batch_processing_simulator(BENCHMARK_SIMULATION_COUNT)
        
        batch_validation = validate_batch_processing_results(
            batch_results=batch_results,
            expected_count=BENCHMARK_SIMULATION_COUNT,
            completion_threshold=0.95
        )
        
        # Validate batch processing results
        assert batch_validation.is_valid
        assert batch_validation.metrics['completion_rate'] >= 0.95
        
        # Step 7: Generate comprehensive validation report
        all_validation_results = [
            comparison_result,
            reproducibility_result,
            compatibility_result,
            batch_validation.to_dict()
        ]
        
        # Convert to ValidationResult objects for reporting
        validation_results = []
        for result in all_validation_results:
            val_result = ValidationResult(
                validation_type="integration_test",
                is_valid=True,
                validation_context="comprehensive_workflow"
            )
            val_result.metadata.update(result)
            validation_results.append(val_result)
        
        # Generate final validation report
        from src.test.utils.test_helpers import create_validation_report
        final_report = create_validation_report(
            validation_results=validation_results,
            report_type="comprehensive",
            include_recommendations=True
        )
        
        # Validate final report
        assert 'validation_summary' in final_report
        assert 'aggregated_statistics' in final_report
        assert 'recommendations' in final_report
        
        # Check overall success
        summary = final_report['validation_summary']
        assert summary['success_rate'] >= 0.95  # 95% success rate required
        
    finally:
        # Stop performance profiling
        performance_report = profiler.stop_profiling()
        test_env['performance_metrics'] = performance_report
        
        # Validate performance requirements
        assert performance_report['session_metrics']['execution_time_seconds'] <= 30.0
        assert performance_report['threshold_validation']['overall_performance_acceptable']


if __name__ == "__main__":
    # Run comprehensive statistical validation tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--maxfail=5",
        f"--junit-xml={pathlib.Path(__file__).parent}/test_results.xml"
    ])