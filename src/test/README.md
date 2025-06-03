# Plume Navigation Simulation Testing Framework

Comprehensive testing infrastructure for scientific plume navigation simulation validation with >95% correlation accuracy, <7.2 seconds per simulation performance, and 4000+ simulation batch processing capabilities.

## Table of Contents

- [Quick Start Guide](#quick-start-guide)
- [Test Framework Architecture](#test-framework-architecture)
- [Test Execution Guide](#test-execution-guide)
- [Test Fixtures and Data Management](#test-fixtures-and-data-management)
- [Performance Validation and Benchmarking](#performance-validation-and-benchmarking)
- [Scientific Computing Standards](#scientific-computing-standards)
- [Error Handling and Recovery Testing](#error-handling-and-recovery-testing)
- [Troubleshooting and Common Issues](#troubleshooting-and-common-issues)
- [Continuous Integration and Automation](#continuous-integration-and-automation)
- [Development and Testing Workflow](#development-and-testing-workflow)

## Overview

This testing framework provides comprehensive validation for the plume navigation simulation system, ensuring scientific accuracy, performance compliance, and cross-format compatibility. The framework supports unit testing, integration testing, and performance testing with automated validation against scientific computing standards.

### Key Features

- **Scientific Accuracy Validation**: >95% correlation with reference implementations
- **Performance Testing**: <7.2 seconds per simulation validation
- **Batch Processing Validation**: 4000+ simulations within 8 hours
- **Cross-Format Compatibility**: Crimaldi and custom AVI format testing
- **Error Handling Testing**: Comprehensive error scenario validation
- **Reproducibility Testing**: >0.99 reproducibility coefficient validation
- **Automated Benchmarking**: Performance regression detection
- **Statistical Validation**: Hypothesis testing and significance analysis

## Requirements

### System Requirements
- Python 3.9+
- 8GB RAM minimum (16GB recommended)
- Multi-core CPU for parallel processing
- 10GB free disk space for test data

### Dependencies
```bash
# Install test dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest --version
pytest-benchmark --version
```

## Quick Start Guide

### Basic Test Execution
```bash
# Run all tests
python -m pytest src/test/

# Run with coverage
python -m pytest --cov=src/backend --cov-report=html

# Run specific test categories
python -m pytest -m unit          # Unit tests only
python -m pytest -m integration   # Integration tests only
python -m pytest -m performance   # Performance tests only

# Run cross-format compatibility tests
python -m pytest -m cross_format

# Run accuracy validation tests
python -m pytest -m accuracy
```

### Installation Steps
1. **Install Python 3.9+ with scientific computing dependencies**
   ```bash
   # Verify Python version
   python --version
   
   # Install scientific computing stack
   pip install numpy>=2.1.3 scipy>=1.15.3 opencv-python>=4.11.0
   ```

2. **Install test dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify test environment**
   ```bash
   python -m pytest --version
   ```

4. **Run basic test suite**
   ```bash
   python -m pytest src/test/unit/ -v
   ```

5. **Execute integration tests**
   ```bash
   python -m pytest src/test/integration/ -v
   ```

6. **Run performance validation**
   ```bash
   python -m pytest src/test/performance/ -v
   ```

### Basic Usage Examples
```bash
# Run all tests with verbose output
python -m pytest src/test/ -v

# Run with coverage report
python -m pytest --cov=src/backend --cov-report=html

# Run performance tests only
python -m pytest -m performance

# Run cross-format compatibility tests
python -m pytest -m cross_format

# Generate HTML test report
python -m pytest --html=report.html

# Run tests in parallel
python -m pytest -n auto

# Run with benchmark profiling
python -m pytest --benchmark-only
```

## Test Framework Architecture

### Unit Testing

Individual component testing with isolation and mocking to ensure each component functions correctly in isolation.

#### Test Categories

- **Data Validation and Format Compatibility Testing**
  - Video file format validation (AVI, Crimaldi formats)
  - Metadata extraction and verification
  - File integrity checking
  - Format conversion accuracy testing

- **Scale Calibration and Intensity Calibration Testing**
  - Pixel-to-physical unit conversion validation
  - Arena size normalization testing
  - Temporal sampling rate standardization
  - Intensity unit conversion accuracy

- **Video Processing and Temporal Normalization Testing**
  - Frame extraction and processing validation
  - Temporal interpolation accuracy testing
  - Memory-mapped video access performance
  - Video preprocessing pipeline validation

- **Memory Mapping and Caching Efficiency Testing**
  - Memory usage optimization validation
  - Disk caching effectiveness testing
  - Multi-level cache hierarchy performance
  - Memory leak detection and prevention

- **Parallel Processing and Simulation Runtime Testing**
  - Multi-core utilization efficiency testing
  - Process pool management validation
  - Task distribution and load balancing
  - Parallel execution correctness verification

- **Algorithm Execution and Analysis Pipeline Testing**
  - Navigation algorithm correctness validation
  - Parameter configuration testing
  - Algorithm output verification
  - Cross-algorithm compatibility testing

- **Metrics Calculation and Statistical Validation Testing**
  - Performance metric computation accuracy
  - Statistical significance testing
  - Correlation analysis validation
  - Hypothesis testing framework verification

- **Visualization and Error Handling Testing**
  - Plot generation and formatting validation
  - Error message clarity and actionability
  - Exception handling completeness
  - Recovery mechanism effectiveness

#### Validation Criteria

- **Numerical Tolerance**: 1e-6 for floating-point comparisons
- **Correlation Threshold**: 0.95 minimum for accuracy validation
- **Performance Timeout**: 7.2 seconds maximum per simulation
- **Reproducibility Threshold**: 0.99 minimum consistency coefficient

### Integration Testing

End-to-end workflow and component interaction testing to validate complete pipeline functionality.

#### Test Workflows

- **Normalization to Simulation Pipeline Integration**
  - Complete data preprocessing workflow validation
  - Parameter passing between pipeline stages
  - Data integrity maintenance across transitions
  - Error propagation and handling verification

- **Simulation to Analysis Pipeline Integration**
  - Result data format consistency validation
  - Analysis pipeline input verification
  - Statistical computation accuracy testing
  - Output format standardization validation

- **Cross-Format Compatibility Workflow Testing**
  - Crimaldi to custom format conversion testing
  - Format-specific parameter handling validation
  - Cross-format result consistency verification
  - Metadata preservation across formats

- **Batch Processing Integration with 4000+ Simulations**
  - Large-scale batch execution validation
  - Resource management under load testing
  - Progress monitoring and reporting validation
  - Batch completion rate verification

- **Error Recovery and Graceful Degradation Testing**
  - Failure scenario simulation and recovery
  - Partial result preservation validation
  - Checkpoint and resume functionality testing
  - Error reporting and logging verification

- **Complete End-to-End Workflow Validation**
  - Full system integration testing
  - User workflow simulation
  - Performance target validation under realistic conditions
  - Scientific reproducibility verification

#### Performance Targets

- **Batch Completion Time**: 8 hours maximum for 4000 simulations
- **Cross-Format Tolerance**: 1e-4 maximum conversion error
- **Error Recovery Rate**: >99% successful recovery from transient failures

### Performance Testing

Performance validation and benchmarking against scientific computing requirements.

#### Test Categories

- **Normalization Performance with Video Processing Optimization**
  - Video loading and preprocessing speed validation
  - Memory-mapped file access performance testing
  - Parallel normalization pipeline throughput testing
  - Cache effectiveness and hit rate validation

- **Simulation Speed Validation with Algorithm Performance Testing**
  - Single simulation execution time measurement
  - Algorithm complexity scaling validation
  - Parameter sensitivity performance testing
  - Resource utilization efficiency measurement

- **Batch Throughput Analysis with Large-Scale Validation**
  - 4000+ simulation batch execution monitoring
  - Parallel processing scalability validation
  - System resource utilization optimization
  - Throughput consistency across extended runs

- **Memory Usage Testing with Leak Detection**
  - Peak memory usage monitoring and validation
  - Memory leak detection in long-running operations
  - Garbage collection effectiveness testing
  - Memory allocation pattern optimization

- **Result Accuracy Validation with Statistical Testing**
  - Correlation coefficient calculation and validation
  - Statistical significance testing automation
  - Reproducibility measurement across runs
  - Numerical precision maintenance validation

- **Parallel Scaling Efficiency with Optimization Analysis**
  - Multi-core scaling performance measurement
  - Load balancing effectiveness validation
  - Communication overhead minimization testing
  - Optimal thread/process count determination

#### Performance Thresholds

- **Simulation Time**: 7.2 seconds maximum per simulation
- **Batch Target**: 4000 simulations within 8 hours
- **Memory Limit**: 8.0 GB maximum peak usage
- **Correlation Accuracy**: 0.95 minimum correlation coefficient
- **Reproducibility Coefficient**: 0.99 minimum consistency

## Test Execution Guide

### Test Discovery

Automated test discovery and execution configuration using pytest conventions.

#### Discovery Patterns

- **Test Files**: `test_*.py` and `*_test.py`
- **Test Classes**: `Test*`, `*Test`, `*TestSuite`
- **Test Functions**: `test_*`
- **Test Directories**: `unit/`, `integration/`, `performance/`

### Test Markers

Pytest markers for selective test execution and categorization.

#### Available Markers

- **unit**: Unit tests for individual components
- **integration**: Integration tests for pipeline workflows
- **performance**: Performance validation tests
- **slow**: Tests that take longer than 30 seconds
- **fast**: Tests that complete within 5 seconds
- **crimaldi**: Tests specific to Crimaldi dataset format
- **custom**: Tests specific to custom AVI format
- **cross_format**: Tests validating cross-format compatibility
- **accuracy**: Tests validating >95% correlation requirement
- **speed**: Tests validating <7.2 seconds per simulation requirement
- **batch**: Tests validating 4000+ simulation batch processing
- **memory**: Tests validating memory usage efficiency
- **parallel**: Tests validating parallel processing capabilities
- **normalization**: Tests for data normalization pipeline
- **simulation**: Tests for simulation execution engine
- **analysis**: Tests for performance analysis pipeline
- **error_handling**: Tests for error detection and recovery
- **reproducibility**: Tests validating >0.99 reproducibility coefficient

### Execution Examples

#### Selective Execution
```bash
# Run unit tests only
pytest -m unit

# Run performance tests
pytest -m performance

# Run cross-format compatibility tests
pytest -m cross_format

# Run fast tests only (< 5 seconds)
pytest -m fast

# Run accuracy validation tests
pytest -m accuracy

# Run batch processing tests
pytest -m batch

# Run specific combinations
pytest -m "unit and fast"
pytest -m "performance and not slow"
pytest -m "accuracy or reproducibility"
```

#### Parallel Execution
```bash
# Run tests in parallel with auto-detection
pytest -n auto

# Distribute tests by scope for better load balancing
pytest -n auto --dist=loadscope

# Use custom worker count
pytest -n 4

# Parallel execution with specific markers
pytest -m unit -n auto
```

#### Coverage Analysis
```bash
# Generate HTML coverage report
pytest --cov=src/backend --cov-report=html

# Terminal coverage report with missing lines
pytest --cov=src/backend --cov-report=term-missing

# XML coverage report for CI systems
pytest --cov=src/backend --cov-report=xml

# Fail if coverage is below threshold
pytest --cov-fail-under=95

# Combined coverage and performance testing
pytest --cov=src/backend -m "not slow"
```

## Test Fixtures and Data Management

### Fixture Categories

#### Session-Scoped Fixtures

- **test_config_loader**: Configuration loading and validation
  ```python
  @pytest.fixture(scope="session")
  def test_config_loader():
      """Load and validate test configuration settings"""
  ```

- **reference_benchmark_data**: Reference data for accuracy validation
  ```python
  @pytest.fixture(scope="session")
  def reference_benchmark_data():
      """Load reference benchmark results for validation"""
  ```

#### Function-Scoped Fixtures

- **test_environment**: Isolated test environment setup
  ```python
  @pytest.fixture
  def test_environment():
      """Setup isolated test environment with cleanup"""
  ```

- **performance_monitor**: Performance monitoring and validation
  ```python
  @pytest.fixture
  def performance_monitor():
      """Monitor and validate performance metrics"""
  ```

- **validation_metrics_calculator**: Metrics calculation and analysis
  ```python
  @pytest.fixture
  def validation_metrics_calculator():
      """Calculate and validate simulation metrics"""
  ```

#### Parametrized Fixtures

- **crimaldi_test_data**: Various Crimaldi format test scenarios
  ```python
  @pytest.fixture(params=["small", "medium", "large"])
  def crimaldi_test_data(request):
      """Parametrized Crimaldi format test data"""
  ```

- **custom_test_data**: Various custom AVI format test scenarios
  ```python
  @pytest.fixture(params=["standard", "high_res", "low_res"])
  def custom_test_data(request):
      """Parametrized custom AVI format test data"""
  ```

- **batch_test_scenario**: Different batch processing scenarios
  ```python
  @pytest.fixture(params=[10, 100, 1000])
  def batch_test_scenario(request):
      """Parametrized batch processing scenarios"""
  ```

#### Mock Fixtures

- **mock_simulation_engine**: Mock simulation engine for testing
  ```python
  @pytest.fixture
  def mock_simulation_engine():
      """Mock simulation engine with predictable behavior"""
  ```

- **synthetic_plume_generator**: Synthetic plume data generation
  ```python
  @pytest.fixture
  def synthetic_plume_generator():
      """Generate synthetic plume data for testing"""
  ```

### Test Data Structure

#### Test Fixtures Directory: `src/test/test_fixtures/`

#### Video Samples
- **crimaldi_sample.avi**: Sample Crimaldi format video
- **custom_sample.avi**: Sample custom format video

#### Configuration Files
- **test_algorithm_config.json**: Algorithm testing configuration
- **test_normalization_config.json**: Normalization testing configuration
- **test_simulation_config.json**: Simulation testing configuration

#### Reference Results
- **analysis_benchmark.npy**: Reference analysis results
- **normalization_benchmark.npy**: Reference normalization results
- **simulation_benchmark.npy**: Reference simulation results

## Performance Validation and Benchmarking

### Performance Requirements

#### Simulation Speed
- **Target**: <7.2 seconds per simulation
- **Validation Method**: Automated timing with statistical analysis
- **Test Scenarios**:
  - Single simulation performance validation
  - Batch simulation throughput testing
  - Large-scale performance with 4000+ simulations

#### Batch Processing
- **Target**: 4000 simulations within 8 hours
- **Validation Method**: End-to-end batch execution monitoring
- **Test Scenarios**:
  - Parallel processing efficiency validation
  - Resource utilization optimization testing
  - Cross-format batch compatibility testing

#### Accuracy Validation
- **Target**: >95% correlation with reference implementations
- **Validation Method**: Statistical correlation analysis
- **Test Scenarios**:
  - Cross-format result accuracy testing
  - Reproducibility validation with >0.99 coefficient
  - Statistical significance testing

#### Memory Efficiency
- **Target**: <8GB peak memory usage
- **Validation Method**: Memory profiling and leak detection
- **Test Scenarios**:
  - Video processing memory usage testing
  - Simulation engine memory efficiency testing
  - Long-running operation leak detection

### Benchmarking Framework

#### Pytest-Benchmark Integration
Automated performance benchmarking with statistical analysis:
```bash
# Run benchmark tests only
pytest --benchmark-only

# Compare benchmarks with baseline
pytest --benchmark-compare

# Save benchmark results
pytest --benchmark-save=baseline

# Generate benchmark histogram
pytest --benchmark-histogram
```

#### Performance Monitoring
Real-time resource utilization tracking:
- CPU usage monitoring during test execution
- Memory allocation and deallocation tracking
- Disk I/O performance measurement
- Network usage monitoring for distributed tests

#### Threshold Validation
Automated compliance checking against performance targets:
```python
@pytest.mark.benchmark
def test_simulation_speed_threshold(benchmark):
    """Validate simulation speed meets <7.2 second requirement"""
    result = benchmark(run_single_simulation)
    assert result.stats.mean < 7.2
```

#### Regression Detection
Performance trend analysis and regression alerts:
- Historical performance data comparison
- Statistical significance testing for performance changes
- Automated alerts for performance degradation
- Trend analysis and prediction

## Scientific Computing Standards

### Numerical Precision

#### Tolerance Thresholds
- **Numerical Tolerance**: 1e-6 for floating-point comparisons
- **Correlation Requirement**: 0.95 minimum for accuracy validation
- **Reproducibility Coefficient**: 0.99 minimum for consistency
- **Statistical Significance**: 0.05 significance level for hypothesis testing

### Validation Methodology

#### Hypothesis Testing
- **Multiple Comparison Correction**: Bonferroni method for family-wise error rate control
- **Power Analysis**: Statistical power calculation for adequate sample sizes
- **Effect Size Analysis**: Cohen's d and eta-squared for practical significance
- **Confidence Intervals**: Bootstrap and parametric methods for interval estimation

#### Effect Size Calculation
- **Cohen's d**: Standardized effect size for mean differences
- **Eta-squared**: Proportion of variance explained by treatment effects
- **Confidence Intervals**: 95% confidence intervals for all effect size estimates
- **Practical Significance**: Minimum detectable effect size determination

#### Confidence Intervals
- **Bootstrap Methods**: Non-parametric confidence interval estimation
- **Parametric Methods**: Normal distribution-based confidence intervals
- **Bias-Corrected Bootstrap**: Improved accuracy for skewed distributions
- **Percentile Methods**: Robust confidence interval estimation

#### Reproducibility Assessment
- **Intraclass Correlation Coefficients**: Consistency measurement across runs
- **Test-Retest Reliability**: Temporal stability validation
- **Inter-Rater Reliability**: Consistency across different configurations
- **Internal Consistency**: Cronbach's alpha for scale reliability

### Cross-Format Compatibility

#### Crimaldi Format Validation
- **Format-Specific Validation**: Specialized testing procedures for Crimaldi datasets
- **Metadata Extraction**: Proper handling of Crimaldi-specific metadata
- **Calibration Parameters**: Validation of format-specific calibration procedures
- **Reference Implementation**: Comparison with established Crimaldi processing methods

#### Custom AVI Validation
- **Generic Format Validation**: Comprehensive AVI format compatibility testing
- **Metadata Handling**: Robust extraction and validation of AVI metadata
- **Codec Compatibility**: Support for various AVI codecs and containers
- **Quality Assessment**: Video quality preservation during processing

#### Conversion Accuracy
- **Lossless Transformation**: Verification of data integrity during format conversion
- **Numerical Precision**: Maintenance of numerical accuracy across formats
- **Metadata Preservation**: Complete metadata transfer between formats
- **Validation Protocols**: Comprehensive testing of conversion procedures

#### Metadata Preservation
- **Format-Specific Information**: Retention of critical format-specific data
- **Calibration Parameters**: Preservation of scale and intensity calibration
- **Temporal Information**: Accurate frame rate and timing preservation
- **Quality Metrics**: Validation of preserved video quality indicators

## Error Handling and Recovery Testing

### Error Scenarios

#### Data Validation Errors
- **Corrupted Video File Handling**
  - Detection of corrupted or incomplete video files
  - Graceful handling of read errors during processing
  - Validation of file integrity before processing
  - Clear error reporting for corrupted data

- **Invalid Configuration Parameter Detection**
  - Validation of algorithm configuration parameters
  - Detection of out-of-range or invalid values
  - Type checking and format validation
  - Configuration file schema validation

- **Format Incompatibility Error Management**
  - Detection of unsupported video formats
  - Handling of format conversion failures
  - Validation of format-specific requirements
  - Clear error messages for format issues

#### Processing Errors
- **Memory Exhaustion Simulation**
  - Testing behavior under memory pressure
  - Graceful handling of out-of-memory conditions
  - Memory usage monitoring and alerting
  - Automatic cleanup and resource recovery

- **Timeout Scenario Handling**
  - Detection and handling of processing timeouts
  - Configurable timeout thresholds
  - Graceful termination of long-running operations
  - Timeout recovery and retry mechanisms

- **File System Error Recovery**
  - Handling of disk space exhaustion
  - Recovery from file permission errors
  - Network file system error handling
  - Temporary file cleanup procedures

#### Simulation Errors
- **Algorithm Execution Failure Handling**
  - Detection of algorithm runtime errors
  - Isolation of failing simulations from batch operations
  - Error reporting with detailed diagnostic information
  - Recovery strategies for algorithm failures

- **Batch Processing Interruption Recovery**
  - Checkpoint-based batch processing resumption
  - Progress preservation during interruptions
  - Partial result recovery and validation
  - Restart mechanisms for interrupted batches

- **Resource Allocation Error Management**
  - Detection of resource allocation failures
  - Dynamic resource adjustment strategies
  - Fallback mechanisms for resource constraints
  - Resource cleanup and recovery procedures

### Recovery Mechanisms

#### Retry Strategies
- **Exponential Backoff**: Progressive delay increase for retry attempts
- **Configurable Retry Limits**: Maximum retry attempts with user configuration
- **Jitter Addition**: Random delay variation to prevent thundering herd
- **Selective Retry**: Different strategies for different error types

#### Checkpoint Recovery
- **Checkpoint-Based Resumption**: Save/restore mechanisms for long operations
- **Progress State Persistence**: Durable storage of processing progress
- **Incremental Checkpointing**: Regular progress saves during batch operations
- **Checkpoint Validation**: Integrity checking of saved state data

#### Graceful Degradation
- **Partial Result Preservation**: Save completed work during failures
- **Progressive Failure Handling**: Continued operation with reduced functionality
- **Quality Degradation**: Acceptable quality reduction under constraints
- **Service Continuation**: Maintained core functionality during partial failures

#### Error Reporting
- **Comprehensive Error Logging**: Detailed error information with context
- **Audit Trail Maintenance**: Complete record of error events and recovery
- **Structured Error Messages**: Consistent format for error reporting
- **Actionable Error Information**: Clear guidance for error resolution

## Troubleshooting and Common Issues

### Common Test Failures

#### Performance Threshold Failures

**Symptoms**: Tests failing due to performance threshold violations
- Simulation time exceeding 7.2 seconds per simulation
- Batch processing time exceeding 8 hours for 4000 simulations
- Memory usage exceeding 8GB peak limit
- Throughput below expected targets

**Causes**:
- **Insufficient System Resources**: Hardware below minimum requirements
- **Suboptimal Parallel Processing Configuration**: Incorrect thread/process counts
- **Memory Allocation Inefficiencies**: Poor memory management patterns
- **I/O Bottlenecks**: Slow disk or network storage access

**Solutions**:
- **Verify System Requirements**: Ensure hardware meets minimum specifications
- **Optimize Parallel Configuration**: Adjust worker count based on CPU cores
- **Monitor Memory Usage**: Use profiling tools to identify memory leaks
- **Optimize I/O Operations**: Use memory mapping and caching strategies

#### Accuracy Validation Failures

**Symptoms**: Correlation accuracy below 95% threshold
- Statistical correlation tests failing
- Numerical precision errors accumulating
- Cross-format comparison discrepancies
- Reproducibility coefficient below 0.99

**Causes**:
- **Numerical Precision Issues**: Floating-point arithmetic limitations
- **Algorithm Implementation Differences**: Inconsistent algorithm implementations
- **Data Normalization Inconsistencies**: Variation in preprocessing procedures
- **Random Seed Variations**: Non-deterministic algorithm behavior

**Solutions**:
- **Verify Numerical Tolerance Settings**: Adjust comparison thresholds appropriately
- **Check Algorithm Implementation Consistency**: Validate algorithm correctness
- **Validate Data Normalization Procedures**: Ensure consistent preprocessing
- **Set Deterministic Random Seeds**: Control randomness for reproducibility

#### Cross-Format Compatibility Issues

**Symptoms**: Format-specific test failures
- Crimaldi format processing errors
- Custom AVI format conversion failures
- Metadata extraction inconsistencies
- Cross-format result discrepancies

**Causes**:
- **Format Conversion Errors**: Incorrect format transformation procedures
- **Metadata Preservation Issues**: Loss of critical format-specific information
- **Calibration Parameter Inconsistencies**: Different calibration procedures
- **Codec Compatibility Problems**: Unsupported video codecs

**Solutions**:
- **Verify Format Conversion Procedures**: Validate conversion algorithms
- **Check Metadata Preservation Mechanisms**: Ensure complete metadata transfer
- **Validate Calibration Parameter Consistency**: Use standardized calibration
- **Test Codec Compatibility**: Verify support for required video codecs

### Debugging Procedures

#### Verbose Test Execution
```bash
# Run tests with maximum verbosity
pytest -vvv

# Show local variables in tracebacks
pytest --tb=long

# Capture and display stdout/stderr
pytest -s

# Show test durations
pytest --durations=10
```

#### Test Isolation
```bash
# Run specific test modules
pytest src/test/unit/test_normalization.py

# Run specific test classes
pytest src/test/unit/test_normalization.py::TestNormalization

# Run specific test functions
pytest src/test/unit/test_normalization.py::TestNormalization::test_scale_calibration

# Run tests matching pattern
pytest -k "normalization and not slow"
```

#### Performance Profiling
```bash
# Profile test execution
pytest --profile

# Memory profiling
pytest --memprof

# Line-by-line profiling
pytest --profile-svg

# Benchmark profiling
pytest --benchmark-only --benchmark-sort=mean
```

#### Log Analysis
```bash
# Enable debug logging
pytest --log-level=DEBUG

# Capture logs to file
pytest --log-file=test.log

# Live log display
pytest --log-cli-level=INFO

# Custom log format
pytest --log-cli-format="%(asctime)s [%(levelname)s] %(message)s"
```

## Continuous Integration and Automation

### CI Configuration

#### GitHub Actions Integration
Automated test execution on code changes:
```yaml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov=src/backend --cov-report=xml
```

#### Test Matrix
Multiple Python versions and operating systems:
- **Python Versions**: 3.9, 3.10, 3.11
- **Operating Systems**: Ubuntu, macOS, Windows
- **Dependency Versions**: Latest stable and pinned versions
- **Configuration Variants**: Different algorithm configurations

#### Performance Regression Detection
Automated performance threshold monitoring:
- **Baseline Comparison**: Compare against established performance baselines
- **Trend Analysis**: Detect performance degradation over time
- **Threshold Alerting**: Automatic alerts for performance violations
- **Regression Reports**: Detailed analysis of performance changes

#### Coverage Reporting
Automated coverage analysis and reporting:
- **Coverage Thresholds**: Minimum 95% code coverage requirement
- **Coverage Trends**: Track coverage changes over time
- **Missing Coverage**: Identify untested code areas
- **Coverage Reports**: Generate detailed coverage reports

### Quality Gates

#### Minimum Test Coverage
- **95% Code Coverage Requirement**: All code must be adequately tested
- **Branch Coverage**: Test both success and failure paths
- **Function Coverage**: Every function must have associated tests
- **Line Coverage**: Critical code lines must be executed during testing

#### Performance Compliance
- **All Performance Thresholds Must Pass**: No performance regressions allowed
- **Simulation Speed**: <7.2 seconds per simulation requirement
- **Batch Processing**: 4000 simulations within 8 hours requirement
- **Memory Usage**: <8GB peak memory usage requirement

#### Accuracy Validation
- **95% Correlation Accuracy Requirement**: Statistical validation required
- **Reproducibility Testing**: >0.99 reproducibility coefficient required
- **Cross-Format Compatibility**: Both Crimaldi and custom formats supported
- **Statistical Significance**: All accuracy claims must be statistically validated

#### Error Handling Coverage
- **Comprehensive Error Scenario Testing**: All error paths must be tested
- **Recovery Mechanism Validation**: Error recovery must be verified
- **Graceful Degradation Testing**: System behavior under stress validated
- **Error Reporting Verification**: Clear error messages and logging required

## Development and Testing Workflow

### Test-Driven Development

#### Unit Test First
1. **Write Unit Tests Before Implementation**
   - Define expected behavior through tests
   - Use tests as specification documentation
   - Ensure comprehensive edge case coverage
   - Validate error handling scenarios

2. **Implementation Guided by Tests**
   - Implement minimum code to pass tests
   - Refactor while maintaining test coverage
   - Add functionality incrementally
   - Validate each change against tests

3. **Continuous Testing During Development**
   - Run tests frequently during development
   - Use fast test subset for rapid feedback
   - Integrate performance tests early
   - Maintain high test coverage throughout

#### Integration Test Validation
1. **Component Interaction Testing**
   - Validate interfaces between components
   - Test data flow through pipeline stages
   - Verify configuration parameter propagation
   - Validate error handling across components

2. **End-to-End Workflow Testing**
   - Complete pipeline execution validation
   - User workflow simulation
   - Performance validation under realistic conditions
   - Cross-format compatibility verification

3. **System Integration Testing**
   - Full system deployment testing
   - External dependency integration
   - Configuration management validation
   - Environment-specific testing

#### Performance Test Integration
1. **Performance Requirements Definition**
   - Define measurable performance targets
   - Establish baseline performance metrics
   - Create performance test scenarios
   - Implement automated performance validation

2. **Continuous Performance Monitoring**
   - Integrate performance tests in CI pipeline
   - Monitor performance trends over time
   - Detect performance regressions early
   - Validate performance under load

3. **Performance Optimization Guidance**
   - Identify performance bottlenecks
   - Guide optimization efforts with data
   - Validate optimization effectiveness
   - Maintain performance documentation

### Code Quality Standards

#### Test Naming Conventions
- **Descriptive Test Names**: Clear indication of test purpose
  ```python
  def test_crimaldi_format_normalization_preserves_temporal_resolution():
      """Test that Crimaldi format normalization maintains temporal accuracy"""
  ```

- **Hierarchical Organization**: Logical grouping of related tests
  ```python
  class TestNormalizationPipeline:
      class TestScaleCalibration:
          def test_pixel_to_physical_conversion_accuracy(self):
  ```

- **Behavior-Driven Naming**: Focus on behavior rather than implementation
  ```python
  def test_simulation_completes_within_time_threshold_for_standard_parameters():
  ```

#### Test Documentation
- **Comprehensive Docstrings**: Detailed description of test purpose and expectations
- **Parameter Documentation**: Clear description of test parameters and fixtures
- **Expected Behavior**: Explicit statement of expected test outcomes
- **Preconditions and Setup**: Documentation of required test setup

#### Assertion Clarity
- **Specific Assertion Messages**: Clear failure messages for debugging
  ```python
  assert correlation > 0.95, f"Correlation {correlation} below required 0.95 threshold"
  ```

- **Multiple Assertions**: Break complex validations into specific assertions
- **Custom Assertion Helpers**: Domain-specific assertion functions
- **Error Context**: Include relevant context in assertion messages

#### Test Isolation
- **Independent Tests**: No dependencies between test functions
- **Clean Setup and Teardown**: Proper resource management
- **Isolated Test Data**: Each test uses independent data
- **No Side Effects**: Tests don't modify global state

## Test Environment Management

### Environment Setup
```bash
# Create isolated test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install test dependencies
pip install -r requirements.txt
pip install -r test-requirements.txt

# Verify environment
python -m pytest --version
python -c "import numpy, scipy, cv2; print('Environment ready')"
```

### Configuration Management
```bash
# Set test configuration
export PYTEST_CURRENT_TEST_CONFIG="test_config.json"
export PLUME_TEST_DATA_PATH="src/test/test_fixtures"
export PLUME_TEMP_DIR="/tmp/plume_tests"

# Verify configuration
python -m pytest --collect-only
```

### Data Management
```bash
# Download test data (if required)
python scripts/download_test_data.py

# Verify test data integrity
python scripts/verify_test_data.py

# Clean test artifacts
python scripts/clean_test_artifacts.py
```

This comprehensive testing framework ensures robust validation of the plume navigation simulation system while maintaining scientific accuracy, performance standards, and cross-format compatibility requirements essential for reproducible research outcomes.