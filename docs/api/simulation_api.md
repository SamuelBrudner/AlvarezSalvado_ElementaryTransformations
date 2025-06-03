# Plume Navigation Simulation System API Documentation

**Version:** 1.0.0  
**Last Updated:** 2024-01-01  
**Target Audience:** Developers and Researchers  
**Format:** Markdown

---

## Table of Contents

1. [Introduction](#introduction)
2. [Simulation Engine API](#simulation-engine-api)
3. [Batch Execution API](#batch-execution-api)
4. [Algorithm Execution API](#algorithm-execution-api)
5. [Performance Metrics API](#performance-metrics-api)
6. [Algorithm Interface API](#algorithm-interface-api)
7. [Error Handling Reference](#error-handling-reference)
8. [Configuration Reference](#configuration-reference)
9. [Examples and Tutorials](#examples-and-tutorials)

---

## Introduction

### Overview

The Plume Navigation Simulation System provides comprehensive APIs for scientific computing workflows focused on navigation algorithm testing, batch processing, and performance analysis. This system supports cross-format plume data processing, standardized algorithm interfaces, and reproducible research outcomes with >95% correlation accuracy and <7.2 seconds average execution time.

### Target Audience

- **Software Developers:** Building applications that integrate plume navigation simulation capabilities
- **Research Scientists:** Conducting scientific studies on navigation algorithms and plume tracking
- **Algorithm Developers:** Implementing and testing custom navigation strategies with standardized interfaces

### Prerequisites

- **Python Version:** 3.9 or higher
- **Core Dependencies:** NumPy 2.1.3+, SciPy 1.15.3+, OpenCV 4.11.0+
- **System Requirements:** 8GB RAM minimum, multi-core CPU recommended for batch processing
- **Knowledge Requirements:** Understanding of navigation algorithms and scientific computing principles

### System Architecture

The simulation system employs a modular architecture with the following key components:

```
┌─────────────────────────────────────────────────────────────┐
│                 Plume Simulation System                     │
├─────────────────────────────────────────────────────────────┤
│  Simulation Engine API          │  Batch Execution API      │
│  • Single simulation execution  │  • Parallel processing    │
│  • Cross-format compatibility   │  • Progress monitoring    │
│  • Performance analysis         │  • Resource optimization  │
├─────────────────────────────────────────────────────────────┤
│  Algorithm Execution API        │  Performance Metrics API  │
│  • Algorithm sandboxing         │  • Statistical analysis   │
│  • Resource coordination        │  • Cross-algorithm comparison│
│  • Result collection           │  • Reproducibility assessment│
├─────────────────────────────────────────────────────────────┤
│  Algorithm Interface API        │  Error Handling System    │
│  • Abstract base classes        │  • Graceful degradation   │
│  • Parameter validation         │  • Recovery mechanisms    │
│  • Scientific standardization   │  • Comprehensive logging  │
└─────────────────────────────────────────────────────────────┘
```

---

## Simulation Engine API

### Overview

The Simulation Engine API provides core simulation orchestration with comprehensive plume navigation simulation execution, algorithm management, data normalization integration, and performance analysis capabilities. It achieves >95% correlation accuracy with reference implementations and maintains <7.2 seconds average execution time for scientific computing standards compliance.

### Key Features

- **Single and batch simulation execution** with comprehensive monitoring
- **Cross-format plume data compatibility** for Crimaldi and custom datasets
- **Real-time performance monitoring** with automated optimization recommendations
- **Scientific reproducibility validation** with >0.99 reproducibility coefficient
- **Comprehensive error handling and recovery** with graceful degradation

### Performance Targets

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Execution Time | <7.2 seconds average per simulation | Automated timing and throughput monitoring |
| Correlation Accuracy | >95% with reference implementations | Statistical validation against benchmarks |
| Reproducibility Coefficient | >0.99 across environments | Cross-platform consistency analysis |
| Batch Completion | 4000+ simulations within 8 hours | High-throughput processing validation |

### Main Classes

#### SimulationEngine

Primary simulation orchestrator with algorithm execution and performance analysis capabilities.

**Constructor:**
```python
def __init__(
    engine_id: str,
    config: SimulationEngineConfig,
    enable_monitoring: bool = True
) -> None
```

**Key Methods:**

##### execute_single_simulation()

Execute single plume navigation simulation with comprehensive data processing, algorithm execution, and performance validation.

```python
def execute_single_simulation(
    self,
    plume_video_path: str,
    algorithm_name: str,
    simulation_config: Dict[str, Any],
    execution_context: Dict[str, Any]
) -> SimulationResult
```

**Parameters:**
- `plume_video_path` (str): Path to the plume video file for simulation
- `algorithm_name` (str): Name of the navigation algorithm to execute
- `simulation_config` (Dict[str, Any]): Configuration parameters for simulation execution
- `execution_context` (Dict[str, Any]): Context information for simulation tracking and correlation

**Returns:** `SimulationResult` - Comprehensive simulation result with performance metrics and quality validation

**Example:**
```python
# Initialize simulation engine
engine = create_simulation_engine(
    engine_id="main_engine",
    engine_config={
        "algorithms": {"infotaxis": {"max_iterations": 1000}},
        "performance_thresholds": {"max_execution_time": 10.0}
    }
)

# Execute single simulation
result = engine.execute_single_simulation(
    plume_video_path="/data/plumes/crimaldi_sample.avi",
    algorithm_name="infotaxis",
    simulation_config={
        "algorithm": {"step_size": 0.1, "convergence_threshold": 1e-6},
        "normalization": {"enable_cross_format": True}
    },
    execution_context={"batch_id": None, "reference_data": {}}
)

print(f"Simulation success: {result.execution_success}")
print(f"Execution time: {result.execution_time_seconds:.3f}s")
print(f"Quality score: {result.calculate_overall_quality_score():.3f}")
```

##### execute_batch_simulation()

Execute comprehensive batch of simulations with parallel processing, progress monitoring, and cross-algorithm analysis.

```python
def execute_batch_simulation(
    self,
    plume_video_paths: List[str],
    algorithm_names: List[str],
    batch_config: Dict[str, Any],
    progress_callback: Callable = None
) -> BatchSimulationResult
```

**Parameters:**
- `plume_video_paths` (List[str]): List of plume video paths for batch processing
- `algorithm_names` (List[str]): List of algorithm names to test
- `batch_config` (Dict[str, Any]): Configuration parameters for batch execution
- `progress_callback` (Callable, optional): Callback function for progress updates

**Returns:** `BatchSimulationResult` - Comprehensive batch simulation result with statistics and analysis

**Example:**
```python
# Execute batch simulation
batch_result = engine.execute_batch_simulation(
    plume_video_paths=[
        "/data/plumes/crimaldi_01.avi",
        "/data/plumes/crimaldi_02.avi",
        "/data/plumes/custom_01.avi"
    ],
    algorithm_names=["infotaxis", "casting", "gradient_following"],
    batch_config={
        "simulation_config": {
            "algorithm": {"max_iterations": 1000},
            "normalization": {"enable_cross_format": True}
        },
        "parallel_workers": 4,
        "checkpoint_interval": 100
    },
    progress_callback=lambda progress, completed, total: 
        print(f"Progress: {progress:.1f}% ({completed}/{total})")
)

print(f"Batch completion: {batch_result.successful_simulations}/{batch_result.total_simulations}")
print(f"Success rate: {batch_result.success_rate:.1%}")
```

##### validate_simulation_setup()

Validate simulation setup including configuration, data compatibility, and algorithm requirements.

```python
def validate_simulation_setup(
    self,
    plume_video_path: str,
    algorithm_name: str,
    simulation_config: Dict[str, Any],
    strict_validation: bool = False
) -> ValidationResult
```

##### analyze_performance()

Analyze simulation performance with comprehensive metrics calculation and validation against scientific thresholds.

```python
def analyze_performance(
    self,
    simulation_results: List[SimulationResult],
    include_cross_algorithm_analysis: bool = True,
    validate_against_thresholds: bool = True
) -> Dict[str, Any]
```

##### optimize_performance()

Optimize simulation engine performance based on execution history and system constraints.

```python
def optimize_performance(
    self,
    optimization_strategy: str,
    apply_optimizations: bool = False
) -> Dict[str, Any]
```

##### get_engine_status()

Get comprehensive engine status including active simulations, performance metrics, and system health.

```python
def get_engine_status(
    self,
    include_detailed_metrics: bool = False,
    include_performance_history: bool = False
) -> Dict[str, Any]
```

#### SimulationEngineConfig

Configuration management for simulation engine operations with comprehensive parameter validation.

**Constructor:**
```python
@dataclass
class SimulationEngineConfig:
    engine_id: str
    algorithm_config: Dict[str, Any]
    performance_thresholds: Dict[str, float]
    enable_batch_processing: bool = True
    enable_performance_monitoring: bool = True
    enable_cross_format_validation: bool = True
    enable_scientific_reproducibility: bool = True
```

**Key Methods:**

##### validate_config()

```python
def validate_config(self) -> ValidationResult
```

##### to_dict()

```python
def to_dict(self) -> Dict[str, Any]
```

##### optimize_for_batch()

```python
def optimize_for_batch(
    self,
    batch_size: int,
    system_resources: Dict[str, Any],
    performance_targets: Dict[str, float]
) -> Dict[str, Any]
```

#### SimulationResult

Comprehensive simulation result with performance metrics and validation status.

**Key Methods:**

##### calculate_overall_quality_score()

```python
def calculate_overall_quality_score(self) -> float
```

##### validate_against_thresholds()

```python
def validate_against_thresholds(
    self,
    performance_thresholds: Dict[str, float]
) -> ValidationResult
```

##### to_dict()

```python
def to_dict(
    self,
    include_detailed_results: bool = False,
    include_scientific_context: bool = True
) -> Dict[str, Any]
```

##### generate_scientific_report()

```python
def generate_scientific_report(
    self,
    include_visualizations: bool = False,
    report_format: str = 'comprehensive'
) -> Dict[str, Any]
```

### Global Functions

#### initialize_simulation_system()

Initialize the comprehensive simulation system with configuration loading and component setup.

```python
def initialize_simulation_system(
    system_config: Dict[str, Any],
    enable_performance_monitoring: bool = True,
    enable_cross_format_validation: bool = True,
    enable_scientific_reproducibility: bool = True
) -> bool
```

#### create_simulation_engine()

Create comprehensive simulation engine instance with algorithm execution and batch processing capabilities.

```python
def create_simulation_engine(
    engine_id: str,
    engine_config: Dict[str, Any],
    enable_batch_processing: bool = True,
    enable_performance_analysis: bool = True
) -> SimulationEngine
```

#### execute_single_simulation()

Execute single plume navigation simulation with comprehensive data normalization and performance analysis.

```python
def execute_single_simulation(
    engine_id: str,
    plume_video_path: str,
    algorithm_name: str,
    simulation_config: Dict[str, Any],
    execution_context: Dict[str, Any]
) -> SimulationResult
```

---

## Batch Execution API

### Overview

The Batch Execution API provides intelligent batch processing coordination for 4000+ simulation execution with parallel processing, resource optimization, and progress monitoring. It achieves 99% completion rate within 8-hour target timeframe with comprehensive error handling and recovery mechanisms.

### Key Features

- **Parallel execution management** with dynamic load balancing
- **Resource optimization** based on system capabilities and constraints
- **Progress monitoring and ETA calculation** with real-time updates
- **Graceful error handling and recovery** with checkpoint-based resumption
- **Comprehensive statistics and analysis** with cross-algorithm comparison

### Performance Specifications

| Specification | Target | Implementation |
|--------------|--------|----------------|
| Target Batch Size | 4000+ simulations | Configurable batch chunking and parallel workers |
| Completion Timeframe | 8 hours maximum | Intelligent resource allocation and optimization |
| Parallel Workers | System-dependent | Dynamic worker count based on available resources |
| Checkpoint Interval | Every 100 simulations | Automatic checkpoint creation and recovery |
| Success Rate Target | 99% completion rate | Robust error handling and retry mechanisms |

### Main Classes

#### BatchExecutor

Comprehensive batch execution with intelligent processing coordination and resource management.

**Constructor:**
```python
def __init__(
    self,
    executor_id: str,
    execution_config: Dict[str, Any],
    enable_parallel_processing: bool = True
) -> None
```

**Key Methods:**

##### execute_batch()

Execute comprehensive batch of simulations with parallel processing and progress monitoring.

```python
def execute_batch(
    self,
    simulation_tasks: List[Callable],
    batch_config: Dict[str, Any],
    progress_callback: Callable = None
) -> ParallelExecutionResult
```

**Parameters:**
- `simulation_tasks` (List[Callable]): List of simulation task functions to execute
- `batch_config` (Dict[str, Any]): Configuration parameters for batch execution
- `progress_callback` (Callable, optional): Optional callback for progress updates

**Returns:** `ParallelExecutionResult` - Comprehensive batch execution result with performance metrics

**Example:**
```python
# Create batch executor
batch_executor = LocalizedBatchExecutor(
    executor_id="batch_main",
    execution_config={
        "worker_count": 8,
        "backend": "threading",
        "memory_mapping": True
    },
    enable_parallel_processing=True
)

# Prepare simulation tasks
simulation_tasks = []
for video_path in plume_video_paths:
    for algorithm_name in algorithm_names:
        task = partial(
            execute_single_simulation,
            engine_id="main_engine",
            plume_video_path=video_path,
            algorithm_name=algorithm_name,
            simulation_config=default_config,
            execution_context={"batch_id": "batch_001"}
        )
        simulation_tasks.append(task)

# Execute batch with progress tracking
result = batch_executor.execute_batch(
    simulation_tasks=simulation_tasks,
    batch_config={
        "chunk_size": 50,
        "timeout_per_task": 300,
        "enable_checkpointing": True
    },
    progress_callback=lambda p, c, t: print(f"Batch progress: {p:.1f}%")
)

print(f"Batch completed: {result.successful_tasks}/{result.total_tasks}")
```

##### validate_batch_setup()

Validate batch execution setup including task compatibility and resource requirements.

```python
def validate_batch_setup(
    self,
    batch_tasks: List[Any],
    validation_config: Dict[str, Any]
) -> ValidationResult
```

##### optimize_execution()

Optimize batch execution parameters based on system resources and performance constraints.

```python
def optimize_execution(
    self,
    performance_metrics: Dict[str, float],
    optimization_strategy: str
) -> Dict[str, Any]
```

##### get_execution_status()

Retrieve current batch execution status with performance metrics and progress information.

```python
def get_execution_status(
    self,
    include_worker_details: bool = False,
    include_performance_history: bool = False
) -> Dict[str, Any]
```

#### BatchExecutionResult

Comprehensive batch execution result with statistics and analysis capabilities.

**Key Methods:**

##### calculate_batch_efficiency()

```python
def calculate_batch_efficiency(self) -> float
```

##### validate_against_targets()

```python
def validate_against_targets(
    self,
    target_metrics: Dict[str, float]
) -> ValidationResult
```

##### generate_cross_algorithm_analysis()

```python
def generate_cross_algorithm_analysis(
    self,
    analysis_config: Dict[str, Any] = None
) -> Dict[str, Any]
```

##### assess_scientific_reproducibility()

```python
def assess_scientific_reproducibility(
    self,
    reproducibility_threshold: float = 0.99
) -> Dict[str, Any]
```

#### BatchExecutionContext

Context manager for scoped batch execution operations with resource management.

```python
class BatchExecutionContext:
    def __enter__(self) -> 'BatchExecutionContext'
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool
    
    def execute_batch(
        self,
        simulation_tasks: List[Callable],
        batch_config: Dict[str, Any]
    ) -> BatchExecutionResult
    
    def get_context_summary(self) -> Dict[str, Any]
```

**Example Usage:**
```python
with BatchExecutionContext(executor_config) as batch_context:
    result = batch_context.execute_batch(
        simulation_tasks=tasks,
        batch_config=config
    )
    summary = batch_context.get_context_summary()
```

### Global Functions

#### execute_batch_simulation()

Execute comprehensive batch of plume navigation simulations with parallel processing and cross-algorithm analysis.

```python
def execute_batch_simulation(
    engine_id: str,
    plume_video_paths: List[str],
    algorithm_names: List[str],
    batch_config: Dict[str, Any],
    progress_callback: Callable = None
) -> BatchSimulationResult
```

---

## Algorithm Execution API

### Overview

The Algorithm Execution API provides centralized algorithm execution management with resource coordination, performance tracking, and result collection for navigation algorithm testing. It supports algorithm sandboxing, parameter validation, and comprehensive performance analysis for scientific computing standards compliance.

### Key Features

- **Algorithm sandboxing and isolation** for reliable execution
- **Parameter validation and optimization** with constraint checking
- **Performance tracking and metrics** with comprehensive analysis
- **Resource allocation and management** with intelligent coordination
- **Execution session management** with context preservation

### Supported Algorithms

- **Infotaxis Navigation:** Information-theoretic navigation with entropy-based decision making
- **Casting Strategies:** Systematic search patterns with adaptive radius adjustment
- **Gradient Following:** Direct gradient ascent with noise handling and optimization
- **Plume Tracking:** Continuous plume boundary tracking with dynamic adaptation
- **Hybrid Navigation Strategies:** Combined approaches with intelligent strategy switching

### Main Classes

#### AlgorithmExecutor

Comprehensive algorithm execution with resource coordination and performance tracking.

**Constructor:**
```python
def __init__(
    self,
    executor_config: Dict[str, Any],
    enable_performance_tracking: bool = True
) -> None
```

**Key Methods:**

##### start_execution_session()

Initialize algorithm execution session with resource allocation and context setup.

```python
def start_execution_session(
    self,
    session_config: Dict[str, Any]
) -> str  # Returns session_id
```

##### execute_single_algorithm()

Execute single algorithm with comprehensive monitoring and resource management.

```python
def execute_single_algorithm(
    self,
    algorithm_name: str,
    normalized_plume_data: np.ndarray,
    algorithm_config: Dict[str, Any],
    execution_context: Dict[str, Any]
) -> ExecutionResult
```

**Parameters:**
- `algorithm_name` (str): Name of the navigation algorithm to execute
- `normalized_plume_data` (np.ndarray): Normalized plume data array for processing
- `algorithm_config` (Dict[str, Any]): Algorithm-specific configuration parameters
- `execution_context` (Dict[str, Any]): Execution context with tracking information

**Returns:** `ExecutionResult` - Comprehensive algorithm execution result with trajectory and metrics

**Example:**
```python
# Initialize algorithm executor
executor = AlgorithmExecutor(
    executor_config={
        "resource_limits": {"memory_mb": 2048, "timeout_seconds": 300},
        "performance_tracking": True,
        "enable_sandboxing": True
    },
    enable_performance_tracking=True
)

# Start execution session
session_id = executor.start_execution_session({
    "session_name": "infotaxis_evaluation",
    "algorithms": ["infotaxis"],
    "resource_allocation": "balanced"
})

# Execute algorithm
result = executor.execute_single_algorithm(
    algorithm_name="infotaxis",
    normalized_plume_data=normalized_data,
    algorithm_config={
        "step_size": 0.1,
        "information_gain_threshold": 0.01,
        "max_iterations": 1000,
        "convergence_tolerance": 1e-6
    },
    execution_context={
        "session_id": session_id,
        "plume_source": "crimaldi_dataset",
        "reference_trajectory": reference_path
    }
)

print(f"Algorithm success: {result.success}")
print(f"Trajectory length: {len(result.trajectory) if result.trajectory is not None else 0}")
print(f"Efficiency score: {result.calculate_efficiency_score():.3f}")
```

##### execute_batch_algorithms()

Execute multiple algorithms in batch with parallel processing and comparison analysis.

```python
def execute_batch_algorithms(
    self,
    algorithm_configs: List[Dict[str, Any]],
    batch_execution_config: Dict[str, Any]
) -> List[ExecutionResult]
```

##### validate_execution_results()

Validate algorithm execution results against performance criteria and scientific standards.

```python
def validate_execution_results(
    self,
    execution_results: List[ExecutionResult],
    validation_criteria: Dict[str, Any]
) -> ValidationResult
```

##### optimize_execution_performance()

Optimize algorithm execution performance based on historical data and system constraints.

```python
def optimize_execution_performance(
    self,
    performance_history: Dict[str, List[float]],
    optimization_strategy: str
) -> Dict[str, Any]
```

#### ExecutionResult

Algorithm execution result with performance metrics and trajectory data.

**Key Methods:**

##### add_performance_metric()

```python
def add_performance_metric(
    self,
    metric_name: str,
    metric_value: float,
    metric_unit: str = ""
) -> None
```

##### set_algorithm_result()

```python
def set_algorithm_result(
    self,
    trajectory: np.ndarray,
    convergence_info: Dict[str, Any]
) -> None
```

##### calculate_efficiency_score()

```python
def calculate_efficiency_score(self) -> float
```

##### to_dict()

```python
def to_dict(
    self,
    include_trajectory: bool = True,
    include_detailed_metrics: bool = False
) -> Dict[str, Any]
```

#### BatchExecutionTask

Individual algorithm execution task for parallel batch processing coordination.

**Key Methods:**

##### validate_task()

```python
def validate_task(self) -> ValidationResult
```

##### estimate_execution_time()

```python
def estimate_execution_time(
    self,
    historical_data: Dict[str, float]
) -> float
```

##### to_dict()

```python
def to_dict(self) -> Dict[str, Any]
```

---

## Performance Metrics API

### Overview

The Performance Metrics API provides advanced statistical analysis and algorithm performance evaluation with cross-format validation and scientific reproducibility assessment. It calculates comprehensive navigation success metrics, path efficiency analysis, temporal dynamics evaluation, and robustness assessment for scientific computing standards compliance.

### Key Features

- **Navigation success metrics calculation** with statistical validation
- **Path efficiency analysis** with optimal path comparison
- **Temporal dynamics evaluation** with response time analysis
- **Robustness assessment** with noise tolerance and adaptability testing
- **Cross-format compatibility validation** with consistency analysis

### Metrics Categories

#### Navigation Success

| Metric | Description | Calculation Method |
|--------|-------------|-------------------|
| `localization_success_rate` | Percentage of successful source localizations | Target reached within tolerance threshold |
| `time_to_target` | Average time to reach plume source | Temporal analysis of trajectory convergence |
| `search_efficiency` | Efficiency of search pattern relative to optimal | Path length ratio and exploration analysis |
| `target_accuracy` | Spatial accuracy of final localization | Distance from true source location |
| `completion_rate` | Percentage of simulations reaching completion | Success/failure ratio analysis |

#### Path Efficiency

| Metric | Description | Calculation Method |
|--------|-------------|-------------------|
| `path_length_ratio` | Actual path length vs. optimal direct path | Trajectory length comparison analysis |
| `directness_index` | Measure of path directness toward target | Vector analysis and deviation metrics |
| `exploration_efficiency` | Effectiveness of area exploration strategy | Coverage analysis and redundancy assessment |
| `redundancy_factor` | Amount of unnecessary revisiting in trajectory | Spatial overlap and backtracking analysis |
| `optimization_score` | Overall path optimization effectiveness | Composite efficiency metric |

#### Temporal Dynamics

| Metric | Description | Calculation Method |
|--------|-------------|-------------------|
| `response_time` | Time to initial plume detection and response | First significant movement analysis |
| `velocity_profile` | Speed characteristics throughout trajectory | Velocity analysis and pattern recognition |
| `acceleration_patterns` | Acceleration/deceleration behavior analysis | Motion dynamics and strategy adaptation |
| `movement_phases` | Distinct phases in navigation strategy | Behavioral pattern segmentation |
| `temporal_consistency` | Consistency of temporal behavior patterns | Statistical variance and stability analysis |

#### Robustness

| Metric | Description | Calculation Method |
|--------|-------------|-------------------|
| `noise_tolerance` | Performance degradation with data noise | Controlled noise injection and comparison |
| `environmental_adaptability` | Adaptation to different plume conditions | Cross-environment performance analysis |
| `parameter_sensitivity` | Sensitivity to algorithm parameter changes | Parameter perturbation analysis |
| `stability_index` | Numerical stability and convergence reliability | Convergence analysis and error propagation |
| `failure_recovery` | Ability to recover from temporary failures | Error recovery and continuation analysis |

### Main Classes

#### PerformanceMetricsCalculator

Comprehensive performance metrics calculation with caching and statistical validation.

**Constructor:**
```python
def __init__(
    self,
    calculation_config: Dict[str, Any],
    enable_caching: bool = True,
    enable_statistical_validation: bool = True
) -> None
```

**Key Methods:**

##### calculate_all_metrics()

Calculate comprehensive performance metrics for simulation results with statistical analysis.

```python
def calculate_all_metrics(
    self,
    simulation_result: SimulationResult,
    include_statistical_analysis: bool = True,
    enable_cross_format_analysis: bool = True
) -> Dict[str, float]
```

**Parameters:**
- `simulation_result` (SimulationResult): Simulation result to analyze
- `include_statistical_analysis` (bool): Whether to include statistical analysis
- `enable_cross_format_analysis` (bool): Whether to enable cross-format analysis

**Returns:** `Dict[str, float]` - Comprehensive metrics dictionary with calculated values

**Example:**
```python
# Initialize performance metrics calculator
calculator = PerformanceMetricsCalculator(
    calculation_config={
        "metrics_categories": ["navigation_success", "path_efficiency", "temporal_dynamics"],
        "statistical_tests": ["correlation", "significance"],
        "reference_benchmarks": "crimaldi_dataset"
    },
    enable_caching=True,
    enable_statistical_validation=True
)

# Calculate metrics for simulation result
metrics = calculator.calculate_all_metrics(
    simulation_result=simulation_result,
    include_statistical_analysis=True,
    enable_cross_format_analysis=True
)

# Display key metrics
print(f"Localization success rate: {metrics['localization_success_rate']:.2%}")
print(f"Path length ratio: {metrics['path_length_ratio']:.3f}")
print(f"Search efficiency: {metrics['search_efficiency']:.3f}")
print(f"Temporal consistency: {metrics['temporal_consistency']:.3f}")
```

##### validate_metrics_accuracy()

Validate calculated metrics against reference implementations and statistical thresholds.

```python
def validate_metrics_accuracy(
    self,
    calculated_metrics: Dict[str, float],
    reference_metrics: Dict[str, float],
    validation_thresholds: Dict[str, float]
) -> ValidationResult
```

##### compare_algorithm_metrics()

Compare performance metrics between different algorithms with statistical significance testing.

```python
def compare_algorithm_metrics(
    self,
    algorithm_metrics: Dict[str, Dict[str, float]],
    comparison_config: Dict[str, Any]
) -> Dict[str, Any]
```

##### generate_metrics_report()

Generate comprehensive metrics report with visualizations and statistical analysis.

```python
def generate_metrics_report(
    self,
    metrics_data: Dict[str, Any],
    report_config: Dict[str, Any]
) -> Dict[str, Any]
```

#### NavigationSuccessAnalyzer

Specialized navigation success metrics analysis with detailed success criteria evaluation.

**Key Methods:**

##### calculate_localization_success_rate()

```python
def calculate_localization_success_rate(
    self,
    trajectories: List[np.ndarray],
    source_locations: List[np.ndarray],
    success_criteria: Dict[str, float]
) -> float
```

##### analyze_time_to_target()

```python
def analyze_time_to_target(
    self,
    trajectory: np.ndarray,
    source_location: np.ndarray,
    temporal_resolution: float
) -> Dict[str, float]
```

#### PathEfficiencyAnalyzer

Specialized path efficiency analysis with optimal path comparison and exploration assessment.

**Key Methods:**

##### calculate_path_optimality()

```python
def calculate_path_optimality(
    self,
    trajectory: np.ndarray,
    start_point: np.ndarray,
    target_point: np.ndarray
) -> Dict[str, float]
```

##### analyze_search_patterns()

```python
def analyze_search_patterns(
    self,
    trajectory: np.ndarray,
    plume_data: np.ndarray,
    pattern_config: Dict[str, Any]
) -> Dict[str, Any]
```

### Global Functions

#### calculate_navigation_success_metrics()

Calculate comprehensive navigation success metrics for trajectory analysis.

```python
def calculate_navigation_success_metrics(
    trajectory: np.ndarray,
    plume_data: np.ndarray,
    source_location: np.ndarray,
    success_criteria: Dict[str, float]
) -> Dict[str, float]
```

#### validate_performance_against_thresholds()

Validate performance metrics against scientific computing thresholds and standards.

```python
def validate_performance_against_thresholds(
    performance_metrics: Dict[str, float],
    performance_thresholds: Dict[str, float],
    validation_config: Dict[str, Any]
) -> ValidationResult
```

---

## Algorithm Interface API

### Overview

The Algorithm Interface API provides standardized algorithm interface and framework for navigation algorithm implementation with scientific computing support. It establishes abstract base classes, parameter validation, performance tracking integration, and result standardization for reproducible research outcomes.

### Key Features

- **Abstract base class for algorithm standardization** with enforced interface compliance
- **Parameter validation and constraints** with scientific computing requirements
- **Performance tracking integration** with comprehensive metrics collection
- **Scientific context management** with reproducibility support
- **Result standardization and validation** with cross-algorithm compatibility

### Implementation Requirements

1. **Inherit from BaseAlgorithm class** and implement required abstract methods
2. **Implement _execute_algorithm() method** with algorithm-specific logic
3. **Define algorithm parameters and constraints** using AlgorithmParameters class
4. **Support performance tracking and validation** with integrated monitoring
5. **Maintain scientific computing standards** with reproducibility and correlation requirements

### Main Classes

#### BaseAlgorithm

Abstract base class for all navigation algorithms providing standardized interface and scientific computing support.

**Constructor:**
```python
def __init__(
    self,
    parameters: AlgorithmParameters,
    execution_config: Dict[str, Any] = None
) -> None
```

**Key Methods:**

##### validate_parameters()

Validate algorithm parameters against constraints and scientific computing requirements.

```python
def validate_parameters(
    self,
    strict_validation: bool = False
) -> ValidationResult
```

**Parameters:**
- `strict_validation` (bool): Enable strict validation with enhanced constraint checking

**Returns:** `ValidationResult` - Parameter validation result with constraint compliance assessment

##### execute()

Execute algorithm with plume data, performance tracking, and comprehensive error handling.

```python
def execute(
    self,
    plume_data: np.ndarray,
    plume_metadata: Dict[str, Any],
    simulation_id: str
) -> AlgorithmResult
```

**Parameters:**
- `plume_data` (np.ndarray): Plume data array for navigation algorithm processing
- `plume_metadata` (Dict[str, Any]): Metadata containing format and calibration information
- `simulation_id` (str): Unique identifier for the simulation run

**Returns:** `AlgorithmResult` - Comprehensive algorithm execution result with trajectory and performance data

**Example Implementation:**
```python
class InfotaxisAlgorithm(BaseAlgorithm):
    """Infotaxis navigation algorithm implementation."""
    
    def __init__(self, parameters: AlgorithmParameters, execution_config: Dict[str, Any] = None):
        # Validate infotaxis-specific parameters
        required_params = ['step_size', 'information_gain_threshold', 'exploration_factor']
        for param in required_params:
            if param not in parameters.parameters:
                raise ValueError(f"Required parameter missing: {param}")
        
        super().__init__(parameters, execution_config)
        
        # Initialize infotaxis-specific state
        self.probability_map = None
        self.information_history = []
    
    def _execute_algorithm(
        self,
        plume_data: np.ndarray,
        plume_metadata: Dict[str, Any],
        context: AlgorithmContext
    ) -> AlgorithmResult:
        """Execute infotaxis algorithm with information-theoretic navigation."""
        
        # Initialize algorithm result
        result = AlgorithmResult(
            algorithm_name=self.algorithm_name,
            simulation_id=context.simulation_id,
            execution_id=context.execution_id
        )
        
        try:
            # Initialize probability map and starting position
            height, width = plume_data.shape[1], plume_data.shape[2]
            self.probability_map = np.ones((height, width)) / (height * width)
            
            start_position = np.array([height // 2, width // 2])
            current_position = start_position.copy()
            trajectory = [current_position.copy()]
            
            # Algorithm parameters
            step_size = self.parameters.parameters['step_size']
            info_threshold = self.parameters.parameters['information_gain_threshold']
            max_iterations = self.parameters.max_iterations
            
            # Infotaxis algorithm main loop
            for iteration in range(max_iterations):
                # Add checkpoint for performance tracking
                context.add_checkpoint(f"iteration_{iteration}", {
                    'position': current_position.tolist(),
                    'information_entropy': float(-np.sum(self.probability_map * np.log(self.probability_map + 1e-10)))
                })
                
                # Calculate information gain for possible moves
                possible_moves = self._get_possible_moves(current_position, step_size, height, width)
                information_gains = []
                
                for move in possible_moves:
                    # Simulate concentration measurement at new position
                    simulated_concentration = self._simulate_measurement(move, plume_data, iteration)
                    
                    # Calculate expected information gain
                    info_gain = self._calculate_information_gain(move, simulated_concentration)
                    information_gains.append(info_gain)
                
                # Select move with highest information gain
                best_move_idx = np.argmax(information_gains)
                next_position = possible_moves[best_move_idx]
                
                # Update position and trajectory
                current_position = next_position
                trajectory.append(current_position.copy())
                
                # Update probability map based on measurement
                actual_concentration = plume_data[iteration % plume_data.shape[0], 
                                                int(current_position[0]), 
                                                int(current_position[1])]
                self._update_probability_map(current_position, actual_concentration)
                
                # Check convergence criteria
                max_probability = np.max(self.probability_map)
                if max_probability > (1.0 - info_threshold):
                    result.converged = True
                    break
                
                # Store information gain history
                self.information_history.append(information_gains[best_move_idx])
            
            # Finalize algorithm result
            result.trajectory = np.array(trajectory)
            result.iterations_completed = iteration + 1
            result.success = True
            
            # Add performance metrics
            result.add_performance_metric('final_entropy', 
                float(-np.sum(self.probability_map * np.log(self.probability_map + 1e-10))))
            result.add_performance_metric('mean_information_gain', 
                float(np.mean(self.information_history)) if self.information_history else 0.0)
            result.add_performance_metric('trajectory_length', float(len(trajectory)))
            
            return result
            
        except Exception as e:
            result.success = False
            result.add_warning(f"Infotaxis execution failed: {str(e)}", "execution_error")
            return result
    
    def _get_possible_moves(self, position: np.ndarray, step_size: float, height: int, width: int) -> List[np.ndarray]:
        """Generate possible moves from current position."""
        moves = []
        directions = [(0, step_size), (0, -step_size), (step_size, 0), (-step_size, 0)]
        
        for dy, dx in directions:
            new_pos = position + np.array([dy, dx])
            if 0 <= new_pos[0] < height and 0 <= new_pos[1] < width:
                moves.append(new_pos)
        
        return moves
    
    def _simulate_measurement(self, position: np.ndarray, plume_data: np.ndarray, time_step: int) -> float:
        """Simulate concentration measurement at given position."""
        time_idx = time_step % plume_data.shape[0]
        y, x = int(position[0]), int(position[1])
        return float(plume_data[time_idx, y, x])
    
    def _calculate_information_gain(self, position: np.ndarray, concentration: float) -> float:
        """Calculate expected information gain for a potential measurement."""
        # Simplified information gain calculation
        current_entropy = -np.sum(self.probability_map * np.log(self.probability_map + 1e-10))
        
        # Estimate entropy after measurement (simplified)
        estimated_entropy = current_entropy * (1.0 - concentration)
        
        return float(current_entropy - estimated_entropy)
    
    def _update_probability_map(self, position: np.ndarray, concentration: float) -> None:
        """Update probability map based on concentration measurement."""
        # Simplified Bayesian update
        y, x = int(position[0]), int(position[1])
        
        # Update probability based on concentration
        if concentration > 0.1:  # Threshold for significant concentration
            # Increase probability near measurement location
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.probability_map.shape[0] and 0 <= nx < self.probability_map.shape[1]:
                        distance = np.sqrt(dy*dy + dx*dx)
                        update_factor = concentration * np.exp(-distance)
                        self.probability_map[ny, nx] *= (1.0 + update_factor)
        
        # Normalize probability map
        self.probability_map /= np.sum(self.probability_map)

# Usage example
infotaxis_params = AlgorithmParameters(
    algorithm_name="infotaxis",
    parameters={
        "step_size": 1.0,
        "information_gain_threshold": 0.01,
        "exploration_factor": 0.1
    },
    convergence_tolerance=1e-6,
    max_iterations=1000
)

infotaxis = InfotaxisAlgorithm(infotaxis_params)
result = infotaxis.execute(plume_data, plume_metadata, simulation_id)
```

##### reset()

Reset algorithm state to initial conditions for fresh execution.

```python
def reset(self) -> None
```

##### get_performance_summary()

Get comprehensive performance summary for algorithm execution history.

```python
def get_performance_summary(
    self,
    history_window: int = 10
) -> Dict[str, Any]
```

##### validate_execution_result()

Validate algorithm execution result against scientific computing standards.

```python
def validate_execution_result(
    self,
    result: AlgorithmResult,
    reference_metrics: Dict[str, float] = None
) -> ValidationResult
```

#### AlgorithmParameters

Data class for algorithm parameters with validation, serialization, and scientific computing context support.

**Constructor:**
```python
@dataclass
class AlgorithmParameters:
    algorithm_name: str
    version: str = ALGORITHM_VERSION
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    convergence_tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE
    max_iterations: int = MAX_ITERATIONS
    enable_performance_tracking: bool = PERFORMANCE_TRACKING_ENABLED
```

**Key Methods:**

##### validate()

```python
def validate(
    self,
    strict_validation: bool = False
) -> ValidationResult
```

##### to_dict()

```python
def to_dict(self) -> Dict[str, Any]
```

##### copy()

```python
def copy(self) -> 'AlgorithmParameters'
```

#### AlgorithmResult

Data container for algorithm execution results with performance metrics and scientific computing context.

**Key Methods:**

##### add_performance_metric()

```python
def add_performance_metric(
    self,
    metric_name: str,
    metric_value: float,
    metric_unit: str = ""
) -> None
```

##### add_warning()

```python
def add_warning(
    self,
    warning_message: str,
    warning_category: str = "general"
) -> None
```

##### to_dict()

```python
def to_dict(
    self,
    include_trajectory: bool = True,
    include_state: bool = True
) -> Dict[str, Any]
```

##### get_summary()

```python
def get_summary(self) -> Dict[str, Any]
```

---

## Error Handling Reference

### Overview

The simulation system implements comprehensive error handling patterns and exception management for robust simulation execution. The error handling framework provides graceful degradation, automatic recovery mechanisms, and detailed error reporting for scientific computing reliability.

### Error Categories

#### Validation Errors

Errors related to parameter validation and data compatibility issues.

**Common Validation Errors:**
- `InvalidAlgorithmParametersError`: Algorithm parameters fail validation constraints
- `IncompatiblePlumeDataError`: Plume data format not compatible with algorithm requirements
- `ConfigurationValidationError`: Configuration parameters violate system requirements

**Example:**
```python
try:
    result = engine.execute_single_simulation(
        plume_video_path="invalid_path.avi",
        algorithm_name="nonexistent_algorithm",
        simulation_config={},
        execution_context={}
    )
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Failed parameters: {e.failed_parameters}")
    print(f"Validation context: {e.validation_context}")
    
    # Handle specific validation errors
    if "plume_video_path" in e.failed_parameters:
        print("Please check that the plume video file exists and is accessible")
    if "algorithm_name" in e.failed_parameters:
        print("Please use a supported algorithm name: infotaxis, casting, gradient_following")
```

#### Execution Errors

Errors occurring during algorithm execution and simulation runtime.

**Common Execution Errors:**
- `AlgorithmConvergenceError`: Algorithm fails to converge within iteration limits
- `ResourceAllocationError`: Insufficient system resources for execution
- `PerformanceThresholdViolationError`: Execution exceeds performance thresholds

**Example:**
```python
try:
    batch_result = engine.execute_batch_simulation(
        plume_video_paths=video_paths,
        algorithm_names=algorithms,
        batch_config=config
    )
except SimulationError as e:
    print(f"Simulation failed: {e.message}")
    print(f"Simulation ID: {e.simulation_id}")
    print(f"Algorithm: {e.algorithm_name}")
    print(f"Context: {e.simulation_context}")
    
    # Implement recovery strategy
    if "timeout" in e.simulation_context.get("error", "").lower():
        # Retry with increased timeout
        config["timeout_seconds"] = config.get("timeout_seconds", 300) * 2
        batch_result = engine.execute_batch_simulation(video_paths, algorithms, config)
```

#### System Errors

Errors related to system resource and infrastructure issues.

**Common System Errors:**
- `MemoryAllocationError`: Insufficient memory for processing large datasets
- `FileSystemAccessError`: Unable to access required files or directories
- `NetworkConnectivityError`: Network-related issues during distributed processing

### Recovery Strategies

#### Graceful Degradation for Batch Processing

When batch processing encounters errors, the system implements graceful degradation to maximize completion rate.

```python
def execute_batch_with_recovery(
    engine: SimulationEngine,
    video_paths: List[str],
    algorithms: List[str],
    config: Dict[str, Any]
) -> BatchSimulationResult:
    """Execute batch with automatic error recovery and graceful degradation."""
    
    try:
        # Attempt full batch execution
        return engine.execute_batch_simulation(video_paths, algorithms, config)
        
    except SimulationError as e:
        print(f"Batch execution failed, attempting recovery: {e.message}")
        
        # Strategy 1: Reduce batch size
        if len(video_paths) > 10:
            chunk_size = len(video_paths) // 2
            chunks = [video_paths[i:i + chunk_size] for i in range(0, len(video_paths), chunk_size)]
            
            results = []
            for chunk in chunks:
                try:
                    chunk_result = engine.execute_batch_simulation(chunk, algorithms, config)
                    results.append(chunk_result)
                except Exception as chunk_error:
                    print(f"Chunk failed, continuing with next: {chunk_error}")
                    continue
            
            # Merge chunk results
            return merge_batch_results(results)
        
        # Strategy 2: Reduce algorithm complexity
        simplified_config = config.copy()
        simplified_config["algorithm"]["max_iterations"] = min(500, config.get("algorithm", {}).get("max_iterations", 1000))
        
        try:
            return engine.execute_batch_simulation(video_paths, algorithms, simplified_config)
        except Exception:
            # Strategy 3: Sequential execution fallback
            return execute_sequential_fallback(engine, video_paths, algorithms, config)
```

#### Automatic Retry with Exponential Backoff

For transient errors, the system implements automatic retry with exponential backoff.

```python
import time
import random

def execute_with_retry(
    operation: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> Any:
    """Execute operation with exponential backoff retry strategy."""
    
    for attempt in range(max_attempts):
        try:
            return operation()
            
        except (ResourceError, ProcessingError) as e:
            if attempt == max_attempts - 1:
                raise e  # Re-raise on final attempt
            
            # Calculate delay with jitter
            delay = base_delay * (backoff_factor ** attempt)
            jitter = random.uniform(0.8, 1.2)
            sleep_time = delay * jitter
            
            print(f"Attempt {attempt + 1} failed: {e.message}. Retrying in {sleep_time:.1f}s")
            time.sleep(sleep_time)
    
    raise Exception("All retry attempts exhausted")

# Usage example
result = execute_with_retry(
    lambda: engine.execute_single_simulation(video_path, algorithm, config, context),
    max_attempts=3,
    base_delay=2.0
)
```

#### Checkpoint-based Recovery for Long Operations

For long-running batch operations, the system supports checkpoint-based recovery.

```python
class CheckpointManager:
    """Manage simulation checkpoints for recovery purposes."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(
        self,
        batch_id: str,
        completed_simulations: List[SimulationResult],
        remaining_tasks: List[Dict[str, Any]]
    ) -> str:
        """Save batch execution checkpoint."""
        checkpoint_data = {
            "batch_id": batch_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "completed_count": len(completed_simulations),
            "remaining_count": len(remaining_tasks),
            "completed_results": [result.to_dict(include_trajectory=False) for result in completed_simulations],
            "remaining_tasks": remaining_tasks
        }
        
        checkpoint_file = self.checkpoint_dir / f"batch_{batch_id}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        return str(checkpoint_file)
    
    def load_checkpoint(self, batch_id: str) -> Dict[str, Any]:
        """Load batch execution checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"batch_{batch_id}.json"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {batch_id}")
        
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    
    def resume_batch_execution(
        self,
        engine: SimulationEngine,
        batch_id: str
    ) -> BatchSimulationResult:
        """Resume batch execution from checkpoint."""
        checkpoint = self.load_checkpoint(batch_id)
        
        print(f"Resuming batch {batch_id}: {checkpoint['completed_count']} completed, {checkpoint['remaining_count']} remaining")
        
        # Execute remaining tasks
        remaining_results = []
        for task in checkpoint["remaining_tasks"]:
            try:
                result = engine.execute_single_simulation(**task)
                remaining_results.append(result)
            except Exception as e:
                print(f"Task failed during recovery: {e}")
                continue
        
        # Combine with completed results
        # Note: In a real implementation, you would reconstruct SimulationResult objects
        # from the checkpoint data
        print(f"Batch recovery completed: {len(remaining_results)} additional simulations")
        
        return create_batch_result_from_checkpoint(checkpoint, remaining_results)
```

### Exception Hierarchy

```python
class PlumeSimulationException(Exception):
    """Base exception for plume simulation system."""
    
    def __init__(self, message: str, error_context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_context = error_context or {}
        self.timestamp = datetime.datetime.now()

class ValidationError(PlumeSimulationException):
    """Exception for validation failures."""
    
    def __init__(
        self,
        message: str,
        validation_type: str,
        validation_context: Dict[str, Any],
        failed_parameters: List[str]
    ):
        super().__init__(message)
        self.validation_type = validation_type
        self.validation_context = validation_context
        self.failed_parameters = failed_parameters

class SimulationError(PlumeSimulationException):
    """Exception for simulation execution failures."""
    
    def __init__(
        self,
        message: str,
        simulation_id: str,
        algorithm_name: str,
        simulation_context: Dict[str, Any]
    ):
        super().__init__(message)
        self.simulation_id = simulation_id
        self.algorithm_name = algorithm_name
        self.simulation_context = simulation_context

class ProcessingError(PlumeSimulationException):
    """Exception for data processing failures."""
    pass

class ResourceError(PlumeSimulationException):
    """Exception for resource allocation failures."""
    pass

class ConfigurationError(PlumeSimulationException):
    """Exception for configuration errors."""
    pass
```

---

## Configuration Reference

### Overview

The simulation system provides comprehensive configuration management for all components and operations. Configuration parameters control simulation behavior, performance characteristics, algorithm settings, and system resources to ensure optimal performance and scientific reproducibility.

### Configuration Categories

#### Simulation Engine Configuration

Core parameters for simulation engine operation and management.

```python
simulation_engine_config = {
    "engine_id": "main_simulation_engine",
    "algorithm_config": {
        "supported_algorithms": ["infotaxis", "casting", "gradient_following", "hybrid"],
        "default_parameters": {
            "max_iterations": 1000,
            "convergence_threshold": 1e-6,
            "timeout_seconds": 300.0
        },
        "algorithm_specific": {
            "infotaxis": {
                "information_gain_threshold": 0.01,
                "exploration_factor": 0.1,
                "probability_map_resolution": 0.5
            },
            "casting": {
                "casting_radius": 2.0,
                "angle_increment": 15.0,
                "search_pattern": "spiral"
            },
            "gradient_following": {
                "gradient_step_size": 0.1,
                "noise_threshold": 0.05,
                "smoothing_window": 5
            }
        }
    },
    "performance_thresholds": {
        "max_execution_time_seconds": 7.2,
        "min_correlation_accuracy": 0.95,
        "min_success_rate": 0.95,
        "max_memory_usage_mb": 1024,
        "min_reproducibility_coefficient": 0.99
    },
    "enable_batch_processing": True,
    "enable_performance_monitoring": True,
    "enable_cross_format_validation": True,
    "enable_scientific_reproducibility": True,
    "plume_normalization_config": {
        "enable_cross_format": True,
        "auto_scaling": True,
        "quality_validation": True,
        "supported_formats": ["avi", "mp4", "crimaldi_custom"],
        "normalization_methods": ["min_max", "z_score", "robust_scaling"]
    },
    "performance_analysis_config": {
        "enable_statistical_analysis": True,
        "enable_performance_tracking": True,
        "enable_reproducibility_assessment": True,
        "metrics_categories": [
            "navigation_success",
            "path_efficiency", 
            "temporal_dynamics",
            "robustness"
        ]
    },
    "error_handling_config": {
        "enable_graceful_degradation": True,
        "max_retry_attempts": 3,
        "enable_checkpoint_recovery": True,
        "retry_backoff_factor": 2.0,
        "enable_partial_results": True
    }
}
```

#### Batch Execution Configuration

Parameters for large-scale batch processing and parallel execution coordination.

```python
batch_execution_config = {
    "batch_size": 4000,
    "parallel_workers": "auto",  # or specific integer
    "parallel_backend": "threading",  # or "multiprocessing"
    "chunk_size": 100,
    "checkpoint_interval": 50,
    "resource_allocation_timeout": 300,
    "memory_mapping_enabled": True,
    "load_balancing_strategy": "dynamic",
    "worker_pool_config": {
        "min_workers": 2,
        "max_workers": 16,
        "idle_timeout": 300,
        "startup_timeout": 60
    },
    "progress_monitoring": {
        "enable_real_time_updates": True,
        "update_interval_seconds": 10,
        "enable_eta_calculation": True,
        "enable_throughput_tracking": True
    },
    "optimization_config": {
        "enable_dynamic_optimization": True,
        "optimization_interval": 1000,
        "performance_threshold_adjustment": True,
        "resource_usage_monitoring": True
    },
    "error_recovery": {
        "enable_automatic_retry": True,
        "max_retry_attempts": 3,
        "retry_delay_seconds": 5.0,
        "enable_graceful_degradation": True,
        "partial_completion_threshold": 0.8
    }
}
```

#### Performance Metrics Configuration

Settings for comprehensive performance analysis and metrics calculation.

```python
performance_metrics_config = {
    "metrics_categories": {
        "navigation_success": {
            "enabled": True,
            "metrics": [
                "localization_success_rate",
                "time_to_target", 
                "search_efficiency",
                "target_accuracy",
                "completion_rate"
            ],
            "calculation_config": {
                "success_distance_threshold": 2.0,
                "time_resolution": 0.1,
                "efficiency_baseline": "optimal_path"
            }
        },
        "path_efficiency": {
            "enabled": True,
            "metrics": [
                "path_length_ratio",
                "directness_index",
                "exploration_efficiency", 
                "redundancy_factor",
                "optimization_score"
            ],
            "calculation_config": {
                "optimal_path_method": "euclidean",
                "exploration_grid_size": 1.0,
                "redundancy_threshold": 0.1
            }
        },
        "temporal_dynamics": {
            "enabled": True,
            "metrics": [
                "response_time",
                "velocity_profile",
                "acceleration_patterns",
                "movement_phases", 
                "temporal_consistency"
            ],
            "calculation_config": {
                "smoothing_window": 5,
                "phase_detection_threshold": 0.2,
                "consistency_metric": "coefficient_of_variation"
            }
        },
        "robustness": {
            "enabled": True,
            "metrics": [
                "noise_tolerance",
                "environmental_adaptability",
                "parameter_sensitivity",
                "stability_index",
                "failure_recovery"
            ],
            "calculation_config": {
                "noise_levels": [0.1, 0.2, 0.5],
                "sensitivity_perturbation": 0.1,
                "stability_iterations": 10
            }
        }
    },
    "correlation_threshold": 0.95,
    "reproducibility_threshold": 0.99,
    "caching_enabled": True,
    "cache_size_mb": 512,
    "statistical_validation": {
        "enable_significance_testing": True,
        "confidence_level": 0.95,
        "multiple_comparison_correction": "bonferroni",
        "minimum_sample_size": 10
    },
    "reference_benchmarks": {
        "crimaldi_dataset": {
            "baseline_metrics_file": "benchmarks/crimaldi_baseline.json",
            "reference_implementations": ["matlab_infotaxis", "python_casting"]
        },
        "synthetic_dataset": {
            "baseline_metrics_file": "benchmarks/synthetic_baseline.json",
            "reference_implementations": ["theoretical_optimal"]
        }
    }
}
```

### Default Values

Core default values used throughout the simulation system when specific configurations are not provided.

```python
DEFAULT_VALUES = {
    # Performance targets
    "target_execution_time_seconds": 7.2,
    "correlation_accuracy_threshold": 0.95,
    "reproducibility_threshold": 0.99,
    "batch_completion_target_hours": 8.0,
    
    # Algorithm parameters
    "default_max_iterations": 1000,
    "default_convergence_tolerance": 1e-6,
    "default_timeout_seconds": 300.0,
    
    # System resources
    "default_memory_limit_mb": 1024,
    "default_parallel_workers": 4,
    "default_batch_chunk_size": 100,
    
    # Error handling
    "default_retry_attempts": 3,
    "default_checkpoint_interval": 50,
    "default_backoff_factor": 2.0,
    
    # Validation thresholds
    "min_plume_temporal_frames": 10,
    "min_plume_spatial_resolution": 32,
    "max_algorithm_warnings": 5,
    
    # Scientific computing
    "numerical_precision_tolerance": 1e-12,
    "statistical_significance_level": 0.05,
    "correlation_significance_threshold": 0.95
}
```

### Configuration Loading and Validation

```python
def load_system_configuration(
    config_file: str = None,
    override_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Load and validate system configuration from file or defaults."""
    
    # Load base configuration
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_configuration()
    
    # Apply parameter overrides
    if override_params:
        config = deep_merge_config(config, override_params)
    
    # Validate configuration
    validation_result = validate_system_configuration(config)
    if not validation_result.is_valid:
        raise ConfigurationError(
            f"Configuration validation failed: {validation_result.validation_errors}",
            config_file=config_file,
            config_section="system_configuration"
        )
    
    return config

def validate_system_configuration(config: Dict[str, Any]) -> ValidationResult:
    """Validate system configuration against requirements."""
    
    validation_result = ValidationResult(
        validation_type="system_configuration",
        is_valid=True
    )
    
    # Validate required sections
    required_sections = [
        "simulation_engine",
        "batch_execution", 
        "performance_metrics",
        "algorithm_interface"
    ]
    
    for section in required_sections:
        if section not in config:
            validation_result.add_error(
                f"Required configuration section missing: {section}",
                severity="HIGH"
            )
            validation_result.is_valid = False
    
    # Validate performance thresholds
    if "performance_thresholds" in config.get("simulation_engine", {}):
        thresholds = config["simulation_engine"]["performance_thresholds"]
        
        if thresholds.get("max_execution_time_seconds", 0) <= 0:
            validation_result.add_error(
                "Maximum execution time must be positive",
                severity="HIGH"
            )
            validation_result.is_valid = False
        
        if not (0 < thresholds.get("min_correlation_accuracy", 0) <= 1):
            validation_result.add_error(
                "Correlation accuracy must be between 0 and 1",
                severity="HIGH"
            )
            validation_result.is_valid = False
    
    # Validate batch configuration
    if "batch_execution" in config:
        batch_config = config["batch_execution"]
        
        if batch_config.get("batch_size", 0) <= 0:
            validation_result.add_error(
                "Batch size must be positive",
                severity="HIGH"
            )
            validation_result.is_valid = False
        
        workers = batch_config.get("parallel_workers")
        if isinstance(workers, int) and workers <= 0:
            validation_result.add_error(
                "Number of parallel workers must be positive",
                severity="MEDIUM"
            )
    
    return validation_result

# Configuration usage example
config = load_system_configuration(
    config_file="config/production.json",
    override_params={
        "simulation_engine": {
            "performance_thresholds": {
                "max_execution_time_seconds": 5.0  # Stricter requirement
            }
        },
        "batch_execution": {
            "parallel_workers": 8  # Use more workers
        }
    }
)

# Initialize system with configuration
success = initialize_simulation_system(
    system_config=config,
    enable_performance_monitoring=True,
    enable_cross_format_validation=True
)
```

---

## Examples and Tutorials

### Overview

This section provides practical examples and tutorials for implementing simulation workflows using the Plume Navigation Simulation System API. Examples progress from basic operations to advanced use cases, demonstrating best practices for scientific computing and reproducible research.

### Basic Simulation Execution

#### Single Algorithm Execution

Learn how to execute a single navigation algorithm with proper configuration and result analysis.

```python
import numpy as np
from pathlib import Path
from plume_simulation import (
    initialize_simulation_system,
    create_simulation_engine,
    SimulationEngineConfig,
    AlgorithmParameters
)

def basic_single_simulation_example():
    """Execute a single infotaxis simulation with comprehensive analysis."""
    
    # Step 1: Initialize the simulation system
    system_config = {
        "parallel_processing": {"worker_count": 4, "backend": "threading"},
        "performance_monitoring": {"enabled": True, "real_time_analysis": True},
        "error_handling": {"graceful_degradation": True, "retry_attempts": 3}
    }
    
    success = initialize_simulation_system(
        system_config=system_config,
        enable_performance_monitoring=True,
        enable_cross_format_validation=True,
        enable_scientific_reproducibility=True
    )
    
    if not success:
        raise RuntimeError("Failed to initialize simulation system")
    
    # Step 2: Create simulation engine with configuration
    engine_config = {
        "algorithms": {
            "infotaxis": {
                "max_iterations": 1000,
                "convergence_threshold": 1e-6,
                "information_gain_threshold": 0.01
            }
        },
        "performance_thresholds": {
            "max_execution_time": 10.0,
            "min_correlation_score": 0.95,
            "min_success_rate": 0.90
        }
    }
    
    engine = create_simulation_engine(
        engine_id="basic_example_engine",
        engine_config=engine_config,
        enable_batch_processing=True,
        enable_performance_analysis=True
    )
    
    # Step 3: Configure simulation parameters
    simulation_config = {
        "algorithm": {
            "step_size": 1.0,
            "information_gain_threshold": 0.01,
            "exploration_factor": 0.1,
            "max_iterations": 1000,
            "convergence_tolerance": 1e-6
        },
        "normalization": {
            "enable_cross_format": True,
            "auto_scaling": True,
            "quality_validation": True
        },
        "performance": {
            "enable_tracking": True,
            "checkpoint_interval": 100,
            "enable_optimization": True
        }
    }
    
    execution_context = {
        "experiment_id": "basic_example_001",
        "researcher": "example_user",
        "purpose": "tutorial_demonstration",
        "reference_data": {},
        "batch_id": None
    }
    
    # Step 4: Execute single simulation
    plume_video_path = "data/samples/crimaldi_plume_001.avi"
    
    try:
        result = engine.execute_single_simulation(
            plume_video_path=plume_video_path,
            algorithm_name="infotaxis",
            simulation_config=simulation_config,
            execution_context=execution_context
        )
        
        # Step 5: Analyze results
        print("=== Simulation Results ===")
        print(f"Execution Success: {result.execution_success}")
        print(f"Execution Time: {result.execution_time_seconds:.3f} seconds")
        print(f"Overall Quality Score: {result.calculate_overall_quality_score():.3f}")
        
        # Display performance metrics
        print("\n=== Performance Metrics ===")
        for metric_name, metric_value in result.performance_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Validate against thresholds
        validation_result = result.validate_against_thresholds({
            "max_execution_time": 10.0,
            "min_correlation": 0.95,
            "min_quality_score": 0.8
        })
        
        print(f"\n=== Validation Results ===")
        print(f"Validation Passed: {validation_result.is_valid}")
        if validation_result.validation_errors:
            print("Validation Errors:")
            for error in validation_result.validation_errors:
                print(f"  - {error}")
        
        if validation_result.warnings:
            print("Warnings:")
            for warning in validation_result.warnings:
                print(f"  - {warning}")
        
        # Generate scientific report
        scientific_report = result.generate_scientific_report(
            include_visualizations=False,
            report_format="comprehensive"
        )
        
        print(f"\n=== Scientific Report Summary ===")
        print(f"Report ID: {scientific_report['report_id']}")
        print(f"Executive Summary: {scientific_report['executive_summary']}")
        
        return result
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return None

# Run the example
if __name__ == "__main__":
    result = basic_single_simulation_example()
```

#### Basic Performance Analysis

Analyze simulation performance and compare against benchmarks.

```python
def basic_performance_analysis_example():
    """Analyze simulation performance with benchmark comparison."""
    
    # Execute multiple simulations for analysis
    simulation_results = []
    algorithms = ["infotaxis", "casting", "gradient_following"]
    
    for algorithm in algorithms:
        for trial in range(5):  # 5 trials per algorithm
            try:
                result = engine.execute_single_simulation(
                    plume_video_path=f"data/samples/trial_{trial:02d}.avi",
                    algorithm_name=algorithm,
                    simulation_config=get_algorithm_config(algorithm),
                    execution_context={"trial_id": trial, "algorithm": algorithm}
                )
                simulation_results.append(result)
            except Exception as e:
                print(f"Trial {trial} failed for {algorithm}: {e}")
                continue
    
    # Analyze performance across algorithms
    performance_analysis = engine.analyze_performance(
        simulation_results=simulation_results,
        include_cross_algorithm_analysis=True,
        validate_against_thresholds=True
    )
    
    print("=== Cross-Algorithm Performance Analysis ===")
    print(f"Total Simulations: {performance_analysis['total_simulations']}")
    print(f"Average Execution Time: {performance_analysis['performance_metrics']['average_execution_time']:.3f}s")
    print(f"Overall Success Rate: {performance_analysis['performance_metrics']['success_rate']:.1%}")
    
    # Display algorithm-specific results
    if 'cross_algorithm_analysis' in performance_analysis:
        cross_analysis = performance_analysis['cross_algorithm_analysis']
        print("\n=== Algorithm Comparison ===")
        for algorithm, metrics in cross_analysis.items():
            print(f"\n{algorithm.upper()}:")
            print(f"  Success Rate: {metrics.get('success_rate', 0):.1%}")
            print(f"  Avg Time: {metrics.get('avg_execution_time', 0):.3f}s")
            print(f"  Efficiency Score: {metrics.get('efficiency_score', 0):.3f}")
    
    return performance_analysis

def get_algorithm_config(algorithm_name: str) -> Dict[str, Any]:
    """Get algorithm-specific configuration."""
    configs = {
        "infotaxis": {
            "algorithm": {
                "step_size": 1.0,
                "information_gain_threshold": 0.01,
                "exploration_factor": 0.1
            }
        },
        "casting": {
            "algorithm": {
                "casting_radius": 2.0,
                "angle_increment": 15.0,
                "search_pattern": "spiral"
            }
        },
        "gradient_following": {
            "algorithm": {
                "gradient_step_size": 0.1,
                "noise_threshold": 0.05,
                "smoothing_window": 5
            }
        }
    }
    
    base_config = {
        "normalization": {"enable_cross_format": True},
        "performance": {"enable_tracking": True}
    }
    
    algorithm_config = configs.get(algorithm_name, {})
    return {**base_config, **algorithm_config}
```

#### Result Validation and Reporting

Comprehensive result validation and scientific reporting.

```python
def result_validation_example(simulation_result):
    """Demonstrate comprehensive result validation and reporting."""
    
    # Step 1: Basic result validation
    print("=== Basic Result Validation ===")
    if not simulation_result.execution_success:
        print("❌ Simulation execution failed")
        return False
    
    print("✅ Simulation execution successful")
    
    # Step 2: Performance threshold validation
    performance_thresholds = {
        "max_execution_time": 7.2,  # Target <7.2 seconds
        "min_correlation": 0.95,    # Target >95% correlation
        "min_quality_score": 0.8    # Minimum quality threshold
    }
    
    validation = simulation_result.validate_against_thresholds(performance_thresholds)
    
    print(f"\n=== Performance Validation ===")
    print(f"Validation Status: {'✅ PASSED' if validation.is_valid else '❌ FAILED'}")
    
    if validation.validation_errors:
        print("Validation Errors:")
        for error in validation.validation_errors:
            print(f"  ❌ {error}")
    
    if validation.warnings:
        print("Validation Warnings:")
        for warning in validation.warnings:
            print(f"  ⚠️  {warning}")
    
    # Step 3: Scientific reproducibility assessment
    print(f"\n=== Scientific Reproducibility ===")
    quality_score = simulation_result.calculate_overall_quality_score()
    print(f"Overall Quality Score: {quality_score:.3f}")
    
    if quality_score >= 0.9:
        print("✅ Excellent scientific quality")
    elif quality_score >= 0.8:
        print("✅ Good scientific quality")
    elif quality_score >= 0.7:
        print("⚠️  Acceptable scientific quality")
    else:
        print("❌ Poor scientific quality - review parameters")
    
    # Step 4: Generate detailed scientific report
    scientific_report = simulation_result.generate_scientific_report(
        include_visualizations=True,
        report_format="comprehensive"
    )
    
    print(f"\n=== Scientific Report Generated ===")
    print(f"Report ID: {scientific_report['report_id']}")
    print(f"Timestamp: {scientific_report['report_timestamp']}")
    
    # Save report to file
    report_filename = f"reports/simulation_report_{scientific_report['report_id']}.json"
    Path("reports").mkdir(exist_ok=True)
    
    with open(report_filename, 'w') as f:
        json.dump(scientific_report, f, indent=2)
    
    print(f"Report saved to: {report_filename}")
    
    return validation.is_valid
```

### Batch Processing Workflows

#### Large-Scale Batch Execution

Execute 4000+ simulations with progress monitoring and optimization.

```python
def large_scale_batch_example():
    """Execute large-scale batch processing with 4000+ simulations."""
    
    # Step 1: Prepare batch configuration
    batch_config = {
        "simulation_config": {
            "algorithm": {"max_iterations": 1000, "convergence_tolerance": 1e-6},
            "normalization": {"enable_cross_format": True, "quality_validation": True},
            "performance": {"enable_tracking": True, "checkpoint_interval": 100}
        },
        "parallel_workers": 8,
        "chunk_size": 100,
        "checkpoint_interval": 500,
        "enable_optimization": True,
        "progress_monitoring": {
            "enable_real_time": True,
            "update_interval": 30,
            "enable_eta": True
        },
        "error_recovery": {
            "enable_retry": True,
            "max_attempts": 3,
            "enable_graceful_degradation": True
        }
    }
    
    # Step 2: Prepare video paths and algorithms
    video_base_path = Path("data/plume_videos")
    video_patterns = [
        "crimaldi/*.avi",
        "custom_dataset/*.mp4",
        "synthetic/*.avi"
    ]
    
    plume_video_paths = []
    for pattern in video_patterns:
        plume_video_paths.extend(list(video_base_path.glob(pattern)))
    
    # Limit to first 4000 videos for demonstration
    plume_video_paths = [str(path) for path in plume_video_paths[:4000]]
    
    algorithms = ["infotaxis", "casting", "gradient_following"]
    
    print(f"=== Batch Execution Setup ===")
    print(f"Video files: {len(plume_video_paths)}")
    print(f"Algorithms: {len(algorithms)}")
    print(f"Total simulations: {len(plume_video_paths) * len(algorithms)}")
    print(f"Estimated duration: {estimate_batch_duration(len(plume_video_paths), len(algorithms))}")
    
    # Step 3: Setup progress tracking
    class BatchProgressTracker:
        def __init__(self, total_simulations):
            self.total_simulations = total_simulations
            self.start_time = datetime.datetime.now()
            self.last_update = self.start_time
            
        def update_progress(self, progress_percent, completed, total):
            current_time = datetime.datetime.now()
            elapsed = (current_time - self.start_time).total_seconds()
            
            if progress_percent > 0:
                eta_seconds = (elapsed / progress_percent * 100) - elapsed
                eta = datetime.timedelta(seconds=int(eta_seconds))
            else:
                eta = "Unknown"
            
            # Update every 30 seconds
            if (current_time - self.last_update).total_seconds() >= 30:
                print(f"Progress: {progress_percent:.1f}% ({completed}/{total}) | "
                      f"Elapsed: {datetime.timedelta(seconds=int(elapsed))} | "
                      f"ETA: {eta}")
                self.last_update = current_time
    
    progress_tracker = BatchProgressTracker(len(plume_video_paths) * len(algorithms))
    
    # Step 4: Execute batch with error handling
    try:
        batch_result = engine.execute_batch_simulation(
            plume_video_paths=plume_video_paths,
            algorithm_names=algorithms,
            batch_config=batch_config,
            progress_callback=progress_tracker.update_progress
        )
        
        # Step 5: Analyze batch results
        print(f"\n=== Batch Execution Results ===")
        print(f"Total Simulations: {batch_result.total_simulations}")
        print(f"Successful: {batch_result.successful_simulations}")
        print(f"Failed: {batch_result.failed_simulations}")
        print(f"Success Rate: {batch_result.success_rate:.1%}")
        print(f"Total Execution Time: {batch_result.total_execution_time_seconds / 3600:.2f} hours")
        print(f"Average Time per Simulation: {batch_result.average_execution_time:.3f} seconds")
        
        # Performance targets validation
        target_completion_hours = 8.0
        target_success_rate = 0.99
        actual_hours = batch_result.total_execution_time_seconds / 3600
        
        print(f"\n=== Performance Target Assessment ===")
        print(f"Time Target: {'✅ MET' if actual_hours <= target_completion_hours else '❌ EXCEEDED'} "
              f"({actual_hours:.2f}h / {target_completion_hours}h)")
        print(f"Success Rate Target: {'✅ MET' if batch_result.success_rate >= target_success_rate else '❌ FAILED'} "
              f"({batch_result.success_rate:.1%} / {target_success_rate:.1%})")
        
        # Generate cross-algorithm analysis
        if hasattr(batch_result, 'cross_algorithm_analysis'):
            cross_analysis = batch_result.cross_algorithm_analysis
            print(f"\n=== Cross-Algorithm Performance ===")
            for algorithm, metrics in cross_analysis.items():
                print(f"{algorithm.upper()}:")
                print(f"  Success Rate: {metrics.get('success_rate', 0):.1%}")
                print(f"  Avg Execution Time: {metrics.get('avg_execution_time', 0):.3f}s")
                print(f"  Performance Score: {metrics.get('performance_score', 0):.3f}")
        
        return batch_result
        
    except Exception as e:
        print(f"❌ Batch execution failed: {e}")
        
        # Attempt recovery if checkpoint exists
        checkpoint_manager = CheckpointManager()
        try:
            recovered_result = checkpoint_manager.resume_batch_execution(
                engine=engine,
                batch_id="large_scale_batch"
            )
            print("✅ Successfully recovered from checkpoint")
            return recovered_result
        except Exception as recovery_error:
            print(f"❌ Recovery failed: {recovery_error}")
            return None

def estimate_batch_duration(num_videos, num_algorithms):
    """Estimate batch execution duration."""
    total_simulations = num_videos * num_algorithms
    avg_time_per_simulation = 7.2  # Target average time
    parallel_factor = 0.3  # Efficiency factor for parallel processing
    
    estimated_seconds = total_simulations * avg_time_per_simulation * parallel_factor
    estimated_hours = estimated_seconds / 3600
    
    return f"{estimated_hours:.1f} hours"
```

#### Progress Monitoring and Optimization

Advanced progress monitoring with real-time optimization.

```python
def advanced_progress_monitoring_example():
    """Demonstrate advanced progress monitoring and optimization."""
    
    class AdvancedProgressMonitor:
        def __init__(self, total_simulations, optimization_interval=1000):
            self.total_simulations = total_simulations
            self.optimization_interval = optimization_interval
            self.start_time = datetime.datetime.now()
            self.completed_simulations = 0
            self.performance_history = {
                "execution_times": [],
                "success_rates": [],
                "throughput": []
            }
            self.last_optimization = 0
            
        def update_progress(self, progress_percent, completed, total):
            self.completed_simulations = completed
            current_time = datetime.datetime.now()
            elapsed_seconds = (current_time - self.start_time).total_seconds()
            
            # Calculate performance metrics
            if elapsed_seconds > 0:
                current_throughput = completed / elapsed_seconds  # simulations per second
                self.performance_history["throughput"].append(current_throughput)
            
            # Display progress
            if completed % 100 == 0:  # Update every 100 simulations
                self._display_progress(progress_percent, completed, total, elapsed_seconds)
            
            # Trigger optimization if interval reached
            if completed - self.last_optimization >= self.optimization_interval:
                self._trigger_optimization()
                self.last_optimization = completed
        
        def _display_progress(self, progress_percent, completed, total, elapsed_seconds):
            """Display detailed progress information."""
            eta_seconds = 0
            if progress_percent > 0:
                eta_seconds = (elapsed_seconds / progress_percent * 100) - elapsed_seconds
            
            throughput = completed / elapsed_seconds if elapsed_seconds > 0 else 0
            
            print(f"\n=== Progress Update ===")
            print(f"Completed: {completed:,} / {total:,} ({progress_percent:.1f}%)")
            print(f"Elapsed: {datetime.timedelta(seconds=int(elapsed_seconds))}")
            print(f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}")
            print(f"Throughput: {throughput:.2f} simulations/second")
            
            # Performance trend analysis
            if len(self.performance_history["throughput"]) > 10:
                recent_throughput = self.performance_history["throughput"][-10:]
                trend = self._calculate_trend(recent_throughput)
                print(f"Performance Trend: {trend}")
        
        def _trigger_optimization(self):
            """Trigger performance optimization based on current metrics."""
            print(f"\n=== Triggering Performance Optimization ===")
            
            try:
                current_metrics = {
                    "average_throughput": np.mean(self.performance_history["throughput"][-100:]) if self.performance_history["throughput"] else 0,
                    "completed_simulations": self.completed_simulations
                }
                
                optimization_result = engine.optimize_performance(
                    optimization_strategy="adaptive",
                    apply_optimizations=True
                )
                
                if "expected_improvements" in optimization_result:
                    improvements = optimization_result["expected_improvements"]
                    print(f"Expected Improvements:")
                    for metric, improvement in improvements.items():
                        print(f"  {metric}: +{improvement:.1f}%")
                
            except Exception as e:
                print(f"Optimization failed: {e}")
        
        def _calculate_trend(self, values):
            """Calculate performance trend."""
            if len(values) < 3:
                return "Stable"
            
            recent_avg = np.mean(values[-5:])
            earlier_avg = np.mean(values[:-5])
            
            change_percent = ((recent_avg - earlier_avg) / earlier_avg) * 100 if earlier_avg > 0 else 0
            
            if change_percent > 5:
                return "Improving ↗️"
            elif change_percent < -5:
                return "Degrading ↘️"
            else:
                return "Stable ➡️"
        
        def generate_final_report(self, batch_result):
            """Generate comprehensive progress monitoring report."""
            total_time = (datetime.datetime.now() - self.start_time).total_seconds()
            
            report = {
                "execution_summary": {
                    "total_simulations": self.total_simulations,
                    "completed_simulations": self.completed_simulations,
                    "total_execution_time_hours": total_time / 3600,
                    "average_throughput": np.mean(self.performance_history["throughput"]) if self.performance_history["throughput"] else 0
                },
                "performance_analysis": {
                    "peak_throughput": max(self.performance_history["throughput"]) if self.performance_history["throughput"] else 0,
                    "throughput_stability": np.std(self.performance_history["throughput"]) if len(self.performance_history["throughput"]) > 1 else 0,
                    "optimization_triggers": self.completed_simulations // self.optimization_interval
                },
                "target_compliance": {
                    "time_target_met": total_time <= (8 * 3600),  # 8 hour target
                    "throughput_target_met": self.completed_simulations >= (self.total_simulations * 0.99),  # 99% completion
                    "efficiency_score": min(1.0, (8 * 3600) / total_time) if total_time > 0 else 0
                }
            }
            
            return report
    
    # Usage example
    monitor = AdvancedProgressMonitor(
        total_simulations=4000,
        optimization_interval=500
    )
    
    # Execute batch with advanced monitoring
    batch_result = engine.execute_batch_simulation(
        plume_video_paths=video_paths,
        algorithm_names=algorithms,
        batch_config=batch_config,
        progress_callback=monitor.update_progress
    )
    
    # Generate final monitoring report
    final_report = monitor.generate_final_report(batch_result)
    
    print(f"\n=== Final Progress Report ===")
    print(f"Execution Summary: {final_report['execution_summary']}")
    print(f"Performance Analysis: {final_report['performance_analysis']}")
    print(f"Target Compliance: {final_report['target_compliance']}")
    
    return final_report
```

### Algorithm Development

#### Implementing Custom Algorithms

Create and integrate custom navigation algorithms with the framework.

```python
from plume_simulation.algorithms import BaseAlgorithm, AlgorithmParameters, AlgorithmResult, AlgorithmContext

class CustomHybridAlgorithm(BaseAlgorithm):
    """
    Custom hybrid navigation algorithm combining infotaxis and gradient following.
    
    This algorithm demonstrates how to implement a custom navigation strategy
    using the standardized algorithm interface.
    """
    
    def __init__(self, parameters: AlgorithmParameters, execution_config: Dict[str, Any] = None):
        # Validate hybrid-specific parameters
        required_params = [
            'infotaxis_weight', 'gradient_weight', 'switching_threshold',
            'step_size', 'exploration_factor', 'gradient_smoothing'
        ]
        
        for param in required_params:
            if param not in parameters.parameters:
                raise ValueError(f"Required parameter missing for HybridAlgorithm: {param}")
        
        super().__init__(parameters, execution_config)
        
        # Initialize hybrid-specific state
        self.current_strategy = "exploration"  # or "exploitation"
        self.probability_map = None
        self.gradient_estimate = None
        self.strategy_history = []
        self.performance_metrics_history = {
            "infotaxis_scores": [],
            "gradient_scores": [],
            "strategy_switches": 0
        }
    
    def _execute_algorithm(
        self,
        plume_data: np.ndarray,
        plume_metadata: Dict[str, Any],
        context: AlgorithmContext
    ) -> AlgorithmResult:
        """Execute hybrid navigation algorithm with dynamic strategy switching."""
        
        # Initialize algorithm result
        result = AlgorithmResult(
            algorithm_name=self.algorithm_name,
            simulation_id=context.simulation_id,
            execution_id=context.execution_id
        )
        
        try:
            # Setup algorithm parameters
            infotaxis_weight = self.parameters.parameters['infotaxis_weight']
            gradient_weight = self.parameters.parameters['gradient_weight']
            switching_threshold = self.parameters.parameters['switching_threshold']
            step_size = self.parameters.parameters['step_size']
            
            # Initialize spatial dimensions and starting position
            time_steps, height, width = plume_data.shape[:3]
            start_position = np.array([height // 2, width // 2], dtype=float)
            current_position = start_position.copy()
            trajectory = [current_position.copy()]
            
            # Initialize probability map for infotaxis component
            self.probability_map = np.ones((height, width)) / (height * width)
            
            # Initialize gradient estimate
            self.gradient_estimate = np.zeros((height, width, 2))  # x, y gradients
            
            # Main algorithm loop
            for iteration in range(self.parameters.max_iterations):
                # Add performance checkpoint
                context.add_checkpoint(f"iteration_{iteration}", {
                    'position': current_position.tolist(),
                    'current_strategy': self.current_strategy,
                    'iteration': iteration
                })
                
                # Get current concentration measurement
                time_idx = iteration % time_steps
                y_idx, x_idx = int(current_position[0]), int(current_position[1])
                current_concentration = float(plume_data[time_idx, y_idx, x_idx])
                
                # Calculate infotaxis component
                infotaxis_move, infotaxis_score = self._calculate_infotaxis_move(
                    current_position, current_concentration, plume_data, time_idx
                )
                
                # Calculate gradient following component
                gradient_move, gradient_score = self._calculate_gradient_move(
                    current_position, plume_data, time_idx
                )
                
                # Dynamic strategy selection based on performance
                strategy_scores = {
                    'infotaxis': infotaxis_score,
                    'gradient': gradient_score
                }
                
                # Determine optimal strategy weighting
                if iteration > 50:  # Allow initial exploration
                    optimal_strategy = self._select_optimal_strategy(strategy_scores)
                    if optimal_strategy != self.current_strategy:
                        self.current_strategy = optimal_strategy
                        self.performance_metrics_history["strategy_switches"] += 1
                        context.add_checkpoint(f"strategy_switch_{iteration}", {
                            'new_strategy': optimal_strategy,
                            'scores': strategy_scores
                        })
                
                # Combine moves based on current strategy weights
                if self.current_strategy == "exploration":
                    # Favor infotaxis during exploration
                    combined_move = (infotaxis_weight * infotaxis_move + 
                                   (1 - infotaxis_weight) * gradient_move)
                else:
                    # Favor gradient following during exploitation
                    combined_move = (gradient_weight * gradient_move + 
                                   (1 - gradient_weight) * infotaxis_move)
                
                # Normalize and scale move
                move_magnitude = np.linalg.norm(combined_move)
                if move_magnitude > 0:
                    combined_move = (combined_move / move_magnitude) * step_size
                
                # Update position with boundary checking
                next_position = current_position + combined_move
                next_position[0] = np.clip(next_position[0], 0, height - 1)
                next_position[1] = np.clip(next_position[1], 0, width - 1)
                
                current_position = next_position
                trajectory.append(current_position.copy())
                
                # Update algorithm state
                self._update_probability_map(current_position, current_concentration)
                self._update_gradient_estimate(current_position, plume_data, time_idx)
                
                # Store performance metrics
                self.performance_metrics_history["infotaxis_scores"].append(infotaxis_score)
                self.performance_metrics_history["gradient_scores"].append(gradient_score)
                self.strategy_history.append(self.current_strategy)
                
                # Check convergence criteria
                if self._check_convergence(current_concentration, iteration):
                    result.converged = True
                    break
            
            # Finalize algorithm result
            result.trajectory = np.array(trajectory)
            result.iterations_completed = iteration + 1
            result.success = True
            
            # Add comprehensive performance metrics
            self._add_performance_metrics(result, iteration)
            
            return result
            
        except Exception as e:
            result.success = False
            result.add_warning(f"Hybrid algorithm execution failed: {str(e)}", "execution_error")
            return result
    
    def _calculate_infotaxis_move(
        self, 
        position: np.ndarray, 
        concentration: float, 
        plume_data: np.ndarray, 
        time_idx: int
    ) -> Tuple[np.ndarray, float]:
        """Calculate infotaxis-based move and score."""
        height, width = plume_data.shape[1], plume_data.shape[2]
        possible_moves = self._get_possible_moves(position, height, width)
        
        best_move = np.array([0.0, 0.0])
        best_score = -np.inf
        
        for move in possible_moves:
            # Calculate expected information gain
            simulated_concentration = self._simulate_concentration_at_position(
                position + move, plume_data, time_idx
            )
            information_gain = self._calculate_information_gain(
                position + move, simulated_concentration
            )
            
            if information_gain > best_score:
                best_score = information_gain
                best_move = move
        
        return best_move, best_score
    
    def _calculate_gradient_move(
        self, 
        position: np.ndarray, 
        plume_data: np.ndarray, 
        time_idx: int
    ) -> Tuple[np.ndarray, float]:
        """Calculate gradient-based move and score."""
        # Calculate local gradient
        gradient = self._estimate_local_gradient(position, plume_data, time_idx)
        gradient_magnitude = np.linalg.norm(gradient)
        
        # Score based on gradient strength and consistency
        score = gradient_magnitude * self._calculate_gradient_consistency(position, plume_data, time_idx)
        
        return gradient, score
    
    def _select_optimal_strategy(self, strategy_scores: Dict[str, float]) -> str:
        """Select optimal strategy based on recent performance."""
        # Analyze recent performance history
        recent_window = 20
        if len(self.performance_metrics_history["infotaxis_scores"]) < recent_window:
            return self.current_strategy
        
        recent_infotaxis = np.mean(self.performance_metrics_history["infotaxis_scores"][-recent_window:])
        recent_gradient = np.mean(self.performance_metrics_history["gradient_scores"][-recent_window:])
        
        # Switch strategy if alternative is significantly better
        switching_threshold = self.parameters.parameters['switching_threshold']
        
        if self.current_strategy == "exploration":
            if recent_gradient > recent_infotaxis * (1 + switching_threshold):
                return "exploitation"
        else:
            if recent_infotaxis > recent_gradient * (1 + switching_threshold):
                return "exploration"
        
        return self.current_strategy
    
    def _get_possible_moves(self, position: np.ndarray, height: int, width: int) -> List[np.ndarray]:
        """Generate possible moves from current position."""
        moves = []
        step_size = self.parameters.parameters['step_size']
        
        # 8-directional movement
        for dy in [-step_size, 0, step_size]:
            for dx in [-step_size, 0, step_size]:
                if dy == 0 and dx == 0:
                    continue
                move = np.array([dy, dx], dtype=float)
                new_pos = position + move
                if 0 <= new_pos[0] < height and 0 <= new_pos[1] < width:
                    moves.append(move)
        
        return moves
    
    def _simulate_concentration_at_position(
        self, 
        position: np.ndarray, 
        plume_data: np.ndarray, 
        time_idx: int
    ) -> float:
        """Simulate concentration measurement at given position."""
        y, x = int(np.clip(position[0], 0, plume_data.shape[1] - 1)), \
               int(np.clip(position[1], 0, plume_data.shape[2] - 1))
        return float(plume_data[time_idx, y, x])
    
    def _calculate_information_gain(self, position: np.ndarray, concentration: float) -> float:
        """Calculate expected information gain for infotaxis component."""
        # Simplified information gain calculation
        current_entropy = -np.sum(self.probability_map * np.log(self.probability_map + 1e-10))
        
        # Estimate entropy reduction
        y, x = int(position[0]), int(position[1])
        if 0 <= y < self.probability_map.shape[0] and 0 <= x < self.probability_map.shape[1]:
            local_probability = self.probability_map[y, x]
            information_content = -np.log(local_probability + 1e-10)
            return information_content * concentration
        
        return 0.0
    
    def _estimate_local_gradient(
        self, 
        position: np.ndarray, 
        plume_data: np.ndarray, 
        time_idx: int
    ) -> np.ndarray:
        """Estimate local concentration gradient."""
        y, x = int(position[0]), int(position[1])
        gradient = np.array([0.0, 0.0])
        
        smoothing = self.parameters.parameters['gradient_smoothing']
        
        # Calculate finite differences with smoothing
        if y > smoothing and y < plume_data.shape[1] - smoothing:
            gradient[0] = (np.mean(plume_data[time_idx, y+smoothing:y+2*smoothing, x]) - 
                          np.mean(plume_data[time_idx, y-2*smoothing:y-smoothing, x]))
        
        if x > smoothing and x < plume_data.shape[2] - smoothing:
            gradient[1] = (np.mean(plume_data[time_idx, y, x+smoothing:x+2*smoothing]) - 
                          np.mean(plume_data[time_idx, y, x-2*smoothing:x-smoothing]))
        
        return gradient
    
    def _calculate_gradient_consistency(
        self, 
        position: np.ndarray, 
        plume_data: np.ndarray, 
        time_idx: int
    ) -> float:
        """Calculate gradient consistency over time."""
        # Simplified consistency measure
        gradients = []
        for t_offset in range(-2, 3):
            t_idx = (time_idx + t_offset) % plume_data.shape[0]
            grad = self._estimate_local_gradient(position, plume_data, t_idx)
            gradients.append(grad)
        
        if gradients:
            gradient_std = np.std([np.linalg.norm(g) for g in gradients])
            gradient_mean = np.mean([np.linalg.norm(g) for g in gradients])
            consistency = 1.0 / (1.0 + gradient_std / (gradient_mean + 1e-10))
            return consistency
        
        return 0.0
    
    def _update_probability_map(self, position: np.ndarray, concentration: float) -> None:
        """Update probability map for infotaxis component."""
        y, x = int(position[0]), int(position[1])
        
        # Bayesian update based on concentration measurement
        if concentration > 0.1:  # Significant concentration threshold
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.probability_map.shape[0] and 0 <= nx < self.probability_map.shape[1]:
                        distance = np.sqrt(dy*dy + dx*dx)
                        update_factor = concentration * np.exp(-distance * 0.5)
                        self.probability_map[ny, nx] *= (1.0 + update_factor)
        
        # Normalize probability map
        self.probability_map /= np.sum(self.probability_map)
    
    def _update_gradient_estimate(
        self, 
        position: np.ndarray, 
        plume_data: np.ndarray, 
        time_idx: int
    ) -> None:
        """Update gradient estimate for gradient following component."""
        local_gradient = self._estimate_local_gradient(position, plume_data, time_idx)
        y, x = int(position[0]), int(position[1])
        
        # Update gradient estimate with exponential smoothing
        if 0 <= y < self.gradient_estimate.shape[0] and 0 <= x < self.gradient_estimate.shape[1]:
            alpha = 0.3  # Smoothing factor
            self.gradient_estimate[y, x] = (alpha * local_gradient + 
                                          (1 - alpha) * self.gradient_estimate[y, x])
    
    def _check_convergence(self, concentration: float, iteration: int) -> bool:
        """Check convergence criteria for hybrid algorithm."""
        # Multiple convergence criteria
        convergence_criteria = []
        
        # Concentration-based convergence
        concentration_threshold = 0.8  # High concentration indicates source proximity
        convergence_criteria.append(concentration > concentration_threshold)
        
        # Probability-based convergence (infotaxis component)
        max_probability = np.max(self.probability_map)
        probability_threshold = 0.9
        convergence_criteria.append(max_probability > probability_threshold)
        
        # Gradient-based convergence
        if iteration > 10:
            recent_positions = self.trajectory[-10:] if hasattr(self, 'trajectory') else []
            if recent_positions:
                position_variance = np.var([np.linalg.norm(pos) for pos in recent_positions])
                convergence_criteria.append(position_variance < 0.1)
        
        # Converge if any criterion is met
        return any(convergence_criteria)
    
    def _add_performance_metrics(self, result: AlgorithmResult, final_iteration: int) -> None:
        """Add comprehensive performance metrics to result."""
        # Strategy usage analysis
        strategy_counts = {}
        for strategy in self.strategy_history:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        total_iterations = len(self.strategy_history)
        if total_iterations > 0:
            result.add_performance_metric(
                "exploration_ratio", 
                strategy_counts.get("exploration", 0) / total_iterations
            )
            result.add_performance_metric(
                "exploitation_ratio", 
                strategy_counts.get("exploitation", 0) / total_iterations
            )
        
        # Performance component analysis
        if self.performance_metrics_history["infotaxis_scores"]:
            result.add_performance_metric(
                "avg_infotaxis_score", 
                np.mean(self.performance_metrics_history["infotaxis_scores"])
            )
        
        if self.performance_metrics_history["gradient_scores"]:
            result.add_performance_metric(
                "avg_gradient_score", 
                np.mean(self.performance_metrics_history["gradient_scores"])
            )
        
        # Strategy switching efficiency
        result.add_performance_metric(
            "strategy_switches", 
            float(self.performance_metrics_history["strategy_switches"])
        )
        
        switching_efficiency = (
            self.performance_metrics_history["strategy_switches"] / max(1, final_iteration)
        )
        result.add_performance_metric("switching_efficiency", switching_efficiency)
        
        # Final state metrics
        result.add_performance_metric(
            "final_probability_entropy", 
            float(-np.sum(self.probability_map * np.log(self.probability_map + 1e-10)))
        )
        
        gradient_magnitude = np.mean(np.linalg.norm(self.gradient_estimate, axis=2))
        result.add_performance_metric("final_gradient_strength", float(gradient_magnitude))

# Example usage of custom algorithm
def custom_algorithm_example():
    """Demonstrate custom algorithm implementation and execution."""
    
    # Create custom algorithm parameters
    hybrid_params = AlgorithmParameters(
        algorithm_name="custom_hybrid",
        parameters={
            "infotaxis_weight": 0.6,
            "gradient_weight": 0.7,
            "switching_threshold": 0.2,
            "step_size": 1.0,
            "exploration_factor": 0.1,
            "gradient_smoothing": 2
        },
        convergence_tolerance=1e-6,
        max_iterations=1500,
        enable_performance_tracking=True
    )
    
    # Initialize custom algorithm
    hybrid_algorithm = CustomHybridAlgorithm(
        parameters=hybrid_params,
        execution_config={
            "validation_enabled": True,
            "performance_tracking": {"enabled": True},
            "error_handling": {"timeout_seconds": 300}
        }
    )
    
    # Load test plume data
    plume_data = load_plume_data("data/test/hybrid_test_plume.avi")
    plume_metadata = {
        "format_type": "avi",
        "frame_rate": 30.0,
        "spatial_units": "pixels",
        "temporal_units": "seconds",
        "pixel_to_meter_ratio": 0.01
    }
    
    # Execute custom algorithm
    try:
        result = hybrid_algorithm.execute(
            plume_data=plume_data,
            plume_metadata=plume_metadata,
            simulation_id="custom_hybrid_test_001"
        )
        
        print("=== Custom Hybrid Algorithm Results ===")
        print(f"Execution Success: {result.success}")
        print(f"Converged: {result.converged}")
        print(f"Iterations: {result.iterations_completed}")
        print(f"Trajectory Length: {len(result.trajectory) if result.trajectory is not None else 0}")
        
        # Display custom performance metrics
        print("\n=== Custom Performance Metrics ===")
        custom_metrics = [
            "exploration_ratio", "exploitation_ratio", 
            "avg_infotaxis_score", "avg_gradient_score",
            "strategy_switches", "switching_efficiency"
        ]
        
        for metric in custom_metrics:
            if metric in result.performance_metrics:
                print(f"{metric}: {result.performance_metrics[metric]:.4f}")
        
        # Validate custom algorithm results
        validation = hybrid_algorithm.validate_execution_result(result)
        print(f"\nValidation Status: {'✅ PASSED' if validation.is_valid else '❌ FAILED'}")
        
        return result
        
    except Exception as e:
        print(f"Custom algorithm execution failed: {e}")
        return None

# Run custom algorithm example
if __name__ == "__main__":
    custom_result = custom_algorithm_example()
```

### Cross-Format Analysis

#### Processing Multiple Plume Formats

Handle different plume data formats with consistency validation.

```python
def cross_format_analysis_example():
    """Demonstrate cross-format plume data processing and analysis."""
    
    # Define different plume data sources
    plume_datasets = {
        "crimaldi": {
            "file_paths": list(Path("data/crimaldi").glob("*.avi")),
            "format_config": {
                "format_type": "crimaldi_avi",
                "frame_rate": 15.0,
                "spatial_units": "pixels",
                "pixel_to_meter_ratio": 0.005,
                "intensity_units": "normalized"
            }
        },
        "custom_avi": {
            "file_paths": list(Path("data/custom").glob("*.avi")),
            "format_config": {
                "format_type": "custom_avi",
                "frame_rate": 30.0,
                "spatial_units": "pixels",
                "pixel_to_meter_ratio": 0.01,
                "intensity_units": "raw"
            }
        },
        "synthetic": {
            "file_paths": list(Path("data/synthetic").glob("*.mp4")),
            "format_config": {
                "format_type": "synthetic_mp4",
                "frame_rate": 60.0,
                "spatial_units": "meters",
                "pixel_to_meter_ratio": 1.0,
                "intensity_units": "concentration_ppm"
            }
        }
    }
    
    print("=== Cross-Format Analysis Setup ===")
    for format_name, dataset in plume_datasets.items():
        print(f"{format_name}: {len(dataset['file_paths'])} files")
    
    # Execute simulations across all formats
    format_results = {}
    algorithms = ["infotaxis", "gradient_following"]
    
    for format_name, dataset in plume_datasets.items():
        print(f"\n=== Processing {format_name.upper()} Format ===")
        
        format_results[format_name] = {}
        
        # Take subset for demonstration
        sample_files = dataset["file_paths"][:5]
        
        for algorithm in algorithms:
            algorithm_results = []
            
            for plume_file in sample_files:
                try:
                    # Configure simulation for specific format
                    simulation_config = {
                        "algorithm": get_algorithm_config(algorithm)["algorithm"],
                        "normalization": {
                            "enable_cross_format": True,
                            "format_specific_config": dataset["format_config"],
                            "target_format": "normalized_standard"
                        }
                    }
                    
                    execution_context = {
                        "format_type": format_name,
                        "source_file": str(plume_file),
                        "cross_format_analysis": True
                    }
                    
                    # Execute simulation
                    result = engine.execute_single_simulation(
                        plume_video_path=str(plume_file),
                        algorithm_name=algorithm,
                        simulation_config=simulation_config,
                        execution_context=execution_context
                    )
                    
                    algorithm_results.append(result)
                    
                except Exception as e:
                    print(f"Failed to process {plume_file}: {e}")
                    continue
            
            format_results[format_name][algorithm] = algorithm_results
            print(f"  {algorithm}: {len(algorithm_results)} successful simulations")
    
    # Perform cross-format consistency analysis
    print(f"\n=== Cross-Format Consistency Analysis ===")
    
    consistency_results = {}
    analysis_metrics = [
        "execution_time_seconds",
        "overall_quality_score", 
        "localization_success_rate",
        "path_efficiency"
    ]
    
    for algorithm in algorithms:
        print(f"\n{algorithm.upper()} Cross-Format Analysis:")
        
        algorithm_format_results = {}
        for format_name in format_results:
            if algorithm in format_results[format_name]:
                results = format_results[format_name][algorithm]
                if results:
                    # Calculate format-specific metrics
                    metrics = {}
                    for metric in analysis_metrics:
                        values = []
                        for result in results:
                            if metric == "overall_quality_score":
                                values.append(result.calculate_overall_quality_score())
                            elif metric in result.performance_metrics:
                                values.append(result.performance_metrics[metric])
                            elif hasattr(result, metric.replace("_", "")):
                                values.append(getattr(result, metric.replace("_", "")))
                        
                        if values:
                            metrics[metric] = {
                                "mean": np.mean(values),
                                "std": np.std(values),
                                "values": values
                            }
                    
                    algorithm_format_results[format_name] = metrics
        
        # Analyze cross-format performance
        cross_format_analysis = analyze_cross_format_performance(
            format_results=algorithm_format_results,
            analysis_metrics=analysis_metrics,
            consistency_threshold=0.95,
            include_detailed_analysis=True
        )
        
        consistency_results[algorithm] = cross_format_analysis
        
        # Display analysis results
        if 'compatibility_assessment' in cross_format_analysis:
            assessment = cross_format_analysis['compatibility_assessment']
            print(f"  Overall Consistency: {assessment['overall_consistency_score']:.3f}")
            print(f"  Compatible: {'✅ YES' if assessment['is_compatible'] else '❌ NO'}")
            print(f"  Consistent Metrics: {assessment['consistent_metrics']}/{assessment['total_metrics']}")
        
        if 'format_comparison' in cross_format_analysis:
            print("  Format Comparison:")
            for metric, comparison in cross_format_analysis['format_comparison'].items():
                crimaldi_val = comparison.get('crimaldi_value', 'N/A')
                custom_val = comparison.get('custom_value', 'N/A')
                significant = comparison.get('significant_difference', False)
                print(f"    {metric}: Crimaldi={crimaldi_val:.3f}, Custom={custom_val:.3f} {'⚠️' if significant else '✅'}")
    
    # Generate comprehensive cross-format report
    cross_format_report = generate_cross_format_report(
        consistency_results=consistency_results,
        format_datasets=plume_datasets,
        analysis_metrics=analysis_metrics
    )
    
    print(f"\n=== Cross-Format Report Generated ===")
    print(f"Report ID: {cross_format_report['report_id']}")
    print(f"Formats Analyzed: {len(cross_format_report['formats_analyzed'])}")
    print(f"Algorithms Tested: {len(cross_format_report['algorithms_tested'])}")
    print(f"Overall Compatibility Score: {cross_format_report['overall_compatibility_score']:.3f}")
    
    # Save cross-format analysis results
    report_file = f"reports/cross_format_analysis_{cross_format_report['report_id']}.json"
    Path("reports").mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(cross_format_report, f, indent=2, default=str)
    
    print(f"Cross-format analysis saved to: {report_file}")
    
    return cross_format_report

def generate_cross_format_report(
    consistency_results: Dict[str, Any],
    format_datasets: Dict[str, Any],
    analysis_metrics: List[str]
) -> Dict[str, Any]:
    """Generate comprehensive cross-format analysis report."""
    
    report = {
        "report_id": str(uuid.uuid4()),
        "report_timestamp": datetime.datetime.now().isoformat(),
        "formats_analyzed": list(format_datasets.keys()),
        "algorithms_tested": list(consistency_results.keys()),
        "analysis_metrics": analysis_metrics,
        "format_specifications": {},
        "algorithm_consistency": {},
        "cross_format_validation": {},
        "recommendations": []
    }
    
    # Document format specifications
    for format_name, dataset in format_datasets.items():
        report["format_specifications"][format_name] = {
            "file_count": len(dataset["file_paths"]),
            "format_config": dataset["format_config"]
        }
    
    # Analyze algorithm consistency across formats
    overall_consistency_scores = []
    
    for algorithm, analysis in consistency_results.items():
        if 'compatibility_assessment' in analysis:
            assessment = analysis['compatibility_assessment']
            consistency_score = assessment['overall_consistency_score']
            overall_consistency_scores.append(consistency_score)
            
            report["algorithm_consistency"][algorithm] = {
                "consistency_score": consistency_score,
                "is_compatible": assessment['is_compatible'],
                "consistent_metrics": assessment['consistent_metrics'],
                "total_metrics": assessment['total_metrics'],
                "recommendations": assessment.get('recommendations', [])
            }
    
    # Calculate overall compatibility score
    if overall_consistency_scores:
        report["overall_compatibility_score"] = np.mean(overall_consistency_scores)
    else:
        report["overall_compatibility_score"] = 0.0
    
    # Generate cross-format validation summary
    report["cross_format_validation"] = {
        "validation_passed": report["overall_compatibility_score"] >= 0.95,
        "critical_issues": [],
        "warnings": [],
        "validation_criteria": {
            "consistency_threshold": 0.95,
            "minimum_correlation": 0.90,
            "maximum_variance": 0.20
        }
    }
    
    # Identify critical issues and warnings
    for algorithm, consistency in report["algorithm_consistency"].items():
        if not consistency["is_compatible"]:
            report["cross_format_validation"]["critical_issues"].append(
                f"Algorithm {algorithm} shows format incompatibility"
            )
        elif consistency["consistency_score"] < 0.90:
            report["cross_format_validation"]["warnings"].append(
                f"Algorithm {algorithm} has low consistency score: {consistency['consistency_score']:.3f}"
            )
    
    # Generate recommendations
    if report["overall_compatibility_score"] >= 0.95:
        report["recommendations"].append(
            "✅ Excellent cross-format compatibility - ready for production use"
        )
    elif report["overall_compatibility_score"] >= 0.85:
        report["recommendations"].extend([
            "⚠️ Good cross-format compatibility with minor inconsistencies",
            "Review format-specific normalization parameters",
            "Consider additional validation testing"
        ])
    else:
        report["recommendations"].extend([
            "❌ Poor cross-format compatibility detected",
            "Review and adjust normalization algorithms",
            "Investigate format-specific processing differences",
            "Consider separate processing pipelines for incompatible formats"
        ])
    
    return report
```

This comprehensive API documentation provides developers and researchers with detailed guidance for implementing navigation algorithm testing, batch simulation execution, and performance evaluation workflows using the Plume Navigation Simulation System. The examples demonstrate practical usage patterns while maintaining scientific computing standards and reproducible research outcomes.

---

**End of Documentation**

For additional support and advanced use cases, please refer to the system's technical specifications and contact the development team for specialized integration requirements.