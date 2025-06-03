# Adding New Navigation Algorithms

## Overview

This comprehensive developer guide provides detailed instructions for implementing new navigation algorithms that integrate seamlessly with the plume simulation system while maintaining scientific computing standards and reproducibility requirements. The guide covers algorithm interface compliance, parameter validation, performance optimization, testing strategies, and documentation requirements for reproducible research outcomes across different plume recording formats.

All algorithms must achieve >95% correlation with reference implementations and maintain >0.99 reproducibility coefficient across different computational environments while meeting the <7.2 seconds average processing time requirement.

## Prerequisites

Before implementing a new navigation algorithm, ensure you have:

- **Python 3.9+**: Core programming language with modern feature support
- **NumPy 2.1.3+**: Fundamental package for scientific computing with arrays
- **SciPy 1.15.3+**: Scientific computing library for optimization and statistics
- **Scientific computing principles**: Understanding of numerical methods and precision
- **Navigation algorithm theory**: Knowledge of odor plume navigation strategies
- **Plume simulation concepts**: Familiarity with turbulent flow and source localization

## Algorithm Implementation Requirements

### BaseAlgorithm Interface Compliance

All algorithms must inherit from the `BaseAlgorithm` class and implement the required abstract methods including `_execute_algorithm`, `validate_parameters`, and `reset` methods. This ensures consistent interface across all navigation algorithms in the system.

**Basic Algorithm Structure:**

```python
from src.backend.algorithms.base_algorithm import BaseAlgorithm, AlgorithmParameters, AlgorithmResult, AlgorithmContext

class MyNavigationAlgorithm(BaseAlgorithm):
    def __init__(self, parameters: AlgorithmParameters, execution_config: Dict[str, Any]):
        super().__init__(parameters, execution_config)
        self.algorithm_name = 'my_navigation_algorithm'
        self.version = '1.0.0'
        # Initialize algorithm-specific state
        
    def _execute_algorithm(self, plume_data: numpy.ndarray, plume_metadata: Dict[str, Any], context: AlgorithmContext) -> AlgorithmResult:
        # Implement algorithm-specific navigation logic
        # Return AlgorithmResult with trajectory and performance metrics
        pass
```

### Parameter Validation

Implement comprehensive parameter validation using the `AlgorithmParameters` class with scientific computing constraints and cross-parameter dependency checking. This ensures algorithm robustness and prevents invalid configurations from affecting simulation results.

**Parameter Validation Implementation:**

```python
def validate_parameters(self, strict_validation: bool = True) -> ValidationResult:
    validation_result = super().validate_parameters(strict_validation)
    
    # Algorithm-specific parameter validation
    if self.parameters.get('search_radius', 0) <= 0:
        validation_result.add_error('search_radius must be positive')
    
    if self.parameters.get('convergence_tolerance', 1e-6) < 1e-12:
        validation_result.add_warning('convergence_tolerance may be too strict')
    
    return validation_result
```

### Performance Tracking Integration

Integrate with the performance tracking system to ensure <7.2 seconds average execution time and >95% correlation with reference implementations. The system provides comprehensive monitoring of algorithm performance across different plume datasets.

**Performance Tracking Integration:**

```python
def _execute_algorithm(self, plume_data: numpy.ndarray, plume_metadata: Dict[str, Any], context: AlgorithmContext) -> AlgorithmResult:
    with context:
        context.add_checkpoint('algorithm_start', {'plume_shape': plume_data.shape})
        
        # Algorithm execution with performance monitoring
        trajectory = self._navigate_plume(plume_data, plume_metadata)
        
        context.add_checkpoint('navigation_complete', {'trajectory_length': len(trajectory)})
        
        # Create result with performance metrics
        result = AlgorithmResult(self.algorithm_name, context.simulation_id, True)
        result.trajectory = trajectory
        result.add_performance_metric('execution_time', context.execution_time, 'seconds')
        
        return result
```

## Algorithm Registry Integration

### Registration Process

Register algorithms with the global algorithm registry for discovery, instantiation, and management across the simulation system. The registry provides dynamic loading capabilities and metadata management for algorithm selection and configuration.

**Algorithm Registration:**

```python
from src.backend.algorithms.algorithm_registry import register_algorithm

# Register algorithm with metadata
algorithm_metadata = {
    'algorithm_type': 'bio_inspired',
    'version': '1.0.0',
    'capabilities': {
        'supports_3d_navigation': True,
        'requires_gradient_information': False,
        'adaptive_search_radius': True
    },
    'performance_characteristics': {
        'typical_execution_time': 5.2,
        'memory_usage_mb': 128,
        'convergence_rate': 0.95
    },
    'supported_formats': ['crimaldi', 'custom_avi'],
    'validation_requirements': {
        'correlation_threshold': 0.95,
        'reproducibility_threshold': 0.99
    }
}

register_algorithm(
    algorithm_name='my_navigation_algorithm',
    algorithm_class=MyNavigationAlgorithm,
    algorithm_metadata=algorithm_metadata,
    validate_interface=True,
    enable_performance_tracking=True
)
```

### Dynamic Loading Support

Implement dynamic loading capabilities to avoid circular dependencies while maintaining full functionality and integration. This approach enables modular algorithm development and deployment without affecting the core simulation system.

**Dynamic Loading Implementation:**

```python
def register_my_algorithm(force_reload: bool = False, validate_integration: bool = True) -> bool:
    """Register my navigation algorithm using dynamic loading."""
    try:
        if not force_reload and 'my_navigation_algorithm' in _global_algorithm_registry:
            return True
            
        # Load algorithm class dynamically
        algorithm_class = load_algorithm_dynamically(
            algorithm_name='my_navigation_algorithm',
            module_path='src.backend.algorithms.my_navigation_algorithm',
            class_name='MyNavigationAlgorithm',
            cache_result=True
        )
        
        if algorithm_class is None:
            return False
            
        # Register with full metadata
        return register_algorithm(
            algorithm_name='my_navigation_algorithm',
            algorithm_class=algorithm_class,
            algorithm_metadata=get_algorithm_metadata(),
            validate_interface=validate_integration
        )
        
    except Exception as e:
        logger.error(f"Failed to register my_navigation_algorithm: {e}")
        return False
```

## Scientific Computing Standards

### Numerical Precision Requirements

Maintain numerical precision with 1e-6 tolerance, implement proper convergence criteria, and ensure reproducible results across different computational environments. This is critical for achieving the >95% correlation requirement with reference implementations.

**Numerical Precision Implementation:**

```python
from src.backend.utils.scientific_constants import NUMERICAL_PRECISION_THRESHOLD, DEFAULT_CORRELATION_THRESHOLD

def _validate_numerical_stability(self, calculation_result: numpy.ndarray) -> bool:
    """Validate numerical stability of algorithm calculations."""
    # Check for NaN or infinite values
    if not numpy.isfinite(calculation_result).all():
        return False
        
    # Validate precision against threshold
    if numpy.any(numpy.abs(calculation_result) < NUMERICAL_PRECISION_THRESHOLD):
        self.logger.warning('Calculation approaching numerical precision limits')
        
    return True

def _ensure_reproducibility(self, random_seed: int = 42) -> None:
    """Ensure reproducible algorithm execution."""
    numpy.random.seed(random_seed)
    # Set any other algorithm-specific random states
```

### Statistical Validation

Implement statistical validation against reference implementations with >95% correlation requirement and >0.99 reproducibility coefficient. This validation ensures scientific rigor and enables meaningful comparison of algorithm performance.

**Statistical Validation Implementation:**

```python
from src.backend.utils.statistical_utils import calculate_correlation_matrix, assess_reproducibility

def validate_algorithm_performance(self, algorithm_result: AlgorithmResult, reference_metrics: Dict[str, float]) -> Dict[str, Any]:
    """Validate algorithm performance against scientific standards."""
    validation_result = {
        'correlation_analysis': {},
        'reproducibility_assessment': {},
        'compliance_status': {}
    }
    
    # Calculate correlation with reference implementation
    correlation_matrix = calculate_correlation_matrix(
        algorithm_result.trajectory,
        reference_metrics.get('reference_trajectory')
    )
    
    correlation_coefficient = correlation_matrix[0, 1]
    validation_result['correlation_analysis'] = {
        'coefficient': correlation_coefficient,
        'meets_threshold': correlation_coefficient >= DEFAULT_CORRELATION_THRESHOLD,
        'threshold': DEFAULT_CORRELATION_THRESHOLD
    }
    
    # Assess reproducibility
    reproducibility_metrics = assess_reproducibility(
        algorithm_result.performance_metrics,
        reference_metrics
    )
    
    validation_result['reproducibility_assessment'] = reproducibility_metrics
    validation_result['compliance_status'] = {
        'correlation_compliant': correlation_coefficient >= DEFAULT_CORRELATION_THRESHOLD,
        'reproducibility_compliant': reproducibility_metrics['icc'] >= 0.99,
        'overall_compliant': all([
            correlation_coefficient >= DEFAULT_CORRELATION_THRESHOLD,
            reproducibility_metrics['icc'] >= 0.99
        ])
    }
    
    return validation_result
```

## Cross-Format Compatibility

### Plume Data Normalization

Implement cross-format compatibility for Crimaldi and custom plume data formats using the plume normalization system. This ensures algorithms work consistently across different experimental datasets and recording formats.

**Cross-Format Data Handling:**

```python
from src.backend.core.data_normalization.plume_normalizer import PlumeNormalizer

def _prepare_plume_data(self, plume_data: numpy.ndarray, plume_metadata: Dict[str, Any]) -> Tuple[numpy.ndarray, Dict[str, Any]]:
    """Prepare plume data for algorithm execution with cross-format support."""
    # Initialize plume normalizer
    normalizer = PlumeNormalizer()
    
    # Normalize plume data for cross-format compatibility
    normalized_data = normalizer.normalize_plume_data(
        plume_data=plume_data,
        source_format=plume_metadata.get('format', 'unknown'),
        target_format='standard',
        preserve_physical_properties=True
    )
    
    # Validate normalization quality
    normalization_quality = normalizer.validate_normalization_quality(
        original_data=plume_data,
        normalized_data=normalized_data,
        metadata=plume_metadata
    )
    
    if not normalization_quality.is_valid:
        raise ValidationError(f"Plume normalization failed: {normalization_quality.errors}")
    
    # Update metadata with normalization information
    updated_metadata = plume_metadata.copy()
    updated_metadata.update({
        'normalized': True,
        'normalization_quality': normalization_quality.to_dict(),
        'original_format': plume_metadata.get('format'),
        'target_format': 'standard'
    })
    
    return normalized_data, updated_metadata
```

### Format-Specific Adaptations

Handle format-specific requirements and adaptations while maintaining algorithm consistency across different plume data sources. This ensures optimal performance regardless of the input data format.

**Format-Specific Handling:**

```python
def _adapt_to_plume_format(self, plume_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt algorithm parameters based on plume data format."""
    format_adaptations = {}
    
    plume_format = plume_metadata.get('format', 'unknown')
    
    if plume_format == 'crimaldi':
        # Crimaldi-specific adaptations
        format_adaptations.update({
            'temporal_resolution_factor': plume_metadata.get('fps', 30) / 30.0,
            'spatial_resolution_factor': plume_metadata.get('pixel_size_mm', 1.0),
            'intensity_scaling': plume_metadata.get('intensity_units', 'normalized')
        })
    elif plume_format == 'custom_avi':
        # Custom AVI format adaptations
        format_adaptations.update({
            'temporal_resolution_factor': plume_metadata.get('sampling_rate', 25) / 25.0,
            'spatial_resolution_factor': plume_metadata.get('resolution_mm_per_pixel', 1.0),
            'intensity_scaling': 'raw_values'
        })
    
    # Apply format adaptations to algorithm parameters
    adapted_parameters = self.parameters.copy()
    if 'temporal_resolution_factor' in format_adaptations:
        adapted_parameters.parameters['search_radius'] *= format_adaptations['spatial_resolution_factor']
        adapted_parameters.parameters['time_step'] *= format_adaptations['temporal_resolution_factor']
    
    return adapted_parameters
```

## Testing and Validation

### Unit Testing Requirements

Implement comprehensive unit tests covering algorithm functionality, parameter validation, performance requirements, and error handling. Testing ensures algorithm reliability and validates compliance with system requirements.

**Algorithm Unit Tests:**

```python
import pytest
from src.test.utils.test_helpers import assert_simulation_accuracy, measure_performance
from src.test.utils.validation_metrics import validate_performance_thresholds

class TestMyNavigationAlgorithm:
    def setup_method(self):
        self.algorithm_params = AlgorithmParameters(
            algorithm_name='my_navigation_algorithm',
            parameters={'search_radius': 5.0, 'convergence_tolerance': 1e-6},
            constraints={'search_radius': {'min': 0.1, 'max': 50.0}}
        )
        self.algorithm = MyNavigationAlgorithm(self.algorithm_params, {})
    
    @measure_performance(time_limit_seconds=7.2)
    def test_algorithm_execution_performance(self):
        """Test algorithm execution meets <7.2 seconds requirement."""
        plume_data = numpy.random.rand(100, 100, 50)  # Test plume data
        plume_metadata = {'format': 'test', 'fps': 30}
        
        result = self.algorithm.execute(plume_data, plume_metadata, 'test_sim_001')
        
        assert result.success
        assert result.execution_time < 7.2
        assert_simulation_accuracy(result, correlation_threshold=0.95)
    
    def test_parameter_validation(self):
        """Test comprehensive parameter validation."""
        # Test valid parameters
        validation_result = self.algorithm.validate_parameters()
        assert validation_result.is_valid
        
        # Test invalid parameters
        invalid_params = AlgorithmParameters(
            algorithm_name='my_navigation_algorithm',
            parameters={'search_radius': -1.0},  # Invalid negative radius
            constraints={'search_radius': {'min': 0.1, 'max': 50.0}}
        )
        invalid_algorithm = MyNavigationAlgorithm(invalid_params, {})
        validation_result = invalid_algorithm.validate_parameters()
        assert not validation_result.is_valid
        assert 'search_radius' in str(validation_result.errors)
    
    def test_cross_format_compatibility(self):
        """Test algorithm works with different plume formats."""
        formats_to_test = ['crimaldi', 'custom_avi']
        
        for format_name in formats_to_test:
            plume_data = numpy.random.rand(50, 50, 25)
            plume_metadata = {'format': format_name, 'fps': 30}
            
            result = self.algorithm.execute(plume_data, plume_metadata, f'test_{format_name}')
            
            assert result.success
            assert result.trajectory is not None
            validate_performance_thresholds(result.performance_metrics)
```

### Integration Testing

Test algorithm integration with the simulation system, batch processing, and cross-algorithm compatibility. Integration tests validate end-to-end system functionality and algorithm interoperability.

**Integration Tests:**

```python
def test_algorithm_registry_integration():
    """Test algorithm integration with registry system."""
    from src.backend.algorithms.algorithm_registry import get_algorithm, create_algorithm_instance
    
    # Test algorithm retrieval from registry
    algorithm_class = get_algorithm('my_navigation_algorithm')
    assert algorithm_class is not None
    assert issubclass(algorithm_class, BaseAlgorithm)
    
    # Test algorithm instance creation
    algorithm_instance = create_algorithm_instance(
        algorithm_name='my_navigation_algorithm',
        parameters=self.algorithm_params,
        execution_config={},
        instance_id='integration_test_001'
    )
    assert isinstance(algorithm_instance, MyNavigationAlgorithm)

def test_batch_processing_integration():
    """Test algorithm integration with batch processing system."""
    from src.backend.core.simulation.batch_executor import create_batch_executor
    
    batch_config = {
        'algorithm_name': 'my_navigation_algorithm',
        'batch_size': 100,
        'parallel_execution': True,
        'max_workers': 4
    }
    
    batch_executor = create_batch_executor(batch_config)
    
    # Execute small batch for integration testing
    plume_datasets = [numpy.random.rand(50, 50, 25) for _ in range(10)]
    batch_result = batch_executor.execute_batch(
        plume_datasets=plume_datasets,
        algorithm_parameters=self.algorithm_params
    )
    
    assert batch_result.success
    assert len(batch_result.individual_results) == 10
    assert all(result.success for result in batch_result.individual_results)
```

## Performance Optimization

### Execution Efficiency

Optimize algorithm execution for <7.2 seconds average performance while maintaining accuracy and scientific validity. Use JIT compilation, vectorized operations, and memory optimization techniques.

**Performance Optimization Techniques:**

```python
import numpy as np
from numba import jit, prange
from src.backend.utils.memory_management import optimize_memory_usage

class OptimizedNavigationAlgorithm(MyNavigationAlgorithm):
    def __init__(self, parameters: AlgorithmParameters, execution_config: Dict[str, Any]):
        super().__init__(parameters, execution_config)
        # Pre-compile JIT functions
        self._precompile_optimizations()
    
    def _precompile_optimizations(self):
        """Pre-compile JIT optimized functions for faster execution."""
        # Trigger JIT compilation with dummy data
        dummy_data = np.random.rand(10, 10)
        self._calculate_gradient_jit(dummy_data)
        self._update_position_jit(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _calculate_gradient_jit(concentration_field: np.ndarray) -> np.ndarray:
        """JIT-optimized gradient calculation for improved performance."""
        gradient = np.zeros((concentration_field.shape[0], concentration_field.shape[1], 2))
        
        for i in prange(1, concentration_field.shape[0] - 1):
            for j in prange(1, concentration_field.shape[1] - 1):
                # Calculate gradient using central differences
                gradient[i, j, 0] = (concentration_field[i+1, j] - concentration_field[i-1, j]) / 2.0
                gradient[i, j, 1] = (concentration_field[i, j+1] - concentration_field[i, j-1]) / 2.0
        
        return gradient
    
    @staticmethod
    @jit(nopython=True)
    def _update_position_jit(current_position: np.ndarray, movement_vector: np.ndarray) -> np.ndarray:
        """JIT-optimized position update for faster navigation steps."""
        return current_position + movement_vector
    
    @optimize_memory_usage
    def _navigate_plume(self, plume_data: np.ndarray, plume_metadata: Dict[str, Any]) -> np.ndarray:
        """Memory-optimized navigation implementation."""
        trajectory = []
        current_position = np.array([plume_data.shape[0] // 2, plume_data.shape[1] // 2], dtype=np.float64)
        
        for time_step in range(plume_data.shape[2]):
            # Use JIT-optimized gradient calculation
            concentration_slice = plume_data[:, :, time_step]
            gradient_field = self._calculate_gradient_jit(concentration_slice)
            
            # Calculate movement based on algorithm-specific logic
            movement_vector = self._calculate_movement(gradient_field, current_position)
            
            # Use JIT-optimized position update
            current_position = self._update_position_jit(current_position, movement_vector)
            
            trajectory.append(current_position.copy())
            
            # Early termination if convergence criteria met
            if self._check_convergence(trajectory):
                break
        
        return np.array(trajectory)
```

### Memory Management

Implement efficient memory management for large plume datasets and batch processing scenarios. Use memory monitoring, caching strategies, and chunked processing for optimal resource utilization.

**Memory Management Implementation:**

```python
from src.backend.utils.memory_management import MemoryManager
from src.backend.cache.memory_cache import MemoryCache

class MemoryEfficientAlgorithm(MyNavigationAlgorithm):
    def __init__(self, parameters: AlgorithmParameters, execution_config: Dict[str, Any]):
        super().__init__(parameters, execution_config)
        self.memory_manager = MemoryManager()
        self.computation_cache = MemoryCache(max_size_mb=256)
    
    def _execute_algorithm(self, plume_data: np.ndarray, plume_metadata: Dict[str, Any], context: AlgorithmContext) -> AlgorithmResult:
        """Memory-efficient algorithm execution with caching and optimization."""
        with self.memory_manager.optimize_for_large_arrays():
            # Check cache for previously computed results
            cache_key = self._generate_cache_key(plume_data, plume_metadata)
            cached_result = self.computation_cache.get(cache_key)
            
            if cached_result is not None:
                self.logger.info("Using cached computation result")
                return cached_result
            
            # Process plume data in chunks to manage memory usage
            chunk_size = self.memory_manager.calculate_optimal_chunk_size(
                data_shape=plume_data.shape,
                available_memory_mb=self.memory_manager.get_available_memory_mb()
            )
            
            trajectory = []
            for chunk_start in range(0, plume_data.shape[2], chunk_size):
                chunk_end = min(chunk_start + chunk_size, plume_data.shape[2])
                plume_chunk = plume_data[:, :, chunk_start:chunk_end]
                
                # Process chunk with memory monitoring
                chunk_trajectory = self._process_plume_chunk(plume_chunk, chunk_start)
                trajectory.extend(chunk_trajectory)
                
                # Force garbage collection if memory usage is high
                if self.memory_manager.get_memory_usage_percent() > 80:
                    self.memory_manager.force_garbage_collection()
            
            # Create result and cache it
            result = AlgorithmResult(self.algorithm_name, context.simulation_id, True)
            result.trajectory = np.array(trajectory)
            
            # Cache result for future use
            self.computation_cache.set(cache_key, result)
            
            return result
    
    def _generate_cache_key(self, plume_data: np.ndarray, plume_metadata: Dict[str, Any]) -> str:
        """Generate cache key for plume data and metadata."""
        import hashlib
        
        # Create hash from plume data shape and key metadata
        data_hash = hashlib.md5()
        data_hash.update(str(plume_data.shape).encode())
        data_hash.update(str(plume_metadata.get('format', '')).encode())
        data_hash.update(str(self.parameters.to_dict()).encode())
        
        return f"algorithm_{self.algorithm_name}_{data_hash.hexdigest()}"
```

## Documentation and Examples

### Algorithm Documentation

Provide comprehensive documentation including algorithm theory, implementation details, parameter descriptions, and usage examples. Documentation ensures reproducibility and facilitates algorithm adoption.

**Algorithm Documentation Template:**

```python
class MyNavigationAlgorithm(BaseAlgorithm):
    """
    My Navigation Algorithm for Plume Source Localization
    
    This algorithm implements [describe algorithm approach] for efficient odor source
    localization in turbulent plume environments. The algorithm uses [key techniques]
    to navigate towards the source while maintaining [performance characteristics].
    
    Algorithm Theory:
        [Detailed description of the theoretical foundation]
    
    Key Features:
        - [Feature 1]: Description and benefits
        - [Feature 2]: Description and benefits
        - [Feature 3]: Description and benefits
    
    Performance Characteristics:
        - Average execution time: [X.X] seconds per simulation
        - Memory usage: [XXX] MB typical
        - Convergence rate: [XX]% on standard test datasets
        - Correlation with reference: [X.XX] (>0.95 required)
    
    Parameters:
        search_radius (float): Radius for local search operations
            - Range: 0.1 to 50.0
            - Default: 5.0
            - Units: grid cells
            
        convergence_tolerance (float): Tolerance for convergence detection
            - Range: 1e-12 to 1e-3
            - Default: 1e-6
            - Units: normalized concentration difference
    
    Example Usage:
        >>> from src.backend.algorithms.my_navigation_algorithm import MyNavigationAlgorithm
        >>> from src.backend.algorithms.base_algorithm import AlgorithmParameters
        >>> 
        >>> # Create algorithm parameters
        >>> params = AlgorithmParameters(
        ...     algorithm_name='my_navigation_algorithm',
        ...     parameters={'search_radius': 5.0, 'convergence_tolerance': 1e-6},
        ...     constraints={'search_radius': {'min': 0.1, 'max': 50.0}}
        ... )
        >>> 
        >>> # Initialize algorithm
        >>> algorithm = MyNavigationAlgorithm(params, {})
        >>> 
        >>> # Execute on plume data
        >>> result = algorithm.execute(plume_data, plume_metadata, 'simulation_001')
        >>> print(f"Navigation success: {result.success}")
        >>> print(f"Trajectory length: {len(result.trajectory)}")
    
    References:
        [1] Author, A. et al. "Algorithm Paper Title", Journal Name, Year
        [2] Author, B. et al. "Related Work Title", Conference Name, Year
    
    See Also:
        - BaseAlgorithm: Abstract base class for all navigation algorithms
        - InfotaxisAlgorithm: Information-theoretic navigation algorithm
        - CastingAlgorithm: Bio-inspired casting search algorithm
    """
```

### Usage Examples

Provide practical examples demonstrating algorithm usage, parameter tuning, and integration with the simulation system. Examples facilitate rapid adoption and proper implementation.

**Complete Algorithm Usage Example:**

```python
#!/usr/bin/env python3
"""
Example: Using My Navigation Algorithm for Plume Source Localization

This example demonstrates how to use the MyNavigationAlgorithm for plume
navigation simulation with proper parameter configuration, execution,
and result analysis.
"""

import numpy as np
from pathlib import Path
from src.backend.algorithms.my_navigation_algorithm import MyNavigationAlgorithm
from src.backend.algorithms.base_algorithm import AlgorithmParameters
from src.backend.algorithms.algorithm_registry import register_algorithm, create_algorithm_instance
from src.backend.core.analysis.visualization import ScientificVisualizer
from src.backend.io.video_reader import VideoReader

def main():
    """Main example function demonstrating algorithm usage."""
    
    # 1. Load plume data
    print("Loading plume data...")
    video_reader = VideoReader()
    plume_data, plume_metadata = video_reader.read_video(
        file_path="data/sample_plume.avi",
        format_type="crimaldi"
    )
    
    # 2. Configure algorithm parameters
    print("Configuring algorithm parameters...")
    algorithm_params = AlgorithmParameters(
        algorithm_name='my_navigation_algorithm',
        parameters={
            'search_radius': 5.0,
            'convergence_tolerance': 1e-6,
            'max_iterations': 1000,
            'adaptive_search': True
        },
        constraints={
            'search_radius': {'min': 0.1, 'max': 50.0},
            'convergence_tolerance': {'min': 1e-12, 'max': 1e-3},
            'max_iterations': {'min': 10, 'max': 10000}
        }
    )
    
    # 3. Validate parameters
    validation_result = algorithm_params.validate(strict_validation=True)
    if not validation_result.is_valid:
        print(f"Parameter validation failed: {validation_result.errors}")
        return
    
    # 4. Create algorithm instance
    print("Creating algorithm instance...")
    algorithm = MyNavigationAlgorithm(
        parameters=algorithm_params,
        execution_config={
            'enable_performance_tracking': True,
            'enable_visualization': True,
            'timeout_seconds': 300
        }
    )
    
    # 5. Execute algorithm
    print("Executing navigation algorithm...")
    result = algorithm.execute(
        plume_data=plume_data,
        plume_metadata=plume_metadata,
        simulation_id='example_simulation_001'
    )
    
    # 6. Analyze results
    print("Analyzing results...")
    if result.success:
        print(f"Navigation completed successfully!")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        print(f"Trajectory length: {len(result.trajectory)} steps")
        print(f"Converged: {result.converged}")
        
        # Get performance summary
        performance_summary = algorithm.get_performance_summary(history_window=1)
        print(f"Performance summary: {performance_summary}")
        
        # 7. Visualize results
        print("Generating visualizations...")
        visualizer = ScientificVisualizer()
        
        # Create trajectory plot
        trajectory_plot = visualizer.create_trajectory_plot(
            trajectory=result.trajectory,
            plume_data=plume_data,
            title="My Navigation Algorithm - Trajectory"
        )
        
        # Save visualization
        output_path = Path("results/my_algorithm_trajectory.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_plot.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory plot saved to: {output_path}")
        
    else:
        print(f"Navigation failed: {result.warnings}")
        print(f"Performance metrics: {result.performance_metrics}")
    
    # 8. Export results
    print("Exporting results...")
    result_dict = result.to_dict(include_trajectory=True, include_state=False)
    
    import json
    with open("results/my_algorithm_results.json", "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    print("Example completed successfully!")

if __name__ == "__main__":
    main()
```

## Best Practices and Guidelines

### Code Quality Standards

Follow Python coding standards, implement comprehensive error handling, and maintain code documentation for scientific reproducibility:

- **Use type hints** for all function parameters and return values
- **Implement comprehensive docstrings** following NumPy documentation style
- **Follow PEP 8 coding standards** with line length limit of 100 characters
- **Use meaningful variable names** that reflect scientific concepts
- **Implement proper error handling** with specific exception types
- **Add logging statements** for debugging and audit trail purposes
- **Use constants** for magic numbers and configuration values
- **Implement parameter validation** with clear error messages

### Scientific Reproducibility

Ensure algorithm implementations support reproducible research outcomes across different computational environments:

- **Set random seeds** for any stochastic components
- **Document all algorithm assumptions** and limitations
- **Provide reference implementations** for validation
- **Include statistical validation** against known benchmarks
- **Implement deterministic behavior** when possible
- **Document computational complexity** and performance characteristics
- **Provide clear parameter sensitivity analysis**
- **Include uncertainty quantification** where appropriate

### Performance Optimization

Optimize algorithm performance while maintaining scientific accuracy and reproducibility requirements:

- **Profile algorithm execution** to identify bottlenecks
- **Use vectorized NumPy operations** instead of Python loops
- **Implement JIT compilation** for computationally intensive functions
- **Cache expensive computations** when appropriate
- **Optimize memory usage** for large datasets
- **Implement early termination conditions** for convergence
- **Use parallel processing** for independent operations
- **Monitor and validate performance** against requirements

## Troubleshooting and Common Issues

### Common Implementation Problems

**Algorithm execution exceeds 7.2 second time limit:**
- Solution: Optimize computational bottlenecks, implement JIT compilation, reduce algorithm complexity, or implement early termination conditions

**Correlation with reference implementation below 95%:**
- Solution: Verify algorithm logic against reference, check numerical precision, validate parameter settings, and ensure proper data normalization

**Memory usage too high for large plume datasets:**
- Solution: Implement chunked processing, optimize data structures, use memory mapping, and implement garbage collection strategies

**Algorithm fails to converge on certain datasets:**
- Solution: Adjust convergence criteria, implement adaptive parameters, add robustness checks, and validate input data quality

**Registry integration fails during algorithm loading:**
- Solution: Check import paths, verify interface compliance, validate metadata format, and ensure proper inheritance from BaseAlgorithm

### Debugging Strategies

Use systematic approaches for debugging algorithm implementations and performance issues:

- **Use logging extensively** to track algorithm execution flow
- **Implement visualization** of intermediate algorithm states
- **Create unit tests** for individual algorithm components
- **Use profiling tools** to identify performance bottlenecks
- **Validate algorithm behavior** on simple test cases
- **Compare results** with reference implementations step-by-step
- **Check numerical stability** and precision issues
- **Verify parameter validation** and constraint enforcement

---

This guide provides comprehensive coverage of all requirements for implementing navigation algorithms that meet the scientific computing standards and performance requirements of the plume simulation system. Follow the guidelines and examples to ensure your algorithm integrates seamlessly with the system while maintaining the required >95% correlation with reference implementations and <7.2 seconds execution time.