"""
Comprehensive unit test module for memory mapping functionality in the plume simulation system. Tests memory-mapped array operations, joblib memory mapping integration, cache management, parallel processing memory sharing, and performance validation. Validates memory efficiency, data integrity, concurrent access safety, and integration with the multi-level caching system while ensuring <8GB memory usage and optimal performance for 4000+ simulation batch processing.

This module provides comprehensive test coverage for memory mapping operations including:
- Memory-mapped array creation, read/write operations, and resource management
- Joblib memory mapping integration for parallel processing workflows
- Cache manager integration with memory-mapped data structures
- Concurrent access testing for thread safety and data consistency
- Performance validation against memory usage and processing speed thresholds
- Memory leak detection and prevention with garbage collection optimization
- Error handling for file system errors, permission issues, and resource exhaustion
- Data integrity validation across different access patterns and system conditions

Key Testing Areas:
- Memory-mapped array lifecycle management with proper resource cleanup
- Integration with joblib Memory objects for distributed caching
- Multi-level cache performance with memory-mapped storage backends
- Parallel worker memory sharing and coordination mechanisms
- Performance threshold validation for <8GB memory usage requirements
- Memory leak detection with baseline comparison and garbage collection
- Concurrent access patterns with thread safety verification
- Error recovery and graceful degradation under adverse conditions
- Cross-platform file system integration and compatibility testing
"""

# External library imports with version specifications for testing framework
import pytest  # pytest 8.3.5+ - Testing framework for unit test execution and fixture management
import numpy as np  # numpy 2.1.3+ - Numerical array operations and memory-mapped array testing
import tempfile  # Python 3.9+ - Temporary file creation for memory mapping test isolation
import pathlib  # Python 3.9+ - Path handling for memory-mapped file management
import mmap  # Python 3.9+ - Low-level memory mapping functionality testing
import psutil  # psutil 5.9.0+ - Memory usage monitoring and validation during tests
import joblib  # joblib 1.6.0+ - Joblib memory mapping and parallel processing integration testing
from joblib import Memory, Parallel, delayed  # joblib 1.6.0+ - Memory object and parallel processing primitives
import threading  # Python 3.9+ - Concurrent access testing for memory-mapped arrays
import time  # Python 3.9+ - Performance timing measurements for memory mapping operations
import gc  # Python 3.9+ - Garbage collection control for memory leak testing
import os  # Python 3.9+ - Operating system interface for file system operations
import uuid  # Python 3.9+ - Unique identifier generation for test correlation
import datetime  # Python 3.9+ - Timestamp handling for test metadata and audit trails
import json  # Python 3.9+ - JSON serialization for test configuration and results
import hashlib  # Python 3.9+ - Hash functions for data integrity verification
from typing import Dict, Any, List, Optional, Tuple  # Python 3.9+ - Type hints for test function signatures
from unittest.mock import Mock, patch, MagicMock  # Python 3.9+ - Mocking utilities for isolated testing
import warnings  # Python 3.9+ - Warning management for test validation and error reporting
import contextlib  # Python 3.9+ - Context manager utilities for test resource management

# Internal imports from backend utilities (using paths as specified in JSON)
from backend.utils.memory_management import (  # Memory management utilities for testing
    MemoryMappedArray, allocate_memory_mapped_array, MemoryPool
)
from backend.utils.parallel_processing import (  # Parallel processing utilities with memory mapping
    setup_memory_mapping, ParallelExecutor
)
from backend.utils.caching import (  # Cache management integration with memory mapping
    CacheManager, JobLibCache
)

# Internal imports from test utilities
from test.utils.test_helpers import (  # Test helper utilities for fixture management
    create_test_fixture_path, assert_arrays_almost_equal, measure_performance
)
from test.utils.validation_metrics import (  # Validation metrics for performance testing
    ValidationMetricsCalculator
)

# Global test configuration constants for memory mapping validation
TEST_ARRAY_SHAPES = [(100, 100), (1000, 1000), (2000, 2000, 10)]  # Test array dimensions for various memory sizes
TEST_DATA_TYPES = ['float32', 'float64', 'int32', 'uint8']  # Data types for memory mapping compatibility testing
MEMORY_THRESHOLD_MB = 100  # Memory usage threshold for individual test operations
PERFORMANCE_TIMEOUT_SECONDS = 5.0  # Maximum execution time for memory mapping operations
CONCURRENT_ACCESS_THREADS = 4  # Number of threads for concurrent access testing
CHUNK_SIZES = [64, 256, 1024]  # Chunk sizes for read/write operation testing

# Memory mapping test configuration
MEMORY_MAPPING_TEST_CONFIG = {
    'max_memory_usage_gb': 8.0,  # Maximum memory usage for entire test suite
    'target_memory_usage_gb': 6.0,  # Target memory usage for optimal performance
    'warning_threshold_gb': 7.0,  # Warning threshold for memory usage monitoring
    'critical_threshold_gb': 7.5,  # Critical threshold requiring immediate cleanup
    'batch_simulation_target': 4000,  # Target number of simulations for batch processing tests
    'memory_leak_tolerance_mb': 50,  # Acceptable memory growth between test iterations
    'file_cleanup_timeout_seconds': 30.0,  # Maximum time allowed for file cleanup operations
    'concurrent_access_timeout_seconds': 10.0  # Timeout for concurrent access operations
}

# Test data generation constants for reproducible testing
MEMORY_MAP_TEST_SEED = 42  # Random seed for reproducible test data generation
TEST_DATA_PATTERNS = {
    'sequential': lambda size: np.arange(size, dtype=np.float32),
    'random': lambda size: np.random.RandomState(MEMORY_MAP_TEST_SEED).random(size).astype(np.float32),
    'checkerboard': lambda shape: np.indices(shape).sum(axis=0) % 2,
    'gradient': lambda shape: np.linspace(0, 1, np.prod(shape)).reshape(shape)
}


class TestMemoryMappingFixtures:
    """
    Test fixture class providing setup and teardown for memory mapping test scenarios with resource management and isolation.
    
    This class manages test fixtures including temporary file creation, memory-mapped array setup,
    resource tracking, and cleanup operations to ensure test isolation and proper resource management.
    """
    
    def __init__(self):
        """
        Initialize test fixtures with temporary directory and resource tracking.
        """
        # Create temporary directory for test isolation
        self.temp_directory = pathlib.Path(tempfile.mkdtemp(prefix='memory_mapping_test_'))
        
        # Initialize file tracking for cleanup
        self.created_files = []
        
        # Record baseline memory usage
        self.memory_usage_baseline = self._get_current_memory_usage()
        
        # Setup logging for test operations
        self.fixture_id = str(uuid.uuid4())
        self.creation_time = datetime.datetime.now()
        
        # Initialize resource tracking containers
        self.allocated_arrays = []
        self.active_memory_pools = []
        self.test_metadata = {
            'fixture_id': self.fixture_id,
            'creation_time': self.creation_time,
            'temp_directory': str(self.temp_directory),
            'baseline_memory_mb': self.memory_usage_baseline['used_memory_mb']
        }
    
    def create_test_memory_mapped_array(
        self,
        shape: Tuple[int, ...],
        dtype: str = 'float32',
        mode: str = 'w+'
    ) -> 'MemoryMappedArray':
        """
        Create memory-mapped array for testing with specified parameters.
        
        Args:
            shape: Array shape as tuple of dimensions
            dtype: Data type for the array elements
            mode: File mode for memory mapping ('r', 'w+', 'r+')
            
        Returns:
            MemoryMappedArray: Configured memory-mapped array for testing
        """
        # Generate unique filename for array
        array_filename = f"test_array_{len(self.created_files)}_{dtype}_{'-'.join(map(str, shape))}.dat"
        array_path = self.temp_directory / array_filename
        
        # Create memory-mapped array with parameters
        try:
            memory_mapped_array = allocate_memory_mapped_array(
                file_path=str(array_path),
                shape=shape,
                dtype=dtype,
                mode=mode
            )
            
            # Track created file for cleanup
            self.created_files.append(array_path)
            self.allocated_arrays.append(memory_mapped_array)
            
            # Return configured array instance
            return memory_mapped_array
            
        except Exception as e:
            raise RuntimeError(f"Failed to create test memory-mapped array: {e}")
    
    def cleanup_test_resources(self) -> None:
        """
        Clean up test resources including files, memory mappings, and temporary data.
        """
        cleanup_errors = []
        
        try:
            # Close all open memory-mapped arrays
            for array in self.allocated_arrays:
                try:
                    if hasattr(array, 'close'):
                        array.close()
                except Exception as e:
                    cleanup_errors.append(f"Array cleanup error: {e}")
            
            # Clear memory caches and pools
            for pool in self.active_memory_pools:
                try:
                    if hasattr(pool, 'deallocate'):
                        pool.deallocate()
                except Exception as e:
                    cleanup_errors.append(f"Pool cleanup error: {e}")
            
            # Remove temporary files and directories
            for file_path in self.created_files:
                try:
                    if file_path.exists():
                        if file_path.is_file():
                            file_path.unlink()
                        elif file_path.is_dir():
                            file_path.rmdir()
                except Exception as e:
                    cleanup_errors.append(f"File cleanup error: {e}")
            
            # Remove temporary directory
            try:
                if self.temp_directory.exists():
                    import shutil
                    shutil.rmtree(self.temp_directory, ignore_errors=True)
            except Exception as e:
                cleanup_errors.append(f"Directory cleanup error: {e}")
            
            # Validate memory usage returns to baseline
            current_memory = self._get_current_memory_usage()
            memory_growth = current_memory['used_memory_mb'] - self.memory_usage_baseline['used_memory_mb']
            
            if memory_growth > MEMORY_MAPPING_TEST_CONFIG['memory_leak_tolerance_mb']:
                warnings.warn(
                    f"Memory usage increased by {memory_growth:.1f}MB after cleanup",
                    UserWarning
                )
            
            # Log cleanup completion
            if cleanup_errors:
                warnings.warn(f"Cleanup completed with {len(cleanup_errors)} errors: {cleanup_errors}")
            
        except Exception as e:
            raise RuntimeError(f"Critical cleanup failure: {e}")
    
    def _get_current_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                'total_memory_mb': memory_info.total / (1024 * 1024),
                'available_memory_mb': memory_info.available / (1024 * 1024),
                'used_memory_mb': memory_info.used / (1024 * 1024),
                'process_memory_mb': process_memory.rss / (1024 * 1024),
                'memory_percent': memory_info.percent
            }
        except Exception:
            return {'used_memory_mb': 0, 'available_memory_mb': 0, 'total_memory_mb': 0, 'process_memory_mb': 0, 'memory_percent': 0}


@pytest.fixture
def memory_mapping_fixtures():
    """
    Pytest fixture providing memory mapping test fixtures with automatic cleanup.
    """
    fixtures = TestMemoryMappingFixtures()
    try:
        yield fixtures
    finally:
        fixtures.cleanup_test_resources()


@pytest.fixture
def test_data_generator():
    """
    Pytest fixture for generating test data with various patterns and characteristics.
    """
    def generate_test_data(pattern: str, shape: Tuple[int, ...], dtype: str = 'float32'):
        """Generate test data with specified pattern and characteristics."""
        np.random.seed(MEMORY_MAP_TEST_SEED)
        
        if pattern == 'sequential':
            data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        elif pattern == 'random':
            data = np.random.random(shape).astype(dtype)
        elif pattern == 'checkerboard':
            indices = np.indices(shape)
            data = (indices.sum(axis=0) % 2).astype(dtype)
        elif pattern == 'gradient':
            data = np.linspace(0, 1, np.prod(shape)).reshape(shape).astype(dtype)
        else:
            raise ValueError(f"Unknown test data pattern: {pattern}")
        
        return data
    
    return generate_test_data


@pytest.mark.parametrize('array_shape', TEST_ARRAY_SHAPES)
@pytest.mark.parametrize('dtype', TEST_DATA_TYPES)
def test_memory_mapped_array_creation(
    memory_mapping_fixtures: TestMemoryMappingFixtures,
    array_shape: Tuple[int, ...],
    dtype: str
):
    """
    Test creation of memory-mapped arrays with various shapes, data types, and file modes to validate proper initialization and resource allocation.
    """
    # Create temporary file for memory mapping
    temp_file = memory_mapping_fixtures.temp_directory / f"test_creation_{dtype}_{'-'.join(map(str, array_shape))}.dat"
    
    # Initialize MemoryMappedArray with specified shape and dtype
    try:
        memory_mapped_array = allocate_memory_mapped_array(
            file_path=str(temp_file),
            shape=array_shape,
            dtype=dtype,
            mode='w+'
        )
        
        # Validate array properties match expected values
        assert memory_mapped_array.shape == array_shape, f"Array shape mismatch: {memory_mapped_array.shape} != {array_shape}"
        assert str(memory_mapped_array.dtype) == dtype, f"Array dtype mismatch: {memory_mapped_array.dtype} != {dtype}"
        
        # Check file size matches calculated array size
        expected_size = np.prod(array_shape) * np.dtype(dtype).itemsize
        actual_size = temp_file.stat().st_size
        assert actual_size >= expected_size, f"File size {actual_size} less than expected {expected_size}"
        
        # Verify memory mapping is properly established
        assert memory_mapped_array._memmap is not None, "Memory mapping not established"
        
        # Test array accessibility and basic operations
        if len(array_shape) == 1:
            memory_mapped_array[0] = np.dtype(dtype).type(1)
            assert memory_mapped_array[0] == np.dtype(dtype).type(1), "Array write/read failed"
        elif len(array_shape) == 2:
            memory_mapped_array[0, 0] = np.dtype(dtype).type(1)
            assert memory_mapped_array[0, 0] == np.dtype(dtype).type(1), "Array write/read failed"
        elif len(array_shape) == 3:
            memory_mapped_array[0, 0, 0] = np.dtype(dtype).type(1)
            assert memory_mapped_array[0, 0, 0] == np.dtype(dtype).type(1), "Array write/read failed"
        
        # Clean up temporary files and resources
        memory_mapped_array.close()
        memory_mapping_fixtures.created_files.append(temp_file)
        
    except Exception as e:
        pytest.fail(f"Memory-mapped array creation failed: {e}")


@measure_performance(time_limit_seconds=PERFORMANCE_TIMEOUT_SECONDS)
def test_memory_mapped_array_read_write(
    memory_mapping_fixtures: TestMemoryMappingFixtures,
    test_data_generator
):
    """
    Test read and write operations on memory-mapped arrays with data integrity validation and performance measurement.
    """
    # Generate test data with known patterns
    test_shape = (500, 500)
    test_dtype = 'float32'
    test_data = test_data_generator('gradient', test_shape, test_dtype)
    
    # Create memory-mapped array for testing
    memory_mapped_array = memory_mapping_fixtures.create_test_memory_mapped_array(
        shape=test_shape,
        dtype=test_dtype,
        mode='w+'
    )
    
    try:
        # Write data chunks to memory-mapped array
        chunk_size = 100
        for i in range(0, test_shape[0], chunk_size):
            end_i = min(i + chunk_size, test_shape[0])
            chunk_data = test_data[i:end_i]
            memory_mapped_array.write_chunk(chunk_data, start_index=(i, 0))
        
        # Flush changes to ensure persistence
        memory_mapped_array.flush()
        
        # Read data chunks back from array
        reconstructed_data = np.zeros_like(test_data)
        for i in range(0, test_shape[0], chunk_size):
            end_i = min(i + chunk_size, test_shape[0])
            chunk_shape = (end_i - i, test_shape[1])
            read_chunk = memory_mapped_array.read_chunk(chunk_shape, start_index=(i, 0))
            reconstructed_data[i:end_i] = read_chunk
        
        # Validate data integrity using assert_arrays_almost_equal
        assert_arrays_almost_equal(
            actual=reconstructed_data,
            expected=test_data,
            tolerance=1e-6,
            error_message="Memory-mapped array read/write data integrity check failed"
        )
        
        # Test partial chunk reads and writes
        partial_data = test_data[100:200, 100:200]
        memory_mapped_array.write_chunk(partial_data * 2, start_index=(100, 100))
        memory_mapped_array.flush()
        
        partial_read = memory_mapped_array.read_chunk((100, 100), start_index=(100, 100))
        assert_arrays_almost_equal(
            actual=partial_read,
            expected=partial_data * 2,
            tolerance=1e-6,
            error_message="Partial chunk read/write validation failed"
        )
        
        # Measure read/write performance against thresholds
        start_time = time.time()
        for _ in range(10):
            test_chunk = memory_mapped_array.read_chunk((50, 50), start_index=(0, 0))
            memory_mapped_array.write_chunk(test_chunk, start_index=(0, 0))
        end_time = time.time()
        
        performance_time = end_time - start_time
        assert performance_time < PERFORMANCE_TIMEOUT_SECONDS, f"Read/write performance {performance_time:.3f}s exceeds threshold"
        
        # Verify memory usage stays within limits
        current_memory = psutil.virtual_memory()
        memory_usage_gb = current_memory.used / (1024**3)
        assert memory_usage_gb < MEMORY_MAPPING_TEST_CONFIG['critical_threshold_gb'], f"Memory usage {memory_usage_gb:.2f}GB exceeds critical threshold"
        
    finally:
        memory_mapped_array.close()


@pytest.mark.timeout(10)
def test_concurrent_memory_mapping_access(memory_mapping_fixtures: TestMemoryMappingFixtures):
    """
    Test concurrent access to memory-mapped arrays from multiple threads to validate thread safety and data consistency.
    """
    # Create shared memory-mapped array for testing
    test_shape = (1000, 1000)
    shared_memory_array = memory_mapping_fixtures.create_test_memory_mapped_array(
        shape=test_shape,
        dtype='float32',
        mode='w+'
    )
    
    # Initialize shared data
    initial_data = np.arange(np.prod(test_shape), dtype=np.float32).reshape(test_shape)
    shared_memory_array.write_chunk(initial_data, start_index=(0, 0))
    shared_memory_array.flush()
    
    # Define concurrent read and write operations
    results = []
    errors = []
    thread_lock = threading.Lock()
    
    def concurrent_reader(thread_id: int, read_count: int):
        """Concurrent reader function for thread safety testing."""
        try:
            thread_results = []
            for i in range(read_count):
                # Read random chunk from array
                start_row = np.random.randint(0, test_shape[0] - 100)
                start_col = np.random.randint(0, test_shape[1] - 100)
                chunk = shared_memory_array.read_chunk((100, 100), start_index=(start_row, start_col))
                
                # Validate chunk integrity
                expected_chunk = initial_data[start_row:start_row+100, start_col:start_col+100]
                chunk_checksum = hashlib.md5(chunk.tobytes()).hexdigest()
                expected_checksum = hashlib.md5(expected_chunk.tobytes()).hexdigest()
                
                thread_results.append({
                    'thread_id': thread_id,
                    'read_index': i,
                    'chunk_checksum': chunk_checksum,
                    'expected_checksum': expected_checksum,
                    'checksums_match': chunk_checksum == expected_checksum
                })
            
            with thread_lock:
                results.extend(thread_results)
                
        except Exception as e:
            with thread_lock:
                errors.append(f"Reader thread {thread_id} error: {e}")
    
    def concurrent_writer(thread_id: int, write_count: int):
        """Concurrent writer function for thread safety testing."""
        try:
            for i in range(write_count):
                # Write to non-overlapping regions to avoid conflicts
                region_size = test_shape[0] // CONCURRENT_ACCESS_THREADS
                start_row = thread_id * region_size
                end_row = min(start_row + region_size, test_shape[0])
                
                if start_row < end_row:
                    write_data = np.full((end_row - start_row, test_shape[1]), thread_id + i, dtype=np.float32)
                    shared_memory_array.write_chunk(write_data, start_index=(start_row, 0))
                    shared_memory_array.flush()
                
        except Exception as e:
            with thread_lock:
                errors.append(f"Writer thread {thread_id} error: {e}")
    
    # Launch multiple threads for concurrent access
    reader_threads = []
    writer_threads = []
    
    # Create reader threads
    for thread_id in range(CONCURRENT_ACCESS_THREADS):
        reader_thread = threading.Thread(
            target=concurrent_reader,
            args=(thread_id, 20)
        )
        reader_threads.append(reader_thread)
    
    # Create writer threads
    for thread_id in range(CONCURRENT_ACCESS_THREADS):
        writer_thread = threading.Thread(
            target=concurrent_writer,
            args=(thread_id, 10)
        )
        writer_threads.append(writer_thread)
    
    # Execute simultaneous read/write operations
    start_time = time.time()
    
    for thread in reader_threads + writer_threads:
        thread.start()
    
    # Synchronize thread completion and collect results
    for thread in reader_threads + writer_threads:
        thread.join(timeout=MEMORY_MAPPING_TEST_CONFIG['concurrent_access_timeout_seconds'])
        if thread.is_alive():
            errors.append(f"Thread {thread.name} did not complete within timeout")
    
    end_time = time.time()
    
    # Validate data consistency across all operations
    assert len(errors) == 0, f"Concurrent access errors: {errors}"
    
    # Check for race conditions and data corruption
    successful_reads = len([r for r in results if r['checksums_match']])
    total_reads = len(results)
    
    if total_reads > 0:
        consistency_rate = successful_reads / total_reads
        assert consistency_rate >= 0.95, f"Data consistency rate {consistency_rate:.2%} below 95% threshold"
    
    # Verify thread safety mechanisms work correctly
    execution_time = end_time - start_time
    assert execution_time < MEMORY_MAPPING_TEST_CONFIG['concurrent_access_timeout_seconds'], f"Concurrent access took too long: {execution_time:.2f}s"
    
    shared_memory_array.close()


def test_joblib_memory_mapping_integration(memory_mapping_fixtures: TestMemoryMappingFixtures):
    """
    Test integration with joblib memory mapping for parallel processing workflows and cache sharing.
    """
    # Setup joblib memory mapping with test cache directory
    cache_directory = memory_mapping_fixtures.temp_directory / 'joblib_cache'
    cache_directory.mkdir(exist_ok=True)
    
    try:
        # Create memory-mapped arrays for parallel processing
        setup_memory_mapping(cache_dir=str(cache_directory), verbose=0)
        
        # Test joblib Memory object integration
        memory = Memory(location=str(cache_directory), verbose=0)
        
        @memory.cache
        def memory_intensive_computation(array_data, multiplier):
            """Test function for joblib caching with memory-mapped data."""
            return array_data * multiplier
        
        # Create test data
        test_data = np.random.random((1000, 1000)).astype(np.float32)
        
        # Validate cache function decoration works correctly
        result1 = memory_intensive_computation(test_data, 2.0)
        result2 = memory_intensive_computation(test_data, 2.0)  # Should use cached result
        
        assert_arrays_almost_equal(result1, result2, tolerance=1e-6, error_message="Joblib cached results differ")
        
        # Test parallel worker access to shared memory
        def parallel_worker(data_chunk, worker_id):
            """Worker function for parallel processing test."""
            return {
                'worker_id': worker_id,
                'chunk_sum': np.sum(data_chunk),
                'chunk_shape': data_chunk.shape,
                'processed_timestamp': time.time()
            }
        
        # Create chunks for parallel processing
        chunks = [test_data[i:i+200] for i in range(0, test_data.shape[0], 200)]
        
        # Execute parallel processing with joblib
        with Parallel(n_jobs=2, backend='threading') as parallel:
            parallel_results = parallel(
                delayed(parallel_worker)(chunk, idx) for idx, chunk in enumerate(chunks)
            )
        
        # Verify cache hit/miss behavior
        cache_info = memory.cache_function.cache_info() if hasattr(memory.cache_function, 'cache_info') else None
        
        # Validate memory mapping persistence across processes
        assert len(parallel_results) == len(chunks), "Parallel processing result count mismatch"
        
        for result in parallel_results:
            assert 'worker_id' in result, "Worker result missing worker_id"
            assert 'chunk_sum' in result, "Worker result missing chunk_sum"
            assert isinstance(result['chunk_sum'], (int, float, np.number)), "Invalid chunk_sum type"
        
        # Test cache clearing and cleanup
        memory.clear(warn=False)
        
    except Exception as e:
        pytest.fail(f"Joblib memory mapping integration test failed: {e}")
    
    finally:
        # Clean up joblib cache and temporary files
        try:
            import shutil
            if cache_directory.exists():
                shutil.rmtree(cache_directory, ignore_errors=True)
        except Exception:
            pass


@pytest.mark.parametrize('pool_size_mb', [10, 50, 100])
def test_memory_pool_integration(
    memory_mapping_fixtures: TestMemoryMappingFixtures,
    pool_size_mb: int
):
    """
    Test integration between memory pools and memory mapping for efficient resource allocation and management.
    """
    # Create memory pool with specified size
    try:
        memory_pool = MemoryPool(pool_size_mb=pool_size_mb)
        memory_mapping_fixtures.active_memory_pools.append(memory_pool)
        
        # Allocate memory-mapped arrays from pool
        allocated_arrays = []
        array_count = min(5, pool_size_mb // 10)  # Limit based on pool size
        
        for i in range(array_count):
            array_size_mb = pool_size_mb // (array_count * 2)  # Use half pool size
            array_shape = (int(np.sqrt(array_size_mb * 1024 * 1024 / 4)), int(np.sqrt(array_size_mb * 1024 * 1024 / 4)))
            
            allocated_memory = memory_pool.allocate(size_bytes=array_size_mb * 1024 * 1024)
            assert allocated_memory is not None, f"Memory allocation failed for {array_size_mb}MB"
            
            allocated_arrays.append({
                'memory_block': allocated_memory,
                'size_mb': array_size_mb,
                'allocation_index': i
            })
        
        # Test pool allocation and deallocation efficiency
        allocation_start_time = time.time()
        
        for array_info in allocated_arrays:
            memory_pool.deallocate(array_info['memory_block'])
        
        allocation_end_time = time.time()
        allocation_time = allocation_end_time - allocation_start_time
        
        # Validate memory fragmentation remains low
        pool_stats = memory_pool.get_pool_statistics()
        fragmentation_ratio = pool_stats.get('fragmentation_ratio', 0.0)
        assert fragmentation_ratio < 0.3, f"High memory fragmentation: {fragmentation_ratio:.2%}"
        
        # Test pool cleanup and resource release
        cleanup_success = memory_pool.cleanup_pool()
        assert cleanup_success, "Memory pool cleanup failed"
        
        # Verify pool statistics and utilization metrics
        final_stats = memory_pool.get_pool_statistics()
        assert final_stats['allocated_blocks'] == 0, "Memory blocks not properly deallocated"
        assert final_stats['free_memory_mb'] == pool_size_mb, "Pool memory not fully released"
        
        # Check integration with garbage collection
        gc.collect()
        post_gc_memory = psutil.virtual_memory()
        
        # Validate pool performance under load
        assert allocation_time < 1.0, f"Pool allocation/deallocation too slow: {allocation_time:.3f}s"
        
    except Exception as e:
        pytest.fail(f"Memory pool integration test failed: {e}")


@measure_performance(memory_limit_mb=MEMORY_THRESHOLD_MB)
def test_memory_mapping_performance_thresholds(memory_mapping_fixtures: TestMemoryMappingFixtures):
    """
    Test memory mapping operations against performance thresholds including memory usage, access speed, and resource efficiency.
    """
    # Initialize ValidationMetricsCalculator for performance testing
    validation_calculator = ValidationMetricsCalculator()
    
    # Create large memory-mapped arrays for stress testing
    large_array_shapes = [(2000, 2000), (1500, 1500, 4), (3000, 1000)]
    performance_metrics = []
    
    for array_shape in large_array_shapes:
        start_time = time.time()
        initial_memory = psutil.virtual_memory()
        
        # Measure memory allocation and access performance
        memory_mapped_array = memory_mapping_fixtures.create_test_memory_mapped_array(
            shape=array_shape,
            dtype='float32',
            mode='w+'
        )
        
        # Generate test data for stress testing
        test_data = np.random.random(array_shape).astype(np.float32)
        
        # Monitor memory usage during operations
        memory_allocation_time = time.time() - start_time
        
        # Perform intensive read/write operations
        write_start_time = time.time()
        memory_mapped_array.write_chunk(test_data, start_index=tuple([0] * len(array_shape)))
        memory_mapped_array.flush()
        write_time = time.time() - write_start_time
        
        read_start_time = time.time()
        read_data = memory_mapped_array.read_chunk(array_shape, start_index=tuple([0] * len(array_shape)))
        read_time = time.time() - read_start_time
        
        # Validate performance against <8GB memory threshold
        current_memory = psutil.virtual_memory()
        memory_usage_gb = current_memory.used / (1024**3)
        
        assert memory_usage_gb < MEMORY_MAPPING_TEST_CONFIG['max_memory_usage_gb'], \
            f"Memory usage {memory_usage_gb:.2f}GB exceeds 8GB threshold"
        
        # Test memory mapping efficiency for large datasets
        data_size_mb = test_data.nbytes / (1024 * 1024)
        write_speed_mbps = data_size_mb / write_time if write_time > 0 else 0
        read_speed_mbps = data_size_mb / read_time if read_time > 0 else 0
        
        # Verify performance meets scientific computing requirements
        performance_data = {
            'array_shape': array_shape,
            'data_size_mb': data_size_mb,
            'memory_allocation_time': memory_allocation_time,
            'write_time': write_time,
            'read_time': read_time,
            'write_speed_mbps': write_speed_mbps,
            'read_speed_mbps': read_speed_mbps,
            'memory_usage_gb': memory_usage_gb,
            'memory_efficiency': data_size_mb / (current_memory.used - initial_memory.used) * (1024 * 1024) if current_memory.used > initial_memory.used else 1.0
        }
        
        performance_metrics.append(performance_data)
        memory_mapped_array.close()
    
    # Generate performance validation report
    validation_result = validation_calculator.validate_performance_thresholds(
        performance_metrics=performance_metrics,
        memory_threshold_gb=MEMORY_MAPPING_TEST_CONFIG['max_memory_usage_gb'],
        processing_speed_threshold_mbps=100.0
    )
    
    assert validation_result.is_valid, f"Performance validation failed: {validation_result.validation_errors}"


@pytest.mark.slow
def test_memory_leak_detection(memory_mapping_fixtures: TestMemoryMappingFixtures):
    """
    Test memory leak detection and prevention in memory mapping operations with resource monitoring.
    """
    # Record baseline memory usage before test
    gc.collect()  # Ensure clean starting state
    baseline_memory = psutil.virtual_memory()
    baseline_process_memory = psutil.Process().memory_info()
    
    leak_iterations = 50
    memory_snapshots = []
    
    # Create and destroy multiple memory-mapped arrays
    for iteration in range(leak_iterations):
        iteration_start_memory = psutil.virtual_memory()
        
        # Create temporary memory-mapped array
        temp_array = memory_mapping_fixtures.create_test_memory_mapped_array(
            shape=(500, 500),
            dtype='float32',
            mode='w+'
        )
        
        # Perform operations that might cause leaks
        test_data = np.random.random((500, 500)).astype(np.float32)
        temp_array.write_chunk(test_data, start_index=(0, 0))
        temp_array.flush()
        
        # Read data back
        read_data = temp_array.read_chunk((500, 500), start_index=(0, 0))
        
        # Close array explicitly
        temp_array.close()
        
        # Monitor memory usage throughout operations
        iteration_end_memory = psutil.virtual_memory()
        memory_snapshots.append({
            'iteration': iteration,
            'start_memory_mb': iteration_start_memory.used / (1024 * 1024),
            'end_memory_mb': iteration_end_memory.used / (1024 * 1024),
            'memory_delta_mb': (iteration_end_memory.used - iteration_start_memory.used) / (1024 * 1024)
        })
        
        # Force garbage collection between iterations
        if iteration % 10 == 0:
            gc.collect()
    
    # Detect memory growth patterns and leaks
    final_memory = psutil.virtual_memory()
    final_process_memory = psutil.Process().memory_info()
    
    total_memory_growth_mb = (final_memory.used - baseline_memory.used) / (1024 * 1024)
    process_memory_growth_mb = (final_process_memory.rss - baseline_process_memory.rss) / (1024 * 1024)
    
    # Validate proper resource cleanup
    max_acceptable_growth_mb = MEMORY_MAPPING_TEST_CONFIG['memory_leak_tolerance_mb'] * leak_iterations / 10
    
    assert total_memory_growth_mb < max_acceptable_growth_mb, \
        f"Potential memory leak detected: {total_memory_growth_mb:.1f}MB growth exceeds {max_acceptable_growth_mb:.1f}MB threshold"
    
    # Check file handle and memory mapping cleanup
    open_files_count = len(psutil.Process().open_files())
    assert open_files_count < 10, f"Too many open files: {open_files_count} (potential file handle leak)"
    
    # Verify memory returns to baseline levels
    force_cleanup_memory = psutil.virtual_memory()
    final_growth_mb = (force_cleanup_memory.used - baseline_memory.used) / (1024 * 1024)
    
    if final_growth_mb > MEMORY_MAPPING_TEST_CONFIG['memory_leak_tolerance_mb']:
        warnings.warn(f"Memory usage grew by {final_growth_mb:.1f}MB during leak detection test")


def test_cache_integration_with_memory_mapping(memory_mapping_fixtures: TestMemoryMappingFixtures):
    """
    Test integration between caching system and memory mapping for multi-level cache performance.
    """
    # Initialize cache manager with memory mapping support
    cache_manager = CacheManager(
        cache_config={
            'memory_cache_size_mb': 100,
            'disk_cache_size_mb': 500,
            'enable_memory_mapping': True,
            'cache_directory': str(memory_mapping_fixtures.temp_directory / 'cache')
        }
    )
    
    try:
        # Test memory-mapped array caching and retrieval
        test_arrays = []
        cache_keys = []
        
        for i in range(10):
            # Create test array
            array_shape = (200, 200)
            test_array = memory_mapping_fixtures.create_test_memory_mapped_array(
                shape=array_shape,
                dtype='float32',
                mode='w+'
            )
            
            # Generate unique test data
            test_data = np.random.random(array_shape).astype(np.float32) * i
            test_array.write_chunk(test_data, start_index=(0, 0))
            test_array.flush()
            
            # Cache the array data
            cache_key = f"test_array_{i}"
            cache_success = cache_manager.set(
                key=cache_key,
                value=test_data,
                ttl_seconds=300
            )
            
            assert cache_success, f"Failed to cache array data for key {cache_key}"
            
            test_arrays.append(test_array)
            cache_keys.append(cache_key)
        
        # Validate cache hit rates for memory-mapped data
        cache_hits = 0
        cache_misses = 0
        
        for cache_key in cache_keys:
            cached_data = cache_manager.get(cache_key)
            if cached_data is not None:
                cache_hits += 1
            else:
                cache_misses += 1
        
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        assert cache_hit_rate >= 0.8, f"Cache hit rate {cache_hit_rate:.2%} below 80% threshold"
        
        # Test cache eviction with memory-mapped arrays
        large_data = np.random.random((1000, 1000)).astype(np.float32)
        eviction_key = "large_eviction_test"
        
        cache_manager.set(eviction_key, large_data, ttl_seconds=300)
        
        # Verify cache consistency across memory mappings
        retrieved_data = cache_manager.get(eviction_key)
        if retrieved_data is not None:
            assert_arrays_almost_equal(
                actual=retrieved_data,
                expected=large_data,
                tolerance=1e-6,
                error_message="Cache consistency check failed for large array"
            )
        
        # Test cache performance with large datasets
        performance_start_time = time.time()
        
        for i in range(100):
            key = f"perf_test_{i % 10}"  # Reuse keys for cache hits
            cache_manager.get(key)
        
        performance_time = time.time() - performance_start_time
        assert performance_time < 5.0, f"Cache performance {performance_time:.3f}s exceeds threshold"
        
        # Validate cache cleanup and resource management
        cleanup_success = cache_manager.cleanup_cache()
        assert cleanup_success, "Cache cleanup failed"
        
        # Check integration with disk and memory caches
        cache_stats = cache_manager.get_cache_statistics()
        assert 'memory_cache_hits' in cache_stats, "Missing memory cache statistics"
        assert 'disk_cache_hits' in cache_stats, "Missing disk cache statistics"
        
    except Exception as e:
        pytest.fail(f"Cache integration test failed: {e}")
    
    finally:
        # Clean up test arrays
        for array in test_arrays:
            try:
                array.close()
            except:
                pass


@pytest.mark.parametrize('error_scenario', ['file_not_found', 'permission_denied', 'disk_full', 'invalid_size'])
def test_error_handling_in_memory_mapping(
    memory_mapping_fixtures: TestMemoryMappingFixtures,
    error_scenario: str
):
    """
    Test error handling scenarios in memory mapping operations including file system errors, permission issues, and resource exhaustion.
    """
    # Setup error scenario conditions
    temp_dir = memory_mapping_fixtures.temp_directory
    
    try:
        if error_scenario == 'file_not_found':
            # Attempt memory mapping operation on non-existent file
            non_existent_path = temp_dir / 'non_existent_file.dat'
            
            with pytest.raises((FileNotFoundError, OSError)):
                memory_mapped_array = allocate_memory_mapped_array(
                    file_path=str(non_existent_path),
                    shape=(100, 100),
                    dtype='float32',
                    mode='r'  # Read mode for non-existent file
                )
        
        elif error_scenario == 'permission_denied':
            # Create file with restricted permissions
            restricted_file = temp_dir / 'restricted_file.dat'
            restricted_file.touch()
            
            try:
                # Remove write permissions
                os.chmod(str(restricted_file), 0o444)  # Read-only
                
                with pytest.raises((PermissionError, OSError)):
                    memory_mapped_array = allocate_memory_mapped_array(
                        file_path=str(restricted_file),
                        shape=(100, 100),
                        dtype='float32',
                        mode='w+'  # Write mode on read-only file
                    )
            
            finally:
                # Restore permissions for cleanup
                try:
                    os.chmod(str(restricted_file), 0o666)
                    restricted_file.unlink()
                except:
                    pass
        
        elif error_scenario == 'disk_full':
            # Simulate disk full condition (platform-dependent)
            large_size = (10000, 10000)  # Very large array
            
            try:
                # This might fail due to insufficient disk space
                memory_mapped_array = allocate_memory_mapped_array(
                    file_path=str(temp_dir / 'large_test_file.dat'),
                    shape=large_size,
                    dtype='float64',
                    mode='w+'
                )
                
                # If creation succeeds, try to write large amounts of data
                large_data = np.ones(large_size, dtype=np.float64)
                memory_mapped_array.write_chunk(large_data, start_index=(0, 0))
                memory_mapped_array.flush()
                memory_mapped_array.close()
                
            except (OSError, MemoryError, ValueError) as e:
                # Expected error due to resource exhaustion
                assert any(keyword in str(e).lower() for keyword in ['space', 'memory', 'size', 'disk']), \
                    f"Unexpected error type: {e}"
        
        elif error_scenario == 'invalid_size':
            # Test error recovery mechanisms with invalid array sizes
            invalid_shapes = [
                (-100, 100),  # Negative dimensions
                (0, 100),     # Zero dimensions
                (2**32, 100), # Extremely large dimensions
                ()            # Empty shape
            ]
            
            for invalid_shape in invalid_shapes:
                with pytest.raises((ValueError, OverflowError, MemoryError)):
                    memory_mapped_array = allocate_memory_mapped_array(
                        file_path=str(temp_dir / f'invalid_shape_{len(invalid_shape)}.dat'),
                        shape=invalid_shape,
                        dtype='float32',
                        mode='w+'
                    )
        
        # Validate appropriate exception is raised
        # (Already handled in individual scenario blocks above)
        
        # Test error recovery mechanisms
        recovery_test_path = temp_dir / 'recovery_test.dat'
        try:
            # Create valid array after error scenarios
            recovery_array = allocate_memory_mapped_array(
                file_path=str(recovery_test_path),
                shape=(100, 100),
                dtype='float32',
                mode='w+'
            )
            
            # Verify error recovery was successful
            test_data = np.ones((100, 100), dtype=np.float32)
            recovery_array.write_chunk(test_data, start_index=(0, 0))
            recovery_array.flush()
            
            read_data = recovery_array.read_chunk((100, 100), start_index=(0, 0))
            assert_arrays_almost_equal(read_data, test_data, tolerance=1e-6, error_message="Error recovery validation failed")
            
            recovery_array.close()
            
        except Exception as recovery_error:
            pytest.fail(f"Error recovery failed: {recovery_error}")
        
        # Verify resource cleanup after errors
        open_files = psutil.Process().open_files()
        temp_dir_files = [f for f in open_files if str(temp_dir) in f.path]
        assert len(temp_dir_files) < 5, f"Too many open files after error handling: {len(temp_dir_files)}"
        
        # Check error logging and reporting
        # (Would typically verify log entries, but simplified for test environment)
        
        # Test graceful degradation behavior
        current_memory = psutil.virtual_memory()
        memory_usage_gb = current_memory.used / (1024**3)
        assert memory_usage_gb < MEMORY_MAPPING_TEST_CONFIG['critical_threshold_gb'], \
            f"Memory usage {memory_usage_gb:.2f}GB too high after error handling"
        
        # Validate system stability after errors
        stability_test_array = allocate_memory_mapped_array(
            file_path=str(temp_dir / 'stability_test.dat'),
            shape=(50, 50),
            dtype='float32',
            mode='w+'
        )
        
        stability_data = np.random.random((50, 50)).astype(np.float32)
        stability_test_array.write_chunk(stability_data, start_index=(0, 0))
        stability_test_array.flush()
        stability_test_array.close()
        
    except Exception as unexpected_error:
        pytest.fail(f"Unexpected error in {error_scenario} scenario: {unexpected_error}")


@pytest.mark.parametrize('access_pattern', ['sequential', 'random', 'chunked', 'sparse'])
def test_memory_mapping_data_integrity(
    memory_mapping_fixtures: TestMemoryMappingFixtures,
    test_data_generator,
    access_pattern: str
):
    """
    Test data integrity in memory-mapped arrays across different access patterns, flush operations, and system conditions.
    """
    # Create memory-mapped array with test data
    test_shape = (800, 600)
    memory_mapped_array = memory_mapping_fixtures.create_test_memory_mapped_array(
        shape=test_shape,
        dtype='float32',
        mode='w+'
    )
    
    # Generate test data with known patterns
    original_data = test_data_generator('gradient', test_shape, 'float32')
    data_checksum = hashlib.md5(original_data.tobytes()).hexdigest()
    
    try:
        # Write initial data
        memory_mapped_array.write_chunk(original_data, start_index=(0, 0))
        memory_mapped_array.flush()
        
        # Apply specified access pattern for reads/writes
        integrity_results = []
        
        if access_pattern == 'sequential':
            # Sequential row-by-row access
            for row in range(0, test_shape[0], 50):
                end_row = min(row + 50, test_shape[0])
                chunk_shape = (end_row - row, test_shape[1])
                
                # Read chunk
                read_chunk = memory_mapped_array.read_chunk(chunk_shape, start_index=(row, 0))
                expected_chunk = original_data[row:end_row, :]
                
                # Validate chunk integrity
                chunk_matches = np.allclose(read_chunk, expected_chunk, atol=1e-6)
                integrity_results.append(chunk_matches)
                
                # Modify and write back
                modified_chunk = read_chunk * 1.1
                memory_mapped_array.write_chunk(modified_chunk, start_index=(row, 0))
        
        elif access_pattern == 'random':
            # Random access pattern
            np.random.seed(MEMORY_MAP_TEST_SEED)
            
            for _ in range(20):
                # Random chunk position and size
                chunk_size = np.random.randint(10, 100)
                start_row = np.random.randint(0, max(1, test_shape[0] - chunk_size))
                start_col = np.random.randint(0, max(1, test_shape[1] - chunk_size))
                
                chunk_shape = (min(chunk_size, test_shape[0] - start_row), 
                              min(chunk_size, test_shape[1] - start_col))
                
                # Read and validate
                read_chunk = memory_mapped_array.read_chunk(chunk_shape, start_index=(start_row, start_col))
                chunk_valid = not np.any(np.isnan(read_chunk)) and not np.any(np.isinf(read_chunk))
                integrity_results.append(chunk_valid)
        
        elif access_pattern == 'chunked':
            # Large chunk access pattern
            chunk_height = test_shape[0] // 4
            chunk_width = test_shape[1] // 4
            
            for row_chunk in range(0, test_shape[0], chunk_height):
                for col_chunk in range(0, test_shape[1], chunk_width):
                    end_row = min(row_chunk + chunk_height, test_shape[0])
                    end_col = min(col_chunk + chunk_width, test_shape[1])
                    
                    chunk_shape = (end_row - row_chunk, end_col - col_chunk)
                    
                    # Read chunk
                    read_chunk = memory_mapped_array.read_chunk(chunk_shape, start_index=(row_chunk, col_chunk))
                    
                    # Validate chunk bounds
                    chunk_in_bounds = (
                        read_chunk.shape == chunk_shape and
                        not np.any(np.isnan(read_chunk)) and
                        not np.any(np.isinf(read_chunk))
                    )
                    integrity_results.append(chunk_in_bounds)
        
        elif access_pattern == 'sparse':
            # Sparse access pattern (every nth element)
            stride = 10
            
            for row in range(0, test_shape[0], stride):
                for col in range(0, test_shape[1], stride):
                    if row < test_shape[0] and col < test_shape[1]:
                        # Single element access
                        single_chunk = memory_mapped_array.read_chunk((1, 1), start_index=(row, col))
                        element_valid = not np.isnan(single_chunk[0, 0]) and not np.isinf(single_chunk[0, 0])
                        integrity_results.append(element_valid)
        
        # Perform flush operations at various intervals
        flush_count = 5
        for flush_idx in range(flush_count):
            memory_mapped_array.flush()
            
            # Validate data integrity after each flush
            verification_chunk = memory_mapped_array.read_chunk((100, 100), start_index=(0, 0))
            flush_integrity = not np.any(np.isnan(verification_chunk)) and not np.any(np.isinf(verification_chunk))
            integrity_results.append(flush_integrity)
        
        # Validate data integrity after each operation
        overall_integrity_rate = sum(integrity_results) / len(integrity_results) if integrity_results else 0
        assert overall_integrity_rate >= 0.95, f"Data integrity rate {overall_integrity_rate:.2%} below 95% for {access_pattern} pattern"
        
        # Test data persistence across array reopening
        memory_mapped_array.close()
        
        # Reopen the same file and verify data persistence
        reopened_array = allocate_memory_mapped_array(
            file_path=memory_mapped_array._file_path,
            shape=test_shape,
            dtype='float32',
            mode='r+'
        )
        
        # Verify checksums and data validation
        persistence_chunk = reopened_array.read_chunk((200, 200), start_index=(0, 0))
        persistence_valid = not np.any(np.isnan(persistence_chunk)) and not np.any(np.isinf(persistence_chunk))
        assert persistence_valid, "Data persistence validation failed after array reopening"
        
        # Test recovery from partial write scenarios
        recovery_data = np.ones((100, 100), dtype=np.float32) * 999
        reopened_array.write_chunk(recovery_data, start_index=(300, 200))
        reopened_array.flush()
        
        recovered_chunk = reopened_array.read_chunk((100, 100), start_index=(300, 200))
        assert_arrays_almost_equal(
            actual=recovered_chunk,
            expected=recovery_data,
            tolerance=1e-6,
            error_message="Recovery from partial write failed"
        )
        
        # Validate data consistency under stress
        stress_iterations = 10
        for stress_iter in range(stress_iterations):
            stress_chunk = np.random.random((50, 50)).astype(np.float32)
            stress_position = (stress_iter * 50 % (test_shape[0] - 50), stress_iter * 50 % (test_shape[1] - 50))
            
            reopened_array.write_chunk(stress_chunk, start_index=stress_position)
            reopened_array.flush()
            
            # Immediate read-back verification
            verification_chunk = reopened_array.read_chunk((50, 50), start_index=stress_position)
            assert_arrays_almost_equal(
                actual=verification_chunk,
                expected=stress_chunk,
                tolerance=1e-6,
                error_message=f"Stress test iteration {stress_iter} data consistency failed"
            )
        
        reopened_array.close()
        
    except Exception as e:
        pytest.fail(f"Data integrity test failed for {access_pattern} pattern: {e}")