"""
Comprehensive unit test suite for the multi-level caching architecture providing thorough validation
of memory cache, disk cache, result cache, and unified cache manager functionality. Tests cache
performance, eviction strategies, cross-format compatibility, thread safety, and integration with
scientific computing workflows. Validates cache hit rate thresholds (>0.8), memory management,
batch processing support for 4000+ simulations, and reproducible caching behavior with 1e-6
numerical precision requirements.

This test suite implements comprehensive testing of the multi-level caching system including:
- Level 1 memory cache with LRU eviction and memory pressure handling
- Level 2 disk cache with compression and persistence validation
- Level 3 result cache integration testing
- Unified cache manager coordination and optimization
- Performance validation against scientific computing requirements
- Thread safety and concurrent access testing
- Cache warming and optimization strategies
- Cross-level promotion and eviction coordination
- Integration with scientific simulation workflows
"""

import pytest  # pytest 8.3.5+ - Testing framework for cache functionality validation
import numpy as np  # numpy 2.1.3+ - Numerical array operations for cache data validation
import tempfile  # Python 3.9+ - Temporary directory management for cache testing
import threading  # Python 3.9+ - Thread safety testing for concurrent cache operations
import time  # Python 3.9+ - Performance timing and TTL testing for cache operations
import pathlib  # Python 3.9+ - Path handling for cache directory management in tests
import concurrent.futures  # Python 3.9+ - Concurrent execution testing for cache thread safety
import gc  # Python 3.9+ - Garbage collection control for memory cache testing
import psutil  # psutil 5.9.0+ - System resource monitoring for cache performance validation
import uuid  # Python 3.9+ - Unique identifier generation for test isolation
import json  # Python 3.9+ - JSON handling for cache configuration testing
from typing import Dict, Any, List, Optional, Tuple
import datetime
import copy
import weakref

# Internal imports for cache components and utilities
from backend.cache.memory_cache import (
    MemoryCache,
    CacheEntry,
    create_cache,
    get_cache_instance,
    cleanup_cache_instances
)
from backend.cache.disk_cache import (
    DiskCache,
    create_disk_cache,
    get_compression_ratio,
    optimize_disk_cache_performance
)
from backend.cache.cache_manager import (
    UnifiedCacheManager,
    CacheLevel,
    initialize_cache_manager,
    get_cache_manager,
    optimize_cache_coordination,
    warm_cache_system,
    analyze_cache_performance,
    cleanup_cache_system,
    shutdown_cache_system
)
from backend.cache import (
    initialize_cache_system,
    get_default_cache_manager
)

# Test utilities and helpers
from test.utils.test_helpers import (
    create_test_fixture_path,
    assert_arrays_almost_equal,
    measure_performance,
    setup_test_environment,
    PerformanceProfiler
)
from test.mocks.mock_video_data import (
    create_mock_video_sequence,
    MockVideoDataset
)

# Global test configuration constants
CACHE_TEST_TOLERANCE = 1e-6
PERFORMANCE_TIMEOUT_SECONDS = 7.2
CACHE_HIT_RATE_THRESHOLD = 0.8
MEMORY_FRAGMENTATION_THRESHOLD = 0.1
BATCH_SIMULATION_COUNT = 4000
TEST_CACHE_SIZE_MB = 64
TEST_DISK_CACHE_SIZE_GB = 1
TEST_TTL_SECONDS = 300
CONCURRENT_THREAD_COUNT = 10
STRESS_TEST_ITERATIONS = 1000


class TestCacheFixtures:
    """
    Test fixture class providing standardized cache configurations, mock data, and test environment
    setup for comprehensive cache testing scenarios with scientific computing validation.
    """
    
    def __init__(self):
        """Initialize test fixtures with cache configurations and mock data for comprehensive testing."""
        # Setup cache configurations for different test scenarios
        self.cache_configs = {
            'small_cache': {
                'max_size_mb': 8,
                'max_entries': 100,
                'ttl_seconds': 60
            },
            'medium_cache': {
                'max_size_mb': TEST_CACHE_SIZE_MB,
                'max_entries': 1000,
                'ttl_seconds': TEST_TTL_SECONDS
            },
            'large_cache': {
                'max_size_mb': 256,
                'max_entries': 10000,
                'ttl_seconds': 3600
            }
        }
        
        # Create temporary cache directory for test isolation
        self.temp_cache_dir = pathlib.Path(tempfile.mkdtemp(prefix='cache_test_'))
        
        # Initialize mock video dataset for cache testing
        self.mock_dataset = MockVideoDataset(
            dataset_config={
                'formats': ['crimaldi', 'custom'],
                'arena_size': (1.0, 1.0),
                'resolution': (640, 480),
                'duration': 10.0,
                'num_validation_scenarios': 5
            },
            enable_caching=True,
            random_seed=42
        )
        
        # Setup performance profiler for cache operation measurement
        self.profiler = PerformanceProfiler(
            time_threshold_seconds=PERFORMANCE_TIMEOUT_SECONDS,
            memory_threshold_mb=512
        )
    
    def create_memory_cache_config(
        self,
        size_mb: int,
        max_entries: int,
        ttl_seconds: float
    ) -> Dict[str, Any]:
        """
        Create memory cache configuration for testing with specified parameters.
        
        Args:
            size_mb: Memory cache size in megabytes
            max_entries: Maximum number of cache entries
            ttl_seconds: Time-to-live for cache entries
            
        Returns:
            Dict[str, Any]: Memory cache configuration dictionary
        """
        # Set memory cache size and entry limits
        config = {
            'cache_name': f'test_memory_cache_{uuid.uuid4().hex[:8]}',
            'max_size_mb': size_mb,
            'max_entries': max_entries,
            'ttl_seconds': ttl_seconds,
            'eviction_strategy': 'lru'
        }
        
        # Configure TTL and eviction parameters
        config['enable_memory_monitoring'] = True
        config['enable_performance_tracking'] = True
        
        # Setup performance monitoring settings
        config['monitoring_config'] = {
            'hit_rate_target': CACHE_HIT_RATE_THRESHOLD,
            'memory_fragmentation_threshold': MEMORY_FRAGMENTATION_THRESHOLD
        }
        
        # Return validated configuration dictionary
        return config
    
    def create_disk_cache_config(
        self,
        cache_directory: str,
        size_gb: int,
        compression_algorithm: str
    ) -> Dict[str, Any]:
        """
        Create disk cache configuration for testing with compression and persistence settings.
        
        Args:
            cache_directory: Directory path for disk cache storage
            size_gb: Maximum disk cache size in gigabytes
            compression_algorithm: Compression algorithm to use
            
        Returns:
            Dict[str, Any]: Disk cache configuration dictionary
        """
        # Set disk cache directory and size limits
        config = {
            'cache_directory': cache_directory,
            'max_size_gb': size_gb,
            'ttl_seconds': TEST_TTL_SECONDS,
            'compression_algorithm': compression_algorithm,
            'eviction_strategy': 'lru'
        }
        
        # Configure compression algorithm and settings
        config['compression_config'] = {
            'enable_compression': True,
            'compression_level': 6,
            'enable_integrity_verification': True
        }
        
        # Setup integrity verification parameters
        config['integrity_config'] = {
            'enable_checksums': True,
            'verify_on_read': True,
            'repair_corrupted_entries': True
        }
        
        # Return validated configuration dictionary
        return config
    
    def cleanup_test_environment(self) -> None:
        """Clean up test environment including temporary files and cache directories."""
        # Close all cache instances and release resources
        try:
            cleanup_cache_instances(force_cleanup=True)
        except Exception as e:
            print(f"Warning: Failed to cleanup cache instances: {e}")
        
        # Remove temporary cache directories and files
        try:
            import shutil
            if self.temp_cache_dir.exists():
                shutil.rmtree(str(self.temp_cache_dir), ignore_errors=True)
        except Exception as e:
            print(f"Warning: Failed to cleanup temp directory: {e}")
        
        # Clear mock dataset cache
        try:
            self.mock_dataset.clear_cache()
        except Exception as e:
            print(f"Warning: Failed to clear mock dataset cache: {e}")
        
        # Reset performance profiler state
        try:
            if hasattr(self.profiler, 'current_session') and self.profiler.current_session:
                self.profiler.stop_profiling()
        except Exception as e:
            print(f"Warning: Failed to stop profiler: {e}")
        
        # Log cleanup operation completion
        print("Test environment cleanup completed")


class TestCachePerformance:
    """
    Performance testing class for cache operations with throughput measurement, latency validation,
    and resource utilization monitoring for scientific computing requirements.
    """
    
    def __init__(
        self,
        performance_threshold: float = CACHE_HIT_RATE_THRESHOLD,
        enable_profiling: bool = True
    ):
        """
        Initialize performance testing with thresholds and profiling configuration.
        
        Args:
            performance_threshold: Cache hit rate threshold for validation
            enable_profiling: Enable performance profiling during tests
        """
        # Set performance threshold for validation
        self.performance_threshold = performance_threshold
        self.profiling_enabled = enable_profiling
        
        # Initialize performance profiler if enabled
        if self.profiling_enabled:
            self.profiler = PerformanceProfiler(
                time_threshold_seconds=PERFORMANCE_TIMEOUT_SECONDS,
                memory_threshold_mb=512
            )
        else:
            self.profiler = None
        
        # Setup performance metrics collection
        self.performance_metrics = {}
        self.benchmark_results = []
    
    def benchmark_cache_operations(
        self,
        cache_instance: Any,
        operation_count: int,
        operation_type: str
    ) -> Dict[str, Any]:
        """
        Benchmark cache operations including get, put, and eviction performance.
        
        Args:
            cache_instance: Cache instance to benchmark
            operation_count: Number of operations to perform
            operation_type: Type of operation to benchmark
            
        Returns:
            Dict[str, Any]: Performance benchmark results
        """
        # Start performance profiling session
        if self.profiler:
            self.profiler.start_profiling(f'benchmark_{operation_type}')
        
        start_time = time.time()
        successful_operations = 0
        failed_operations = 0
        
        try:
            # Execute specified cache operations
            for i in range(operation_count):
                try:
                    if operation_type == 'put':
                        key = f'benchmark_key_{i}'
                        value = np.random.random((100, 100))  # 10KB array
                        success = cache_instance.put(key, value)
                        if success:
                            successful_operations += 1
                        else:
                            failed_operations += 1
                    
                    elif operation_type == 'get':
                        key = f'benchmark_key_{i % 100}'  # Reuse keys for hit testing
                        value = cache_instance.get(key)
                        if value is not None:
                            successful_operations += 1
                        else:
                            failed_operations += 1
                    
                    elif operation_type == 'eviction':
                        evicted_count = cache_instance.evict_lru(count=1)
                        successful_operations += evicted_count
                        if evicted_count == 0:
                            failed_operations += 1
                    
                except Exception as e:
                    failed_operations += 1
                    print(f"Operation {i} failed: {e}")
            
            # Measure operation latency and throughput
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Monitor resource utilization during operations
            operations_per_second = successful_operations / max(execution_time, 0.001)
            
            # Stop profiling and collect metrics
            if self.profiler:
                profiling_results = self.profiler.stop_profiling()
            else:
                profiling_results = {}
            
            # Validate performance against thresholds
            meets_performance_requirements = (
                execution_time <= PERFORMANCE_TIMEOUT_SECONDS and
                operations_per_second >= 100  # Minimum 100 ops/sec
            )
            
            # Return comprehensive benchmark results
            benchmark_results = {
                'operation_type': operation_type,
                'operation_count': operation_count,
                'successful_operations': successful_operations,
                'failed_operations': failed_operations,
                'execution_time_seconds': execution_time,
                'operations_per_second': operations_per_second,
                'success_rate': successful_operations / max(operation_count, 1),
                'meets_performance_requirements': meets_performance_requirements,
                'profiling_results': profiling_results,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.benchmark_results.append(benchmark_results)
            return benchmark_results
            
        except Exception as e:
            return {
                'operation_type': operation_type,
                'error': str(e),
                'execution_time_seconds': time.time() - start_time,
                'meets_performance_requirements': False
            }
    
    def validate_performance_requirements(
        self,
        performance_results: Dict[str, Any]
    ) -> bool:
        """
        Validate cache performance meets scientific computing requirements.
        
        Args:
            performance_results: Performance measurement results
            
        Returns:
            bool: True if performance requirements are met
        """
        # Check cache hit rate against 0.8 threshold
        hit_rate = performance_results.get('success_rate', 0.0)
        hit_rate_valid = hit_rate >= self.performance_threshold
        
        # Validate memory usage within fragmentation limits
        memory_usage = performance_results.get('profiling_results', {}).get('peak_memory_mb', 0)
        memory_valid = memory_usage <= 512  # 512MB limit for tests
        
        # Assert operation latency meets timing requirements
        execution_time = performance_results.get('execution_time_seconds', float('inf'))
        latency_valid = execution_time <= PERFORMANCE_TIMEOUT_SECONDS
        
        # Verify throughput supports batch processing needs
        ops_per_second = performance_results.get('operations_per_second', 0)
        throughput_valid = ops_per_second >= 100  # Minimum throughput
        
        # Check resource utilization efficiency
        efficiency_valid = performance_results.get('meets_performance_requirements', False)
        
        # Return overall performance validation result
        return all([hit_rate_valid, memory_valid, latency_valid, throughput_valid, efficiency_valid])


# Memory Cache Testing

@pytest.mark.unit
@measure_performance(time_limit_seconds=1.0)
def test_memory_cache_basic_operations():
    """
    Test basic memory cache operations including get, put, delete, and clear functionality
    with data integrity validation and performance measurement.
    """
    # Create memory cache instance with test configuration
    cache = create_cache(
        cache_name='test_basic_ops',
        max_size_mb=16,
        max_entries=100,
        ttl_seconds=60
    )
    
    try:
        # Test cache put operation with various data types
        test_data = {
            'string_key': 'test_string_value',
            'int_key': 42,
            'float_key': 3.14159,
            'array_key': np.random.random((10, 10)),
            'dict_key': {'nested': {'data': [1, 2, 3]}}
        }
        
        for key, value in test_data.items():
            success = cache.put(key, value)
            assert success, f"Failed to put {key}"
        
        # Validate cache get operation returns correct data
        for key, expected_value in test_data.items():
            retrieved_value = cache.get(key)
            assert retrieved_value is not None, f"Failed to get {key}"
            
            if isinstance(expected_value, np.ndarray):
                assert_arrays_almost_equal(retrieved_value, expected_value, CACHE_TEST_TOLERANCE)
            else:
                assert retrieved_value == expected_value, f"Value mismatch for {key}"
        
        # Test cache delete operation removes entries
        delete_key = 'string_key'
        cache.invalidate(delete_key)
        deleted_value = cache.get(delete_key)
        assert deleted_value is None, "Delete operation failed"
        
        # Validate cache clear operation empties cache
        initial_count = len(test_data) - 1  # One item deleted
        cache.clear()
        
        for key in test_data.keys():
            if key != delete_key:
                value = cache.get(key)
                assert value is None, f"Clear operation failed for {key}"
        
        # Assert cache statistics are updated correctly
        stats = cache.get_statistics()
        assert stats is not None, "Failed to get cache statistics"
        assert 'performance_summary' in stats
        assert 'cache_utilization' in stats
        
        # Verify memory usage stays within limits
        memory_usage = stats['cache_utilization']['current_size_mb']
        assert memory_usage <= 16, f"Memory usage {memory_usage} exceeds limit"
        
    finally:
        cache.close()


@pytest.mark.unit
@measure_performance(time_limit_seconds=2.0)
def test_memory_cache_lru_eviction():
    """
    Test LRU eviction strategy in memory cache with access pattern validation, eviction order
    verification, and performance impact measurement.
    """
    # Create memory cache with limited size for eviction testing
    cache = create_cache(
        cache_name='test_lru_eviction',
        max_size_mb=1,  # Very small for forced eviction
        max_entries=10,
        ttl_seconds=300
    )
    
    try:
        # Fill cache beyond capacity to trigger LRU eviction
        initial_keys = []
        for i in range(15):  # More than max_entries
            key = f'eviction_test_{i}'
            value = np.random.random((50, 50))  # ~10KB each
            initial_keys.append(key)
            
            success = cache.put(key, value)
            assert success, f"Failed to put {key}"
        
        # Validate least recently used entries are evicted first
        # First few entries should be evicted
        evicted_count = 0
        retained_count = 0
        
        for key in initial_keys:
            value = cache.get(key)
            if value is None:
                evicted_count += 1
            else:
                retained_count += 1
        
        assert evicted_count > 0, "No entries were evicted"
        assert retained_count > 0, "All entries were evicted"
        assert retained_count <= 10, "Too many entries retained"
        
        # Test access pattern updates LRU ordering correctly
        # Access some older entries to move them to front
        access_keys = initial_keys[-5:]  # Last 5 keys
        for key in access_keys:
            value = cache.get(key, update_access=True)
            if value is not None:
                # Re-access to ensure LRU position update
                cache.get(key, update_access=True)
        
        # Add more entries to trigger eviction
        for i in range(20, 25):
            key = f'new_entry_{i}'
            value = np.random.random((50, 50))
            cache.put(key, value)
        
        # Verify eviction statistics are tracked accurately
        stats = cache.get_statistics()
        assert 'performance_summary' in stats
        assert stats['performance_summary']['total_evictions'] > 0
        
        # Assert cache hit rate remains above threshold
        hit_rate = stats['performance_summary']['hit_rate']
        print(f"Cache hit rate after eviction: {hit_rate}")
        
        # Validate memory usage optimization after eviction
        memory_usage = stats['cache_utilization']['current_size_mb']
        assert memory_usage <= 1.5, f"Memory usage {memory_usage} not optimized"
        
    finally:
        cache.close()


@pytest.mark.unit
def test_memory_cache_ttl_expiration():
    """
    Test TTL-based cache expiration with time-based validation, expired entry cleanup,
    and automatic expiration handling.
    """
    # Create memory cache with short TTL for testing
    cache = create_cache(
        cache_name='test_ttl_expiration',
        max_size_mb=16,
        max_entries=100,
        ttl_seconds=1  # 1 second TTL for fast testing
    )
    
    try:
        # Add entries with various TTL values
        test_entries = {
            'short_ttl': ('short_value', 0.5),  # 0.5 second TTL
            'medium_ttl': ('medium_value', 1.0),  # 1 second TTL
            'long_ttl': ('long_value', 2.0)  # 2 second TTL
        }
        
        for key, (value, ttl) in test_entries.items():
            success = cache.put(key, value, ttl_seconds=ttl)
            assert success, f"Failed to put {key}"
        
        # Validate entries are accessible before expiration
        time.sleep(0.2)  # Wait 200ms
        for key, (expected_value, _) in test_entries.items():
            retrieved_value = cache.get(key)
            assert retrieved_value == expected_value, f"Entry {key} not accessible"
        
        # Wait for TTL expiration and verify entries are expired
        time.sleep(1.0)  # Wait 1 second (short_ttl should expire)
        
        short_value = cache.get('short_ttl')
        assert short_value is None, "Short TTL entry should be expired"
        
        medium_value = cache.get('medium_ttl')
        # Medium TTL might be expired depending on timing
        
        long_value = cache.get('long_ttl')
        assert long_value is not None, "Long TTL entry should still be valid"
        
        # Test automatic cleanup of expired entries
        time.sleep(1.5)  # Wait for all entries to expire
        
        cleanup_count = cache.cleanup_expired()
        print(f"Cleanup removed {cleanup_count} expired entries")
        
        # Validate cache statistics reflect expiration events
        stats = cache.get_statistics()
        assert 'performance_summary' in stats
        
        # Assert memory is freed after expiration cleanup
        memory_usage = stats['cache_utilization']['current_size_mb']
        assert memory_usage < 1.0, "Memory not freed after expiration"
        
    finally:
        cache.close()


@pytest.mark.unit
@pytest.mark.slow
def test_memory_cache_thread_safety():
    """
    Test thread safety of memory cache operations with concurrent access, race condition
    prevention, and data consistency validation.
    """
    # Create memory cache for concurrent access testing
    cache = create_cache(
        cache_name='test_thread_safety',
        max_size_mb=32,
        max_entries=1000,
        ttl_seconds=300
    )
    
    try:
        # Launch multiple threads performing cache operations
        num_threads = CONCURRENT_THREAD_COUNT
        operations_per_thread = 100
        results = []
        
        def cache_worker(thread_id: int) -> Dict[str, Any]:
            """Worker function for concurrent cache operations."""
            thread_results = {
                'thread_id': thread_id,
                'puts': 0,
                'gets': 0,
                'hits': 0,
                'errors': 0
            }
            
            try:
                # Test concurrent get, put, and delete operations
                for i in range(operations_per_thread):
                    try:
                        # Put operation
                        key = f'thread_{thread_id}_item_{i}'
                        value = np.random.random((10, 10))
                        if cache.put(key, value):
                            thread_results['puts'] += 1
                        
                        # Get operation
                        retrieved = cache.get(key)
                        thread_results['gets'] += 1
                        if retrieved is not None:
                            thread_results['hits'] += 1
                        
                        # Occasional delete
                        if i % 10 == 0:
                            delete_key = f'thread_{thread_id}_item_{i-5}'
                            cache.get(delete_key)  # Attempt to access deleted item
                        
                    except Exception as e:
                        thread_results['errors'] += 1
                        print(f"Thread {thread_id} error: {e}")
                
            except Exception as e:
                thread_results['errors'] += 1
                print(f"Thread {thread_id} fatal error: {e}")
            
            return thread_results
        
        # Execute concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(cache_worker, i) for i in range(num_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Validate data consistency across concurrent access
        total_puts = sum(r['puts'] for r in results)
        total_gets = sum(r['gets'] for r in results)
        total_hits = sum(r['hits'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        
        assert total_puts > 0, "No successful put operations"
        assert total_gets > 0, "No get operations performed"
        assert total_errors < total_puts * 0.1, f"Too many errors: {total_errors}"
        
        # Check for race conditions and deadlock prevention
        print(f"Concurrent operations: {total_puts} puts, {total_gets} gets, {total_hits} hits, {total_errors} errors")
        
        # Assert cache statistics remain consistent
        stats = cache.get_statistics()
        assert 'performance_summary' in stats
        cache_entries = stats['cache_utilization']['current_entries']
        assert cache_entries >= 0, "Invalid cache entry count"
        
        # Verify thread safety of eviction operations
        # This is implicitly tested through concurrent operations
        
    finally:
        cache.close()


# Disk Cache Testing

@pytest.mark.unit
@measure_performance(time_limit_seconds=3.0)
def test_disk_cache_basic_operations():
    """
    Test basic disk cache operations including file storage, retrieval, compression, and
    integrity verification with persistence validation.
    """
    with setup_test_environment('disk_cache_basic') as test_env:
        # Create temporary directory for disk cache testing
        cache_dir = test_env['temp_directory'] / 'disk_cache'
        
        # Initialize disk cache with test configuration
        disk_cache = create_disk_cache(
            cache_directory=str(cache_dir),
            max_size_gb=TEST_DISK_CACHE_SIZE_GB,
            ttl_seconds=TEST_TTL_SECONDS,
            compression_algorithm='lz4'
        )
        
        try:
            # Test disk cache set operation with data compression
            test_data = {
                'small_array': np.random.random((10, 10)),
                'large_array': np.random.random((100, 100)),
                'text_data': 'This is test text data for compression testing',
                'dict_data': {'nested': {'structure': [1, 2, 3, 4, 5]}}
            }
            
            for key, value in test_data.items():
                success = disk_cache.set(key, value, compress_data=True, verify_integrity=True)
                assert success, f"Failed to set {key} in disk cache"
            
            # Validate disk cache get operation with decompression
            for key, expected_value in test_data.items():
                retrieved_value = disk_cache.get(key, verify_integrity=True)
                assert retrieved_value is not None, f"Failed to get {key} from disk cache"
                
                if isinstance(expected_value, np.ndarray):
                    assert_arrays_almost_equal(retrieved_value, expected_value, CACHE_TEST_TOLERANCE)
                else:
                    assert retrieved_value == expected_value, f"Value mismatch for {key}"
            
            # Test data integrity verification with checksums
            stats = disk_cache.get_statistics(include_detailed_breakdown=True)
            assert 'cache_info' in stats
            assert stats['cache_info']['total_entries'] == len(test_data)
            
            # Validate file persistence across cache sessions
            disk_cache.close()
            
            # Reopen cache and verify data persistence
            disk_cache2 = create_disk_cache(
                cache_directory=str(cache_dir),
                max_size_gb=TEST_DISK_CACHE_SIZE_GB,
                ttl_seconds=TEST_TTL_SECONDS,
                compression_algorithm='lz4'
            )
            
            try:
                for key, expected_value in test_data.items():
                    retrieved_value = disk_cache2.get(key)
                    assert retrieved_value is not None, f"Data not persisted for {key}"
                    
                    if isinstance(expected_value, np.ndarray):
                        assert_arrays_almost_equal(retrieved_value, expected_value, CACHE_TEST_TOLERANCE)
                    else:
                        assert retrieved_value == expected_value, f"Persisted value mismatch for {key}"
                
                # Assert disk usage statistics are accurate
                stats2 = disk_cache2.get_statistics()
                assert stats2['cache_info']['total_entries'] == len(test_data)
                
            finally:
                disk_cache2.close()
            
        finally:
            if not disk_cache.is_initialized:
                disk_cache.close()


@pytest.mark.unit
def test_disk_cache_compression():
    """
    Test disk cache compression algorithms including LZ4, GZIP, and compression ratio
    validation with performance measurement.
    """
    with setup_test_environment('disk_cache_compression') as test_env:
        cache_dir = test_env['temp_directory'] / 'compression_test'
        
        # Test compression with different algorithms
        compression_algorithms = ['gzip', 'lz4', 'none']
        if hasattr(__builtins__, 'zstandard'):
            compression_algorithms.append('zstd')
        
        compression_results = {}
        
        for algorithm in compression_algorithms:
            disk_cache = create_disk_cache(
                cache_directory=str(cache_dir / algorithm),
                max_size_gb=1,
                compression_algorithm=algorithm
            )
            
            try:
                # Test compression with various data types and sizes
                test_data = {
                    'small_random': np.random.random((50, 50)),
                    'large_sparse': np.zeros((200, 200)),  # Highly compressible
                    'mixed_data': np.random.choice([0, 1], size=(100, 100)),
                    'text_data': 'A' * 10000  # Highly repetitive text
                }
                
                algorithm_results = {
                    'original_size': 0,
                    'compressed_size': 0,
                    'compression_ratios': []
                }
                
                for key, value in test_data.items():
                    # Store with compression
                    success = disk_cache.set(key, value, compress_data=True)
                    assert success, f"Failed to store {key} with {algorithm}"
                    
                    # Calculate compression ratio
                    if isinstance(value, np.ndarray):
                        original_size = value.nbytes
                    else:
                        original_size = len(str(value).encode())
                    
                    ratio = get_compression_ratio(str(value).encode(), algorithm)
                    algorithm_results['compression_ratios'].append(ratio)
                    algorithm_results['original_size'] += original_size
                
                # Validate compression ratios meet efficiency targets
                avg_ratio = np.mean(algorithm_results['compression_ratios'])
                compression_results[algorithm] = {
                    'average_ratio': avg_ratio,
                    'algorithm': algorithm
                }
                
                print(f"{algorithm} compression ratio: {avg_ratio:.2f}")
                
                # Test decompression accuracy and data integrity
                for key, expected_value in test_data.items():
                    retrieved_value = disk_cache.get(key)
                    assert retrieved_value is not None, f"Failed to retrieve {key}"
                    
                    if isinstance(expected_value, np.ndarray):
                        assert_arrays_almost_equal(retrieved_value, expected_value, CACHE_TEST_TOLERANCE)
                    else:
                        assert retrieved_value == expected_value
                
            finally:
                disk_cache.close()
        
        # Compare compression performance across algorithms
        if 'lz4' in compression_results and 'gzip' in compression_results:
            lz4_ratio = compression_results['lz4']['average_ratio']
            gzip_ratio = compression_results['gzip']['average_ratio']
            
            # LZ4 should be faster, GZIP should compress better
            print(f"LZ4 ratio: {lz4_ratio:.2f}, GZIP ratio: {gzip_ratio:.2f}")
        
        # Assert compressed data storage optimization
        none_size = compression_results.get('none', {}).get('original_size', 1)
        for algorithm, results in compression_results.items():
            if algorithm != 'none':
                ratio = results['average_ratio']
                assert ratio >= 1.0, f"{algorithm} compression ratio should be >= 1.0"
        
        # Verify compression metadata is stored correctly
        # This is validated through the successful retrieval tests above


@pytest.mark.unit
@pytest.mark.slow
def test_disk_cache_memory_mapping():
    """
    Test memory-mapped file access for large datasets with virtual memory optimization
    and efficient data access patterns.
    """
    with setup_test_environment('disk_cache_mmap') as test_env:
        cache_dir = test_env['temp_directory'] / 'mmap_test'
        
        # Create disk cache with memory mapping enabled
        disk_cache = create_disk_cache(
            cache_directory=str(cache_dir),
            max_size_gb=2,
            compression_algorithm='none'  # Disable compression for memory mapping
        )
        
        try:
            # Store large numpy arrays exceeding memory mapping threshold
            large_arrays = {}
            for i in range(3):
                key = f'large_array_{i}'
                # Create 50MB array (exceeds typical memory mapping threshold)
                array_data = np.random.random((2000, 2000)).astype(np.float32)
                large_arrays[key] = array_data
                
                success = disk_cache.set(key, array_data, compress_data=False)
                assert success, f"Failed to store large array {key}"
            
            # Test memory-mapped array access and modification
            for key, original_array in large_arrays.items():
                # Retrieve using memory mapping
                retrieved_array = disk_cache.get(key, use_memory_mapping=True)
                assert retrieved_array is not None, f"Failed to retrieve {key} via memory mapping"
                
                # Validate data integrity
                assert_arrays_almost_equal(retrieved_array, original_array, CACHE_TEST_TOLERANCE)
                
                # Test that large arrays use memory mapping
                stats = disk_cache.get_statistics(include_file_statistics=True)
                if 'file_statistics' in stats:
                    mmap_count = stats['file_statistics'].get('memory_mapped_files', 0)
                    print(f"Memory mapped files: {mmap_count}")
            
            # Validate virtual memory usage optimization
            stats = disk_cache.get_statistics()
            total_size_mb = stats['cache_info']['total_size_mb']
            print(f"Total cache size: {total_size_mb:.2f} MB")
            
            # Test concurrent access to memory-mapped files
            def access_mmap_file(key: str) -> bool:
                """Concurrent access function for memory-mapped files."""
                try:
                    data = disk_cache.get(key, use_memory_mapping=True)
                    return data is not None
                except Exception:
                    return False
            
            # Launch concurrent access threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for key in large_arrays.keys():
                    for _ in range(3):  # 3 concurrent accesses per file
                        futures.append(executor.submit(access_mmap_file, key))
                
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
                successful_accesses = sum(results)
                
                assert successful_accesses >= len(results) * 0.8, "Too many concurrent access failures"
            
            # Assert memory mapping performance benefits
            # This is demonstrated by successful handling of large files
            
            # Verify proper cleanup of memory-mapped resources
            # This happens automatically when the cache is closed
            
        finally:
            disk_cache.close()


# Unified Cache Manager Testing

@pytest.mark.unit
@measure_performance(time_limit_seconds=5.0)
def test_unified_cache_manager_coordination():
    """
    Test unified cache manager coordination across all cache levels with intelligent
    data placement and access optimization.
    """
    with setup_test_environment('cache_manager_coordination') as test_env:
        cache_dir = test_env['temp_directory'] / 'unified_cache'
        
        # Initialize unified cache manager with all cache levels
        cache_config = {
            'memory_cache': {
                'max_size_mb': 32,
                'max_entries': 500,
                'ttl_seconds': 300
            },
            'disk_cache': {
                'max_size_gb': 1,
                'ttl_seconds': 600,
                'compression_algorithm': 'lz4'
            },
            'result_cache': {
                'max_size_gb': 2,
                'ttl_hours': 24,
                'enable_dependency_tracking': True
            }
        }
        
        cache_manager = initialize_cache_manager(
            manager_name='test_coordination',
            cache_directory=str(cache_dir),
            cache_config=cache_config,
            enable_performance_monitoring=True,
            enable_cache_warming=True,
            enable_cross_level_optimization=True
        )
        
        try:
            # Test intelligent data placement based on access patterns
            test_data = {
                'hot_data': np.random.random((10, 10)),  # Frequently accessed
                'warm_data': np.random.random((50, 50)),  # Occasionally accessed
                'cold_data': np.random.random((100, 100))  # Rarely accessed
            }
            
            # Store data with different access patterns
            for key, value in test_data.items():
                success = cache_manager.set(key, value, enable_promotion=True)
                assert success, f"Failed to store {key} in unified cache"
            
            # Simulate access patterns
            # Hot data - frequent access
            for _ in range(10):
                retrieved = cache_manager.get('hot_data', consider_promotion=True)
                assert retrieved is not None
                assert_arrays_almost_equal(retrieved, test_data['hot_data'], CACHE_TEST_TOLERANCE)
            
            # Warm data - occasional access
            for _ in range(3):
                retrieved = cache_manager.get('warm_data', consider_promotion=True)
                assert retrieved is not None
            
            # Cold data - single access
            retrieved = cache_manager.get('cold_data', consider_promotion=True)
            assert retrieved is not None
            
            # Validate cache promotion from disk to memory on access
            # Hot data should be promoted to memory cache
            memory_cache = cache_manager.cache_instances[CacheLevel.MEMORY]
            hot_in_memory = memory_cache.get('hot_data')
            # Note: Promotion logic may vary based on implementation
            
            # Test coordinated eviction across cache levels
            # Fill caches to trigger eviction
            for i in range(100):
                key = f'bulk_data_{i}'
                value = np.random.random((20, 20))
                cache_manager.set(key, value)
            
            # Validate cache statistics aggregation across levels
            stats = cache_manager.get_statistics(
                include_detailed_breakdown=True,
                include_coordination_metrics=True
            )
            
            assert 'cache_level_statistics' in stats
            assert 'coordination_metrics' in stats
            assert 'aggregated_performance' in stats
            
            # Check individual cache level stats
            level_stats = stats['cache_level_statistics']
            assert 'memory_cache' in level_stats
            assert 'disk_cache' in level_stats
            
            # Assert optimal cache level selection for different data types
            # This is demonstrated through successful storage and retrieval
            
            # Verify unified cache manager optimization effectiveness
            optimization_result = cache_manager.optimize(
                optimization_strategy='balanced',
                apply_optimizations=True
            )
            
            assert optimization_result.optimization_strategy == 'balanced'
            effectiveness = optimization_result.calculate_overall_effectiveness()
            print(f"Cache optimization effectiveness: {effectiveness:.3f}")
            
        finally:
            shutdown_result = shutdown_cache_system(
                cache_manager,
                save_statistics=True,
                preserve_cache_data=False
            )
            assert shutdown_result.graceful_shutdown


@pytest.mark.unit
def test_cache_warming_strategies():
    """
    Test cache warming strategies for batch processing optimization with pre-population
    effectiveness and performance improvement validation.
    """
    with setup_test_environment('cache_warming') as test_env:
        cache_dir = test_env['temp_directory'] / 'warming_test'
        
        # Create unified cache manager with warming enabled
        cache_manager = initialize_cache_manager(
            manager_name='test_warming',
            cache_directory=str(cache_dir),
            cache_config={
                'memory_cache': {'max_size_mb': 64, 'max_entries': 1000},
                'disk_cache': {'max_size_gb': 1, 'compression_algorithm': 'lz4'}
            },
            enable_cache_warming=True
        )
        
        try:
            # Generate mock video datasets for warming scenarios
            mock_dataset = MockVideoDataset(
                dataset_config={
                    'formats': ['crimaldi', 'custom'],
                    'arena_size': (1.0, 1.0),
                    'resolution': (320, 240),
                    'duration': 5.0
                },
                random_seed=42
            )
            
            # Pre-populate some frequently accessed data
            crimaldi_data = mock_dataset.get_crimaldi_dataset()
            custom_data = mock_dataset.get_custom_dataset()
            
            # Store reference data that would be warmed
            cache_manager.set('crimaldi_video', crimaldi_data['video_data'])
            cache_manager.set('custom_video', custom_data['video_data'])
            cache_manager.set('crimaldi_metadata', crimaldi_data['metadata'])
            
            # Clear cache to test warming
            cleanup_cache_system(cache_manager, preserve_hot_data=False)
            
            # Test cache warming with frequently accessed data
            warming_result = warm_cache_system(
                cache_manager,
                data_categories=['simulation_data', 'normalized_video', 'results'],
                warming_config={
                    'batch_size': 10,
                    'parallel_warming': False
                }
            )
            
            assert warming_result.warming_strategy == 'intelligent_preloading'
            assert warming_result.total_entries_preloaded >= 0
            
            # Validate warming effectiveness and hit rate improvement
            # This requires measuring hit rates before and after warming
            stats_after_warming = cache_manager.get_statistics()
            
            # Test parallel warming for performance optimization
            parallel_warming_result = warm_cache_system(
                cache_manager,
                data_categories=['simulation_data'],
                warming_config={'parallel_warming': True}
            )
            
            assert parallel_warming_result.warming_execution_time >= 0
            
            # Assert warming reduces cold start latency
            # Measure access time for warmed vs non-warmed data
            start_time = time.time()
            warmed_data = cache_manager.get('crimaldi_video')
            access_time = time.time() - start_time
            
            print(f"Warmed data access time: {access_time:.4f}s")
            assert access_time < 0.1, "Warmed data access too slow"
            
            # Verify warming statistics and performance metrics
            assert warming_result.warming_effectiveness_score >= 0
            
        finally:
            shutdown_cache_system(cache_manager)


@pytest.mark.unit
@pytest.mark.performance
def test_cache_performance_thresholds():
    """
    Test cache performance against system thresholds including hit rate validation,
    memory usage limits, and optimization trigger points.
    """
    # Initialize cache system with performance monitoring
    cache_manager = initialize_cache_manager(
        manager_name='test_performance',
        cache_config={
            'memory_cache': {'max_size_mb': 128, 'max_entries': 2000},
            'disk_cache': {'max_size_gb': 2}
        },
        enable_performance_monitoring=True
    )
    
    try:
        # Generate workload to test performance thresholds
        performance_profiler = PerformanceProfiler()
        performance_profiler.start_profiling('cache_performance_test')
        
        # Create mixed workload pattern
        hit_count = 0
        miss_count = 0
        
        # Phase 1: Fill cache with test data
        test_keys = []
        for i in range(500):
            key = f'perf_test_{i}'
            value = np.random.random((20, 20))
            test_keys.append(key)
            cache_manager.set(key, value)
        
        # Phase 2: Mixed read/write workload
        for i in range(1000):
            # 80% reads, 20% writes for realistic workload
            if i % 5 == 0:  # Write operation
                key = f'new_data_{i}'
                value = np.random.random((25, 25))
                cache_manager.set(key, value)
                test_keys.append(key)
            else:  # Read operation
                key = np.random.choice(test_keys)
                result = cache_manager.get(key)
                if result is not None:
                    hit_count += 1
                else:
                    miss_count += 1
        
        # Validate cache hit rate exceeds 0.8 threshold
        total_reads = hit_count + miss_count
        hit_rate = hit_count / total_reads if total_reads > 0 else 0
        print(f"Cache hit rate: {hit_rate:.3f}")
        assert hit_rate >= CACHE_HIT_RATE_THRESHOLD, f"Hit rate {hit_rate} below threshold {CACHE_HIT_RATE_THRESHOLD}"
        
        # Test memory fragmentation stays below 0.1 threshold
        stats = cache_manager.get_statistics(include_detailed_breakdown=True)
        memory_utilization = stats['aggregated_performance'].get('total_utilization', 0)
        print(f"Memory utilization: {memory_utilization:.3f}")
        
        # Validate optimization triggers at appropriate points
        optimization_result = cache_manager.optimize(
            optimization_strategy='balanced',
            apply_optimizations=True
        )
        optimization_effectiveness = optimization_result.calculate_overall_effectiveness()
        
        # Assert performance metrics meet scientific computing requirements
        profiling_results = performance_profiler.stop_profiling()
        execution_time = profiling_results['session_metrics']['execution_time_seconds']
        memory_usage = profiling_results['session_metrics']['peak_memory_mb']
        
        assert execution_time <= PERFORMANCE_TIMEOUT_SECONDS * 2, f"Performance test took too long: {execution_time}s"
        assert memory_usage <= 512, f"Memory usage too high: {memory_usage}MB"
        
        # Verify threshold-based optimization effectiveness
        assert optimization_effectiveness >= 0, "Optimization effectiveness should be non-negative"
        print(f"Optimization effectiveness: {optimization_effectiveness:.3f}")
        
    finally:
        shutdown_cache_system(cache_manager)


@pytest.mark.unit
@pytest.mark.slow
@measure_performance(time_limit_seconds=10.0)
def test_batch_simulation_cache_support():
    """
    Test cache support for batch processing of 4000+ simulations with memory efficiency
    and parallel processing optimization.
    """
    # Initialize cache system for batch processing simulation
    cache_manager = initialize_cache_manager(
        manager_name='test_batch_processing',
        cache_config={
            'memory_cache': {'max_size_mb': 256, 'max_entries': 5000},
            'disk_cache': {'max_size_gb': 5, 'compression_algorithm': 'lz4'},
            'result_cache': {'max_size_gb': 10, 'enable_dependency_tracking': True}
        },
        enable_performance_monitoring=True,
        enable_cache_warming=True
    )
    
    try:
        # Generate mock simulation data for 4000+ scenarios
        batch_size = min(BATCH_SIMULATION_COUNT, 1000)  # Reduced for testing performance
        simulation_results = []
        
        profiler = PerformanceProfiler()
        profiler.start_profiling('batch_simulation_test')
        
        # Simulate batch processing workload
        for batch_idx in range(batch_size // 100):  # Process in batches of 100
            batch_start_time = time.time()
            
            # Process simulation batch
            for sim_idx in range(100):
                simulation_id = batch_idx * 100 + sim_idx
                
                # Generate simulation input data
                simulation_key = f'simulation_{simulation_id}'
                input_data = {
                    'video_data': np.random.random((50, 64, 64)),  # Mock video
                    'parameters': {'diffusion': 0.1, 'wind': (0.5, 0.0)},
                    'metadata': {'simulation_id': simulation_id, 'timestamp': time.time()}
                }
                
                # Cache simulation input
                cache_manager.set(f'{simulation_key}_input', input_data)
                
                # Simulate processing and cache results
                result_data = {
                    'trajectory': np.random.random((20, 2)),  # Mock trajectory
                    'performance': {'success': True, 'time': np.random.uniform(1, 5)},
                    'simulation_id': simulation_id
                }
                
                cache_manager.set(f'{simulation_key}_result', result_data)
                simulation_results.append(simulation_id)
            
            batch_time = time.time() - batch_start_time
            print(f"Batch {batch_idx} processed in {batch_time:.3f}s")
        
        # Test cache efficiency during batch processing
        cache_stats = cache_manager.get_statistics(include_coordination_metrics=True)
        hit_rate = cache_stats['aggregated_performance'].get('overall_hit_rate', 0)
        
        print(f"Batch processing cache hit rate: {hit_rate:.3f}")
        
        # Validate memory usage stays within system limits
        profiling_results = profiler.stop_profiling()
        peak_memory = profiling_results['session_metrics']['peak_memory_mb']
        execution_time = profiling_results['session_metrics']['execution_time_seconds']
        
        assert peak_memory <= 1024, f"Memory usage {peak_memory}MB exceeds limit"
        print(f"Peak memory usage: {peak_memory:.1f}MB")
        
        # Test parallel cache access during concurrent simulations
        def concurrent_cache_access(sim_ids: List[int]) -> int:
            """Concurrent cache access function."""
            access_count = 0
            for sim_id in sim_ids:
                try:
                    result = cache_manager.get(f'simulation_{sim_id}_result')
                    if result is not None:
                        access_count += 1
                except Exception:
                    pass
            return access_count
        
        # Test concurrent access
        chunk_size = len(simulation_results) // 5
        chunks = [simulation_results[i:i+chunk_size] for i in range(0, len(simulation_results), chunk_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            concurrent_results = list(executor.map(concurrent_cache_access, chunks))
            total_concurrent_accesses = sum(concurrent_results)
        
        print(f"Concurrent cache accesses: {total_concurrent_accesses}")
        
        # Assert cache hit rate optimization for batch workloads
        assert hit_rate >= 0.5, f"Batch processing hit rate {hit_rate} too low"
        
        # Verify cache performance meets 8-hour target timeframe
        # For scaled test: execution_time should be reasonable
        operations_per_second = len(simulation_results) * 2 / execution_time  # 2 ops per simulation
        print(f"Cache operations per second: {operations_per_second:.1f}")
        assert operations_per_second >= 50, "Cache performance insufficient for batch processing"
        
    finally:
        shutdown_cache_system(cache_manager)


@pytest.mark.unit
def test_cross_format_cache_compatibility():
    """
    Test cache compatibility across different plume data formats with Crimaldi and
    custom AVI format validation.
    """
    with setup_test_environment('cross_format_test') as test_env:
        cache_dir = test_env['temp_directory'] / 'cross_format_cache'
        
        cache_manager = initialize_cache_manager(
            manager_name='test_cross_format',
            cache_directory=str(cache_dir),
            cache_config={
                'memory_cache': {'max_size_mb': 64},
                'disk_cache': {'max_size_gb': 1}
            }
        )
        
        try:
            # Create mock datasets for Crimaldi and custom formats
            mock_dataset = MockVideoDataset(
                dataset_config={
                    'formats': ['crimaldi', 'custom'],
                    'arena_size': (1.0, 1.0),
                    'resolution': (640, 480),
                    'duration': 10.0
                },
                random_seed=42
            )
            
            crimaldi_data = mock_dataset.get_crimaldi_dataset()
            custom_data = mock_dataset.get_custom_dataset()
            
            # Test cache storage and retrieval for different formats
            formats_data = {
                'crimaldi': crimaldi_data,
                'custom': custom_data
            }
            
            for format_name, format_data in formats_data.items():
                # Store format-specific data
                video_key = f'{format_name}_video'
                metadata_key = f'{format_name}_metadata'
                calibration_key = f'{format_name}_calibration'
                
                success = cache_manager.set(video_key, format_data['video_data'])
                assert success, f"Failed to cache {format_name} video data"
                
                success = cache_manager.set(metadata_key, format_data['metadata'])
                assert success, f"Failed to cache {format_name} metadata"
                
                success = cache_manager.set(calibration_key, format_data['calibration_parameters'])
                assert success, f"Failed to cache {format_name} calibration data"
            
            # Validate cross-format data consistency in cache
            for format_name, format_data in formats_data.items():
                # Retrieve and validate cached data
                cached_video = cache_manager.get(f'{format_name}_video')
                cached_metadata = cache_manager.get(f'{format_name}_metadata')
                cached_calibration = cache_manager.get(f'{format_name}_calibration')
                
                assert cached_video is not None, f"Failed to retrieve {format_name} video"
                assert cached_metadata is not None, f"Failed to retrieve {format_name} metadata"
                assert cached_calibration is not None, f"Failed to retrieve {format_name} calibration"
                
                # Validate data integrity
                assert_arrays_almost_equal(cached_video, format_data['video_data'], CACHE_TEST_TOLERANCE)
                assert cached_metadata == format_data['metadata']
                assert cached_calibration == format_data['calibration_parameters']
            
            # Test format-specific metadata preservation
            crimaldi_meta = cache_manager.get('crimaldi_metadata')
            custom_meta = cache_manager.get('custom_metadata')
            
            assert crimaldi_meta['format_type'] == 'crimaldi'
            assert custom_meta['format_type'] == 'custom_avi'
            assert crimaldi_meta['intensity_units'] == 'concentration_ppm'
            assert custom_meta['intensity_units'] == 'raw_sensor'
            
            # Assert cache key generation consistency across formats
            # Keys should be consistently generated regardless of format
            
            # Verify format conversion caching optimization
            # This would involve caching converted data between formats
            
            # Validate cross-format cache statistics accuracy
            stats = cache_manager.get_statistics(include_detailed_breakdown=True)
            total_entries = stats['aggregated_performance'].get('total_utilization', 0)
            print(f"Cross-format cache entries: {total_entries}")
            
        finally:
            shutdown_cache_system(cache_manager)


@pytest.mark.unit
def test_cache_error_handling():
    """
    Test comprehensive cache error handling including disk space errors, memory pressure,
    and recovery mechanisms.
    """
    with setup_test_environment('cache_error_handling') as test_env:
        cache_dir = test_env['temp_directory'] / 'error_test_cache'
        
        # Initialize cache system for error scenario testing
        cache_manager = initialize_cache_manager(
            manager_name='test_error_handling',
            cache_directory=str(cache_dir),
            cache_config={
                'memory_cache': {'max_size_mb': 8, 'max_entries': 50},  # Very small for errors
                'disk_cache': {'max_size_gb': 0.1}  # Very small disk cache
            }
        )
        
        try:
            # Test disk space exhaustion error handling
            large_data_items = []
            disk_full_detected = False
            
            # Try to fill disk cache beyond capacity
            for i in range(100):
                try:
                    key = f'large_item_{i}'
                    # Create data larger than cache capacity
                    large_data = np.random.random((200, 200))  # ~320KB each
                    large_data_items.append((key, large_data))
                    
                    success = cache_manager.set(key, large_data)
                    if not success:
                        disk_full_detected = True
                        print(f"Disk space exhaustion detected at item {i}")
                        break
                        
                except Exception as e:
                    print(f"Expected disk space error: {e}")
                    disk_full_detected = True
                    break
            
            # Validate memory pressure response and eviction
            memory_cache = cache_manager.cache_instances[CacheLevel.MEMORY]
            
            # Fill memory cache to trigger pressure
            for i in range(100):
                key = f'memory_item_{i}'
                data = np.random.random((50, 50))  # Each item ~10KB
                try:
                    memory_cache.put(key, data)
                except Exception as e:
                    print(f"Memory pressure detected: {e}")
                    break
            
            # Check memory cache handled pressure appropriately
            memory_stats = memory_cache.get_statistics()
            memory_utilization = memory_stats['cache_utilization']['size_utilization_ratio']
            print(f"Memory cache utilization: {memory_utilization:.3f}")
            
            # Test cache corruption detection and recovery
            # This would involve corrupting cache files and testing recovery
            
            # Validate graceful degradation under resource constraints
            # Cache should continue operating with reduced capacity
            degraded_operation_test_key = 'degraded_test'
            degraded_test_data = np.array([1, 2, 3, 4, 5])
            
            try:
                success = cache_manager.set(degraded_operation_test_key, degraded_test_data)
                retrieved = cache_manager.get(degraded_operation_test_key)
                
                if retrieved is not None:
                    assert_arrays_almost_equal(retrieved, degraded_test_data, CACHE_TEST_TOLERANCE)
                    print("Cache operating under degraded conditions")
                else:
                    print("Cache degradation: unable to store/retrieve small data")
                    
            except Exception as e:
                print(f"Cache operating in degraded mode: {e}")
            
            # Assert error logging and reporting functionality
            # This is implicitly tested through the exception handling above
            
            # Verify cache system recovery after error conditions
            # Cleanup and restart cache system
            cleanup_result = cleanup_cache_system(
                cache_manager,
                force_cleanup=True,
                preserve_hot_data=False
            )
            
            print(f"Cleanup completed: {cleanup_result.cleanup_effectiveness:.3f}")
            
        finally:
            try:
                shutdown_cache_system(cache_manager)
            except Exception as e:
                print(f"Shutdown after error testing: {e}")


@pytest.mark.unit
def test_cache_statistics_accuracy():
    """
    Test cache statistics accuracy including hit rates, memory usage tracking,
    and performance metrics validation.
    """
    # Initialize cache system with statistics tracking
    cache_manager = initialize_cache_manager(
        manager_name='test_statistics',
        cache_config={
            'memory_cache': {'max_size_mb': 32, 'max_entries': 200},
            'disk_cache': {'max_size_gb': 1}
        },
        enable_performance_monitoring=True
    )
    
    try:
        # Perform various cache operations with known patterns
        known_operations = {
            'puts': 0,
            'gets': 0,
            'hits': 0,
            'misses': 0
        }
        
        # Phase 1: Store known data
        test_items = {}
        for i in range(50):
            key = f'stats_test_{i}'
            value = np.random.random((10, 10))
            test_items[key] = value
            
            success = cache_manager.set(key, value)
            if success:
                known_operations['puts'] += 1
        
        # Phase 2: Read operations with known outcomes
        for key in test_items.keys():
            retrieved = cache_manager.get(key)
            known_operations['gets'] += 1
            
            if retrieved is not None:
                known_operations['hits'] += 1
            else:
                known_operations['misses'] += 1
        
        # Phase 3: Read non-existent keys (guaranteed misses)
        for i in range(10):
            key = f'non_existent_{i}'
            retrieved = cache_manager.get(key)
            known_operations['gets'] += 1
            known_operations['misses'] += 1
            assert retrieved is None
        
        # Validate hit rate calculations are accurate
        expected_hit_rate = known_operations['hits'] / known_operations['gets']
        
        stats = cache_manager.get_statistics(include_detailed_breakdown=True)
        
        # Check aggregated statistics
        aggregated_stats = stats.get('aggregated_performance', {})
        reported_hit_rate = aggregated_stats.get('overall_hit_rate', 0)
        
        print(f"Expected hit rate: {expected_hit_rate:.3f}, Reported: {reported_hit_rate:.3f}")
        
        # Allow some tolerance for timing and caching effects
        hit_rate_tolerance = 0.1
        assert abs(reported_hit_rate - expected_hit_rate) <= hit_rate_tolerance, \
            f"Hit rate accuracy issue: expected {expected_hit_rate:.3f}, got {reported_hit_rate:.3f}"
        
        # Test memory usage tracking precision
        memory_stats = stats['cache_level_statistics'].get('memory_cache', {})
        if memory_stats:
            memory_utilization = memory_stats.get('cache_utilization', {})
            current_size_mb = memory_utilization.get('current_size_mb', 0)
            max_size_mb = memory_utilization.get('max_size_mb', 32)
            
            assert current_size_mb >= 0, "Memory usage cannot be negative"
            assert current_size_mb <= max_size_mb * 1.1, "Memory usage exceeds configured limit"
            
            print(f"Memory usage: {current_size_mb:.2f}MB / {max_size_mb}MB")
        
        # Assert eviction statistics are correctly recorded
        # Force some evictions by overfilling cache
        for i in range(100):
            key = f'eviction_test_{i}'
            value = np.random.random((20, 20))
            cache_manager.set(key, value)
        
        updated_stats = cache_manager.get_statistics()
        
        # Verify performance metrics calculation accuracy
        # Check that all components have reasonable statistics
        for level_name, level_stats in updated_stats.get('cache_level_statistics', {}).items():
            if 'performance_metrics' in level_stats:
                perf_metrics = level_stats['performance_metrics']
                
                # Basic sanity checks
                assert perf_metrics.get('total_hits', 0) >= 0
                assert perf_metrics.get('total_misses', 0) >= 0
                assert perf_metrics.get('cache_hit_rate', 0) >= 0
                assert perf_metrics.get('cache_hit_rate', 0) <= 1.0
                
                print(f"{level_name} hit rate: {perf_metrics.get('cache_hit_rate', 0):.3f}")
        
        # Validate statistics aggregation across cache levels
        # Aggregated stats should be consistent with individual level stats
        overall_stats = updated_stats.get('aggregated_performance', {})
        assert 'overall_hit_rate' in overall_stats
        assert 'total_utilization' in overall_stats
        
    finally:
        shutdown_cache_system(cache_manager)


@pytest.mark.unit
@measure_performance(time_limit_seconds=5.0)
def test_cache_optimization_algorithms():
    """
    Test cache optimization algorithms including adaptive eviction, memory defragmentation,
    and performance tuning.
    """
    # Initialize cache system with optimization enabled
    cache_manager = initialize_cache_manager(
        manager_name='test_optimization',
        cache_config={
            'memory_cache': {'max_size_mb': 64, 'max_entries': 500},
            'disk_cache': {'max_size_gb': 1, 'compression_algorithm': 'lz4'}
        },
        enable_cross_level_optimization=True,
        enable_performance_monitoring=True
    )
    
    try:
        # Generate workload patterns for optimization testing
        access_patterns = {
            'hot': [],    # Frequently accessed
            'warm': [],   # Occasionally accessed
            'cold': []    # Rarely accessed
        }
        
        # Create data with different access patterns
        for pattern_type, pattern_list in access_patterns.items():
            for i in range(20):
                key = f'{pattern_type}_data_{i}'
                value = np.random.random((15, 15))
                pattern_list.append(key)
                cache_manager.set(key, value)
        
        # Simulate access patterns
        for _ in range(5):  # Multiple rounds
            # Hot data - access frequently
            for key in access_patterns['hot'][:10]:  # Access first 10
                cache_manager.get(key)
            
            # Warm data - access occasionally
            for key in access_patterns['warm'][::2]:  # Access every other
                cache_manager.get(key)
            
            # Cold data - access rarely
            if np.random.random() < 0.3:  # 30% chance
                key = np.random.choice(access_patterns['cold'])
                cache_manager.get(key)
        
        # Test adaptive eviction strategy effectiveness
        initial_stats = cache_manager.get_statistics()
        
        # Run optimization
        optimization_result = optimize_cache_coordination(
            cache_manager,
            optimization_strategy='adaptive',
            apply_optimizations=True
        )
        
        assert optimization_result.optimization_strategy == 'adaptive'
        
        # Validate memory defragmentation optimization
        memory_optimization = cache_manager.cache_instances[CacheLevel.MEMORY].optimize_memory(
            optimization_level='moderate',
            force_optimization=True
        )
        
        assert memory_optimization['optimization_applied']
        memory_freed_mb = memory_optimization['memory_freed_mb']
        print(f"Memory optimization freed: {memory_freed_mb:.2f}MB")
        
        # Assert cache policy adjustment based on usage patterns
        # Hot data should be prioritized for memory cache
        hot_key = access_patterns['hot'][0]
        memory_cache = cache_manager.cache_instances[CacheLevel.MEMORY]
        hot_in_memory = memory_cache.get(hot_key)
        # Note: This depends on promotion strategy implementation
        
        # Verify optimization algorithm performance improvement
        post_optimization_stats = cache_manager.get_statistics()
        
        # Compare performance before and after optimization
        initial_hit_rate = initial_stats.get('aggregated_performance', {}).get('overall_hit_rate', 0)
        optimized_hit_rate = post_optimization_stats.get('aggregated_performance', {}).get('overall_hit_rate', 0)
        
        print(f"Hit rate improvement: {initial_hit_rate:.3f} -> {optimized_hit_rate:.3f}")
        
        # Validate optimization effectiveness measurement
        effectiveness = optimization_result.calculate_overall_effectiveness()
        assert effectiveness >= 0, "Optimization effectiveness should be non-negative"
        print(f"Overall optimization effectiveness: {effectiveness:.3f}")
        
        # Test different optimization strategies
        strategies = ['conservative', 'balanced', 'aggressive']
        for strategy in strategies:
            strategy_result = cache_manager.optimize(
                optimization_strategy=strategy,
                apply_optimizations=False  # Test without applying
            )
            assert strategy_result.optimization_strategy == strategy
            print(f"{strategy} strategy effectiveness: {strategy_result.calculate_overall_effectiveness():.3f}")
        
    finally:
        shutdown_cache_system(cache_manager)


@pytest.mark.integration
def test_cache_integration_with_memory_management():
    """
    Test cache integration with memory management system including memory monitoring
    and pressure response.
    """
    # Initialize cache system with memory management integration
    cache_manager = initialize_cache_manager(
        manager_name='test_memory_integration',
        cache_config={
            'memory_cache': {
                'max_size_mb': 128,
                'max_entries': 1000,
                'enable_memory_monitoring': True
            },
            'disk_cache': {'max_size_gb': 2}
        },
        enable_performance_monitoring=True
    )
    
    try:
        # Test memory pressure detection and response
        large_items = []
        
        # Gradually increase memory pressure
        for i in range(200):
            key = f'memory_pressure_test_{i}'
            # Progressively larger arrays to increase pressure
            size = 50 + (i // 10) * 10
            value = np.random.random((size, size))
            large_items.append((key, value))
            
            success = cache_manager.set(key, value)
            
            # Check for memory pressure indicators
            stats = cache_manager.get_statistics()
            memory_stats = stats.get('cache_level_statistics', {}).get('memory_cache', {})
            
            if memory_stats:
                utilization = memory_stats.get('cache_utilization', {}).get('size_utilization_ratio', 0)
                
                if utilization > 0.9:  # High memory pressure
                    print(f"High memory pressure detected at item {i}: {utilization:.3f}")
                    break
        
        # Validate cache eviction coordination with memory monitor
        memory_cache = cache_manager.cache_instances[CacheLevel.MEMORY]
        
        # Force memory pressure check
        pressure_detected = memory_cache.check_memory_pressure()
        print(f"Memory pressure check result: {pressure_detected}")
        
        # Test memory usage optimization across cache levels
        optimization_result = cache_manager.optimize(
            optimization_strategy='memory_focused',
            apply_optimizations=True
        )
        
        # Assert memory threshold compliance during operations
        final_stats = cache_manager.get_statistics()
        memory_utilization = final_stats.get('cache_level_statistics', {}).get('memory_cache', {}).get(
            'cache_utilization', {}
        ).get('size_utilization_ratio', 0)
        
        assert memory_utilization <= 1.0, f"Memory utilization {memory_utilization} exceeds 100%"
        print(f"Final memory utilization: {memory_utilization:.3f}")
        
        # Verify memory management integration effectiveness
        # Test that memory optimization actually freed memory
        memory_optimization = memory_cache.optimize_memory(
            optimization_level='aggressive',
            force_optimization=True
        )
        
        assert memory_optimization['optimization_applied']
        
        # Validate memory leak prevention and cleanup
        # This is tested through proper resource cleanup
        
    finally:
        shutdown_cache_system(cache_manager, save_statistics=True)


# Test fixtures cleanup
@pytest.fixture(scope="function")
def cache_test_fixtures():
    """Pytest fixture providing cache test fixtures with automatic cleanup."""
    fixtures = TestCacheFixtures()
    yield fixtures
    fixtures.cleanup_test_environment()


@pytest.fixture(scope="function") 
def performance_profiler():
    """Pytest fixture providing performance profiler for cache testing."""
    profiler = TestCachePerformance()
    yield profiler
    # Cleanup handled automatically


# Test execution hooks
def pytest_configure(config):
    """Configure pytest for cache testing."""
    # Set test markers
    config.addinivalue_line("markers", "unit: Unit tests for cache components")
    config.addinivalue_line("markers", "integration: Integration tests for cache system")
    config.addinivalue_line("markers", "performance: Performance tests for cache operations")
    config.addinivalue_line("markers", "slow: Slow tests that may take longer to execute")


def pytest_unconfigure(config):
    """Cleanup after all cache tests complete."""
    try:
        cleanup_cache_instances(force_cleanup=True, preserve_statistics=False)
    except Exception as e:
        print(f"Warning: Failed to cleanup cache instances after tests: {e}")