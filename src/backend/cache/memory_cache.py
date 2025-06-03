"""
High-performance thread-safe in-memory cache implementation providing Level 1 caching for active 
simulation data in the multi-level caching architecture. Implements LRU eviction strategy, memory 
pressure monitoring, cache entry management with TTL support, and integration with memory management 
system.

This module provides enterprise-grade caching optimized for scientific computing workloads with 4000+ 
simulation batch processing, supporting memory-efficient caching that maintains cache hit rates above 
0.8 threshold while preventing memory exhaustion during large video dataset processing within 8GB 
system limits.

Key Features:
- Thread-safe cache operations with concurrent access support
- LRU eviction strategy with memory pressure awareness  
- TTL-based cache entry expiration and cleanup
- Memory usage monitoring and optimization
- Performance metrics collection and analysis
- Scientific computing context integration
- Batch processing optimization for large workloads
- Cache coordination across multi-level architecture
- Intelligent cache warming and pre-loading
- Comprehensive error handling and recovery
"""

import threading  # Python 3.9+ - Thread-safe cache operations and concurrent access management
import time  # Python 3.9+ - Timestamp tracking for cache entries and TTL management
import datetime  # Python 3.9+ - Timestamp generation and TTL expiration calculations
import collections  # Python 3.9+ - OrderedDict for LRU implementation and efficient cache entry management
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Type  # Python 3.9+ - Type hints for cache function signatures and data structures
import weakref  # Python 3.9+ - Weak references for cache entry tracking without preventing garbage collection
import sys  # Python 3.9+ - System-specific memory information and cache size calculations
import gc  # Python 3.9+ - Garbage collection integration for memory optimization
import copy  # Python 3.9+ - Deep copying for cache data isolation and thread safety
import pickle  # Python 3.9+ - Object serialization for cache entry size calculation
import json  # Python 3.9+ - JSON serialization for cache statistics and configuration
import uuid  # Python 3.9+ - Unique identifier generation for cache instances and operations
import functools  # Python 3.9+ - Decorator utilities for cache operation optimization
import contextlib  # Python 3.9+ - Context manager utilities for cache operation scoping

# Import eviction strategy components for LRU and adaptive eviction management
from .eviction_strategy import (
    LRUEvictionStrategy,
    AdaptiveEvictionStrategy,
    EvictionCandidate,
    EvictionResult,
    create_eviction_strategy,
    EvictionStrategy
)

# Import memory management utilities for cache management and memory pressure detection
try:
    from ..utils.memory_management import (
        MemoryMonitor,
        get_memory_usage,
        memory_pressure_handler
    )
except ImportError:
    # Fallback implementations for missing memory management utilities
    class MemoryMonitor:
        """Fallback MemoryMonitor class with basic functionality."""
        def __init__(self):
            self.current_usage = 0.5
            
        def get_current_usage(self) -> float:
            """Get current memory usage ratio."""
            return self.current_usage
            
        def check_thresholds(self) -> Dict[str, bool]:
            """Check memory thresholds against configured limits."""
            return {
                'warning': self.current_usage > 0.8,
                'critical': self.current_usage > 0.9
            }
            
        def optimize_memory(self) -> Dict[str, Any]:
            """Optimize memory usage through cleanup and optimization."""
            return {'optimization_applied': False, 'memory_freed': 0}
    
    def get_memory_usage() -> float:
        """Fallback memory usage function returning default usage ratio."""
        return 0.5
    
    def memory_pressure_handler(pressure_level: float) -> None:
        """Fallback memory pressure handler for memory management."""
        pass

# Import caching utilities for statistics tracking and cache key generation
try:
    from ..utils.caching import (
        CacheStatistics,
        create_cache_key,
        validate_cache_key
    )
except ImportError:
    # Fallback implementations for missing caching utilities
    class CacheStatistics:
        """Fallback CacheStatistics class for cache performance tracking."""
        def __init__(self):
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.puts = 0
            self.deletes = 0
            self.start_time = datetime.datetime.now()
            
        def record_hit(self) -> None:
            """Record cache hit for statistics tracking."""
            self.hits += 1
            
        def record_miss(self) -> None:
            """Record cache miss for statistics tracking."""
            self.misses += 1
            
        def record_eviction(self) -> None:
            """Record cache eviction for statistics tracking."""
            self.evictions += 1
            
        def record_put(self) -> None:
            """Record cache put operation for statistics tracking."""
            self.puts += 1
            
        def record_delete(self) -> None:
            """Record cache delete operation for statistics tracking."""
            self.deletes += 1
            
        def get_hit_rate(self) -> float:
            """Calculate cache hit rate from recorded statistics."""
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0
            
        def get_performance_summary(self) -> Dict[str, Any]:
            """Get comprehensive performance summary with all metrics."""
            uptime = (datetime.datetime.now() - self.start_time).total_seconds()
            total_operations = self.hits + self.misses + self.puts + self.deletes
            
            return {
                'hit_rate': self.get_hit_rate(),
                'total_hits': self.hits,
                'total_misses': self.misses,
                'total_evictions': self.evictions,
                'total_puts': self.puts,
                'total_deletes': self.deletes,
                'total_operations': total_operations,
                'operations_per_second': total_operations / max(uptime, 1),
                'uptime_seconds': uptime
            }
    
    def create_cache_key(*args, **kwargs) -> str:
        """Fallback cache key generation function."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return ":".join(key_parts)
    
    def validate_cache_key(key: str) -> bool:
        """Fallback cache key validation function."""
        return isinstance(key, str) and len(key) > 0 and len(key) < 1000

# Import logging utilities for cache operations and performance tracking
from ..utils.logging_utils import (
    get_logger,
    log_performance_metrics,
    set_scientific_context,
    log_simulation_event,
    LoggingContext
)

# Import performance thresholds for cache optimization and memory management
try:
    with open('src/backend/config/performance_thresholds.json', 'r') as f:
        performance_thresholds = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    # Fallback performance thresholds if configuration file is not available
    performance_thresholds = {
        'resource_utilization': {
            'memory': {
                'max_usage_gb': 8.0,
                'warning_threshold_gb': 6.4,
                'critical_threshold_gb': 7.5,
                'per_simulation_limit_mb': 1024
            }
        },
        'optimization_targets': {
            'throughput_optimization': {
                'cache_hit_rate_target': 0.8,
                'memory_allocation_efficiency_target': 0.95
            }
        },
        'batch_processing': {
            'chunk_size_per_worker': 10,
            'optimal_batch_size': 1000
        }
    }

# Global configuration constants for memory cache behavior and performance optimization
DEFAULT_CACHE_SIZE_MB: int = 512
DEFAULT_MAX_ENTRIES: int = 10000
DEFAULT_TTL_SECONDS: float = 3600.0
DEFAULT_EVICTION_STRATEGY: str = 'lru'
MEMORY_PRESSURE_THRESHOLD: float = 0.8
CRITICAL_MEMORY_THRESHOLD: float = 0.9
CACHE_HIT_RATE_TARGET: float = 0.8
EVICTION_BATCH_SIZE: int = 10
STATISTICS_UPDATE_INTERVAL: float = 60.0
MEMORY_CHECK_INTERVAL: float = 30.0
TTL_CHECK_INTERVAL: float = 300.0
MAX_ENTRY_SIZE_MB: int = 100
CACHE_WARMING_ENABLED: bool = True
PERFORMANCE_MONITORING_ENABLED: bool = True

# Global registry for cache instances and memory monitoring coordination
_cache_instances: Dict[str, 'MemoryCache'] = {}
_global_memory_monitor: Optional[MemoryMonitor] = None
_cache_registry_lock: threading.RLock = threading.RLock()


class CacheEntry:
    """
    Cache entry container class providing data storage, metadata management, TTL tracking, 
    and access statistics for memory cache entries with thread-safe operations and memory 
    optimization.
    
    This class encapsulates cache entry data with comprehensive metadata tracking, TTL 
    management, and access statistics for intelligent cache management and eviction decisions.
    """
    
    def __init__(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize cache entry with data, TTL configuration, and metadata tracking for 
        comprehensive cache management.
        
        Args:
            key: Unique cache key identifier
            value: Data value to store in cache entry
            ttl_seconds: Time-to-live in seconds (None for no expiration)
            metadata: Additional metadata dictionary for entry context
        """
        # Store cache key and value with deep copy for isolation
        self.key = key
        self.value = copy.deepcopy(value) if not isinstance(value, (str, int, float, bool, type(None))) else value
        
        # Set TTL configuration and calculate expiration time
        self.ttl_seconds = ttl_seconds
        self.expiration_time = None
        if ttl_seconds is not None:
            self.expiration_time = datetime.datetime.now() + datetime.timedelta(seconds=ttl_seconds)
        
        # Initialize metadata dictionary with provided data
        self.metadata = metadata.copy() if metadata else {}
        
        # Record creation timestamp and initial access time
        self.created_time = datetime.datetime.now()
        self.last_accessed = self.created_time
        
        # Initialize access count and calculate entry size
        self.access_count = 0
        self.size_bytes = self._calculate_size()
        
        # Create thread lock for entry-level thread safety
        self.entry_lock = threading.Lock()
        
        # Mark entry as not expired initially
        self._is_expired_cache = False
        self._last_expiry_check = datetime.datetime.now()
    
    def update_access(self) -> None:
        """
        Update access statistics and timestamp for LRU tracking with thread safety.
        
        This method updates access time and frequency counters for LRU eviction 
        strategy and cache performance analysis.
        """
        with self.entry_lock:
            # Update last accessed timestamp
            self.last_accessed = datetime.datetime.now()
            
            # Increment access count
            self.access_count += 1
            
            # Update metadata with access information
            self.metadata['last_access'] = self.last_accessed.isoformat()
            self.metadata['total_accesses'] = self.access_count
            
            # Clear expired cache flag for re-evaluation
            self._is_expired_cache = False
    
    def is_expired(self) -> bool:
        """
        Check if cache entry has expired based on TTL configuration with caching 
        for performance optimization.
        
        Returns:
            bool: True if entry has expired, False otherwise
        """
        # Return cached result if recent check was performed
        current_time = datetime.datetime.now()
        if (current_time - self._last_expiry_check).total_seconds() < 1.0:
            return self._is_expired_cache
        
        # Check if TTL is configured for entry
        if self.ttl_seconds is None or self.expiration_time is None:
            self._is_expired_cache = False
        else:
            # Calculate if entry has expired based on expiration time
            self._is_expired_cache = current_time >= self.expiration_time
        
        # Update last expiry check timestamp
        self._last_expiry_check = current_time
        
        return self._is_expired_cache
    
    def verify_integrity(self) -> bool:
        """
        Verify cache entry data integrity and consistency for scientific computing 
        reliability with comprehensive validation.
        
        Returns:
            bool: True if entry integrity is valid
        """
        try:
            # Validate entry key format and constraints
            if not isinstance(self.key, str) or len(self.key) == 0:
                return False
            
            # Check value data consistency and type
            if self.value is None:
                return True  # None values are valid
            
            # Verify metadata structure and content
            if not isinstance(self.metadata, dict):
                return False
            
            # Validate timestamp consistency
            if self.created_time > datetime.datetime.now():
                return False
            
            if self.last_accessed < self.created_time:
                return False
            
            # Verify access count consistency
            if self.access_count < 0:
                return False
            
            # Check size calculation accuracy
            calculated_size = self._calculate_size()
            if abs(calculated_size - self.size_bytes) > 1024:  # Allow 1KB tolerance
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_size(self) -> int:
        """
        Calculate and return cache entry size in bytes for memory management with 
        comprehensive size accounting.
        
        Returns:
            int: Entry size in bytes including key, value, and metadata
        """
        return self.size_bytes
    
    def _calculate_size(self) -> int:
        """
        Calculate comprehensive size of cache entry including all components.
        
        Returns:
            int: Total size in bytes
        """
        try:
            # Calculate key size using string encoding
            key_size = len(self.key.encode('utf-8'))
            
            # Calculate value size using pickle serialization
            try:
                value_size = len(pickle.dumps(self.value, protocol=pickle.HIGHEST_PROTOCOL))
            except (pickle.PicklingError, TypeError):
                # Fallback size estimation for unpicklable objects
                value_size = sys.getsizeof(self.value)
            
            # Calculate metadata size including timestamps
            metadata_size = sys.getsizeof(self.metadata)
            timestamp_size = sys.getsizeof(self.created_time) + sys.getsizeof(self.last_accessed)
            
            # Account for object overhead and references
            overhead_size = 64  # Estimated object overhead
            
            # Sum all components for total entry size
            total_size = key_size + value_size + metadata_size + timestamp_size + overhead_size
            
            return total_size
            
        except Exception:
            # Fallback size estimation if calculation fails
            return sys.getsizeof(self.value) + len(self.key) * 2 + 128
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert cache entry to dictionary format for serialization and logging with 
        comprehensive metadata inclusion.
        
        Returns:
            Dict[str, Any]: Cache entry as dictionary with all metadata
        """
        return {
            'key': self.key,
            'value_type': type(self.value).__name__,
            'value_size_bytes': self.size_bytes,
            'ttl_seconds': self.ttl_seconds,
            'expiration_time': self.expiration_time.isoformat() if self.expiration_time else None,
            'is_expired': self.is_expired(),
            'created_time': self.created_time.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'metadata': self.metadata.copy(),
            'integrity_valid': self.verify_integrity()
        }


class MemoryCache:
    """
    High-performance thread-safe in-memory cache implementation with LRU eviction, memory 
    pressure monitoring, TTL support, and integration with memory management system for 
    scientific computing workloads with 4000+ simulation batch processing support.
    
    This cache implementation provides Level 1 caching for active simulation data with 
    intelligent eviction strategies, memory optimization, and performance monitoring 
    optimized for scientific computing requirements.
    """
    
    def __init__(
        self,
        cache_name: str,
        max_size_mb: int = DEFAULT_CACHE_SIZE_MB,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        eviction_strategy: str = DEFAULT_EVICTION_STRATEGY,
        memory_monitor: Optional[MemoryMonitor] = None
    ):
        """
        Initialize memory cache with configuration, eviction strategy, memory monitoring, 
        and performance tracking for scientific computing optimization.
        
        Args:
            cache_name: Unique identifier for the cache instance
            max_size_mb: Maximum cache size in megabytes
            max_entries: Maximum number of cache entries
            ttl_seconds: Default TTL for cache entries
            eviction_strategy: Eviction strategy type ('lru', 'adaptive')
            memory_monitor: Optional memory monitor for pressure detection
        """
        # Store cache configuration and size limits
        self.cache_name = cache_name
        self.max_size_mb = max_size_mb
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.eviction_strategy_name = eviction_strategy
        
        # Initialize memory monitor integration if provided
        self.memory_monitor = memory_monitor or _global_memory_monitor
        
        # Initialize cache data dictionary and access order tracking
        self.cache_data: Dict[str, CacheEntry] = {}
        self.access_order: collections.OrderedDict = collections.OrderedDict()
        
        # Setup cache statistics tracking and performance monitoring
        self.statistics = CacheStatistics()
        
        # Create cache lock for thread-safe operations
        self.cache_lock = threading.RLock()
        
        # Initialize eviction strategies (LRU and adaptive)
        self._initialize_eviction_strategies()
        
        # Record creation time and initialize performance metrics
        self.creation_time = datetime.datetime.now()
        self.current_size_bytes = 0
        self.performance_metrics: Dict[str, Any] = {}
        
        # Configure monitoring and cleanup timers
        self.is_monitoring_enabled = PERFORMANCE_MONITORING_ENABLED
        self._initialize_monitoring_timers()
        
        # Setup cache instance logging
        self.logger = get_logger(f'cache.memory.{cache_name}', 'CACHE')
        
        # Log cache initialization with configuration details
        self.logger.info(
            f"Memory cache initialized: {cache_name} "
            f"(max_size: {max_size_mb}MB, max_entries: {max_entries}, "
            f"ttl: {ttl_seconds}s, strategy: {eviction_strategy})"
        )
        
        # Set scientific context for cache operations
        set_scientific_context(
            simulation_id=f'cache_{cache_name}',
            algorithm_name='memory_cache',
            processing_stage='INITIALIZATION'
        )
    
    def _initialize_eviction_strategies(self) -> None:
        """Initialize eviction strategies with cache configuration."""
        try:
            # Create LRU eviction strategy
            self.lru_eviction = LRUEvictionStrategy(
                strategy_name=f'{self.cache_name}_lru',
                config={
                    'batch_size': EVICTION_BATCH_SIZE,
                    'cache_config': {
                        'max_memory_gb': self.max_size_mb / 1024.0,
                        'max_entries': self.max_entries
                    }
                },
                memory_monitor=self.memory_monitor
            )
            
            # Create adaptive eviction strategy
            self.adaptive_eviction = AdaptiveEvictionStrategy(
                strategy_name=f'{self.cache_name}_adaptive',
                config={
                    'initial_strategy': 'lru',
                    'evaluation_interval': 300.0,
                    'auto_switching': True,
                    'lru_config': {
                        'batch_size': EVICTION_BATCH_SIZE,
                        'access_weight': 1.0
                    }
                },
                memory_monitor=self.memory_monitor
            )
            
            # Set current eviction strategy based on configuration
            if self.eviction_strategy_name == 'adaptive':
                self.current_eviction_strategy = self.adaptive_eviction
            else:
                self.current_eviction_strategy = self.lru_eviction
                
        except Exception as e:
            self.logger.error(f"Failed to initialize eviction strategies: {e}")
            # Fallback to simple LRU strategy
            self.current_eviction_strategy = self.lru_eviction
    
    def _initialize_monitoring_timers(self) -> None:
        """Initialize monitoring and cleanup timers for cache management."""
        if not self.is_monitoring_enabled:
            return
        
        try:
            # Setup TTL cleanup timer
            self.ttl_cleanup_timer = threading.Timer(
                TTL_CHECK_INTERVAL,
                self._ttl_cleanup_callback
            )
            self.ttl_cleanup_timer.daemon = True
            self.ttl_cleanup_timer.start()
            
            # Setup memory check timer
            self.memory_check_timer = threading.Timer(
                MEMORY_CHECK_INTERVAL,
                self._memory_check_callback
            )
            self.memory_check_timer.daemon = True
            self.memory_check_timer.start()
            
            # Setup statistics update timer
            self.statistics_timer = threading.Timer(
                STATISTICS_UPDATE_INTERVAL,
                self._statistics_update_callback
            )
            self.statistics_timer.daemon = True
            self.statistics_timer.start()
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize monitoring timers: {e}")
    
    def get(
        self,
        key: str,
        default: Any = None,
        update_access: bool = True
    ) -> Any:
        """
        Retrieve data from cache with access tracking, TTL validation, and performance 
        monitoring for optimal cache hit rates.
        
        This method provides thread-safe cache retrieval with comprehensive access 
        tracking and performance optimization for scientific computing workloads.
        
        Args:
            key: Cache key to retrieve
            default: Default value if key not found or expired
            update_access: Whether to update access statistics
            
        Returns:
            Any: Cached data or default value if not found or expired
        """
        try:
            # Acquire cache lock for thread-safe access
            with self.cache_lock:
                # Validate cache key format and constraints
                if not validate_cache_key(key):
                    self.logger.warning(f"Invalid cache key format: {key}")
                    self.statistics.record_miss()
                    return default
                
                # Check if key exists in cache data
                if key not in self.cache_data:
                    self.statistics.record_miss()
                    self.logger.debug(f"Cache miss: {key}")
                    return default
                
                # Get cache entry and verify it hasn't expired
                entry = self.cache_data[key]
                
                if entry.is_expired():
                    # Remove expired entry from cache
                    self._remove_entry_internal(key)
                    self.statistics.record_miss()
                    self.logger.debug(f"Cache miss (expired): {key}")
                    return default
                
                # Update access statistics and LRU order if requested
                if update_access:
                    entry.update_access()
                    self._update_access_order(key)
                
                # Record cache hit in statistics
                self.statistics.record_hit()
                
                # Log cache hit for performance tracking
                self.logger.debug(f"Cache hit: {key}")
                
                # Return cached data
                return entry.value
                
        except Exception as e:
            self.logger.error(f"Error retrieving cache entry {key}: {e}")
            self.statistics.record_miss()
            return default
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store data in cache with eviction management, memory pressure handling, and 
        performance optimization for scientific computing workloads.
        
        This method provides thread-safe cache storage with intelligent eviction 
        management and memory optimization for large-scale batch processing.
        
        Args:
            key: Cache key for the entry
            value: Data value to store
            ttl_seconds: Time-to-live override (uses default if None)
            metadata: Additional metadata for the entry
            
        Returns:
            bool: Success status of cache storage operation
        """
        try:
            # Acquire cache lock for exclusive access
            with self.cache_lock:
                # Validate cache key and value for storage
                if not validate_cache_key(key):
                    self.logger.warning(f"Invalid cache key format: {key}")
                    return False
                
                # Create cache entry with TTL and metadata
                entry_ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
                cache_entry = CacheEntry(
                    key=key,
                    value=value,
                    ttl_seconds=entry_ttl,
                    metadata=metadata
                )
                
                # Check if entry size exceeds maximum allowed size
                if cache_entry.get_size() > MAX_ENTRY_SIZE_MB * 1024 * 1024:
                    self.logger.warning(
                        f"Entry size ({cache_entry.get_size() / (1024*1024):.2f}MB) "
                        f"exceeds maximum ({MAX_ENTRY_SIZE_MB}MB): {key}"
                    )
                    return False
                
                # Check memory pressure and trigger eviction if needed
                memory_pressure_level = self._get_memory_pressure_level()
                if memory_pressure_level > MEMORY_PRESSURE_THRESHOLD:
                    self._handle_memory_pressure(memory_pressure_level)
                
                # Ensure cache size limits are not exceeded
                self._ensure_cache_capacity(cache_entry.get_size())
                
                # Handle existing entry update or new entry insertion
                is_update = key in self.cache_data
                if is_update:
                    old_entry = self.cache_data[key]
                    self.current_size_bytes -= old_entry.get_size()
                
                # Store entry in cache data and update access order
                self.cache_data[key] = cache_entry
                self.current_size_bytes += cache_entry.get_size()
                self._update_access_order(key)
                
                # Record put operation in statistics
                self.statistics.record_put()
                
                # Log cache put operation
                operation_type = "updated" if is_update else "stored"
                self.logger.debug(
                    f"Cache entry {operation_type}: {key} "
                    f"(size: {cache_entry.get_size()}B, ttl: {entry_ttl}s)"
                )
                
                # Log performance metrics for cache operations
                log_performance_metrics(
                    metric_name='cache_put_operation',
                    metric_value=1.0,
                    metric_unit='count',
                    component='CACHE',
                    metric_context={
                        'cache_name': self.cache_name,
                        'entry_size_bytes': cache_entry.get_size(),
                        'total_entries': len(self.cache_data),
                        'cache_size_mb': self.current_size_bytes / (1024 * 1024),
                        'operation_type': operation_type
                    }
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing cache entry {key}: {e}")
            return False
    
    def evict_lru(
        self,
        count: int = EVICTION_BATCH_SIZE,
        force_eviction: bool = False
    ) -> int:
        """
        Evict least recently used cache entries using configured eviction strategy with 
        memory pressure consideration and performance tracking.
        
        This method provides intelligent cache eviction with performance monitoring 
        and coordination with memory management systems.
        
        Args:
            count: Number of entries to evict
            force_eviction: Force eviction even if not needed
            
        Returns:
            int: Number of entries successfully evicted
        """
        try:
            # Acquire cache lock for exclusive eviction access
            with self.cache_lock:
                # Check if eviction is needed or forced
                if not force_eviction and not self._should_trigger_eviction():
                    return 0
                
                # Prepare cache entries for eviction strategy
                cache_entries_for_eviction = {}
                for key, entry in self.cache_data.items():
                    cache_entries_for_eviction[key] = {
                        'size_bytes': entry.get_size(),
                        'last_access': entry.last_accessed,
                        'access_count': entry.access_count,
                        'created_time': entry.created_time,
                        'is_expired': entry.is_expired()
                    }
                
                # Get memory pressure level for eviction strategy
                memory_pressure_level = self._get_memory_pressure_level()
                
                # Select eviction candidates using current strategy
                eviction_candidates = self.current_eviction_strategy.select_eviction_candidates(
                    cache_entries=cache_entries_for_eviction,
                    target_eviction_count=count,
                    memory_pressure_level=memory_pressure_level,
                    selection_context={
                        'cache_name': self.cache_name,
                        'current_size': self.current_size_bytes,
                        'max_size': self.max_size_mb * 1024 * 1024,
                        'force_eviction': force_eviction
                    }
                )
                
                # Execute eviction for selected candidates
                if eviction_candidates:
                    eviction_result = self._execute_eviction_candidates(eviction_candidates)
                    
                    # Log eviction operation with results
                    self.logger.info(
                        f"Cache eviction completed: {eviction_result.successful_evictions}/{count} "
                        f"entries evicted, {eviction_result.memory_freed_bytes / (1024*1024):.2f}MB freed"
                    )
                    
                    # Update eviction statistics
                    for _ in range(eviction_result.successful_evictions):
                        self.statistics.record_eviction()
                    
                    return eviction_result.successful_evictions
                
                return 0
                
        except Exception as e:
            self.logger.error(f"Error during LRU eviction: {e}")
            return 0
    
    def _execute_eviction_candidates(self, candidates: List[EvictionCandidate]) -> EvictionResult:
        """Execute eviction for selected candidates with error handling."""
        successful_evictions = 0
        memory_freed_bytes = 0
        failed_evictions = []
        execution_start_time = time.time()
        
        try:
            for candidate in candidates:
                try:
                    if candidate.cache_key in self.cache_data:
                        entry = self.cache_data[candidate.cache_key]
                        entry_size = entry.get_size()
                        
                        # Remove entry from cache
                        self._remove_entry_internal(candidate.cache_key)
                        
                        successful_evictions += 1
                        memory_freed_bytes += entry_size
                        
                        self.logger.debug(f"Evicted cache entry: {candidate.cache_key}")
                        
                except Exception as e:
                    failed_evictions.append(candidate.cache_key)
                    self.logger.warning(f"Failed to evict entry {candidate.cache_key}: {e}")
            
            # Calculate execution time
            execution_time = time.time() - execution_start_time
            
            # Create eviction result
            result = EvictionResult(
                candidates_processed=len(candidates),
                successful_evictions=successful_evictions,
                execution_time_seconds=execution_time,
                memory_freed_bytes=memory_freed_bytes,
                failed_evictions=failed_evictions
            )
            
            # Calculate effectiveness score
            result.calculate_effectiveness()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Critical error during eviction execution: {e}")
            return EvictionResult(
                candidates_processed=len(candidates),
                successful_evictions=successful_evictions,
                execution_time_seconds=time.time() - execution_start_time,
                memory_freed_bytes=memory_freed_bytes,
                failed_evictions=failed_evictions,
                error_messages=[str(e)]
            )
    
    def clear(self, preserve_statistics: bool = False) -> int:
        """
        Clear all cache entries and reset cache state with optional statistics 
        preservation for testing and maintenance.
        
        Args:
            preserve_statistics: Whether to preserve cache statistics
            
        Returns:
            int: Number of entries cleared from cache
        """
        try:
            # Acquire cache lock for exclusive clear access
            with self.cache_lock:
                # Count current cache entries for return value
                entries_count = len(self.cache_data)
                
                # Clear cache data dictionary and access order
                self.cache_data.clear()
                self.access_order.clear()
                
                # Reset current size tracking to zero
                self.current_size_bytes = 0
                
                # Preserve or reset cache statistics based on parameter
                if not preserve_statistics:
                    self.statistics = CacheStatistics()
                
                # Log cache clear operation with entry count
                self.logger.info(f"Cache cleared: {entries_count} entries removed")
                
                # Log performance metrics for clear operation
                log_performance_metrics(
                    metric_name='cache_clear_operation',
                    metric_value=entries_count,
                    metric_unit='count',
                    component='CACHE',
                    metric_context={
                        'cache_name': self.cache_name,
                        'preserve_statistics': preserve_statistics
                    }
                )
                
                return entries_count
                
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return 0
    
    def get_statistics(
        self,
        include_detailed_breakdown: bool = True,
        include_performance_trends: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve comprehensive cache statistics including hit rates, memory usage, and 
        performance metrics for monitoring and optimization.
        
        Args:
            include_detailed_breakdown: Include detailed breakdown statistics
            include_performance_trends: Include performance trend analysis
            
        Returns:
            Dict[str, Any]: Comprehensive cache statistics with performance analysis
        """
        try:
            with self.cache_lock:
                # Get base performance summary from statistics
                performance_summary = self.statistics.get_performance_summary()
                
                # Calculate current cache utilization metrics
                cache_utilization = {
                    'current_entries': len(self.cache_data),
                    'max_entries': self.max_entries,
                    'entry_utilization_ratio': len(self.cache_data) / self.max_entries,
                    'current_size_mb': self.current_size_bytes / (1024 * 1024),
                    'max_size_mb': self.max_size_mb,
                    'size_utilization_ratio': self.current_size_bytes / (self.max_size_mb * 1024 * 1024),
                    'average_entry_size_bytes': self.current_size_bytes / max(len(self.cache_data), 1)
                }
                
                # Collect configuration and operational information
                cache_info = {
                    'cache_name': self.cache_name,
                    'creation_time': self.creation_time.isoformat(),
                    'eviction_strategy': self.eviction_strategy_name,
                    'ttl_seconds': self.ttl_seconds,
                    'monitoring_enabled': self.is_monitoring_enabled,
                    'memory_monitor_active': self.memory_monitor is not None
                }
                
                # Combine all statistics
                statistics = {
                    'cache_info': cache_info,
                    'performance_summary': performance_summary,
                    'cache_utilization': cache_utilization,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                # Include detailed breakdown if requested
                if include_detailed_breakdown:
                    statistics['detailed_breakdown'] = self._generate_detailed_breakdown()
                
                # Include performance trends if requested
                if include_performance_trends:
                    statistics['performance_trends'] = self._generate_performance_trends()
                
                # Add memory pressure information if monitor available
                if self.memory_monitor:
                    try:
                        memory_info = {
                            'current_memory_usage': self.memory_monitor.get_current_usage(),
                            'memory_thresholds': self.memory_monitor.check_thresholds(),
                            'memory_pressure_level': self._get_memory_pressure_level()
                        }
                        statistics['memory_info'] = memory_info
                    except Exception as e:
                        statistics['memory_info'] = {'error': str(e)}
                
                return statistics
                
        except Exception as e:
            self.logger.error(f"Error generating cache statistics: {e}")
            return {
                'cache_name': self.cache_name,
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }
    
    def optimize_memory(
        self,
        optimization_level: str = 'moderate',
        force_optimization: bool = False
    ) -> Dict[str, Any]:
        """
        Optimize cache memory usage through intelligent eviction, garbage collection, 
        and memory pressure management for scientific computing performance.
        
        Args:
            optimization_level: Level of optimization ('conservative', 'moderate', 'aggressive')
            force_optimization: Force optimization regardless of current state
            
        Returns:
            Dict[str, Any]: Memory optimization results with freed memory and performance impact
        """
        optimization_start_time = time.time()
        initial_size = self.current_size_bytes
        initial_entries = len(self.cache_data)
        
        try:
            with self.cache_lock:
                # Analyze current memory usage and pressure
                memory_pressure = self._get_memory_pressure_level()
                optimization_needed = force_optimization or memory_pressure > MEMORY_PRESSURE_THRESHOLD
                
                if not optimization_needed:
                    return {
                        'optimization_applied': False,
                        'reason': 'optimization_not_needed',
                        'memory_pressure': memory_pressure
                    }
                
                # Execute memory optimization based on level
                optimization_actions = []
                
                # Cleanup expired entries first
                expired_removed = self.cleanup_expired()
                if expired_removed > 0:
                    optimization_actions.append(f"removed_{expired_removed}_expired_entries")
                
                # Determine eviction count based on optimization level
                eviction_count = 0
                if optimization_level == 'conservative':
                    eviction_count = max(5, int(len(self.cache_data) * 0.1))
                elif optimization_level == 'moderate':
                    eviction_count = max(10, int(len(self.cache_data) * 0.2))
                elif optimization_level == 'aggressive':
                    eviction_count = max(20, int(len(self.cache_data) * 0.4))
                
                # Execute cache eviction if needed
                if eviction_count > 0:
                    evicted_count = self.evict_lru(eviction_count, force_eviction=True)
                    if evicted_count > 0:
                        optimization_actions.append(f"evicted_{evicted_count}_entries")
                
                # Trigger garbage collection if beneficial
                if optimization_level in ['moderate', 'aggressive']:
                    gc.collect()
                    optimization_actions.append("garbage_collection")
                
                # Update memory monitor if available
                if self.memory_monitor:
                    try:
                        monitor_result = self.memory_monitor.optimize_memory()
                        if monitor_result.get('optimization_applied', False):
                            optimization_actions.append("memory_monitor_optimization")
                    except Exception as e:
                        self.logger.warning(f"Memory monitor optimization failed: {e}")
                
                # Calculate optimization results
                final_size = self.current_size_bytes
                final_entries = len(self.cache_data)
                memory_freed = initial_size - final_size
                entries_removed = initial_entries - final_entries
                optimization_time = time.time() - optimization_start_time
                
                # Create optimization results
                optimization_results = {
                    'optimization_applied': True,
                    'optimization_level': optimization_level,
                    'memory_freed_bytes': memory_freed,
                    'memory_freed_mb': memory_freed / (1024 * 1024),
                    'entries_removed': entries_removed,
                    'optimization_time_seconds': optimization_time,
                    'optimization_actions': optimization_actions,
                    'initial_state': {
                        'size_bytes': initial_size,
                        'entries': initial_entries
                    },
                    'final_state': {
                        'size_bytes': final_size,
                        'entries': final_entries
                    },
                    'effectiveness_score': memory_freed / max(initial_size, 1)
                }
                
                # Log optimization results
                self.logger.info(
                    f"Memory optimization completed: {optimization_level} level, "
                    f"{memory_freed / (1024*1024):.2f}MB freed, {entries_removed} entries removed"
                )
                
                # Log performance metrics for optimization
                log_performance_metrics(
                    metric_name='cache_memory_optimization',
                    metric_value=memory_freed / (1024 * 1024),
                    metric_unit='MB',
                    component='CACHE',
                    metric_context={
                        'cache_name': self.cache_name,
                        'optimization_level': optimization_level,
                        'entries_removed': entries_removed,
                        'optimization_time': optimization_time
                    }
                )
                
                return optimization_results
                
        except Exception as e:
            self.logger.error(f"Error during memory optimization: {e}")
            return {
                'optimization_applied': False,
                'error': str(e),
                'optimization_time_seconds': time.time() - optimization_start_time
            }
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries based on TTL configuration with performance tracking 
        and memory optimization.
        
        Returns:
            int: Number of expired entries removed
        """
        try:
            # Acquire cache lock for cleanup operation
            with self.cache_lock:
                expired_keys = []
                
                # Iterate through cache entries checking expiration
                for key, entry in self.cache_data.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                # Remove expired entries from cache data and access order
                for key in expired_keys:
                    self._remove_entry_internal(key)
                
                # Log cleanup operation with removed entry count
                if expired_keys:
                    self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                return len(expired_keys)
                
        except Exception as e:
            self.logger.error(f"Error during expired entry cleanup: {e}")
            return 0
    
    def check_memory_pressure(self) -> bool:
        """
        Check memory pressure and trigger appropriate cache management actions for 
        system stability and performance.
        
        Returns:
            bool: True if memory pressure detected and handled
        """
        try:
            # Get current memory usage from memory monitor
            memory_pressure_level = self._get_memory_pressure_level()
            
            # Check if pressure exceeds thresholds
            pressure_detected = memory_pressure_level > MEMORY_PRESSURE_THRESHOLD
            
            if pressure_detected:
                # Trigger cache eviction if pressure detected
                self._handle_memory_pressure(memory_pressure_level)
                
                # Log memory pressure handling actions
                self.logger.warning(
                    f"Memory pressure detected and handled: level={memory_pressure_level:.3f}, "
                    f"threshold={MEMORY_PRESSURE_THRESHOLD}"
                )
                
                # Coordinate with memory management system
                if self.memory_monitor:
                    try:
                        memory_pressure_handler(memory_pressure_level)
                    except Exception as e:
                        self.logger.warning(f"Memory pressure handler failed: {e}")
            
            return pressure_detected
            
        except Exception as e:
            self.logger.error(f"Error checking memory pressure: {e}")
            return False
    
    def close(self) -> Dict[str, Any]:
        """
        Close cache instance and cleanup resources including timers, monitoring, and 
        final statistics reporting.
        
        Returns:
            Dict[str, Any]: Final cache statistics and cleanup results
        """
        try:
            # Stop all cache monitoring timers
            if hasattr(self, 'ttl_cleanup_timer') and self.ttl_cleanup_timer:
                self.ttl_cleanup_timer.cancel()
            if hasattr(self, 'memory_check_timer') and self.memory_check_timer:
                self.memory_check_timer.cancel()
            if hasattr(self, 'statistics_timer') and self.statistics_timer:
                self.statistics_timer.cancel()
            
            # Generate final cache statistics
            final_statistics = self.get_statistics(
                include_detailed_breakdown=True,
                include_performance_trends=False
            )
            
            # Clear all cache entries and release memory
            entries_cleared = self.clear(preserve_statistics=True)
            
            # Cleanup memory monitor integration
            if self.memory_monitor:
                try:
                    self.memory_monitor.optimize_memory()
                except Exception as e:
                    self.logger.warning(f"Final memory optimization failed: {e}")
            
            # Log cache closure with final statistics
            self.logger.info(
                f"Cache closed: {self.cache_name}, "
                f"final_hit_rate={final_statistics['performance_summary']['hit_rate']:.3f}, "
                f"entries_cleared={entries_cleared}"
            )
            
            # Return comprehensive closure results
            return {
                'cache_name': self.cache_name,
                'entries_cleared': entries_cleared,
                'final_statistics': final_statistics,
                'closure_time': datetime.datetime.now().isoformat(),
                'timers_stopped': True,
                'memory_optimized': True
            }
            
        except Exception as e:
            self.logger.error(f"Error closing cache: {e}")
            return {
                'cache_name': self.cache_name,
                'error': str(e),
                'closure_time': datetime.datetime.now().isoformat()
            }
    
    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU tracking."""
        if key in self.access_order:
            del self.access_order[key]
        self.access_order[key] = datetime.datetime.now()
    
    def _remove_entry_internal(self, key: str) -> bool:
        """Internal method to remove cache entry and update tracking."""
        if key in self.cache_data:
            entry = self.cache_data[key]
            self.current_size_bytes -= entry.get_size()
            del self.cache_data[key]
            
            if key in self.access_order:
                del self.access_order[key]
            
            return True
        return False
    
    def _get_memory_pressure_level(self) -> float:
        """Get current memory pressure level."""
        if self.memory_monitor:
            try:
                return self.memory_monitor.get_current_usage()
            except Exception:
                pass
        
        # Fallback calculation based on cache utilization
        cache_utilization = self.current_size_bytes / (self.max_size_mb * 1024 * 1024)
        return min(cache_utilization * 1.2, 1.0)  # Slightly amplify for pressure detection
    
    def _should_trigger_eviction(self) -> bool:
        """Determine if eviction should be triggered."""
        # Check size limits
        if self.current_size_bytes > self.max_size_mb * 1024 * 1024 * 0.9:
            return True
        
        # Check entry count limits
        if len(self.cache_data) > self.max_entries * 0.9:
            return True
        
        # Check memory pressure
        if self._get_memory_pressure_level() > MEMORY_PRESSURE_THRESHOLD:
            return True
        
        return False
    
    def _ensure_cache_capacity(self, new_entry_size: int) -> None:
        """Ensure cache has capacity for new entry."""
        # Check if adding new entry would exceed size limit
        if self.current_size_bytes + new_entry_size > self.max_size_mb * 1024 * 1024:
            # Calculate how much to evict
            excess_size = (self.current_size_bytes + new_entry_size) - (self.max_size_mb * 1024 * 1024)
            eviction_count = max(1, excess_size // (self.current_size_bytes // max(len(self.cache_data), 1)))
            
            self.evict_lru(int(eviction_count))
        
        # Check if adding new entry would exceed entry limit
        if len(self.cache_data) >= self.max_entries:
            self.evict_lru(1)
    
    def _handle_memory_pressure(self, pressure_level: float) -> None:
        """Handle memory pressure through cache eviction."""
        if pressure_level > CRITICAL_MEMORY_THRESHOLD:
            # Aggressive eviction for critical pressure
            eviction_count = max(10, int(len(self.cache_data) * 0.3))
        elif pressure_level > MEMORY_PRESSURE_THRESHOLD:
            # Moderate eviction for warning pressure
            eviction_count = max(5, int(len(self.cache_data) * 0.1))
        else:
            return
        
        evicted = self.evict_lru(eviction_count, force_eviction=True)
        self.logger.info(f"Memory pressure handling: evicted {evicted} entries")
    
    def _generate_detailed_breakdown(self) -> Dict[str, Any]:
        """Generate detailed breakdown of cache contents."""
        if not self.cache_data:
            return {}
        
        entry_sizes = [entry.get_size() for entry in self.cache_data.values()]
        access_counts = [entry.access_count for entry in self.cache_data.values()]
        
        return {
            'size_distribution': {
                'min_entry_size': min(entry_sizes),
                'max_entry_size': max(entry_sizes),
                'avg_entry_size': sum(entry_sizes) / len(entry_sizes),
                'total_size': sum(entry_sizes)
            },
            'access_distribution': {
                'min_access_count': min(access_counts),
                'max_access_count': max(access_counts),
                'avg_access_count': sum(access_counts) / len(access_counts),
                'total_accesses': sum(access_counts)
            },
            'entry_metadata': {
                'total_entries': len(self.cache_data),
                'expired_entries': sum(1 for entry in self.cache_data.values() if entry.is_expired()),
                'entries_with_ttl': sum(1 for entry in self.cache_data.values() if entry.ttl_seconds is not None)
            }
        }
    
    def _generate_performance_trends(self) -> Dict[str, Any]:
        """Generate performance trend analysis."""
        # Placeholder for performance trend analysis
        # This would typically track metrics over time
        return {
            'hit_rate_trend': 'stable',
            'memory_usage_trend': 'increasing',
            'eviction_frequency_trend': 'stable'
        }
    
    def _ttl_cleanup_callback(self) -> None:
        """Callback for TTL cleanup timer."""
        try:
            self.cleanup_expired()
        except Exception as e:
            self.logger.error(f"TTL cleanup callback failed: {e}")
        finally:
            # Reschedule timer
            if self.is_monitoring_enabled:
                self.ttl_cleanup_timer = threading.Timer(TTL_CHECK_INTERVAL, self._ttl_cleanup_callback)
                self.ttl_cleanup_timer.daemon = True
                self.ttl_cleanup_timer.start()
    
    def _memory_check_callback(self) -> None:
        """Callback for memory check timer."""
        try:
            self.check_memory_pressure()
        except Exception as e:
            self.logger.error(f"Memory check callback failed: {e}")
        finally:
            # Reschedule timer
            if self.is_monitoring_enabled:
                self.memory_check_timer = threading.Timer(MEMORY_CHECK_INTERVAL, self._memory_check_callback)
                self.memory_check_timer.daemon = True
                self.memory_check_timer.start()
    
    def _statistics_update_callback(self) -> None:
        """Callback for statistics update timer."""
        try:
            # Update performance metrics
            current_stats = self.get_statistics(include_detailed_breakdown=False)
            self.performance_metrics.update(current_stats.get('performance_summary', {}))
            
            # Log performance metrics
            log_performance_metrics(
                metric_name='cache_hit_rate',
                metric_value=self.performance_metrics.get('hit_rate', 0.0),
                metric_unit='ratio',
                component='CACHE',
                metric_context={
                    'cache_name': self.cache_name,
                    'total_entries': len(self.cache_data),
                    'cache_size_mb': self.current_size_bytes / (1024 * 1024)
                }
            )
        except Exception as e:
            self.logger.error(f"Statistics update callback failed: {e}")
        finally:
            # Reschedule timer
            if self.is_monitoring_enabled:
                self.statistics_timer = threading.Timer(STATISTICS_UPDATE_INTERVAL, self._statistics_update_callback)
                self.statistics_timer.daemon = True
                self.statistics_timer.start()


def create_cache(
    cache_name: Optional[str] = None,
    max_size_mb: Optional[int] = None,
    max_entries: Optional[int] = None,
    ttl_seconds: Optional[float] = None,
    eviction_strategy: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> MemoryCache:
    """
    Factory function to create and configure memory cache instances with specified parameters, 
    eviction strategy, and memory monitoring integration for optimal scientific computing performance.
    
    This function provides centralized cache creation with configuration loading, validation, 
    and performance monitoring setup for different cache requirements.
    
    Args:
        cache_name: Unique name for the cache instance
        max_size_mb: Maximum cache size in megabytes
        max_entries: Maximum number of cache entries
        ttl_seconds: Default time-to-live for cache entries
        eviction_strategy: Eviction strategy type ('lru', 'adaptive')
        config: Additional configuration dictionary
        
    Returns:
        MemoryCache: Configured memory cache instance ready for active data caching
    """
    try:
        # Set default cache name if not provided
        if cache_name is None:
            cache_name = f"cache_{uuid.uuid4().hex[:8]}"
        
        # Apply default size limits and TTL settings from performance thresholds
        if max_size_mb is None:
            max_size_mb = DEFAULT_CACHE_SIZE_MB
        
        if max_entries is None:
            max_entries = DEFAULT_MAX_ENTRIES
        
        if ttl_seconds is None:
            ttl_seconds = DEFAULT_TTL_SECONDS
        
        if eviction_strategy is None:
            eviction_strategy = DEFAULT_EVICTION_STRATEGY
        
        # Merge provided configuration with defaults
        merged_config = {
            'cache_name': cache_name,
            'max_size_mb': max_size_mb,
            'max_entries': max_entries,
            'ttl_seconds': ttl_seconds,
            'eviction_strategy': eviction_strategy,
            'monitoring_enabled': PERFORMANCE_MONITORING_ENABLED,
            'cache_warming_enabled': CACHE_WARMING_ENABLED
        }
        
        if config:
            merged_config.update(config)
        
        # Setup memory monitoring integration
        memory_monitor = None
        if merged_config.get('enable_memory_monitoring', True):
            try:
                memory_monitor = MemoryMonitor()
                global _global_memory_monitor
                if _global_memory_monitor is None:
                    _global_memory_monitor = memory_monitor
            except Exception as e:
                logger = get_logger('cache.factory', 'CACHE')
                logger.warning(f"Failed to create memory monitor: {e}")
        
        # Create memory cache instance with specified parameters
        cache_instance = MemoryCache(
            cache_name=cache_name,
            max_size_mb=max_size_mb,
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
            eviction_strategy=eviction_strategy,
            memory_monitor=memory_monitor
        )
        
        # Register cache instance in global cache registry
        with _cache_registry_lock:
            _cache_instances[cache_name] = cache_instance
        
        # Initialize performance monitoring and statistics tracking
        logger = get_logger('cache.factory', 'CACHE')
        logger.info(f"Created memory cache: {cache_name} with {eviction_strategy} eviction strategy")
        
        # Set scientific context for cache creation
        set_scientific_context(
            simulation_id=f'cache_creation_{cache_name}',
            algorithm_name='memory_cache_factory',
            processing_stage='CACHE_CREATION'
        )
        
        # Log cache creation metrics
        log_performance_metrics(
            metric_name='cache_creation',
            metric_value=1.0,
            metric_unit='count',
            component='CACHE',
            metric_context={
                'cache_name': cache_name,
                'max_size_mb': max_size_mb,
                'max_entries': max_entries,
                'eviction_strategy': eviction_strategy
            }
        )
        
        return cache_instance
        
    except Exception as e:
        logger = get_logger('cache.factory', 'CACHE')
        logger.error(f"Failed to create memory cache '{cache_name}': {e}")
        raise


def get_cache_instance(
    cache_name: str,
    create_if_missing: bool = True,
    default_config: Optional[Dict[str, Any]] = None
) -> Optional[MemoryCache]:
    """
    Retrieve existing memory cache instance by name or create new instance with default 
    configuration for convenient access to cache functionality.
    
    Args:
        cache_name: Name of the cache instance to retrieve
        create_if_missing: Create new cache if not found
        default_config: Default configuration for new cache creation
        
    Returns:
        Optional[MemoryCache]: Memory cache instance or None if not found and creation disabled
    """
    try:
        # Acquire cache registry lock for thread safety
        with _cache_registry_lock:
            # Check if cache instance exists in registry
            if cache_name in _cache_instances:
                return _cache_instances[cache_name]
            
            # Create new cache instance if missing and creation enabled
            if create_if_missing:
                # Apply default configuration if provided
                config = default_config or {}
                
                cache_instance = create_cache(
                    cache_name=cache_name,
                    max_size_mb=config.get('max_size_mb'),
                    max_entries=config.get('max_entries'),
                    ttl_seconds=config.get('ttl_seconds'),
                    eviction_strategy=config.get('eviction_strategy'),
                    config=config
                )
                
                # Cache instance is already registered by create_cache
                return cache_instance
            
            return None
            
    except Exception as e:
        logger = get_logger('cache.registry', 'CACHE')
        logger.error(f"Error retrieving cache instance '{cache_name}': {e}")
        return None


def cleanup_cache_instances(
    force_cleanup: bool = False,
    preserve_statistics: bool = False
) -> Dict[str, Any]:
    """
    Cleanup all memory cache instances and release resources for system shutdown or 
    testing scenarios with comprehensive resource deallocation.
    
    Args:
        force_cleanup: Force cleanup even if caches are active
        preserve_statistics: Preserve cache statistics during cleanup
        
    Returns:
        Dict[str, Any]: Cleanup results with freed memory and final statistics
    """
    cleanup_start_time = time.time()
    cleanup_results = {
        'caches_cleaned': 0,
        'total_memory_freed_mb': 0.0,
        'final_statistics': {},
        'cleanup_time_seconds': 0.0,
        'errors': []
    }
    
    try:
        # Acquire cache registry lock for exclusive access
        with _cache_registry_lock:
            logger = get_logger('cache.cleanup', 'CACHE')
            logger.info(f"Starting cleanup of {len(_cache_instances)} cache instances")
            
            # Stop all cache monitoring and optimization timers
            for cache_name, cache_instance in _cache_instances.items():
                try:
                    # Close cache instance and collect statistics
                    cache_closure_result = cache_instance.close()
                    
                    # Accumulate cleanup statistics
                    cleanup_results['final_statistics'][cache_name] = cache_closure_result
                    cleanup_results['caches_cleaned'] += 1
                    
                    # Calculate memory freed
                    if 'final_statistics' in cache_closure_result:
                        cache_size_mb = cache_closure_result['final_statistics'].get(
                            'cache_utilization', {}
                        ).get('current_size_mb', 0.0)
                        cleanup_results['total_memory_freed_mb'] += cache_size_mb
                    
                    logger.debug(f"Cleaned up cache: {cache_name}")
                    
                except Exception as e:
                    error_msg = f"Failed to cleanup cache {cache_name}: {e}"
                    cleanup_results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            # Clear cache registry and release locks
            _cache_instances.clear()
            
            # Cleanup global memory monitor integration
            global _global_memory_monitor
            if _global_memory_monitor:
                try:
                    _global_memory_monitor.optimize_memory()
                    _global_memory_monitor = None
                except Exception as e:
                    cleanup_results['errors'].append(f"Failed to cleanup memory monitor: {e}")
            
            # Calculate total cleanup time
            cleanup_results['cleanup_time_seconds'] = time.time() - cleanup_start_time
            
            # Log cache cleanup operation with results
            logger.info(
                f"Cache cleanup completed: {cleanup_results['caches_cleaned']} caches cleaned, "
                f"{cleanup_results['total_memory_freed_mb']:.2f}MB freed, "
                f"{cleanup_results['cleanup_time_seconds']:.3f}s"
            )
            
            # Log cleanup performance metrics
            log_performance_metrics(
                metric_name='cache_cleanup_operation',
                metric_value=cleanup_results['caches_cleaned'],
                metric_unit='count',
                component='CACHE',
                metric_context={
                    'memory_freed_mb': cleanup_results['total_memory_freed_mb'],
                    'cleanup_time': cleanup_results['cleanup_time_seconds'],
                    'errors_count': len(cleanup_results['errors'])
                }
            )
            
            return cleanup_results
            
    except Exception as e:
        cleanup_results['cleanup_time_seconds'] = time.time() - cleanup_start_time
        cleanup_results['errors'].append(f"Critical cleanup error: {e}")
        
        logger = get_logger('cache.cleanup', 'CACHE')
        logger.error(f"Critical error during cache cleanup: {e}")
        
        return cleanup_results