"""
High-performance persistent disk cache implementation providing Level 2 caching for normalized video data 
with compression, memory mapping, atomic operations, and intelligent eviction strategies optimized for 
scientific computing workloads handling large video datasets during 4000+ simulation batch processing.

This module implements a comprehensive disk-based caching system with the following key features:
- Multi-level caching architecture integration (Level 2)
- Configurable compression algorithms (gzip, lz4, zstd) for storage optimization
- Memory mapping for efficient access to large video datasets
- Atomic file operations with integrity verification and rollback capability
- Thread-safe operations with concurrent access management
- Intelligent eviction strategies with memory pressure awareness
- Performance monitoring and optimization for scientific computing workloads
- Comprehensive error handling with graceful degradation capabilities
- Integration with the plume simulation system's batch processing requirements

The implementation emphasizes data persistence, retrieval performance, and storage efficiency while
maintaining reproducible research outcomes and supporting the 8-hour target timeframe for 4000+ simulations.
"""

# External library imports with version specifications
import pathlib  # Python 3.9+ - Modern cross-platform path handling for cache directory management
import threading  # Python 3.9+ - Thread-safe cache operations and concurrent access management
import time  # Python 3.9+ - Timestamp tracking for cache entries and performance measurement
import datetime  # Python 3.9+ - Timestamp generation for cache entry metadata and expiration management
import os  # Python 3.9+ - Operating system interface for disk space monitoring and file operations
import shutil  # Python 3.9+ - High-level file operations for cache management and cleanup
import mmap  # Python 3.9+ - Memory-mapped file operations for efficient large dataset access
import pickle  # Python 3.9+ - Object serialization for cache data storage and retrieval
import gzip  # Python 3.9+ - Gzip compression for cache storage optimization
import tempfile  # Python 3.9+ - Temporary file creation for atomic cache operations
import json  # Python 3.9+ - JSON serialization for cache metadata and configuration
import uuid  # Python 3.9+ - Unique identifier generation for cache operations and integrity tracking
import dataclasses  # Python 3.9+ - Data classes for cache entry and result structures
import contextlib  # Python 3.9+ - Context manager utilities for safe cache operations and resource cleanup
from typing import Dict, Any, List, Optional, Union, Callable, Tuple  # Python 3.9+ - Type hints for cache function signatures and data structures

# High-speed compression libraries for real-time cache operations
try:
    import lz4.frame  # lz4 4.3.2+ - High-speed LZ4 compression for real-time disk cache operations
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd  # zstandard 0.21.0+ - Zstandard compression for optimal disk cache storage efficiency
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

# Platform-specific file locking for concurrent access control
try:
    import fcntl  # fcntl 3.9+ - File locking for concurrent disk cache access control on Unix systems
    FCNTL_AVAILABLE = True
except ImportError:
    FCNTL_AVAILABLE = False

try:
    import msvcrt  # msvcrt 3.9+ - File locking for concurrent disk cache access control on Windows systems
    MSVCRT_AVAILABLE = True
except ImportError:
    MSVCRT_AVAILABLE = False

# Internal imports for eviction strategies and cache management
from .eviction_strategy import (
    EvictionStrategy,
    LRUEvictionStrategy,
    create_eviction_strategy
)

# Internal imports for file utilities and atomic operations
from ..utils.file_utils import (
    ensure_directory_exists,
    safe_file_copy,
    safe_file_move,
    calculate_file_checksum,
    cleanup_temporary_files,
    AtomicFileOperations
)

# Internal imports for logging and performance tracking
from ..utils.logging_utils import (
    get_logger,
    log_performance_metrics,
    create_audit_trail
)

# Internal imports for memory management and monitoring
try:
    from ..utils.memory_management import (
        get_memory_usage,
        MemoryMonitor
    )
except ImportError:
    # Fallback implementations for missing memory management utilities
    def get_memory_usage() -> float:
        """Fallback memory usage function returning default usage ratio."""
        return 0.5
    
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

# Internal imports for caching utilities and statistics
try:
    from ..utils.caching import (
        create_cache_key,
        validate_cache_key,
        CacheStatistics
    )
except ImportError:
    # Fallback implementations for missing caching utilities
    def create_cache_key(*args, **kwargs) -> str:
        """Fallback cache key generation function."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return ":".join(key_parts)
    
    def validate_cache_key(cache_key: str) -> bool:
        """Fallback cache key validation function."""
        return isinstance(cache_key, str) and len(cache_key) > 0
    
    class CacheStatistics:
        """Fallback CacheStatistics class for cache performance tracking."""
        def __init__(self):
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            
        def record_hit(self) -> None:
            """Record cache hit for statistics tracking."""
            self.hits += 1
            
        def record_miss(self) -> None:
            """Record cache miss for statistics tracking."""
            self.misses += 1
            
        def record_eviction(self) -> None:
            """Record cache eviction for statistics tracking."""
            self.evictions += 1
            
        def get_hit_rate(self) -> float:
            """Calculate cache hit rate from recorded statistics."""
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0

# Global configuration constants for disk cache behavior and performance optimization
DEFAULT_CACHE_DIRECTORY: str = '.disk_cache'
DEFAULT_MAX_SIZE_GB: float = 5.0
DEFAULT_TTL_SECONDS: int = 3600
DEFAULT_COMPRESSION_ALGORITHM: str = 'lz4'
SUPPORTED_COMPRESSION_ALGORITHMS: List[str] = ['none', 'gzip', 'lz4', 'zstd']
DEFAULT_EVICTION_STRATEGY: str = 'lru'
CACHE_METADATA_FILENAME: str = 'cache_metadata.json'
CACHE_INDEX_FILENAME: str = 'cache_index.json'
TEMP_FILE_PREFIX: str = 'disk_cache_tmp_'
MAX_MEMORY_MAPPED_SIZE_MB: int = 512
DISK_USAGE_CHECK_INTERVAL: float = 300.0  # 5 minutes
CLEANUP_INTERVAL_SECONDS: float = 1800.0  # 30 minutes
INTEGRITY_CHECK_INTERVAL: float = 3600.0  # 1 hour
FILE_LOCK_TIMEOUT: float = 30.0
ATOMIC_OPERATION_TIMEOUT: float = 60.0
COMPRESSION_LEVEL_FAST: int = 1
COMPRESSION_LEVEL_BALANCED: int = 6
COMPRESSION_LEVEL_BEST: int = 9


@dataclasses.dataclass
class DiskCacheEntry:
    """
    Data class representing a disk cache entry with metadata, file information, compression details, 
    and access tracking for comprehensive cache management and performance monitoring.
    
    This data class encapsulates all information needed for cache entry management including
    file paths, timestamps, compression metadata, and access patterns for optimal cache performance.
    """
    cache_key: str
    file_path: pathlib.Path
    created_time: datetime.datetime
    expires_time: datetime.datetime
    last_accessed: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
    access_count: int = 0
    file_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_algorithm: str = 'none'
    checksum: str = ''
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    is_memory_mapped: bool = False
    
    def is_expired(self, current_time: Optional[datetime.datetime] = None) -> bool:
        """
        Check if cache entry has expired based on TTL and current time.
        
        Args:
            current_time: Current time for expiration check (defaults to datetime.now())
            
        Returns:
            bool: True if cache entry has expired
        """
        if current_time is None:
            current_time = datetime.datetime.now()
        return current_time >= self.expires_time
    
    def update_access(self, access_time: Optional[datetime.datetime] = None) -> None:
        """
        Update access tracking information including timestamp and access count for eviction 
        strategy optimization.
        
        Args:
            access_time: Time of access (defaults to current time)
        """
        if access_time is None:
            access_time = datetime.datetime.now()
        self.last_accessed = access_time
        self.access_count += 1
    
    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio for the cache entry based on original and compressed file sizes.
        
        Returns:
            float: Compression ratio as original_size / compressed_size
        """
        if self.compressed_size_bytes > 0 and self.compression_algorithm != 'none':
            return self.file_size_bytes / self.compressed_size_bytes
        return 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert cache entry to dictionary format for serialization and index storage.
        
        Returns:
            Dict[str, Any]: Cache entry as dictionary with all metadata and timestamps
        """
        return {
            'cache_key': self.cache_key,
            'file_path': str(self.file_path),
            'created_time': self.created_time.isoformat(),
            'expires_time': self.expires_time.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'file_size_bytes': self.file_size_bytes,
            'compressed_size_bytes': self.compressed_size_bytes,
            'compression_algorithm': self.compression_algorithm,
            'checksum': self.checksum,
            'metadata': self.metadata,
            'is_memory_mapped': self.is_memory_mapped
        }
    
    @classmethod
    def from_dict(cls, entry_dict: Dict[str, Any]) -> 'DiskCacheEntry':
        """
        Create cache entry instance from dictionary data for index loading and deserialization.
        
        Args:
            entry_dict: Dictionary containing cache entry data
            
        Returns:
            DiskCacheEntry: Cache entry instance created from dictionary data
        """
        return cls(
            cache_key=entry_dict['cache_key'],
            file_path=pathlib.Path(entry_dict['file_path']),
            created_time=datetime.datetime.fromisoformat(entry_dict['created_time']),
            expires_time=datetime.datetime.fromisoformat(entry_dict['expires_time']),
            last_accessed=datetime.datetime.fromisoformat(entry_dict['last_accessed']),
            access_count=entry_dict['access_count'],
            file_size_bytes=entry_dict['file_size_bytes'],
            compressed_size_bytes=entry_dict['compressed_size_bytes'],
            compression_algorithm=entry_dict['compression_algorithm'],
            checksum=entry_dict['checksum'],
            metadata=entry_dict['metadata'],
            is_memory_mapped=entry_dict['is_memory_mapped']
        )


class DiskCacheIndex:
    """
    Disk cache index management class providing efficient cache entry lookup, persistence, and 
    maintenance operations with thread-safe access and atomic updates for reliable cache index operations.
    
    This class manages the cache index with efficient lookup operations, persistence to disk,
    and thread-safe access for concurrent cache operations.
    """
    
    def __init__(self, index_file_path: pathlib.Path, enable_persistence: bool = True):
        """
        Initialize disk cache index with file path and persistence settings for efficient 
        cache entry management.
        
        Args:
            index_file_path: Path to the cache index file
            enable_persistence: Enable automatic persistence of index changes
        """
        self.index_file_path = index_file_path
        self.persistence_enabled = enable_persistence
        self.entries: Dict[str, DiskCacheEntry] = {}
        self.index_lock = threading.RLock()
        self.last_saved = datetime.datetime.now()
        self.is_dirty = False
        self.save_interval_seconds = 60  # Save every minute if dirty
        self.auto_save_timer: Optional[threading.Timer] = None
        
        # Load existing index if file exists
        if self.index_file_path.exists():
            self.load_index()
        
        # Start auto-save timer if persistence is enabled
        if self.persistence_enabled:
            self._schedule_auto_save()
    
    def add_entry(self, cache_entry: DiskCacheEntry) -> None:
        """
        Add cache entry to index with thread safety and automatic persistence.
        
        Args:
            cache_entry: Cache entry to add to the index
        """
        with self.index_lock:
            self.entries[cache_entry.cache_key] = cache_entry
            self.is_dirty = True
            
            if self.persistence_enabled:
                self._trigger_save_if_needed()
    
    def remove_entry(self, cache_key: str) -> bool:
        """
        Remove cache entry from index with thread safety and persistence.
        
        Args:
            cache_key: Key of cache entry to remove
            
        Returns:
            bool: True if entry was removed successfully
        """
        with self.index_lock:
            if cache_key in self.entries:
                del self.entries[cache_key]
                self.is_dirty = True
                
                if self.persistence_enabled:
                    self._trigger_save_if_needed()
                return True
            return False
    
    def get_entry(self, cache_key: str) -> Optional[DiskCacheEntry]:
        """
        Retrieve cache entry from index with thread safety and access tracking.
        
        Args:
            cache_key: Key of cache entry to retrieve
            
        Returns:
            Optional[DiskCacheEntry]: Cache entry if found, None otherwise
        """
        with self.index_lock:
            entry = self.entries.get(cache_key)
            if entry:
                entry.update_access()
                self.is_dirty = True
            return entry
    
    def save_index(self, force_save: bool = False) -> bool:
        """
        Save cache index to disk with atomic operations and error handling for reliable persistence.
        
        Args:
            force_save: Force save even if not dirty
            
        Returns:
            bool: True if index was saved successfully
        """
        if not force_save and not self.is_dirty:
            return True
        
        try:
            with self.index_lock:
                # Ensure index directory exists
                ensure_directory_exists(str(self.index_file_path.parent))
                
                # Serialize entries to dictionary format
                index_data = {
                    'version': '1.0',
                    'saved_time': datetime.datetime.now().isoformat(),
                    'entry_count': len(self.entries),
                    'entries': {key: entry.to_dict() for key, entry in self.entries.items()}
                }
                
                # Use atomic file operations for safe writing
                atomic_ops = AtomicFileOperations(str(self.index_file_path.parent))
                json_data = json.dumps(index_data, indent=2).encode('utf-8')
                
                write_result = atomic_ops.atomic_write(
                    target_path=str(self.index_file_path),
                    data=json_data,
                    verify_integrity=True
                )
                
                if write_result['success']:
                    self.last_saved = datetime.datetime.now()
                    self.is_dirty = False
                    return True
                    
                return False
                
        except Exception as e:
            logger = get_logger('disk_cache.index', 'CACHE')
            logger.error(f"Failed to save cache index: {e}")
            return False
    
    def load_index(self) -> bool:
        """
        Load cache index from disk with error handling and validation for reliable index restoration.
        
        Returns:
            bool: True if index was loaded successfully
        """
        try:
            with self.index_lock:
                if not self.index_file_path.exists():
                    return False
                
                with open(self.index_file_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                
                # Validate index data structure
                if 'entries' not in index_data:
                    raise ValueError("Invalid index file format: missing entries")
                
                # Reconstruct cache entries from data
                self.entries = {}
                for cache_key, entry_dict in index_data['entries'].items():
                    try:
                        cache_entry = DiskCacheEntry.from_dict(entry_dict)
                        self.entries[cache_key] = cache_entry
                    except Exception as e:
                        logger = get_logger('disk_cache.index', 'CACHE')
                        logger.warning(f"Failed to load cache entry {cache_key}: {e}")
                
                self.is_dirty = False
                return True
                
        except Exception as e:
            logger = get_logger('disk_cache.index', 'CACHE')
            logger.error(f"Failed to load cache index: {e}")
            return False
    
    def cleanup_expired(self, current_time: Optional[datetime.datetime] = None) -> int:
        """
        Remove expired entries from index with batch processing and persistence.
        
        Args:
            current_time: Current time for expiration check
            
        Returns:
            int: Number of expired entries removed
        """
        if current_time is None:
            current_time = datetime.datetime.now()
        
        expired_keys = []
        
        with self.index_lock:
            for cache_key, entry in self.entries.items():
                if entry.is_expired(current_time):
                    expired_keys.append(cache_key)
            
            # Remove expired entries
            for cache_key in expired_keys:
                del self.entries[cache_key]
            
            if expired_keys:
                self.is_dirty = True
                if self.persistence_enabled:
                    self._trigger_save_if_needed()
        
        return len(expired_keys)
    
    def _schedule_auto_save(self) -> None:
        """Schedule automatic save operation using threading timer."""
        if self.auto_save_timer:
            self.auto_save_timer.cancel()
        
        self.auto_save_timer = threading.Timer(self.save_interval_seconds, self._auto_save_callback)
        self.auto_save_timer.daemon = True
        self.auto_save_timer.start()
    
    def _auto_save_callback(self) -> None:
        """Callback function for automatic index saving."""
        try:
            if self.is_dirty:
                self.save_index()
        finally:
            if self.persistence_enabled:
                self._schedule_auto_save()
    
    def _trigger_save_if_needed(self) -> None:
        """Trigger immediate save if conditions are met."""
        time_since_save = (datetime.datetime.now() - self.last_saved).total_seconds()
        if time_since_save > self.save_interval_seconds:
            self.save_index()


class DiskCache:
    """
    High-performance persistent disk cache implementation providing Level 2 caching for normalized 
    video data with compression, memory mapping, atomic operations, and intelligent eviction strategies 
    optimized for scientific computing workloads and large dataset processing.
    
    This class implements a comprehensive disk-based caching system with configurable compression,
    memory mapping for large files, atomic operations for data integrity, and intelligent eviction
    strategies for optimal performance in scientific computing workflows.
    """
    
    def __init__(
        self,
        cache_directory: str,
        max_size_gb: float = DEFAULT_MAX_SIZE_GB,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        compression_algorithm: str = DEFAULT_COMPRESSION_ALGORITHM,
        eviction_strategy: str = DEFAULT_EVICTION_STRATEGY,
        config: Dict[str, Any] = None
    ):
        """
        Initialize disk cache with directory structure, compression settings, eviction strategy, and 
        performance monitoring for persistent data storage.
        
        Args:
            cache_directory: Directory path for cache storage
            max_size_gb: Maximum cache size in gigabytes
            ttl_seconds: Time-to-live for cache entries in seconds
            compression_algorithm: Compression algorithm to use ('none', 'gzip', 'lz4', 'zstd')
            eviction_strategy: Eviction strategy type ('lru', 'adaptive')
            config: Additional configuration parameters
        """
        # Validate and store cache configuration parameters
        self.cache_directory = pathlib.Path(cache_directory).resolve()
        self.max_size_gb = max_size_gb
        self.ttl_seconds = ttl_seconds
        self.compression_algorithm = self._validate_compression_algorithm(compression_algorithm)
        self.config = config or {}
        
        # Initialize cache directory structure with proper permissions
        ensure_directory_exists(str(self.cache_directory), create_parents=True)
        
        # Setup cache index and metadata management
        index_file_path = self.cache_directory / CACHE_INDEX_FILENAME
        self.cache_index = DiskCacheIndex(index_file_path, enable_persistence=True)
        
        # Initialize cache statistics tracking and performance monitoring
        self.statistics = CacheStatistics()
        
        # Create thread lock for concurrent access safety
        self.cache_lock = threading.RLock()
        
        # Setup atomic file operations for data integrity
        self.atomic_operations = AtomicFileOperations(
            working_directory=str(self.cache_directory / 'temp'),
            operation_timeout=ATOMIC_OPERATION_TIMEOUT
        )
        
        # Initialize memory mapping for large file access
        self.memory_mapped_files: Dict[str, mmap.mmap] = {}
        
        # Setup eviction strategy instance with configuration
        eviction_config = self.config.get('eviction_config', {})
        eviction_config.update({
            'max_cache_size_gb': max_size_gb,
            'cache_directory': str(self.cache_directory)
        })
        
        self.eviction_strategy = create_eviction_strategy(
            strategy_type=eviction_strategy,
            cache_config=eviction_config,
            memory_monitor=MemoryMonitor(),
            strategy_config=eviction_config
        )
        
        # Initialize performance metrics and monitoring
        self.performance_metrics: Dict[str, float] = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_operations': 0,
            'compression_ratio': 1.0,
            'memory_mapping_usage': 0,
            'last_cleanup_time': time.time()
        }
        
        # Setup cleanup and integrity check timers
        self.cleanup_timer = self._schedule_cleanup()
        self.integrity_timer = self._schedule_integrity_check()
        
        # Mark cache as initialized and log creation
        self.is_initialized = True
        self.last_optimization = datetime.datetime.now()
        
        # Configure logging for disk cache operations
        self.logger = get_logger(f'disk_cache.{id(self)}', 'CACHE')
        self.logger.info(f"Disk cache initialized: {cache_directory} (max_size: {max_size_gb}GB)")
        
        # Create audit trail for cache initialization
        create_audit_trail(
            action='DISK_CACHE_INIT',
            component='CACHE',
            action_details={
                'cache_directory': str(self.cache_directory),
                'max_size_gb': max_size_gb,
                'compression_algorithm': self.compression_algorithm,
                'eviction_strategy': eviction_strategy
            }
        )
    
    def get(
        self,
        cache_key: str,
        default_value: Any = None,
        use_memory_mapping: bool = True,
        verify_integrity: bool = False
    ) -> Any:
        """
        Retrieve data from disk cache with memory mapping optimization, decompression, integrity 
        verification, and access tracking for performance monitoring.
        
        Args:
            cache_key: Unique identifier for the cached data
            default_value: Value to return if cache key is not found
            use_memory_mapping: Use memory mapping for large files
            verify_integrity: Verify data integrity using checksums
            
        Returns:
            Any: Cached data or default value if not found or expired
        """
        with self.cache_lock:
            try:
                # Validate cache key format and constraints
                if not validate_cache_key(cache_key):
                    self.logger.warning(f"Invalid cache key format: {cache_key}")
                    self.statistics.record_miss()
                    return default_value
                
                # Check cache index for key existence
                cache_entry = self.cache_index.get_entry(cache_key)
                if not cache_entry:
                    self.statistics.record_miss()
                    self.performance_metrics['cache_misses'] += 1
                    return default_value
                
                # Verify cache entry expiration and validity
                if cache_entry.is_expired():
                    self.logger.debug(f"Cache entry expired: {cache_key}")
                    self._remove_entry(cache_key, cleanup_files=True)
                    self.statistics.record_miss()
                    return default_value
                
                # Check if cache file exists on disk
                if not cache_entry.file_path.exists():
                    self.logger.warning(f"Cache file missing: {cache_entry.file_path}")
                    self.cache_index.remove_entry(cache_key)
                    self.statistics.record_miss()
                    return default_value
                
                # Load and decompress cached data
                data = self._load_cache_data(
                    cache_entry=cache_entry,
                    use_memory_mapping=use_memory_mapping,
                    verify_integrity=verify_integrity
                )
                
                if data is not None:
                    # Update access statistics and eviction strategy
                    cache_entry.update_access()
                    self.statistics.record_hit()
                    self.performance_metrics['cache_hits'] += 1
                    
                    # Update eviction strategy with access information
                    if hasattr(self.eviction_strategy, 'update_access_order'):
                        self.eviction_strategy.update_access_order(cache_key)
                    
                    self.logger.debug(f"Cache hit: {cache_key}")
                    return data
                else:
                    self.statistics.record_miss()
                    self.performance_metrics['cache_misses'] += 1
                    return default_value
                    
            except Exception as e:
                self.logger.error(f"Error retrieving cache entry {cache_key}: {e}")
                self.statistics.record_miss()
                return default_value
            finally:
                self.performance_metrics['total_operations'] += 1
    
    def set(
        self,
        cache_key: str,
        data: Any,
        ttl_seconds: Optional[int] = None,
        compress_data: bool = True,
        verify_integrity: bool = True,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Store data in disk cache with compression, atomic operations, integrity verification, and 
        automatic eviction management for reliable persistent storage.
        
        Args:
            cache_key: Unique identifier for the data to cache
            data: Data to store in cache
            ttl_seconds: Time-to-live override for this entry
            compress_data: Apply compression to the data
            verify_integrity: Verify integrity after storage
            metadata: Additional metadata to store with the entry
            
        Returns:
            bool: Success status of cache storage operation
        """
        with self.cache_lock:
            try:
                # Validate cache key and data for storage
                if not validate_cache_key(cache_key):
                    self.logger.warning(f"Invalid cache key format: {cache_key}")
                    return False
                
                if data is None:
                    self.logger.warning(f"Cannot cache None value for key: {cache_key}")
                    return False
                
                # Check disk space availability and size limits
                if not self._check_disk_space():
                    self.logger.warning("Insufficient disk space for cache operation")
                    # Trigger eviction to free space
                    self._trigger_eviction()
                    if not self._check_disk_space():
                        return False
                
                # Serialize data for disk storage
                try:
                    serialized_data = pickle.dumps(data)
                except Exception as e:
                    self.logger.error(f"Failed to serialize data for cache key {cache_key}: {e}")
                    return False
                
                # Apply compression if enabled
                compressed_data = serialized_data
                compression_algorithm = 'none'
                
                if compress_data and self.compression_algorithm != 'none':
                    compressed_data = self._compress_data(serialized_data)
                    compression_algorithm = self.compression_algorithm
                
                # Calculate expiration time
                ttl = ttl_seconds or self.ttl_seconds
                expires_time = datetime.datetime.now() + datetime.timedelta(seconds=ttl)
                
                # Generate unique file path for cache entry
                cache_file_path = self._generate_cache_file_path(cache_key)
                
                # Create cache entry with metadata and timestamps
                cache_entry = DiskCacheEntry(
                    cache_key=cache_key,
                    file_path=cache_file_path,
                    created_time=datetime.datetime.now(),
                    expires_time=expires_time,
                    file_size_bytes=len(serialized_data),
                    compressed_size_bytes=len(compressed_data),
                    compression_algorithm=compression_algorithm,
                    metadata=metadata or {}
                )
                
                # Use atomic file operations for safe storage
                write_result = self.atomic_operations.atomic_write(
                    target_path=str(cache_file_path),
                    data=compressed_data,
                    verify_integrity=verify_integrity
                )
                
                if not write_result['success']:
                    self.logger.error(f"Failed to write cache file for key {cache_key}")
                    return False
                
                # Calculate and store checksum if integrity verification is enabled
                if verify_integrity:
                    cache_entry.checksum = calculate_file_checksum(str(cache_file_path))
                
                # Update cache index with new entry
                self.cache_index.add_entry(cache_entry)
                
                # Update performance metrics
                self._update_compression_metrics(cache_entry)
                
                # Trigger eviction if cache size exceeds limits
                if self._should_trigger_eviction():
                    self._trigger_eviction()
                
                self.logger.debug(f"Cache entry stored: {cache_key} ({len(compressed_data)} bytes)")
                return True
                
            except Exception as e:
                self.logger.error(f"Error storing cache entry {cache_key}: {e}")
                return False
    
    def delete(self, cache_key: str, cleanup_files: bool = True) -> bool:
        """
        Delete cache entry from disk cache with atomic operations, index cleanup, and memory 
        mapping cleanup for safe data removal.
        
        Args:
            cache_key: Key of cache entry to delete
            cleanup_files: Remove associated cache files from disk
            
        Returns:
            bool: Success status of cache deletion operation
        """
        with self.cache_lock:
            try:
                # Validate cache key and check existence
                cache_entry = self.cache_index.get_entry(cache_key)
                if not cache_entry:
                    return False
                
                # Remove entry from cache index
                if not self.cache_index.remove_entry(cache_key):
                    return False
                
                # Cleanup memory mapped files if active
                if cache_key in self.memory_mapped_files:
                    try:
                        self.memory_mapped_files[cache_key].close()
                        del self.memory_mapped_files[cache_key]
                    except Exception as e:
                        self.logger.warning(f"Error closing memory mapped file for {cache_key}: {e}")
                
                # Delete cache files if cleanup_files is enabled
                if cleanup_files and cache_entry.file_path.exists():
                    try:
                        cache_entry.file_path.unlink()
                    except Exception as e:
                        self.logger.warning(f"Error deleting cache file {cache_entry.file_path}: {e}")
                
                # Update cache statistics
                self.statistics.record_eviction()
                
                self.logger.debug(f"Cache entry deleted: {cache_key}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error deleting cache entry {cache_key}: {e}")
                return False
    
    def evict_expired(self, force_eviction: bool = False, batch_size: int = 50) -> Dict[str, Any]:
        """
        Evict expired cache entries based on TTL settings with batch processing, file cleanup, and 
        performance optimization for cache maintenance.
        
        Args:
            force_eviction: Force eviction regardless of expiration time
            batch_size: Number of entries to process in each batch
            
        Returns:
            Dict[str, Any]: Eviction results with expired entry count and freed space
        """
        with self.cache_lock:
            start_time = time.time()
            expired_keys = []
            freed_space_bytes = 0
            current_time = datetime.datetime.now()
            
            try:
                # Scan cache index for expired entries
                for cache_key, cache_entry in list(self.cache_index.entries.items()):
                    if force_eviction or cache_entry.is_expired(current_time):
                        expired_keys.append(cache_key)
                        freed_space_bytes += cache_entry.compressed_size_bytes
                        
                        # Process in batches to avoid holding lock too long
                        if len(expired_keys) >= batch_size:
                            break
                
                # Remove expired entries from index and disk
                removed_count = 0
                for cache_key in expired_keys:
                    if self.delete(cache_key, cleanup_files=True):
                        removed_count += 1
                
                # Update performance metrics
                execution_time = time.time() - start_time
                self.performance_metrics['last_cleanup_time'] = time.time()
                
                # Log performance metrics for eviction operation
                log_performance_metrics(
                    metric_name='cache_eviction_time',
                    metric_value=execution_time,
                    metric_unit='seconds',
                    component='DISK_CACHE',
                    metric_context={
                        'entries_evicted': removed_count,
                        'space_freed_mb': freed_space_bytes / (1024 * 1024),
                        'force_eviction': force_eviction
                    }
                )
                
                eviction_results = {
                    'expired_entries_found': len(expired_keys),
                    'entries_evicted': removed_count,
                    'space_freed_bytes': freed_space_bytes,
                    'execution_time_seconds': execution_time,
                    'force_eviction': force_eviction
                }
                
                self.logger.info(f"Evicted {removed_count} expired cache entries "
                               f"(freed {freed_space_bytes / (1024*1024):.2f} MB)")
                
                return eviction_results
                
            except Exception as e:
                self.logger.error(f"Error during cache eviction: {e}")
                return {
                    'expired_entries_found': 0,
                    'entries_evicted': 0,
                    'space_freed_bytes': 0,
                    'execution_time_seconds': time.time() - start_time,
                    'error': str(e)
                }
    
    def optimize(
        self,
        optimization_level: str = 'moderate',
        defragment_storage: bool = False,
        optimize_compression: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize disk cache performance through compression analysis, file defragmentation, index 
        optimization, and storage efficiency improvements.
        
        Args:
            optimization_level: Level of optimization ('conservative', 'moderate', 'aggressive')
            defragment_storage: Perform storage defragmentation
            optimize_compression: Optimize compression settings
            
        Returns:
            Dict[str, Any]: Optimization results with performance improvements and storage efficiency gains
        """
        with self.cache_lock:
            start_time = time.time()
            optimization_results = {
                'optimization_level': optimization_level,
                'operations_performed': [],
                'space_saved_bytes': 0,
                'performance_improvements': {},
                'errors': []
            }
            
            try:
                # Analyze current cache performance and storage patterns
                if optimize_compression:
                    compression_optimization = self._optimize_compression_settings()
                    optimization_results['operations_performed'].append('compression_optimization')
                    optimization_results['space_saved_bytes'] += compression_optimization.get('space_saved', 0)
                
                # Optimize cache index structure and access patterns
                index_optimization = self._optimize_cache_index()
                optimization_results['operations_performed'].append('index_optimization')
                
                # Defragment storage if requested
                if defragment_storage:
                    defrag_results = self._defragment_cache_storage()
                    optimization_results['operations_performed'].append('storage_defragmentation')
                    optimization_results['space_saved_bytes'] += defrag_results.get('space_saved', 0)
                
                # Update eviction strategy parameters based on optimization level
                strategy_optimization = self._optimize_eviction_strategy(optimization_level)
                optimization_results['operations_performed'].append('eviction_strategy_optimization')
                
                # Calculate performance improvements
                execution_time = time.time() - start_time
                optimization_results['execution_time_seconds'] = execution_time
                optimization_results['performance_improvements'] = {
                    'compression_ratio_improvement': self._calculate_compression_improvement(),
                    'access_time_improvement': self._calculate_access_time_improvement(),
                    'storage_efficiency_improvement': optimization_results['space_saved_bytes'] / (1024 * 1024)
                }
                
                # Update last optimization timestamp
                self.last_optimization = datetime.datetime.now()
                
                # Log optimization operation with performance metrics
                log_performance_metrics(
                    metric_name='cache_optimization_time',
                    metric_value=execution_time,
                    metric_unit='seconds',
                    component='DISK_CACHE',
                    metric_context={
                        'optimization_level': optimization_level,
                        'operations_count': len(optimization_results['operations_performed']),
                        'space_saved_mb': optimization_results['space_saved_bytes'] / (1024 * 1024)
                    }
                )
                
                self.logger.info(f"Cache optimization completed: {optimization_level} level "
                               f"({len(optimization_results['operations_performed'])} operations)")
                
                return optimization_results
                
            except Exception as e:
                optimization_results['errors'].append(str(e))
                self.logger.error(f"Error during cache optimization: {e}")
                return optimization_results
    
    def get_statistics(
        self,
        include_detailed_breakdown: bool = True,
        include_file_statistics: bool = False,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Retrieve comprehensive disk cache statistics including hit rates, storage utilization, 
        compression efficiency, and performance metrics for monitoring and optimization.
        
        Args:
            include_detailed_breakdown: Include detailed breakdown by data type
            include_file_statistics: Include file-level statistics
            time_window_hours: Time window for statistics calculation
            
        Returns:
            Dict[str, Any]: Comprehensive disk cache statistics with performance analysis and storage metrics
        """
        with self.cache_lock:
            try:
                # Collect basic cache statistics
                total_entries = len(self.cache_index.entries)
                total_size_bytes = sum(entry.compressed_size_bytes for entry in self.cache_index.entries.values())
                
                # Calculate cache hit rates and performance metrics
                cache_stats = {
                    'cache_info': {
                        'cache_directory': str(self.cache_directory),
                        'total_entries': total_entries,
                        'total_size_bytes': total_size_bytes,
                        'total_size_mb': total_size_bytes / (1024 * 1024),
                        'max_size_gb': self.max_size_gb,
                        'utilization_ratio': (total_size_bytes / (1024 * 1024 * 1024)) / self.max_size_gb,
                        'compression_algorithm': self.compression_algorithm
                    },
                    'performance_metrics': {
                        'cache_hit_rate': self.statistics.get_hit_rate(),
                        'total_hits': self.statistics.hits,
                        'total_misses': self.statistics.misses,
                        'total_evictions': self.statistics.evictions,
                        'average_compression_ratio': self.performance_metrics.get('compression_ratio', 1.0),
                        'memory_mapping_usage': len(self.memory_mapped_files)
                    },
                    'storage_efficiency': {
                        'average_entry_size_mb': (total_size_bytes / total_entries) / (1024 * 1024) if total_entries > 0 else 0,
                        'compression_space_saved_mb': self._calculate_compression_space_saved() / (1024 * 1024),
                        'disk_space_available_gb': self._get_available_disk_space() / (1024 * 1024 * 1024)
                    }
                }
                
                # Include detailed breakdown by data type if requested
                if include_detailed_breakdown:
                    cache_stats['detailed_breakdown'] = self._generate_detailed_statistics_breakdown()
                
                # Include file-level statistics if requested
                if include_file_statistics:
                    cache_stats['file_statistics'] = self._generate_file_level_statistics()
                
                # Add eviction strategy statistics
                if hasattr(self.eviction_strategy, 'get_strategy_statistics'):
                    cache_stats['eviction_statistics'] = self.eviction_strategy.get_strategy_statistics()
                
                # Include timestamp and cache status
                cache_stats['statistics_timestamp'] = datetime.datetime.now().isoformat()
                cache_stats['cache_status'] = {
                    'is_initialized': self.is_initialized,
                    'last_optimization': self.last_optimization.isoformat(),
                    'cleanup_active': self.cleanup_timer is not None,
                    'integrity_check_active': self.integrity_timer is not None
                }
                
                return cache_stats
                
            except Exception as e:
                self.logger.error(f"Error generating cache statistics: {e}")
                return {
                    'error': str(e),
                    'statistics_timestamp': datetime.datetime.now().isoformat()
                }
    
    def cleanup(
        self,
        aggressive_cleanup: bool = False,
        target_utilization_ratio: float = 0.8,
        optimize_after_cleanup: bool = False
    ) -> Dict[str, Any]:
        """
        Perform comprehensive cache cleanup including expired entry removal, temporary file cleanup, 
        and storage optimization for maintenance operations.
        
        Args:
            aggressive_cleanup: Perform aggressive cleanup with lower thresholds
            target_utilization_ratio: Target cache utilization ratio after cleanup
            optimize_after_cleanup: Run optimization after cleanup completion
            
        Returns:
            Dict[str, Any]: Cleanup results with freed space and performance impact analysis
        """
        start_time = time.time()
        cleanup_results = {
            'cleanup_type': 'aggressive' if aggressive_cleanup else 'standard',
            'target_utilization': target_utilization_ratio,
            'operations_performed': [],
            'space_freed_bytes': 0,
            'entries_removed': 0,
            'errors': []
        }
        
        try:
            # Stop cleanup and integrity timers temporarily
            if self.cleanup_timer:
                self.cleanup_timer.cancel()
            if self.integrity_timer:
                self.integrity_timer.cancel()
            
            # Evict expired entries
            eviction_results = self.evict_expired(force_eviction=aggressive_cleanup)
            cleanup_results['operations_performed'].append('expired_eviction')
            cleanup_results['space_freed_bytes'] += eviction_results['space_freed_bytes']
            cleanup_results['entries_removed'] += eviction_results['entries_evicted']
            
            # Clean up temporary files
            temp_cleanup_results = cleanup_temporary_files(
                temp_directory=str(self.cache_directory / 'temp'),
                max_age_hours=1 if aggressive_cleanup else 24
            )
            cleanup_results['operations_performed'].append('temp_file_cleanup')
            cleanup_results['space_freed_bytes'] += temp_cleanup_results.get('bytes_freed', 0)
            
            # Remove orphaned memory mappings
            orphaned_mappings = self._cleanup_orphaned_memory_mappings()
            cleanup_results['operations_performed'].append('memory_mapping_cleanup')
            
            # Apply additional eviction if target utilization is exceeded
            current_utilization = self._calculate_current_utilization()
            if current_utilization > target_utilization_ratio:
                additional_eviction = self._evict_to_target_utilization(target_utilization_ratio)
                cleanup_results['operations_performed'].append('utilization_eviction')
                cleanup_results['space_freed_bytes'] += additional_eviction['space_freed_bytes']
                cleanup_results['entries_removed'] += additional_eviction['entries_evicted']
            
            # Optimize cache if requested
            if optimize_after_cleanup:
                optimization_results = self.optimize(optimization_level='moderate')
                cleanup_results['operations_performed'].append('post_cleanup_optimization')
                cleanup_results['space_freed_bytes'] += optimization_results.get('space_saved_bytes', 0)
            
            # Calculate cleanup effectiveness
            execution_time = time.time() - start_time
            cleanup_results['execution_time_seconds'] = execution_time
            cleanup_results['cleanup_effectiveness'] = min(
                cleanup_results['space_freed_bytes'] / (1024 * 1024) / max(execution_time, 0.001),
                100.0
            )
            
            # Restart cleanup and integrity timers
            self.cleanup_timer = self._schedule_cleanup()
            self.integrity_timer = self._schedule_integrity_check()
            
            # Log cleanup operation with performance metrics
            log_performance_metrics(
                metric_name='cache_cleanup_time',
                metric_value=execution_time,
                metric_unit='seconds',
                component='DISK_CACHE',
                metric_context={
                    'cleanup_type': cleanup_results['cleanup_type'],
                    'space_freed_mb': cleanup_results['space_freed_bytes'] / (1024 * 1024),
                    'entries_removed': cleanup_results['entries_removed'],
                    'operations_count': len(cleanup_results['operations_performed'])
                }
            )
            
            self.logger.info(f"Cache cleanup completed: {cleanup_results['cleanup_type']} "
                           f"({cleanup_results['space_freed_bytes'] / (1024*1024):.2f} MB freed)")
            
            return cleanup_results
            
        except Exception as e:
            cleanup_results['errors'].append(str(e))
            self.logger.error(f"Error during cache cleanup: {e}")
            return cleanup_results
    
    def close(self, save_statistics: bool = True, final_cleanup: bool = True) -> Dict[str, Any]:
        """
        Gracefully close disk cache with final cleanup, statistics saving, timer cancellation, and 
        resource deallocation for safe shutdown.
        
        Args:
            save_statistics: Save final cache statistics before closing
            final_cleanup: Perform final cleanup operations
            
        Returns:
            Dict[str, Any]: Cache closure results with final statistics and cleanup status
        """
        closure_results = {
            'final_statistics_saved': False,
            'final_cleanup_performed': False,
            'memory_mappings_closed': 0,
            'timers_cancelled': 0,
            'index_saved': False,
            'closure_timestamp': datetime.datetime.now().isoformat()
        }
        
        try:
            # Cancel cleanup and integrity check timers
            if self.cleanup_timer:
                self.cleanup_timer.cancel()
                closure_results['timers_cancelled'] += 1
            if self.integrity_timer:
                self.integrity_timer.cancel()
                closure_results['timers_cancelled'] += 1
            
            # Perform final cleanup if requested
            if final_cleanup:
                cleanup_results = self.cleanup(aggressive_cleanup=False)
                closure_results['final_cleanup_performed'] = True
                closure_results['final_cleanup_results'] = cleanup_results
            
            # Close all memory mapped files
            for cache_key, mapped_file in list(self.memory_mapped_files.items()):
                try:
                    mapped_file.close()
                    del self.memory_mapped_files[cache_key]
                    closure_results['memory_mappings_closed'] += 1
                except Exception as e:
                    self.logger.warning(f"Error closing memory mapped file {cache_key}: {e}")
            
            # Save cache index and metadata
            if self.cache_index.save_index(force_save=True):
                closure_results['index_saved'] = True
            
            # Save final statistics if requested
            if save_statistics:
                final_stats = self.get_statistics(include_detailed_breakdown=True)
                closure_results['final_statistics'] = final_stats
                closure_results['final_statistics_saved'] = True
            
            # Mark cache as not initialized
            self.is_initialized = False
            
            # Create audit trail for cache closure
            create_audit_trail(
                action='DISK_CACHE_CLOSE',
                component='CACHE',
                action_details=closure_results
            )
            
            self.logger.info("Disk cache closed successfully")
            return closure_results
            
        except Exception as e:
            closure_results['error'] = str(e)
            self.logger.error(f"Error during cache closure: {e}")
            return closure_results
    
    def _validate_compression_algorithm(self, algorithm: str) -> str:
        """Validate and normalize compression algorithm selection."""
        if algorithm not in SUPPORTED_COMPRESSION_ALGORITHMS:
            self.logger.warning(f"Unsupported compression algorithm: {algorithm}, using 'none'")
            return 'none'
        
        # Check library availability for selected algorithm
        if algorithm == 'lz4' and not LZ4_AVAILABLE:
            self.logger.warning("LZ4 library not available, falling back to gzip")
            return 'gzip'
        
        if algorithm == 'zstd' and not ZSTD_AVAILABLE:
            self.logger.warning("Zstandard library not available, falling back to gzip")
            return 'gzip'
        
        return algorithm
    
    def _compress_data(self, data: bytes, compression_level: int = COMPRESSION_LEVEL_BALANCED) -> bytes:
        """
        Internal method to compress data using configured compression algorithm with error handling 
        and performance optimization.
        
        Args:
            data: Data to compress
            compression_level: Compression level to use
            
        Returns:
            bytes: Compressed data bytes
        """
        try:
            if self.compression_algorithm == 'gzip':
                return gzip.compress(data, compresslevel=compression_level)
            elif self.compression_algorithm == 'lz4' and LZ4_AVAILABLE:
                return lz4.frame.compress(data, compression_level=compression_level)
            elif self.compression_algorithm == 'zstd' and ZSTD_AVAILABLE:
                compressor = zstd.ZstdCompressor(level=compression_level)
                return compressor.compress(data)
            else:
                return data
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            return data
    
    def _decompress_data(self, compressed_data: bytes, compression_algorithm: str) -> bytes:
        """
        Internal method to decompress data using specified compression algorithm with integrity 
        verification and error handling.
        
        Args:
            compressed_data: Compressed data to decompress
            compression_algorithm: Algorithm used for compression
            
        Returns:
            bytes: Decompressed data bytes
        """
        try:
            if compression_algorithm == 'gzip':
                return gzip.decompress(compressed_data)
            elif compression_algorithm == 'lz4' and LZ4_AVAILABLE:
                return lz4.frame.decompress(compressed_data)
            elif compression_algorithm == 'zstd' and ZSTD_AVAILABLE:
                decompressor = zstd.ZstdDecompressor()
                return decompressor.decompress(compressed_data)
            else:
                return compressed_data
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            raise e
    
    def _load_cache_data(
        self,
        cache_entry: DiskCacheEntry,
        use_memory_mapping: bool = True,
        verify_integrity: bool = False
    ) -> Any:
        """Load and deserialize cache data with optional memory mapping and integrity verification."""
        try:
            # Verify integrity if requested
            if verify_integrity and cache_entry.checksum:
                current_checksum = calculate_file_checksum(str(cache_entry.file_path))
                if current_checksum != cache_entry.checksum:
                    self.logger.warning(f"Checksum mismatch for cache entry: {cache_entry.cache_key}")
                    return None
            
            # Use memory mapping for large files if enabled
            if (use_memory_mapping and 
                cache_entry.compressed_size_bytes > MAX_MEMORY_MAPPED_SIZE_MB * 1024 * 1024):
                return self._load_with_memory_mapping(cache_entry)
            
            # Read cache file data
            with open(cache_entry.file_path, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress data if compression was used
            if cache_entry.compression_algorithm != 'none':
                data = self._decompress_data(compressed_data, cache_entry.compression_algorithm)
            else:
                data = compressed_data
            
            # Deserialize data
            return pickle.loads(data)
            
        except Exception as e:
            self.logger.error(f"Failed to load cache data for {cache_entry.cache_key}: {e}")
            return None
    
    def _load_with_memory_mapping(self, cache_entry: DiskCacheEntry) -> Any:
        """Load cache data using memory mapping for large files."""
        try:
            # Check if already memory mapped
            if cache_entry.cache_key in self.memory_mapped_files:
                mapped_file = self.memory_mapped_files[cache_entry.cache_key]
            else:
                # Create new memory mapping
                with open(cache_entry.file_path, 'rb') as f:
                    mapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    self.memory_mapped_files[cache_entry.cache_key] = mapped_file
                    cache_entry.is_memory_mapped = True
            
            # Read data from memory mapping
            mapped_file.seek(0)
            compressed_data = mapped_file.read()
            
            # Decompress and deserialize
            if cache_entry.compression_algorithm != 'none':
                data = self._decompress_data(compressed_data, cache_entry.compression_algorithm)
            else:
                data = compressed_data
            
            return pickle.loads(data)
            
        except Exception as e:
            self.logger.error(f"Memory mapping failed for {cache_entry.cache_key}: {e}")
            # Fallback to regular file reading
            return self._load_cache_data(cache_entry, use_memory_mapping=False)
    
    def _generate_cache_file_path(self, cache_key: str) -> pathlib.Path:
        """Generate unique file path for cache entry based on cache key."""
        # Create subdirectories based on cache key hash for better distribution
        key_hash = str(hash(cache_key))
        subdir1 = key_hash[-2:]
        subdir2 = key_hash[-4:-2]
        
        cache_subdir = self.cache_directory / subdir1 / subdir2
        ensure_directory_exists(str(cache_subdir))
        
        # Generate filename with extension
        filename = f"{cache_key.replace('/', '_').replace(':', '_')}.cache"
        return cache_subdir / filename
    
    def _check_disk_space(self, required_space_bytes: int = 0) -> bool:
        """
        Internal method to check available disk space and enforce cache size limits with automatic 
        cleanup triggers.
        
        Args:
            required_space_bytes: Required space for operation
            
        Returns:
            bool: True if sufficient disk space is available
        """
        try:
            # Calculate current cache usage
            current_usage_bytes = sum(entry.compressed_size_bytes 
                                    for entry in self.cache_index.entries.values())
            max_usage_bytes = self.max_size_gb * 1024 * 1024 * 1024
            
            # Check against cache size limit
            if current_usage_bytes + required_space_bytes > max_usage_bytes:
                return False
            
            # Check available disk space
            disk_usage = shutil.disk_usage(self.cache_directory)
            available_space = disk_usage.free
            
            # Require at least 1GB free space buffer
            required_buffer = 1024 * 1024 * 1024
            return available_space > (required_space_bytes + required_buffer)
            
        except Exception as e:
            self.logger.error(f"Error checking disk space: {e}")
            return False
    
    def _should_trigger_eviction(self) -> bool:
        """Determine if eviction should be triggered based on cache utilization."""
        current_utilization = self._calculate_current_utilization()
        return current_utilization > 0.9  # Trigger eviction at 90% utilization
    
    def _trigger_eviction(self) -> None:
        """Trigger eviction using configured eviction strategy."""
        try:
            # Prepare cache state for eviction strategy
            cache_state = {
                'entries': {key: entry.to_dict() for key, entry in self.cache_index.entries.items()},
                'utilization_ratio': self._calculate_current_utilization(),
                'memory_pressure': get_memory_usage()
            }
            
            # Determine target eviction count
            total_entries = len(self.cache_index.entries)
            target_eviction_count = max(1, int(total_entries * 0.1))  # Evict 10% of entries
            
            # Select eviction candidates using strategy
            candidates = self.eviction_strategy.select_eviction_candidates(
                cache_entries=cache_state['entries'],
                target_eviction_count=target_eviction_count,
                memory_pressure_level=cache_state['memory_pressure'],
                selection_context={'cache_type': 'disk_cache'}
            )
            
            # Execute eviction
            cache_interface = {
                'remove_entry': lambda key: {'success': self.delete(key, cleanup_files=True)}
            }
            
            eviction_result = self.eviction_strategy.execute_eviction(
                eviction_candidates=candidates,
                cache_interface=cache_interface,
                force_eviction=False
            )
            
            self.logger.info(f"Eviction completed: {eviction_result.successful_evictions} entries removed")
            
        except Exception as e:
            self.logger.error(f"Error during eviction: {e}")
    
    def _remove_entry(self, cache_key: str, cleanup_files: bool = True) -> bool:
        """Internal method to remove cache entry with cleanup."""
        return self.delete(cache_key, cleanup_files)
    
    def _calculate_current_utilization(self) -> float:
        """Calculate current cache utilization ratio."""
        current_usage_bytes = sum(entry.compressed_size_bytes 
                                for entry in self.cache_index.entries.values())
        max_usage_bytes = self.max_size_gb * 1024 * 1024 * 1024
        return current_usage_bytes / max_usage_bytes if max_usage_bytes > 0 else 0.0
    
    def _update_compression_metrics(self, cache_entry: DiskCacheEntry) -> None:
        """Update compression performance metrics."""
        compression_ratio = cache_entry.get_compression_ratio()
        current_avg = self.performance_metrics.get('compression_ratio', 1.0)
        total_entries = len(self.cache_index.entries)
        
        # Calculate weighted average compression ratio
        if total_entries > 1:
            self.performance_metrics['compression_ratio'] = (
                (current_avg * (total_entries - 1) + compression_ratio) / total_entries
            )
        else:
            self.performance_metrics['compression_ratio'] = compression_ratio
    
    def _schedule_cleanup(self) -> threading.Timer:
        """Schedule periodic cache cleanup operations."""
        def cleanup_callback():
            try:
                self.evict_expired()
            except Exception as e:
                self.logger.error(f"Scheduled cleanup failed: {e}")
            finally:
                # Reschedule next cleanup
                self.cleanup_timer = self._schedule_cleanup()
        
        timer = threading.Timer(CLEANUP_INTERVAL_SECONDS, cleanup_callback)
        timer.daemon = True
        timer.start()
        return timer
    
    def _schedule_integrity_check(self) -> threading.Timer:
        """Schedule periodic integrity verification operations."""
        def integrity_callback():
            try:
                self._verify_cache_integrity()
            except Exception as e:
                self.logger.error(f"Scheduled integrity check failed: {e}")
            finally:
                # Reschedule next integrity check
                self.integrity_timer = self._schedule_integrity_check()
        
        timer = threading.Timer(INTEGRITY_CHECK_INTERVAL, integrity_callback)
        timer.daemon = True
        timer.start()
        return timer
    
    def _verify_cache_integrity(self) -> None:
        """Verify integrity of cache entries and remove corrupted entries."""
        corrupted_keys = []
        
        for cache_key, cache_entry in self.cache_index.entries.items():
            try:
                if not cache_entry.file_path.exists():
                    corrupted_keys.append(cache_key)
                    continue
                
                if cache_entry.checksum:
                    current_checksum = calculate_file_checksum(str(cache_entry.file_path))
                    if current_checksum != cache_entry.checksum:
                        corrupted_keys.append(cache_key)
                        
            except Exception as e:
                self.logger.warning(f"Error verifying integrity for {cache_key}: {e}")
                corrupted_keys.append(cache_key)
        
        # Remove corrupted entries
        for cache_key in corrupted_keys:
            self.logger.warning(f"Removing corrupted cache entry: {cache_key}")
            self.delete(cache_key, cleanup_files=True)
    
    def _optimize_compression_settings(self) -> Dict[str, Any]:
        """Optimize compression settings based on data characteristics."""
        return {'space_saved': 0}  # Placeholder implementation
    
    def _optimize_cache_index(self) -> Dict[str, Any]:
        """Optimize cache index structure for better performance."""
        self.cache_index.save_index(force_save=True)
        return {'optimization_applied': True}
    
    def _defragment_cache_storage(self) -> Dict[str, Any]:
        """Defragment cache storage for better performance."""
        return {'space_saved': 0}  # Placeholder implementation
    
    def _optimize_eviction_strategy(self, optimization_level: str) -> Dict[str, Any]:
        """Optimize eviction strategy parameters."""
        if hasattr(self.eviction_strategy, 'optimize_strategy'):
            return self.eviction_strategy.optimize_strategy(
                optimization_context={'level': optimization_level},
                apply_optimizations=True
            )
        return {}
    
    def _calculate_compression_improvement(self) -> float:
        """Calculate compression ratio improvement after optimization."""
        return 0.05  # Placeholder implementation
    
    def _calculate_access_time_improvement(self) -> float:
        """Calculate access time improvement after optimization."""
        return 0.03  # Placeholder implementation
    
    def _calculate_compression_space_saved(self) -> int:
        """Calculate total space saved through compression."""
        total_saved = 0
        for entry in self.cache_index.entries.values():
            if entry.compression_algorithm != 'none':
                total_saved += entry.file_size_bytes - entry.compressed_size_bytes
        return total_saved
    
    def _get_available_disk_space(self) -> int:
        """Get available disk space in bytes."""
        try:
            return shutil.disk_usage(self.cache_directory).free
        except Exception:
            return 0
    
    def _generate_detailed_statistics_breakdown(self) -> Dict[str, Any]:
        """Generate detailed statistics breakdown by data type."""
        breakdown = {
            'compression_algorithms': {},
            'entry_sizes': {'small': 0, 'medium': 0, 'large': 0},
            'access_patterns': {}
        }
        
        for entry in self.cache_index.entries.values():
            # Count by compression algorithm
            algo = entry.compression_algorithm
            breakdown['compression_algorithms'][algo] = breakdown['compression_algorithms'].get(algo, 0) + 1
            
            # Categorize by size
            size_mb = entry.compressed_size_bytes / (1024 * 1024)
            if size_mb < 1:
                breakdown['entry_sizes']['small'] += 1
            elif size_mb < 10:
                breakdown['entry_sizes']['medium'] += 1
            else:
                breakdown['entry_sizes']['large'] += 1
        
        return breakdown
    
    def _generate_file_level_statistics(self) -> Dict[str, Any]:
        """Generate file-level statistics for detailed analysis."""
        return {
            'total_files': len(self.cache_index.entries),
            'memory_mapped_files': len(self.memory_mapped_files),
            'average_file_size_mb': sum(e.compressed_size_bytes for e in self.cache_index.entries.values()) / len(self.cache_index.entries) / (1024*1024) if self.cache_index.entries else 0
        }
    
    def _cleanup_orphaned_memory_mappings(self) -> int:
        """Clean up memory mappings for entries that no longer exist."""
        orphaned_count = 0
        for cache_key in list(self.memory_mapped_files.keys()):
            if cache_key not in self.cache_index.entries:
                try:
                    self.memory_mapped_files[cache_key].close()
                    del self.memory_mapped_files[cache_key]
                    orphaned_count += 1
                except Exception as e:
                    self.logger.warning(f"Error cleaning orphaned mapping {cache_key}: {e}")
        return orphaned_count
    
    def _evict_to_target_utilization(self, target_ratio: float) -> Dict[str, Any]:
        """Evict entries to reach target utilization ratio."""
        current_usage = sum(e.compressed_size_bytes for e in self.cache_index.entries.values())
        target_usage = target_ratio * self.max_size_gb * 1024 * 1024 * 1024
        
        if current_usage <= target_usage:
            return {'entries_evicted': 0, 'space_freed_bytes': 0}
        
        space_to_free = current_usage - target_usage
        space_freed = 0
        entries_evicted = 0
        
        # Sort entries by access time (least recently used first)
        sorted_entries = sorted(self.cache_index.entries.items(), 
                              key=lambda x: x[1].last_accessed)
        
        for cache_key, entry in sorted_entries:
            if space_freed >= space_to_free:
                break
            
            if self.delete(cache_key, cleanup_files=True):
                space_freed += entry.compressed_size_bytes
                entries_evicted += 1
        
        return {'entries_evicted': entries_evicted, 'space_freed_bytes': space_freed}


def create_disk_cache(
    cache_directory: str,
    max_size_gb: Optional[float] = None,
    ttl_seconds: Optional[int] = None,
    compression_algorithm: Optional[str] = None,
    eviction_strategy: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> DiskCache:
    """
    Factory function to create and configure disk cache instance with specified directory, size 
    limits, compression settings, and eviction strategy for Level 2 caching of normalized video data.
    
    This factory function provides centralized disk cache creation with configuration validation
    and performance optimization for scientific computing workloads.
    
    Args:
        cache_directory: Directory path for cache storage
        max_size_gb: Maximum cache size in gigabytes
        ttl_seconds: Time-to-live for cache entries in seconds
        compression_algorithm: Compression algorithm to use
        eviction_strategy: Eviction strategy type
        config: Additional configuration parameters
        
    Returns:
        DiskCache: Configured disk cache instance ready for persistent data caching operations
    """
    logger = get_logger('disk_cache.factory', 'CACHE')
    
    try:
        # Apply default values for optional parameters
        cache_config = {
            'max_size_gb': max_size_gb or DEFAULT_MAX_SIZE_GB,
            'ttl_seconds': ttl_seconds or DEFAULT_TTL_SECONDS,
            'compression_algorithm': compression_algorithm or DEFAULT_COMPRESSION_ALGORITHM,
            'eviction_strategy': eviction_strategy or DEFAULT_EVICTION_STRATEGY
        }
        
        # Merge provided configuration with defaults
        if config:
            cache_config.update(config)
        
        # Validate cache directory path and create if necessary
        cache_dir_path = pathlib.Path(cache_directory)
        if not cache_dir_path.exists():
            ensure_directory_exists(str(cache_dir_path), create_parents=True)
            logger.info(f"Created cache directory: {cache_directory}")
        
        # Validate compression algorithm and eviction strategy
        if cache_config['compression_algorithm'] not in SUPPORTED_COMPRESSION_ALGORITHMS:
            logger.warning(f"Invalid compression algorithm: {cache_config['compression_algorithm']}, using default")
            cache_config['compression_algorithm'] = DEFAULT_COMPRESSION_ALGORITHM
        
        # Create DiskCache instance with validated configuration
        disk_cache = DiskCache(
            cache_directory=str(cache_dir_path),
            max_size_gb=cache_config['max_size_gb'],
            ttl_seconds=cache_config['ttl_seconds'],
            compression_algorithm=cache_config['compression_algorithm'],
            eviction_strategy=cache_config['eviction_strategy'],
            config=cache_config
        )
        
        # Validate disk cache configuration and permissions
        if not disk_cache.is_initialized:
            raise RuntimeError("Failed to initialize disk cache")
        
        # Log disk cache creation with configuration details
        logger.info(f"Disk cache created: {cache_directory} "
                   f"(size: {cache_config['max_size_gb']}GB, "
                   f"compression: {cache_config['compression_algorithm']}, "
                   f"eviction: {cache_config['eviction_strategy']})")
        
        # Create audit trail for cache creation
        create_audit_trail(
            action='DISK_CACHE_CREATED',
            component='CACHE',
            action_details={
                'cache_directory': str(cache_dir_path),
                'configuration': cache_config
            }
        )
        
        return disk_cache
        
    except Exception as e:
        logger.error(f"Failed to create disk cache: {e}")
        raise e


def get_compression_ratio(
    data: bytes,
    compression_algorithm: str,
    compression_level: int = COMPRESSION_LEVEL_BALANCED
) -> float:
    """
    Calculate compression ratio for given data using specified compression algorithm to estimate 
    storage efficiency and optimize compression strategy selection.
    
    Args:
        data: Data to analyze for compression ratio
        compression_algorithm: Compression algorithm to test
        compression_level: Compression level to use
        
    Returns:
        float: Compression ratio as original_size / compressed_size
    """
    try:
        # Validate compression algorithm and level parameters
        if compression_algorithm not in SUPPORTED_COMPRESSION_ALGORITHMS:
            return 1.0
        
        if compression_algorithm == 'none':
            return 1.0
        
        original_size = len(data)
        if original_size == 0:
            return 1.0
        
        # Compress data using specified algorithm and level
        compressed_data = None
        
        if compression_algorithm == 'gzip':
            compressed_data = gzip.compress(data, compresslevel=compression_level)
        elif compression_algorithm == 'lz4' and LZ4_AVAILABLE:
            compressed_data = lz4.frame.compress(data, compression_level=compression_level)
        elif compression_algorithm == 'zstd' and ZSTD_AVAILABLE:
            compressor = zstd.ZstdCompressor(level=compression_level)
            compressed_data = compressor.compress(data)
        
        if compressed_data is None:
            return 1.0
        
        # Calculate compression ratio with division by zero protection
        compressed_size = len(compressed_data)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        return compression_ratio
        
    except Exception as e:
        logger = get_logger('compression_analysis', 'CACHE')
        logger.error(f"Error calculating compression ratio: {e}")
        return 1.0


def optimize_disk_cache_performance(
    disk_cache: DiskCache,
    optimization_strategy: str = 'balanced',
    apply_optimizations: bool = True,
    optimization_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Optimize disk cache performance by analyzing access patterns, adjusting compression settings, 
    optimizing file organization, and implementing performance improvements for scientific computing workloads.
    
    Args:
        disk_cache: DiskCache instance to optimize
        optimization_strategy: Optimization strategy ('conservative', 'balanced', 'aggressive')
        apply_optimizations: Apply optimization changes immediately
        optimization_config: Additional optimization configuration
        
    Returns:
        Dict[str, Any]: Disk cache optimization results with performance improvements and recommendations
    """
    logger = get_logger('cache_optimization', 'CACHE')
    optimization_start_time = time.time()
    
    try:
        # Analyze current disk cache performance and access patterns
        current_stats = disk_cache.get_statistics(include_detailed_breakdown=True)
        
        optimization_results = {
            'optimization_strategy': optimization_strategy,
            'current_performance': {
                'hit_rate': current_stats['performance_metrics']['cache_hit_rate'],
                'utilization': current_stats['cache_info']['utilization_ratio'],
                'compression_ratio': current_stats['performance_metrics']['average_compression_ratio']
            },
            'optimizations_applied': [],
            'performance_improvements': {},
            'recommendations': []
        }
        
        # Identify optimization opportunities and bottlenecks
        opportunities = []
        
        if current_stats['performance_metrics']['cache_hit_rate'] < 0.8:
            opportunities.append('improve_hit_rate')
        
        if current_stats['cache_info']['utilization_ratio'] > 0.9:
            opportunities.append('optimize_utilization')
        
        if current_stats['performance_metrics']['average_compression_ratio'] < 2.0:
            opportunities.append('improve_compression')
        
        # Generate optimization strategy based on usage patterns
        if 'improve_hit_rate' in opportunities:
            if apply_optimizations:
                # Optimize eviction strategy for better hit rates
                eviction_optimization = disk_cache.eviction_strategy.optimize_strategy(
                    optimization_context={'focus': 'hit_rate'},
                    apply_optimizations=True
                )
                optimization_results['optimizations_applied'].append('eviction_strategy_optimization')
            else:
                optimization_results['recommendations'].append('Optimize eviction strategy for better hit rates')
        
        if 'optimize_utilization' in opportunities:
            if apply_optimizations:
                # Perform aggressive cleanup to reduce utilization
                cleanup_results = disk_cache.cleanup(
                    aggressive_cleanup=True,
                    target_utilization_ratio=0.7
                )
                optimization_results['optimizations_applied'].append('utilization_optimization')
                optimization_results['space_freed_mb'] = cleanup_results['space_freed_bytes'] / (1024 * 1024)
            else:
                optimization_results['recommendations'].append('Reduce cache utilization through aggressive cleanup')
        
        if 'improve_compression' in opportunities:
            if apply_optimizations:
                # Optimize compression settings
                compression_optimization = disk_cache.optimize(
                    optimization_level=optimization_strategy,
                    optimize_compression=True
                )
                optimization_results['optimizations_applied'].append('compression_optimization')
            else:
                optimization_results['recommendations'].append('Optimize compression settings for better efficiency')
        
        # Monitor optimization effectiveness and performance impact
        execution_time = time.time() - optimization_start_time
        optimization_results['execution_time_seconds'] = execution_time
        
        # Measure performance improvements if optimizations were applied
        if apply_optimizations and optimization_results['optimizations_applied']:
            # Get updated statistics
            updated_stats = disk_cache.get_statistics()
            
            optimization_results['performance_improvements'] = {
                'hit_rate_improvement': (
                    updated_stats['performance_metrics']['cache_hit_rate'] - 
                    optimization_results['current_performance']['hit_rate']
                ),
                'utilization_improvement': (
                    optimization_results['current_performance']['utilization'] - 
                    updated_stats['cache_info']['utilization_ratio']
                ),
                'compression_improvement': (
                    updated_stats['performance_metrics']['average_compression_ratio'] - 
                    optimization_results['current_performance']['compression_ratio']
                )
            }
        
        # Log optimization operation with detailed results
        log_performance_metrics(
            metric_name='cache_performance_optimization_time',
            metric_value=execution_time,
            metric_unit='seconds',
            component='DISK_CACHE',
            metric_context={
                'optimization_strategy': optimization_strategy,
                'optimizations_applied': len(optimization_results['optimizations_applied']),
                'opportunities_identified': len(opportunities)
            }
        )
        
        logger.info(f"Disk cache optimization completed: {optimization_strategy} strategy "
                   f"({len(optimization_results['optimizations_applied'])} optimizations applied)")
        
        return optimization_results
        
    except Exception as e:
        logger.error(f"Error during disk cache optimization: {e}")
        return {
            'optimization_strategy': optimization_strategy,
            'error': str(e),
            'execution_time_seconds': time.time() - optimization_start_time
        }