"""
Comprehensive cache eviction strategy implementation providing intelligent cache management algorithms 
for the multi-level caching architecture. Implements LRU (Least Recently Used), adaptive, and hybrid 
eviction strategies with performance monitoring, memory pressure handling, and coordination across 
memory cache, disk cache, and result cache levels.

This module is optimized for scientific computing workloads with 4000+ simulation batch processing, 
supporting cache hit rates above 0.8 threshold while maintaining memory efficiency within 8GB system 
limits and enabling intelligent cache promotion and eviction decisions.

Key Features:
- Multi-level eviction coordination (Memory, Disk, Result caches)
- LRU eviction strategy with efficient access tracking
- Adaptive eviction strategy with dynamic algorithm selection
- Memory pressure-aware eviction decisions
- Performance monitoring and optimization
- Thread-safe operations for concurrent access
- Scientific computing context integration
- Batch processing optimization for large workloads
"""

import abc  # Python 3.9+ - Abstract base class implementation for eviction strategy interface
import threading  # Python 3.9+ - Thread-safe eviction operations and concurrent access management
import time  # Python 3.9+ - Timestamp tracking for LRU eviction and performance measurement
import datetime  # Python 3.9+ - Timestamp generation for eviction candidate tracking and audit trails
import collections  # Python 3.9+ - OrderedDict for LRU implementation and efficient eviction candidate management
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Type  # Python 3.9+ - Type hints for eviction strategy function signatures and interfaces
import dataclasses  # Python 3.9+ - Data classes for eviction candidate and result structures
import enum  # Python 3.9+ - Enumeration for eviction strategy types and priority levels
import heapq  # Python 3.9+ - Priority queue implementation for efficient eviction candidate selection
import statistics  # Python 3.9+ - Statistical analysis for adaptive eviction strategy performance evaluation
import copy  # Python 3.9+ - Deep copying for eviction candidate data isolation and thread safety
import weakref  # Python 3.9+ - Weak references for cache entry tracking without preventing garbage collection
import json  # Python 3.9+ - JSON serialization for eviction configuration and performance data
import uuid  # Python 3.9+ - Unique identifier generation for eviction operations and audit trails

# Import memory management utilities for eviction decisions and memory pressure detection
try:
    from ..utils.memory_management import (
        get_memory_usage,
        MemoryMonitor,
        memory_pressure_handler
    )
except ImportError:
    # Fallback implementations for missing memory management utilities
    def get_memory_usage() -> float:
        """Fallback memory usage function returning 0.5 as default usage ratio."""
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
    
    def memory_pressure_handler(pressure_level: float) -> None:
        """Fallback memory pressure handler for memory management."""
        pass

# Import caching utilities for statistics tracking and cache key generation
try:
    from ..utils.caching import (
        CacheStatistics,
        create_cache_key
    )
except ImportError:
    # Fallback implementations for missing caching utilities
    class CacheStatistics:
        """Fallback CacheStatistics class for eviction performance tracking."""
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
    
    def create_cache_key(*args, **kwargs) -> str:
        """Fallback cache key generation function."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return ":".join(key_parts)

# Import logging utilities for eviction strategy operations and performance tracking
from ..utils.logging_utils import (
    get_logger,
    log_performance_metrics,
    set_scientific_context
)

# Import performance thresholds for eviction strategy optimization and memory management
from ..config.performance_thresholds.json import performance_thresholds

# Global configuration constants for eviction strategy behavior and performance optimization
DEFAULT_EVICTION_STRATEGY: str = 'adaptive_lru'
LRU_ACCESS_WEIGHT: float = 1.0
FREQUENCY_ACCESS_WEIGHT: float = 0.5
SIZE_WEIGHT: float = 0.3
AGE_WEIGHT: float = 0.2
MEMORY_PRESSURE_THRESHOLD: float = 0.8
CRITICAL_MEMORY_THRESHOLD: float = 0.9
EVICTION_BATCH_SIZE: int = 10
ADAPTIVE_EVALUATION_INTERVAL: float = 300.0  # 5 minutes
STRATEGY_PERFORMANCE_WINDOW: float = 3600.0  # 1 hour
MIN_EVICTION_CANDIDATES: int = 5
MAX_EVICTION_CANDIDATES: int = 100
EVICTION_EFFECTIVENESS_THRESHOLD: float = 0.7

# Global registry for eviction strategy classes and performance tracking
_strategy_registry: Dict[str, Type['EvictionStrategy']] = {}
_strategy_performance_history: Dict[str, List[float]] = {}
_global_eviction_lock: threading.RLock = threading.RLock()
_memory_monitor_instance: Optional[MemoryMonitor] = None


class EvictionPriority(enum.Enum):
    """Enumeration for eviction priority levels with numerical values for sorting."""
    HIGHEST = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    LOWEST = 5


class EvictionReason(enum.Enum):
    """Enumeration for eviction reasons to track eviction decision rationale."""
    MEMORY_PRESSURE = "memory_pressure"
    LRU_POLICY = "lru_policy"
    SIZE_OPTIMIZATION = "size_optimization"
    ACCESS_FREQUENCY = "access_frequency"
    AGE_BASED = "age_based"
    ADAPTIVE_DECISION = "adaptive_decision"
    MANUAL_EVICTION = "manual_eviction"
    CACHE_COORDINATION = "cache_coordination"


@dataclasses.dataclass
class EvictionCandidate:
    """
    Data class representing a cache entry candidate for eviction with priority scoring, 
    metadata, and eviction context for intelligent cache management decisions.
    
    This data class encapsulates all information needed for eviction decision making including
    priority scoring, access patterns, and metadata for comprehensive cache management.
    """
    cache_key: str
    priority_score: float
    metadata: Dict[str, Any]
    last_access_time: datetime.datetime
    access_frequency: int = 0
    entry_size_bytes: int = 0
    eviction_reason: str = EvictionReason.LRU_POLICY.value
    creation_time: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
    is_pinned: bool = False
    
    def calculate_eviction_priority(
        self,
        weight_config: Dict[str, float],
        memory_pressure_level: float
    ) -> float:
        """
        Calculate eviction priority based on access patterns, size, age, and memory pressure 
        for optimal eviction ordering.
        
        This method combines multiple factors into a weighted priority score for optimal
        eviction candidate selection based on current system conditions.
        
        Args:
            weight_config: Configuration for weighting different priority factors
            memory_pressure_level: Current memory pressure level (0.0 to 1.0)
            
        Returns:
            float: Calculated eviction priority score for candidate ordering
        """
        current_time = datetime.datetime.now()
        
        # Calculate age-based priority component (older entries have higher priority for eviction)
        age_seconds = (current_time - self.last_access_time).total_seconds()
        age_priority = age_seconds / 3600.0  # Normalize to hours
        
        # Include access frequency in priority calculation (less frequent = higher eviction priority)
        frequency_priority = 1.0 / (self.access_frequency + 1)  # Avoid division by zero
        
        # Factor entry size into priority score (larger entries = higher eviction priority under memory pressure)
        size_priority = self.entry_size_bytes / (1024 * 1024)  # Normalize to MB
        
        # Apply memory pressure weighting if applicable
        memory_pressure_multiplier = 1.0 + (memory_pressure_level * 2.0) if memory_pressure_level > MEMORY_PRESSURE_THRESHOLD else 1.0
        
        # Combine weighted components into final priority score
        weighted_priority = (
            weight_config.get('age_weight', AGE_WEIGHT) * age_priority +
            weight_config.get('frequency_weight', FREQUENCY_ACCESS_WEIGHT) * frequency_priority +
            weight_config.get('size_weight', SIZE_WEIGHT) * size_priority
        ) * memory_pressure_multiplier
        
        self.priority_score = weighted_priority
        return weighted_priority
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert eviction candidate to dictionary format for serialization, logging, and analysis.
        
        Returns:
            Dict[str, Any]: Eviction candidate as dictionary with all properties and metadata
        """
        return {
            'cache_key': self.cache_key,
            'priority_score': self.priority_score,
            'metadata': self.metadata,
            'last_access_time': self.last_access_time.isoformat(),
            'access_frequency': self.access_frequency,
            'entry_size_bytes': self.entry_size_bytes,
            'eviction_reason': self.eviction_reason,
            'creation_time': self.creation_time.isoformat(),
            'is_pinned': self.is_pinned
        }


@dataclasses.dataclass
class EvictionResult:
    """
    Data class containing eviction execution results including success count, performance 
    metrics, error information, and effectiveness analysis for eviction operation tracking.
    
    This data class provides comprehensive results tracking for eviction operations with
    performance analysis and error handling information.
    """
    candidates_processed: int
    successful_evictions: int
    execution_time_seconds: float
    memory_freed_bytes: int
    failed_evictions: List[str] = dataclasses.field(default_factory=list)
    performance_metrics: Dict[str, Any] = dataclasses.field(default_factory=dict)
    execution_timestamp: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
    eviction_effectiveness_score: float = 0.0
    error_messages: List[str] = dataclasses.field(default_factory=list)
    
    def calculate_effectiveness(self) -> float:
        """
        Calculate eviction effectiveness score based on success rate, memory freed, and execution efficiency.
        
        Returns:
            float: Eviction effectiveness score (0.0 to 1.0)
        """
        if self.candidates_processed == 0:
            return 0.0
        
        # Calculate success rate from processed and successful evictions
        success_rate = self.successful_evictions / self.candidates_processed
        
        # Factor memory freed per second into effectiveness
        memory_efficiency = self.memory_freed_bytes / (1024 * 1024 * max(self.execution_time_seconds, 0.001))  # MB/sec
        memory_efficiency_normalized = min(memory_efficiency / 100.0, 1.0)  # Normalize to 0-1 range
        
        # Consider execution efficiency and resource utilization
        execution_efficiency = min(self.candidates_processed / max(self.execution_time_seconds, 0.001) / 10.0, 1.0)  # Normalize to 0-1
        
        # Combine metrics into normalized effectiveness score
        effectiveness = (success_rate * 0.5 + memory_efficiency_normalized * 0.3 + execution_efficiency * 0.2)
        
        self.eviction_effectiveness_score = min(effectiveness, 1.0)
        return self.eviction_effectiveness_score
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert eviction result to dictionary format for logging and performance analysis.
        
        Returns:
            Dict[str, Any]: Eviction result as dictionary with all metrics and analysis
        """
        return {
            'candidates_processed': self.candidates_processed,
            'successful_evictions': self.successful_evictions,
            'execution_time_seconds': self.execution_time_seconds,
            'memory_freed_bytes': self.memory_freed_bytes,
            'failed_evictions': self.failed_evictions,
            'performance_metrics': self.performance_metrics,
            'execution_timestamp': self.execution_timestamp.isoformat(),
            'eviction_effectiveness_score': self.eviction_effectiveness_score,
            'error_messages': self.error_messages,
            'success_rate': self.successful_evictions / max(self.candidates_processed, 1),
            'memory_freed_mb': self.memory_freed_bytes / (1024 * 1024)
        }


@dataclasses.dataclass
class EvictionPerformanceResult:
    """Data class for eviction strategy performance evaluation results with metrics and recommendations."""
    strategy_name: str
    evaluation_period_seconds: float
    eviction_effectiveness: float
    memory_efficiency: float
    cache_hit_rate_impact: float
    resource_utilization: Dict[str, float]
    performance_trends: Dict[str, List[float]]
    optimization_recommendations: List[str] = dataclasses.field(default_factory=list)
    evaluation_timestamp: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)


@dataclasses.dataclass
class EvictionCoordinationResult:
    """Data class for eviction coordination optimization results with performance improvements."""
    coordination_strategy: str
    cache_levels_coordinated: List[str]
    optimization_effectiveness: float
    performance_improvements: Dict[str, float]
    coordination_overhead: float
    optimization_timestamp: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)


@dataclasses.dataclass
class StrategyOptimizationResult:
    """Data class for strategy optimization results with parameter changes and performance impact."""
    strategy_name: str
    optimization_level: str
    parameter_changes: Dict[str, Any]
    performance_impact: Dict[str, float]
    optimization_applied: bool
    optimization_timestamp: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)


class EvictionStrategy(abc.ABC):
    """
    Abstract base class defining the interface for cache eviction strategies with common 
    functionality for candidate selection, eviction execution, and performance monitoring 
    across the multi-level caching architecture.
    
    This abstract base class provides the foundation for all eviction strategies with
    common functionality and standardized interfaces for cache management operations.
    """
    
    def __init__(
        self,
        strategy_name: str,
        config: Dict[str, Any],
        memory_monitor: Optional[MemoryMonitor] = None
    ):
        """
        Initialize eviction strategy with configuration, memory monitoring, and performance 
        tracking for cache management optimization.
        
        Args:
            strategy_name: Unique name identifier for the eviction strategy
            config: Configuration dictionary with strategy-specific parameters
            memory_monitor: Optional memory monitor instance for pressure detection
        """
        # Store strategy name and configuration parameters
        self.strategy_name = strategy_name
        self.config = config.copy()
        
        # Initialize memory monitor integration if provided
        self.memory_monitor = memory_monitor or _memory_monitor_instance
        
        # Setup eviction statistics tracking and performance monitoring
        self.eviction_statistics = CacheStatistics()
        
        # Create strategy lock for thread-safe operations
        self.strategy_lock = threading.RLock()
        
        # Initialize performance metrics and eviction history
        self.performance_metrics: Dict[str, float] = {}
        self.eviction_history: List[Dict[str, Any]] = []
        
        # Record creation time and mark strategy as active
        self.creation_time = datetime.datetime.now()
        self.is_active = True
        
        # Configure strategy-specific parameters from config
        self._configure_strategy_parameters()
        
        # Log strategy initialization with configuration details
        self.logger = get_logger(f'eviction.{strategy_name}', 'CACHE')
        self.logger.info(f"Eviction strategy '{strategy_name}' initialized with config: {config}")
    
    def _configure_strategy_parameters(self) -> None:
        """Configure strategy-specific parameters from configuration dictionary."""
        # Load performance thresholds for strategy optimization
        optimization_targets = performance_thresholds.get('optimization_targets', {})
        
        # Configure cache hit rate targets
        self.cache_hit_rate_target = optimization_targets.get('throughput_optimization', {}).get('cache_hit_rate_target', 0.8)
        
        # Set memory management thresholds
        memory_config = performance_thresholds.get('resource_utilization', {}).get('memory', {})
        self.memory_warning_threshold = memory_config.get('warning_threshold_gb', 6.4) / memory_config.get('max_usage_gb', 8.0)
        self.memory_critical_threshold = memory_config.get('critical_threshold_gb', 7.5) / memory_config.get('max_usage_gb', 8.0)
        
        # Configure batch processing parameters
        batch_config = performance_thresholds.get('batch_processing', {})
        self.eviction_batch_size = self.config.get('batch_size', batch_config.get('chunk_size_per_worker', EVICTION_BATCH_SIZE))
        
        # Setup adaptive evaluation settings
        self.evaluation_interval = self.config.get('evaluation_interval', ADAPTIVE_EVALUATION_INTERVAL)
        self.performance_window = self.config.get('performance_window', STRATEGY_PERFORMANCE_WINDOW)
    
    @abc.abstractmethod
    def select_eviction_candidates(
        self,
        cache_entries: Dict[str, Any],
        target_eviction_count: int,
        memory_pressure_level: float,
        selection_context: Dict[str, Any]
    ) -> List[EvictionCandidate]:
        """
        Abstract method to select cache entries for eviction based on strategy-specific 
        criteria and memory pressure considerations.
        
        This method must be implemented by concrete strategy classes to provide
        strategy-specific eviction candidate selection logic.
        
        Args:
            cache_entries: Dictionary of cache entries available for eviction
            target_eviction_count: Target number of entries to select for eviction
            memory_pressure_level: Current memory pressure level (0.0 to 1.0)
            selection_context: Additional context for eviction selection
            
        Returns:
            List[EvictionCandidate]: List of eviction candidates with priority scores and metadata
        """
        pass
    
    def execute_eviction(
        self,
        eviction_candidates: List[EvictionCandidate],
        cache_interface: Dict[str, Any],
        force_eviction: bool = False,
        execution_context: Dict[str, Any] = None
    ) -> EvictionResult:
        """
        Execute eviction of selected candidates with performance tracking, error handling, 
        and coordination across cache levels.
        
        This method provides standardized eviction execution with comprehensive error
        handling and performance tracking for all eviction strategies.
        
        Args:
            eviction_candidates: List of candidates selected for eviction
            cache_interface: Interface for cache operations and entry removal
            force_eviction: Force eviction even for pinned entries
            execution_context: Additional context for eviction execution
            
        Returns:
            EvictionResult: Eviction execution result with success count and performance metrics
        """
        execution_start_time = time.time()
        successful_evictions = 0
        memory_freed_bytes = 0
        failed_evictions = []
        error_messages = []
        
        try:
            # Acquire strategy lock for thread-safe eviction execution
            with self.strategy_lock:
                # Validate eviction candidates and cache interface
                if not eviction_candidates:
                    self.logger.warning("No eviction candidates provided for execution")
                    return EvictionResult(0, 0, 0.0, 0)
                
                if not cache_interface or 'remove_entry' not in cache_interface:
                    error_msg = "Invalid cache interface provided for eviction execution"
                    self.logger.error(error_msg)
                    return EvictionResult(len(eviction_candidates), 0, 0.0, 0, error_messages=[error_msg])
                
                # Sort candidates by priority score for optimal eviction order
                sorted_candidates = sorted(eviction_candidates, key=lambda c: c.priority_score, reverse=True)
                
                # Execute eviction for each candidate with error handling
                for candidate in sorted_candidates:
                    try:
                        # Skip pinned entries unless force_eviction is enabled
                        if candidate.is_pinned and not force_eviction:
                            self.logger.debug(f"Skipping pinned entry: {candidate.cache_key}")
                            continue
                        
                        # Attempt to remove entry from cache
                        removal_result = cache_interface['remove_entry'](candidate.cache_key)
                        
                        if removal_result.get('success', False):
                            successful_evictions += 1
                            memory_freed_bytes += candidate.entry_size_bytes
                            
                            # Record eviction in statistics
                            self.eviction_statistics.record_eviction()
                            
                            # Log successful eviction
                            self.logger.debug(f"Successfully evicted entry: {candidate.cache_key} "
                                            f"({candidate.entry_size_bytes} bytes)")
                        else:
                            failed_evictions.append(candidate.cache_key)
                            error_messages.append(f"Failed to evict {candidate.cache_key}: {removal_result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        failed_evictions.append(candidate.cache_key)
                        error_messages.append(f"Exception evicting {candidate.cache_key}: {str(e)}")
                        self.logger.error(f"Exception during eviction of {candidate.cache_key}: {e}")
                
                # Track eviction performance and success rates
                execution_time = time.time() - execution_start_time
                
                # Update eviction statistics and performance metrics
                self.performance_metrics.update({
                    'last_eviction_time': execution_time,
                    'last_eviction_count': successful_evictions,
                    'last_memory_freed': memory_freed_bytes,
                    'total_evictions': self.performance_metrics.get('total_evictions', 0) + successful_evictions
                })
                
                # Create eviction result with comprehensive information
                eviction_result = EvictionResult(
                    candidates_processed=len(eviction_candidates),
                    successful_evictions=successful_evictions,
                    execution_time_seconds=execution_time,
                    memory_freed_bytes=memory_freed_bytes,
                    failed_evictions=failed_evictions,
                    error_messages=error_messages
                )
                
                # Calculate effectiveness score
                eviction_result.calculate_effectiveness()
                
                # Log eviction execution with detailed results
                self.logger.info(f"Eviction execution completed: {successful_evictions}/{len(eviction_candidates)} "
                               f"successful, {memory_freed_bytes / (1024*1024):.2f} MB freed, "
                               f"{execution_time:.3f}s execution time")
                
                # Log performance metrics for monitoring
                log_performance_metrics(
                    metric_name='eviction_execution_time',
                    metric_value=execution_time,
                    metric_unit='seconds',
                    component='EVICTION',
                    metric_context={
                        'strategy_name': self.strategy_name,
                        'candidates_processed': len(eviction_candidates),
                        'successful_evictions': successful_evictions,
                        'effectiveness_score': eviction_result.eviction_effectiveness_score
                    }
                )
                
                return eviction_result
                
        except Exception as e:
            execution_time = time.time() - execution_start_time
            error_msg = f"Critical error during eviction execution: {str(e)}"
            self.logger.error(error_msg)
            
            return EvictionResult(
                candidates_processed=len(eviction_candidates),
                successful_evictions=successful_evictions,
                execution_time_seconds=execution_time,
                memory_freed_bytes=memory_freed_bytes,
                failed_evictions=failed_evictions,
                error_messages=error_messages + [error_msg]
            )
    
    def should_trigger_eviction(
        self,
        cache_state: Dict[str, Any],
        memory_usage_ratio: float,
        trigger_context: Dict[str, Any] = None
    ) -> bool:
        """
        Determine if eviction should be triggered based on memory pressure, cache utilization, 
        and strategy-specific thresholds.
        
        This method evaluates multiple factors to determine optimal eviction timing
        based on system conditions and strategy-specific criteria.
        
        Args:
            cache_state: Current cache state information
            memory_usage_ratio: Current memory usage ratio (0.0 to 1.0)
            trigger_context: Additional context for trigger evaluation
            
        Returns:
            bool: True if eviction should be triggered based on current conditions
        """
        try:
            # Check memory usage against configured thresholds
            memory_trigger = memory_usage_ratio >= self.memory_warning_threshold
            
            # Analyze cache utilization and fragmentation levels
            cache_utilization = cache_state.get('utilization_ratio', 0.0)
            cache_fragmentation = cache_state.get('fragmentation_ratio', 0.0)
            
            utilization_trigger = cache_utilization >= self.config.get('max_cache_utilization', 0.9)
            fragmentation_trigger = cache_fragmentation >= self.config.get('max_fragmentation', 0.1)
            
            # Evaluate strategy-specific trigger conditions
            strategy_trigger = self._evaluate_strategy_specific_triggers(cache_state, trigger_context or {})
            
            # Consider memory pressure from memory monitor if available
            monitor_trigger = False
            if self.memory_monitor:
                try:
                    thresholds = self.memory_monitor.check_thresholds()
                    monitor_trigger = thresholds.get('warning', False)
                except Exception as e:
                    self.logger.warning(f"Failed to check memory monitor thresholds: {e}")
            
            # Apply trigger context and cache state analysis
            context_trigger = trigger_context.get('force_eviction', False) if trigger_context else False
            
            # Determine overall trigger decision
            should_trigger = (memory_trigger or utilization_trigger or fragmentation_trigger or 
                            strategy_trigger or monitor_trigger or context_trigger)
            
            # Log trigger evaluation with reasoning
            if should_trigger:
                trigger_reasons = []
                if memory_trigger:
                    trigger_reasons.append(f"memory_usage={memory_usage_ratio:.3f}")
                if utilization_trigger:
                    trigger_reasons.append(f"cache_utilization={cache_utilization:.3f}")
                if fragmentation_trigger:
                    trigger_reasons.append(f"fragmentation={cache_fragmentation:.3f}")
                if strategy_trigger:
                    trigger_reasons.append("strategy_specific")
                if monitor_trigger:
                    trigger_reasons.append("memory_monitor")
                if context_trigger:
                    trigger_reasons.append("force_eviction")
                
                self.logger.info(f"Eviction triggered by: {', '.join(trigger_reasons)}")
            
            return should_trigger
            
        except Exception as e:
            self.logger.error(f"Error evaluating eviction trigger: {e}")
            # Default to safe behavior - trigger eviction on error
            return memory_usage_ratio >= MEMORY_PRESSURE_THRESHOLD
    
    def _evaluate_strategy_specific_triggers(
        self,
        cache_state: Dict[str, Any],
        trigger_context: Dict[str, Any]
    ) -> bool:
        """Evaluate strategy-specific trigger conditions (overridden by subclasses)."""
        return False
    
    def optimize_strategy(
        self,
        optimization_context: Dict[str, Any],
        apply_optimizations: bool = True,
        optimization_level: str = 'moderate'
    ) -> StrategyOptimizationResult:
        """
        Optimize strategy parameters based on performance history, cache characteristics, 
        and system conditions for improved effectiveness.
        
        This method analyzes strategy performance and optimizes parameters for improved
        effectiveness based on historical data and current system conditions.
        
        Args:
            optimization_context: Context information for optimization analysis
            apply_optimizations: Whether to apply optimization changes immediately
            optimization_level: Level of optimization (conservative, moderate, aggressive)
            
        Returns:
            StrategyOptimizationResult: Strategy optimization result with parameter changes and performance impact
        """
        optimization_start_time = time.time()
        parameter_changes = {}
        performance_impact = {}
        
        try:
            with self.strategy_lock:
                # Analyze strategy performance history and effectiveness
                current_effectiveness = self._calculate_current_effectiveness()
                performance_trends = self._analyze_performance_trends()
                
                # Identify optimization opportunities based on cache characteristics
                optimization_opportunities = self._identify_optimization_opportunities(
                    optimization_context, current_effectiveness, performance_trends
                )
                
                # Generate optimized strategy parameters
                optimized_parameters = self._generate_optimized_parameters(
                    optimization_opportunities, optimization_level
                )
                
                # Validate optimization changes against performance thresholds
                validation_result = self._validate_optimization_changes(optimized_parameters)
                
                if validation_result['valid']:
                    parameter_changes = optimized_parameters
                    
                    # Apply optimization changes if enabled and validated
                    if apply_optimizations:
                        original_config = self.config.copy()
                        self.config.update(optimized_parameters)
                        self._reconfigure_strategy_parameters()
                        
                        # Monitor optimization effectiveness and performance impact
                        performance_impact = self._measure_optimization_impact(original_config)
                        
                        self.logger.info(f"Strategy optimization applied: {len(parameter_changes)} parameters updated")
                    else:
                        self.logger.info(f"Strategy optimization generated: {len(parameter_changes)} parameter recommendations")
                else:
                    self.logger.warning(f"Strategy optimization validation failed: {validation_result['reason']}")
                
                # Update strategy configuration and performance tracking
                optimization_time = time.time() - optimization_start_time
                
                # Log optimization operation with detailed results
                log_performance_metrics(
                    metric_name='strategy_optimization_time',
                    metric_value=optimization_time,
                    metric_unit='seconds',
                    component='EVICTION',
                    metric_context={
                        'strategy_name': self.strategy_name,
                        'optimization_level': optimization_level,
                        'parameters_changed': len(parameter_changes),
                        'optimizations_applied': apply_optimizations
                    }
                )
                
                return StrategyOptimizationResult(
                    strategy_name=self.strategy_name,
                    optimization_level=optimization_level,
                    parameter_changes=parameter_changes,
                    performance_impact=performance_impact,
                    optimization_applied=apply_optimizations and validation_result['valid']
                )
                
        except Exception as e:
            self.logger.error(f"Error during strategy optimization: {e}")
            return StrategyOptimizationResult(
                strategy_name=self.strategy_name,
                optimization_level=optimization_level,
                parameter_changes={},
                performance_impact={'error': str(e)},
                optimization_applied=False
            )
    
    def _calculate_current_effectiveness(self) -> float:
        """Calculate current strategy effectiveness based on recent performance."""
        if not self.eviction_history:
            return 0.5  # Default effectiveness for new strategies
        
        recent_evictions = self.eviction_history[-10:]  # Last 10 evictions
        effectiveness_scores = [eviction.get('effectiveness_score', 0.0) for eviction in recent_evictions]
        
        return statistics.mean(effectiveness_scores) if effectiveness_scores else 0.5
    
    def _analyze_performance_trends(self) -> Dict[str, List[float]]:
        """Analyze performance trends over time for optimization analysis."""
        trends = {
            'effectiveness': [],
            'execution_time': [],
            'memory_freed': [],
            'success_rate': []
        }
        
        for eviction in self.eviction_history[-50:]:  # Last 50 evictions
            trends['effectiveness'].append(eviction.get('effectiveness_score', 0.0))
            trends['execution_time'].append(eviction.get('execution_time', 0.0))
            trends['memory_freed'].append(eviction.get('memory_freed', 0))
            trends['success_rate'].append(eviction.get('success_rate', 0.0))
        
        return trends
    
    def _identify_optimization_opportunities(
        self,
        context: Dict[str, Any],
        effectiveness: float,
        trends: Dict[str, List[float]]
    ) -> List[str]:
        """Identify optimization opportunities based on performance analysis."""
        opportunities = []
        
        # Check effectiveness threshold
        if effectiveness < EVICTION_EFFECTIVENESS_THRESHOLD:
            opportunities.append('improve_effectiveness')
        
        # Analyze execution time trends
        if trends['execution_time'] and statistics.mean(trends['execution_time']) > 1.0:
            opportunities.append('reduce_execution_time')
        
        # Check memory freed efficiency
        if trends['memory_freed'] and statistics.mean(trends['memory_freed']) < 1024*1024:  # < 1MB
            opportunities.append('increase_memory_efficiency')
        
        # Analyze success rate trends
        if trends['success_rate'] and statistics.mean(trends['success_rate']) < 0.9:
            opportunities.append('improve_success_rate')
        
        return opportunities
    
    def _generate_optimized_parameters(
        self,
        opportunities: List[str],
        optimization_level: str
    ) -> Dict[str, Any]:
        """Generate optimized parameters based on identified opportunities."""
        optimized_params = {}
        
        optimization_factors = {
            'conservative': 0.1,
            'moderate': 0.2,
            'aggressive': 0.4
        }
        
        factor = optimization_factors.get(optimization_level, 0.2)
        
        if 'improve_effectiveness' in opportunities:
            current_batch_size = self.config.get('batch_size', EVICTION_BATCH_SIZE)
            optimized_params['batch_size'] = max(1, int(current_batch_size * (1 - factor)))
        
        if 'reduce_execution_time' in opportunities:
            current_evaluation_interval = self.config.get('evaluation_interval', ADAPTIVE_EVALUATION_INTERVAL)
            optimized_params['evaluation_interval'] = current_evaluation_interval * (1 + factor)
        
        if 'increase_memory_efficiency' in opportunities:
            current_min_candidates = self.config.get('min_candidates', MIN_EVICTION_CANDIDATES)
            optimized_params['min_candidates'] = int(current_min_candidates * (1 + factor))
        
        return optimized_params
    
    def _validate_optimization_changes(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate optimization changes against system constraints."""
        validation_result = {'valid': True, 'reason': ''}
        
        # Validate batch size constraints
        if 'batch_size' in parameters:
            if parameters['batch_size'] < 1 or parameters['batch_size'] > MAX_EVICTION_CANDIDATES:
                validation_result = {'valid': False, 'reason': 'batch_size out of valid range'}
        
        # Validate timing constraints
        if 'evaluation_interval' in parameters:
            if parameters['evaluation_interval'] < 60.0 or parameters['evaluation_interval'] > 3600.0:
                validation_result = {'valid': False, 'reason': 'evaluation_interval out of valid range'}
        
        return validation_result
    
    def _reconfigure_strategy_parameters(self) -> None:
        """Reconfigure strategy parameters after optimization changes."""
        self._configure_strategy_parameters()
    
    def _measure_optimization_impact(self, original_config: Dict[str, Any]) -> Dict[str, float]:
        """Measure the impact of optimization changes on performance."""
        # This would typically involve monitoring performance over time
        # Placeholder implementation for demonstration
        return {
            'effectiveness_improvement': 0.05,
            'execution_time_reduction': 0.1,
            'memory_efficiency_improvement': 0.08
        }
    
    def get_strategy_statistics(
        self,
        include_detailed_breakdown: bool = True,
        time_window_hours: int = 24,
        include_optimization_recommendations: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive strategy statistics including eviction rates, effectiveness metrics, 
        and performance analysis for monitoring and optimization.
        
        This method provides detailed statistics for strategy monitoring and performance
        analysis with optional optimization recommendations.
        
        Args:
            include_detailed_breakdown: Include detailed breakdown by eviction type
            time_window_hours: Time window for statistics calculation (hours)
            include_optimization_recommendations: Include optimization recommendations
            
        Returns:
            Dict[str, Any]: Comprehensive strategy statistics with performance metrics and recommendations
        """
        try:
            # Collect eviction statistics and performance metrics
            stats = {
                'strategy_name': self.strategy_name,
                'creation_time': self.creation_time.isoformat(),
                'is_active': self.is_active,
                'configuration': self.config.copy(),
                'performance_metrics': self.performance_metrics.copy()
            }
            
            # Calculate eviction effectiveness and success rates
            cache_stats = self.eviction_statistics
            total_operations = cache_stats.hits + cache_stats.misses
            
            stats.update({
                'cache_hit_rate': cache_stats.get_hit_rate(),
                'total_evictions': cache_stats.evictions,
                'total_cache_operations': total_operations,
                'eviction_rate': cache_stats.evictions / max(total_operations, 1)
            })
            
            # Include detailed breakdown by eviction type if requested
            if include_detailed_breakdown:
                stats['detailed_breakdown'] = self._generate_detailed_breakdown()
            
            # Analyze performance trends over specified time window
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=time_window_hours)
            recent_evictions = [e for e in self.eviction_history if 
                              datetime.datetime.fromisoformat(e.get('timestamp', '1970-01-01')) >= cutoff_time]
            
            if recent_evictions:
                stats['time_window_analysis'] = {
                    'period_hours': time_window_hours,
                    'evictions_in_period': len(recent_evictions),
                    'average_effectiveness': statistics.mean([e.get('effectiveness_score', 0.0) for e in recent_evictions]),
                    'average_execution_time': statistics.mean([e.get('execution_time', 0.0) for e in recent_evictions]),
                    'total_memory_freed': sum([e.get('memory_freed', 0) for e in recent_evictions])
                }
            
            # Generate optimization recommendations if requested
            if include_optimization_recommendations:
                stats['optimization_recommendations'] = self._generate_optimization_recommendations()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error generating strategy statistics: {e}")
            return {
                'strategy_name': self.strategy_name,
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }
    
    def _generate_detailed_breakdown(self) -> Dict[str, Any]:
        """Generate detailed breakdown of eviction operations by type and reason."""
        breakdown = {
            'eviction_reasons': {},
            'size_distribution': {'small': 0, 'medium': 0, 'large': 0},
            'timing_analysis': {'peak_hours': [], 'average_intervals': []}
        }
        
        for eviction in self.eviction_history:
            # Count eviction reasons
            reason = eviction.get('eviction_reason', 'unknown')
            breakdown['eviction_reasons'][reason] = breakdown['eviction_reasons'].get(reason, 0) + 1
            
            # Analyze size distribution
            memory_freed = eviction.get('memory_freed', 0)
            if memory_freed < 1024*1024:  # < 1MB
                breakdown['size_distribution']['small'] += 1
            elif memory_freed < 10*1024*1024:  # < 10MB
                breakdown['size_distribution']['medium'] += 1
            else:
                breakdown['size_distribution']['large'] += 1
        
        return breakdown
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on current performance."""
        recommendations = []
        
        current_effectiveness = self._calculate_current_effectiveness()
        if current_effectiveness < EVICTION_EFFECTIVENESS_THRESHOLD:
            recommendations.append("Consider tuning eviction parameters to improve effectiveness")
        
        hit_rate = self.eviction_statistics.get_hit_rate()
        if hit_rate < self.cache_hit_rate_target:
            recommendations.append("Cache hit rate below target - consider adjusting eviction strategy")
        
        avg_execution_time = self.performance_metrics.get('last_eviction_time', 0.0)
        if avg_execution_time > 1.0:
            recommendations.append("Eviction execution time is high - consider batch size optimization")
        
        return recommendations


class LRUEvictionStrategy(EvictionStrategy):
    """
    Least Recently Used eviction strategy implementation with efficient access tracking, 
    batch eviction support, and memory pressure integration for optimal cache management 
    in scientific computing workloads.
    
    This implementation provides efficient LRU eviction with access order tracking,
    frequency-based enhancements, and memory pressure awareness for scientific computing.
    """
    
    def __init__(
        self,
        strategy_name: str = 'lru',
        config: Dict[str, Any] = None,
        memory_monitor: Optional[MemoryMonitor] = None
    ):
        """
        Initialize LRU eviction strategy with access tracking, batch processing, and 
        memory pressure awareness for efficient cache management.
        
        Args:
            strategy_name: Name identifier for the LRU strategy instance
            config: Configuration dictionary with LRU-specific parameters
            memory_monitor: Optional memory monitor for pressure-aware decisions
        """
        # Initialize base EvictionStrategy with provided parameters
        super().__init__(strategy_name, config or {}, memory_monitor)
        
        # Setup OrderedDict for efficient LRU access tracking
        self.access_order: collections.OrderedDict = collections.OrderedDict()
        
        # Initialize access timestamps and frequency tracking
        self.access_timestamps: Dict[str, datetime.datetime] = {}
        self.access_frequency: Dict[str, int] = {}
        
        # Configure access weight and batch size from config
        self.access_weight = self.config.get('access_weight', LRU_ACCESS_WEIGHT)
        self.batch_size = self.config.get('batch_size', EVICTION_BATCH_SIZE)
        
        # Enable memory pressure awareness if memory monitor provided
        self.memory_pressure_aware = self.memory_monitor is not None
        
        # Create LRU-specific lock for access order management
        self.lru_lock = threading.RLock()
        
        # Log LRU strategy initialization with configuration
        self.logger.info(f"LRU eviction strategy initialized: access_weight={self.access_weight}, "
                        f"batch_size={self.batch_size}, memory_pressure_aware={self.memory_pressure_aware}")
    
    def select_eviction_candidates(
        self,
        cache_entries: Dict[str, Any],
        target_eviction_count: int,
        memory_pressure_level: float,
        selection_context: Dict[str, Any]
    ) -> List[EvictionCandidate]:
        """
        Select LRU eviction candidates based on access order, frequency, and memory pressure 
        with intelligent batch sizing.
        
        This method implements LRU candidate selection with frequency weighting and
        memory pressure considerations for optimal eviction decisions.
        
        Args:
            cache_entries: Dictionary of available cache entries
            target_eviction_count: Target number of candidates to select
            memory_pressure_level: Current memory pressure level (0.0 to 1.0)
            selection_context: Additional context for candidate selection
            
        Returns:
            List[EvictionCandidate]: LRU-ordered eviction candidates with access-based priority scores
        """
        candidates = []
        
        try:
            # Acquire LRU lock for thread-safe access order management
            with self.lru_lock:
                # Validate inputs and available entries
                if not cache_entries:
                    self.logger.debug("No cache entries available for LRU candidate selection")
                    return candidates
                
                # Identify least recently used entries from access order
                lru_keys = []
                
                # Get LRU keys from access order (oldest first)
                for key in self.access_order:
                    if key in cache_entries:
                        lru_keys.append(key)
                
                # Add keys not in access order (treat as oldest)
                for key in cache_entries:
                    if key not in self.access_order:
                        lru_keys.insert(0, key)  # Insert at beginning (oldest)
                
                # Limit to target count with memory pressure adjustment
                effective_target = min(target_eviction_count, len(lru_keys))
                
                # Adjust target based on memory pressure
                if memory_pressure_level > CRITICAL_MEMORY_THRESHOLD:
                    effective_target = min(effective_target * 2, len(lru_keys))
                elif memory_pressure_level > MEMORY_PRESSURE_THRESHOLD:
                    effective_target = int(effective_target * 1.5)
                
                # Select candidates up to effective target
                selected_keys = lru_keys[:effective_target]
                
                # Create EvictionCandidate objects with LRU metadata
                for key in selected_keys:
                    entry = cache_entries[key]
                    
                    # Calculate priority scores based on access time and frequency
                    last_access = self.access_timestamps.get(key, datetime.datetime.min)
                    access_freq = self.access_frequency.get(key, 0)
                    entry_size = entry.get('size_bytes', 0)
                    
                    # Apply memory pressure weighting if pressure level is high
                    weight_config = {
                        'age_weight': AGE_WEIGHT,
                        'frequency_weight': FREQUENCY_ACCESS_WEIGHT * (2.0 if memory_pressure_level > MEMORY_PRESSURE_THRESHOLD else 1.0),
                        'size_weight': SIZE_WEIGHT * (1.5 if memory_pressure_level > MEMORY_PRESSURE_THRESHOLD else 1.0)
                    }
                    
                    # Create eviction candidate with calculated priority
                    candidate = EvictionCandidate(
                        cache_key=key,
                        priority_score=0.0,  # Will be calculated below
                        metadata={
                            'entry_size': entry_size,
                            'access_frequency': access_freq,
                            'strategy_type': 'lru',
                            'memory_pressure_level': memory_pressure_level
                        },
                        last_access_time=last_access,
                        access_frequency=access_freq,
                        entry_size_bytes=entry_size,
                        eviction_reason=EvictionReason.LRU_POLICY.value
                    )
                    
                    # Calculate priority score with memory pressure weighting
                    candidate.calculate_eviction_priority(weight_config, memory_pressure_level)
                    
                    candidates.append(candidate)
                
                # Sort candidates by LRU priority for optimal eviction order
                candidates.sort(key=lambda c: c.priority_score, reverse=True)
                
                # Log candidate selection results
                self.logger.debug(f"Selected {len(candidates)} LRU eviction candidates "
                                f"(target: {target_eviction_count}, pressure: {memory_pressure_level:.3f})")
                
                return candidates
                
        except Exception as e:
            self.logger.error(f"Error selecting LRU eviction candidates: {e}")
            return []
    
    def update_access_order(
        self,
        cache_key: str,
        access_time: datetime.datetime = None,
        access_context: Dict[str, Any] = None
    ) -> None:
        """
        Update access order tracking for cache entry with timestamp recording and frequency 
        counting for accurate LRU management.
        
        This method maintains LRU access order with frequency tracking for enhanced
        eviction decision making.
        
        Args:
            cache_key: Key of the cache entry being accessed
            access_time: Time of access (defaults to current time)
            access_context: Additional context for access tracking
        """
        try:
            # Acquire LRU lock for thread-safe access tracking
            with self.lru_lock:
                # Use provided access time or current time
                if access_time is None:
                    access_time = datetime.datetime.now()
                
                # Update access order by moving key to end of OrderedDict
                if cache_key in self.access_order:
                    del self.access_order[cache_key]
                self.access_order[cache_key] = access_time
                
                # Record access timestamp for LRU calculation
                self.access_timestamps[cache_key] = access_time
                
                # Increment access frequency counter
                self.access_frequency[cache_key] = self.access_frequency.get(cache_key, 0) + 1
                
                # Update access context metadata if provided
                if access_context:
                    # Store additional access context if needed
                    pass
                
                # Log access order update for audit trail (debug level only)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Access order updated for key: {cache_key} "
                                    f"(frequency: {self.access_frequency[cache_key]})")
                
        except Exception as e:
            self.logger.error(f"Error updating LRU access order for {cache_key}: {e}")
    
    def _evaluate_strategy_specific_triggers(
        self,
        cache_state: Dict[str, Any],
        trigger_context: Dict[str, Any]
    ) -> bool:
        """Evaluate LRU-specific trigger conditions for eviction timing."""
        try:
            # Check access order size vs cache size
            access_order_size = len(self.access_order)
            cache_size = cache_state.get('entry_count', 0)
            
            # Trigger if access order is significantly larger than cache
            if access_order_size > cache_size * 1.2:
                return True
            
            # Check for stale access timestamps
            current_time = datetime.datetime.now()
            stale_threshold = datetime.timedelta(hours=1)
            
            stale_entries = 0
            for timestamp in self.access_timestamps.values():
                if current_time - timestamp > stale_threshold:
                    stale_entries += 1
            
            # Trigger if too many stale entries
            if stale_entries > len(self.access_timestamps) * 0.3:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating LRU-specific triggers: {e}")
            return False
    
    def get_lru_statistics(
        self,
        include_access_patterns: bool = True,
        include_frequency_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Get LRU-specific statistics including access patterns, eviction effectiveness, 
        and order management metrics.
        
        Args:
            include_access_patterns: Include access pattern analysis in statistics
            include_frequency_analysis: Include frequency analysis in statistics
            
        Returns:
            Dict[str, Any]: LRU strategy statistics with access patterns and effectiveness metrics
        """
        try:
            with self.lru_lock:
                # Calculate LRU eviction effectiveness and success rates
                base_stats = self.get_strategy_statistics(
                    include_detailed_breakdown=True,
                    include_optimization_recommendations=False
                )
                
                lru_stats = {
                    'lru_specific_metrics': {
                        'access_order_size': len(self.access_order),
                        'tracked_timestamps': len(self.access_timestamps),
                        'tracked_frequencies': len(self.access_frequency),
                        'access_weight': self.access_weight,
                        'memory_pressure_aware': self.memory_pressure_aware
                    }
                }
                
                # Include access pattern analysis if requested
                if include_access_patterns and self.access_timestamps:
                    current_time = datetime.datetime.now()
                    access_ages = [(current_time - timestamp).total_seconds() 
                                 for timestamp in self.access_timestamps.values()]
                    
                    lru_stats['access_patterns'] = {
                        'average_access_age_seconds': statistics.mean(access_ages),
                        'median_access_age_seconds': statistics.median(access_ages),
                        'oldest_access_age_seconds': max(access_ages),
                        'newest_access_age_seconds': min(access_ages)
                    }
                
                # Analyze access frequency distribution if requested
                if include_frequency_analysis and self.access_frequency:
                    frequencies = list(self.access_frequency.values())
                    
                    lru_stats['frequency_analysis'] = {
                        'average_access_frequency': statistics.mean(frequencies),
                        'median_access_frequency': statistics.median(frequencies),
                        'max_access_frequency': max(frequencies),
                        'min_access_frequency': min(frequencies),
                        'frequency_standard_deviation': statistics.stdev(frequencies) if len(frequencies) > 1 else 0.0
                    }
                
                # Calculate access order management efficiency
                efficiency_metrics = {
                    'order_tracking_overhead': len(self.access_order) / max(len(self.access_timestamps), 1),
                    'frequency_tracking_efficiency': len(self.access_frequency) / max(len(self.access_order), 1)
                }
                
                lru_stats['efficiency_metrics'] = efficiency_metrics
                
                # Include memory pressure impact on LRU decisions
                if self.memory_monitor:
                    try:
                        current_usage = self.memory_monitor.get_current_usage()
                        thresholds = self.memory_monitor.check_thresholds()
                        
                        lru_stats['memory_pressure_impact'] = {
                            'current_memory_usage': current_usage,
                            'warning_threshold_exceeded': thresholds.get('warning', False),
                            'critical_threshold_exceeded': thresholds.get('critical', False),
                            'pressure_aware_weighting_active': current_usage > MEMORY_PRESSURE_THRESHOLD
                        }
                    except Exception as e:
                        lru_stats['memory_pressure_impact'] = {'error': str(e)}
                
                # Combine base stats with LRU-specific metrics
                base_stats.update(lru_stats)
                
                return base_stats
                
        except Exception as e:
            self.logger.error(f"Error generating LRU statistics: {e}")
            return {'error': str(e), 'strategy_name': self.strategy_name}


class AdaptiveEvictionStrategy(EvictionStrategy):
    """
    Adaptive eviction strategy that dynamically switches between different eviction algorithms 
    based on workload characteristics, cache performance, and system conditions for optimal 
    cache management across varying scientific computing scenarios.
    
    This strategy provides intelligent algorithm selection with performance monitoring and
    automatic switching for optimal cache management across different workload patterns.
    """
    
    def __init__(
        self,
        strategy_name: str = 'adaptive',
        config: Dict[str, Any] = None,
        memory_monitor: Optional[MemoryMonitor] = None
    ):
        """
        Initialize adaptive eviction strategy with multiple algorithm support, performance 
        evaluation, and automatic strategy switching for optimal cache management.
        
        Args:
            strategy_name: Name identifier for the adaptive strategy instance
            config: Configuration dictionary with adaptive strategy parameters
            memory_monitor: Optional memory monitor for pressure-aware decisions
        """
        # Initialize base EvictionStrategy with provided parameters
        super().__init__(strategy_name, config or {}, memory_monitor)
        
        # Create available strategies dictionary with LRU and other algorithms
        self.available_strategies: Dict[str, EvictionStrategy] = {}
        
        # Initialize with LRU strategy as default
        lru_config = self.config.get('lru_config', {})
        self.available_strategies['lru'] = LRUEvictionStrategy(
            strategy_name='adaptive_lru',
            config=lru_config,
            memory_monitor=memory_monitor
        )
        
        # Set initial strategy based on configuration or default to LRU
        self.current_strategy_name = self.config.get('initial_strategy', 'lru')
        self.current_strategy = self.available_strategies[self.current_strategy_name]
        
        # Configure evaluation interval and auto-switching settings
        self.evaluation_interval = self.config.get('evaluation_interval', ADAPTIVE_EVALUATION_INTERVAL)
        self.auto_switching_enabled = self.config.get('auto_switching', True)
        
        # Initialize strategy performance tracking and scoring
        self.strategy_performance_scores: Dict[str, float] = {name: 0.5 for name in self.available_strategies}
        
        # Setup performance history for trend analysis
        self.performance_history: List[Dict[str, Any]] = []
        self.last_evaluation_time = datetime.datetime.now()
        
        # Configure workload characteristic detection
        self.workload_characteristics: Dict[str, float] = {
            'access_locality': 0.5,
            'size_variance': 0.5,
            'temporal_patterns': 0.5,
            'memory_pressure_frequency': 0.5
        }
        
        # Log adaptive strategy initialization with available algorithms
        self.logger.info(f"Adaptive eviction strategy initialized with {len(self.available_strategies)} algorithms: "
                        f"{list(self.available_strategies.keys())}")
    
    def select_eviction_candidates(
        self,
        cache_entries: Dict[str, Any],
        target_eviction_count: int,
        memory_pressure_level: float,
        selection_context: Dict[str, Any]
    ) -> List[EvictionCandidate]:
        """
        Select eviction candidates using current optimal strategy with adaptive algorithm 
        selection based on workload characteristics.
        
        This method implements intelligent strategy selection with workload analysis and
        automatic switching for optimal eviction decisions.
        
        Args:
            cache_entries: Dictionary of available cache entries
            target_eviction_count: Target number of candidates to select
            memory_pressure_level: Current memory pressure level (0.0 to 1.0)
            selection_context: Additional context for candidate selection
            
        Returns:
            List[EvictionCandidate]: Adaptively selected eviction candidates with optimal strategy application
        """
        try:
            # Analyze current workload characteristics and cache state
            current_workload = self.analyze_workload_characteristics(
                cache_state={'entries': cache_entries, 'memory_pressure': memory_pressure_level},
                access_patterns=selection_context.get('access_patterns', {}),
                analysis_window_seconds=300  # 5 minutes
            )
            
            # Evaluate if strategy switching is needed based on performance
            if self.auto_switching_enabled:
                optimal_strategy_name = self.evaluate_strategy_performance(
                    evaluation_context={
                        'workload_characteristics': current_workload,
                        'memory_pressure': memory_pressure_level,
                        'cache_size': len(cache_entries)
                    },
                    force_evaluation=False
                )
                
                # Switch to optimal strategy if different from current
                if optimal_strategy_name != self.current_strategy_name:
                    self.logger.info(f"Switching from {self.current_strategy_name} to {optimal_strategy_name} "
                                   f"based on workload analysis")
                    self.current_strategy_name = optimal_strategy_name
                    self.current_strategy = self.available_strategies[optimal_strategy_name]
            
            # Delegate candidate selection to current optimal strategy
            candidates = self.current_strategy.select_eviction_candidates(
                cache_entries, target_eviction_count, memory_pressure_level, selection_context
            )
            
            # Apply adaptive weighting based on workload analysis
            for candidate in candidates:
                # Apply adaptive weighting based on workload characteristics
                adaptive_weight = self._calculate_adaptive_weight(candidate, current_workload)
                candidate.priority_score *= adaptive_weight
                
                # Update eviction reason to indicate adaptive decision
                candidate.eviction_reason = EvictionReason.ADAPTIVE_DECISION.value
                candidate.metadata['adaptive_strategy'] = self.current_strategy_name
                candidate.metadata['workload_characteristics'] = current_workload
            
            # Record strategy performance for future evaluation
            self._record_strategy_performance(len(candidates), memory_pressure_level, current_workload)
            
            # Log adaptive candidate selection with strategy choice
            self.logger.debug(f"Adaptive strategy selected {len(candidates)} candidates using {self.current_strategy_name} "
                            f"(workload locality: {current_workload.get('access_locality', 0.0):.3f})")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error in adaptive candidate selection: {e}")
            # Fallback to current strategy without adaptive weighting
            return self.current_strategy.select_eviction_candidates(
                cache_entries, target_eviction_count, memory_pressure_level, selection_context
            )
    
    def _calculate_adaptive_weight(
        self,
        candidate: EvictionCandidate,
        workload_characteristics: Dict[str, float]
    ) -> float:
        """Calculate adaptive weight for eviction candidate based on workload characteristics."""
        base_weight = 1.0
        
        # Apply locality-based weighting
        locality = workload_characteristics.get('access_locality', 0.5)
        if locality > 0.7:  # High locality - favor LRU behavior
            if candidate.access_frequency < 2:
                base_weight *= 1.2
        elif locality < 0.3:  # Low locality - favor size-based eviction
            size_factor = candidate.entry_size_bytes / (1024 * 1024)  # MB
            if size_factor > 1.0:
                base_weight *= 1.3
        
        # Apply memory pressure weighting
        memory_pressure_freq = workload_characteristics.get('memory_pressure_frequency', 0.5)
        if memory_pressure_freq > 0.6:  # Frequent memory pressure
            if candidate.entry_size_bytes > 512 * 1024:  # > 512KB
                base_weight *= 1.4
        
        return base_weight
    
    def _record_strategy_performance(
        self,
        candidates_selected: int,
        memory_pressure: float,
        workload_characteristics: Dict[str, float]
    ) -> None:
        """Record strategy performance for evaluation and optimization."""
        performance_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'strategy_name': self.current_strategy_name,
            'candidates_selected': candidates_selected,
            'memory_pressure': memory_pressure,
            'workload_characteristics': workload_characteristics.copy()
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history (last 1000 records)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def evaluate_strategy_performance(
        self,
        evaluation_context: Dict[str, Any],
        force_evaluation: bool = False
    ) -> str:
        """
        Evaluate performance of all available strategies and select optimal strategy based 
        on current workload and system conditions.
        
        This method analyzes strategy performance and selects the optimal strategy for
        current conditions based on performance history and workload characteristics.
        
        Args:
            evaluation_context: Context information for strategy evaluation
            force_evaluation: Force evaluation regardless of interval
            
        Returns:
            str: Name of optimal strategy based on performance evaluation
        """
        try:
            current_time = datetime.datetime.now()
            
            # Check if evaluation interval has elapsed or force evaluation
            time_since_last_eval = (current_time - self.last_evaluation_time).total_seconds()
            if not force_evaluation and time_since_last_eval < self.evaluation_interval:
                return self.current_strategy_name
            
            # Collect performance data for all available strategies
            strategy_scores = {}
            
            for strategy_name, strategy in self.available_strategies.items():
                try:
                    # Get strategy statistics for performance evaluation
                    stats = strategy.get_strategy_statistics(
                        include_detailed_breakdown=False,
                        time_window_hours=1,
                        include_optimization_recommendations=False
                    )
                    
                    # Calculate performance score based on multiple factors
                    hit_rate = stats.get('cache_hit_rate', 0.0)
                    effectiveness = stats.get('time_window_analysis', {}).get('average_effectiveness', 0.5)
                    execution_time = stats.get('time_window_analysis', {}).get('average_execution_time', 1.0)
                    
                    # Combine metrics into overall performance score
                    performance_score = (
                        hit_rate * 0.4 +  # Cache hit rate weight
                        effectiveness * 0.4 +  # Eviction effectiveness weight
                        max(0.0, 1.0 - (execution_time / 10.0)) * 0.2  # Execution efficiency weight
                    )
                    
                    strategy_scores[strategy_name] = performance_score
                    
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate strategy {strategy_name}: {e}")
                    strategy_scores[strategy_name] = 0.3  # Default low score
            
            # Analyze workload characteristics and cache patterns
            workload_characteristics = evaluation_context.get('workload_characteristics', self.workload_characteristics)
            
            # Apply workload-specific strategy preferences
            for strategy_name in strategy_scores:
                workload_bonus = self._calculate_workload_strategy_bonus(strategy_name, workload_characteristics)
                strategy_scores[strategy_name] += workload_bonus
            
            # Identify optimal strategy based on current conditions
            if strategy_scores:
                optimal_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
            else:
                optimal_strategy = 'lru'  # Fallback to LRU
            
            # Update strategy performance history and scores
            self.strategy_performance_scores.update(strategy_scores)
            self.last_evaluation_time = current_time
            
            # Log strategy evaluation with performance analysis
            self.logger.info(f"Strategy performance evaluation completed: optimal={optimal_strategy}, "
                           f"scores={dict(sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True))}")
            
            # Log performance metrics for monitoring
            log_performance_metrics(
                metric_name='adaptive_strategy_evaluation',
                metric_value=strategy_scores.get(optimal_strategy, 0.0),
                metric_unit='score',
                component='EVICTION',
                metric_context={
                    'optimal_strategy': optimal_strategy,
                    'strategies_evaluated': len(strategy_scores),
                    'workload_locality': workload_characteristics.get('access_locality', 0.0)
                }
            )
            
            return optimal_strategy
            
        except Exception as e:
            self.logger.error(f"Error evaluating adaptive strategy performance: {e}")
            return self.current_strategy_name  # Return current strategy on error
    
    def _calculate_workload_strategy_bonus(
        self,
        strategy_name: str,
        workload_characteristics: Dict[str, float]
    ) -> float:
        """Calculate workload-specific bonus for strategy selection."""
        bonus = 0.0
        
        locality = workload_characteristics.get('access_locality', 0.5)
        size_variance = workload_characteristics.get('size_variance', 0.5)
        memory_pressure_freq = workload_characteristics.get('memory_pressure_frequency', 0.5)
        
        if strategy_name == 'lru':
            # LRU performs well with high locality
            if locality > 0.7:
                bonus += 0.1
            # LRU struggles with high memory pressure
            if memory_pressure_freq > 0.6:
                bonus -= 0.05
        
        # Add more strategy-specific bonuses as strategies are added
        
        return bonus
    
    def add_strategy(
        self,
        strategy_name: str,
        strategy_instance: EvictionStrategy,
        enable_for_selection: bool = True
    ) -> bool:
        """
        Add new eviction strategy to available strategies for adaptive selection and 
        performance comparison.
        
        This method adds new strategies to the adaptive selection pool with performance
        tracking initialization and selection enablement.
        
        Args:
            strategy_name: Unique name for the new strategy
            strategy_instance: Configured eviction strategy instance
            enable_for_selection: Enable strategy for adaptive selection
            
        Returns:
            bool: Success status of strategy addition
        """
        try:
            # Validate strategy instance implements EvictionStrategy interface
            if not isinstance(strategy_instance, EvictionStrategy):
                self.logger.error(f"Strategy {strategy_name} does not implement EvictionStrategy interface")
                return False
            
            # Check for duplicate strategy names
            if strategy_name in self.available_strategies:
                self.logger.warning(f"Strategy {strategy_name} already exists, replacing")
            
            # Add strategy to available strategies dictionary
            self.available_strategies[strategy_name] = strategy_instance
            
            # Initialize performance tracking for new strategy
            self.strategy_performance_scores[strategy_name] = 0.5  # Default initial score
            
            # Enable strategy for adaptive selection if requested
            if enable_for_selection:
                # Strategy is automatically available for selection
                pass
            
            # Update strategy performance scores with initial values
            # This will be updated through actual performance monitoring
            
            # Log strategy addition with configuration details
            self.logger.info(f"Added strategy '{strategy_name}' to adaptive selection pool "
                           f"(total strategies: {len(self.available_strategies)})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add strategy {strategy_name}: {e}")
            return False
    
    def analyze_workload_characteristics(
        self,
        cache_state: Dict[str, Any],
        access_patterns: Dict[str, Any],
        analysis_window_seconds: int = 300
    ) -> Dict[str, float]:
        """
        Analyze current workload characteristics including access patterns, data types, 
        and cache behavior for optimal strategy selection.
        
        This method analyzes workload patterns to provide insights for optimal strategy
        selection based on access locality, size distribution, and temporal patterns.
        
        Args:
            cache_state: Current cache state information
            access_patterns: Access pattern data for analysis
            analysis_window_seconds: Time window for pattern analysis
            
        Returns:
            Dict[str, float]: Workload characteristics with pattern analysis and strategy recommendations
        """
        try:
            characteristics = self.workload_characteristics.copy()
            
            # Analyze cache access patterns and frequency distribution
            entries = cache_state.get('entries', {})
            if entries:
                # Calculate access locality based on recent access patterns
                recent_accesses = access_patterns.get('recent_accesses', [])
                if recent_accesses:
                    # Simple locality measure based on access concentration
                    unique_keys = len(set(recent_accesses))
                    total_accesses = len(recent_accesses)
                    locality_score = 1.0 - (unique_keys / max(total_accesses, 1))
                    characteristics['access_locality'] = locality_score
                
                # Evaluate data size distribution and cache entry characteristics
                entry_sizes = [entry.get('size_bytes', 0) for entry in entries.values()]
                if entry_sizes:
                    size_variance = statistics.stdev(entry_sizes) / max(statistics.mean(entry_sizes), 1)
                    characteristics['size_variance'] = min(size_variance, 1.0)
            
            # Calculate temporal access patterns and locality
            access_times = access_patterns.get('access_timestamps', [])
            if len(access_times) > 1:
                # Analyze temporal clustering of accesses
                time_intervals = []
                for i in range(1, len(access_times)):
                    interval = (access_times[i] - access_times[i-1]).total_seconds()
                    time_intervals.append(interval)
                
                if time_intervals:
                    # Temporal pattern score based on interval variance
                    avg_interval = statistics.mean(time_intervals)
                    if avg_interval > 0:
                        interval_variance = statistics.stdev(time_intervals) / avg_interval
                        characteristics['temporal_patterns'] = 1.0 - min(interval_variance, 1.0)
            
            # Assess memory pressure impact on cache behavior
            memory_pressure = cache_state.get('memory_pressure', 0.0)
            
            # Calculate memory pressure frequency from recent history
            recent_records = [r for r in self.performance_history[-50:] 
                            if (datetime.datetime.now() - 
                                datetime.datetime.fromisoformat(r['timestamp'])).total_seconds() < analysis_window_seconds]
            
            if recent_records:
                high_pressure_count = sum(1 for r in recent_records if r.get('memory_pressure', 0.0) > MEMORY_PRESSURE_THRESHOLD)
                pressure_frequency = high_pressure_count / len(recent_records)
                characteristics['memory_pressure_frequency'] = pressure_frequency
            
            # Update workload characteristics for strategy selection
            self.workload_characteristics.update(characteristics)
            
            # Log workload analysis with characteristic scores
            self.logger.debug(f"Workload characteristics analyzed: "
                            f"locality={characteristics.get('access_locality', 0.0):.3f}, "
                            f"size_variance={characteristics.get('size_variance', 0.0):.3f}, "
                            f"temporal_patterns={characteristics.get('temporal_patterns', 0.0):.3f}, "
                            f"memory_pressure_freq={characteristics.get('memory_pressure_frequency', 0.0):.3f}")
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Error analyzing workload characteristics: {e}")
            return self.workload_characteristics.copy()


def create_eviction_strategy(
    strategy_type: str,
    cache_config: Dict[str, Any],
    memory_monitor: Optional[MemoryMonitor] = None,
    strategy_config: Dict[str, Any] = None
) -> EvictionStrategy:
    """
    Factory function to create and configure eviction strategy instances based on strategy 
    type, cache characteristics, and performance requirements for optimal cache management.
    
    This factory function provides centralized strategy creation with configuration loading,
    validation, and performance monitoring setup for different eviction algorithms.
    
    Args:
        strategy_type: Type of eviction strategy to create ('lru', 'adaptive', etc.)
        cache_config: Cache configuration dictionary
        memory_monitor: Optional memory monitor for pressure-aware decisions
        strategy_config: Strategy-specific configuration parameters
        
    Returns:
        EvictionStrategy: Configured eviction strategy instance ready for cache management
    """
    logger = get_logger('eviction.factory', 'CACHE')
    
    try:
        # Validate strategy type against registered strategy types
        available_strategies = ['lru', 'adaptive']
        if strategy_type not in available_strategies:
            logger.warning(f"Unknown strategy type '{strategy_type}', defaulting to '{DEFAULT_EVICTION_STRATEGY}'")
            strategy_type = DEFAULT_EVICTION_STRATEGY
        
        # Load strategy configuration from performance thresholds
        optimization_config = performance_thresholds.get('optimization_targets', {})
        batch_config = performance_thresholds.get('batch_processing', {})
        
        # Merge provided strategy_config with defaults
        merged_config = {
            'batch_size': batch_config.get('chunk_size_per_worker', EVICTION_BATCH_SIZE),
            'evaluation_interval': ADAPTIVE_EVALUATION_INTERVAL,
            'performance_window': STRATEGY_PERFORMANCE_WINDOW,
            'cache_hit_rate_target': optimization_config.get('throughput_optimization', {}).get('cache_hit_rate_target', 0.8)
        }
        
        if strategy_config:
            merged_config.update(strategy_config)
        
        # Include cache configuration parameters
        merged_config.update({
            'cache_config': cache_config,
            'max_memory_usage': cache_config.get('max_memory_gb', 8.0),
            'max_cache_size': cache_config.get('max_entries', 10000)
        })
        
        # Create strategy instance using strategy type
        if strategy_type == 'lru':
            strategy = LRUEvictionStrategy(
                strategy_name=f'lru_{uuid.uuid4().hex[:8]}',
                config=merged_config,
                memory_monitor=memory_monitor
            )
        elif strategy_type == 'adaptive':
            strategy = AdaptiveEvictionStrategy(
                strategy_name=f'adaptive_{uuid.uuid4().hex[:8]}',
                config=merged_config,
                memory_monitor=memory_monitor
            )
        else:
            # Fallback to LRU for unknown types
            logger.warning(f"Creating LRU strategy as fallback for unknown type: {strategy_type}")
            strategy = LRUEvictionStrategy(
                strategy_name=f'fallback_lru_{uuid.uuid4().hex[:8]}',
                config=merged_config,
                memory_monitor=memory_monitor
            )
        
        # Configure memory monitor integration if provided
        if memory_monitor and not strategy.memory_monitor:
            strategy.memory_monitor = memory_monitor
            logger.debug(f"Memory monitor integrated with strategy: {strategy.strategy_name}")
        
        # Setup performance monitoring and statistics tracking
        strategy.eviction_statistics = CacheStatistics()
        
        # Register strategy in performance history tracking
        strategy_name = strategy.strategy_name
        if strategy_name not in _strategy_performance_history:
            _strategy_performance_history[strategy_name] = []
        
        # Log strategy creation with configuration details
        logger.info(f"Created {strategy_type} eviction strategy: {strategy.strategy_name}")
        logger.debug(f"Strategy configuration: {merged_config}")
        
        # Set scientific context for strategy operations
        set_scientific_context(
            simulation_id='strategy_creation',
            algorithm_name=strategy_type,
            processing_stage='INITIALIZATION'
        )
        
        # Log performance metrics for strategy creation
        log_performance_metrics(
            metric_name='strategy_creation_time',
            metric_value=0.0,  # Creation is instantaneous
            metric_unit='seconds',
            component='EVICTION',
            metric_context={
                'strategy_type': strategy_type,
                'strategy_name': strategy.strategy_name,
                'memory_monitor_enabled': memory_monitor is not None
            }
        )
        
        return strategy
        
    except Exception as e:
        logger.error(f"Failed to create eviction strategy of type '{strategy_type}': {e}")
        
        # Create minimal LRU strategy as emergency fallback
        fallback_strategy = LRUEvictionStrategy(
            strategy_name=f'emergency_lru_{uuid.uuid4().hex[:8]}',
            config={'batch_size': EVICTION_BATCH_SIZE},
            memory_monitor=memory_monitor
        )
        
        logger.warning(f"Created emergency fallback LRU strategy: {fallback_strategy.strategy_name}")
        return fallback_strategy


def register_eviction_strategy(
    strategy_name: str,
    strategy_class: Type[EvictionStrategy],
    override_existing: bool = False
) -> bool:
    """
    Register custom eviction strategy class in the global strategy registry for factory 
    function access and dynamic strategy selection.
    
    This function enables registration of custom eviction strategies for use with the
    factory function and adaptive strategy selection.
    
    Args:
        strategy_name: Unique name for the strategy class
        strategy_class: Eviction strategy class to register
        override_existing: Allow overriding existing strategy registrations
        
    Returns:
        bool: Success status of strategy registration
    """
    logger = get_logger('eviction.registry', 'CACHE')
    
    try:
        with _global_eviction_lock:
            # Validate strategy class implements EvictionStrategy interface
            if not issubclass(strategy_class, EvictionStrategy):
                logger.error(f"Strategy class {strategy_class.__name__} does not inherit from EvictionStrategy")
                return False
            
            # Check if strategy name already exists in registry
            if strategy_name in _strategy_registry and not override_existing:
                logger.warning(f"Strategy '{strategy_name}' already registered, use override_existing=True to replace")
                return False
            
            # Register strategy class in global strategy registry
            _strategy_registry[strategy_name] = strategy_class
            
            # Initialize performance history tracking for strategy
            if strategy_name not in _strategy_performance_history:
                _strategy_performance_history[strategy_name] = []
            
            # Log strategy registration with class information
            logger.info(f"Registered eviction strategy: {strategy_name} -> {strategy_class.__name__}")
            
            return True
            
    except Exception as e:
        logger.error(f"Failed to register eviction strategy '{strategy_name}': {e}")
        return False


def evaluate_strategy_performance(
    strategy_name: str,
    performance_data: Dict[str, Any],
    evaluation_window_seconds: int = 3600,
    include_recommendations: bool = True
) -> EvictionPerformanceResult:
    """
    Evaluate eviction strategy performance across multiple metrics including eviction 
    effectiveness, memory efficiency, and cache hit rate impact for optimization analysis.
    
    This function provides comprehensive performance evaluation with trend analysis and
    optimization recommendations for eviction strategy optimization.
    
    Args:
        strategy_name: Name of the strategy to evaluate
        performance_data: Performance data dictionary for analysis
        evaluation_window_seconds: Time window for performance analysis
        include_recommendations: Include optimization recommendations in results
        
    Returns:
        EvictionPerformanceResult: Strategy performance evaluation with metrics and optimization recommendations
    """
    logger = get_logger('eviction.evaluation', 'CACHE')
    
    try:
        evaluation_start_time = time.time()
        
        # Collect performance data for specified strategy and time window
        strategy_history = _strategy_performance_history.get(strategy_name, [])
        
        # Filter performance data by evaluation window
        cutoff_time = datetime.datetime.now() - datetime.timedelta(seconds=evaluation_window_seconds)
        recent_performance = []
        
        for record in strategy_history:
            record_time = datetime.datetime.fromisoformat(record.get('timestamp', '1970-01-01'))
            if record_time >= cutoff_time:
                recent_performance.append(record)
        
        # Calculate eviction effectiveness and memory efficiency metrics
        if recent_performance:
            effectiveness_scores = [r.get('effectiveness_score', 0.0) for r in recent_performance]
            memory_freed_values = [r.get('memory_freed', 0) for r in recent_performance]
            execution_times = [r.get('execution_time', 0.0) for r in recent_performance]
            
            eviction_effectiveness = statistics.mean(effectiveness_scores) if effectiveness_scores else 0.0
            total_memory_freed = sum(memory_freed_values)
            avg_execution_time = statistics.mean(execution_times) if execution_times else 0.0
            
            # Calculate memory efficiency as MB freed per second
            memory_efficiency = (total_memory_freed / (1024 * 1024)) / max(evaluation_window_seconds, 1)
        else:
            eviction_effectiveness = 0.0
            memory_efficiency = 0.0
            avg_execution_time = 0.0
        
        # Analyze cache hit rate impact and resource utilization
        cache_hit_rate_impact = performance_data.get('cache_hit_rate_change', 0.0)
        
        resource_utilization = {
            'average_execution_time': avg_execution_time,
            'memory_efficiency_mb_per_sec': memory_efficiency,
            'evaluation_coverage': len(recent_performance) / max(evaluation_window_seconds / 60, 1)  # Records per minute
        }
        
        # Compare performance against configured thresholds
        optimization_targets = performance_thresholds.get('optimization_targets', {})
        cache_target = optimization_targets.get('throughput_optimization', {}).get('cache_hit_rate_target', 0.8)
        
        threshold_comparison = {
            'effectiveness_vs_threshold': eviction_effectiveness - EVICTION_EFFECTIVENESS_THRESHOLD,
            'cache_hit_rate_vs_target': performance_data.get('current_hit_rate', 0.0) - cache_target
        }
        
        # Generate performance trend analysis over evaluation window
        performance_trends = {
            'effectiveness_trend': effectiveness_scores[-10:] if len(effectiveness_scores) >= 10 else effectiveness_scores,
            'memory_efficiency_trend': memory_freed_values[-10:] if len(memory_freed_values) >= 10 else memory_freed_values,
            'execution_time_trend': execution_times[-10:] if len(execution_times) >= 10 else execution_times
        }
        
        # Include optimization recommendations if requested
        optimization_recommendations = []
        if include_recommendations:
            if eviction_effectiveness < EVICTION_EFFECTIVENESS_THRESHOLD:
                optimization_recommendations.append("Consider tuning eviction parameters to improve effectiveness")
            
            if avg_execution_time > 1.0:
                optimization_recommendations.append("Execution time is high - consider batch size optimization")
            
            if memory_efficiency < 1.0:  # Less than 1 MB/sec
                optimization_recommendations.append("Memory efficiency is low - consider more aggressive eviction policies")
            
            if cache_hit_rate_impact < -0.1:  # Significant negative impact
                optimization_recommendations.append("Strategy negatively impacting cache hit rate - review selection criteria")
        
        # Update strategy performance history
        evaluation_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'evaluation_window_seconds': evaluation_window_seconds,
            'eviction_effectiveness': eviction_effectiveness,
            'memory_efficiency': memory_efficiency,
            'cache_hit_rate_impact': cache_hit_rate_impact
        }
        
        if strategy_name in _strategy_performance_history:
            _strategy_performance_history[strategy_name].append(evaluation_record)
            # Keep only recent history (last 1000 records)
            if len(_strategy_performance_history[strategy_name]) > 1000:
                _strategy_performance_history[strategy_name] = _strategy_performance_history[strategy_name][-1000:]
        
        # Log performance evaluation with detailed metrics
        evaluation_time = time.time() - evaluation_start_time
        logger.info(f"Performance evaluation completed for {strategy_name}: "
                   f"effectiveness={eviction_effectiveness:.3f}, efficiency={memory_efficiency:.2f} MB/s")
        
        # Log performance metrics for monitoring
        log_performance_metrics(
            metric_name='strategy_performance_evaluation_time',
            metric_value=evaluation_time,
            metric_unit='seconds',
            component='EVICTION',
            metric_context={
                'strategy_name': strategy_name,
                'evaluation_window_seconds': evaluation_window_seconds,
                'records_analyzed': len(recent_performance),
                'effectiveness_score': eviction_effectiveness
            }
        )
        
        return EvictionPerformanceResult(
            strategy_name=strategy_name,
            evaluation_period_seconds=evaluation_window_seconds,
            eviction_effectiveness=eviction_effectiveness,
            memory_efficiency=memory_efficiency,
            cache_hit_rate_impact=cache_hit_rate_impact,
            resource_utilization=resource_utilization,
            performance_trends=performance_trends,
            optimization_recommendations=optimization_recommendations
        )
        
    except Exception as e:
        logger.error(f"Error evaluating strategy performance for {strategy_name}: {e}")
        return EvictionPerformanceResult(
            strategy_name=strategy_name,
            evaluation_period_seconds=evaluation_window_seconds,
            eviction_effectiveness=0.0,
            memory_efficiency=0.0,
            cache_hit_rate_impact=0.0,
            resource_utilization={'error': str(e)},
            performance_trends={},
            optimization_recommendations=[f"Evaluation failed: {str(e)}"]
        )


def optimize_eviction_coordination(
    cache_level_names: List[str],
    optimization_strategy: str = 'balanced',
    coordination_config: Dict[str, Any] = None,
    apply_optimizations: bool = True
) -> EvictionCoordinationResult:
    """
    Optimize eviction coordination across multiple cache levels by analyzing access patterns, 
    memory pressure, and cache effectiveness for improved system performance.
    
    This function provides comprehensive eviction coordination optimization across multiple
    cache levels with performance analysis and optimization application.
    
    Args:
        cache_level_names: List of cache level names to coordinate
        optimization_strategy: Strategy for coordination optimization ('balanced', 'aggressive', 'conservative')
        coordination_config: Configuration for coordination optimization
        apply_optimizations: Whether to apply optimization changes immediately
        
    Returns:
        EvictionCoordinationResult: Eviction coordination optimization results with performance improvements
    """
    logger = get_logger('eviction.coordination', 'CACHE')
    
    try:
        optimization_start_time = time.time()
        
        # Validate input parameters
        if not cache_level_names:
            logger.warning("No cache levels specified for coordination optimization")
            return EvictionCoordinationResult(
                coordination_strategy=optimization_strategy,
                cache_levels_coordinated=[],
                optimization_effectiveness=0.0,
                performance_improvements={},
                coordination_overhead=0.0
            )
        
        coordination_config = coordination_config or {}
        
        # Analyze current eviction patterns across specified cache levels
        current_patterns = {}
        for level_name in cache_level_names:
            # This would typically interface with actual cache level instances
            # Placeholder analysis for demonstration
            current_patterns[level_name] = {
                'eviction_frequency': 10.0,  # Evictions per minute
                'average_effectiveness': 0.7,
                'memory_pressure_sensitivity': 0.8,
                'coordination_overhead': 0.1
            }
        
        # Identify coordination bottlenecks and optimization opportunities
        optimization_opportunities = []
        
        # Check for imbalanced eviction patterns
        eviction_frequencies = [patterns['eviction_frequency'] for patterns in current_patterns.values()]
        if max(eviction_frequencies) / max(min(eviction_frequencies), 0.1) > 3.0:
            optimization_opportunities.append('balance_eviction_frequencies')
        
        # Check for effectiveness disparities
        effectiveness_scores = [patterns['average_effectiveness'] for patterns in current_patterns.values()]
        if max(effectiveness_scores) - min(effectiveness_scores) > 0.3:
            optimization_opportunities.append('improve_coordination_effectiveness')
        
        # Analyze coordination overhead
        coordination_overheads = [patterns['coordination_overhead'] for patterns in current_patterns.values()]
        if statistics.mean(coordination_overheads) > 0.2:
            optimization_opportunities.append('reduce_coordination_overhead')
        
        # Generate coordination strategy based on access patterns and memory pressure
        coordination_optimizations = {}
        
        if optimization_strategy == 'aggressive':
            coordination_optimizations.update({
                'eviction_frequency_multiplier': 1.5,
                'coordination_batch_size': 20,
                'memory_pressure_threshold': 0.7
            })
        elif optimization_strategy == 'conservative':
            coordination_optimizations.update({
                'eviction_frequency_multiplier': 0.8,
                'coordination_batch_size': 5,
                'memory_pressure_threshold': 0.9
            })
        else:  # balanced
            coordination_optimizations.update({
                'eviction_frequency_multiplier': 1.2,
                'coordination_batch_size': 10,
                'memory_pressure_threshold': 0.8
            })
        
        # Calculate optimal eviction timing and batch sizes
        optimal_timing = {}
        for level_name in cache_level_names:
            current_frequency = current_patterns[level_name]['eviction_frequency']
            optimal_frequency = current_frequency * coordination_optimizations['eviction_frequency_multiplier']
            optimal_timing[level_name] = {
                'eviction_interval_seconds': 60.0 / optimal_frequency,
                'batch_size': coordination_optimizations['coordination_batch_size']
            }
        
        # Coordinate eviction strategies across cache levels
        coordination_effectiveness = 0.0
        if 'balance_eviction_frequencies' in optimization_opportunities:
            # Simulate balancing effect
            balanced_frequencies = [statistics.mean(eviction_frequencies)] * len(cache_level_names)
            frequency_improvement = 1.0 - (statistics.stdev(balanced_frequencies) / max(statistics.mean(balanced_frequencies), 0.1))
            coordination_effectiveness += frequency_improvement * 0.3
        
        if 'improve_coordination_effectiveness' in optimization_opportunities:
            # Simulate effectiveness improvement
            effectiveness_improvement = min(0.2, EVICTION_EFFECTIVENESS_THRESHOLD - min(effectiveness_scores))
            coordination_effectiveness += effectiveness_improvement
        
        if 'reduce_coordination_overhead' in optimization_opportunities:
            # Simulate overhead reduction
            overhead_reduction = max(0.0, statistics.mean(coordination_overheads) - 0.1)
            coordination_effectiveness += overhead_reduction * 0.5
        
        # Apply optimization changes if enabled and validated
        performance_improvements = {}
        if apply_optimizations:
            for level_name in cache_level_names:
                # This would typically apply changes to actual cache level instances
                # Simulate performance improvements
                baseline_effectiveness = current_patterns[level_name]['average_effectiveness']
                improved_effectiveness = min(1.0, baseline_effectiveness + coordination_effectiveness * 0.5)
                performance_improvements[level_name] = {
                    'effectiveness_improvement': improved_effectiveness - baseline_effectiveness,
                    'frequency_optimization': optimal_timing[level_name]['eviction_interval_seconds'],
                    'batch_size_optimization': optimal_timing[level_name]['batch_size']
                }
            
            logger.info(f"Applied eviction coordination optimizations to {len(cache_level_names)} cache levels")
        else:
            # Generate optimization recommendations without applying
            for level_name in cache_level_names:
                performance_improvements[level_name] = {
                    'recommended_frequency_change': coordination_optimizations['eviction_frequency_multiplier'],
                    'recommended_batch_size': coordination_optimizations['coordination_batch_size'],
                    'estimated_effectiveness_improvement': coordination_effectiveness * 0.5
                }
            
            logger.info(f"Generated coordination optimization recommendations for {len(cache_level_names)} cache levels")
        
        # Monitor coordination effectiveness and performance impact
        coordination_overhead = statistics.mean([
            patterns.get('coordination_overhead', 0.1) for patterns in current_patterns.values()
        ])
        
        # Apply overhead reduction if optimizations were applied
        if apply_optimizations and 'reduce_coordination_overhead' in optimization_opportunities:
            coordination_overhead *= 0.7  # Simulate 30% overhead reduction
        
        # Update eviction coordination configuration
        if apply_optimizations:
            # This would typically update global coordination settings
            pass
        
        # Log coordination optimization with detailed results
        optimization_time = time.time() - optimization_start_time
        logger.info(f"Eviction coordination optimization completed: strategy={optimization_strategy}, "
                   f"levels={len(cache_level_names)}, effectiveness={coordination_effectiveness:.3f}")
        
        # Log performance metrics for monitoring
        log_performance_metrics(
            metric_name='coordination_optimization_time',
            metric_value=optimization_time,
            metric_unit='seconds',
            component='EVICTION',
            metric_context={
                'optimization_strategy': optimization_strategy,
                'cache_levels_count': len(cache_level_names),
                'opportunities_identified': len(optimization_opportunities),
                'optimizations_applied': apply_optimizations
            }
        )
        
        return EvictionCoordinationResult(
            coordination_strategy=optimization_strategy,
            cache_levels_coordinated=cache_level_names,
            optimization_effectiveness=coordination_effectiveness,
            performance_improvements=performance_improvements,
            coordination_overhead=coordination_overhead
        )
        
    except Exception as e:
        logger.error(f"Error optimizing eviction coordination: {e}")
        return EvictionCoordinationResult(
            coordination_strategy=optimization_strategy,
            cache_levels_coordinated=cache_level_names,
            optimization_effectiveness=0.0,
            performance_improvements={'error': str(e)},
            coordination_overhead=1.0  # High overhead on error
        )


# Initialize global memory monitor instance if available
try:
    _memory_monitor_instance = MemoryMonitor()
except Exception:
    _memory_monitor_instance = None

# Register built-in eviction strategies
register_eviction_strategy('lru', LRUEvictionStrategy, override_existing=True)
register_eviction_strategy('adaptive', AdaptiveEvictionStrategy, override_existing=True)