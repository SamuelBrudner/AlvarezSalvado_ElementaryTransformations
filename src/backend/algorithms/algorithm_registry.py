"""
Comprehensive algorithm registry module providing centralized algorithm discovery, registration, 
instantiation, and management for the plume simulation system.

This module implements dynamic algorithm loading, metadata management, compatibility validation, 
performance tracking, and cross-format algorithm coordination with scientific computing standards 
compliance. Supports registration of navigation algorithms including infotaxis, casting, gradient 
following, plume tracking, hybrid strategies, and reference implementations with comprehensive 
validation, audit trail integration, and reproducibility assessment for >95% correlation 
requirements and >0.99 reproducibility coefficients.

Key Features:
- Centralized algorithm registry with thread-safe operations
- Dynamic algorithm loading with caching and validation
- Performance tracking with <7.2 seconds target execution time
- Cross-platform compatibility for different plume formats
- Comprehensive validation and audit trail integration
- Algorithm discovery and metadata management
- Batch processing support for 4000+ simulation requirements
- Scientific computing standards compliance and reproducibility assessment
"""

# External imports with version specifications
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Type  # Python 3.9+ - Type hints for complex data structures
from abc import ABC, abstractmethod  # Python 3.9+ - Abstract base class functionality for algorithm interface validation
import inspect  # Python 3.9+ - Runtime inspection of algorithm classes and methods for validation
import importlib  # Python 3.9+ - Dynamic module loading for algorithm discovery and registration
import threading  # Python 3.9+ - Thread-safe algorithm registry operations and concurrent access management
import copy  # Python 3.9+ - Deep copying for algorithm instance isolation and parameter preservation
import datetime  # Python 3.9+ - Timestamp generation for algorithm registration tracking and audit trails
import json  # Python 3.9+ - JSON serialization for algorithm metadata and configuration management
import pathlib  # Python 3.9+ - Path handling for algorithm module discovery and configuration files
import collections  # Python 3.9+ - Efficient data structures for algorithm registry and metadata management
import weakref  # Python 3.9+ - Weak references for algorithm instance management and memory optimization
import dataclasses  # Python 3.9+ - Data classes for algorithm registry entries and metadata
import uuid  # Python 3.9+ - Unique identifier generation for algorithm instances and correlation tracking
import time  # Python 3.9+ - Performance timing for algorithm execution monitoring

# Internal imports from base algorithm framework
from .base_algorithm import (
    BaseAlgorithm, AlgorithmParameters, AlgorithmResult, AlgorithmContext,
    validate_plume_data, create_algorithm_context, calculate_performance_metrics
)
from .reference_implementation import (
    ReferenceImplementation, validate_against_benchmark, 
    calculate_reproducibility_metrics, generate_benchmark_report
)
from .infotaxis import InfotaxisAlgorithm
from .casting import CastingAlgorithm
from .gradient_following import GradientFollowing

# Internal imports from utility modules
from ..utils.logging_utils import (
    get_logger, set_scientific_context, log_simulation_event, create_audit_trail
)
from ..error.exceptions import (
    PlumeSimulationException, ValidationError, ConfigurationError
)

# Global registry for algorithm storage and thread-safe access
_global_algorithm_registry: Dict[str, 'AlgorithmRegistryEntry'] = {}
_algorithm_metadata_cache: Dict[str, Dict[str, Any]] = {}
_algorithm_performance_cache: Dict[str, Dict[str, float]] = {}
_registry_lock: threading.RLock = threading.RLock()
_algorithm_instances: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_dynamic_algorithm_cache: Dict[str, type] = {}

# Algorithm registry constants and configuration
SUPPORTED_ALGORITHM_TYPES = [
    'reference_implementation', 'infotaxis', 'casting', 'gradient_following', 
    'plume_tracking', 'hybrid_strategies'
]
ALGORITHM_INTERFACE_VERSION = '1.0.0'
REGISTRY_VERSION = '1.0.0'
DEFAULT_VALIDATION_TIMEOUT = 30.0
PERFORMANCE_CACHE_TTL = 3600.0
MAX_ALGORITHM_INSTANCES = 100
CORRELATION_THRESHOLD = 0.95
REPRODUCIBILITY_THRESHOLD = 0.99
DYNAMIC_LOADING_ENABLED = True

# Dynamic algorithm module paths for circular dependency avoidance
ALGORITHM_MODULE_PATHS = {
    'plume_tracking': 'src.backend.algorithms.plume_tracking',
    'hybrid_strategies': 'src.backend.algorithms.hybrid_strategies'
}


@dataclasses.dataclass
class AlgorithmRegistryEntry:
    """
    Data class representing a single algorithm registry entry with comprehensive metadata, 
    performance tracking, validation status, and audit trail information for complete 
    algorithm lifecycle management in the registry system.
    
    This class provides complete algorithm registration information with performance tracking,
    validation status, and audit trail integration for reproducible scientific computing.
    """
    
    # Core algorithm identification and class reference
    algorithm_name: str
    algorithm_class: Type[BaseAlgorithm]
    metadata: Dict[str, Any]
    
    # Algorithm classification and versioning
    algorithm_type: str = dataclasses.field(init=False)
    version: str = dataclasses.field(init=False)
    registration_timestamp: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
    
    # Availability and capability information
    is_available: bool = True
    capabilities: Dict[str, Any] = dataclasses.field(default_factory=dict)
    performance_characteristics: Dict[str, float] = dataclasses.field(default_factory=dict)
    supported_formats: List[str] = dataclasses.field(default_factory=list)
    validation_requirements: Dict[str, Any] = dataclasses.field(default_factory=dict)
    
    # Tracking and audit information
    audit_trail_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    instance_count: int = 0
    last_accessed: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
    dynamically_loaded: bool = False
    
    def __post_init__(self):
        """Initialize algorithm registry entry with metadata extraction, validation setup, and audit trail creation."""
        # Extract algorithm type and version from metadata
        self.algorithm_type = self.metadata.get('algorithm_type', self.algorithm_name)
        self.version = self.metadata.get('version', ALGORITHM_INTERFACE_VERSION)
        
        # Extract capabilities and performance characteristics
        self.capabilities = self.metadata.get('capabilities', {})
        self.performance_characteristics = self.metadata.get('performance_characteristics', {})
        
        # Setup supported formats and validation requirements
        self.supported_formats = self.metadata.get('supported_formats', ['generic'])
        self.validation_requirements = self.metadata.get('validation_requirements', {})
        
        # Set dynamically loaded flag based on loading method
        self.dynamically_loaded = self.metadata.get('dynamically_loaded', False)
        
        # Create initial audit trail entry for algorithm registration
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.debug(f"Algorithm registry entry created: {self.algorithm_name} [{self.algorithm_type}]")
    
    def validate(self, strict_validation: bool = False) -> bool:
        """
        Validate registry entry completeness and consistency with interface compliance checking.
        
        Args:
            strict_validation: Enable strict validation criteria
            
        Returns:
            bool: True if entry is valid with detailed validation status
        """
        try:
            # Validate algorithm name and class consistency
            if not self.algorithm_name or not isinstance(self.algorithm_name, str):
                return False
            
            if not self.algorithm_class or not issubclass(self.algorithm_class, BaseAlgorithm):
                return False
            
            # Check metadata completeness and format
            required_metadata = ['algorithm_type', 'version', 'description']
            for key in required_metadata:
                if key not in self.metadata:
                    if strict_validation:
                        return False
            
            # Validate capabilities and performance characteristics
            if not isinstance(self.capabilities, dict):
                return False
            
            if not isinstance(self.performance_characteristics, dict):
                return False
            
            # Apply strict validation criteria if enabled
            if strict_validation:
                # Verify algorithm class has required methods
                required_methods = ['execute', 'validate_parameters', 'reset']
                for method in required_methods:
                    if not hasattr(self.algorithm_class, method):
                        return False
                
                # Check version format validity
                if not self.version or len(self.version.split('.')) != 3:
                    return False
            
            return True
            
        except Exception as e:
            logger = get_logger('algorithm_registry', 'ALGORITHM')
            logger.error(f"Registry entry validation failed: {e}")
            return False
    
    def update_metadata(self, new_metadata: Dict[str, Any], merge_existing: bool = True) -> None:
        """
        Update registry entry metadata with validation and audit trail creation.
        
        Args:
            new_metadata: New metadata to update
            merge_existing: Whether to merge with existing metadata
        """
        # Validate new metadata format and content
        if not isinstance(new_metadata, dict):
            raise ValueError("Metadata must be a dictionary")
        
        # Merge with existing metadata if requested
        if merge_existing:
            self.metadata.update(new_metadata)
        else:
            self.metadata = new_metadata.copy()
        
        # Update derived fields from metadata
        self.algorithm_type = self.metadata.get('algorithm_type', self.algorithm_name)
        self.version = self.metadata.get('version', ALGORITHM_INTERFACE_VERSION)
        self.capabilities = self.metadata.get('capabilities', {})
        self.performance_characteristics = self.metadata.get('performance_characteristics', {})
        
        # Update last accessed timestamp
        self.last_accessed = datetime.datetime.now()
        
        # Create audit trail entry for metadata update
        create_audit_trail(
            action='ALGORITHM_METADATA_UPDATED',
            component='ALGORITHM_REGISTRY',
            action_details={
                'algorithm_name': self.algorithm_name,
                'metadata_keys': list(new_metadata.keys()),
                'merge_existing': merge_existing
            },
            user_context='SYSTEM'
        )
    
    def increment_instance_count(self) -> int:
        """
        Increment instance count for tracking algorithm usage and resource management.
        
        Returns:
            int: Updated instance count
        """
        # Increment instance count atomically
        self.instance_count += 1
        
        # Update last accessed timestamp
        self.last_accessed = datetime.datetime.now()
        
        # Return updated instance count
        return self.instance_count
    
    def to_dict(self, include_class_reference: bool = False) -> Dict[str, Any]:
        """
        Convert registry entry to dictionary format for serialization and export.
        
        Args:
            include_class_reference: Whether to include algorithm class reference
            
        Returns:
            Dict[str, Any]: Registry entry as dictionary with all properties and metadata
        """
        entry_dict = {
            'algorithm_name': self.algorithm_name,
            'algorithm_type': self.algorithm_type,
            'version': self.version,
            'metadata': self.metadata.copy(),
            'registration_timestamp': self.registration_timestamp.isoformat(),
            'is_available': self.is_available,
            'capabilities': self.capabilities.copy(),
            'performance_characteristics': self.performance_characteristics.copy(),
            'supported_formats': self.supported_formats.copy(),
            'validation_requirements': self.validation_requirements.copy(),
            'audit_trail_id': self.audit_trail_id,
            'instance_count': self.instance_count,
            'last_accessed': self.last_accessed.isoformat(),
            'dynamically_loaded': self.dynamically_loaded
        }
        
        # Include algorithm class reference if requested
        if include_class_reference:
            entry_dict['algorithm_class'] = self.algorithm_class
        
        return entry_dict


class AlgorithmRegistry:
    """
    Comprehensive algorithm registry class providing centralized algorithm management, discovery, 
    instantiation, validation, and performance tracking with thread-safe operations, audit trail 
    integration, dynamic loading support, and scientific computing standards compliance for 
    reproducible research outcomes.
    
    This class serves as the central hub for algorithm management with comprehensive validation,
    performance tracking, and dynamic loading capabilities for scientific computing workflows.
    """
    
    def __init__(
        self,
        enable_performance_tracking: bool = True,
        enable_validation: bool = True,
        registry_name: str = 'default_registry',
        enable_dynamic_loading: bool = True
    ):
        """
        Initialize algorithm registry with performance tracking, validation, dynamic loading, 
        and audit trail setup for comprehensive algorithm management.
        
        Args:
            enable_performance_tracking: Enable performance tracking for algorithms
            enable_validation: Enable validation for algorithm operations
            registry_name: Name identifier for the registry instance
            enable_dynamic_loading: Enable dynamic algorithm loading capability
        """
        # Set registry name and version information
        self.registry_name = registry_name
        self.registry_version = REGISTRY_VERSION
        
        # Enable performance tracking and validation if requested
        self.performance_tracking_enabled = enable_performance_tracking
        self.validation_enabled = enable_validation
        self.dynamic_loading_enabled = enable_dynamic_loading
        
        # Initialize algorithm registry dictionary and caches
        self.algorithms: Dict[str, AlgorithmRegistryEntry] = {}
        self.performance_cache: Dict[str, Dict[str, float]] = {}
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}
        self.dynamic_algorithm_cache: Dict[str, type] = {}
        
        # Create thread-safe registry lock and instance management
        self.registry_lock = threading.RLock()
        self.algorithm_instances: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        
        # Set creation timestamp and initialize logger
        self.creation_timestamp = datetime.datetime.now()
        self.logger = get_logger(f'algorithm_registry.{registry_name}', 'ALGORITHM')
        
        # Initialize registry statistics tracking
        self.registry_statistics = {
            'total_registrations': 0,
            'successful_instantiations': 0,
            'validation_failures': 0,
            'performance_cache_hits': 0,
            'dynamic_loads': 0
        }
        
        # Register built-in algorithms automatically
        self._register_builtin_algorithms()
        
        # Create audit trail entry for registry initialization
        create_audit_trail(
            action='ALGORITHM_REGISTRY_INITIALIZED',
            component='ALGORITHM_REGISTRY',
            action_details={
                'registry_name': registry_name,
                'performance_tracking': enable_performance_tracking,
                'validation_enabled': enable_validation,
                'dynamic_loading': enable_dynamic_loading
            },
            user_context='SYSTEM'
        )
        
        self.logger.info(f"Algorithm registry initialized: {registry_name}")
    
    def register(
        self,
        algorithm_name: str,
        algorithm_class: Type[BaseAlgorithm],
        metadata: Dict[str, Any] = None,
        validate_interface: bool = True
    ) -> bool:
        """
        Register algorithm in registry with comprehensive validation, metadata extraction, 
        and audit trail creation.
        
        Args:
            algorithm_name: Name of the algorithm to register
            algorithm_class: Algorithm class to register
            metadata: Algorithm metadata and configuration
            validate_interface: Whether to validate algorithm interface
            
        Returns:
            bool: True if registration successful with validation status
        """
        with self.registry_lock:
            try:
                # Validate algorithm name and class
                if not algorithm_name or not isinstance(algorithm_name, str):
                    raise ValueError("Algorithm name must be a non-empty string")
                
                if not algorithm_class or not issubclass(algorithm_class, BaseAlgorithm):
                    raise TypeError("Algorithm class must inherit from BaseAlgorithm")
                
                # Check if algorithm already registered
                if algorithm_name in self.algorithms:
                    self.logger.warning(f"Algorithm already registered: {algorithm_name}")
                    return False
                
                # Perform interface validation if enabled
                if validate_interface and self.validation_enabled:
                    validation_result = self.validate_interface(algorithm_class, strict_validation=True)
                    if not validation_result.is_valid:
                        self.registry_statistics['validation_failures'] += 1
                        raise ValidationError(
                            message=f"Algorithm interface validation failed: {algorithm_name}",
                            validation_type="interface_validation",
                            validation_context={'algorithm_name': algorithm_name},
                            failed_parameters=['algorithm_interface']
                        )
                
                # Extract and validate algorithm metadata
                algorithm_metadata = metadata or {}
                if not algorithm_metadata.get('description'):
                    algorithm_metadata['description'] = f"Algorithm: {algorithm_name}"
                if not algorithm_metadata.get('version'):
                    algorithm_metadata['version'] = ALGORITHM_INTERFACE_VERSION
                if not algorithm_metadata.get('algorithm_type'):
                    algorithm_metadata['algorithm_type'] = algorithm_name
                
                # Create algorithm registry entry
                registry_entry = AlgorithmRegistryEntry(
                    algorithm_name=algorithm_name,
                    algorithm_class=algorithm_class,
                    metadata=algorithm_metadata
                )
                
                # Validate registry entry
                if not registry_entry.validate(strict_validation=self.validation_enabled):
                    raise ValidationError(
                        message=f"Registry entry validation failed: {algorithm_name}",
                        validation_type="registry_entry_validation",
                        validation_context={'algorithm_name': algorithm_name}
                    )
                
                # Store algorithm in registry
                self.algorithms[algorithm_name] = registry_entry
                
                # Update metadata cache
                self.metadata_cache[algorithm_name] = algorithm_metadata.copy()
                
                # Update registry statistics
                self.registry_statistics['total_registrations'] += 1
                
                # Create audit trail entry for algorithm registration
                create_audit_trail(
                    action='ALGORITHM_REGISTERED',
                    component='ALGORITHM_REGISTRY',
                    action_details={
                        'algorithm_name': algorithm_name,
                        'registry_name': self.registry_name,
                        'interface_validated': validate_interface,
                        'metadata_keys': list(algorithm_metadata.keys())
                    },
                    user_context='SYSTEM'
                )
                
                self.logger.info(f"Algorithm registered successfully: {algorithm_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Algorithm registration failed: {algorithm_name} - {e}")
                raise
    
    def unregister(self, algorithm_name: str, force_cleanup: bool = False) -> bool:
        """
        Unregister algorithm from registry with cleanup and audit trail creation.
        
        Args:
            algorithm_name: Name of the algorithm to unregister
            force_cleanup: Whether to force cleanup of active instances
            
        Returns:
            bool: True if unregistration successful
        """
        with self.registry_lock:
            try:
                # Validate algorithm exists in registry
                if algorithm_name not in self.algorithms:
                    self.logger.warning(f"Algorithm not found for unregistration: {algorithm_name}")
                    return False
                
                # Check for active algorithm instances
                active_instances = [
                    instance_id for instance_id, instance in self.algorithm_instances.items()
                    if hasattr(instance, 'algorithm_name') and instance.algorithm_name == algorithm_name
                ]
                
                if active_instances and not force_cleanup:
                    self.logger.warning(f"Active instances exist for algorithm: {algorithm_name}")
                    return False
                
                # Remove algorithm from registry
                registry_entry = self.algorithms.pop(algorithm_name)
                
                # Clean up metadata and performance caches
                self.metadata_cache.pop(algorithm_name, None)
                self.performance_cache.pop(algorithm_name, None)
                
                # Clean up dynamic algorithm cache
                self.dynamic_algorithm_cache.pop(algorithm_name, None)
                
                # Force cleanup of weak references if requested
                if force_cleanup:
                    for instance_id in active_instances:
                        self.algorithm_instances.pop(instance_id, None)
                
                # Create audit trail entry for algorithm unregistration
                create_audit_trail(
                    action='ALGORITHM_UNREGISTERED',
                    component='ALGORITHM_REGISTRY',
                    action_details={
                        'algorithm_name': algorithm_name,
                        'registry_name': self.registry_name,
                        'force_cleanup': force_cleanup,
                        'active_instances_cleaned': len(active_instances)
                    },
                    user_context='SYSTEM'
                )
                
                self.logger.info(f"Algorithm unregistered successfully: {algorithm_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Algorithm unregistration failed: {algorithm_name} - {e}")
                return False
    
    def get(
        self,
        algorithm_name: str,
        include_metadata: bool = False,
        enable_dynamic_loading: bool = True
    ) -> Union[Type[BaseAlgorithm], Tuple[Type[BaseAlgorithm], Dict[str, Any]]]:
        """
        Retrieve algorithm class from registry with dynamic loading support, validation and metadata access.
        
        Args:
            algorithm_name: Name of the algorithm to retrieve
            include_metadata: Whether to include metadata in return
            enable_dynamic_loading: Whether to attempt dynamic loading if not found
            
        Returns:
            Union[Type[BaseAlgorithm], Tuple[Type[BaseAlgorithm], Dict[str, Any]]]: Algorithm class or tuple with metadata
        """
        with self.registry_lock:
            try:
                # Check if algorithm exists in registry
                if algorithm_name in self.algorithms:
                    registry_entry = self.algorithms[algorithm_name]
                    
                    # Update last accessed timestamp
                    registry_entry.last_accessed = datetime.datetime.now()
                    
                    # Extract algorithm class
                    algorithm_class = registry_entry.algorithm_class
                    
                    # Include metadata if requested
                    if include_metadata:
                        return algorithm_class, registry_entry.metadata.copy()
                    else:
                        return algorithm_class
                
                # Attempt dynamic loading if algorithm not found and dynamic loading enabled
                if enable_dynamic_loading and self.dynamic_loading_enabled:
                    algorithm_class = self.load_algorithm_dynamically(
                        algorithm_name=algorithm_name,
                        module_path=ALGORITHM_MODULE_PATHS.get(algorithm_name),
                        class_name=self._get_class_name_from_algorithm_name(algorithm_name),
                        cache_result=True
                    )
                    
                    if algorithm_class:
                        # Auto-register dynamically loaded algorithm
                        self.register(
                            algorithm_name=algorithm_name,
                            algorithm_class=algorithm_class,
                            metadata={'dynamically_loaded': True, 'algorithm_type': algorithm_name},
                            validate_interface=True
                        )
                        
                        if include_metadata:
                            return algorithm_class, self.algorithms[algorithm_name].metadata.copy()
                        else:
                            return algorithm_class
                
                # Algorithm not found
                raise ValueError(f"Algorithm not found: {algorithm_name}")
                
            except Exception as e:
                self.logger.error(f"Algorithm retrieval failed: {algorithm_name} - {e}")
                raise
    
    def create_instance(
        self,
        algorithm_name: str,
        parameters: AlgorithmParameters,
        execution_config: Dict[str, Any] = None,
        instance_id: str = None
    ) -> BaseAlgorithm:
        """
        Create algorithm instance with parameter validation and performance tracking setup.
        
        Args:
            algorithm_name: Name of the algorithm to instantiate
            parameters: Algorithm parameters for instance creation
            execution_config: Configuration for algorithm execution
            instance_id: Unique identifier for the instance
            
        Returns:
            BaseAlgorithm: Configured algorithm instance
        """
        try:
            # Retrieve algorithm class from registry with dynamic loading support
            algorithm_class = self.get(
                algorithm_name=algorithm_name,
                include_metadata=False,
                enable_dynamic_loading=True
            )
            
            # Validate parameters if validation enabled
            if self.validation_enabled:
                validation_result = parameters.validate(strict_validation=True)
                if not validation_result.is_valid:
                    raise ValidationError(
                        message=f"Algorithm parameters validation failed: {algorithm_name}",
                        validation_type="parameter_validation",
                        validation_context={'algorithm_name': algorithm_name},
                        failed_parameters=['algorithm_parameters']
                    )
            
            # Generate unique instance ID if not provided
            if not instance_id:
                instance_id = f"{algorithm_name}_{uuid.uuid4().hex[:8]}"
            
            # Create algorithm instance with parameters and configuration
            algorithm_instance = algorithm_class(
                parameters=parameters,
                execution_config=execution_config or {}
            )
            
            # Setup performance tracking if enabled
            if self.performance_tracking_enabled:
                algorithm_instance.performance_tracking_enabled = True
            
            # Store instance in weak reference dictionary
            self.algorithm_instances[instance_id] = algorithm_instance
            
            # Update instance count in registry entry
            if algorithm_name in self.algorithms:
                self.algorithms[algorithm_name].increment_instance_count()
            
            # Update registry statistics
            self.registry_statistics['successful_instantiations'] += 1
            
            # Create audit trail entry for instance creation
            create_audit_trail(
                action='ALGORITHM_INSTANCE_CREATED',
                component='ALGORITHM_REGISTRY',
                action_details={
                    'algorithm_name': algorithm_name,
                    'instance_id': instance_id,
                    'registry_name': self.registry_name,
                    'performance_tracking': self.performance_tracking_enabled
                },
                user_context='SYSTEM'
            )
            
            self.logger.info(f"Algorithm instance created: {algorithm_name} [{instance_id}]")
            return algorithm_instance
            
        except Exception as e:
            self.logger.error(f"Algorithm instance creation failed: {algorithm_name} - {e}")
            raise
    
    def list_algorithms(
        self,
        algorithm_types: List[str] = None,
        include_metadata: bool = False,
        only_available: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        List all registered algorithms with filtering and metadata options.
        
        Args:
            algorithm_types: List of algorithm types to filter
            include_metadata: Whether to include metadata in listing
            only_available: Whether to include only available algorithms
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of algorithms with metadata
        """
        with self.registry_lock:
            algorithm_listing = {}
            
            for algorithm_name, registry_entry in self.algorithms.items():
                # Filter by availability status if only_available is True
                if only_available and not registry_entry.is_available:
                    continue
                
                # Filter by algorithm types if specified
                if algorithm_types and registry_entry.algorithm_type not in algorithm_types:
                    continue
                
                # Create algorithm information entry
                algorithm_info = {
                    'algorithm_name': algorithm_name,
                    'algorithm_type': registry_entry.algorithm_type,
                    'version': registry_entry.version,
                    'is_available': registry_entry.is_available,
                    'instance_count': registry_entry.instance_count,
                    'last_accessed': registry_entry.last_accessed.isoformat(),
                    'dynamically_loaded': registry_entry.dynamically_loaded
                }
                
                # Include metadata if requested
                if include_metadata:
                    algorithm_info['metadata'] = registry_entry.metadata.copy()
                    algorithm_info['capabilities'] = registry_entry.capabilities.copy()
                    algorithm_info['performance_characteristics'] = registry_entry.performance_characteristics.copy()
                    algorithm_info['supported_formats'] = registry_entry.supported_formats.copy()
                
                algorithm_listing[algorithm_name] = algorithm_info
            
            return algorithm_listing
    
    def validate_interface(
        self,
        algorithm_class: Type[BaseAlgorithm],
        strict_validation: bool = False
    ) -> 'ValidationResult':
        """
        Validate algorithm interface compliance with BaseAlgorithm requirements.
        
        Args:
            algorithm_class: Algorithm class to validate
            strict_validation: Enable strict validation criteria
            
        Returns:
            ValidationResult: Interface validation result with compliance assessment
        """
        from ..utils.validation_utils import ValidationResult
        
        # Initialize validation result
        validation_result = ValidationResult(
            validation_type="algorithm_interface_validation",
            is_valid=True,
            validation_context=f"algorithm_class={algorithm_class.__name__}, strict={strict_validation}"
        )
        
        try:
            # Check algorithm class inheritance from BaseAlgorithm
            if not issubclass(algorithm_class, BaseAlgorithm):
                validation_result.add_error(
                    "Algorithm class must inherit from BaseAlgorithm",
                    severity="CRITICAL"
                )
                validation_result.is_valid = False
            
            # Validate required method implementations
            required_methods = ['execute', 'validate_parameters', 'reset', 'get_performance_summary']
            for method_name in required_methods:
                if not hasattr(algorithm_class, method_name):
                    validation_result.add_error(
                        f"Missing required method: {method_name}",
                        severity="HIGH"
                    )
                    validation_result.is_valid = False
                else:
                    # Check if method is callable
                    method = getattr(algorithm_class, method_name)
                    if not callable(method):
                        validation_result.add_error(
                            f"Method {method_name} is not callable",
                            severity="HIGH"
                        )
                        validation_result.is_valid = False
            
            # Check method signatures and return types using inspect
            if hasattr(algorithm_class, 'execute'):
                execute_method = getattr(algorithm_class, 'execute')
                sig = inspect.signature(execute_method)
                
                # Validate execute method signature
                expected_params = ['self', 'plume_data', 'plume_metadata', 'simulation_id']
                actual_params = list(sig.parameters.keys())
                
                for param in expected_params:
                    if param not in actual_params:
                        validation_result.add_warning(
                            f"Execute method missing expected parameter: {param}"
                        )
            
            # Apply strict validation criteria if enabled
            if strict_validation:
                # Check for abstract method implementations
                abstract_methods = getattr(algorithm_class, '__abstractmethods__', set())
                if abstract_methods:
                    validation_result.add_error(
                        f"Algorithm has unimplemented abstract methods: {abstract_methods}",
                        severity="CRITICAL"
                    )
                    validation_result.is_valid = False
                
                # Validate constructor signature
                init_method = getattr(algorithm_class, '__init__')
                init_sig = inspect.signature(init_method)
                if 'parameters' not in init_sig.parameters:
                    validation_result.add_warning(
                        "Constructor missing 'parameters' parameter"
                    )
            
            # Add validation metrics
            validation_result.add_metric("required_methods_present", float(len(required_methods)))
            validation_result.add_metric("inheritance_valid", float(issubclass(algorithm_class, BaseAlgorithm)))
            
            # Generate validation recommendations
            if validation_result.is_valid:
                validation_result.add_recommendation(
                    "Algorithm interface complies with BaseAlgorithm requirements",
                    priority="INFO"
                )
            else:
                validation_result.add_recommendation(
                    "Address interface compliance issues before registration",
                    priority="HIGH"
                )
        
        except Exception as e:
            validation_result.add_error(
                f"Interface validation failed: {str(e)}",
                severity="CRITICAL"
            )
            validation_result.is_valid = False
        
        validation_result.finalize_validation()
        return validation_result
    
    def update_performance(
        self,
        algorithm_name: str,
        metrics: Dict[str, float],
        context: str = 'execution'
    ) -> None:
        """
        Update algorithm performance metrics with validation and trend analysis.
        
        Args:
            algorithm_name: Name of the algorithm
            metrics: Performance metrics to update
            context: Context for the performance update
        """
        with self.registry_lock:
            try:
                # Validate algorithm exists in registry
                if algorithm_name not in self.algorithms:
                    raise ValueError(f"Algorithm not found: {algorithm_name}")
                
                # Update performance cache with new metrics
                if algorithm_name not in self.performance_cache:
                    self.performance_cache[algorithm_name] = {}
                
                # Store metrics with timestamp
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        self.performance_cache[algorithm_name][metric_name] = float(metric_value)
                
                # Update performance characteristics in registry entry
                registry_entry = self.algorithms[algorithm_name]
                registry_entry.performance_characteristics.update(metrics)
                
                # Update registry statistics
                self.registry_statistics['performance_cache_hits'] += 1
                
                # Create audit trail entry for performance update
                create_audit_trail(
                    action='ALGORITHM_PERFORMANCE_UPDATED',
                    component='ALGORITHM_REGISTRY',
                    action_details={
                        'algorithm_name': algorithm_name,
                        'metrics_count': len(metrics),
                        'context': context,
                        'registry_name': self.registry_name
                    },
                    user_context='SYSTEM'
                )
                
                self.logger.debug(f"Performance metrics updated: {algorithm_name} [{len(metrics)} metrics]")
                
            except Exception as e:
                self.logger.error(f"Performance metrics update failed: {algorithm_name} - {e}")
                raise
    
    def load_algorithm_dynamically(
        self,
        algorithm_name: str,
        module_path: str,
        class_name: str,
        cache_result: bool = True
    ) -> Optional[Type[BaseAlgorithm]]:
        """
        Load algorithm class dynamically from module with caching and validation.
        
        Args:
            algorithm_name: Name of the algorithm to load
            module_path: Python module path for the algorithm
            class_name: Class name within the module
            cache_result: Whether to cache the loaded class
            
        Returns:
            Optional[Type[BaseAlgorithm]]: Dynamically loaded algorithm class or None if failed
        """
        try:
            # Check dynamic algorithm cache for existing class
            cache_key = f"{algorithm_name}_{module_path}_{class_name}"
            if cache_key in self.dynamic_algorithm_cache:
                self.logger.debug(f"Dynamic algorithm cache hit: {algorithm_name}")
                return self.dynamic_algorithm_cache[cache_key]
            
            # Validate module path and class name
            if not module_path or not class_name:
                self.logger.warning(f"Invalid module path or class name for: {algorithm_name}")
                return None
            
            # Import module using importlib with error handling
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                self.logger.warning(f"Failed to import module {module_path}: {e}")
                return None
            
            # Extract algorithm class from module
            if not hasattr(module, class_name):
                self.logger.warning(f"Class {class_name} not found in module {module_path}")
                return None
            
            algorithm_class = getattr(module, class_name)
            
            # Validate algorithm class inheritance from BaseAlgorithm
            if not issubclass(algorithm_class, BaseAlgorithm):
                self.logger.warning(f"Class {class_name} does not inherit from BaseAlgorithm")
                return None
            
            # Cache algorithm class if caching enabled
            if cache_result:
                self.dynamic_algorithm_cache[cache_key] = algorithm_class
            
            # Update registry statistics
            self.registry_statistics['dynamic_loads'] += 1
            
            # Log dynamic loading operation
            self.logger.info(f"Algorithm loaded dynamically: {algorithm_name} from {module_path}")
            
            return algorithm_class
            
        except Exception as e:
            self.logger.error(f"Dynamic algorithm loading failed: {algorithm_name} - {e}")
            return None
    
    def get_statistics(
        self,
        include_performance_trends: bool = False,
        include_usage_patterns: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive registry statistics including usage, performance, and health metrics.
        
        Args:
            include_performance_trends: Whether to include performance trends
            include_usage_patterns: Whether to include usage patterns
            
        Returns:
            Dict[str, Any]: Registry statistics with performance and usage information
        """
        with self.registry_lock:
            # Compile basic registry statistics
            stats = {
                'registry_name': self.registry_name,
                'registry_version': self.registry_version,
                'creation_timestamp': self.creation_timestamp.isoformat(),
                'total_algorithms': len(self.algorithms),
                'available_algorithms': sum(1 for entry in self.algorithms.values() if entry.is_available),
                'dynamically_loaded_algorithms': sum(1 for entry in self.algorithms.values() if entry.dynamically_loaded),
                'total_instances': len(self.algorithm_instances),
                'performance_tracking_enabled': self.performance_tracking_enabled,
                'validation_enabled': self.validation_enabled,
                'dynamic_loading_enabled': self.dynamic_loading_enabled,
                'registry_statistics': self.registry_statistics.copy()
            }
            
            # Include algorithm type distribution
            algorithm_types = {}
            for entry in self.algorithms.values():
                algorithm_type = entry.algorithm_type
                algorithm_types[algorithm_type] = algorithm_types.get(algorithm_type, 0) + 1
            stats['algorithm_type_distribution'] = algorithm_types
            
            # Include performance trends if requested
            if include_performance_trends:
                performance_trends = {}
                for algorithm_name, performance_data in self.performance_cache.items():
                    if performance_data:
                        performance_trends[algorithm_name] = {
                            'metrics_count': len(performance_data),
                            'avg_execution_time': performance_data.get('execution_time_seconds', 0.0),
                            'success_rate': performance_data.get('success_rate', 0.0)
                        }
                stats['performance_trends'] = performance_trends
            
            # Include usage patterns if requested
            if include_usage_patterns:
                usage_patterns = {}
                for algorithm_name, entry in self.algorithms.items():
                    usage_patterns[algorithm_name] = {
                        'instance_count': entry.instance_count,
                        'last_accessed': entry.last_accessed.isoformat(),
                        'registration_age_hours': (datetime.datetime.now() - entry.registration_timestamp).total_seconds() / 3600
                    }
                stats['usage_patterns'] = usage_patterns
            
            return stats
    
    def _register_builtin_algorithms(self) -> None:
        """Register built-in algorithms automatically during registry initialization."""
        builtin_algorithms = [
            ('reference_implementation', ReferenceImplementation, {
                'description': 'Reference implementation for plume source localization',
                'algorithm_type': 'reference_implementation',
                'capabilities': ['gradient_following', 'benchmarking', 'validation'],
                'supported_formats': ['crimaldi', 'custom', 'generic']
            }),
            ('infotaxis', InfotaxisAlgorithm, {
                'description': 'Information-theoretic source localization algorithm',
                'algorithm_type': 'infotaxis',
                'capabilities': ['information_theory', 'source_localization'],
                'supported_formats': ['crimaldi', 'custom']
            }),
            ('casting', CastingAlgorithm, {
                'description': 'Bio-inspired casting algorithm for search patterns',
                'algorithm_type': 'casting',
                'capabilities': ['bio_inspired', 'search_patterns'],
                'supported_formats': ['crimaldi', 'custom']
            }),
            ('gradient_following', GradientFollowing, {
                'description': 'Gradient following algorithm for concentration-based navigation',
                'algorithm_type': 'gradient_following',
                'capabilities': ['gradient_following', 'concentration_navigation'],
                'supported_formats': ['crimaldi', 'custom']
            })
        ]
        
        for algorithm_name, algorithm_class, metadata in builtin_algorithms:
            try:
                self.register(
                    algorithm_name=algorithm_name,
                    algorithm_class=algorithm_class,
                    metadata=metadata,
                    validate_interface=False  # Skip validation for built-in algorithms
                )
            except Exception as e:
                self.logger.warning(f"Failed to register built-in algorithm {algorithm_name}: {e}")
    
    def _get_class_name_from_algorithm_name(self, algorithm_name: str) -> str:
        """Convert algorithm name to expected class name."""
        # Convert snake_case to PascalCase
        words = algorithm_name.split('_')
        return ''.join(word.capitalize() for word in words)


# Global algorithm registry instance for system-wide access
_global_registry = AlgorithmRegistry(
    enable_performance_tracking=True,
    enable_validation=True,
    registry_name='global_plume_simulation_registry',
    enable_dynamic_loading=True
)


def register_algorithm(
    algorithm_name: str,
    algorithm_class: Type[BaseAlgorithm],
    algorithm_metadata: Dict[str, Any] = None,
    validate_interface: bool = True,
    enable_performance_tracking: bool = True
) -> bool:
    """
    Register navigation algorithm in the global registry with comprehensive validation, 
    metadata extraction, interface compliance checking, and audit trail creation for 
    dynamic algorithm discovery and instantiation.
    
    Args:
        algorithm_name: Name of the algorithm to register
        algorithm_class: Algorithm class to register
        algorithm_metadata: Algorithm metadata and configuration
        validate_interface: Whether to validate algorithm interface
        enable_performance_tracking: Whether to enable performance tracking
        
    Returns:
        bool: True if registration successful with validation status and registry update confirmation
    """
    try:
        # Set performance tracking in metadata
        if algorithm_metadata is None:
            algorithm_metadata = {}
        algorithm_metadata['performance_tracking_enabled'] = enable_performance_tracking
        
        # Register algorithm in global registry
        success = _global_registry.register(
            algorithm_name=algorithm_name,
            algorithm_class=algorithm_class,
            metadata=algorithm_metadata,
            validate_interface=validate_interface
        )
        
        # Log registration with scientific context
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        if success:
            logger.info(f"Algorithm registered globally: {algorithm_name}")
        else:
            logger.warning(f"Algorithm registration failed: {algorithm_name}")
        
        return success
        
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.error(f"Global algorithm registration failed: {algorithm_name} - {e}")
        return False


def unregister_algorithm(
    algorithm_name: str,
    force_cleanup: bool = False,
    preserve_metadata: bool = False
) -> bool:
    """
    Unregister algorithm from the global registry with cleanup of metadata, performance cache, 
    active instances, and audit trail creation for registry maintenance and algorithm lifecycle management.
    
    Args:
        algorithm_name: Name of the algorithm to unregister
        force_cleanup: Whether to force cleanup of active instances
        preserve_metadata: Whether to preserve metadata during unregistration
        
    Returns:
        bool: True if unregistration successful with cleanup confirmation
    """
    try:
        # Preserve metadata in cache before unregistration if requested
        if preserve_metadata and algorithm_name in _global_registry.algorithms:
            _algorithm_metadata_cache[f"{algorithm_name}_preserved"] = _global_registry.algorithms[algorithm_name].metadata.copy()
        
        # Unregister algorithm from global registry
        success = _global_registry.unregister(
            algorithm_name=algorithm_name,
            force_cleanup=force_cleanup
        )
        
        # Log unregistration with context
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        if success:
            logger.info(f"Algorithm unregistered globally: {algorithm_name}")
        else:
            logger.warning(f"Algorithm unregistration failed: {algorithm_name}")
        
        return success
        
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.error(f"Global algorithm unregistration failed: {algorithm_name} - {e}")
        return False


def get_algorithm(
    algorithm_name: str,
    validate_availability: bool = True,
    include_metadata: bool = False,
    enable_dynamic_loading: bool = True
) -> Union[Type[BaseAlgorithm], Tuple[Type[BaseAlgorithm], Dict[str, Any]]]:
    """
    Retrieve algorithm class from registry with dynamic loading support, validation, 
    metadata access, and instance management for algorithm instantiation with comprehensive 
    error handling and performance tracking.
    
    Args:
        algorithm_name: Name of the algorithm to retrieve
        validate_availability: Whether to validate algorithm availability
        include_metadata: Whether to include metadata in return
        enable_dynamic_loading: Whether to enable dynamic loading
        
    Returns:
        Union[Type[BaseAlgorithm], Tuple[Type[BaseAlgorithm], Dict[str, Any]]]: Algorithm class or tuple with metadata
    """
    try:
        # Validate algorithm availability if requested
        if validate_availability:
            if algorithm_name not in _global_registry.algorithms:
                if not enable_dynamic_loading:
                    raise ValueError(f"Algorithm not available: {algorithm_name}")
        
        # Retrieve algorithm class from global registry
        result = _global_registry.get(
            algorithm_name=algorithm_name,
            include_metadata=include_metadata,
            enable_dynamic_loading=enable_dynamic_loading
        )
        
        # Log algorithm retrieval for audit trail
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.debug(f"Algorithm retrieved: {algorithm_name}")
        
        return result
        
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.error(f"Algorithm retrieval failed: {algorithm_name} - {e}")
        raise


def create_algorithm_instance(
    algorithm_name: str,
    parameters: AlgorithmParameters,
    execution_config: Dict[str, Any] = None,
    enable_validation: bool = True,
    instance_id: str = None
) -> BaseAlgorithm:
    """
    Create algorithm instance with parameter validation, configuration setup, performance 
    tracking initialization, and scientific context management for isolated algorithm execution.
    
    Args:
        algorithm_name: Name of the algorithm to instantiate
        parameters: Algorithm parameters for instance creation
        execution_config: Configuration for algorithm execution
        enable_validation: Whether to enable parameter validation
        instance_id: Unique identifier for the instance
        
    Returns:
        BaseAlgorithm: Configured algorithm instance with validation and performance tracking
    """
    try:
        # Create algorithm instance using global registry
        algorithm_instance = _global_registry.create_instance(
            algorithm_name=algorithm_name,
            parameters=parameters,
            execution_config=execution_config,
            instance_id=instance_id
        )
        
        # Setup scientific context for algorithm execution
        set_scientific_context(
            simulation_id=execution_config.get('simulation_id', 'unknown') if execution_config else 'unknown',
            algorithm_name=algorithm_name,
            processing_stage='ALGORITHM_INSTANTIATION'
        )
        
        # Log instance creation with context
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.info(f"Algorithm instance created: {algorithm_name} [{instance_id or 'auto-generated'}]")
        
        return algorithm_instance
        
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.error(f"Algorithm instance creation failed: {algorithm_name} - {e}")
        raise


def list_algorithms(
    algorithm_types: List[str] = None,
    include_metadata: bool = False,
    include_performance_metrics: bool = False,
    only_available: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    List all registered algorithms with metadata, availability status, performance metrics, 
    and filtering capabilities for algorithm discovery and selection.
    
    Args:
        algorithm_types: List of algorithm types to filter
        include_metadata: Whether to include metadata in listing
        include_performance_metrics: Whether to include performance metrics
        only_available: Whether to include only available algorithms
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of algorithm names with metadata and performance information
    """
    try:
        # Get algorithm listing from global registry
        algorithm_listing = _global_registry.list_algorithms(
            algorithm_types=algorithm_types,
            include_metadata=include_metadata,
            only_available=only_available
        )
        
        # Include performance metrics if requested
        if include_performance_metrics:
            for algorithm_name, algorithm_info in algorithm_listing.items():
                if algorithm_name in _global_registry.performance_cache:
                    algorithm_info['performance_metrics'] = _global_registry.performance_cache[algorithm_name].copy()
        
        # Log algorithm listing request
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.debug(f"Algorithm listing requested: {len(algorithm_listing)} algorithms")
        
        return algorithm_listing
        
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.error(f"Algorithm listing failed: {e}")
        return {}


def validate_algorithm_interface(
    algorithm_class: Type[BaseAlgorithm],
    strict_validation: bool = False,
    validation_config: Dict[str, Any] = None
) -> 'ValidationResult':
    """
    Validate algorithm class interface compliance with BaseAlgorithm requirements, method 
    signatures, and scientific computing standards for registry compatibility.
    
    Args:
        algorithm_class: Algorithm class to validate
        strict_validation: Enable strict validation criteria
        validation_config: Configuration for validation process
        
    Returns:
        ValidationResult: Interface validation result with compliance assessment and recommendations
    """
    try:
        # Validate algorithm interface using global registry
        validation_result = _global_registry.validate_interface(
            algorithm_class=algorithm_class,
            strict_validation=strict_validation
        )
        
        # Apply additional validation configuration if provided
        if validation_config:
            # Additional validation logic based on configuration
            pass
        
        # Log validation request
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.debug(f"Algorithm interface validation: {algorithm_class.__name__} - valid: {validation_result.is_valid}")
        
        return validation_result
        
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.error(f"Algorithm interface validation failed: {algorithm_class.__name__ if algorithm_class else 'unknown'} - {e}")
        raise


def get_algorithm_metadata(
    algorithm_name: str,
    include_performance_history: bool = False,
    include_validation_status: bool = False
) -> Dict[str, Any]:
    """
    Retrieve comprehensive algorithm metadata including capabilities, performance characteristics, 
    validation requirements, and compatibility information for algorithm selection and configuration.
    
    Args:
        algorithm_name: Name of the algorithm
        include_performance_history: Whether to include performance history
        include_validation_status: Whether to include validation status
        
    Returns:
        Dict[str, Any]: Comprehensive algorithm metadata with performance and validation information
    """
    try:
        # Retrieve algorithm and metadata from global registry
        algorithm_class, metadata = _global_registry.get(
            algorithm_name=algorithm_name,
            include_metadata=True,
            enable_dynamic_loading=True
        )
        
        # Compile comprehensive metadata
        comprehensive_metadata = metadata.copy()
        
        # Include performance history if requested
        if include_performance_history:
            if algorithm_name in _global_registry.performance_cache:
                comprehensive_metadata['performance_history'] = _global_registry.performance_cache[algorithm_name].copy()
        
        # Include validation status if requested
        if include_validation_status:
            validation_result = _global_registry.validate_interface(algorithm_class)
            comprehensive_metadata['validation_status'] = {
                'is_valid': validation_result.is_valid,
                'validation_errors': validation_result.errors,
                'validation_warnings': validation_result.warnings
            }
        
        # Add registry information
        if algorithm_name in _global_registry.algorithms:
            registry_entry = _global_registry.algorithms[algorithm_name]
            comprehensive_metadata['registry_info'] = {
                'registration_timestamp': registry_entry.registration_timestamp.isoformat(),
                'last_accessed': registry_entry.last_accessed.isoformat(),
                'instance_count': registry_entry.instance_count,
                'is_available': registry_entry.is_available
            }
        
        return comprehensive_metadata
        
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.error(f"Algorithm metadata retrieval failed: {algorithm_name} - {e}")
        return {}


def update_algorithm_performance(
    algorithm_name: str,
    performance_metrics: Dict[str, float],
    execution_context: str = 'execution',
    validate_metrics: bool = True
) -> None:
    """
    Update algorithm performance metrics in registry cache with statistical validation, 
    trend analysis, and performance threshold checking for algorithm optimization and selection.
    
    Args:
        algorithm_name: Name of the algorithm
        performance_metrics: Performance metrics to update
        execution_context: Context for the performance update
        validate_metrics: Whether to validate metrics
    """
    try:
        # Validate performance metrics if validation enabled
        if validate_metrics:
            for metric_name, metric_value in performance_metrics.items():
                if not isinstance(metric_value, (int, float)):
                    raise ValueError(f"Invalid metric value type: {metric_name} = {type(metric_value)}")
                if metric_value < 0 and 'time' in metric_name.lower():
                    raise ValueError(f"Negative time metric: {metric_name} = {metric_value}")
        
        # Update performance metrics in global registry
        _global_registry.update_performance(
            algorithm_name=algorithm_name,
            metrics=performance_metrics,
            context=execution_context
        )
        
        # Log performance metrics update with scientific context
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.debug(f"Performance metrics updated: {algorithm_name} [{len(performance_metrics)} metrics]")
        
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.error(f"Performance metrics update failed: {algorithm_name} - {e}")
        raise


def compare_algorithms(
    algorithm_names: List[str],
    comparison_metrics: List[str] = None,
    include_statistical_analysis: bool = False,
    validate_reproducibility: bool = False
) -> Dict[str, Any]:
    """
    Compare multiple algorithms using performance metrics, statistical analysis, correlation 
    assessment, and reproducibility validation for algorithm selection and optimization.
    
    Args:
        algorithm_names: List of algorithm names to compare
        comparison_metrics: List of metrics to use for comparison
        include_statistical_analysis: Whether to include statistical analysis
        validate_reproducibility: Whether to validate reproducibility
        
    Returns:
        Dict[str, Any]: Algorithm comparison results with statistical analysis and rankings
    """
    try:
        # Validate all algorithms exist in registry
        missing_algorithms = []
        for algorithm_name in algorithm_names:
            if algorithm_name not in _global_registry.algorithms:
                missing_algorithms.append(algorithm_name)
        
        if missing_algorithms:
            raise ValueError(f"Algorithms not found: {missing_algorithms}")
        
        # Initialize comparison results
        comparison_results = {
            'comparison_id': str(uuid.uuid4()),
            'comparison_timestamp': datetime.datetime.now().isoformat(),
            'algorithms_compared': algorithm_names,
            'comparison_metrics': comparison_metrics or ['execution_time_seconds', 'success_rate'],
            'algorithm_performance': {},
            'rankings': {},
            'statistical_analysis': {},
            'reproducibility_analysis': {}
        }
        
        # Retrieve performance metrics for all algorithms
        for algorithm_name in algorithm_names:
            if algorithm_name in _global_registry.performance_cache:
                comparison_results['algorithm_performance'][algorithm_name] = _global_registry.performance_cache[algorithm_name].copy()
            else:
                comparison_results['algorithm_performance'][algorithm_name] = {}
        
        # Calculate rankings for each comparison metric
        for metric_name in comparison_results['comparison_metrics']:
            metric_values = {}
            for algorithm_name in algorithm_names:
                performance_data = comparison_results['algorithm_performance'][algorithm_name]
                metric_values[algorithm_name] = performance_data.get(metric_name, 0.0)
            
            # Rank algorithms by metric (higher is better except for time metrics)
            if 'time' in metric_name.lower():
                # Lower is better for time metrics
                ranked_algorithms = sorted(metric_values.items(), key=lambda x: x[1])
            else:
                # Higher is better for other metrics
                ranked_algorithms = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            
            comparison_results['rankings'][metric_name] = ranked_algorithms
        
        # Include statistical analysis if requested
        if include_statistical_analysis:
            # Perform statistical analysis on metrics
            statistical_data = {}
            for metric_name in comparison_results['comparison_metrics']:
                metric_values = []
                for algorithm_name in algorithm_names:
                    performance_data = comparison_results['algorithm_performance'][algorithm_name]
                    metric_values.append(performance_data.get(metric_name, 0.0))
                
                # Calculate statistical measures
                if metric_values:
                    import statistics
                    statistical_data[metric_name] = {
                        'mean': statistics.mean(metric_values),
                        'median': statistics.median(metric_values),
                        'stdev': statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0,
                        'min': min(metric_values),
                        'max': max(metric_values)
                    }
            
            comparison_results['statistical_analysis'] = statistical_data
        
        # Validate reproducibility if requested
        if validate_reproducibility:
            reproducibility_data = {}
            for algorithm_name in algorithm_names:
                # Check if algorithm meets reproducibility threshold
                performance_data = comparison_results['algorithm_performance'][algorithm_name]
                correlation_score = performance_data.get('overall_correlation', 0.0)
                reproducibility_score = performance_data.get('reproducibility_score', 0.0)
                
                reproducibility_data[algorithm_name] = {
                    'correlation_score': correlation_score,
                    'reproducibility_score': reproducibility_score,
                    'meets_correlation_threshold': correlation_score >= CORRELATION_THRESHOLD,
                    'meets_reproducibility_threshold': reproducibility_score >= REPRODUCIBILITY_THRESHOLD
                }
            
            comparison_results['reproducibility_analysis'] = reproducibility_data
        
        # Generate overall recommendations
        comparison_results['recommendations'] = _generate_comparison_recommendations(comparison_results)
        
        # Log algorithm comparison operation
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.info(f"Algorithm comparison completed: {len(algorithm_names)} algorithms compared")
        
        return comparison_results
        
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.error(f"Algorithm comparison failed: {e}")
        return {}


def validate_registry_integrity(
    deep_validation: bool = False,
    repair_inconsistencies: bool = False
) -> Dict[str, Any]:
    """
    Validate registry integrity including algorithm availability, metadata consistency, 
    performance cache validity, and interface compliance for registry maintenance and quality assurance.
    
    Args:
        deep_validation: Whether to perform deep validation
        repair_inconsistencies: Whether to repair found inconsistencies
        
    Returns:
        Dict[str, Any]: Registry integrity validation report with issues and repair status
    """
    with _registry_lock:
        try:
            # Initialize validation report
            validation_report = {
                'validation_id': str(uuid.uuid4()),
                'validation_timestamp': datetime.datetime.now().isoformat(),
                'deep_validation': deep_validation,
                'repair_enabled': repair_inconsistencies,
                'registry_statistics': _global_registry.get_statistics(),
                'validation_results': {
                    'algorithm_entries': {'valid': 0, 'invalid': 0, 'repaired': 0},
                    'metadata_consistency': {'consistent': 0, 'inconsistent': 0, 'repaired': 0},
                    'performance_cache': {'valid': 0, 'invalid': 0, 'cleaned': 0},
                    'interface_compliance': {'compliant': 0, 'non_compliant': 0}
                },
                'issues_found': [],
                'repairs_performed': [],
                'recommendations': []
            }
            
            # Validate algorithm registry entries
            for algorithm_name, registry_entry in _global_registry.algorithms.items():
                if registry_entry.validate(strict_validation=deep_validation):
                    validation_report['validation_results']['algorithm_entries']['valid'] += 1
                else:
                    validation_report['validation_results']['algorithm_entries']['invalid'] += 1
                    validation_report['issues_found'].append(f"Invalid registry entry: {algorithm_name}")
                    
                    if repair_inconsistencies:
                        # Attempt to repair registry entry
                        try:
                            # Basic repair - ensure required fields
                            if not registry_entry.metadata.get('description'):
                                registry_entry.metadata['description'] = f"Algorithm: {algorithm_name}"
                            if not registry_entry.metadata.get('version'):
                                registry_entry.metadata['version'] = ALGORITHM_INTERFACE_VERSION
                            
                            validation_report['validation_results']['algorithm_entries']['repaired'] += 1
                            validation_report['repairs_performed'].append(f"Repaired registry entry: {algorithm_name}")
                        except Exception:
                            pass
            
            # Check metadata consistency between registry and cache
            for algorithm_name in _global_registry.algorithms:
                registry_metadata = _global_registry.algorithms[algorithm_name].metadata
                cached_metadata = _global_registry.metadata_cache.get(algorithm_name, {})
                
                if registry_metadata == cached_metadata:
                    validation_report['validation_results']['metadata_consistency']['consistent'] += 1
                else:
                    validation_report['validation_results']['metadata_consistency']['inconsistent'] += 1
                    validation_report['issues_found'].append(f"Metadata inconsistency: {algorithm_name}")
                    
                    if repair_inconsistencies:
                        # Sync metadata cache with registry
                        _global_registry.metadata_cache[algorithm_name] = registry_metadata.copy()
                        validation_report['validation_results']['metadata_consistency']['repaired'] += 1
                        validation_report['repairs_performed'].append(f"Synced metadata cache: {algorithm_name}")
            
            # Validate performance cache integrity
            for algorithm_name, performance_data in _global_registry.performance_cache.items():
                valid_metrics = True
                for metric_name, metric_value in performance_data.items():
                    if not isinstance(metric_value, (int, float)):
                        valid_metrics = False
                        break
                
                if valid_metrics:
                    validation_report['validation_results']['performance_cache']['valid'] += 1
                else:
                    validation_report['validation_results']['performance_cache']['invalid'] += 1
                    validation_report['issues_found'].append(f"Invalid performance cache: {algorithm_name}")
                    
                    if repair_inconsistencies:
                        # Clean invalid metrics
                        cleaned_metrics = {k: v for k, v in performance_data.items() if isinstance(v, (int, float))}
                        _global_registry.performance_cache[algorithm_name] = cleaned_metrics
                        validation_report['validation_results']['performance_cache']['cleaned'] += 1
                        validation_report['repairs_performed'].append(f"Cleaned performance cache: {algorithm_name}")
            
            # Perform interface compliance checking if deep validation enabled
            if deep_validation:
                for algorithm_name, registry_entry in _global_registry.algorithms.items():
                    try:
                        validation_result = _global_registry.validate_interface(registry_entry.algorithm_class, strict_validation=True)
                        if validation_result.is_valid:
                            validation_report['validation_results']['interface_compliance']['compliant'] += 1
                        else:
                            validation_report['validation_results']['interface_compliance']['non_compliant'] += 1
                            validation_report['issues_found'].append(f"Interface non-compliance: {algorithm_name}")
                    except Exception as e:
                        validation_report['issues_found'].append(f"Interface validation error: {algorithm_name} - {e}")
            
            # Generate validation recommendations
            if validation_report['issues_found']:
                validation_report['recommendations'].append("Address identified issues to maintain registry integrity")
                if not repair_inconsistencies:
                    validation_report['recommendations'].append("Consider enabling repair mode for automatic issue resolution")
            else:
                validation_report['recommendations'].append("Registry integrity is good - no issues found")
            
            # Calculate overall integrity score
            total_checks = sum(sum(category.values()) for category in validation_report['validation_results'].values())
            valid_checks = sum(category.get('valid', 0) + category.get('consistent', 0) + category.get('compliant', 0) 
                             for category in validation_report['validation_results'].values())
            
            validation_report['integrity_score'] = (valid_checks / total_checks) if total_checks > 0 else 1.0
            validation_report['overall_status'] = 'good' if validation_report['integrity_score'] >= 0.95 else 'needs_attention'
            
            # Create audit trail entry for integrity validation
            create_audit_trail(
                action='REGISTRY_INTEGRITY_VALIDATION',
                component='ALGORITHM_REGISTRY',
                action_details={
                    'validation_id': validation_report['validation_id'],
                    'deep_validation': deep_validation,
                    'issues_found': len(validation_report['issues_found']),
                    'repairs_performed': len(validation_report['repairs_performed']),
                    'integrity_score': validation_report['integrity_score']
                },
                user_context='SYSTEM'
            )
            
            # Log integrity validation completion
            logger = get_logger('algorithm_registry', 'ALGORITHM')
            logger.info(f"Registry integrity validation completed: score={validation_report['integrity_score']:.3f}")
            
            return validation_report
            
        except Exception as e:
            logger = get_logger('algorithm_registry', 'ALGORITHM')
            logger.error(f"Registry integrity validation failed: {e}")
            return {
                'validation_error': str(e),
                'validation_timestamp': datetime.datetime.now().isoformat()
            }


def clear_performance_cache(
    algorithm_name: Optional[str] = None,
    preserve_recent_data: bool = True,
    age_threshold: float = PERFORMANCE_CACHE_TTL
) -> Dict[str, int]:
    """
    Clear algorithm performance cache with selective clearing options, cache statistics, 
    and audit trail creation for cache maintenance and performance optimization.
    
    Args:
        algorithm_name: Specific algorithm to clear (None for all)
        preserve_recent_data: Whether to preserve recent performance data
        age_threshold: Age threshold in seconds for cache entry removal
        
    Returns:
        Dict[str, int]: Cache clearing statistics with counts of cleared entries
    """
    with _registry_lock:
        try:
            # Initialize cache clearing statistics
            clearing_stats = {
                'entries_cleared': 0,
                'entries_preserved': 0,
                'total_entries_before': len(_global_registry.performance_cache),
                'algorithms_affected': 0
            }
            
            current_time = time.time()
            
            # Clear specific algorithm cache if algorithm_name provided
            if algorithm_name:
                if algorithm_name in _global_registry.performance_cache:
                    if preserve_recent_data:
                        # Check if data is recent enough to preserve
                        registry_entry = _global_registry.algorithms.get(algorithm_name)
                        if registry_entry:
                            last_accessed = registry_entry.last_accessed.timestamp()
                            if (current_time - last_accessed) < age_threshold:
                                clearing_stats['entries_preserved'] += 1
                            else:
                                _global_registry.performance_cache.pop(algorithm_name)
                                clearing_stats['entries_cleared'] += 1
                                clearing_stats['algorithms_affected'] += 1
                        else:
                            _global_registry.performance_cache.pop(algorithm_name)
                            clearing_stats['entries_cleared'] += 1
                            clearing_stats['algorithms_affected'] += 1
                    else:
                        _global_registry.performance_cache.pop(algorithm_name)
                        clearing_stats['entries_cleared'] += 1
                        clearing_stats['algorithms_affected'] += 1
            else:
                # Clear all algorithm caches with age threshold consideration
                algorithms_to_clear = []
                
                for algorithm_name in list(_global_registry.performance_cache.keys()):
                    should_clear = True
                    
                    if preserve_recent_data:
                        registry_entry = _global_registry.algorithms.get(algorithm_name)
                        if registry_entry:
                            last_accessed = registry_entry.last_accessed.timestamp()
                            if (current_time - last_accessed) < age_threshold:
                                should_clear = False
                                clearing_stats['entries_preserved'] += 1
                    
                    if should_clear:
                        algorithms_to_clear.append(algorithm_name)
                
                # Clear identified algorithms
                for algorithm_name in algorithms_to_clear:
                    _global_registry.performance_cache.pop(algorithm_name, None)
                    clearing_stats['entries_cleared'] += 1
                
                clearing_stats['algorithms_affected'] = len(algorithms_to_clear)
            
            # Update final statistics
            clearing_stats['total_entries_after'] = len(_global_registry.performance_cache)
            
            # Create audit trail entry for cache clearing
            create_audit_trail(
                action='PERFORMANCE_CACHE_CLEARED',
                component='ALGORITHM_REGISTRY',
                action_details={
                    'specific_algorithm': algorithm_name,
                    'preserve_recent_data': preserve_recent_data,
                    'age_threshold': age_threshold,
                    'clearing_statistics': clearing_stats
                },
                user_context='SYSTEM'
            )
            
            # Log cache clearing operation
            logger = get_logger('algorithm_registry', 'ALGORITHM')
            logger.info(f"Performance cache cleared: {clearing_stats['entries_cleared']} entries, {clearing_stats['algorithms_affected']} algorithms affected")
            
            return clearing_stats
            
        except Exception as e:
            logger = get_logger('algorithm_registry', 'ALGORITHM')
            logger.error(f"Performance cache clearing failed: {e}")
            return {'error': str(e)}


def export_registry_configuration(
    output_path: str,
    include_performance_data: bool = False,
    include_validation_history: bool = False,
    export_format: str = 'json'
) -> bool:
    """
    Export registry configuration including algorithm metadata, performance data, and validation 
    status for backup, documentation, and reproducibility support.
    
    Args:
        output_path: Path for output file
        include_performance_data: Whether to include performance data
        include_validation_history: Whether to include validation history
        export_format: Format for export ('json')
        
    Returns:
        bool: True if export successful with file validation and integrity checking
    """
    try:
        # Compile registry configuration data
        export_data = {
            'export_metadata': {
                'export_timestamp': datetime.datetime.now().isoformat(),
                'export_format': export_format,
                'registry_version': REGISTRY_VERSION,
                'algorithm_interface_version': ALGORITHM_INTERFACE_VERSION
            },
            'registry_configuration': {
                'registry_name': _global_registry.registry_name,
                'creation_timestamp': _global_registry.creation_timestamp.isoformat(),
                'performance_tracking_enabled': _global_registry.performance_tracking_enabled,
                'validation_enabled': _global_registry.validation_enabled,
                'dynamic_loading_enabled': _global_registry.dynamic_loading_enabled
            },
            'algorithms': {},
            'registry_statistics': _global_registry.get_statistics()
        }
        
        # Export algorithm entries with metadata
        for algorithm_name, registry_entry in _global_registry.algorithms.items():
            export_data['algorithms'][algorithm_name] = registry_entry.to_dict(include_class_reference=False)
        
        # Include performance data if requested
        if include_performance_data:
            export_data['performance_data'] = _global_registry.performance_cache.copy()
        
        # Include validation history if requested (placeholder for future implementation)
        if include_validation_history:
            export_data['validation_history'] = {}
        
        # Export data to specified output path
        output_file = pathlib.Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if export_format.lower() == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        # Validate exported file integrity
        if output_file.exists() and output_file.stat().st_size > 0:
            # Basic file validation - try to read back
            with open(output_file, 'r', encoding='utf-8') as f:
                json.load(f)  # Verify JSON format
            
            # Create audit trail entry for export
            create_audit_trail(
                action='REGISTRY_CONFIGURATION_EXPORTED',
                component='ALGORITHM_REGISTRY',
                action_details={
                    'output_path': str(output_path),
                    'export_format': export_format,
                    'include_performance_data': include_performance_data,
                    'include_validation_history': include_validation_history,
                    'algorithms_exported': len(export_data['algorithms'])
                },
                user_context='SYSTEM'
            )
            
            logger = get_logger('algorithm_registry', 'ALGORITHM')
            logger.info(f"Registry configuration exported successfully: {output_path}")
            return True
        else:
            logger = get_logger('algorithm_registry', 'ALGORITHM')
            logger.error(f"Export validation failed: {output_path}")
            return False
        
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.error(f"Registry configuration export failed: {e}")
        return False


def load_algorithm_dynamically(
    algorithm_name: str,
    module_path: str,
    class_name: str,
    cache_result: bool = True
) -> Optional[Type[BaseAlgorithm]]:
    """
    Load algorithm class dynamically from module path with caching, validation, and error 
    handling for on-demand algorithm discovery without circular dependencies.
    
    Args:
        algorithm_name: Name of the algorithm to load
        module_path: Python module path for dynamic loading
        class_name: Class name within the module
        cache_result: Whether to cache the loaded algorithm class
        
    Returns:
        Optional[Type[BaseAlgorithm]]: Dynamically loaded algorithm class or None if loading failed
    """
    try:
        # Load algorithm using global registry
        algorithm_class = _global_registry.load_algorithm_dynamically(
            algorithm_name=algorithm_name,
            module_path=module_path,
            class_name=class_name,
            cache_result=cache_result
        )
        
        if algorithm_class:
            logger = get_logger('algorithm_registry', 'ALGORITHM')
            logger.info(f"Algorithm loaded dynamically: {algorithm_name}")
        
        return algorithm_class
        
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.error(f"Dynamic algorithm loading failed: {algorithm_name} - {e}")
        return None


def register_plume_tracking_algorithm(
    force_reload: bool = False,
    validate_integration: bool = True
) -> bool:
    """
    Register plume tracking algorithm using dynamic loading to avoid circular dependencies 
    while maintaining full functionality and integration with the registry system.
    
    Args:
        force_reload: Whether to force reload if already registered
        validate_integration: Whether to validate integration after registration
        
    Returns:
        bool: True if plume tracking algorithm successfully registered
    """
    try:
        algorithm_name = 'plume_tracking'
        
        # Check if already registered and handle force reload
        if algorithm_name in _global_registry.algorithms:
            if not force_reload:
                logger = get_logger('algorithm_registry', 'ALGORITHM')
                logger.info(f"Plume tracking algorithm already registered")
                return True
            else:
                unregister_algorithm(algorithm_name, force_cleanup=True)
        
        # Load plume tracking algorithm dynamically
        module_path = ALGORITHM_MODULE_PATHS.get(algorithm_name)
        class_name = 'PlumeTracking'
        
        algorithm_class = load_algorithm_dynamically(
            algorithm_name=algorithm_name,
            module_path=module_path,
            class_name=class_name,
            cache_result=True
        )
        
        if not algorithm_class:
            logger = get_logger('algorithm_registry', 'ALGORITHM')
            logger.warning(f"Failed to load plume tracking algorithm from {module_path}")
            return False
        
        # Extract algorithm metadata
        algorithm_metadata = {
            'description': 'Plume tracking algorithm for source localization',
            'algorithm_type': 'plume_tracking',
            'capabilities': ['plume_tracking', 'source_localization', 'adaptive_search'],
            'supported_formats': ['crimaldi', 'custom'],
            'dynamically_loaded': True
        }
        
        # Register algorithm in global registry
        success = register_algorithm(
            algorithm_name=algorithm_name,
            algorithm_class=algorithm_class,
            algorithm_metadata=algorithm_metadata,
            validate_interface=validate_integration,
            enable_performance_tracking=True
        )
        
        if success:
            logger = get_logger('algorithm_registry', 'ALGORITHM')
            logger.info(f"Plume tracking algorithm registered successfully")
        
        return success
        
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.error(f"Plume tracking algorithm registration failed: {e}")
        return False


def register_hybrid_strategies_algorithm(
    force_reload: bool = False,
    validate_integration: bool = True
) -> bool:
    """
    Register hybrid strategies algorithm using dynamic loading to avoid circular dependencies 
    while maintaining full functionality and integration with the registry system.
    
    Args:
        force_reload: Whether to force reload if already registered
        validate_integration: Whether to validate integration after registration
        
    Returns:
        bool: True if hybrid strategies algorithm successfully registered
    """
    try:
        algorithm_name = 'hybrid_strategies'
        
        # Check if already registered and handle force reload
        if algorithm_name in _global_registry.algorithms:
            if not force_reload:
                logger = get_logger('algorithm_registry', 'ALGORITHM')
                logger.info(f"Hybrid strategies algorithm already registered")
                return True
            else:
                unregister_algorithm(algorithm_name, force_cleanup=True)
        
        # Load hybrid strategies algorithm dynamically
        module_path = ALGORITHM_MODULE_PATHS.get(algorithm_name)
        class_name = 'HybridStrategies'
        
        algorithm_class = load_algorithm_dynamically(
            algorithm_name=algorithm_name,
            module_path=module_path,
            class_name=class_name,
            cache_result=True
        )
        
        if not algorithm_class:
            logger = get_logger('algorithm_registry', 'ALGORITHM')
            logger.warning(f"Failed to load hybrid strategies algorithm from {module_path}")
            return False
        
        # Extract algorithm metadata
        algorithm_metadata = {
            'description': 'Hybrid strategies algorithm combining multiple navigation approaches',
            'algorithm_type': 'hybrid_strategies',
            'capabilities': ['hybrid_navigation', 'multi_strategy', 'adaptive_switching'],
            'supported_formats': ['crimaldi', 'custom'],
            'dynamically_loaded': True
        }
        
        # Register algorithm in global registry
        success = register_algorithm(
            algorithm_name=algorithm_name,
            algorithm_class=algorithm_class,
            algorithm_metadata=algorithm_metadata,
            validate_interface=validate_integration,
            enable_performance_tracking=True
        )
        
        if success:
            logger = get_logger('algorithm_registry', 'ALGORITHM')
            logger.info(f"Hybrid strategies algorithm registered successfully")
        
        return success
        
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.error(f"Hybrid strategies algorithm registration failed: {e}")
        return False


def discover_available_algorithms(
    algorithms_directory: str = 'src/backend/algorithms',
    auto_register: bool = False,
    exclude_modules: List[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Discover all available algorithms in the algorithms directory with dynamic loading support, 
    metadata extraction, and compatibility validation for comprehensive algorithm registry population.
    
    Args:
        algorithms_directory: Directory to scan for algorithm modules
        auto_register: Whether to automatically register discovered algorithms
        exclude_modules: List of module names to exclude from discovery
        
    Returns:
        Dict[str, Dict[str, Any]]: Discovered algorithms with metadata and loading status
    """
    try:
        discovery_results = {
            'discovery_id': str(uuid.uuid4()),
            'discovery_timestamp': datetime.datetime.now().isoformat(),
            'algorithms_directory': algorithms_directory,
            'auto_register': auto_register,
            'discovered_algorithms': {},
            'discovery_statistics': {
                'total_modules_scanned': 0,
                'algorithms_discovered': 0,
                'algorithms_registered': 0,
                'loading_failures': 0
            }
        }
        
        # Setup exclusion list
        exclude_modules = exclude_modules or ['__init__', '__pycache__', 'base_algorithm', 'algorithm_registry']
        
        # Scan algorithms directory for Python modules
        algorithms_path = pathlib.Path(algorithms_directory)
        if not algorithms_path.exists():
            logger = get_logger('algorithm_registry', 'ALGORITHM')
            logger.warning(f"Algorithms directory not found: {algorithms_directory}")
            return discovery_results
        
        # Scan for Python module files
        for module_file in algorithms_path.glob('*.py'):
            module_name = module_file.stem
            
            # Skip excluded modules
            if module_name in exclude_modules:
                continue
            
            discovery_results['discovery_statistics']['total_modules_scanned'] += 1
            
            try:
                # Attempt dynamic loading for each algorithm module
                module_path = f"src.backend.algorithms.{module_name}"
                
                # Import module dynamically
                module = importlib.import_module(module_path)
                
                # Find algorithm classes in module
                algorithm_classes = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (inspect.isclass(attr) and 
                        issubclass(attr, BaseAlgorithm) and 
                        attr != BaseAlgorithm and
                        not getattr(attr, '__abstractmethods__', None)):
                        algorithm_classes.append((attr_name, attr))
                
                # Process discovered algorithm classes
                for class_name, algorithm_class in algorithm_classes:
                    algorithm_name = module_name  # Use module name as algorithm name
                    
                    # Extract metadata from algorithm class
                    algorithm_metadata = {
                        'description': getattr(algorithm_class, '__doc__', f'Algorithm: {class_name}').split('\n')[0],
                        'algorithm_type': algorithm_name,
                        'class_name': class_name,
                        'module_path': module_path,
                        'discovered': True,
                        'dynamically_loaded': True
                    }
                    
                    # Validate interface compliance
                    validation_result = validate_algorithm_interface(algorithm_class, strict_validation=False)
                    
                    discovery_info = {
                        'algorithm_name': algorithm_name,
                        'class_name': class_name,
                        'module_path': module_path,
                        'metadata': algorithm_metadata,
                        'interface_valid': validation_result.is_valid,
                        'validation_errors': validation_result.errors,
                        'registration_status': 'not_registered'
                    }
                    
                    # Auto-register if enabled and validation passed
                    if auto_register and validation_result.is_valid:
                        try:
                            registration_success = register_algorithm(
                                algorithm_name=algorithm_name,
                                algorithm_class=algorithm_class,
                                algorithm_metadata=algorithm_metadata,
                                validate_interface=True
                            )
                            
                            if registration_success:
                                discovery_info['registration_status'] = 'registered'
                                discovery_results['discovery_statistics']['algorithms_registered'] += 1
                            else:
                                discovery_info['registration_status'] = 'registration_failed'
                        except Exception as reg_error:
                            discovery_info['registration_status'] = 'registration_error'
                            discovery_info['registration_error'] = str(reg_error)
                    
                    discovery_results['discovered_algorithms'][algorithm_name] = discovery_info
                    discovery_results['discovery_statistics']['algorithms_discovered'] += 1
                
            except Exception as e:
                discovery_results['discovery_statistics']['loading_failures'] += 1
                logger = get_logger('algorithm_registry', 'ALGORITHM')
                logger.warning(f"Failed to load module {module_name}: {e}")
        
        # Create audit trail entry for algorithm discovery
        create_audit_trail(
            action='ALGORITHMS_DISCOVERED',
            component='ALGORITHM_REGISTRY',
            action_details={
                'discovery_id': discovery_results['discovery_id'],
                'algorithms_directory': algorithms_directory,
                'auto_register': auto_register,
                'discovery_statistics': discovery_results['discovery_statistics']
            },
            user_context='SYSTEM'
        )
        
        # Log algorithm discovery completion
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.info(f"Algorithm discovery completed: {discovery_results['discovery_statistics']['algorithms_discovered']} algorithms discovered")
        
        return discovery_results
        
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.error(f"Algorithm discovery failed: {e}")
        return {'error': str(e)}


def _generate_comparison_recommendations(comparison_results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on algorithm comparison results."""
    recommendations = []
    
    # Analyze overall performance rankings
    if 'rankings' in comparison_results:
        best_performers = {}
        for metric_name, ranked_algorithms in comparison_results['rankings'].items():
            if ranked_algorithms:
                best_algorithm = ranked_algorithms[0][0]
                best_performers[metric_name] = best_algorithm
        
        # Find overall best performer
        algorithm_scores = {}
        for algorithm in best_performers.values():
            algorithm_scores[algorithm] = algorithm_scores.get(algorithm, 0) + 1
        
        if algorithm_scores:
            best_overall = max(algorithm_scores.items(), key=lambda x: x[1])
            recommendations.append(f"Overall best performer: {best_overall[0]} (top in {best_overall[1]} metrics)")
    
    # Check reproducibility compliance
    if 'reproducibility_analysis' in comparison_results:
        compliant_algorithms = []
        for algorithm_name, repro_data in comparison_results['reproducibility_analysis'].items():
            if (repro_data.get('meets_correlation_threshold', False) and 
                repro_data.get('meets_reproducibility_threshold', False)):
                compliant_algorithms.append(algorithm_name)
        
        if compliant_algorithms:
            recommendations.append(f"Reproducibility compliant algorithms: {', '.join(compliant_algorithms)}")
        else:
            recommendations.append("No algorithms meet reproducibility thresholds - consider parameter optimization")
    
    # Performance threshold analysis
    algorithms_compared = comparison_results.get('algorithms_compared', [])
    if len(algorithms_compared) > 1:
        recommendations.append(f"Compared {len(algorithms_compared)} algorithms - review rankings for optimal selection")
    
    return recommendations


# Auto-register dynamic algorithms if enabled
if DYNAMIC_LOADING_ENABLED:
    # Register plume tracking and hybrid strategies algorithms on module import
    try:
        register_plume_tracking_algorithm(force_reload=False, validate_integration=False)
        register_hybrid_strategies_algorithm(force_reload=False, validate_integration=False)
    except Exception as e:
        logger = get_logger('algorithm_registry', 'ALGORITHM')
        logger.warning(f"Auto-registration of dynamic algorithms failed: {e}")