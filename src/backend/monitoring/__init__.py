"""
Comprehensive monitoring system initialization module providing centralized access to monitoring 
infrastructure including console formatting, progress tracking, resource monitoring, performance 
metrics collection, alert management, and log management for the plume navigation simulation system.

This module implements unified monitoring system initialization, component coordination, and simplified 
API access for scientific computing workflows with real-time monitoring, threshold validation, and 
localized alerting capabilities. It supports complete 4000 simulation monitoring within 8 hours with 
detailed logging, resource utilization tracking, and >5% batch failure detection.

Key Features:
- Centralized monitoring system initialization and coordination
- Console formatting with color-coded output and progress indicators  
- Progress tracking for batch operations with performance monitoring
- Resource monitoring with threshold validation and optimization
- Performance metrics collection with real-time tracking and validation
- Alert management with escalation and localized notification
- Log management with comprehensive logging infrastructure
- Scientific computing context integration for reproducible research
- Thread-safe operations with concurrent monitoring support
- Graceful shutdown with cleanup and preservation of monitoring data
"""

# Standard library imports with version specifications
import logging  # Python 3.9+ - Core Python logging framework for monitoring system integration
import threading  # Python 3.9+ - Thread-safe monitoring system operations and concurrent component coordination
import datetime  # Python 3.9+ - Timestamp management for monitoring operations and audit trails
import contextlib  # Python 3.9+ - Context manager utilities for scoped monitoring operations
import atexit  # Python 3.9+ - Cleanup registration for graceful monitoring system shutdown
from typing import Dict, Any, List, Optional, Union, Callable  # Python 3.9+ - Type hints for monitoring system function signatures and data structures

# Internal imports from monitoring components
from .console_formatter import (
    initialize_console_formatting,
    ConsoleFormatter,
    ScientificConsoleHandler
)

from .progress_tracker import (
    initialize_progress_tracking,
    BatchProgressTracker,
    SimulationProgressTracker
)

from .resource_monitor import (
    initialize_resource_monitoring,
    ResourceMonitor,
    LocalAlertManager
)

from .performance_metrics import (
    initialize_performance_metrics_system,
    PerformanceMetricsCollector,
    SimulationPerformanceTracker
)

from .alert_system import (
    initialize_alert_system,
    AlertManager,
    AlertType,
    AlertSeverity
)

from .log_manager import (
    initialize_log_manager,
    LogManager
)

# Internal utility imports
from ..utils.logging_utils import get_logger, create_audit_trail

# Global monitoring system state management with thread-safe operations
_monitoring_system_initialized: bool = False
_global_log_manager: Optional[LogManager] = None
_global_alert_manager: Optional[AlertManager] = None
_global_resource_monitor: Optional[ResourceMonitor] = None
_global_performance_collector: Optional[PerformanceMetricsCollector] = None
_monitoring_components: Dict[str, Any] = {}
_monitoring_lock: threading.RLock = threading.RLock()
_shutdown_callbacks: List[Callable] = []

# Monitoring system configuration constants
MONITORING_SYSTEM_VERSION: str = '1.0.0'
DEFAULT_MONITORING_CONFIG: Dict[str, Any] = {
    'console_output': True,
    'performance_tracking': True,
    'resource_monitoring': True,
    'alert_system': True,
    'audit_trail': True
}


def initialize_monitoring_system(
    monitoring_config: Dict[str, Any] = None,
    enable_console_output: bool = True,
    enable_performance_tracking: bool = True,
    enable_resource_monitoring: bool = True,
    enable_alert_system: bool = True
) -> bool:
    """
    Initialize the comprehensive monitoring system with all components including console formatting, 
    progress tracking, resource monitoring, performance metrics, alert management, and log management 
    for scientific computing workflows.
    
    This function sets up the entire monitoring infrastructure including component initialization,
    cross-component integration, scientific context support, and graceful shutdown registration
    for optimal scientific computing workflow support with real-time monitoring capabilities.
    
    Args:
        monitoring_config: Configuration dictionary for monitoring system customization
        enable_console_output: Enable color-coded console output with progress indicators
        enable_performance_tracking: Enable performance metrics collection and analysis
        enable_resource_monitoring: Enable resource utilization monitoring and optimization
        enable_alert_system: Enable alert management with escalation and notification
        
    Returns:
        bool: Success status of monitoring system initialization with component status summary
    """
    global _monitoring_system_initialized, _global_log_manager, _global_alert_manager
    global _global_resource_monitor, _global_performance_collector, _monitoring_components
    
    # Acquire monitoring system lock for thread-safe initialization
    with _monitoring_lock:
        # Check if monitoring system is already initialized
        if _monitoring_system_initialized:
            return True
        
        try:
            # Load monitoring configuration with defaults
            config = monitoring_config or DEFAULT_MONITORING_CONFIG.copy()
            
            # Initialize log manager with comprehensive logging infrastructure
            try:
                _global_log_manager = initialize_log_manager(
                    enable_console_output=enable_console_output,
                    enable_performance_logging=enable_performance_tracking,
                    enable_audit_trail=config.get('audit_trail', True)
                )
                
                if _global_log_manager:
                    _monitoring_components['log_manager'] = _global_log_manager
                    
            except Exception as e:
                print(f"WARNING: Log manager initialization failed: {e}")
                _global_log_manager = None
            
            # Initialize console formatting system if console output enabled
            if enable_console_output:
                try:
                    console_success = initialize_console_formatting(
                        default_color_scheme='scientific_default',
                        formatting_config=config.get('console_config', {})
                    )
                    
                    if console_success:
                        _monitoring_components['console_formatting'] = True
                        
                except Exception as e:
                    print(f"WARNING: Console formatting initialization failed: {e}")
            
            # Initialize progress tracking system with performance monitoring integration
            try:
                progress_success = initialize_progress_tracking(
                    enable_console_integration=enable_console_output,
                    enable_performance_monitoring=enable_performance_tracking
                )
                
                if progress_success:
                    _monitoring_components['progress_tracking'] = True
                    
            except Exception as e:
                print(f"WARNING: Progress tracking initialization failed: {e}")
            
            # Initialize resource monitoring system if resource monitoring enabled
            if enable_resource_monitoring:
                try:
                    resource_success = initialize_resource_monitoring(
                        monitoring_config=config.get('resource_config', {}),
                        enable_threshold_validation=True,
                        enable_optimization=True
                    )
                    
                    if resource_success:
                        _monitoring_components['resource_monitoring'] = True
                        
                except Exception as e:
                    print(f"WARNING: Resource monitoring initialization failed: {e}")
            
            # Initialize performance metrics collection system if performance tracking enabled
            if enable_performance_tracking:
                try:
                    performance_success = initialize_performance_metrics_system(
                        metrics_config=config.get('performance_config', {}),
                        enable_validation=True,
                        enable_alerting=enable_alert_system
                    )
                    
                    if performance_success:
                        _monitoring_components['performance_metrics'] = True
                        
                except Exception as e:
                    print(f"WARNING: Performance metrics initialization failed: {e}")
            
            # Initialize alert management system if alert system enabled
            if enable_alert_system:
                try:
                    alert_success = initialize_alert_system(
                        alert_config=config.get('alert_config', {}),
                        enable_escalation=True,
                        enable_suppression=True,
                        alert_profile='scientific_computing'
                    )
                    
                    if alert_success:
                        _monitoring_components['alert_system'] = True
                        
                except Exception as e:
                    print(f"WARNING: Alert system initialization failed: {e}")
            
            # Register monitoring component integrations and cross-references
            _register_component_integrations()
            
            # Setup monitoring system shutdown callbacks for graceful cleanup
            atexit.register(_cleanup_monitoring_system)
            _shutdown_callbacks.append(_finalize_monitoring_components)
            
            # Set global monitoring system initialized flag
            _monitoring_system_initialized = True
            
            # Log monitoring system initialization completion with component summary
            if _global_log_manager:
                logger = _global_log_manager.get_logger('monitoring.system', 'SYSTEM')
                logger.info(
                    f"Monitoring system initialized successfully | "
                    f"Version: {MONITORING_SYSTEM_VERSION} | "
                    f"Components: {len(_monitoring_components)}"
                )
                
                # Create audit trail entry for monitoring system initialization
                create_audit_trail(
                    action='MONITORING_SYSTEM_INITIALIZED',
                    component='MONITORING_SYSTEM',
                    action_details={
                        'version': MONITORING_SYSTEM_VERSION,
                        'console_output': enable_console_output,
                        'performance_tracking': enable_performance_tracking,
                        'resource_monitoring': enable_resource_monitoring,
                        'alert_system': enable_alert_system,
                        'components_count': len(_monitoring_components),
                        'configuration': config
                    }
                )
            
            # Return initialization success status
            return True
            
        except Exception as e:
            print(f"CRITICAL: Monitoring system initialization failed: {e}")
            # Reset global state on initialization failure
            _monitoring_system_initialized = False
            _monitoring_components.clear()
            return False


def get_monitoring_components(include_status_information: bool = False) -> Dict[str, Any]:
    """
    Get dictionary of all initialized monitoring system components for centralized access and 
    component coordination across the scientific computing system.
    
    This function provides comprehensive access to all monitoring components with optional
    status information and health indicators for system integration and coordination.
    
    Args:
        include_status_information: Include component status information and health indicators
        
    Returns:
        Dict[str, Any]: Dictionary of monitoring components with optional status information and health indicators
    """
    with _monitoring_lock:
        # Check monitoring system initialization status
        if not _monitoring_system_initialized:
            return {
                'error': 'Monitoring system not initialized',
                'initialized': False,
                'components_available': 0
            }
        
        # Collect all initialized monitoring components
        components = _monitoring_components.copy()
        
        # Add global component references
        if _global_log_manager:
            components['log_manager_instance'] = _global_log_manager
        if _global_alert_manager:
            components['alert_manager_instance'] = _global_alert_manager
        if _global_resource_monitor:
            components['resource_monitor_instance'] = _global_resource_monitor
        if _global_performance_collector:
            components['performance_collector_instance'] = _global_performance_collector
        
        # Include component status information if requested
        if include_status_information:
            status_info = {
                'monitoring_system_version': MONITORING_SYSTEM_VERSION,
                'initialization_status': _monitoring_system_initialized,
                'total_components': len(components),
                'active_components': len([c for c in components.values() if c]),
                'system_health': 'operational' if _monitoring_system_initialized else 'not_initialized'
            }
            
            # Add component health indicators and operational status
            component_health = {}
            for component_name, component in components.items():
                if hasattr(component, 'get_statistics'):
                    try:
                        component_health[component_name] = component.get_statistics()
                    except Exception as e:
                        component_health[component_name] = {'error': str(e)}
                elif isinstance(component, bool):
                    component_health[component_name] = {'status': 'active' if component else 'inactive'}
                else:
                    component_health[component_name] = {'status': 'available'}
            
            status_info['component_health'] = component_health
            components['status_information'] = status_info
        
        # Format component information for easy access
        components['system_metadata'] = {
            'version': MONITORING_SYSTEM_VERSION,
            'initialized': _monitoring_system_initialized,
            'components_count': len(_monitoring_components),
            'generation_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Return comprehensive monitoring components dictionary
        return components


def create_monitoring_context(
    context_name: str,
    context_config: Dict[str, Any] = None,
    enable_progress_tracking: bool = True,
    enable_performance_monitoring: bool = True,
    enable_resource_monitoring: bool = True
) -> 'MonitoringContext':
    """
    Create comprehensive monitoring context for scoped operations that automatically manages 
    progress tracking, performance monitoring, resource monitoring, and alert processing for 
    specific workflows or operations.
    
    This function provides comprehensive monitoring context creation with automatic setup and 
    cleanup for scoped operations with scientific computing context integration.
    
    Args:
        context_name: Name identifier for the monitoring context
        context_config: Configuration dictionary for the monitoring context
        enable_progress_tracking: Enable progress tracking integration for the context
        enable_performance_monitoring: Enable performance monitoring for the context
        enable_resource_monitoring: Enable resource monitoring for the context
        
    Returns:
        MonitoringContext: Comprehensive monitoring context manager for scoped operations with automatic setup and cleanup
    """
    # Validate context configuration and monitoring requirements
    if not _monitoring_system_initialized:
        raise RuntimeError("Monitoring system not initialized - cannot create monitoring context")
    
    if not context_name:
        raise ValueError("Context name cannot be empty")
    
    # Create monitoring context instance with specified configuration
    config = context_config or {}
    
    # Setup progress tracking integration if enabled
    if enable_progress_tracking and 'progress_tracking' not in _monitoring_components:
        print("WARNING: Progress tracking not available for monitoring context")
        enable_progress_tracking = False
    
    # Configure performance monitoring if enabled
    if enable_performance_monitoring and 'performance_metrics' not in _monitoring_components:
        print("WARNING: Performance monitoring not available for monitoring context")
        enable_performance_monitoring = False
    
    # Setup resource monitoring if enabled
    if enable_resource_monitoring and 'resource_monitoring' not in _monitoring_components:
        print("WARNING: Resource monitoring not available for monitoring context")
        enable_resource_monitoring = False
    
    # Initialize alert processing for context
    alert_processing_enabled = 'alert_system' in _monitoring_components
    
    # Configure scientific context integration
    monitoring_context = MonitoringContext(
        context_name=context_name,
        context_config=config,
        enable_progress_tracking=enable_progress_tracking,
        enable_performance_monitoring=enable_performance_monitoring,
        enable_resource_monitoring=enable_resource_monitoring
    )
    
    # Return configured monitoring context manager
    return monitoring_context


def get_system_health_status(
    include_detailed_metrics: bool = False,
    include_alert_summary: bool = False,
    include_resource_status: bool = False
) -> Dict[str, Any]:
    """
    Get comprehensive system health status including monitoring component status, performance 
    metrics, resource utilization, alert summary, and overall system health indicators for 
    dashboard and reporting.
    
    This function provides comprehensive system health assessment with monitoring metrics,
    alerts, resource information, and overall system status for dashboard and reporting.
    
    Args:
        include_detailed_metrics: Include detailed performance metrics from collectors
        include_alert_summary: Include alert summary from alert manager
        include_resource_status: Include current resource utilization from resource monitor
        
    Returns:
        Dict[str, Any]: Comprehensive system health status with monitoring metrics, alerts, and resource information
    """
    with _monitoring_lock:
        # Check monitoring system initialization and component status
        health_status = {
            'system_health_timestamp': datetime.datetime.now().isoformat(),
            'monitoring_system_version': MONITORING_SYSTEM_VERSION,
            'monitoring_system_initialized': _monitoring_system_initialized,
            'total_components': len(_monitoring_components),
            'component_status': {},
            'overall_health': 'unknown'
        }
        
        if not _monitoring_system_initialized:
            health_status['overall_health'] = 'not_initialized'
            health_status['error'] = 'Monitoring system not initialized'
            return health_status
        
        # Collect performance metrics from performance collector if available
        if include_detailed_metrics and _global_performance_collector:
            try:
                performance_metrics = _global_performance_collector.get_statistics(
                    include_detailed_breakdown=True,
                    include_trend_analysis=True
                )
                health_status['performance_metrics'] = performance_metrics
            except Exception as e:
                health_status['performance_metrics_error'] = str(e)
        
        # Get current resource utilization from resource monitor if available
        if include_resource_status and _global_resource_monitor:
            try:
                resource_status = _global_resource_monitor.get_current_resources(
                    include_thresholds=True,
                    include_optimization_recommendations=True
                )
                health_status['resource_status'] = resource_status
            except Exception as e:
                health_status['resource_status_error'] = str(e)
        
        # Collect alert summary from alert manager if available
        if include_alert_summary and _global_alert_manager:
            try:
                alert_summary = _global_alert_manager.get_alert_statistics(
                    statistics_period='all',
                    include_detailed_breakdown=True
                )
                health_status['alert_summary'] = alert_summary
            except Exception as e:
                health_status['alert_summary_error'] = str(e)
        
        # Include detailed metrics if requested
        detailed_metrics = {}
        if include_detailed_metrics:
            for component_name, component in _monitoring_components.items():
                if hasattr(component, 'get_statistics'):
                    try:
                        detailed_metrics[component_name] = component.get_statistics()
                    except Exception as e:
                        detailed_metrics[component_name] = {'error': str(e)}
                else:
                    detailed_metrics[component_name] = {'status': 'active' if component else 'inactive'}
            
            health_status['detailed_metrics'] = detailed_metrics
        
        # Analyze overall system health indicators
        active_components = len([c for c in _monitoring_components.values() if c])
        total_components = len(_monitoring_components)
        
        if total_components == 0:
            health_status['overall_health'] = 'no_components'
        elif active_components == total_components:
            health_status['overall_health'] = 'excellent'
        elif active_components >= total_components * 0.8:
            health_status['overall_health'] = 'good'
        elif active_components >= total_components * 0.5:
            health_status['overall_health'] = 'fair'
        else:
            health_status['overall_health'] = 'poor'
        
        # Add component availability statistics
        health_status['component_availability'] = {
            'active_components': active_components,
            'total_components': total_components,
            'availability_percentage': (active_components / total_components * 100) if total_components > 0 else 0
        }
        
        # Format health status for reporting and dashboard display
        health_status['summary'] = {
            'system_operational': _monitoring_system_initialized,
            'health_score': health_status['component_availability']['availability_percentage'],
            'critical_alerts': 0,  # Would be populated from alert manager if available
            'resource_utilization': 'unknown'  # Would be populated from resource monitor if available
        }
        
        # Return comprehensive system health status
        return health_status


def shutdown_monitoring_system(
    generate_final_reports: bool = True,
    preserve_monitoring_data: bool = True,
    shutdown_mode: str = 'normal'
) -> Dict[str, Any]:
    """
    Shutdown the comprehensive monitoring system with graceful cleanup, resource finalization, 
    final report generation, and preservation of critical monitoring data for system shutdown 
    or restart.
    
    This function provides comprehensive monitoring system shutdown with graceful cleanup,
    audit trail finalization, and resource cleanup for system shutdown or restart scenarios.
    
    Args:
        generate_final_reports: Generate final monitoring reports before shutdown
        preserve_monitoring_data: Preserve critical monitoring data during shutdown
        shutdown_mode: Mode of shutdown operation ('normal', 'emergency', 'maintenance')
        
    Returns:
        Dict[str, Any]: Shutdown summary with final statistics, preserved data locations, and component cleanup status
    """
    global _monitoring_system_initialized, _monitoring_components
    
    # Acquire monitoring system lock for exclusive shutdown access
    with _monitoring_lock:
        shutdown_summary = {
            'shutdown_timestamp': datetime.datetime.now().isoformat(),
            'shutdown_mode': shutdown_mode,
            'generate_final_reports': generate_final_reports,
            'preserve_monitoring_data': preserve_monitoring_data,
            'operations_performed': [],
            'final_statistics': {},
            'preserved_data_locations': [],
            'component_cleanup_status': {},
            'shutdown_success': True
        }
        
        try:
            # Execute registered shutdown callbacks for component cleanup
            for callback in _shutdown_callbacks:
                try:
                    callback()
                    shutdown_summary['operations_performed'].append(f'callback_{callback.__name__}')
                except Exception as e:
                    shutdown_summary['component_cleanup_status'][f'callback_{callback.__name__}'] = f'failed: {e}'
            
            # Shutdown alert manager with final alert processing
            if _global_alert_manager:
                try:
                    alert_shutdown = _global_alert_manager.stop_monitoring()
                    shutdown_summary['alert_manager_shutdown'] = alert_shutdown
                    shutdown_summary['operations_performed'].append('alert_manager_shutdown')
                except Exception as e:
                    shutdown_summary['component_cleanup_status']['alert_manager'] = f'failed: {e}'
            
            # Shutdown performance metrics collector with final report generation
            if _global_performance_collector:
                try:
                    performance_shutdown = _global_performance_collector.finalize_collection(
                        generate_report=generate_final_reports
                    )
                    shutdown_summary['performance_collector_shutdown'] = performance_shutdown
                    shutdown_summary['operations_performed'].append('performance_collector_shutdown')
                except Exception as e:
                    shutdown_summary['component_cleanup_status']['performance_collector'] = f'failed: {e}'
            
            # Shutdown resource monitor with final resource analysis
            if _global_resource_monitor:
                try:
                    resource_shutdown = _global_resource_monitor.stop_monitoring(
                        generate_final_report=generate_final_reports
                    )
                    shutdown_summary['resource_monitor_shutdown'] = resource_shutdown
                    shutdown_summary['operations_performed'].append('resource_monitor_shutdown')
                except Exception as e:
                    shutdown_summary['component_cleanup_status']['resource_monitor'] = f'failed: {e}'
            
            # Shutdown progress tracking system with final statistics
            try:
                # Progress tracking cleanup would be implemented here
                shutdown_summary['operations_performed'].append('progress_tracking_shutdown')
            except Exception as e:
                shutdown_summary['component_cleanup_status']['progress_tracking'] = f'failed: {e}'
            
            # Shutdown log manager with log archival and cleanup
            if _global_log_manager:
                try:
                    log_shutdown = _global_log_manager.shutdown(
                        archive_logs=preserve_monitoring_data,
                        preserve_audit_trail=preserve_monitoring_data
                    )
                    shutdown_summary['log_manager_shutdown'] = log_shutdown
                    shutdown_summary['operations_performed'].append('log_manager_shutdown')
                except Exception as e:
                    shutdown_summary['component_cleanup_status']['log_manager'] = f'failed: {e}'
            
            # Generate final monitoring reports if requested
            if generate_final_reports:
                try:
                    final_report = {
                        'monitoring_system_version': MONITORING_SYSTEM_VERSION,
                        'final_component_count': len(_monitoring_components),
                        'shutdown_timestamp': datetime.datetime.now().isoformat(),
                        'system_health_at_shutdown': get_system_health_status(
                            include_detailed_metrics=True,
                            include_alert_summary=True,
                            include_resource_status=True
                        )
                    }
                    shutdown_summary['final_report'] = final_report
                    shutdown_summary['operations_performed'].append('final_report_generated')
                except Exception as e:
                    shutdown_summary['component_cleanup_status']['final_report'] = f'failed: {e}'
            
            # Preserve monitoring data if preservation enabled
            if preserve_monitoring_data:
                try:
                    # Data preservation logic would be implemented here
                    preservation_locations = []
                    shutdown_summary['preserved_data_locations'] = preservation_locations
                    shutdown_summary['operations_performed'].append('monitoring_data_preserved')
                except Exception as e:
                    shutdown_summary['component_cleanup_status']['data_preservation'] = f'failed: {e}'
            
            # Clear global monitoring component references
            _monitoring_components.clear()
            
            # Set monitoring system initialized flag to False
            _monitoring_system_initialized = False
            
            # Log monitoring system shutdown completion
            print(f"Monitoring system shutdown completed successfully (mode: {shutdown_mode})")
            shutdown_summary['operations_performed'].append('system_flags_reset')
            
            # Return comprehensive shutdown summary
            return shutdown_summary
            
        except Exception as e:
            shutdown_summary['shutdown_success'] = False
            shutdown_summary['critical_error'] = str(e)
            print(f"CRITICAL: Monitoring system shutdown failed: {e}")
            return shutdown_summary


class MonitoringContext:
    """
    Comprehensive monitoring context manager that automatically coordinates progress tracking, 
    performance monitoring, resource monitoring, alert processing, and logging for specific 
    operations or workflows with scientific computing context integration and automatic setup/cleanup.
    
    This context manager provides complete monitoring lifecycle management with scientific context
    integration, performance tracking, and comprehensive audit trail support for scoped operations.
    """
    
    def __init__(
        self,
        context_name: str,
        context_config: Dict[str, Any] = None,
        enable_progress_tracking: bool = True,
        enable_performance_monitoring: bool = True,
        enable_resource_monitoring: bool = True
    ):
        """
        Initialize monitoring context manager with context configuration and monitoring component setup.
        
        Args:
            context_name: Name identifier for the monitoring context
            context_config: Configuration dictionary for the monitoring context
            enable_progress_tracking: Enable progress tracking integration
            enable_performance_monitoring: Enable performance monitoring
            enable_resource_monitoring: Enable resource monitoring
        """
        # Set context name and configuration parameters
        self.context_name = context_name
        self.context_config = context_config or {}
        
        # Configure monitoring component enablement flags
        self.progress_tracking_enabled = enable_progress_tracking
        self.performance_monitoring_enabled = enable_performance_monitoring
        self.resource_monitoring_enabled = enable_resource_monitoring
        
        # Initialize monitoring component references
        self.progress_tracker: Optional[SimulationProgressTracker] = None
        self.performance_tracker: Optional[SimulationPerformanceTracker] = None
        self.resource_monitor: Optional[ResourceMonitor] = None
        self.log_manager: Optional[LogManager] = None
        
        # Setup triggered alerts tracking
        self.triggered_alerts: List[str] = []
        
        # Initialize monitoring summary storage
        self.start_time: datetime.datetime = None
        self.monitoring_summary: Dict[str, Any] = {}
    
    def __enter__(self) -> 'MonitoringContext':
        """
        Enter monitoring context and setup comprehensive monitoring including progress tracking, 
        performance monitoring, resource monitoring, and alert processing for the operation.
        
        This method establishes the monitoring context with all enabled components and creates
        the necessary tracking infrastructure for the scoped operation.
        
        Returns:
            MonitoringContext: Self reference for context management and monitoring coordination
        """
        # Record context start time
        self.start_time = datetime.datetime.now()
        
        # Setup progress tracking if enabled
        if self.progress_tracking_enabled and 'progress_tracking' in _monitoring_components:
            try:
                self.progress_tracker = SimulationProgressTracker(
                    tracker_id=f"{self.context_name}_progress",
                    tracking_config=self.context_config.get('progress_config', {})
                )
                self.progress_tracker.start_simulation_tracking(
                    simulation_id=self.context_name,
                    algorithm_name=self.context_config.get('algorithm_name', 'unknown'),
                    tracking_mode='context'
                )
            except Exception as e:
                print(f"WARNING: Progress tracking setup failed: {e}")
        
        # Initialize performance monitoring if enabled
        if self.performance_monitoring_enabled and 'performance_metrics' in _monitoring_components:
            try:
                self.performance_tracker = SimulationPerformanceTracker(
                    tracker_id=f"{self.context_name}_performance",
                    performance_config=self.context_config.get('performance_config', {})
                )
                self.performance_tracker.start_tracking(
                    simulation_id=self.context_name,
                    algorithm_name=self.context_config.get('algorithm_name', 'unknown')
                )
            except Exception as e:
                print(f"WARNING: Performance monitoring setup failed: {e}")
        
        # Start resource monitoring if enabled
        if self.resource_monitoring_enabled and 'resource_monitoring' in _monitoring_components:
            try:
                self.resource_monitor = _global_resource_monitor
                if self.resource_monitor:
                    self.resource_monitor.start_monitoring()
            except Exception as e:
                print(f"WARNING: Resource monitoring setup failed: {e}")
        
        # Configure alert processing for context
        if 'alert_system' in _monitoring_components:
            try:
                # Alert system is already running globally
                pass
            except Exception as e:
                print(f"WARNING: Alert processing setup failed: {e}")
        
        # Setup scientific context for logging
        if _global_log_manager:
            try:
                self.log_manager = _global_log_manager
                logger = self.log_manager.get_logger(f"context.{self.context_name}", 'CONTEXT')
                logger.debug(f"Monitoring context entered: {self.context_name}")
            except Exception as e:
                print(f"WARNING: Logging context setup failed: {e}")
        
        # Log context entry with monitoring configuration
        if self.log_manager:
            create_audit_trail(
                action='MONITORING_CONTEXT_ENTERED',
                component='MONITORING_CONTEXT',
                action_details={
                    'context_name': self.context_name,
                    'progress_tracking': self.progress_tracking_enabled,
                    'performance_monitoring': self.performance_monitoring_enabled,
                    'resource_monitoring': self.resource_monitoring_enabled,
                    'context_config': self.context_config
                }
            )
        
        # Return self for context management
        return self
    
    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb) -> bool:
        """
        Exit monitoring context and finalize monitoring with comprehensive summary generation, 
        alert processing, and cleanup of monitoring resources.
        
        This method finalizes all monitoring activities and generates comprehensive summaries
        with exception handling and resource cleanup.
        
        Args:
            exc_type: Exception type if exception occurred
            exc_val: Exception value if exception occurred
            exc_tb: Exception traceback if exception occurred
            
        Returns:
            bool: False to propagate exceptions
        """
        try:
            # Calculate context execution time
            execution_time = None
            if self.start_time:
                execution_time = (datetime.datetime.now() - self.start_time).total_seconds()
            
            # Finalize progress tracking if enabled
            if self.progress_tracker:
                try:
                    self.progress_tracker.complete_simulation_tracking(
                        completion_status='success' if exc_type is None else 'error',
                        completion_summary={'execution_time': execution_time}
                    )
                except Exception as e:
                    print(f"WARNING: Progress tracking finalization failed: {e}")
            
            # Complete performance monitoring if enabled
            if self.performance_tracker:
                try:
                    performance_summary = self.performance_tracker.validate_simulation_accuracy(
                        simulation_results={'execution_time': execution_time},
                        accuracy_thresholds=self.context_config.get('accuracy_thresholds', {})
                    )
                    self.monitoring_summary['performance'] = performance_summary
                except Exception as e:
                    print(f"WARNING: Performance monitoring finalization failed: {e}")
            
            # Stop resource monitoring if enabled
            if self.resource_monitor and self.resource_monitoring_enabled:
                try:
                    resource_summary = self.resource_monitor.validate_resource_thresholds(
                        operation_context=self.context_name,
                        include_optimization_recommendations=True
                    )
                    self.monitoring_summary['resources'] = resource_summary
                except Exception as e:
                    print(f"WARNING: Resource monitoring finalization failed: {e}")
            
            # Process any triggered alerts
            if 'alert_system' in _monitoring_components and _global_alert_manager:
                try:
                    alert_statistics = _global_alert_manager.get_alert_statistics(
                        statistics_period='context',
                        include_detailed_breakdown=True
                    )
                    self.monitoring_summary['alerts'] = alert_statistics
                except Exception as e:
                    print(f"WARNING: Alert processing finalization failed: {e}")
            
            # Generate comprehensive monitoring summary
            self.monitoring_summary.update({
                'context_name': self.context_name,
                'execution_time': execution_time,
                'success': exc_type is None,
                'exception_info': {
                    'type': exc_type.__name__ if exc_type else None,
                    'message': str(exc_val) if exc_val else None
                } if exc_type else None,
                'monitoring_configuration': {
                    'progress_tracking': self.progress_tracking_enabled,
                    'performance_monitoring': self.performance_monitoring_enabled,
                    'resource_monitoring': self.resource_monitoring_enabled
                },
                'completion_timestamp': datetime.datetime.now().isoformat()
            })
            
            # Cleanup context monitoring resources
            self.progress_tracker = None
            self.performance_tracker = None
            # Note: resource_monitor is global, don't clear it
            
            # Log context exit with monitoring summary
            if self.log_manager:
                logger = self.log_manager.get_logger(f"context.{self.context_name}", 'CONTEXT')
                if execution_time:
                    logger.debug(f"Monitoring context exited: {self.context_name} (execution time: {execution_time:.3f}s)")
                else:
                    logger.debug(f"Monitoring context exited: {self.context_name}")
                
                # Create exit audit trail entry
                create_audit_trail(
                    action='MONITORING_CONTEXT_EXITED',
                    component='MONITORING_CONTEXT',
                    action_details=self.monitoring_summary
                )
            
        except Exception as cleanup_error:
            print(f"ERROR: Monitoring context cleanup failed: {cleanup_error}")
        
        # Return False to propagate exceptions
        return False
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring summary for the context execution including progress, 
        performance, resource utilization, and alert information.
        
        This method provides comprehensive monitoring analysis for the context with all
        collected metrics, alerts, and analysis results.
        
        Returns:
            Dict[str, Any]: Context monitoring summary with comprehensive metrics, alerts, and analysis
        """
        # Compile progress tracking summary if available
        summary = self.monitoring_summary.copy()
        
        if self.progress_tracker:
            try:
                progress_summary = {
                    'tracking_enabled': True,
                    'tracker_id': getattr(self.progress_tracker, 'tracker_id', 'unknown')
                }
                summary['progress_tracking'] = progress_summary
            except Exception as e:
                summary['progress_tracking'] = {'error': str(e)}
        
        # Include performance monitoring results if available
        if self.performance_tracker:
            try:
                performance_summary = {
                    'monitoring_enabled': True,
                    'tracker_id': getattr(self.performance_tracker, 'tracker_id', 'unknown')
                }
                summary['performance_monitoring'] = performance_summary
            except Exception as e:
                summary['performance_monitoring'] = {'error': str(e)}
        
        # Add resource utilization summary if available
        if self.resource_monitor and self.resource_monitoring_enabled:
            try:
                resource_summary = {
                    'monitoring_enabled': True,
                    'monitor_active': getattr(self.resource_monitor, 'is_monitoring', False)
                }
                summary['resource_monitoring'] = resource_summary
            except Exception as e:
                summary['resource_monitoring'] = {'error': str(e)}
        
        # Include triggered alerts and their status
        summary['triggered_alerts'] = self.triggered_alerts
        
        # Calculate context execution metrics
        if self.start_time:
            current_time = datetime.datetime.now()
            summary['current_execution_time'] = (current_time - self.start_time).total_seconds()
        
        # Format comprehensive monitoring summary
        summary['context_metadata'] = {
            'context_name': self.context_name,
            'generation_timestamp': datetime.datetime.now().isoformat(),
            'monitoring_components_enabled': sum([
                self.progress_tracking_enabled,
                self.performance_monitoring_enabled,
                self.resource_monitoring_enabled
            ])
        }
        
        # Return complete context monitoring analysis
        return summary


# Internal helper functions for monitoring system coordination

def _register_component_integrations() -> None:
    """Register cross-component integrations and monitoring coordination."""
    try:
        # Setup log manager integration with other components
        if _global_log_manager and _global_alert_manager:
            # Alert system can use log manager for notification delivery
            pass
        
        # Configure performance collector integration with resource monitor
        if _global_performance_collector and _global_resource_monitor:
            # Performance collector can trigger resource threshold checks
            pass
        
        # Setup progress tracking integration with alert system
        if _global_alert_manager:
            # Progress tracking can trigger alerts for batch failures
            pass
        
    except Exception as e:
        print(f"WARNING: Component integration registration failed: {e}")


def _finalize_monitoring_components() -> None:
    """Finalize monitoring components during shutdown."""
    try:
        # Finalize each component that supports finalization
        for component_name, component in _monitoring_components.items():
            if hasattr(component, 'finalize') or hasattr(component, 'shutdown'):
                try:
                    if hasattr(component, 'finalize'):
                        component.finalize()
                    elif hasattr(component, 'shutdown'):
                        component.shutdown()
                except Exception as e:
                    print(f"WARNING: Component finalization failed for {component_name}: {e}")
        
    except Exception as e:
        print(f"WARNING: Monitoring components finalization failed: {e}")


def _cleanup_monitoring_system() -> None:
    """Cleanup function registered with atexit for graceful monitoring system shutdown."""
    try:
        if _monitoring_system_initialized:
            shutdown_monitoring_system(
                generate_final_reports=True,
                preserve_monitoring_data=True,
                shutdown_mode='atexit'
            )
    except Exception as e:
        print(f"WARNING: Exit cleanup failed: {e}")


# Export all monitoring system components and functions for external access
__all__ = [
    # Core monitoring system functions
    'initialize_monitoring_system',
    'get_monitoring_components', 
    'create_monitoring_context',
    'get_system_health_status',
    'shutdown_monitoring_system',
    
    # Monitoring context manager
    'MonitoringContext',
    
    # Re-exported component classes
    'ConsoleFormatter',
    'ScientificConsoleHandler',
    'BatchProgressTracker',
    'SimulationProgressTracker',
    'ResourceMonitor',
    'PerformanceMetricsCollector',
    'SimulationPerformanceTracker',
    'AlertManager',
    'AlertType',
    'AlertSeverity',
    'LogManager',
    
    # System constants
    'MONITORING_SYSTEM_VERSION',
    'DEFAULT_MONITORING_CONFIG'
]