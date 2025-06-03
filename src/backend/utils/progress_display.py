"""
Comprehensive progress display utilities module providing advanced terminal-based progress visualization, 
status indicators, performance metrics display, and interactive console formatting for the plume navigation 
simulation system.

This module implements ASCII progress bars, hierarchical status trees, real-time counters, scientific value 
formatting, and color-coded status displays optimized for batch processing operations with 4000+ simulations, 
supporting both interactive and non-interactive terminal environments with responsive formatting and scientific 
computing presentation standards.

Key Features:
- Advanced ASCII progress bars with customizable styling and performance metrics integration
- Hierarchical status displays with tree-like organization for complex operations
- Real-time performance metrics visualization with threshold indicators and color coding
- Scientific value formatting with appropriate precision and units
- Terminal management with cursor control and screen clearing capabilities
- Thread-safe operations for concurrent progress tracking and updates
- Responsive formatting that adapts to different terminal sizes and capabilities
- Color-coded status displays with consistent visual indicators for scientific workflows
"""

import sys  # Python 3.9+ - System interface for stdout/stderr management and terminal output control
import os  # Python 3.9+ - Operating system interface for environment variables and terminal detection
import shutil  # Python 3.9+ - Terminal size detection for responsive progress bar formatting
import time  # Python 3.9+ - High-precision timing for progress rate calculations and ETA estimation
import datetime  # Python 3.9+ - Timestamp formatting and duration calculations for progress tracking
import threading  # Python 3.9+ - Thread-safe progress display updates and concurrent progress tracking
import math  # Python 3.9+ - Mathematical calculations for progress percentages and rate calculations
from typing import Dict, Any, List, Optional, Union, Tuple  # Python 3.9+ - Type hints for progress display function signatures and data structures
from dataclasses import dataclass, field  # Python 3.9+ - Data classes for progress state and display configuration storage
from enum import Enum, auto  # Python 3.9+ - Enumeration types for progress states and display modes
import re  # Python 3.9+ - Regular expression processing for text formatting and ANSI escape sequence handling
import collections  # Python 3.9+ - Efficient data structures for progress history and status tracking
import json  # Python 3.9+ - JSON parsing for loading performance thresholds configuration

# Import internal logging utilities for scientific context and value formatting
from .logging_utils import (
    get_logger,
    format_scientific_value,
    detect_terminal_capabilities,
    ScientificContextFilter
)

# Import performance monitoring utilities (with fallback handling)
try:
    from .performance_monitoring import collect_system_metrics, PerformanceMonitor
except ImportError:
    # Fallback implementations for missing performance monitoring module
    def collect_system_metrics() -> Dict[str, float]:
        """Fallback implementation for system metrics collection."""
        return {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_io': 0.0,
            'network_io': 0.0
        }
    
    class PerformanceMonitor:
        """Fallback implementation for performance monitoring."""
        def __init__(self):
            pass
        
        def get_current_metrics(self) -> Dict[str, float]:
            """Fallback implementation for current metrics retrieval."""
            return collect_system_metrics()

# Terminal color constants for consistent color coding across progress displays
TERMINAL_COLORS: Dict[str, str] = {
    'GREEN': '\033[92m',    # Successful operations, completed simulations
    'YELLOW': '\033[93m',   # Warnings, non-critical issues
    'RED': '\033[91m',      # Errors, failed simulations
    'BLUE': '\033[94m',     # Information, status updates
    'CYAN': '\033[96m',     # File paths, configuration values
    'WHITE': '\033[97m',    # Default text
    'BOLD': '\033[1m',      # Emphasis
    'DIM': '\033[2m',       # Secondary information
    'RESET': '\033[0m'      # Reset formatting
}

# Status icon constants for visual status indicators in progress displays
STATUS_ICONS: Dict[str, str] = {
    'SUCCESS': '✓',         # Successful operations completion
    'WARNING': '⚠',         # Warning conditions and non-critical issues
    'ERROR': '✗',           # Error conditions and failures
    'INFO': 'ℹ',            # Information and status updates
    'PROGRESS': '▶',        # Operations in progress
    'COMPLETE': '✓',        # Completed operations
    'FAILED': '✗',          # Failed operations
    'PENDING': '○',         # Pending operations
    'RUNNING': '●'          # Currently running operations
}

# Progress bar character constants for ASCII progress visualization
PROGRESS_BAR_CHARS: Dict[str, Union[str, List[str]]] = {
    'FILLED': '█',          # Filled portion of progress bar
    'EMPTY': '░',           # Empty portion of progress bar
    'PARTIAL': ['▏', '▎', '▍', '▌', '▋', '▊', '▉'],  # Partial fill characters
    'SPINNER': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']  # Spinner animation characters
}

# Progress bar configuration constants
DEFAULT_PROGRESS_BAR_WIDTH: int = 40
MIN_PROGRESS_BAR_WIDTH: int = 20
MAX_PROGRESS_BAR_WIDTH: int = 80
DEFAULT_TERMINAL_WIDTH: int = 80

# Scientific formatting configuration
SCIENTIFIC_PRECISION_DIGITS: int = 3

# Table formatting configuration
TABLE_COLUMN_PADDING: int = 2

# Update interval configuration for animations and status updates
STATUS_UPDATE_INTERVAL_SECONDS: float = 0.1
PROGRESS_ANIMATION_INTERVAL_SECONDS: float = 0.2

# Global state variables for terminal capabilities and configuration
_terminal_capabilities: Optional[Dict[str, Any]] = None
_color_support_enabled: bool = True
_unicode_support_enabled: bool = True
_terminal_width: int = 80

# Thread-safe global state management
_progress_display_lock: threading.Lock = threading.Lock()
_active_progress_bars: Dict[str, 'ProgressBar'] = {}
_status_displays: Dict[str, 'StatusDisplay'] = {}
_performance_thresholds: Dict[str, Any] = {}


class ProgressState(Enum):
    """Enumeration for progress bar states and status indicators."""
    PENDING = auto()        # Progress bar is initialized but not started
    RUNNING = auto()        # Progress bar is actively updating
    PAUSED = auto()         # Progress bar is temporarily paused
    COMPLETE = auto()       # Progress bar has completed successfully
    FAILED = auto()         # Progress bar has failed with errors
    CANCELLED = auto()      # Progress bar was cancelled by user


class DisplayMode(Enum):
    """Enumeration for display mode configurations and formatting options."""
    INTERACTIVE = auto()    # Interactive terminal with full color and animation support
    NON_INTERACTIVE = auto() # Non-interactive terminal with simplified output
    COMPACT = auto()        # Compact display mode for minimal screen space
    VERBOSE = auto()        # Verbose display mode with detailed information
    SCIENTIFIC = auto()     # Scientific display mode with precise metrics


@dataclass
class ProgressBarConfig:
    """Configuration data class for progress bar appearance and behavior settings."""
    bar_width: int = DEFAULT_PROGRESS_BAR_WIDTH
    show_percentage: bool = True
    show_eta: bool = True
    show_rate: bool = True
    color_scheme: str = 'default'
    update_interval: float = STATUS_UPDATE_INTERVAL_SECONDS
    enable_animation: bool = True
    scientific_formatting: bool = True


@dataclass
class StatusDisplayConfig:
    """Configuration data class for status display hierarchy and formatting options."""
    max_display_lines: int = 50
    show_timestamps: bool = True
    show_performance_metrics: bool = True
    auto_scroll: bool = True
    tree_indent_size: int = 2
    color_coding_enabled: bool = True


@dataclass
class PerformanceThreshold:
    """Data class for performance threshold definitions and violation detection."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    unit: str
    violation_action: str = 'log'


def initialize_progress_display(
    force_color_detection: bool = False,
    enable_unicode_support: bool = True,
    terminal_width_override: Optional[int] = None,
    display_config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Initialize the progress display system with terminal capability detection, color support configuration, 
    performance thresholds loading, and display optimization settings for scientific computing workflows.
    
    This function sets up the entire progress display infrastructure including terminal capability detection,
    color scheme configuration, performance threshold loading, and global state initialization for optimal
    scientific computing workflow support.
    
    Args:
        force_color_detection: Force re-detection of terminal color capabilities
        enable_unicode_support: Enable Unicode character support for enhanced progress indicators
        terminal_width_override: Override detected terminal width with specified value
        display_config: Additional display configuration options and customizations
        
    Returns:
        bool: Success status of progress display initialization
    """
    global _terminal_capabilities, _color_support_enabled, _unicode_support_enabled, _terminal_width
    
    try:
        # Initialize logging for progress display system
        logger = get_logger('progress_display.init', 'SYSTEM')
        logger.info("Initializing progress display system")
        
        # Load performance thresholds configuration from config file
        config_path = display_config.get('thresholds_config', 'config/performance_thresholds.json') if display_config else 'config/performance_thresholds.json'
        thresholds = load_performance_thresholds(config_path)
        global _performance_thresholds
        _performance_thresholds = thresholds
        
        # Detect terminal capabilities including color and Unicode support
        if force_color_detection or _terminal_capabilities is None:
            _terminal_capabilities = detect_terminal_capabilities()
            logger.debug(f"Terminal capabilities detected: {_terminal_capabilities}")
        
        # Configure terminal width and responsive formatting settings
        if terminal_width_override:
            _terminal_width = terminal_width_override
        else:
            _terminal_width = _terminal_capabilities.get('width', DEFAULT_TERMINAL_WIDTH)
        
        # Initialize color scheme based on terminal capabilities
        _color_support_enabled = _terminal_capabilities.get('color_support', False)
        if not _color_support_enabled:
            logger.warning("Color support not detected, using monochrome display")
        
        # Setup Unicode character support for progress indicators
        _unicode_support_enabled = enable_unicode_support and _terminal_capabilities.get('unicode_support', False)
        if not _unicode_support_enabled:
            logger.warning("Unicode support disabled or not detected, using ASCII fallback")
        
        # Configure display optimization settings
        if display_config:
            # Apply additional configuration options
            for key, value in display_config.items():
                if key == 'update_interval':
                    globals()['STATUS_UPDATE_INTERVAL_SECONDS'] = value
                elif key == 'animation_interval':
                    globals()['PROGRESS_ANIMATION_INTERVAL_SECONDS'] = value
                elif key == 'precision_digits':
                    globals()['SCIENTIFIC_PRECISION_DIGITS'] = value
        
        # Initialize progress display registry and tracking
        _active_progress_bars.clear()
        _status_displays.clear()
        
        # Setup thread-safe display update mechanisms
        if not _progress_display_lock:
            globals()['_progress_display_lock'] = threading.Lock()
        
        # Validate progress display system configuration
        validation_tests = [
            ('terminal_width', _terminal_width > 0),
            ('color_support', isinstance(_color_support_enabled, bool)),
            ('unicode_support', isinstance(_unicode_support_enabled, bool)),
            ('performance_thresholds', isinstance(_performance_thresholds, dict))
        ]
        
        for test_name, test_result in validation_tests:
            if not test_result:
                logger.error(f"Progress display validation failed: {test_name}")
                return False
        
        # Log progress display initialization completion
        logger.info("Progress display system initialized successfully")
        logger.debug(f"Configuration: width={_terminal_width}, color={_color_support_enabled}, unicode={_unicode_support_enabled}")
        
        return True
        
    except Exception as e:
        # Handle initialization errors with fallback configuration
        print(f"WARNING: Progress display initialization failed: {e}", file=sys.stderr)
        
        # Apply fallback configuration
        _terminal_width = DEFAULT_TERMINAL_WIDTH
        _color_support_enabled = False
        _unicode_support_enabled = False
        _terminal_capabilities = {'width': DEFAULT_TERMINAL_WIDTH, 'color_support': False, 'unicode_support': False}
        
        return False


def load_performance_thresholds(config_path: str) -> Dict[str, Any]:
    """
    Load performance thresholds configuration from JSON file for progress display color coding and status indicators.
    
    This function loads and validates performance threshold configuration for color coding and status indicators
    in progress displays, providing fallback defaults for missing or invalid configuration files.
    
    Args:
        config_path: Path to the performance thresholds JSON configuration file
        
    Returns:
        Dict[str, Any]: Loaded performance thresholds configuration with defaults applied
    """
    logger = get_logger('progress_display.config', 'SYSTEM')
    
    try:
        # Read performance thresholds JSON configuration file
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Performance thresholds loaded from: {config_path}")
        else:
            logger.warning(f"Configuration file not found: {config_path}, using defaults")
            config = {}
        
        # Parse JSON configuration with error handling
        if not isinstance(config, dict):
            logger.error("Invalid configuration format, using defaults")
            config = {}
        
        # Validate configuration structure and required fields
        required_sections = ['processing_time', 'memory_usage', 'cpu_usage', 'batch_completion']
        for section in required_sections:
            if section not in config:
                config[section] = {}
        
        # Extract relevant threshold values for progress display
        thresholds = {
            'processing_time': {
                'warning': config.get('processing_time', {}).get('warning', 7.2),  # seconds per simulation
                'critical': config.get('processing_time', {}).get('critical', 10.0),
                'unit': 'seconds'
            },
            'memory_usage': {
                'warning': config.get('memory_usage', {}).get('warning', 80.0),  # percentage
                'critical': config.get('memory_usage', {}).get('critical', 90.0),
                'unit': 'percent'
            },
            'cpu_usage': {
                'warning': config.get('cpu_usage', {}).get('warning', 85.0),  # percentage
                'critical': config.get('cpu_usage', {}).get('critical', 95.0),
                'unit': 'percent'
            },
            'batch_completion': {
                'warning': config.get('batch_completion', {}).get('warning', 95.0),  # percentage
                'critical': config.get('batch_completion', {}).get('critical', 90.0),
                'unit': 'percent'
            }
        }
        
        # Cache configuration for progress display operations
        logger.debug("Performance thresholds configuration cached")
        
        # Return loaded thresholds configuration
        return thresholds
        
    except Exception as e:
        # Log error and return default configuration
        logger.error(f"Failed to load performance thresholds: {e}")
        
        # Return default performance thresholds
        return {
            'processing_time': {'warning': 7.2, 'critical': 10.0, 'unit': 'seconds'},
            'memory_usage': {'warning': 80.0, 'critical': 90.0, 'unit': 'percent'},
            'cpu_usage': {'warning': 85.0, 'critical': 95.0, 'unit': 'percent'},
            'batch_completion': {'warning': 95.0, 'critical': 90.0, 'unit': 'percent'}
        }


def format_progress_update(
    message: str,
    status_level: str,
    context: Optional[Dict[str, Any]] = None,
    include_timestamp: bool = True,
    apply_colors: bool = True
) -> str:
    """
    Format progress update message with color coding, performance context, and scientific formatting for terminal display.
    
    This function provides comprehensive formatting for progress update messages with color coding based on status level,
    scientific value formatting for numerical context, and consistent visual styling for terminal display.
    
    Args:
        message: Core progress update message content
        status_level: Status level for color coding (SUCCESS, WARNING, ERROR, INFO)
        context: Additional context information for scientific formatting
        include_timestamp: Include timestamp in formatted message
        apply_colors: Apply color coding based on status level and terminal capabilities
        
    Returns:
        str: Formatted progress update message with appropriate styling and context
    """
    logger = get_logger('progress_display.format', 'DISPLAY')
    
    try:
        # Determine color scheme based on status level and terminal capabilities
        color_code = ''
        reset_code = ''
        status_icon = STATUS_ICONS.get('INFO', 'ℹ')
        
        if apply_colors and _color_support_enabled:
            color_mapping = {
                'SUCCESS': TERMINAL_COLORS['GREEN'],
                'WARNING': TERMINAL_COLORS['YELLOW'],
                'ERROR': TERMINAL_COLORS['RED'],
                'INFO': TERMINAL_COLORS['BLUE'],
                'PROGRESS': TERMINAL_COLORS['CYAN']
            }
            color_code = color_mapping.get(status_level, TERMINAL_COLORS['WHITE'])
            reset_code = TERMINAL_COLORS['RESET']
            
            # Get appropriate status icon for level
            icon_mapping = {
                'SUCCESS': STATUS_ICONS['SUCCESS'],
                'WARNING': STATUS_ICONS['WARNING'],
                'ERROR': STATUS_ICONS['ERROR'],
                'INFO': STATUS_ICONS['INFO'],
                'PROGRESS': STATUS_ICONS['PROGRESS']
            }
            status_icon = icon_mapping.get(status_level, STATUS_ICONS['INFO'])
        
        # Apply scientific value formatting to numerical context data
        formatted_context = []
        if context:
            for key, value in context.items():
                if isinstance(value, (int, float)):
                    # Format numerical values with scientific precision
                    formatted_value = format_scientific_value(
                        value=float(value),
                        precision=SCIENTIFIC_PRECISION_DIGITS
                    )
                    formatted_context.append(f"{key}={formatted_value}")
                else:
                    formatted_context.append(f"{key}={value}")
        
        # Include timestamp if requested with appropriate precision
        timestamp_str = ''
        if include_timestamp:
            timestamp = datetime.datetime.now()
            timestamp_str = f"[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] "
        
        # Add status icons and visual indicators based on status level
        icon_str = f"{status_icon} " if _unicode_support_enabled else f"[{status_level}] "
        
        # Apply color coding for status level and context elements
        formatted_message = f"{timestamp_str}{color_code}{icon_str}{message}{reset_code}"
        
        # Add context information if provided
        if formatted_context:
            context_str = " | ".join(formatted_context)
            if apply_colors and _color_support_enabled:
                context_str = f"{TERMINAL_COLORS['DIM']}{context_str}{TERMINAL_COLORS['RESET']}"
            formatted_message += f" | {context_str}"
        
        # Format message with appropriate terminal width wrapping
        if len(formatted_message) > _terminal_width:
            # Simple line wrapping for long messages
            wrapped_lines = []
            current_line = ''
            words = formatted_message.split(' ')
            
            for word in words:
                # Calculate line length without ANSI escape sequences
                clean_line = re.sub(r'\033\[[0-9;]*m', '', current_line + word)
                if len(clean_line) < _terminal_width - 5:  # Leave margin
                    current_line += word + ' '
                else:
                    if current_line:
                        wrapped_lines.append(current_line.rstrip())
                    current_line = word + ' '
            
            if current_line:
                wrapped_lines.append(current_line.rstrip())
            
            formatted_message = '\n'.join(wrapped_lines)
        
        # Return formatted progress update message
        return formatted_message
        
    except Exception as e:
        logger.error(f"Error formatting progress update: {e}")
        # Return basic formatted message as fallback
        return f"[{status_level}] {message}"


def create_progress_bar(
    bar_id: str,
    total_items: int,
    description: str = '',
    bar_width: Optional[int] = None,
    show_percentage: bool = True,
    show_eta: bool = True,
    show_rate: bool = True,
    color_scheme: str = 'default'
) -> 'ProgressBar':
    """
    Create ASCII progress bar with customizable width, style, color coding, and performance metrics display 
    for batch simulation operations and long-running tasks.
    
    This function creates and configures an advanced ASCII progress bar with real-time updates, performance
    metrics integration, and scientific computing optimizations for batch processing operations.
    
    Args:
        bar_id: Unique identifier for the progress bar instance
        total_items: Total number of items to process
        description: Descriptive text for the progress operation
        bar_width: Width of the progress bar (auto-calculated if None)
        show_percentage: Display completion percentage
        show_eta: Display estimated time to completion
        show_rate: Display processing rate
        color_scheme: Color scheme for progress bar styling
        
    Returns:
        ProgressBar: Configured progress bar instance for real-time updates
    """
    logger = get_logger('progress_display.create', 'DISPLAY')
    
    try:
        # Validate progress bar parameters and configuration
        if total_items <= 0:
            raise ValueError(f"Total items must be positive: {total_items}")
        
        if bar_id in _active_progress_bars:
            logger.warning(f"Progress bar already exists: {bar_id}")
            return _active_progress_bars[bar_id]
        
        # Determine optimal bar width based on terminal size
        if bar_width is None:
            # Calculate optimal width based on terminal size and display elements
            reserved_space = 50  # Space for percentage, ETA, rate, etc.
            available_width = _terminal_width - reserved_space
            bar_width = max(MIN_PROGRESS_BAR_WIDTH, min(available_width, MAX_PROGRESS_BAR_WIDTH))
        
        bar_width = max(MIN_PROGRESS_BAR_WIDTH, min(bar_width, MAX_PROGRESS_BAR_WIDTH))
        
        # Initialize progress bar state and tracking variables
        display_config = ProgressBarConfig(
            bar_width=bar_width,
            show_percentage=show_percentage,
            show_eta=show_eta,
            show_rate=show_rate,
            color_scheme=color_scheme,
            scientific_formatting=True
        )
        
        # Configure color scheme and visual indicators
        color_config = {
            'default': {
                'filled': TERMINAL_COLORS['GREEN'] if _color_support_enabled else '',
                'empty': TERMINAL_COLORS['DIM'] if _color_support_enabled else '',
                'text': TERMINAL_COLORS['WHITE'] if _color_support_enabled else '',
                'reset': TERMINAL_COLORS['RESET'] if _color_support_enabled else ''
            }
        }
        
        # Setup performance metrics tracking if enabled
        performance_monitor = PerformanceMonitor() if show_rate else None
        
        # Register progress bar in active displays registry
        with _progress_display_lock:
            progress_bar = ProgressBar(
                bar_id=bar_id,
                total_items=total_items,
                description=description,
                bar_width=bar_width,
                display_config=display_config.__dict__
            )
            
            _active_progress_bars[bar_id] = progress_bar
        
        # Initialize thread-safe update mechanisms
        logger.debug(f"Created progress bar: {bar_id} ({total_items} items, width={bar_width})")
        
        # Return configured progress bar instance
        return progress_bar
        
    except Exception as e:
        logger.error(f"Failed to create progress bar {bar_id}: {e}")
        raise


def update_progress_bar(
    bar_id: str,
    current_items: int,
    status_message: str = '',
    performance_metrics: Optional[Dict[str, float]] = None,
    force_redraw: bool = False
) -> None:
    """
    Update progress bar with current completion status, performance metrics, estimated time remaining, 
    and visual indicators with thread-safe operations.
    
    This function provides thread-safe progress bar updates with performance metrics integration,
    real-time visual updates, and scientific value formatting for batch processing operations.
    
    Args:
        bar_id: Unique identifier for the progress bar to update
        current_items: Current number of completed items
        status_message: Optional status message to display with progress
        performance_metrics: Performance metrics to display with progress
        force_redraw: Force complete redraw of progress bar
    """
    logger = get_logger('progress_display.update', 'DISPLAY')
    
    try:
        # Acquire progress display lock for thread safety
        with _progress_display_lock:
            # Validate bar_id exists in active displays registry
            if bar_id not in _active_progress_bars:
                logger.warning(f"Progress bar not found: {bar_id}")
                return
            
            progress_bar = _active_progress_bars[bar_id]
            
            # Calculate progress percentage and remaining items
            progress_bar.update(
                current_items=current_items,
                status_message=status_message,
                performance_metrics=performance_metrics
            )
            
            # Redraw progress bar to terminal if needed
            if force_redraw or progress_bar.should_update():
                rendered_bar = progress_bar.render(force_redraw=force_redraw)
                
                # Output rendered progress bar to terminal
                print(f"\r{rendered_bar}", end='', flush=True)
        
        logger.debug(f"Updated progress bar {bar_id}: {current_items} items")
        
    except Exception as e:
        logger.error(f"Error updating progress bar {bar_id}: {e}")


def display_simulation_status(
    simulation_id: str,
    algorithm_name: str,
    current_stage: str,
    progress_percentage: float,
    performance_data: Optional[Dict[str, Any]] = None,
    include_resource_metrics: bool = False
) -> None:
    """
    Display comprehensive simulation status including algorithm execution, resource utilization, performance metrics, 
    and progress indicators for scientific computing monitoring.
    
    This function provides comprehensive simulation status display with hierarchical structure, performance metrics
    integration, and scientific value formatting optimized for research computing workflows.
    
    Args:
        simulation_id: Unique identifier for the simulation
        algorithm_name: Name of the navigation algorithm being executed
        current_stage: Current processing stage of the simulation
        progress_percentage: Current progress percentage (0-100)
        performance_data: Performance metrics and timing information
        include_resource_metrics: Include system resource utilization metrics
    """
    logger = get_logger('progress_display.simulation', 'SIMULATION')
    
    try:
        # Format simulation identification and algorithm context
        header = format_progress_update(
            message=f"Simulation {simulation_id[:8]}... | {algorithm_name}",
            status_level='INFO',
            include_timestamp=True,
            apply_colors=_color_support_enabled
        )
        
        # Create status display with hierarchical structure
        status_lines = [header]
        
        # Include current processing stage with progress indicator
        stage_icon = STATUS_ICONS['RUNNING'] if _unicode_support_enabled else '[RUN]'
        stage_color = TERMINAL_COLORS['BLUE'] if _color_support_enabled else ''
        reset_color = TERMINAL_COLORS['RESET'] if _color_support_enabled else ''
        
        stage_line = f"  {stage_color}{stage_icon} Stage: {current_stage}{reset_color}"
        status_lines.append(stage_line)
        
        # Display progress percentage with visual bar
        progress_width = min(30, _terminal_width // 3)
        filled_chars = int(progress_width * progress_percentage / 100)
        empty_chars = progress_width - filled_chars
        
        filled_bar = PROGRESS_BAR_CHARS['FILLED'] * filled_chars
        empty_bar = PROGRESS_BAR_CHARS['EMPTY'] * empty_chars
        
        progress_color = TERMINAL_COLORS['GREEN'] if _color_support_enabled else ''
        progress_line = f"  Progress: [{progress_color}{filled_bar}{empty_bar}{reset_color}] {progress_percentage:.1f}%"
        status_lines.append(progress_line)
        
        # Include performance metrics with threshold indicators
        if performance_data:
            perf_lines = []
            for metric_name, metric_value in performance_data.items():
                if isinstance(metric_value, (int, float)):
                    # Format scientific values with appropriate precision
                    formatted_value = format_scientific_value(
                        value=float(metric_value),
                        precision=SCIENTIFIC_PRECISION_DIGITS
                    )
                    
                    # Apply color coding based on performance thresholds
                    metric_color = _get_metric_color(metric_name, metric_value)
                    metric_line = f"    {metric_color}{metric_name}: {formatted_value}{reset_color}"
                    perf_lines.append(metric_line)
            
            if perf_lines:
                status_lines.append("  Performance:")
                status_lines.extend(perf_lines)
        
        # Add resource utilization metrics if requested
        if include_resource_metrics:
            try:
                system_metrics = collect_system_metrics()
                resource_lines = ["  Resources:"]
                
                for resource_name, resource_value in system_metrics.items():
                    formatted_resource = format_scientific_value(
                        value=resource_value,
                        unit='%' if 'usage' in resource_name else '',
                        precision=1
                    )
                    resource_color = _get_metric_color(resource_name, resource_value)
                    resource_line = f"    {resource_color}{resource_name}: {formatted_resource}{reset_color}"
                    resource_lines.append(resource_line)
                
                status_lines.extend(resource_lines)
                
            except Exception as e:
                logger.debug(f"Could not collect resource metrics: {e}")
        
        # Output formatted status to terminal
        status_output = '\n'.join(status_lines)
        print(status_output)
        
        logger.debug(f"Displayed simulation status for {simulation_id}")
        
    except Exception as e:
        logger.error(f"Error displaying simulation status: {e}")


def display_performance_metrics(
    metrics: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None,
    display_format: str = 'table',
    show_trends: bool = False,
    highlight_violations: bool = True
) -> None:
    """
    Display real-time performance metrics with scientific formatting, threshold indicators, color coding, 
    and trend analysis for batch processing optimization.
    
    This function provides comprehensive performance metrics display with threshold comparison, color coding,
    trend analysis, and scientific value formatting for performance monitoring and optimization.
    
    Args:
        metrics: Dictionary of performance metrics to display
        thresholds: Performance thresholds for comparison and color coding
        display_format: Format for metrics display (table, list, compact)
        show_trends: Include trend indicators in the display
        highlight_violations: Highlight threshold violations with color coding
    """
    logger = get_logger('progress_display.metrics', 'PERFORMANCE')
    
    try:
        # Format performance metrics with scientific precision
        formatted_metrics = {}
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                formatted_metrics[metric_name] = format_scientific_value(
                    value=float(metric_value),
                    precision=SCIENTIFIC_PRECISION_DIGITS
                )
            else:
                formatted_metrics[metric_name] = str(metric_value)
        
        # Compare metrics against configured thresholds
        threshold_config = thresholds or _performance_thresholds
        violations = []
        
        if display_format == 'table':
            # Create aligned table format for multiple metrics
            table_data = []
            headers = ['Metric', 'Value', 'Status']
            
            if show_trends:
                headers.append('Trend')
            
            for metric_name, formatted_value in formatted_metrics.items():
                row = [metric_name, formatted_value]
                
                # Determine status based on thresholds
                status = 'OK'
                status_color = TERMINAL_COLORS['GREEN'] if _color_support_enabled else ''
                
                if metric_name in threshold_config:
                    threshold_data = threshold_config[metric_name]
                    raw_value = metrics[metric_name]
                    
                    if raw_value >= threshold_data.get('critical', float('inf')):
                        status = 'CRITICAL'
                        status_color = TERMINAL_COLORS['RED'] if _color_support_enabled else ''
                        if highlight_violations:
                            violations.append((metric_name, raw_value, 'critical'))
                    elif raw_value >= threshold_data.get('warning', float('inf')):
                        status = 'WARNING'
                        status_color = TERMINAL_COLORS['YELLOW'] if _color_support_enabled else ''
                        if highlight_violations:
                            violations.append((metric_name, raw_value, 'warning'))
                
                reset_color = TERMINAL_COLORS['RESET'] if _color_support_enabled else ''
                colored_status = f"{status_color}{status}{reset_color}"
                row.append(colored_status)
                
                # Include trend indicators if show_trends is enabled
                if show_trends:
                    # Placeholder for trend analysis - would integrate with historical data
                    trend_indicator = '→'  # Stable trend
                    row.append(trend_indicator)
                
                table_data.append(row)
            
            # Format and display table
            table_output = create_status_table(
                table_data=[dict(zip(headers, row)) for row in table_data],
                column_headers=headers,
                column_formats={},
                table_width=_terminal_width,
                include_borders=True,
                color_scheme='default'
            )
            
            print("Performance Metrics:")
            print(table_output)
            
        else:
            # Simple list format for compact display
            print("Performance Metrics:")
            for metric_name, formatted_value in formatted_metrics.items():
                metric_color = _get_metric_color(metric_name, metrics[metric_name])
                reset_color = TERMINAL_COLORS['RESET'] if _color_support_enabled else ''
                print(f"  {metric_color}{metric_name}: {formatted_value}{reset_color}")
        
        # Highlight threshold violations if enabled
        if highlight_violations and violations:
            print("\nThreshold Violations:")
            for metric_name, value, severity in violations:
                severity_color = TERMINAL_COLORS['RED'] if severity == 'critical' else TERMINAL_COLORS['YELLOW']
                reset_color = TERMINAL_COLORS['RESET'] if _color_support_enabled else ''
                formatted_value = format_scientific_value(value, precision=SCIENTIFIC_PRECISION_DIGITS)
                print(f"  {severity_color}[{severity.upper()}] {metric_name}: {formatted_value}{reset_color}")
        
        logger.debug(f"Displayed performance metrics: {len(metrics)} metrics, {len(violations)} violations")
        
    except Exception as e:
        logger.error(f"Error displaying performance metrics: {e}")


def create_status_table(
    table_data: List[Dict[str, Any]],
    column_headers: List[str],
    column_formats: Dict[str, str],
    table_width: int,
    include_borders: bool = True,
    color_scheme: str = 'default'
) -> str:
    """
    Create formatted status table with aligned columns, headers, scientific value formatting, and color coding 
    for structured data presentation.
    
    This function creates well-formatted tables with proper column alignment, scientific value formatting,
    and color coding for structured data presentation in scientific computing environments.
    
    Args:
        table_data: List of dictionaries containing table row data
        column_headers: List of column header names
        column_formats: Dictionary of column-specific formatting options
        table_width: Maximum width for the table
        include_borders: Include table borders and separators
        color_scheme: Color scheme for table styling
        
    Returns:
        str: Formatted table string ready for terminal display
    """
    logger = get_logger('progress_display.table', 'DISPLAY')
    
    try:
        if not table_data or not column_headers:
            return "No data to display"
        
        # Calculate optimal column widths based on content
        column_widths = {}
        for header in column_headers:
            # Start with header width
            column_widths[header] = len(header)
            
            # Check data widths (without ANSI escape sequences)
            for row in table_data:
                if header in row:
                    # Remove ANSI escape sequences for width calculation
                    clean_value = re.sub(r'\033\[[0-9;]*m', '', str(row[header]))
                    column_widths[header] = max(column_widths[header], len(clean_value))
        
        # Adjust column widths to fit within table_width
        total_content_width = sum(column_widths.values())
        total_padding = len(column_headers) * TABLE_COLUMN_PADDING * 2
        border_width = 3 if include_borders else 0  # Account for borders
        
        available_width = table_width - total_padding - border_width
        
        if total_content_width > available_width:
            # Proportionally reduce column widths
            scale_factor = available_width / total_content_width
            for header in column_headers:
                column_widths[header] = max(8, int(column_widths[header] * scale_factor))
        
        # Format table headers with appropriate alignment
        header_color = TERMINAL_COLORS['BOLD'] if _color_support_enabled else ''
        reset_color = TERMINAL_COLORS['RESET'] if _color_support_enabled else ''
        
        formatted_headers = []
        for header in column_headers:
            padded_header = header.ljust(column_widths[header])
            formatted_headers.append(f"{header_color}{padded_header}{reset_color}")
        
        # Create table lines
        table_lines = []
        
        # Add top border if enabled
        if include_borders:
            border_line = '+' + '+'.join('-' * (column_widths[header] + TABLE_COLUMN_PADDING * 2) for header in column_headers) + '+'
            table_lines.append(border_line)
        
        # Add header row
        if include_borders:
            header_line = '| ' + ' | '.join(formatted_headers) + ' |'
        else:
            header_line = '  '.join(formatted_headers)
        table_lines.append(header_line)
        
        # Add header separator
        if include_borders:
            separator_line = '+' + '+'.join('-' * (column_widths[header] + TABLE_COLUMN_PADDING * 2) for header in column_headers) + '+'
            table_lines.append(separator_line)
        else:
            separator_chars = ['-' * column_widths[header] for header in column_headers]
            separator_line = '  '.join(separator_chars)
            table_lines.append(separator_line)
        
        # Apply column-specific formatting to data values
        for row in table_data:
            formatted_row = []
            for header in column_headers:
                cell_value = row.get(header, '')
                
                # Apply column-specific formatting
                if header in column_formats:
                    format_spec = column_formats[header]
                    if format_spec == 'scientific' and isinstance(cell_value, (int, float)):
                        cell_value = format_scientific_value(
                            value=float(cell_value),
                            precision=SCIENTIFIC_PRECISION_DIGITS
                        )
                
                # Pad cell value to column width
                cell_str = str(cell_value)
                # Calculate display width without ANSI codes
                display_width = len(re.sub(r'\033\[[0-9;]*m', '', cell_str))
                padding_needed = column_widths[header] - display_width
                padded_cell = cell_str + ' ' * max(0, padding_needed)
                
                formatted_row.append(padded_cell)
            
            # Add data row
            if include_borders:
                row_line = '| ' + ' | '.join(formatted_row) + ' |'
            else:
                row_line = '  '.join(formatted_row)
            table_lines.append(row_line)
        
        # Add bottom border if enabled
        if include_borders:
            table_lines.append(border_line)
        
        # Return formatted table string
        return '\n'.join(table_lines)
        
    except Exception as e:
        logger.error(f"Error creating status table: {e}")
        return f"Error creating table: {e}"


def display_batch_summary(
    batch_id: str,
    total_simulations: int,
    completed_simulations: int,
    failed_simulations: int,
    elapsed_time_hours: float,
    performance_summary: Optional[Dict[str, Any]] = None,
    include_recommendations: bool = False
) -> None:
    """
    Display comprehensive batch processing summary including completion statistics, performance analysis, 
    error rates, and optimization recommendations.
    
    This function provides comprehensive batch processing summary with statistical analysis, performance
    metrics, error rate analysis, and optimization recommendations for scientific computing workflows.
    
    Args:
        batch_id: Unique identifier for the batch operation
        total_simulations: Total number of simulations in the batch
        completed_simulations: Number of successfully completed simulations
        failed_simulations: Number of failed simulations
        elapsed_time_hours: Total elapsed time for batch processing in hours
        performance_summary: Summary of performance metrics for the batch
        include_recommendations: Include optimization recommendations in the summary
    """
    logger = get_logger('progress_display.batch', 'BATCH')
    
    try:
        # Calculate batch completion statistics and rates
        success_rate = (completed_simulations / total_simulations * 100) if total_simulations > 0 else 0
        failure_rate = (failed_simulations / total_simulations * 100) if total_simulations > 0 else 0
        pending_simulations = total_simulations - completed_simulations - failed_simulations
        
        # Calculate processing rates and efficiency
        simulations_per_hour = completed_simulations / elapsed_time_hours if elapsed_time_hours > 0 else 0
        average_time_per_simulation = (elapsed_time_hours * 3600) / completed_simulations if completed_simulations > 0 else 0
        
        # Format batch identification and timing information
        header = format_progress_update(
            message=f"Batch Summary: {batch_id}",
            status_level='INFO',
            include_timestamp=True,
            apply_colors=_color_support_enabled
        )
        
        print(header)
        print("=" * min(60, _terminal_width))
        
        # Create summary table with completion metrics
        summary_data = [
            {'Metric': 'Total Simulations', 'Value': f"{total_simulations:,}"},
            {'Metric': 'Completed', 'Value': f"{completed_simulations:,} ({success_rate:.1f}%)"},
            {'Metric': 'Failed', 'Value': f"{failed_simulations:,} ({failure_rate:.1f}%)"},
            {'Metric': 'Pending', 'Value': f"{pending_simulations:,}"},
            {'Metric': 'Elapsed Time', 'Value': format_duration(elapsed_time_hours * 3600)},
            {'Metric': 'Processing Rate', 'Value': format_scientific_value(simulations_per_hour, unit='sim/hour', precision=2)},
            {'Metric': 'Avg Time/Simulation', 'Value': format_scientific_value(average_time_per_simulation, unit='seconds', precision=2)}
        ]
        
        # Add color coding based on success rate and performance
        for row in summary_data:
            metric_name = row['Metric']
            if metric_name == 'Completed' and success_rate >= 95:
                row['Value'] = f"{TERMINAL_COLORS['GREEN']}{row['Value']}{TERMINAL_COLORS['RESET']}" if _color_support_enabled else row['Value']
            elif metric_name == 'Failed' and failure_rate > 5:
                row['Value'] = f"{TERMINAL_COLORS['RED']}{row['Value']}{TERMINAL_COLORS['RESET']}" if _color_support_enabled else row['Value']
            elif metric_name == 'Avg Time/Simulation':
                # Color code based on performance threshold
                if average_time_per_simulation > _performance_thresholds.get('processing_time', {}).get('critical', 10.0):
                    row['Value'] = f"{TERMINAL_COLORS['RED']}{row['Value']}{TERMINAL_COLORS['RESET']}" if _color_support_enabled else row['Value']
                elif average_time_per_simulation > _performance_thresholds.get('processing_time', {}).get('warning', 7.2):
                    row['Value'] = f"{TERMINAL_COLORS['YELLOW']}{row['Value']}{TERMINAL_COLORS['RESET']}" if _color_support_enabled else row['Value']
                else:
                    row['Value'] = f"{TERMINAL_COLORS['GREEN']}{row['Value']}{TERMINAL_COLORS['RESET']}" if _color_support_enabled else row['Value']
        
        # Display summary table
        summary_table = create_status_table(
            table_data=summary_data,
            column_headers=['Metric', 'Value'],
            column_formats={'Value': 'scientific'},
            table_width=_terminal_width,
            include_borders=True,
            color_scheme='default'
        )
        
        print(summary_table)
        
        # Include performance analysis and efficiency metrics
        if performance_summary:
            print("\nPerformance Analysis:")
            print("-" * min(40, _terminal_width))
            
            display_performance_metrics(
                metrics=performance_summary,
                thresholds=_performance_thresholds,
                display_format='table',
                show_trends=False,
                highlight_violations=True
            )
        
        # Add optimization recommendations if requested
        if include_recommendations:
            print("\nOptimization Recommendations:")
            print("-" * min(40, _terminal_width))
            
            recommendations = []
            
            # Analyze failure rate
            if failure_rate > 10:
                recommendations.append(f"{TERMINAL_COLORS['YELLOW']}• High failure rate ({failure_rate:.1f}%) detected - review error logs{TERMINAL_COLORS['RESET']}" if _color_support_enabled else f"• High failure rate ({failure_rate:.1f}%) detected - review error logs")
            
            # Analyze processing time
            if average_time_per_simulation > _performance_thresholds.get('processing_time', {}).get('warning', 7.2):
                recommendations.append(f"{TERMINAL_COLORS['YELLOW']}• Processing time exceeds target - consider algorithm optimization{TERMINAL_COLORS['RESET']}" if _color_support_enabled else "• Processing time exceeds target - consider algorithm optimization")
            
            # Analyze completion rate
            if success_rate < 95:
                recommendations.append(f"{TERMINAL_COLORS['YELLOW']}• Completion rate below target (95%) - investigate failure causes{TERMINAL_COLORS['RESET']}" if _color_support_enabled else "• Completion rate below target (95%) - investigate failure causes")
            
            # Resource utilization recommendations
            if performance_summary and 'cpu_usage' in performance_summary:
                cpu_usage = performance_summary['cpu_usage']
                if cpu_usage < 70:
                    recommendations.append(f"{TERMINAL_COLORS['BLUE']}• CPU utilization low ({cpu_usage:.1f}%) - consider increasing parallelism{TERMINAL_COLORS['RESET']}" if _color_support_enabled else f"• CPU utilization low ({cpu_usage:.1f}%) - consider increasing parallelism")
            
            if recommendations:
                for rec in recommendations:
                    print(rec)
            else:
                print(f"{TERMINAL_COLORS['GREEN']}• No optimization recommendations - batch performance is within targets{TERMINAL_COLORS['RESET']}" if _color_support_enabled else "• No optimization recommendations - batch performance is within targets")
        
        logger.info(f"Displayed batch summary for {batch_id}: {completed_simulations}/{total_simulations} completed")
        
    except Exception as e:
        logger.error(f"Error displaying batch summary: {e}")


def clear_progress_display(
    display_id: Optional[str] = None,
    lines_to_clear: int = 1,
    reset_cursor: bool = False
) -> None:
    """
    Clear progress display elements from terminal with proper cursor management and line clearing 
    for clean status updates.
    
    This function provides terminal clearing capabilities with cursor management and line clearing
    for clean progress display updates and terminal maintenance.
    
    Args:
        display_id: Specific display ID to clear (None for general clearing)
        lines_to_clear: Number of lines to clear from current cursor position
        reset_cursor: Reset cursor to home position after clearing
    """
    logger = get_logger('progress_display.clear', 'DISPLAY')
    
    try:
        # Acquire progress display lock for exclusive access
        with _progress_display_lock:
            # Move cursor and clear lines using ANSI escape sequences
            if lines_to_clear > 0:
                # Move cursor up and clear lines
                for _ in range(lines_to_clear):
                    sys.stdout.write('\033[1A')  # Move cursor up one line
                    sys.stdout.write('\033[2K')  # Clear entire line
            
            # Reset cursor position if requested
            if reset_cursor:
                sys.stdout.write('\033[H')  # Move cursor to home position
            
            # Remove display from active registry if specified
            if display_id:
                if display_id in _active_progress_bars:
                    del _active_progress_bars[display_id]
                    logger.debug(f"Removed progress bar: {display_id}")
                
                if display_id in _status_displays:
                    del _status_displays[display_id]
                    logger.debug(f"Removed status display: {display_id}")
            
            # Flush terminal output buffer
            sys.stdout.flush()
        
        logger.debug(f"Cleared progress display: {lines_to_clear} lines")
        
    except Exception as e:
        logger.error(f"Error clearing progress display: {e}")


def animate_progress_spinner(
    spinner_id: str,
    message: str,
    spinner_chars: Optional[List[str]] = None,
    animation_interval: float = PROGRESS_ANIMATION_INTERVAL_SECONDS,
    color: str = 'default'
) -> None:
    """
    Animate progress spinner for indeterminate progress operations with customizable characters, colors, 
    and update intervals.
    
    This function provides animated spinner display for indeterminate progress operations with customizable
    animation characters, colors, and timing for enhanced user experience during long-running operations.
    
    Args:
        spinner_id: Unique identifier for the spinner instance
        message: Message to display alongside the spinner
        spinner_chars: Custom spinner animation characters
        animation_interval: Time interval between animation frames
        color: Color scheme for spinner display
    """
    logger = get_logger('progress_display.spinner', 'DISPLAY')
    
    try:
        # Initialize spinner state and animation tracking
        if spinner_chars is None:
            spinner_chars = PROGRESS_BAR_CHARS['SPINNER'] if _unicode_support_enabled else ['|', '/', '-', '\\']
        
        spinner_color = TERMINAL_COLORS.get(color.upper(), '') if _color_support_enabled else ''
        reset_color = TERMINAL_COLORS['RESET'] if _color_support_enabled else ''
        
        # Setup animation timer with specified interval
        frame_index = 0
        
        # Display initial spinner frame with message
        def update_spinner():
            nonlocal frame_index
            spinner_char = spinner_chars[frame_index % len(spinner_chars)]
            display_text = f"\r{spinner_color}{spinner_char}{reset_color} {message}"
            
            # Apply color coding to spinner characters
            sys.stdout.write(display_text)
            sys.stdout.flush()
            
            frame_index += 1
        
        # Start animation loop with character rotation
        # Note: This is a simplified implementation
        # In a full implementation, this would use threading.Timer for continuous animation
        update_spinner()
        
        logger.debug(f"Started spinner animation: {spinner_id}")
        
    except Exception as e:
        logger.error(f"Error animating spinner {spinner_id}: {e}")


def format_duration(
    duration_seconds: float,
    format_style: str = 'compact',
    include_milliseconds: bool = False
) -> str:
    """
    Format time duration with appropriate units, precision, and human-readable format for progress display 
    and ETA calculations.
    
    This function provides comprehensive duration formatting with automatic unit selection, precision control,
    and human-readable output for progress displays and time estimation features.
    
    Args:
        duration_seconds: Duration in seconds to format
        format_style: Format style (compact, verbose, scientific)
        include_milliseconds: Include milliseconds in the formatted output
        
    Returns:
        str: Formatted duration string with appropriate units and precision
    """
    try:
        # Determine appropriate time units based on duration magnitude
        if duration_seconds < 0:
            return "0s"
        
        # Calculate hours, minutes, seconds, and milliseconds
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        milliseconds = int((duration_seconds - int(duration_seconds)) * 1000)
        
        # Apply format style (compact, verbose, scientific)
        if format_style == 'verbose':
            parts = []
            if hours > 0:
                parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            if seconds > 0 or (hours == 0 and minutes == 0):
                if include_milliseconds and milliseconds > 0:
                    parts.append(f"{seconds}.{milliseconds:03d} second{'s' if seconds != 1 else ''}")
                else:
                    parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
            
            return ', '.join(parts)
            
        elif format_style == 'scientific':
            if duration_seconds >= 3600:
                return format_scientific_value(duration_seconds / 3600, unit='hours', precision=3)
            elif duration_seconds >= 60:
                return format_scientific_value(duration_seconds / 60, unit='minutes', precision=3)
            else:
                return format_scientific_value(duration_seconds, unit='seconds', precision=3)
        
        else:  # compact format
            if hours > 0:
                if include_milliseconds:
                    return f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
                else:
                    return f"{hours}:{minutes:02d}:{seconds:02d}"
            elif minutes > 0:
                if include_milliseconds:
                    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"
                else:
                    return f"{minutes}:{seconds:02d}"
            else:
                if include_milliseconds and milliseconds > 0:
                    return f"{seconds}.{milliseconds:03d}s"
                else:
                    return f"{seconds}s"
    
    except Exception:
        return "0s"


def cleanup_progress_displays(
    clear_terminal: bool = False,
    preserve_final_status: bool = True
) -> Dict[str, Any]:
    """
    Cleanup all active progress displays, clear terminal output, and reset display system for shutdown or restart.
    
    This function provides comprehensive cleanup of the progress display system with optional terminal clearing
    and final status preservation for system shutdown or restart operations.
    
    Args:
        clear_terminal: Clear terminal screen after cleanup
        preserve_final_status: Preserve final status information before cleanup
        
    Returns:
        Dict[str, Any]: Cleanup summary with final display statistics
    """
    logger = get_logger('progress_display.cleanup', 'SYSTEM')
    
    try:
        cleanup_summary = {
            'cleanup_timestamp': datetime.datetime.now().isoformat(),
            'active_progress_bars': len(_active_progress_bars),
            'active_status_displays': len(_status_displays),
            'final_status_preserved': preserve_final_status,
            'terminal_cleared': clear_terminal
        }
        
        # Stop all active progress animations and updates
        with _progress_display_lock:
            # Preserve final status information if requested
            final_status = {}
            if preserve_final_status:
                for bar_id, progress_bar in _active_progress_bars.items():
                    final_status[bar_id] = {
                        'type': 'progress_bar',
                        'current_items': progress_bar.current_items,
                        'total_items': progress_bar.total_items,
                        'progress_percentage': progress_bar.progress_percentage,
                        'is_complete': progress_bar.is_complete
                    }
                
                for display_id, status_display in _status_displays.items():
                    final_status[display_id] = {
                        'type': 'status_display',
                        'component_count': len(status_display.component_status),
                        'last_update': status_display.last_update_time.isoformat() if status_display.last_update_time else None
                    }
                
                cleanup_summary['final_status'] = final_status
            
            # Clear all progress bars and status displays
            _active_progress_bars.clear()
            _status_displays.clear()
        
        # Clear terminal output if specified
        if clear_terminal:
            # Clear entire screen
            sys.stdout.write('\033[2J')  # Clear screen
            sys.stdout.write('\033[H')   # Move cursor to home
            sys.stdout.flush()
        
        # Reset terminal cursor and formatting
        sys.stdout.write(TERMINAL_COLORS['RESET'])
        sys.stdout.flush()
        
        # Log progress display cleanup completion
        logger.info(f"Progress display cleanup completed: {cleanup_summary['active_progress_bars']} bars, {cleanup_summary['active_status_displays']} displays")
        
        # Return cleanup summary
        return cleanup_summary
        
    except Exception as e:
        logger.error(f"Error during progress display cleanup: {e}")
        return {
            'cleanup_timestamp': datetime.datetime.now().isoformat(),
            'error': str(e),
            'partial_cleanup': True
        }


class ProgressBar:
    """
    Advanced ASCII progress bar class providing real-time progress visualization with customizable styling, 
    performance metrics integration, ETA calculation, and thread-safe updates for batch simulation operations.
    
    This class implements a comprehensive progress bar with real-time updates, performance metrics integration,
    scientific value formatting, and thread-safe operations optimized for scientific computing workflows.
    """
    
    def __init__(
        self,
        bar_id: str,
        total_items: int,
        description: str = '',
        bar_width: int = DEFAULT_PROGRESS_BAR_WIDTH,
        display_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize progress bar with configuration, styling, and performance tracking setup.
        
        Args:
            bar_id: Unique identifier for the progress bar
            total_items: Total number of items to process
            description: Descriptive text for the progress operation
            bar_width: Width of the ASCII progress bar
            display_config: Configuration dictionary for display options
        """
        # Set bar identification and total items count
        self.bar_id = bar_id
        self.total_items = total_items
        self.current_items = 0
        self.description = description
        self.bar_width = bar_width
        
        # Initialize progress tracking variables
        self.display_config = display_config or {}
        self.start_time = datetime.datetime.now()
        self.last_update_time = self.start_time
        self.progress_percentage = 0.0
        self.current_status = ''
        self.performance_metrics = {}
        
        # Configure display settings and styling
        self.show_percentage = self.display_config.get('show_percentage', True)
        self.show_eta = self.display_config.get('show_eta', True)
        self.show_rate = self.display_config.get('show_rate', True)
        self.color_scheme = self.display_config.get('color_scheme', 'default')
        
        # Setup performance metrics tracking
        self.update_lock = threading.Lock()
        self.rate_history = collections.deque(maxlen=10)  # Keep last 10 rate measurements
        self.is_complete = False
        
        # Initialize logger for progress bar operations
        self.logger = get_logger(f'progress_bar.{bar_id}', 'PROGRESS')
        
        self.logger.debug(f"Initialized progress bar: {bar_id} ({total_items} items)")
    
    def update(
        self,
        current_items: int,
        status_message: str = '',
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update progress bar with current completion status, performance metrics, and visual refresh.
        
        This method provides thread-safe progress updates with performance metrics integration and
        automatic rate calculation for comprehensive progress tracking.
        
        Args:
            current_items: Current number of completed items
            status_message: Optional status message to display
            performance_metrics: Performance metrics to track with progress
        """
        try:
            # Acquire update lock for thread safety
            with self.update_lock:
                # Update current items and calculate progress percentage
                self.current_items = min(current_items, self.total_items)
                self.progress_percentage = (self.current_items / self.total_items * 100) if self.total_items > 0 else 0
                
                # Update performance metrics if provided
                if performance_metrics:
                    self.performance_metrics.update(performance_metrics)
                
                # Calculate processing rate and ETA
                current_time = datetime.datetime.now()
                elapsed_time = (current_time - self.start_time).total_seconds()
                
                if elapsed_time > 0 and self.current_items > 0:
                    current_rate = self.current_items / elapsed_time
                    self.rate_history.append(current_rate)
                
                # Update rate history for trend analysis
                self.last_update_time = current_time
                
                # Update status message and timestamp
                if status_message:
                    self.current_status = status_message
                
                # Check for completion
                if self.current_items >= self.total_items:
                    self.is_complete = True
                    self.logger.debug(f"Progress bar completed: {self.bar_id}")
        
        except Exception as e:
            self.logger.error(f"Error updating progress bar {self.bar_id}: {e}")
    
    def increment(self, increment_amount: int = 1, status_message: str = '') -> None:
        """
        Increment progress by specified amount with automatic display update.
        
        This method provides convenient progress incrementation with automatic bounds checking
        and display update for simple progress tracking scenarios.
        
        Args:
            increment_amount: Amount to increment progress (default: 1)
            status_message: Optional status message to display
        """
        try:
            # Validate increment amount against remaining items
            new_current = self.current_items + increment_amount
            
            # Update current items count and display
            self.update(
                current_items=new_current,
                status_message=status_message
            )
            
        except Exception as e:
            self.logger.error(f"Error incrementing progress bar {self.bar_id}: {e}")
    
    def finish(
        self,
        completion_message: str = 'Complete',
        show_final_stats: bool = True
    ) -> Dict[str, Any]:
        """
        Complete progress bar with final status, performance summary, and cleanup.
        
        This method finalizes the progress bar with completion status, performance statistics,
        and comprehensive summary for analysis and reporting.
        
        Args:
            completion_message: Message to display upon completion
            show_final_stats: Display final performance statistics
            
        Returns:
            Dict[str, Any]: Final progress statistics and performance summary
        """
        try:
            # Set progress to 100% completion
            with self.update_lock:
                self.current_items = self.total_items
                self.progress_percentage = 100.0
                self.is_complete = True
                self.current_status = completion_message
                
                end_time = datetime.datetime.now()
                total_elapsed = (end_time - self.start_time).total_seconds()
                
                # Calculate final performance statistics
                final_stats = {
                    'bar_id': self.bar_id,
                    'total_items': self.total_items,
                    'completed_items': self.current_items,
                    'start_time': self.start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'total_elapsed_seconds': total_elapsed,
                    'average_rate': self.current_items / total_elapsed if total_elapsed > 0 else 0,
                    'final_status': completion_message
                }
                
                # Include performance metrics summary
                if self.performance_metrics:
                    final_stats['performance_metrics'] = self.performance_metrics.copy()
                
                # Display completion message with final status
                if show_final_stats:
                    completion_display = format_progress_update(
                        message=f"{self.description} - {completion_message}",
                        status_level='SUCCESS',
                        context={
                            'items': self.current_items,
                            'elapsed': format_duration(total_elapsed),
                            'rate': format_scientific_value(final_stats['average_rate'], unit='items/sec', precision=2)
                        },
                        apply_colors=_color_support_enabled
                    )
                    print(f"\r{completion_display}")
                
                self.logger.info(f"Progress bar finished: {self.bar_id} - {completion_message}")
                
                # Return comprehensive completion statistics
                return final_stats
        
        except Exception as e:
            self.logger.error(f"Error finishing progress bar {self.bar_id}: {e}")
            return {'error': str(e)}
    
    def get_eta(self) -> float:
        """
        Calculate estimated time to completion based on current progress rate and remaining work.
        
        This method calculates ETA using historical rate data with smoothing to reduce fluctuations
        and provide more stable time estimates for user planning.
        
        Returns:
            float: Estimated time to completion in seconds
        """
        try:
            if self.is_complete or self.current_items >= self.total_items:
                return 0.0
            
            # Calculate current processing rate from rate history
            if not self.rate_history:
                return 0.0
            
            # Use smoothed average rate for more stable ETA
            average_rate = sum(self.rate_history) / len(self.rate_history)
            
            # Determine remaining items to process
            remaining_items = self.total_items - self.current_items
            
            # Calculate estimated time based on current rate
            if average_rate > 0:
                estimated_seconds = remaining_items / average_rate
                return estimated_seconds
            else:
                return 0.0
        
        except Exception as e:
            self.logger.error(f"Error calculating ETA for {self.bar_id}: {e}")
            return 0.0
    
    def render(self, force_redraw: bool = False) -> str:
        """
        Render progress bar visual representation with colors, metrics, and status information.
        
        This method generates the complete visual representation of the progress bar with color coding,
        performance metrics, status information, and scientific value formatting.
        
        Args:
            force_redraw: Force complete redraw of progress bar
            
        Returns:
            str: Formatted progress bar string ready for terminal display
        """
        try:
            # Calculate filled and empty portions of progress bar
            filled_width = int(self.bar_width * self.progress_percentage / 100)
            empty_width = self.bar_width - filled_width
            
            # Create progress bar characters
            filled_char = PROGRESS_BAR_CHARS['FILLED']
            empty_char = PROGRESS_BAR_CHARS['EMPTY']
            
            # Apply color coding based on progress status and performance thresholds
            if _color_support_enabled:
                if self.is_complete:
                    bar_color = TERMINAL_COLORS['GREEN']
                elif self.progress_percentage >= 80:
                    bar_color = TERMINAL_COLORS['BLUE']
                elif self.progress_percentage >= 50:
                    bar_color = TERMINAL_COLORS['YELLOW']
                else:
                    bar_color = TERMINAL_COLORS['WHITE']
                
                reset_color = TERMINAL_COLORS['RESET']
                filled_bar = f"{bar_color}{filled_char * filled_width}{reset_color}"
                empty_bar = f"{TERMINAL_COLORS['DIM']}{empty_char * empty_width}{reset_color}"
            else:
                filled_bar = filled_char * filled_width
                empty_bar = empty_char * empty_width
            
            # Create base progress bar
            progress_bar = f"[{filled_bar}{empty_bar}]"
            
            # Format percentage display if enabled
            percentage_str = ''
            if self.show_percentage:
                percentage_str = f" {self.progress_percentage:5.1f}%"
            
            # Include ETA and rate information if enabled
            eta_str = ''
            if self.show_eta and not self.is_complete:
                eta_seconds = self.get_eta()
                if eta_seconds > 0:
                    eta_formatted = format_duration(eta_seconds, format_style='compact')
                    eta_str = f" ETA: {eta_formatted}"
            
            rate_str = ''
            if self.show_rate and self.rate_history:
                current_rate = sum(self.rate_history) / len(self.rate_history)
                rate_formatted = format_scientific_value(current_rate, unit='items/sec', precision=1)
                rate_str = f" Rate: {rate_formatted}"
            
            # Add performance metrics display
            metrics_str = ''
            if self.performance_metrics:
                metrics_parts = []
                for metric_name, metric_value in self.performance_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        formatted_metric = format_scientific_value(metric_value, precision=2)
                        metrics_parts.append(f"{metric_name}: {formatted_metric}")
                
                if metrics_parts:
                    metrics_str = f" | {', '.join(metrics_parts[:2])}"  # Limit to 2 metrics for space
            
            # Format complete progress bar string
            components = [
                self.description if self.description else self.bar_id,
                progress_bar,
                percentage_str,
                f" ({self.current_items:,}/{self.total_items:,})",
                eta_str,
                rate_str,
                metrics_str
            ]
            
            # Add current status if available
            if self.current_status:
                components.append(f" | {self.current_status}")
            
            # Combine all components
            complete_display = ''.join(components)
            
            # Ensure display fits within terminal width
            if len(re.sub(r'\033\[[0-9;]*m', '', complete_display)) > _terminal_width:
                # Truncate description if needed
                max_desc_length = _terminal_width // 4
                if self.description and len(self.description) > max_desc_length:
                    truncated_desc = self.description[:max_desc_length-3] + '...'
                    complete_display = complete_display.replace(self.description, truncated_desc, 1)
            
            return complete_display
        
        except Exception as e:
            self.logger.error(f"Error rendering progress bar {self.bar_id}: {e}")
            return f"[ERROR] Progress bar {self.bar_id}: {e}"
    
    def should_update(self) -> bool:
        """Determine if progress bar should be updated based on timing and change threshold."""
        try:
            current_time = datetime.datetime.now()
            time_since_update = (current_time - self.last_update_time).total_seconds()
            
            # Update if enough time has passed or if completed
            return time_since_update >= STATUS_UPDATE_INTERVAL_SECONDS or self.is_complete
        
        except Exception:
            return True  # Default to allowing updates


class StatusDisplay:
    """
    Hierarchical status display class providing structured status visualization with tree-like organization, 
    component status tracking, and real-time updates for complex multi-component operations.
    
    This class implements a comprehensive hierarchical status display system with tree organization,
    component tracking, performance metrics integration, and real-time updates for complex operations.
    """
    
    def __init__(
        self,
        display_id: str,
        title: str,
        display_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hierarchical status display with configuration and component tracking setup.
        
        Args:
            display_id: Unique identifier for the status display
            title: Title for the status display
            display_config: Configuration dictionary for display options
        """
        # Set display identification and title
        self.display_id = display_id
        self.title = title
        self.display_config = display_config or {}
        
        # Initialize component status tracking
        self.component_status: Dict[str, Dict[str, Any]] = {}
        self.status_hierarchy: List[str] = []
        self.last_update_time = datetime.datetime.now()
        
        # Configure display format and styling
        self.show_timestamps = self.display_config.get('show_timestamps', True)
        self.show_performance_metrics = self.display_config.get('show_performance_metrics', True)
        self.display_format = self.display_config.get('display_format', 'tree')
        
        # Create thread lock for status updates
        self.status_lock = threading.Lock()
        
        # Configure display limits and scrolling
        self.max_display_lines = self.display_config.get('max_display_lines', 50)
        self.auto_scroll = self.display_config.get('auto_scroll', True)
        
        # Initialize logger for status display operations
        self.logger = get_logger(f'status_display.{display_id}', 'STATUS')
        
        self.logger.debug(f"Initialized status display: {display_id}")
    
    def add_component(
        self,
        component_id: str,
        component_name: str,
        initial_status: str = 'PENDING',
        hierarchy_level: int = 0
    ) -> None:
        """
        Add component to status display hierarchy with initial status and configuration.
        
        This method adds a new component to the hierarchical status display with proper
        organization and initial status configuration for structured status tracking.
        
        Args:
            component_id: Unique identifier for the component
            component_name: Human-readable name for the component
            initial_status: Initial status for the component
            hierarchy_level: Level in the hierarchy tree (0 = root level)
        """
        try:
            with self.status_lock:
                # Validate component ID uniqueness
                if component_id in self.component_status:
                    self.logger.warning(f"Component already exists: {component_id}")
                    return
                
                # Create component status entry
                component_entry = {
                    'component_id': component_id,
                    'component_name': component_name,
                    'status': initial_status,
                    'hierarchy_level': hierarchy_level,
                    'creation_time': datetime.datetime.now(),
                    'last_update_time': datetime.datetime.now(),
                    'status_history': [initial_status],
                    'performance_data': {},
                    'status_icon': STATUS_ICONS.get(initial_status, STATUS_ICONS['PENDING'])
                }
                
                # Add component to status hierarchy
                self.component_status[component_id] = component_entry
                
                # Update hierarchy list maintaining order
                if component_id not in self.status_hierarchy:
                    # Insert at appropriate position based on hierarchy level
                    insert_index = len(self.status_hierarchy)
                    for i, existing_id in enumerate(self.status_hierarchy):
                        if self.component_status[existing_id]['hierarchy_level'] > hierarchy_level:
                            insert_index = i
                            break
                    
                    self.status_hierarchy.insert(insert_index, component_id)
                
                # Log component addition
                self.logger.debug(f"Added component {component_id} at level {hierarchy_level}")
        
        except Exception as e:
            self.logger.error(f"Error adding component {component_id}: {e}")
    
    def update_component_status(
        self,
        component_id: str,
        new_status: str,
        status_data: Optional[Dict[str, Any]] = None,
        status_icon: Optional[str] = None
    ) -> None:
        """
        Update status for specific component with new status, metrics, and visual indicators.
        
        This method updates component status with thread safety, status history tracking,
        and performance metrics integration for comprehensive status monitoring.
        
        Args:
            component_id: Identifier for the component to update
            new_status: New status value for the component
            status_data: Additional status data and metrics
            status_icon: Custom status icon (uses default if None)
        """
        try:
            # Acquire status lock for thread safety
            with self.status_lock:
                # Validate component exists in hierarchy
                if component_id not in self.component_status:
                    self.logger.warning(f"Component not found: {component_id}")
                    return
                
                component = self.component_status[component_id]
                
                # Update component status and data
                old_status = component['status']
                component['status'] = new_status
                component['last_update_time'] = datetime.datetime.now()
                
                # Update status history
                if new_status != old_status:
                    component['status_history'].append(new_status)
                
                # Update status icon and visual indicators
                if status_icon:
                    component['status_icon'] = status_icon
                else:
                    # Auto-select icon based on status
                    icon_mapping = {
                        'PENDING': STATUS_ICONS['PENDING'],
                        'RUNNING': STATUS_ICONS['RUNNING'],
                        'COMPLETE': STATUS_ICONS['COMPLETE'],
                        'SUCCESS': STATUS_ICONS['SUCCESS'],
                        'FAILED': STATUS_ICONS['FAILED'],
                        'ERROR': STATUS_ICONS['ERROR'],
                        'WARNING': STATUS_ICONS['WARNING']
                    }
                    component['status_icon'] = icon_mapping.get(new_status, STATUS_ICONS['INFO'])
                
                # Include additional status data
                if status_data:
                    component['status_data'] = status_data
                    
                    # Extract performance metrics if present
                    if 'performance' in status_data:
                        component['performance_data'].update(status_data['performance'])
                
                # Update last update timestamp
                self.last_update_time = datetime.datetime.now()
                
                self.logger.debug(f"Updated component {component_id}: {old_status} -> {new_status}")
        
        except Exception as e:
            self.logger.error(f"Error updating component status {component_id}: {e}")
    
    def display_status_tree(
        self,
        include_performance_data: bool = False,
        compact_format: bool = False
    ) -> None:
        """
        Display complete status hierarchy as formatted tree structure with indentation and visual indicators.
        
        This method generates and displays the complete hierarchical status tree with proper indentation,
        visual indicators, performance metrics, and color coding for comprehensive status visualization.
        
        Args:
            include_performance_data: Include performance metrics in the display
            compact_format: Use compact formatting to save screen space
        """
        try:
            # Generate hierarchical tree structure
            display_lines = []
            
            # Add title header
            title_color = TERMINAL_COLORS['BOLD'] if _color_support_enabled else ''
            reset_color = TERMINAL_COLORS['RESET'] if _color_support_enabled else ''
            header = f"{title_color}{self.title}{reset_color}"
            
            if self.show_timestamps:
                timestamp = self.last_update_time.strftime('%H:%M:%S')
                header += f" (updated: {timestamp})"
            
            display_lines.append(header)
            display_lines.append("=" * min(len(self.title), _terminal_width // 2))
            
            # Display components in hierarchy order
            for component_id in self.status_hierarchy:
                if component_id not in self.component_status:
                    continue
                
                component = self.component_status[component_id]
                
                # Apply indentation for hierarchy levels
                indent = "  " * component['hierarchy_level']
                
                # Get status icon and color
                status_icon = component['status_icon'] if _unicode_support_enabled else f"[{component['status']}]"
                
                # Apply color coding based on status
                status_color = ''
                if _color_support_enabled:
                    color_mapping = {
                        'SUCCESS': TERMINAL_COLORS['GREEN'],
                        'COMPLETE': TERMINAL_COLORS['GREEN'],
                        'RUNNING': TERMINAL_COLORS['BLUE'],
                        'WARNING': TERMINAL_COLORS['YELLOW'],
                        'ERROR': TERMINAL_COLORS['RED'],
                        'FAILED': TERMINAL_COLORS['RED'],
                        'PENDING': TERMINAL_COLORS['WHITE']
                    }
                    status_color = color_mapping.get(component['status'], TERMINAL_COLORS['WHITE'])
                
                # Format component line
                component_line = f"{indent}{status_color}{status_icon} {component['component_name']}: {component['status']}{reset_color}"
                
                # Add timing information if not compact
                if not compact_format and self.show_timestamps:
                    update_time = component['last_update_time'].strftime('%H:%M:%S')
                    component_line += f" ({update_time})"
                
                display_lines.append(component_line)
                
                # Include performance data if requested
                if include_performance_data and component['performance_data']:
                    perf_indent = "  " * (component['hierarchy_level'] + 1)
                    for metric_name, metric_value in component['performance_data'].items():
                        if isinstance(metric_value, (int, float)):
                            formatted_metric = format_scientific_value(metric_value, precision=2)
                            metric_line = f"{perf_indent}{TERMINAL_COLORS['DIM'] if _color_support_enabled else ''}{metric_name}: {formatted_metric}{reset_color}"
                            display_lines.append(metric_line)
                
                # Add status data if present and not compact
                if not compact_format and 'status_data' in component:
                    status_data = component['status_data']
                    for key, value in status_data.items():
                        if key != 'performance':  # Performance data handled separately
                            data_indent = "  " * (component['hierarchy_level'] + 1)
                            data_line = f"{data_indent}{TERMINAL_COLORS['DIM'] if _color_support_enabled else ''}{key}: {value}{reset_color}"
                            display_lines.append(data_line)
            
            # Apply auto-scroll if enabled and needed
            if self.auto_scroll and len(display_lines) > self.max_display_lines:
                display_lines = display_lines[-self.max_display_lines:]
                display_lines.insert(0, f"{TERMINAL_COLORS['DIM'] if _color_support_enabled else ''}... (showing last {self.max_display_lines} lines){reset_color}")
            
            # Output formatted status tree to terminal
            status_output = '\n'.join(display_lines)
            print(status_output)
            
            self.logger.debug(f"Displayed status tree for {self.display_id}: {len(display_lines)} lines")
        
        except Exception as e:
            self.logger.error(f"Error displaying status tree: {e}")
    
    def get_status_summary(self, include_detailed_metrics: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive status summary for all components with statistics and analysis.
        
        This method generates comprehensive status summary with statistics, component analysis,
        and performance metrics for reporting and monitoring purposes.
        
        Args:
            include_detailed_metrics: Include detailed performance metrics in summary
            
        Returns:
            Dict[str, Any]: Status summary with component statistics and analysis
        """
        try:
            # Collect status from all components
            status_counts = {}
            component_details = []
            
            for component_id, component in self.component_status.items():
                status = component['status']
                status_counts[status] = status_counts.get(status, 0) + 1
                
                component_summary = {
                    'component_id': component_id,
                    'component_name': component['component_name'],
                    'status': status,
                    'hierarchy_level': component['hierarchy_level'],
                    'last_update': component['last_update_time'].isoformat()
                }
                
                # Include detailed metrics if requested
                if include_detailed_metrics and component['performance_data']:
                    component_summary['performance_metrics'] = component['performance_data']
                
                component_details.append(component_summary)
            
            # Calculate status distribution statistics
            total_components = len(self.component_status)
            status_percentages = {}
            for status, count in status_counts.items():
                status_percentages[status] = (count / total_components * 100) if total_components > 0 else 0
            
            # Generate status summary
            summary = {
                'display_id': self.display_id,
                'title': self.title,
                'total_components': total_components,
                'status_counts': status_counts,
                'status_percentages': status_percentages,
                'last_update_time': self.last_update_time.isoformat(),
                'component_details': component_details
            }
            
            # Include overall health assessment
            if total_components > 0:
                success_rate = status_percentages.get('SUCCESS', 0) + status_percentages.get('COMPLETE', 0)
                error_rate = status_percentages.get('ERROR', 0) + status_percentages.get('FAILED', 0)
                
                if error_rate > 10:
                    health_status = 'CRITICAL'
                elif error_rate > 5:
                    health_status = 'WARNING'
                elif success_rate > 80:
                    health_status = 'HEALTHY'
                else:
                    health_status = 'UNKNOWN'
                
                summary['overall_health'] = {
                    'status': health_status,
                    'success_rate': success_rate,
                    'error_rate': error_rate
                }
            
            return summary
        
        except Exception as e:
            self.logger.error(f"Error generating status summary: {e}")
            return {'error': str(e)}
    
    def clear_display(self, preserve_configuration: bool = True) -> None:
        """
        Clear status display and reset component hierarchy for fresh status tracking.
        
        This method clears all status information while optionally preserving configuration
        for fresh status tracking sessions or display reset operations.
        
        Args:
            preserve_configuration: Keep display configuration settings
        """
        try:
            with self.status_lock:
                # Clear all component status entries
                self.component_status.clear()
                self.status_hierarchy.clear()
                
                # Reset update timestamps
                self.last_update_time = datetime.datetime.now()
                
                # Preserve configuration if requested
                if not preserve_configuration:
                    self.display_config.clear()
                    self.show_timestamps = True
                    self.show_performance_metrics = True
                    self.display_format = 'tree'
                
                self.logger.debug(f"Cleared status display: {self.display_id}")
        
        except Exception as e:
            self.logger.error(f"Error clearing status display: {e}")


class PerformanceMetricsDisplay:
    """
    Specialized display class for real-time performance metrics visualization with threshold indicators, 
    trend analysis, and scientific formatting optimized for batch processing monitoring.
    
    This class provides comprehensive performance metrics visualization with threshold monitoring,
    trend analysis, scientific formatting, and real-time updates for scientific computing workflows.
    """
    
    def __init__(
        self,
        display_id: str,
        performance_thresholds: Dict[str, float],
        display_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize performance metrics display with thresholds, formatting, and trend tracking configuration.
        
        Args:
            display_id: Unique identifier for the metrics display
            performance_thresholds: Performance thresholds for warning and critical levels
            display_config: Configuration dictionary for display options
        """
        # Set display identification and thresholds
        self.display_id = display_id
        self.performance_thresholds = performance_thresholds
        self.display_config = display_config or {}
        
        # Initialize metrics tracking and history
        self.current_metrics: Dict[str, float] = {}
        self.metrics_history = collections.deque(maxlen=100)  # Keep last 100 measurements
        
        # Configure metric formatting options
        self.metric_units: Dict[str, str] = {
            'processing_time': 'seconds',
            'memory_usage': 'percent',
            'cpu_usage': 'percent',
            'disk_io': 'MB/s',
            'network_io': 'MB/s',
            'throughput': 'items/sec'
        }
        
        self.metric_precision: Dict[str, int] = {
            'processing_time': 3,
            'memory_usage': 1,
            'cpu_usage': 1,
            'disk_io': 2,
            'network_io': 2,
            'throughput': 2
        }
        
        # Configure display options
        self.show_trends = self.display_config.get('show_trends', False)
        self.highlight_violations = self.display_config.get('highlight_violations', True)
        self.display_format = self.display_config.get('display_format', 'table')
        
        # Initialize timing and threading
        self.last_update_time = datetime.datetime.now()
        self.metrics_lock = threading.Lock()
        
        # Initialize logger for metrics display
        self.logger = get_logger(f'metrics_display.{display_id}', 'PERFORMANCE')
        
        self.logger.debug(f"Initialized performance metrics display: {display_id}")
    
    def update_metrics(
        self,
        new_metrics: Dict[str, float],
        trigger_threshold_check: bool = True
    ) -> List[str]:
        """
        Update performance metrics with new values, threshold validation, and trend analysis.
        
        This method updates performance metrics with thread safety, threshold validation,
        and trend tracking for comprehensive performance monitoring and analysis.
        
        Args:
            new_metrics: Dictionary of new metric values to update
            trigger_threshold_check: Perform threshold validation on update
            
        Returns:
            List[str]: List of threshold violations detected
        """
        violations = []
        
        try:
            # Acquire metrics lock for thread safety
            with self.metrics_lock:
                # Update current metrics with new values
                self.current_metrics.update(new_metrics)
                
                # Add metrics to history for trend analysis
                history_entry = {
                    'timestamp': datetime.datetime.now(),
                    'metrics': self.current_metrics.copy()
                }
                self.metrics_history.append(history_entry)
                
                # Update last update timestamp
                self.last_update_time = datetime.datetime.now()
                
                # Validate metrics against thresholds if requested
                if trigger_threshold_check:
                    violations = self.check_threshold_violations(new_metrics)
            
            self.logger.debug(f"Updated metrics: {len(new_metrics)} metrics, {len(violations)} violations")
            
            # Return list of violations detected
            return violations
        
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
            return []
    
    def display_metrics_table(
        self,
        metrics_to_display: Optional[List[str]] = None,
        include_trends: bool = False
    ) -> None:
        """
        Display performance metrics as formatted table with threshold indicators and color coding.
        
        This method creates and displays a comprehensive metrics table with threshold indicators,
        color coding, trend analysis, and scientific value formatting.
        
        Args:
            metrics_to_display: Specific metrics to display (all if None)
            include_trends: Include trend indicators in the table
        """
        try:
            if not self.current_metrics:
                print("No performance metrics available")
                return
            
            # Determine metrics to display
            if metrics_to_display is None:
                display_metrics = list(self.current_metrics.keys())
            else:
                display_metrics = [m for m in metrics_to_display if m in self.current_metrics]
            
            # Create table data structure
            table_data = []
            headers = ['Metric', 'Current Value', 'Status']
            
            if include_trends:
                headers.append('Trend')
            
            headers.extend(['Warning Threshold', 'Critical Threshold'])
            
            # Format metrics with scientific precision
            for metric_name in display_metrics:
                metric_value = self.current_metrics[metric_name]
                
                # Format current value with appropriate precision and units
                precision = self.metric_precision.get(metric_name, SCIENTIFIC_PRECISION_DIGITS)
                unit = self.metric_units.get(metric_name, '')
                formatted_value = format_scientific_value(metric_value, unit=unit, precision=precision)
                
                # Determine status and color coding
                status = 'OK'
                status_color = TERMINAL_COLORS['GREEN'] if _color_support_enabled else ''
                
                threshold_data = self.performance_thresholds.get(metric_name, {})
                warning_threshold = threshold_data.get('warning', float('inf'))
                critical_threshold = threshold_data.get('critical', float('inf'))
                
                if metric_value >= critical_threshold:
                    status = 'CRITICAL'
                    status_color = TERMINAL_COLORS['RED']
                elif metric_value >= warning_threshold:
                    status = 'WARNING'
                    status_color = TERMINAL_COLORS['YELLOW']
                
                reset_color = TERMINAL_COLORS['RESET'] if _color_support_enabled else ''
                colored_status = f"{status_color}{status}{reset_color}"
                
                # Create table row
                row_data = {
                    'Metric': metric_name,
                    'Current Value': formatted_value,
                    'Status': colored_status,
                    'Warning Threshold': format_scientific_value(warning_threshold, unit=unit, precision=precision) if warning_threshold != float('inf') else 'N/A',
                    'Critical Threshold': format_scientific_value(critical_threshold, unit=unit, precision=precision) if critical_threshold != float('inf') else 'N/A'
                }
                
                # Include trend indicators if requested
                if include_trends:
                    trend_analysis = self.get_trend_analysis(metric_name, analysis_window=10)
                    trend_symbol = trend_analysis.get('trend_symbol', '→')
                    row_data['Trend'] = trend_symbol
                
                table_data.append(row_data)
            
            # Create and display formatted table
            table_output = create_status_table(
                table_data=table_data,
                column_headers=headers,
                column_formats={'Current Value': 'scientific'},
                table_width=_terminal_width,
                include_borders=True,
                color_scheme='default'
            )
            
            # Display table with header
            print("\nPerformance Metrics:")
            print(table_output)
            
            # Show update timestamp
            timestamp = self.last_update_time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"Last updated: {timestamp}")
            
            self.logger.debug(f"Displayed metrics table: {len(display_metrics)} metrics")
        
        except Exception as e:
            self.logger.error(f"Error displaying metrics table: {e}")
    
    def get_trend_analysis(
        self,
        metric_name: str,
        analysis_window: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze performance metrics trends and generate trend indicators for display.
        
        This method analyzes historical metric data to determine trends, rates of change,
        and statistical indicators for performance trend visualization.
        
        Args:
            metric_name: Name of the metric to analyze
            analysis_window: Number of historical data points to analyze
            
        Returns:
            Dict[str, Any]: Trend analysis with direction, rate, and statistical indicators
        """
        try:
            # Extract metric history for analysis window
            recent_history = list(self.metrics_history)[-analysis_window:]
            
            if len(recent_history) < 2:
                return {
                    'trend_direction': 'unknown',
                    'trend_symbol': '?',
                    'rate_of_change': 0.0,
                    'confidence': 0.0
                }
            
            # Extract metric values and timestamps
            values = []
            timestamps = []
            
            for entry in recent_history:
                if metric_name in entry['metrics']:
                    values.append(entry['metrics'][metric_name])
                    timestamps.append(entry['timestamp'])
            
            if len(values) < 2:
                return {
                    'trend_direction': 'unknown',
                    'trend_symbol': '?',
                    'rate_of_change': 0.0,
                    'confidence': 0.0
                }
            
            # Calculate trend direction and rate of change
            first_value = values[0]
            last_value = values[-1]
            value_change = last_value - first_value
            
            time_span = (timestamps[-1] - timestamps[0]).total_seconds()
            rate_of_change = value_change / time_span if time_span > 0 else 0.0
            
            # Determine trend direction
            change_threshold = abs(first_value) * 0.05  # 5% change threshold
            
            if abs(value_change) < change_threshold:
                trend_direction = 'stable'
                trend_symbol = '→'
            elif value_change > 0:
                trend_direction = 'increasing'
                trend_symbol = '↗' if _unicode_support_enabled else '^'
            else:
                trend_direction = 'decreasing'
                trend_symbol = '↘' if _unicode_support_enabled else 'v'
            
            # Calculate trend confidence based on data consistency
            if len(values) >= 3:
                # Simple confidence calculation based on monotonicity
                increasing_count = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
                decreasing_count = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
                total_comparisons = len(values) - 1
                
                if trend_direction == 'increasing':
                    confidence = increasing_count / total_comparisons
                elif trend_direction == 'decreasing':
                    confidence = decreasing_count / total_comparisons
                else:
                    confidence = 1.0 - max(increasing_count, decreasing_count) / total_comparisons
            else:
                confidence = 0.5  # Low confidence with limited data
            
            # Return comprehensive trend analysis
            return {
                'trend_direction': trend_direction,
                'trend_symbol': trend_symbol,
                'rate_of_change': rate_of_change,
                'confidence': confidence,
                'data_points': len(values),
                'time_span_seconds': time_span
            }
        
        except Exception as e:
            self.logger.error(f"Error analyzing trend for {metric_name}: {e}")
            return {
                'trend_direction': 'error',
                'trend_symbol': '!',
                'rate_of_change': 0.0,
                'confidence': 0.0
            }
    
    def check_threshold_violations(
        self,
        metrics_to_check: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Check current metrics against configured thresholds and identify violations.
        
        This method compares metrics against warning and critical thresholds to identify
        violations and generate alerts with severity levels and recommended actions.
        
        Args:
            metrics_to_check: Dictionary of metrics to validate against thresholds
            
        Returns:
            List[Dict[str, Any]]: List of threshold violations with severity and details
        """
        violations = []
        
        try:
            for metric_name, metric_value in metrics_to_check.items():
                if metric_name not in self.performance_thresholds:
                    continue
                
                threshold_data = self.performance_thresholds[metric_name]
                warning_threshold = threshold_data.get('warning', float('inf'))
                critical_threshold = threshold_data.get('critical', float('inf'))
                
                violation = None
                
                # Compare metrics against critical thresholds
                if metric_value >= critical_threshold:
                    violation = {
                        'metric_name': metric_name,
                        'current_value': metric_value,
                        'threshold_value': critical_threshold,
                        'severity': 'CRITICAL',
                        'violation_percentage': ((metric_value - critical_threshold) / critical_threshold * 100) if critical_threshold > 0 else 0,
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                
                # Compare metrics against warning thresholds
                elif metric_value >= warning_threshold:
                    violation = {
                        'metric_name': metric_name,
                        'current_value': metric_value,
                        'threshold_value': warning_threshold,
                        'severity': 'WARNING',
                        'violation_percentage': ((metric_value - warning_threshold) / warning_threshold * 100) if warning_threshold > 0 else 0,
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                
                # Add violation details and recommendations
                if violation:
                    # Format values for display
                    unit = self.metric_units.get(metric_name, '')
                    precision = self.metric_precision.get(metric_name, SCIENTIFIC_PRECISION_DIGITS)
                    
                    violation['formatted_current'] = format_scientific_value(metric_value, unit=unit, precision=precision)
                    violation['formatted_threshold'] = format_scientific_value(violation['threshold_value'], unit=unit, precision=precision)
                    
                    # Add recommendations based on metric type
                    recommendations = self._get_violation_recommendations(metric_name, violation['severity'])
                    violation['recommendations'] = recommendations
                    
                    violations.append(violation)
            
            # Sort violations by severity (CRITICAL first)
            violations.sort(key=lambda x: 0 if x['severity'] == 'CRITICAL' else 1)
            
            return violations
        
        except Exception as e:
            self.logger.error(f"Error checking threshold violations: {e}")
            return []
    
    def _get_violation_recommendations(self, metric_name: str, severity: str) -> List[str]:
        """Generate recommendations for threshold violations based on metric type and severity."""
        recommendations = []
        
        if metric_name == 'processing_time':
            if severity == 'CRITICAL':
                recommendations.extend([
                    "Consider algorithm optimization or parameter tuning",
                    "Review system resource allocation",
                    "Check for I/O bottlenecks or memory constraints"
                ])
            else:
                recommendations.extend([
                    "Monitor for sustained performance degradation",
                    "Consider parallel processing optimization"
                ])
        
        elif metric_name == 'memory_usage':
            if severity == 'CRITICAL':
                recommendations.extend([
                    "Immediate memory cleanup required",
                    "Consider reducing batch size or data caching",
                    "Monitor for memory leaks"
                ])
            else:
                recommendations.extend([
                    "Monitor memory usage trends",
                    "Consider optimizing data structures"
                ])
        
        elif metric_name == 'cpu_usage':
            if severity == 'CRITICAL':
                recommendations.extend([
                    "Reduce computational load or increase parallelism",
                    "Check for inefficient algorithms",
                    "Consider load balancing"
                ])
            else:
                recommendations.extend([
                    "Monitor CPU usage patterns",
                    "Consider optimization opportunities"
                ])
        
        return recommendations


class TerminalManager:
    """
    Terminal management class providing comprehensive terminal control, cursor management, screen clearing, 
    and output optimization for scientific computing console applications.
    
    This class provides comprehensive terminal management with cursor control, screen clearing,
    output optimization, and cross-platform compatibility for scientific computing applications.
    """
    
    def __init__(
        self,
        enable_color_support: bool = True,
        enable_unicode_support: bool = True,
        terminal_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize terminal manager with capability detection and output optimization configuration.
        
        Args:
            enable_color_support: Enable color output support
            enable_unicode_support: Enable Unicode character support
            terminal_config: Terminal configuration options
        """
        # Configure color and Unicode support
        self.color_support_enabled = enable_color_support
        self.unicode_support_enabled = enable_unicode_support
        self.terminal_config = terminal_config or {}
        
        # Detect terminal capabilities and features
        capabilities = detect_terminal_capabilities()
        self.terminal_capabilities = capabilities
        
        # Initialize terminal dimensions and properties
        self.terminal_width = capabilities.get('width', DEFAULT_TERMINAL_WIDTH)
        self.terminal_height = capabilities.get('height', 24)
        self.is_interactive = capabilities.get('interactive', False)
        
        # Configure terminal type and features
        self.terminal_type = os.getenv('TERM', 'unknown')
        
        # Setup output buffering and optimization
        self.output_lock = threading.Lock()
        self.output_buffer: List[str] = []
        self.buffering_enabled = self.terminal_config.get('buffering_enabled', False)
        
        # Initialize logger for terminal management
        self.logger = get_logger('terminal_manager', 'SYSTEM')
        
        # Validate color and Unicode support against capabilities
        if self.color_support_enabled and not capabilities.get('color_support', False):
            self.color_support_enabled = False
            self.logger.warning("Color support disabled due to terminal limitations")
        
        if self.unicode_support_enabled and not capabilities.get('unicode_support', False):
            self.unicode_support_enabled = False
            self.logger.warning("Unicode support disabled due to terminal limitations")
        
        self.logger.debug(f"Terminal manager initialized: {self.terminal_width}x{self.terminal_height}, color={self.color_support_enabled}, unicode={self.unicode_support_enabled}")
    
    def clear_screen(self, reset_cursor: bool = True) -> None:
        """
        Clear terminal screen with appropriate escape sequences and cursor positioning.
        
        This method clears the terminal screen using appropriate ANSI escape sequences
        with optional cursor reset for clean display initialization.
        
        Args:
            reset_cursor: Reset cursor to home position after clearing
        """
        try:
            with self.output_lock:
                # Send screen clearing escape sequence
                if self.is_interactive:
                    sys.stdout.write('\033[2J')  # Clear entire screen
                    
                    # Reset cursor to home position if requested
                    if reset_cursor:
                        sys.stdout.write('\033[H')  # Move cursor to home position
                else:
                    # Non-interactive fallback: print empty lines
                    sys.stdout.write('\n' * self.terminal_height)
                
                # Flush terminal output buffer
                sys.stdout.flush()
            
            self.logger.debug("Terminal screen cleared")
        
        except Exception as e:
            self.logger.error(f"Error clearing terminal screen: {e}")
    
    def move_cursor(self, row: int, column: int) -> bool:
        """
        Move terminal cursor to specified position with bounds checking and validation.
        
        This method moves the cursor to a specific terminal position with bounds checking
        and validation to ensure cursor positioning within terminal limits.
        
        Args:
            row: Target row position (1-based)
            column: Target column position (1-based)
            
        Returns:
            bool: Success status of cursor movement
        """
        try:
            # Validate cursor position within terminal bounds
            if row < 1 or row > self.terminal_height:
                self.logger.warning(f"Row position out of bounds: {row}")
                return False
            
            if column < 1 or column > self.terminal_width:
                self.logger.warning(f"Column position out of bounds: {column}")
                return False
            
            with self.output_lock:
                # Generate cursor positioning escape sequence
                if self.is_interactive:
                    escape_sequence = f'\033[{row};{column}H'
                    sys.stdout.write(escape_sequence)
                    sys.stdout.flush()
                    return True
                else:
                    # Non-interactive terminals don't support cursor positioning
                    return False
        
        except Exception as e:
            self.logger.error(f"Error moving cursor to {row},{column}: {e}")
            return False
    
    def get_terminal_size(self) -> Tuple[int, int]:
        """
        Get current terminal dimensions with fallback to default values.
        
        This method retrieves current terminal dimensions with automatic fallback
        to default values for environments where size detection fails.
        
        Returns:
            Tuple[int, int]: Terminal width and height in characters
        """
        try:
            # Query terminal size using system calls
            try:
                size = shutil.get_terminal_size(fallback=(DEFAULT_TERMINAL_WIDTH, 24))
                width, height = size.columns, size.lines
            except (AttributeError, OSError):
                # Apply fallback values if query fails
                width, height = DEFAULT_TERMINAL_WIDTH, 24
            
            # Update cached terminal dimensions
            self.terminal_width = width
            self.terminal_height = height
            
            # Return width and height tuple
            return width, height
        
        except Exception as e:
            self.logger.error(f"Error getting terminal size: {e}")
            return DEFAULT_TERMINAL_WIDTH, 24
    
    def write_output(
        self,
        output_text: str,
        flush_immediately: bool = False
    ) -> None:
        """
        Write output to terminal with buffering, formatting, and thread safety.
        
        This method provides thread-safe output writing with optional buffering
        and immediate flushing for optimal performance and output control.
        
        Args:
            output_text: Text to write to terminal
            flush_immediately: Force immediate output flush
        """
        try:
            # Acquire output lock for thread safety
            with self.output_lock:
                if self.buffering_enabled and not flush_immediately:
                    # Add output to buffer if buffering enabled
                    self.output_buffer.append(output_text)
                else:
                    # Write output directly if immediate flush requested
                    sys.stdout.write(output_text)
                    
                    # Flush output buffer if needed
                    if flush_immediately:
                        sys.stdout.flush()
        
        except Exception as e:
            self.logger.error(f"Error writing terminal output: {e}")
    
    def flush_output(self) -> None:
        """
        Flush buffered output to terminal and ensure all content is displayed.
        
        This method flushes all buffered output to the terminal with thread safety
        and proper error handling for reliable output delivery.
        """
        try:
            # Acquire output lock for exclusive access
            with self.output_lock:
                # Write all buffered output to terminal
                if self.output_buffer:
                    for output_line in self.output_buffer:
                        sys.stdout.write(output_line)
                    
                    # Clear output buffer
                    self.output_buffer.clear()
                
                # Flush system output streams
                sys.stdout.flush()
        
        except Exception as e:
            self.logger.error(f"Error flushing terminal output: {e}")


def _get_metric_color(metric_name: str, metric_value: float) -> str:
    """Get color code for metric based on threshold comparison."""
    if not _color_support_enabled:
        return ''
    
    if metric_name in _performance_thresholds:
        threshold_data = _performance_thresholds[metric_name]
        
        if metric_value >= threshold_data.get('critical', float('inf')):
            return TERMINAL_COLORS['RED']
        elif metric_value >= threshold_data.get('warning', float('inf')):
            return TERMINAL_COLORS['YELLOW']
        else:
            return TERMINAL_COLORS['GREEN']
    
    return TERMINAL_COLORS['WHITE']