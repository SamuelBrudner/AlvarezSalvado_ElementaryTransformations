"""
Comprehensive analysis module initialization providing centralized access to all plume navigation 
simulation analysis components including performance metrics calculation, statistical comparison, 
trajectory analysis, success rate calculation, path efficiency analysis, temporal dynamics analysis, 
robustness assessment, scientific visualization, and report generation.

This module implements a unified analysis interface with >95% correlation validation, >0.99 
reproducibility requirements, and scientific computing excellence for reproducible research 
outcomes and algorithm optimization. The module provides enterprise-grade analysis capabilities 
for the plume navigation simulation system with comprehensive performance metrics, statistical 
validation, and scientific documentation standards.

Key Features:
- Performance metrics calculation with caching and statistical validation
- Statistical comparison framework for algorithm validation across experimental conditions
- Comprehensive trajectory analysis with similarity metrics and pattern recognition
- Success rate calculation with real-time monitoring and validation
- Path efficiency analysis with optimization assessment and comparison
- Temporal dynamics analysis with response times and velocity profiles
- Robustness assessment with noise tolerance and environmental adaptation analysis
- Scientific visualization with publication-ready figure generation
- Report generation with templating and multi-format output capabilities
- Cross-platform compatibility analysis for Crimaldi and custom plume formats

Compliance Standards:
- Simulation accuracy >95% correlation with reference implementations
- Result reproducibility coefficient >0.99 across computational environments
- Scientific computing precision with numerical validation and error handling
- Comprehensive audit trail and traceability for reproducible research
- Real-time performance monitoring and quality assurance validation
"""

# =============================================================================
# ANALYSIS MODULE METADATA AND VERSION INFORMATION
# =============================================================================

# Analysis module version for compatibility tracking and validation
ANALYSIS_MODULE_VERSION: str = '1.0.0'

# Default analysis configuration with performance optimization and validation settings
DEFAULT_ANALYSIS_CONFIG: dict = {
    'enable_caching': True,
    'enable_statistical_validation': True,
    'enable_real_time_monitoring': True,
    'correlation_threshold': 0.95,
    'reproducibility_threshold': 0.99,
    'numerical_precision_threshold': 1e-6,
    'performance_tracking_enabled': True,
    'audit_trail_enabled': True,
    'cross_format_validation': True,
    'scientific_documentation_enabled': True
}

# Supported analysis types for comprehensive system coverage
SUPPORTED_ANALYSIS_TYPES: list = [
    'performance_metrics',
    'statistical_comparison', 
    'trajectory_analysis',
    'success_rate_calculation',
    'path_efficiency',
    'temporal_dynamics',
    'robustness_analysis',
    'visualization',
    'report_generation',
    'cross_platform_compatibility'
]

# Analysis validation and reproducibility control flags
ANALYSIS_VALIDATION_ENABLED: bool = True
CROSS_ALGORITHM_COMPARISON_ENABLED: bool = True
SCIENTIFIC_REPRODUCIBILITY_ENABLED: bool = True
CORRELATION_VALIDATION_THRESHOLD: float = 0.95
REPRODUCIBILITY_VALIDATION_THRESHOLD: float = 0.99

# =============================================================================
# PERFORMANCE METRICS ANALYSIS IMPORTS
# =============================================================================

# Core performance metrics calculation with caching and statistical validation
from .performance_metrics import (
    PerformanceMetricsCalculator,  # Comprehensive performance metrics calculation with caching and statistical validation
    NavigationSuccessAnalyzer,     # Specialized navigation success metrics analysis with statistical validation
    PathEfficiencyAnalyzer,        # Specialized path efficiency analysis with optimal path comparison
    calculate_navigation_success_metrics,  # Calculate comprehensive navigation success metrics with statistical validation
    calculate_path_efficiency_metrics      # Calculate path efficiency metrics with optimization analysis
)

# =============================================================================
# STATISTICAL COMPARISON AND VALIDATION IMPORTS
# =============================================================================

# Statistical comparison framework for algorithm validation across experimental conditions
from .statistical_comparison import (
    StatisticalComparator,          # Comprehensive statistical comparison with advanced analysis methods and hypothesis testing
    AlgorithmRankingAnalyzer,       # Specialized algorithm ranking with statistical significance testing and stability assessment
    compare_algorithm_performance,   # Compare performance metrics across multiple navigation algorithms with comprehensive statistical analysis
    validate_simulation_reproducibility  # Validate simulation reproducibility with ICC analysis and >0.99 threshold validation
)

# =============================================================================
# TRAJECTORY ANALYSIS AND PATTERN RECOGNITION IMPORTS
# =============================================================================

# Comprehensive trajectory analysis with similarity metrics, efficiency analysis, and pattern recognition
from .trajectory_analysis import (
    TrajectoryAnalyzer,                    # Comprehensive trajectory analysis with similarity metrics, efficiency analysis, and pattern recognition
    MovementPatternClassifier,             # Movement pattern classification and navigation strategy identification using machine learning
    calculate_trajectory_similarity_matrix, # Calculate comprehensive trajectory similarity matrices using multiple distance metrics
    extract_trajectory_features            # Extract comprehensive trajectory features for movement pattern analysis
)

# =============================================================================
# SUCCESS RATE CALCULATION AND MONITORING IMPORTS
# =============================================================================

# Comprehensive success rate calculation with caching, statistical validation, and real-time monitoring
from .success_rate_calculator import (
    SuccessRateCalculator,           # Comprehensive success rate calculation with caching, statistical validation, and real-time monitoring
    LocalizationSuccessAnalyzer,     # Specialized localization success analysis with accuracy metrics and spatial error assessment
    calculate_overall_success_rate   # Calculate overall navigation success rate with comprehensive criteria evaluation and statistical validation
)

# =============================================================================
# PATH EFFICIENCY ANALYSIS AND OPTIMIZATION IMPORTS
# =============================================================================

# Comprehensive path efficiency analysis with optimization assessment and cross-algorithm comparison
from .path_efficiency_analyzer import (
    PathEfficiencyAnalyzer,    # Comprehensive path efficiency analysis with optimization assessment and cross-algorithm comparison
    OptimalPathCalculator,     # Specialized optimal path calculation with multiple optimization algorithms and performance comparison
    calculate_path_length_ratio  # Calculate path length ratio for trajectory efficiency assessment
)

# =============================================================================
# TEMPORAL DYNAMICS ANALYSIS IMPORTS
# =============================================================================

# Comprehensive temporal dynamics analysis with response times, velocity profiles, and movement phases
from .temporal_dynamics_analyzer import (
    TemporalDynamicsAnalyzer,  # Comprehensive temporal dynamics analysis with response times, velocity profiles, and movement phases
    ResponseTimeAnalyzer,      # Specialized response time analysis with statistical validation and confidence intervals
    calculate_response_times   # Calculate comprehensive response times for real-time navigation capability assessment
)

# =============================================================================
# ROBUSTNESS ANALYSIS AND ENVIRONMENTAL ADAPTATION IMPORTS
# =============================================================================

# Comprehensive robustness analysis with advanced metrics and statistical validation across environmental conditions
from .robustness_analyzer import (
    RobustnessAnalyzer,        # Comprehensive robustness analysis with advanced metrics and statistical validation across environmental conditions
    NoiseToleranceAnalyzer,    # Specialized noise tolerance analysis with detailed noise type assessment and adaptation analysis
    analyze_noise_tolerance    # Analyze algorithm noise tolerance across different noise types with statistical validation
)

# =============================================================================
# SCIENTIFIC VISUALIZATION AND PLOTTING IMPORTS
# =============================================================================

# Comprehensive scientific visualization with publication-ready figure generation and interactive capabilities
from .visualization import (
    ScientificVisualizer,      # Comprehensive scientific visualization with publication-ready figure generation and interactive capabilities
    TrajectoryPlotter,         # Specialized trajectory plotting with advanced visualization and animation capabilities
    create_trajectory_plot,    # Create comprehensive trajectory visualization with scientific formatting and publication standards
    create_performance_chart   # Create performance comparison charts with statistical analysis and algorithm ranking
)

# =============================================================================
# REPORT GENERATION AND DOCUMENTATION IMPORTS
# =============================================================================

# Comprehensive report generation with templating, multi-format output, and scientific documentation standards
from .report_generator import (
    ReportGenerator,           # Comprehensive report generation with templating, multi-format output, and scientific documentation standards
    GeneratedReport,           # Generated report container with content management and export capabilities
    generate_simulation_report, # Generate comprehensive simulation reports with performance analysis and scientific documentation
    generate_batch_report      # Generate batch analysis reports with cross-algorithm comparison and performance trends
)

# =============================================================================
# UTILITY AND SUPPORT MODULE IMPORTS
# =============================================================================

# Import comprehensive utilities for logging, statistics, and scientific constants
from ...utils.logging_utils import (
    get_logger,
    set_scientific_context,
    log_performance_metrics,
    create_audit_trail,
    LoggingContext
)

from ...utils.scientific_constants import (
    get_performance_thresholds,
    get_statistical_constants,
    validate_constant_precision,
    PhysicalConstants,
    PerformanceThresholds
)

# =============================================================================
# ANALYSIS MODULE INITIALIZATION AND CONFIGURATION
# =============================================================================

def initialize_analysis_module(
    config: dict = None,
    enable_logging: bool = True,
    enable_performance_tracking: bool = True,
    validate_initialization: bool = True
) -> bool:
    """
    Initialize the comprehensive analysis module with configuration validation, logging setup,
    performance tracking, and scientific context management for reproducible research outcomes.
    
    This function sets up the entire analysis infrastructure including performance thresholds,
    statistical validation parameters, logging configuration, and scientific context management
    to ensure >95% correlation validation and >0.99 reproducibility requirements.
    
    Args:
        config: Custom configuration dictionary for analysis parameters
        enable_logging: Enable comprehensive logging and audit trail tracking
        enable_performance_tracking: Enable real-time performance monitoring
        validate_initialization: Enable initialization validation and verification
        
    Returns:
        bool: Success status of analysis module initialization with validation results
    """
    try:
        # Set up comprehensive configuration with defaults and validation
        analysis_config = DEFAULT_ANALYSIS_CONFIG.copy()
        if config:
            analysis_config.update(config)
        
        # Initialize logging infrastructure if enable_logging is True
        if enable_logging:
            analysis_logger = get_logger('analysis.module', 'ANALYSIS')
            analysis_logger.info(f"Initializing analysis module v{ANALYSIS_MODULE_VERSION}")
            
            # Set scientific context for analysis module initialization
            set_scientific_context(
                simulation_id='ANALYSIS_INIT',
                algorithm_name='MODULE_INITIALIZATION',
                processing_stage='INITIALIZATION',
                additional_context={
                    'module_version': ANALYSIS_MODULE_VERSION,
                    'config_validation': validate_initialization,
                    'performance_tracking': enable_performance_tracking
                }
            )
        
        # Configure performance thresholds and validation parameters
        performance_thresholds = get_performance_thresholds(
            threshold_category='all',
            include_derived_thresholds=True
        )
        
        # Validate correlation and reproducibility thresholds
        if analysis_config['correlation_threshold'] < CORRELATION_VALIDATION_THRESHOLD:
            raise ValueError(f"Correlation threshold {analysis_config['correlation_threshold']} below minimum {CORRELATION_VALIDATION_THRESHOLD}")
        
        if analysis_config['reproducibility_threshold'] < REPRODUCIBILITY_VALIDATION_THRESHOLD:
            raise ValueError(f"Reproducibility threshold {analysis_config['reproducibility_threshold']} below minimum {REPRODUCIBILITY_VALIDATION_THRESHOLD}")
        
        # Initialize statistical constants for comprehensive analysis
        statistical_constants = get_statistical_constants(
            analysis_type='general',
            confidence_level=analysis_config.get('confidence_level', 0.95)
        )
        
        # Setup performance tracking if enable_performance_tracking is True
        if enable_performance_tracking:
            # Initialize performance monitoring for analysis components
            for analysis_type in SUPPORTED_ANALYSIS_TYPES:
                if enable_logging:
                    log_performance_metrics(
                        metric_name='initialization_time',
                        metric_value=0.0,
                        metric_unit='seconds',
                        component=f'ANALYSIS_{analysis_type.upper()}',
                        metric_context={
                            'module_version': ANALYSIS_MODULE_VERSION,
                            'initialization_stage': 'SETUP'
                        }
                    )
        
        # Validate initialization components if validate_initialization is True
        if validate_initialization:
            # Verify all analysis components are properly imported and accessible
            required_components = [
                'PerformanceMetricsCalculator',
                'StatisticalComparator',
                'TrajectoryAnalyzer',
                'SuccessRateCalculator',
                'PathEfficiencyAnalyzer',
                'TemporalDynamicsAnalyzer',
                'RobustnessAnalyzer',
                'ScientificVisualizer',
                'ReportGenerator'
            ]
            
            for component in required_components:
                if component not in globals():
                    raise ImportError(f"Required analysis component not available: {component}")
            
            # Validate numerical precision requirements
            if not validate_constant_precision(
                analysis_config['correlation_threshold'], 
                'correlation_threshold',
                analysis_config['numerical_precision_threshold']
            ):
                raise ValueError("Correlation threshold precision validation failed")
        
        # Create audit trail for successful initialization
        if enable_logging:
            create_audit_trail(
                action='ANALYSIS_MODULE_INIT',
                component='ANALYSIS_MODULE',
                action_details={
                    'module_version': ANALYSIS_MODULE_VERSION,
                    'configuration': analysis_config,
                    'supported_analysis_types': SUPPORTED_ANALYSIS_TYPES,
                    'validation_enabled': ANALYSIS_VALIDATION_ENABLED,
                    'cross_algorithm_comparison': CROSS_ALGORITHM_COMPARISON_ENABLED,
                    'scientific_reproducibility': SCIENTIFIC_REPRODUCIBILITY_ENABLED
                },
                user_context='SYSTEM'
            )
            
            analysis_logger.info("Analysis module initialization completed successfully")
            analysis_logger.debug(f"Configuration: {analysis_config}")
            analysis_logger.debug(f"Supported analysis types: {SUPPORTED_ANALYSIS_TYPES}")
        
        return True
        
    except Exception as e:
        if enable_logging:
            error_logger = get_logger('analysis.module.error', 'ANALYSIS')
            error_logger.error(f"Analysis module initialization failed: {e}", exc_info=True)
            
            # Create audit trail for initialization failure
            create_audit_trail(
                action='ANALYSIS_MODULE_INIT_FAILED',
                component='ANALYSIS_MODULE',
                action_details={
                    'error_message': str(e),
                    'error_type': type(e).__name__,
                    'configuration_attempted': config or DEFAULT_ANALYSIS_CONFIG
                },
                user_context='SYSTEM'
            )
        
        return False


def validate_analysis_configuration(
    config: dict,
    strict_validation: bool = True,
    generate_recommendations: bool = True
) -> dict:
    """
    Validate analysis configuration against scientific computing requirements, performance
    thresholds, and reproducibility standards with comprehensive validation reporting.
    
    This function performs thorough validation of analysis configuration parameters to ensure
    compliance with scientific computing standards, performance requirements, and reproducibility
    criteria for reliable research outcomes.
    
    Args:
        config: Analysis configuration dictionary to validate
        strict_validation: Enable strict validation against all requirements
        generate_recommendations: Generate configuration improvement recommendations
        
    Returns:
        dict: Validation results with status, issues, and recommendations
    """
    validation_results = {
        'overall_valid': True,
        'validation_issues': [],
        'configuration_recommendations': [],
        'compliance_status': {},
        'performance_assessment': {}
    }
    
    try:
        # Validate required configuration parameters
        required_parameters = [
            'correlation_threshold',
            'reproducibility_threshold',
            'enable_caching',
            'enable_statistical_validation'
        ]
        
        for param in required_parameters:
            if param not in config:
                validation_results['overall_valid'] = False
                validation_results['validation_issues'].append(f"Missing required parameter: {param}")
        
        # Validate correlation threshold requirements
        if 'correlation_threshold' in config:
            correlation_threshold = config['correlation_threshold']
            if correlation_threshold < CORRELATION_VALIDATION_THRESHOLD:
                validation_results['overall_valid'] = False
                validation_results['validation_issues'].append(
                    f"Correlation threshold {correlation_threshold} below minimum {CORRELATION_VALIDATION_THRESHOLD}"
                )
            
            validation_results['compliance_status']['correlation_threshold'] = {
                'valid': correlation_threshold >= CORRELATION_VALIDATION_THRESHOLD,
                'value': correlation_threshold,
                'requirement': CORRELATION_VALIDATION_THRESHOLD
            }
        
        # Validate reproducibility threshold requirements
        if 'reproducibility_threshold' in config:
            reproducibility_threshold = config['reproducibility_threshold']
            if reproducibility_threshold < REPRODUCIBILITY_VALIDATION_THRESHOLD:
                validation_results['overall_valid'] = False
                validation_results['validation_issues'].append(
                    f"Reproducibility threshold {reproducibility_threshold} below minimum {REPRODUCIBILITY_VALIDATION_THRESHOLD}"
                )
            
            validation_results['compliance_status']['reproducibility_threshold'] = {
                'valid': reproducibility_threshold >= REPRODUCIBILITY_VALIDATION_THRESHOLD,
                'value': reproducibility_threshold,
                'requirement': REPRODUCIBILITY_VALIDATION_THRESHOLD
            }
        
        # Apply strict validation if strict_validation is enabled
        if strict_validation:
            # Validate numerical precision requirements
            if 'numerical_precision_threshold' in config:
                precision_threshold = config['numerical_precision_threshold']
                if precision_threshold > 1e-3:  # Stricter requirement for scientific computing
                    validation_results['validation_issues'].append(
                        f"Numerical precision threshold {precision_threshold} may be too loose for scientific computing"
                    )
            
            # Validate performance tracking configuration
            if not config.get('enable_real_time_monitoring', False):
                validation_results['configuration_recommendations'].append(
                    "Consider enabling real-time monitoring for better performance tracking"
                )
            
            # Validate audit trail configuration
            if not config.get('audit_trail_enabled', False):
                validation_results['configuration_recommendations'].append(
                    "Consider enabling audit trail for scientific reproducibility"
                )
        
        # Generate configuration recommendations if generate_recommendations is enabled
        if generate_recommendations:
            # Performance optimization recommendations
            if not config.get('enable_caching', True):
                validation_results['configuration_recommendations'].append(
                    "Enable caching for improved performance in large-scale analysis"
                )
            
            # Scientific reproducibility recommendations
            if not config.get('scientific_documentation_enabled', True):
                validation_results['configuration_recommendations'].append(
                    "Enable scientific documentation for better research reproducibility"
                )
            
            # Cross-platform compatibility recommendations
            if not config.get('cross_format_validation', True):
                validation_results['configuration_recommendations'].append(
                    "Enable cross-format validation for Crimaldi and custom plume compatibility"
                )
        
        # Assess performance implications of configuration
        validation_results['performance_assessment'] = {
            'caching_enabled': config.get('enable_caching', False),
            'real_time_monitoring': config.get('enable_real_time_monitoring', False),
            'statistical_validation_overhead': config.get('enable_statistical_validation', False),
            'expected_performance_impact': 'low' if config.get('enable_caching', False) else 'medium'
        }
        
        return validation_results
        
    except Exception as e:
        validation_results['overall_valid'] = False
        validation_results['validation_issues'].append(f"Configuration validation error: {e}")
        return validation_results


def get_analysis_component(
    component_name: str,
    configuration: dict = None,
    enable_validation: bool = True
) -> object:
    """
    Get analysis component instance with configuration and validation for modular analysis access.
    
    This function provides factory-pattern access to analysis components with proper configuration,
    validation, and scientific context setup for consistent analysis operations.
    
    Args:
        component_name: Name of the analysis component to retrieve
        configuration: Component-specific configuration parameters
        enable_validation: Enable component validation and verification
        
    Returns:
        object: Configured analysis component instance with validation and context setup
    """
    # Validate component name against available components
    component_mapping = {
        'performance_metrics': PerformanceMetricsCalculator,
        'navigation_success': NavigationSuccessAnalyzer,
        'statistical_comparison': StatisticalComparator,
        'algorithm_ranking': AlgorithmRankingAnalyzer,
        'trajectory_analysis': TrajectoryAnalyzer,
        'movement_pattern': MovementPatternClassifier,
        'success_rate': SuccessRateCalculator,
        'localization_success': LocalizationSuccessAnalyzer,
        'path_efficiency': PathEfficiencyAnalyzer,
        'optimal_path': OptimalPathCalculator,
        'temporal_dynamics': TemporalDynamicsAnalyzer,
        'response_time': ResponseTimeAnalyzer,
        'robustness_analysis': RobustnessAnalyzer,
        'noise_tolerance': NoiseToleranceAnalyzer,
        'scientific_visualization': ScientificVisualizer,
        'trajectory_plotting': TrajectoryPlotter,
        'report_generation': ReportGenerator
    }
    
    if component_name not in component_mapping:
        raise ValueError(f"Unknown analysis component: {component_name}. Available: {list(component_mapping.keys())}")
    
    # Get component class and create instance with configuration
    component_class = component_mapping[component_name]
    
    try:
        # Create component instance with configuration if provided
        if configuration:
            component_instance = component_class(**configuration)
        else:
            component_instance = component_class()
        
        # Apply validation if enable_validation is True
        if enable_validation:
            # Verify component instance has required methods and attributes
            required_methods = ['validate', 'get_version'] if hasattr(component_class, 'validate') else []
            for method in required_methods:
                if not hasattr(component_instance, method):
                    raise AttributeError(f"Component {component_name} missing required method: {method}")
        
        # Set up logging context for component usage
        component_logger = get_logger(f'analysis.{component_name}', 'ANALYSIS')
        component_logger.debug(f"Created analysis component: {component_name}")
        
        return component_instance
        
    except Exception as e:
        error_logger = get_logger('analysis.component.error', 'ANALYSIS')
        error_logger.error(f"Failed to create analysis component {component_name}: {e}", exc_info=True)
        raise


# =============================================================================
# COMPREHENSIVE ANALYSIS MODULE EXPORTS
# =============================================================================

# Export all analysis components for comprehensive public API access
__all__ = [
    # Module metadata and configuration
    'ANALYSIS_MODULE_VERSION',
    'DEFAULT_ANALYSIS_CONFIG',
    'SUPPORTED_ANALYSIS_TYPES',
    'ANALYSIS_VALIDATION_ENABLED',
    'CROSS_ALGORITHM_COMPARISON_ENABLED',
    'SCIENTIFIC_REPRODUCIBILITY_ENABLED',
    'CORRELATION_VALIDATION_THRESHOLD',
    'REPRODUCIBILITY_VALIDATION_THRESHOLD',
    
    # Performance metrics analysis components
    'PerformanceMetricsCalculator',
    'NavigationSuccessAnalyzer',
    'PathEfficiencyAnalyzer',
    'calculate_navigation_success_metrics',
    'calculate_path_efficiency_metrics',
    
    # Statistical comparison and validation components
    'StatisticalComparator',
    'AlgorithmRankingAnalyzer',
    'compare_algorithm_performance',
    'validate_simulation_reproducibility',
    
    # Trajectory analysis and pattern recognition components
    'TrajectoryAnalyzer',
    'MovementPatternClassifier',
    'calculate_trajectory_similarity_matrix',
    'extract_trajectory_features',
    
    # Success rate calculation and monitoring components
    'SuccessRateCalculator',
    'LocalizationSuccessAnalyzer',
    'calculate_overall_success_rate',
    
    # Path efficiency analysis and optimization components
    'PathEfficiencyAnalyzer',
    'OptimalPathCalculator',
    'calculate_path_length_ratio',
    
    # Temporal dynamics analysis components
    'TemporalDynamicsAnalyzer',
    'ResponseTimeAnalyzer',
    'calculate_response_times',
    
    # Robustness analysis and environmental adaptation components
    'RobustnessAnalyzer',
    'NoiseToleranceAnalyzer',
    'analyze_noise_tolerance',
    
    # Scientific visualization and plotting components
    'ScientificVisualizer',
    'TrajectoryPlotter',
    'create_trajectory_plot',
    'create_performance_chart',
    
    # Report generation and documentation components
    'ReportGenerator',
    'GeneratedReport',
    'generate_simulation_report',
    'generate_batch_report',
    
    # Module management and utility functions
    'initialize_analysis_module',
    'validate_analysis_configuration',
    'get_analysis_component'
]

# =============================================================================
# AUTOMATIC MODULE INITIALIZATION
# =============================================================================

# Initialize analysis module with default configuration on import
try:
    # Perform automatic initialization with basic configuration
    _initialization_success = initialize_analysis_module(
        config=DEFAULT_ANALYSIS_CONFIG,
        enable_logging=True,
        enable_performance_tracking=True,
        validate_initialization=True
    )
    
    if not _initialization_success:
        import warnings
        warnings.warn(
            "Analysis module initialization failed. Some functionality may be limited.",
            RuntimeWarning,
            stacklevel=2
        )

except Exception as init_error:
    import warnings
    warnings.warn(
        f"Analysis module initialization error: {init_error}. Manual initialization may be required.",
        RuntimeWarning,
        stacklevel=2
    )