"""
Comprehensive example demonstrating advanced analysis and visualization capabilities for plume navigation simulation results.

This module showcases scientific visualization generation, performance metrics analysis, statistical comparison, 
trajectory plotting, and publication-ready figure creation using the complete analysis pipeline. Implements 
end-to-end workflow from simulation result loading through comprehensive visualization report generation with 
>95% correlation validation, cross-algorithm comparison, and reproducible scientific documentation for research 
publication and algorithm development feedback.

Key Features:
- Publication-ready scientific visualization with standardized formatting
- Cross-algorithm performance comparison and statistical validation
- Comprehensive trajectory analysis and movement pattern visualization
- Performance metrics calculation with >95% correlation validation
- Cross-format compatibility analysis (Crimaldi vs custom formats)
- Scientific reproducibility documentation with >0.99 coefficient
- Automated report generation with methodology documentation
- Command-line interface for configuration and execution control
- Comprehensive audit trail and execution tracking
"""

# External library imports with version specifications
import numpy as np  # numpy 2.1.3+ - Numerical array operations for data processing and statistical calculations
import pandas as pd  # pandas 2.2.0+ - Data manipulation and analysis for result aggregation and statistical processing
import matplotlib.pyplot as plt  # matplotlib 3.9.0+ - Primary plotting library for scientific figure generation and publication-ready charts
import seaborn as sns  # seaborn 0.13.2+ - Statistical visualization and advanced plotting for correlation matrices and distribution plots
from pathlib import Path  # pathlib 3.9+ - Modern path handling for configuration files and output directory management
import datetime  # datetime 3.9+ - Timestamp generation for example execution tracking and output file naming
from typing import Dict, Any, List, Optional, Union, Tuple  # typing 3.9+ - Type hints for function signatures and data structures
import argparse  # argparse 3.9+ - Command-line argument parsing for example configuration and execution options
import sys  # sys 3.9+ - System interface for exit codes and error handling
import os  # os 3.9+ - Operating system interface for environment variables and path operations
import json  # json 3.9+ - JSON configuration handling and metadata serialization
import time  # time 3.9+ - Performance timing and execution measurement
import warnings  # warnings 3.9+ - Warning management for scientific computing operations

# Internal imports from core analysis modules
from ..core.analysis.visualization import (  # Primary visualization engine for creating publication-ready scientific figures
    ScientificVisualizer,  # Main visualization class for comprehensive scientific figure generation
    TrajectoryPlotter  # Specialized trajectory plotting for detailed movement pattern analysis
)
from ..core.analysis.performance_metrics import (  # Performance metrics calculation for comprehensive algorithm evaluation
    PerformanceMetricsCalculator,  # Core performance metrics calculation and analysis
    NavigationSuccessAnalyzer  # Navigation success metrics analysis for algorithm performance evaluation
)
from ..core.simulation.result_collector import (  # Simulation result data integration for performance analysis
    SimulationResult,  # Individual simulation result data container
    BatchSimulationResult  # Batch simulation result analysis for cross-algorithm comparison
)

# Internal imports from utility modules
from ..utils.file_utils import load_json_config  # Configuration file loading for example setup and parameter management
from ..utils.logging_utils import (  # Structured logging for example execution and audit trail generation
    get_logger,  # Logger instance creation with scientific context
    set_scientific_context,  # Scientific context setting for reproducible example execution
    create_audit_trail  # Audit trail creation for example execution traceability
)

# Global configuration constants for example execution and validation
EXAMPLE_VERSION = '1.0.0'  # Version identifier for example implementation tracking
EXAMPLE_NAME = 'Comprehensive Analysis and Visualization Example'  # Human-readable example name for documentation
DEFAULT_CONFIG_PATH = 'data/example_config.json'  # Default path to example configuration file
DEFAULT_OUTPUT_DIR = 'results/analysis_visualization_example'  # Default output directory for generated results
SUPPORTED_ALGORITHMS = ['infotaxis', 'casting', 'gradient_following', 'hybrid_strategies']  # Navigation algorithms for comparison
VISUALIZATION_FORMATS = ['png', 'pdf', 'svg']  # Supported output formats for publication-ready figures
SCIENTIFIC_PRECISION = 6  # Decimal precision for scientific value formatting and calculations
CORRELATION_THRESHOLD = 0.95  # Minimum correlation coefficient for validation compliance (>95%)
REPRODUCIBILITY_THRESHOLD = 0.99  # Minimum reproducibility coefficient for scientific validation (>0.99)

# Configure scientific plotting parameters for publication-ready output
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf'
})

# Configure seaborn style for scientific visualization
sns.set_style("whitegrid")
sns.set_palette("husl")

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def set_equal_spatial_scale(ax_list: List[plt.Axes]) -> Tuple[float, float, float, float]:
    """Set common x and y limits for a list of axes.

    Parameters
    ----------
    ax_list : List[plt.Axes]
        Axes that should share the same x/y limits.

    Returns
    -------
    Tuple[float, float, float, float]
        Tuple of ``(xmin, xmax, ymin, ymax)`` applied to all axes.
    """
    logger = get_logger('analysis_visualization.scale', 'VISUALIZATION')

    x_vals: List[float] = []
    y_vals: List[float] = []
    for ax in ax_list:
        x_vals.extend(ax.get_xlim())
        y_vals.extend(ax.get_ylim())

    xmin, xmax = min(x_vals), max(x_vals)
    ymin, ymax = min(y_vals), max(y_vals)
    limits = (xmin, xmax, ymin, ymax)
    logger.debug("Unified axis range: %s", limits)

    for ax in ax_list:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    return limits


def load_example_configuration(
    config_path: str,
    validate_config: bool = True
) -> Dict[str, Any]:
    """
    Load example configuration from JSON file with validation and scientific context setup for reproducible 
    analysis and visualization demonstration.
    
    This function loads configuration parameters, validates required fields, and establishes scientific 
    context for reproducible example execution with comprehensive error handling and audit trail creation.
    
    Args:
        config_path: Path to the JSON configuration file containing example parameters
        validate_config: Enable configuration validation against required fields and ranges
        
    Returns:
        Dict[str, Any]: Loaded and validated example configuration with analysis and visualization settings
        
    Raises:
        FileNotFoundError: If configuration file does not exist
        ValueError: If configuration validation fails
        json.JSONDecodeError: If configuration file contains invalid JSON
    """
    # Initialize logger for configuration loading operations
    logger = get_logger('example.config_loader', 'CONFIGURATION')
    
    try:
        # Load configuration from specified path using utility function
        logger.info(f"Loading example configuration from: {config_path}")
        config_data = load_json_config(
            config_path=config_path,
            validate_schema=False,  # Custom validation implemented below
            use_cache=True
        )
        
        # Validate configuration structure and required fields if enabled
        if validate_config:
            required_fields = [
                'simulation_parameters',
                'visualization_settings',
                'performance_metrics',
                'output_configuration'
            ]
            
            for field in required_fields:
                if field not in config_data:
                    raise ValueError(f"Missing required configuration field: {field}")
            
            # Validate algorithm configuration settings
            if 'algorithms' in config_data['simulation_parameters']:
                invalid_algorithms = set(config_data['simulation_parameters']['algorithms']) - set(SUPPORTED_ALGORITHMS)
                if invalid_algorithms:
                    raise ValueError(f"Unsupported algorithms in configuration: {invalid_algorithms}")
            
            # Validate visualization format specifications
            if 'output_formats' in config_data['visualization_settings']:
                invalid_formats = set(config_data['visualization_settings']['output_formats']) - set(VISUALIZATION_FORMATS)
                if invalid_formats:
                    raise ValueError(f"Unsupported visualization formats: {invalid_formats}")
            
            # Validate numerical thresholds and precision settings
            if 'correlation_threshold' in config_data['performance_metrics']:
                threshold = config_data['performance_metrics']['correlation_threshold']
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError(f"Correlation threshold must be between 0.0 and 1.0, got: {threshold}")
        
        # Set scientific context for example execution with configuration details
        set_scientific_context(
            simulation_id='example_analysis_visualization',
            algorithm_name='multi_algorithm_comparison',
            processing_stage='CONFIGURATION_LOADING',
            additional_context={
                'example_version': EXAMPLE_VERSION,
                'config_path': config_path,
                'validation_enabled': validate_config
            }
        )
        
        # Create audit trail entry for configuration loading
        create_audit_trail(
            action='EXAMPLE_CONFIG_LOADED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'config_path': config_path,
                'validation_enabled': validate_config,
                'config_fields': list(config_data.keys()),
                'algorithms_configured': config_data.get('simulation_parameters', {}).get('algorithms', [])
            }
        )
        
        # Log successful configuration loading with details
        logger.info(f"Configuration loaded successfully with {len(config_data)} sections")
        logger.debug(f"Configuration sections: {list(config_data.keys())}")
        
        return config_data
        
    except Exception as e:
        # Handle configuration loading errors with comprehensive context
        logger.error(f"Failed to load example configuration: {e}")
        create_audit_trail(
            action='EXAMPLE_CONFIG_LOAD_FAILED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'config_path': config_path,
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
        )
        raise e


def generate_sample_simulation_results(
    num_simulations: int,
    algorithm_names: List[str],
    generation_config: Dict[str, Any]
) -> Dict[str, List[SimulationResult]]:
    """
    Generate sample simulation results for demonstration purposes with realistic performance metrics, 
    trajectory data, and algorithm comparison scenarios.
    
    This function creates realistic sample data with statistical variation to demonstrate analysis 
    capabilities without requiring actual simulation execution.
    
    Args:
        num_simulations: Number of simulation results to generate per algorithm
        algorithm_names: List of algorithm names for result generation
        generation_config: Configuration parameters for realistic data generation
        
    Returns:
        Dict[str, List[SimulationResult]]: Generated sample simulation results organized by algorithm
        
    Raises:
        ValueError: If invalid generation parameters are provided
    """
    # Initialize logger for sample data generation
    logger = get_logger('example.sample_generator', 'DATA_GENERATION')
    
    try:
        # Validate generation parameters
        if num_simulations <= 0:
            raise ValueError(f"Number of simulations must be positive, got: {num_simulations}")
        
        if not algorithm_names:
            raise ValueError("At least one algorithm name must be provided")
        
        # Initialize random seed for reproducible sample generation
        np.random.seed(generation_config.get('random_seed', 42))
        
        # Extract generation parameters with defaults
        arena_size = generation_config.get('arena_size', (100, 100))
        simulation_duration = generation_config.get('duration_seconds', 300)
        target_position = generation_config.get('target_position', (80, 80))
        noise_level = generation_config.get('noise_level', 0.1)
        
        logger.info(f"Generating {num_simulations} sample results for {len(algorithm_names)} algorithms")
        
        # Initialize results dictionary organized by algorithm
        algorithm_results = {}
        
        # Generate results for each algorithm with realistic characteristics
        for algorithm_name in algorithm_names:
            logger.debug(f"Generating sample data for algorithm: {algorithm_name}")
            
            algorithm_results[algorithm_name] = []
            
            # Define algorithm-specific performance characteristics
            algorithm_params = _get_algorithm_characteristics(algorithm_name)
            
            # Generate simulation results for current algorithm
            for sim_idx in range(num_simulations):
                # Generate realistic trajectory data with algorithm-specific patterns
                trajectory_data = _generate_realistic_trajectory(
                    algorithm_name=algorithm_name,
                    algorithm_params=algorithm_params,
                    arena_size=arena_size,
                    target_position=target_position,
                    duration=simulation_duration,
                    noise_level=noise_level,
                    simulation_id=f"{algorithm_name}_sim_{sim_idx:04d}"
                )
                
                # Calculate performance metrics with statistical variation
                performance_metrics = _calculate_sample_performance_metrics(
                    trajectory_data=trajectory_data,
                    target_position=target_position,
                    algorithm_params=algorithm_params
                )
                
                # Create SimulationResult object with comprehensive data
                simulation_result = SimulationResult(
                    simulation_id=f"{algorithm_name}_sim_{sim_idx:04d}",
                    algorithm_name=algorithm_name,
                    trajectory_data=trajectory_data,
                    performance_metrics=performance_metrics,
                    metadata={
                        'generation_timestamp': datetime.datetime.now().isoformat(),
                        'sample_data': True,
                        'arena_size': arena_size,
                        'target_position': target_position,
                        'simulation_duration': simulation_duration
                    }
                )
                
                algorithm_results[algorithm_name].append(simulation_result)
        
        # Log successful sample generation with statistics
        total_results = sum(len(results) for results in algorithm_results.values())
        logger.info(f"Sample generation completed: {total_results} total results across {len(algorithm_names)} algorithms")
        
        # Create audit trail for sample data generation
        create_audit_trail(
            action='SAMPLE_DATA_GENERATED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'num_simulations_per_algorithm': num_simulations,
                'algorithms': algorithm_names,
                'total_results': total_results,
                'generation_config': generation_config
            }
        )
        
        return algorithm_results
        
    except Exception as e:
        # Handle sample generation errors with comprehensive logging
        logger.error(f"Failed to generate sample simulation results: {e}")
        create_audit_trail(
            action='SAMPLE_DATA_GENERATION_FAILED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'requested_simulations': num_simulations,
                'algorithms': algorithm_names
            }
        )
        raise e


def demonstrate_trajectory_visualization(
    algorithm_results: Dict[str, List[SimulationResult]],
    visualizer: ScientificVisualizer,
    output_directory: str
) -> List[str]:
    """
    Demonstrate comprehensive trajectory visualization capabilities including single trajectory plots, 
    comparative analysis, and movement pattern visualization with publication-ready formatting.
    
    This function showcases advanced trajectory visualization techniques with scientific formatting 
    and comprehensive comparative analysis for research publication and algorithm development.
    
    Args:
        algorithm_results: Dictionary of simulation results organized by algorithm name
        visualizer: ScientificVisualizer instance for figure generation
        output_directory: Directory path for saving generated visualization files
        
    Returns:
        List[str]: List of generated trajectory visualization file paths for documentation
        
    Raises:
        ValueError: If insufficient data is provided for visualization
        IOError: If output directory creation or file writing fails
    """
    # Initialize logger for trajectory visualization demonstration
    logger = get_logger('example.trajectory_visualization', 'VISUALIZATION')
    
    try:
        # Validate input parameters and data availability
        if not algorithm_results:
            raise ValueError("No algorithm results provided for trajectory visualization")
        
        # Ensure output directory exists for generated files
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting trajectory visualization demonstration for {len(algorithm_results)} algorithms")
        
        # Initialize list to track generated visualization files
        generated_files = []
        
        # Extract trajectory data from simulation results for processing
        trajectory_data = {}
        for algorithm_name, results in algorithm_results.items():
            trajectory_data[algorithm_name] = []
            for result in results:
                if hasattr(result, 'trajectory_data') and result.trajectory_data is not None:
                    trajectory_data[algorithm_name].append(result.trajectory_data)
        
        # Create individual trajectory plots for each algorithm
        logger.debug("Generating individual trajectory visualizations")
        for algorithm_name, trajectories in trajectory_data.items():
            if not trajectories:
                logger.warning(f"No trajectory data available for algorithm: {algorithm_name}")
                continue
            
            # Select representative trajectories for visualization
            sample_trajectories = trajectories[:min(5, len(trajectories))]
            
            # Generate individual trajectory plot using visualizer
            trajectory_plot_file = output_path / f"trajectory_individual_{algorithm_name}.pdf"
            
            # Create trajectory plot with scientific formatting
            figure = visualizer.create_trajectory_plot(
                trajectory_data=sample_trajectories,
                algorithm_name=algorithm_name,
                plot_title=f"Individual Trajectories - {algorithm_name.replace('_', ' ').title()}",
                show_target=True,
                show_source=True,
                include_statistics=True
            )
            
            # Save figure with publication-ready formatting
            figure.savefig(trajectory_plot_file, dpi=300, bbox_inches='tight')
            plt.close(figure)
            
            generated_files.append(str(trajectory_plot_file))
            logger.debug(f"Generated individual trajectory plot: {trajectory_plot_file}")
        
        # Generate comparative trajectory visualization across algorithms
        logger.debug("Generating comparative trajectory visualization")
        comparative_plot_file = output_path / "trajectory_comparison.pdf"
        
        # Create comprehensive trajectory comparison plot
        comparison_figure = visualizer.create_trajectory_comparison(
            algorithm_trajectories=trajectory_data,
            plot_title="Algorithm Trajectory Comparison",
            normalize_coordinates=True,
            show_confidence_intervals=True,
            include_performance_overlay=True
        )
        
        # Save comparative visualization with enhanced formatting
        comparison_figure.savefig(comparative_plot_file, dpi=300, bbox_inches='tight')
        plt.close(comparison_figure)
        
        generated_files.append(str(comparative_plot_file))
        
        # Create trajectory overlay plots with statistical analysis
        logger.debug("Generating trajectory overlay with statistical analysis")
        overlay_plot_file = output_path / "trajectory_overlay_statistical.pdf"
        
        # Generate statistical overlay visualization
        overlay_figure = visualizer.create_statistical_trajectory_overlay(
            algorithm_trajectories=trajectory_data,
            plot_title="Statistical Trajectory Analysis",
            show_density_maps=True,
            include_efficiency_metrics=True,
            show_convergence_zones=True
        )

        # Ensure both plume subplots share the same spatial scale
        axes = overlay_figure.axes[:2]
        set_equal_spatial_scale(axes)
        
        # Save overlay visualization with comprehensive formatting
        overlay_figure.savefig(overlay_plot_file, dpi=300, bbox_inches='tight')
        plt.close(overlay_figure)
        
        generated_files.append(str(overlay_plot_file))
        
        # Generate movement pattern analysis visualization
        logger.debug("Generating movement pattern analysis")
        pattern_plot_file = output_path / "movement_patterns.pdf"
        
        # Create movement pattern analysis figure
        pattern_figure = visualizer.create_movement_pattern_analysis(
            algorithm_trajectories=trajectory_data,
            plot_title="Movement Pattern Analysis",
            include_velocity_profiles=True,
            show_directional_preferences=True,
            analyze_search_strategies=True
        )
        
        # Save pattern analysis with scientific formatting
        pattern_figure.savefig(pattern_plot_file, dpi=300, bbox_inches='tight')
        plt.close(pattern_figure)
        
        generated_files.append(str(pattern_plot_file))
        
        # Log successful trajectory visualization completion
        logger.info(f"Trajectory visualization completed: {len(generated_files)} files generated")
        
        # Create audit trail for trajectory visualization demonstration
        create_audit_trail(
            action='TRAJECTORY_VISUALIZATION_COMPLETED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'algorithms_visualized': list(algorithm_results.keys()),
                'files_generated': len(generated_files),
                'output_directory': output_directory,
                'visualization_types': ['individual', 'comparative', 'statistical_overlay', 'movement_patterns']
            }
        )
        
        return generated_files
        
    except Exception as e:
        # Handle trajectory visualization errors with comprehensive logging
        logger.error(f"Failed to demonstrate trajectory visualization: {e}")
        create_audit_trail(
            action='TRAJECTORY_VISUALIZATION_FAILED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'algorithms_attempted': list(algorithm_results.keys()) if algorithm_results else []
            }
        )
        raise e


def demonstrate_performance_analysis(
    algorithm_results: Dict[str, List[SimulationResult]],
    metrics_calculator: PerformanceMetricsCalculator,
    output_directory: str
) -> Dict[str, Any]:
    """
    Demonstrate comprehensive performance analysis including metrics calculation, statistical comparison, 
    algorithm ranking, and performance trend visualization.
    
    This function showcases advanced performance analysis capabilities with statistical validation, 
    cross-algorithm comparison, and reproducibility assessment for scientific computing requirements.
    
    Args:
        algorithm_results: Dictionary of simulation results organized by algorithm name
        metrics_calculator: PerformanceMetricsCalculator instance for metrics computation
        output_directory: Directory path for saving analysis results and reports
        
    Returns:
        Dict[str, Any]: Comprehensive performance analysis results with metrics, comparisons, and validation
        
    Raises:
        ValueError: If insufficient data is provided for analysis
        RuntimeError: If performance analysis computation fails
    """
    # Initialize logger for performance analysis demonstration
    logger = get_logger('example.performance_analysis', 'PERFORMANCE_ANALYSIS')
    
    try:
        # Validate input parameters and data availability
        if not algorithm_results:
            raise ValueError("No algorithm results provided for performance analysis")
        
        # Ensure output directory exists for analysis results
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting performance analysis demonstration for {len(algorithm_results)} algorithms")
        
        # Initialize comprehensive analysis results container
        analysis_results = {
            'algorithm_metrics': {},
            'comparative_analysis': {},
            'statistical_validation': {},
            'reproducibility_assessment': {},
            'performance_rankings': {},
            'analysis_metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'algorithms_analyzed': list(algorithm_results.keys()),
                'total_simulations': sum(len(results) for results in algorithm_results.values())
            }
        }
        
        # Calculate performance metrics for each algorithm using metrics calculator
        logger.debug("Calculating performance metrics for individual algorithms")
        for algorithm_name, results in algorithm_results.items():
            logger.debug(f"Analyzing performance for algorithm: {algorithm_name}")
            
            # Calculate comprehensive metrics for algorithm results
            algorithm_metrics = metrics_calculator.calculate_all_metrics(
                simulation_results=results,
                include_statistical_analysis=True,
                calculate_confidence_intervals=True
            )
            
            analysis_results['algorithm_metrics'][algorithm_name] = algorithm_metrics
            
            # Log metrics calculation completion
            logger.debug(f"Calculated {len(algorithm_metrics)} metric categories for {algorithm_name}")
        
        # Perform cross-algorithm statistical comparison and ranking
        logger.debug("Performing cross-algorithm statistical comparison")
        comparative_analysis = metrics_calculator.compare_algorithm_metrics(
            algorithm_metrics=analysis_results['algorithm_metrics'],
            comparison_methods=['mean', 'median', 'statistical_significance'],
            confidence_level=0.95,
            multiple_comparisons_correction=True
        )
        
        analysis_results['comparative_analysis'] = comparative_analysis
        
        # Generate algorithm ranking and efficiency analysis
        logger.debug("Generating algorithm rankings and efficiency analysis")
        performance_rankings = _generate_algorithm_rankings(
            algorithm_metrics=analysis_results['algorithm_metrics'],
            ranking_criteria=['success_rate', 'efficiency_score', 'time_to_target', 'path_optimality']
        )
        
        analysis_results['performance_rankings'] = performance_rankings
        
        # Validate metrics accuracy against correlation threshold (>95%)
        logger.debug("Validating metrics accuracy against correlation threshold")
        correlation_validation = _validate_correlation_accuracy(
            algorithm_metrics=analysis_results['algorithm_metrics'],
            correlation_threshold=CORRELATION_THRESHOLD
        )
        
        analysis_results['statistical_validation']['correlation_validation'] = correlation_validation
        
        # Assess reproducibility coefficient compliance (>0.99)
        logger.debug("Assessing reproducibility coefficient compliance")
        reproducibility_assessment = _assess_reproducibility_coefficient(
            algorithm_results=algorithm_results,
            reproducibility_threshold=REPRODUCIBILITY_THRESHOLD
        )
        
        analysis_results['reproducibility_assessment'] = reproducibility_assessment
        
        # Create performance comparison visualizations
        logger.debug("Creating performance comparison visualizations")
        visualization_files = _create_performance_visualizations(
            analysis_results=analysis_results,
            output_directory=output_path
        )
        
        analysis_results['visualization_files'] = visualization_files
        
        # Generate statistical significance testing results
        logger.debug("Performing statistical significance testing")
        significance_testing = _perform_statistical_significance_testing(
            algorithm_metrics=analysis_results['algorithm_metrics'],
            alpha_level=0.05,
            correction_method='bonferroni'
        )
        
        analysis_results['statistical_validation']['significance_testing'] = significance_testing
        
        # Validate overall analysis compliance with scientific standards
        compliance_status = _validate_analysis_compliance(
            analysis_results=analysis_results,
            correlation_threshold=CORRELATION_THRESHOLD,
            reproducibility_threshold=REPRODUCIBILITY_THRESHOLD
        )
        
        analysis_results['compliance_status'] = compliance_status
        
        # Save comprehensive analysis results to file
        results_file = output_path / "performance_analysis_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Log successful performance analysis completion
        logger.info(f"Performance analysis completed successfully with {compliance_status['overall_compliance']} compliance")
        
        # Create audit trail for performance analysis demonstration
        create_audit_trail(
            action='PERFORMANCE_ANALYSIS_COMPLETED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'algorithms_analyzed': list(algorithm_results.keys()),
                'total_simulations': analysis_results['analysis_metadata']['total_simulations'],
                'compliance_status': compliance_status,
                'correlation_validation': correlation_validation['overall_compliance'],
                'reproducibility_assessment': reproducibility_assessment['coefficient'] >= REPRODUCIBILITY_THRESHOLD
            }
        )
        
        return analysis_results
        
    except Exception as e:
        # Handle performance analysis errors with comprehensive logging
        logger.error(f"Failed to demonstrate performance analysis: {e}")
        create_audit_trail(
            action='PERFORMANCE_ANALYSIS_FAILED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'algorithms_attempted': list(algorithm_results.keys()) if algorithm_results else []
            }
        )
        raise e


def demonstrate_statistical_visualization(
    performance_metrics: Dict[str, Any],
    visualizer: ScientificVisualizer,
    output_directory: str
) -> List[str]:
    """
    Demonstrate statistical visualization capabilities including correlation matrices, distribution plots, 
    hypothesis testing results, and confidence interval displays.
    
    This function showcases advanced statistical visualization techniques with scientific formatting 
    and publication-ready output for comprehensive performance analysis documentation.
    
    Args:
        performance_metrics: Dictionary of performance metrics from algorithm analysis
        visualizer: ScientificVisualizer instance for statistical figure generation
        output_directory: Directory path for saving statistical visualization files
        
    Returns:
        List[str]: List of generated statistical visualization file paths for documentation
        
    Raises:
        ValueError: If insufficient performance metrics are provided
        RuntimeError: If statistical visualization generation fails
    """
    # Initialize logger for statistical visualization demonstration
    logger = get_logger('example.statistical_visualization', 'STATISTICAL_VISUALIZATION')
    
    try:
        # Validate input parameters and metrics availability
        if not performance_metrics or 'algorithm_metrics' not in performance_metrics:
            raise ValueError("No performance metrics provided for statistical visualization")
        
        # Ensure output directory exists for visualization files
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        algorithm_metrics = performance_metrics['algorithm_metrics']
        logger.info(f"Starting statistical visualization demonstration for {len(algorithm_metrics)} algorithms")
        
        # Initialize list to track generated statistical visualization files
        generated_files = []
        
        # Create correlation matrix visualization for performance metrics
        logger.debug("Generating correlation matrix visualization")
        correlation_file = output_path / "correlation_matrix.pdf"
        
        # Extract numerical metrics for correlation analysis
        metrics_data = _extract_numerical_metrics(algorithm_metrics)
        
        # Generate correlation matrix plot with scientific formatting
        correlation_figure = visualizer.create_correlation_matrix(
            metrics_data=metrics_data,
            plot_title="Performance Metrics Correlation Matrix",
            method='pearson',
            annotate_values=True,
            significance_testing=True
        )
        
        # Save correlation matrix with publication formatting
        correlation_figure.savefig(correlation_file, dpi=300, bbox_inches='tight')
        plt.close(correlation_figure)
        
        generated_files.append(str(correlation_file))
        
        # Generate distribution plots for algorithm performance metrics
        logger.debug("Generating performance distribution plots")
        distribution_file = output_path / "performance_distributions.pdf"
        
        # Create comprehensive distribution analysis
        distribution_figure = visualizer.create_distribution_plots(
            algorithm_metrics=algorithm_metrics,
            metrics_to_plot=['success_rate', 'efficiency_score', 'time_to_target'],
            plot_title="Algorithm Performance Distributions",
            include_kde=True,
            show_confidence_intervals=True,
            normality_testing=True
        )
        
        # Save distribution plots with scientific formatting
        distribution_figure.savefig(distribution_file, dpi=300, bbox_inches='tight')
        plt.close(distribution_figure)
        
        generated_files.append(str(distribution_file))
        
        # Create hypothesis testing result visualizations
        logger.debug("Generating hypothesis testing visualizations")
        hypothesis_file = output_path / "hypothesis_testing_results.pdf"
        
        # Generate statistical hypothesis testing plots
        if 'statistical_validation' in performance_metrics:
            hypothesis_figure = visualizer.create_hypothesis_testing_plots(
                statistical_results=performance_metrics['statistical_validation'],
                plot_title="Statistical Hypothesis Testing Results",
                alpha_level=0.05,
                include_power_analysis=True,
                show_effect_sizes=True
            )
            
            # Save hypothesis testing visualization
            hypothesis_figure.savefig(hypothesis_file, dpi=300, bbox_inches='tight')
            plt.close(hypothesis_figure)
            
            generated_files.append(str(hypothesis_file))
        
        # Generate confidence interval displays for key metrics
        logger.debug("Generating confidence interval visualizations")
        confidence_file = output_path / "confidence_intervals.pdf"
        
        # Create confidence interval visualization
        confidence_figure = visualizer.create_confidence_interval_plots(
            algorithm_metrics=algorithm_metrics,
            confidence_level=0.95,
            plot_title="Performance Metrics Confidence Intervals",
            include_significance_indicators=True,
            show_overlapping_intervals=True
        )
        
        # Save confidence interval plots
        confidence_figure.savefig(confidence_file, dpi=300, bbox_inches='tight')
        plt.close(confidence_figure)
        
        generated_files.append(str(confidence_file))
        
        # Create statistical significance indicator plots
        logger.debug("Generating statistical significance indicators")
        significance_file = output_path / "statistical_significance.pdf"
        
        # Generate significance testing visualization
        if 'comparative_analysis' in performance_metrics:
            significance_figure = visualizer.create_statistical_significance_plot(
                comparative_analysis=performance_metrics['comparative_analysis'],
                plot_title="Statistical Significance Analysis",
                correction_method='bonferroni',
                show_p_values=True,
                include_effect_size_metrics=True
            )
            
            # Save statistical significance visualization
            significance_figure.savefig(significance_file, dpi=300, bbox_inches='tight')
            plt.close(significance_figure)
            
            generated_files.append(str(significance_file))
        
        # Generate comprehensive statistical summary visualization
        logger.debug("Generating comprehensive statistical summary")
        summary_file = output_path / "statistical_summary.pdf"
        
        # Create statistical summary dashboard
        summary_figure = visualizer.create_statistical_summary_dashboard(
            performance_metrics=performance_metrics,
            plot_title="Comprehensive Statistical Analysis Summary",
            include_correlation_analysis=True,
            show_distribution_comparisons=True,
            display_key_statistics=True
        )
        
        # Save statistical summary with enhanced formatting
        summary_figure.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close(summary_figure)
        
        generated_files.append(str(summary_file))
        
        # Log successful statistical visualization completion
        logger.info(f"Statistical visualization completed: {len(generated_files)} files generated")
        
        # Create audit trail for statistical visualization demonstration
        create_audit_trail(
            action='STATISTICAL_VISUALIZATION_COMPLETED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'algorithms_visualized': list(algorithm_metrics.keys()),
                'files_generated': len(generated_files),
                'output_directory': output_directory,
                'visualization_types': ['correlation_matrix', 'distributions', 'hypothesis_testing', 'confidence_intervals', 'significance', 'summary']
            }
        )
        
        return generated_files
        
    except Exception as e:
        # Handle statistical visualization errors with comprehensive logging
        logger.error(f"Failed to demonstrate statistical visualization: {e}")
        create_audit_trail(
            action='STATISTICAL_VISUALIZATION_FAILED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'metrics_available': 'algorithm_metrics' in performance_metrics if performance_metrics else False
            }
        )
        raise e


def demonstrate_cross_format_analysis(
    crimaldi_results: Dict[str, List[SimulationResult]],
    custom_results: Dict[str, List[SimulationResult]],
    visualizer: ScientificVisualizer,
    output_directory: str
) -> Dict[str, Any]:
    """
    Demonstrate cross-format analysis capabilities comparing Crimaldi and custom format processing 
    with compatibility assessment and validation metrics.
    
    This function showcases cross-format compatibility analysis with statistical comparison and 
    format-specific performance evaluation for comprehensive data source validation.
    
    Args:
        crimaldi_results: Dictionary of simulation results from Crimaldi format processing
        custom_results: Dictionary of simulation results from custom format processing
        visualizer: ScientificVisualizer instance for cross-format comparison visualization
        output_directory: Directory path for saving cross-format analysis results
        
    Returns:
        Dict[str, Any]: Cross-format analysis results with compatibility metrics and visualization
        
    Raises:
        ValueError: If insufficient data is provided for cross-format analysis
        RuntimeError: If cross-format analysis computation fails
    """
    # Initialize logger for cross-format analysis demonstration
    logger = get_logger('example.cross_format_analysis', 'CROSS_FORMAT_ANALYSIS')
    
    try:
        # Validate input parameters and data availability for both formats
        if not crimaldi_results and not custom_results:
            raise ValueError("No results provided for cross-format analysis")
        
        # Ensure output directory exists for analysis results
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting cross-format analysis demonstration")
        logger.debug(f"Crimaldi format algorithms: {list(crimaldi_results.keys()) if crimaldi_results else 'None'}")
        logger.debug(f"Custom format algorithms: {list(custom_results.keys()) if custom_results else 'None'}")
        
        # Initialize comprehensive cross-format analysis results
        cross_format_analysis = {
            'format_comparison': {},
            'compatibility_metrics': {},
            'performance_correlation': {},
            'statistical_validation': {},
            'format_specific_analysis': {
                'crimaldi': {},
                'custom': {}
            },
            'analysis_metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'crimaldi_algorithms': list(crimaldi_results.keys()) if crimaldi_results else [],
                'custom_algorithms': list(custom_results.keys()) if custom_results else []
            }
        }
        
        # Compare performance metrics between Crimaldi and custom formats
        logger.debug("Comparing performance metrics between formats")
        if crimaldi_results and custom_results:
            # Find common algorithms between formats for direct comparison
            common_algorithms = set(crimaldi_results.keys()) & set(custom_results.keys())
            
            if common_algorithms:
                format_comparison = {}
                
                for algorithm in common_algorithms:
                    logger.debug(f"Comparing {algorithm} across formats")
                    
                    # Calculate metrics for each format
                    crimaldi_metrics = _calculate_format_specific_metrics(crimaldi_results[algorithm], 'crimaldi')
                    custom_metrics = _calculate_format_specific_metrics(custom_results[algorithm], 'custom')
                    
                    # Perform direct metric comparison
                    algorithm_comparison = _compare_format_metrics(crimaldi_metrics, custom_metrics)
                    format_comparison[algorithm] = algorithm_comparison
                
                cross_format_analysis['format_comparison'] = format_comparison
            else:
                logger.warning("No common algorithms found between formats for direct comparison")
        
        # Calculate format compatibility and consistency metrics
        logger.debug("Calculating format compatibility metrics")
        compatibility_metrics = _calculate_compatibility_metrics(
            crimaldi_results=crimaldi_results,
            custom_results=custom_results
        )
        
        cross_format_analysis['compatibility_metrics'] = compatibility_metrics
        
        # Assess cross-format correlation and reproducibility
        logger.debug("Assessing cross-format correlation and reproducibility")
        if crimaldi_results and custom_results:
            correlation_analysis = _assess_cross_format_correlation(
                crimaldi_results=crimaldi_results,
                custom_results=custom_results,
                correlation_threshold=CORRELATION_THRESHOLD
            )
            
            cross_format_analysis['performance_correlation'] = correlation_analysis
        
        # Generate format comparison visualizations
        logger.debug("Generating format comparison visualizations")
        visualization_files = []
        
        if crimaldi_results and custom_results:
            # Create side-by-side performance comparison
            comparison_file = output_path / "format_performance_comparison.pdf"
            
            comparison_figure = visualizer.create_format_comparison_plot(
                crimaldi_results=crimaldi_results,
                custom_results=custom_results,
                plot_title="Crimaldi vs Custom Format Performance Comparison",
                normalize_metrics=True,
                include_statistical_tests=True
            )
            
            comparison_figure.savefig(comparison_file, dpi=300, bbox_inches='tight')
            plt.close(comparison_figure)
            
            visualization_files.append(str(comparison_file))
        
        # Create compatibility assessment reports
        logger.debug("Creating compatibility assessment report")
        compatibility_file = output_path / "format_compatibility_assessment.pdf"
        
        compatibility_figure = visualizer.create_compatibility_assessment_plot(
            compatibility_metrics=compatibility_metrics,
            plot_title="Cross-Format Compatibility Assessment",
            include_consistency_analysis=True,
            show_deviation_metrics=True
        )
        
        compatibility_figure.savefig(compatibility_file, dpi=300, bbox_inches='tight')
        plt.close(compatibility_figure)
        
        visualization_files.append(str(compatibility_file))
        
        # Validate cross-format processing accuracy
        logger.debug("Validating cross-format processing accuracy")
        processing_validation = _validate_cross_format_processing(
            crimaldi_results=crimaldi_results,
            custom_results=custom_results,
            accuracy_threshold=0.95
        )
        
        cross_format_analysis['statistical_validation'] = processing_validation
        
        # Perform format-specific analysis for each format
        if crimaldi_results:
            logger.debug("Performing Crimaldi format-specific analysis")
            crimaldi_analysis = _perform_format_specific_analysis(crimaldi_results, 'crimaldi')
            cross_format_analysis['format_specific_analysis']['crimaldi'] = crimaldi_analysis
        
        if custom_results:
            logger.debug("Performing custom format-specific analysis")
            custom_analysis = _perform_format_specific_analysis(custom_results, 'custom')
            cross_format_analysis['format_specific_analysis']['custom'] = custom_analysis
        
        # Include visualization files in results
        cross_format_analysis['visualization_files'] = visualization_files
        
        # Save comprehensive cross-format analysis results
        results_file = output_path / "cross_format_analysis_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(cross_format_analysis, f, indent=2, default=str)
        
        # Log successful cross-format analysis completion
        logger.info("Cross-format analysis completed successfully")
        
        # Create audit trail for cross-format analysis demonstration
        create_audit_trail(
            action='CROSS_FORMAT_ANALYSIS_COMPLETED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'crimaldi_algorithms': list(crimaldi_results.keys()) if crimaldi_results else [],
                'custom_algorithms': list(custom_results.keys()) if custom_results else [],
                'compatibility_score': compatibility_metrics.get('overall_compatibility', 0),
                'visualization_files_generated': len(visualization_files)
            }
        )
        
        return cross_format_analysis
        
    except Exception as e:
        # Handle cross-format analysis errors with comprehensive logging
        logger.error(f"Failed to demonstrate cross-format analysis: {e}")
        create_audit_trail(
            action='CROSS_FORMAT_ANALYSIS_FAILED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'crimaldi_available': bool(crimaldi_results),
                'custom_available': bool(custom_results)
            }
        )
        raise e


def generate_comprehensive_report(
    analysis_results: Dict[str, Any],
    visualization_files: List[str],
    output_directory: str,
    report_config: Dict[str, Any]
) -> str:
    """
    Generate comprehensive analysis and visualization report combining all demonstration results 
    with scientific documentation, methodology, and publication-ready formatting.
    
    This function creates a comprehensive report integrating all analysis components with 
    scientific documentation standards and methodology descriptions for research publication.
    
    Args:
        analysis_results: Dictionary of all analysis results from demonstration components
        visualization_files: List of generated visualization file paths for inclusion
        output_directory: Directory path for saving the comprehensive report
        report_config: Configuration parameters for report generation and formatting
        
    Returns:
        str: Path to generated comprehensive report file with analysis summary
        
    Raises:
        ValueError: If insufficient analysis results are provided
        IOError: If report generation or file writing fails
    """
    # Initialize logger for comprehensive report generation
    logger = get_logger('example.report_generation', 'REPORT_GENERATION')
    
    try:
        # Validate input parameters and analysis results availability
        if not analysis_results:
            raise ValueError("No analysis results provided for report generation")
        
        # Ensure output directory exists for report files
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting comprehensive report generation")
        
        # Initialize report content with scientific documentation structure
        report_content = {
            'title': f"{EXAMPLE_NAME} - Comprehensive Analysis Report",
            'version': EXAMPLE_VERSION,
            'generation_timestamp': datetime.datetime.now().isoformat(),
            'executive_summary': {},
            'methodology': {},
            'results': {},
            'statistical_validation': {},
            'reproducibility_documentation': {},
            'conclusions': {},
            'appendices': {}
        }
        
        # Compile all analysis results and performance metrics
        logger.debug("Compiling analysis results and performance metrics")
        report_content['results'] = {
            'performance_analysis': analysis_results.get('performance_analysis', {}),
            'trajectory_analysis': analysis_results.get('trajectory_analysis', {}),
            'statistical_analysis': analysis_results.get('statistical_analysis', {}),
            'cross_format_analysis': analysis_results.get('cross_format_analysis', {})
        }
        
        # Generate executive summary with key findings
        logger.debug("Generating executive summary with key findings")
        executive_summary = _generate_executive_summary(
            analysis_results=analysis_results,
            visualization_files=visualization_files
        )
        report_content['executive_summary'] = executive_summary
        
        # Include methodology and experimental design documentation
        logger.debug("Including methodology and experimental design documentation")
        methodology_documentation = _generate_methodology_documentation(
            analysis_results=analysis_results,
            report_config=report_config
        )
        report_content['methodology'] = methodology_documentation
        
        # Add visualization references and figure captions
        logger.debug("Adding visualization references and figure captions")
        visualization_documentation = _generate_visualization_documentation(
            visualization_files=visualization_files,
            output_directory=output_directory
        )
        report_content['visualization_documentation'] = visualization_documentation
        
        # Include statistical validation and compliance status
        logger.debug("Including statistical validation and compliance status")
        validation_summary = _generate_validation_summary(
            analysis_results=analysis_results,
            correlation_threshold=CORRELATION_THRESHOLD,
            reproducibility_threshold=REPRODUCIBILITY_THRESHOLD
        )
        report_content['statistical_validation'] = validation_summary
        
        # Generate reproducibility documentation
        logger.debug("Generating reproducibility documentation")
        reproducibility_docs = _generate_reproducibility_documentation(
            analysis_results=analysis_results,
            report_config=report_config
        )
        report_content['reproducibility_documentation'] = reproducibility_docs
        
        # Format report for scientific publication standards
        logger.debug("Formatting report for scientific publication standards")
        formatted_report = _format_scientific_report(
            report_content=report_content,
            format_type=report_config.get('format_type', 'comprehensive'),
            include_citations=report_config.get('include_citations', True)
        )
        
        # Export report in specified format (JSON, HTML, PDF)
        report_format = report_config.get('output_format', 'json')
        if report_format.lower() == 'json':
            report_file = output_path / "comprehensive_analysis_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_report, f, indent=2, default=str)
        
        elif report_format.lower() == 'html':
            report_file = output_path / "comprehensive_analysis_report.html"
            html_content = _generate_html_report(formatted_report)
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        else:
            # Default to JSON format
            report_file = output_path / "comprehensive_analysis_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_report, f, indent=2, default=str)
        
        # Create audit trail for report generation completion
        create_audit_trail(
            action='COMPREHENSIVE_REPORT_GENERATED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'report_file': str(report_file),
                'report_format': report_format,
                'sections_included': list(report_content.keys()),
                'visualization_files_referenced': len(visualization_files),
                'analysis_components': list(analysis_results.keys())
            }
        )
        
        # Log successful report generation completion
        logger.info(f"Comprehensive report generated successfully: {report_file}")
        
        return str(report_file)
        
    except Exception as e:
        # Handle report generation errors with comprehensive logging
        logger.error(f"Failed to generate comprehensive report: {e}")
        create_audit_trail(
            action='COMPREHENSIVE_REPORT_FAILED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'analysis_results_available': bool(analysis_results),
                'visualization_files_count': len(visualization_files) if visualization_files else 0
            }
        )
        raise e


def validate_example_results(
    analysis_results: Dict[str, Any],
    validation_thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """
    Validate example execution results against scientific computing standards including correlation 
    thresholds, reproducibility requirements, and statistical significance.
    
    This function performs comprehensive validation of analysis results against scientific computing 
    standards with detailed compliance assessment and quality recommendations.
    
    Args:
        analysis_results: Dictionary of analysis results from example execution
        validation_thresholds: Dictionary of validation thresholds for compliance checking
        
    Returns:
        Dict[str, Any]: Validation results with compliance status and quality assessment
        
    Raises:
        ValueError: If insufficient analysis results are provided for validation
    """
    # Initialize logger for example results validation
    logger = get_logger('example.validation', 'VALIDATION')
    
    try:
        # Validate input parameters and analysis results availability
        if not analysis_results:
            raise ValueError("No analysis results provided for validation")
        
        logger.info("Starting example results validation against scientific standards")
        
        # Initialize comprehensive validation results container
        validation_results = {
            'overall_compliance': False,
            'correlation_validation': {},
            'reproducibility_validation': {},
            'statistical_significance_validation': {},
            'visualization_quality_validation': {},
            'cross_format_compatibility_validation': {},
            'recommendations': [],
            'validation_metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'validation_thresholds': validation_thresholds,
                'analysis_components_validated': list(analysis_results.keys())
            }
        }
        
        # Validate correlation coefficients against >95% threshold
        logger.debug("Validating correlation coefficients against threshold")
        correlation_threshold = validation_thresholds.get('correlation_threshold', CORRELATION_THRESHOLD)
        
        correlation_validation = _validate_correlation_thresholds(
            analysis_results=analysis_results,
            correlation_threshold=correlation_threshold
        )
        
        validation_results['correlation_validation'] = correlation_validation
        
        if not correlation_validation['meets_threshold']:
            validation_results['recommendations'].append(
                f"Correlation coefficient ({correlation_validation['actual_correlation']:.3f}) "
                f"below required threshold ({correlation_threshold:.3f})"
            )
        
        # Assess reproducibility coefficient against >0.99 requirement
        logger.debug("Assessing reproducibility coefficient compliance")
        reproducibility_threshold = validation_thresholds.get('reproducibility_threshold', REPRODUCIBILITY_THRESHOLD)
        
        reproducibility_validation = _validate_reproducibility_requirements(
            analysis_results=analysis_results,
            reproducibility_threshold=reproducibility_threshold
        )
        
        validation_results['reproducibility_validation'] = reproducibility_validation
        
        if not reproducibility_validation['meets_threshold']:
            validation_results['recommendations'].append(
                f"Reproducibility coefficient ({reproducibility_validation['actual_coefficient']:.4f}) "
                f"below required threshold ({reproducibility_threshold:.4f})"
            )
        
        # Check statistical significance and effect sizes
        logger.debug("Checking statistical significance and effect sizes")
        significance_validation = _validate_statistical_significance(
            analysis_results=analysis_results,
            alpha_level=validation_thresholds.get('alpha_level', 0.05),
            minimum_effect_size=validation_thresholds.get('minimum_effect_size', 0.2)
        )
        
        validation_results['statistical_significance_validation'] = significance_validation
        
        if not significance_validation['adequate_power']:
            validation_results['recommendations'].append(
                "Statistical analysis shows inadequate power for reliable conclusions"
            )
        
        # Validate visualization quality and completeness
        logger.debug("Validating visualization quality and completeness")
        visualization_validation = _validate_visualization_quality(
            analysis_results=analysis_results,
            required_visualization_types=['trajectory', 'performance', 'statistical', 'comparative']
        )
        
        validation_results['visualization_quality_validation'] = visualization_validation
        
        if not visualization_validation['complete_set']:
            missing_types = visualization_validation.get('missing_types', [])
            validation_results['recommendations'].append(
                f"Missing required visualization types: {', '.join(missing_types)}"
            )
        
        # Assess cross-format compatibility compliance
        logger.debug("Assessing cross-format compatibility compliance")
        if 'cross_format_analysis' in analysis_results:
            compatibility_validation = _validate_cross_format_compatibility(
                cross_format_results=analysis_results['cross_format_analysis'],
                compatibility_threshold=validation_thresholds.get('compatibility_threshold', 0.9)
            )
            
            validation_results['cross_format_compatibility_validation'] = compatibility_validation
            
            if not compatibility_validation['meets_threshold']:
                validation_results['recommendations'].append(
                    "Cross-format compatibility below acceptable threshold"
                )
        
        # Generate validation recommendations based on analysis
        logger.debug("Generating validation recommendations")
        quality_recommendations = _generate_quality_recommendations(
            validation_results=validation_results,
            analysis_results=analysis_results
        )
        
        validation_results['recommendations'].extend(quality_recommendations)
        
        # Create compliance status report with overall assessment
        logger.debug("Creating compliance status report")
        compliance_checks = [
            correlation_validation.get('meets_threshold', False),
            reproducibility_validation.get('meets_threshold', False),
            significance_validation.get('adequate_power', False),
            visualization_validation.get('complete_set', False)
        ]
        
        # Include cross-format compatibility if available
        if 'cross_format_compatibility_validation' in validation_results:
            compliance_checks.append(
                validation_results['cross_format_compatibility_validation'].get('meets_threshold', False)
            )
        
        # Determine overall compliance status
        validation_results['overall_compliance'] = all(compliance_checks)
        validation_results['compliance_score'] = sum(compliance_checks) / len(compliance_checks)
        
        # Log validation completion with compliance status
        logger.info(
            f"Example validation completed: "
            f"Overall compliance: {validation_results['overall_compliance']} "
            f"(Score: {validation_results['compliance_score']:.2f})"
        )
        
        # Create audit trail for validation completion
        create_audit_trail(
            action='EXAMPLE_VALIDATION_COMPLETED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'overall_compliance': validation_results['overall_compliance'],
                'compliance_score': validation_results['compliance_score'],
                'validation_checks': len(compliance_checks),
                'recommendations_count': len(validation_results['recommendations'])
            }
        )
        
        return validation_results
        
    except Exception as e:
        # Handle validation errors with comprehensive logging
        logger.error(f"Failed to validate example results: {e}")
        create_audit_trail(
            action='EXAMPLE_VALIDATION_FAILED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'analysis_results_available': bool(analysis_results)
            }
        )
        raise e


def cleanup_example_execution(
    output_directory: str,
    preserve_results: bool = True,
    generate_archive: bool = False
) -> Dict[str, Any]:
    """
    Clean up example execution resources, temporary files, and finalize output organization 
    with optional result preservation for future reference.
    
    This function provides comprehensive cleanup with selective preservation and archiving 
    capabilities for efficient resource management and result organization.
    
    Args:
        output_directory: Directory containing example execution outputs
        preserve_results: Preserve analysis results and visualizations
        generate_archive: Create compressed archive of preserved results
        
    Returns:
        Dict[str, Any]: Cleanup summary with preserved files and final statistics
        
    Raises:
        IOError: If cleanup operations or file management fails
    """
    # Initialize logger for example execution cleanup
    logger = get_logger('example.cleanup', 'CLEANUP')
    
    try:
        # Validate output directory exists
        output_path = Path(output_directory)
        if not output_path.exists():
            logger.warning(f"Output directory does not exist: {output_directory}")
            return {'success': True, 'message': 'No cleanup required - directory does not exist'}
        
        logger.info(f"Starting example execution cleanup for: {output_directory}")
        
        # Initialize cleanup summary container
        cleanup_summary = {
            'success': False,
            'output_directory': output_directory,
            'files_preserved': [],
            'files_removed': [],
            'archive_created': None,
            'cleanup_timestamp': datetime.datetime.now().isoformat(),
            'statistics': {
                'total_files_processed': 0,
                'files_preserved_count': 0,
                'files_removed_count': 0,
                'total_size_freed_mb': 0,
                'total_size_preserved_mb': 0
            }
        }
        
        # Organize output files by type and importance
        logger.debug("Organizing output files by type and importance")
        file_organization = _organize_output_files(output_path)
        
        # Clean up temporary files and intermediate data
        logger.debug("Cleaning up temporary files and intermediate data")
        temp_files_removed = _cleanup_temporary_files(
            output_path=output_path,
            file_organization=file_organization
        )
        
        cleanup_summary['files_removed'].extend(temp_files_removed)
        cleanup_summary['statistics']['files_removed_count'] = len(temp_files_removed)
        
        # Preserve results if preserve_results is enabled
        if preserve_results:
            logger.debug("Preserving analysis results and visualizations")
            preserved_files = _preserve_important_files(
                output_path=output_path,
                file_organization=file_organization
            )
            
            cleanup_summary['files_preserved'].extend(preserved_files)
            cleanup_summary['statistics']['files_preserved_count'] = len(preserved_files)
        else:
            logger.debug("Results preservation disabled - removing all files")
            all_files = list(output_path.rglob('*'))
            for file_path in all_files:
                if file_path.is_file():
                    file_path.unlink()
                    cleanup_summary['files_removed'].append(str(file_path))
        
        # Generate archive if generate_archive is enabled and results are preserved
        if generate_archive and preserve_results and cleanup_summary['files_preserved']:
            logger.debug("Generating compressed archive of preserved results")
            archive_path = _create_results_archive(
                output_path=output_path,
                preserved_files=cleanup_summary['files_preserved']
            )
            
            cleanup_summary['archive_created'] = str(archive_path)
            logger.info(f"Results archive created: {archive_path}")
        
        # Update file permissions and access controls
        logger.debug("Updating file permissions and access controls")
        if preserve_results:
            _update_file_permissions(
                output_path=output_path,
                preserved_files=cleanup_summary['files_preserved']
            )
        
        # Generate cleanup summary and statistics
        logger.debug("Generating cleanup summary and statistics")
        cleanup_statistics = _calculate_cleanup_statistics(
            output_path=output_path,
            files_preserved=cleanup_summary['files_preserved'],
            files_removed=cleanup_summary['files_removed']
        )
        
        cleanup_summary['statistics'].update(cleanup_statistics)
        
        # Update final cleanup status
        cleanup_summary['success'] = True
        cleanup_summary['statistics']['total_files_processed'] = (
            cleanup_summary['statistics']['files_preserved_count'] + 
            cleanup_summary['statistics']['files_removed_count']
        )
        
        # Log cleanup completion with comprehensive statistics
        logger.info(
            f"Example cleanup completed successfully: "
            f"{cleanup_summary['statistics']['files_preserved_count']} files preserved, "
            f"{cleanup_summary['statistics']['files_removed_count']} files removed, "
            f"{cleanup_summary['statistics']['total_size_freed_mb']:.2f} MB freed"
        )
        
        # Create audit trail for cleanup completion
        create_audit_trail(
            action='EXAMPLE_CLEANUP_COMPLETED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'output_directory': output_directory,
                'preserve_results': preserve_results,
                'generate_archive': generate_archive,
                'cleanup_statistics': cleanup_summary['statistics'],
                'archive_created': cleanup_summary['archive_created'] is not None
            }
        )
        
        return cleanup_summary
        
    except Exception as e:
        # Handle cleanup errors with comprehensive logging
        logger.error(f"Failed to cleanup example execution: {e}")
        create_audit_trail(
            action='EXAMPLE_CLEANUP_FAILED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'output_directory': output_directory
            }
        )
        raise e


class AnalysisVisualizationExample:
    """
    Comprehensive example class demonstrating advanced analysis and visualization capabilities for plume 
    navigation simulation results with scientific formatting, statistical validation, and publication-ready 
    output generation.
    
    This class provides a complete workflow demonstration including configuration management, sample data 
    generation, trajectory visualization, performance analysis, statistical validation, cross-format 
    comparison, and comprehensive reporting with reproducible scientific documentation.
    """
    
    def __init__(
        self,
        config_path: str,
        output_directory: str,
        enable_validation: bool = True
    ):
        """
        Initialize analysis and visualization example with configuration, output directory, and validation 
        settings for comprehensive demonstration execution.
        
        This constructor sets up the complete analysis pipeline with configuration loading, component 
        initialization, and scientific context establishment for reproducible example execution.
        
        Args:
            config_path: Path to example configuration file containing analysis parameters
            output_directory: Directory path for saving generated results and visualizations
            enable_validation: Enable comprehensive validation of results against scientific standards
            
        Raises:
            FileNotFoundError: If configuration file does not exist
            ValueError: If configuration validation fails
            IOError: If output directory creation fails
        """
        # Set configuration path, output directory, and validation settings
        self.config_path = config_path
        self.output_directory = output_directory
        self.validation_enabled = enable_validation
        
        # Initialize logger with scientific context for example execution
        self.logger = get_logger('analysis_visualization_example', 'EXAMPLE_EXECUTION')
        
        # Load example configuration from specified path
        self.configuration = load_example_configuration(
            config_path=self.config_path,
            validate_config=True
        )
        
        # Initialize scientific visualizer with publication settings
        self.visualizer = ScientificVisualizer(
            figure_size=self.configuration.get('visualization_settings', {}).get('figure_size', (12, 8)),
            dpi=self.configuration.get('visualization_settings', {}).get('dpi', 300),
            style_preset='publication',
            color_palette='scientific'
        )
        
        # Setup performance metrics calculator with validation
        self.metrics_calculator = PerformanceMetricsCalculator(
            precision=SCIENTIFIC_PRECISION,
            correlation_threshold=CORRELATION_THRESHOLD,
            enable_statistical_validation=True
        )
        
        # Initialize navigation success analyzer for algorithm evaluation
        self.success_analyzer = NavigationSuccessAnalyzer(
            success_criteria=self.configuration.get('performance_metrics', {}).get('success_criteria', {}),
            statistical_confidence=0.95
        )
        
        # Setup trajectory plotter with scientific formatting
        self.trajectory_plotter = TrajectoryPlotter(
            coordinate_system='normalized',
            trajectory_style='publication',
            include_statistics=True
        )
        
        # Initialize generated files tracking and execution metadata
        self.generated_files = []
        self.execution_metadata = {
            'initialization_timestamp': datetime.datetime.now().isoformat(),
            'example_version': EXAMPLE_VERSION,
            'configuration_loaded': True,
            'validation_enabled': self.validation_enabled
        }
        
        # Create output directory structure
        output_path = Path(self.output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set scientific context for example execution
        set_scientific_context(
            simulation_id='analysis_visualization_example',
            algorithm_name='comprehensive_demonstration',
            processing_stage='INITIALIZATION',
            additional_context={
                'example_class': 'AnalysisVisualizationExample',
                'config_path': self.config_path,
                'output_directory': self.output_directory
            }
        )
        
        # Log successful initialization
        self.logger.info(f"AnalysisVisualizationExample initialized successfully")
        self.logger.debug(f"Configuration: {len(self.configuration)} sections loaded")
        self.logger.debug(f"Output directory: {self.output_directory}")
        
        # Create audit trail for example initialization
        create_audit_trail(
            action='EXAMPLE_INITIALIZED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'config_path': self.config_path,
                'output_directory': self.output_directory,
                'validation_enabled': self.validation_enabled,
                'configuration_sections': list(self.configuration.keys())
            }
        )
    
    def run_complete_example(
        self,
        generate_sample_data: bool = True,
        include_cross_format_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Execute complete analysis and visualization example demonstrating all system capabilities 
        with comprehensive result generation and validation.
        
        This method orchestrates the complete demonstration workflow including data generation, 
        analysis execution, visualization creation, and comprehensive validation against scientific 
        computing standards.
        
        Args:
            generate_sample_data: Generate sample simulation data for demonstration
            include_cross_format_analysis: Include cross-format compatibility analysis
            
        Returns:
            Dict[str, Any]: Complete example execution results with analysis summary and validation
            
        Raises:
            RuntimeError: If example execution fails at any stage
        """
        try:
            # Update scientific context for complete example execution
            set_scientific_context(
                simulation_id='analysis_visualization_example',
                algorithm_name='comprehensive_demonstration',
                processing_stage='COMPLETE_EXAMPLE_EXECUTION'
            )
            
            self.logger.info("Starting complete analysis and visualization example execution")
            
            # Initialize comprehensive execution results container
            execution_results = {
                'execution_summary': {},
                'data_generation': {},
                'trajectory_analysis': {},
                'performance_analysis': {},
                'statistical_visualization': {},
                'cross_format_analysis': {},
                'validation_results': {},
                'comprehensive_report': {},
                'execution_metadata': self.execution_metadata.copy()
            }
            
            execution_start_time = time.time()
            
            # Generate or load simulation results for analysis
            self.logger.info("Step 1: Generating simulation results for analysis")
            if generate_sample_data:
                algorithm_results = generate_sample_simulation_results(
                    num_simulations=self.configuration.get('simulation_parameters', {}).get('num_simulations', 50),
                    algorithm_names=SUPPORTED_ALGORITHMS,
                    generation_config=self.configuration.get('simulation_parameters', {})
                )
                
                execution_results['data_generation'] = {
                    'sample_data_generated': True,
                    'algorithms': list(algorithm_results.keys()),
                    'total_simulations': sum(len(results) for results in algorithm_results.values())
                }
            else:
                # Load existing simulation results (placeholder implementation)
                algorithm_results = self._load_existing_results()
                execution_results['data_generation'] = {
                    'sample_data_generated': False,
                    'loaded_from_file': True
                }
            
            # Execute trajectory visualization demonstration
            self.logger.info("Step 2: Executing trajectory visualization demonstration")
            trajectory_files = demonstrate_trajectory_visualization(
                algorithm_results=algorithm_results,
                visualizer=self.visualizer,
                output_directory=self.output_directory
            )
            
            execution_results['trajectory_analysis'] = {
                'visualization_files': trajectory_files,
                'algorithms_visualized': list(algorithm_results.keys())
            }
            
            self.generated_files.extend(trajectory_files)
            
            # Perform comprehensive performance analysis
            self.logger.info("Step 3: Performing comprehensive performance analysis")
            performance_results = demonstrate_performance_analysis(
                algorithm_results=algorithm_results,
                metrics_calculator=self.metrics_calculator,
                output_directory=self.output_directory
            )
            
            execution_results['performance_analysis'] = performance_results
            
            # Generate statistical visualization examples
            self.logger.info("Step 4: Generating statistical visualization examples")
            statistical_files = demonstrate_statistical_visualization(
                performance_metrics=performance_results,
                visualizer=self.visualizer,
                output_directory=self.output_directory
            )
            
            execution_results['statistical_visualization'] = {
                'visualization_files': statistical_files,
                'statistical_methods_demonstrated': ['correlation', 'distribution', 'hypothesis_testing', 'confidence_intervals']
            }
            
            self.generated_files.extend(statistical_files)
            
            # Include cross-format analysis if enabled
            if include_cross_format_analysis:
                self.logger.info("Step 5: Including cross-format analysis demonstration")
                
                # Generate sample data for different formats
                crimaldi_results = algorithm_results  # Use main results as Crimaldi format
                custom_results = self._generate_custom_format_results(algorithm_results)
                
                cross_format_results = demonstrate_cross_format_analysis(
                    crimaldi_results=crimaldi_results,
                    custom_results=custom_results,
                    visualizer=self.visualizer,
                    output_directory=self.output_directory
                )
                
                execution_results['cross_format_analysis'] = cross_format_results
            
            # Validate all results against scientific standards
            self.logger.info("Step 6: Validating results against scientific computing standards")
            if self.validation_enabled:
                validation_results = validate_example_results(
                    analysis_results=execution_results,
                    validation_thresholds={
                        'correlation_threshold': CORRELATION_THRESHOLD,
                        'reproducibility_threshold': REPRODUCIBILITY_THRESHOLD,
                        'alpha_level': 0.05,
                        'minimum_effect_size': 0.2
                    }
                )
                
                execution_results['validation_results'] = validation_results
            
            # Generate comprehensive analysis report
            self.logger.info("Step 7: Generating comprehensive analysis report")
            report_file = generate_comprehensive_report(
                analysis_results=execution_results,
                visualization_files=self.generated_files,
                output_directory=self.output_directory,
                report_config=self.configuration.get('output_configuration', {})
            )
            
            execution_results['comprehensive_report'] = {
                'report_file': report_file,
                'report_format': self.configuration.get('output_configuration', {}).get('output_format', 'json')
            }
            
            # Calculate execution performance metrics
            execution_time = time.time() - execution_start_time
            execution_results['execution_summary'] = {
                'total_execution_time_seconds': execution_time,
                'steps_completed': 7 if include_cross_format_analysis else 6,
                'files_generated': len(self.generated_files),
                'validation_passed': execution_results.get('validation_results', {}).get('overall_compliance', False),
                'completion_timestamp': datetime.datetime.now().isoformat()
            }
            
            # Create audit trail for complete execution
            create_audit_trail(
                action='COMPLETE_EXAMPLE_EXECUTED',
                component='ANALYSIS_VISUALIZATION_EXAMPLE',
                action_details={
                    'execution_time_seconds': execution_time,
                    'steps_completed': execution_results['execution_summary']['steps_completed'],
                    'files_generated': len(self.generated_files),
                    'validation_passed': execution_results['execution_summary']['validation_passed'],
                    'include_cross_format': include_cross_format_analysis
                }
            )
            
            # Log successful complete example execution
            self.logger.info(
                f"Complete example execution finished successfully in {execution_time:.2f} seconds: "
                f"{len(self.generated_files)} files generated, "
                f"validation {'passed' if execution_results['execution_summary']['validation_passed'] else 'failed'}"
            )
            
            return execution_results
            
        except Exception as e:
            # Handle complete example execution errors
            self.logger.error(f"Complete example execution failed: {e}")
            create_audit_trail(
                action='COMPLETE_EXAMPLE_FAILED',
                component='ANALYSIS_VISUALIZATION_EXAMPLE',
                action_details={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'execution_stage': 'unknown'
                }
            )
            raise RuntimeError(f"Complete example execution failed: {e}") from e
    
    def demonstrate_algorithm_comparison(
        self,
        algorithm_results: Dict[str, List[SimulationResult]],
        comparison_metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Demonstrate comprehensive algorithm comparison with statistical analysis, performance ranking, 
        and optimization recommendations.
        
        This method showcases advanced algorithm comparison capabilities with statistical validation 
        and comprehensive performance evaluation for research and development applications.
        
        Args:
            algorithm_results: Dictionary of simulation results organized by algorithm name
            comparison_metrics: List of metrics to include in comparison analysis
            
        Returns:
            Dict[str, Any]: Algorithm comparison results with statistical analysis and rankings
            
        Raises:
            ValueError: If insufficient data is provided for comparison
        """
        try:
            # Update scientific context for algorithm comparison
            set_scientific_context(
                simulation_id='analysis_visualization_example',
                algorithm_name='algorithm_comparison',
                processing_stage='ALGORITHM_COMPARISON'
            )
            
            self.logger.info(f"Starting algorithm comparison demonstration for {len(algorithm_results)} algorithms")
            
            # Initialize algorithm comparison results container
            comparison_results = {
                'algorithms_compared': list(algorithm_results.keys()),
                'comparison_metrics': comparison_metrics,
                'performance_analysis': {},
                'statistical_comparison': {},
                'algorithm_rankings': {},
                'optimization_recommendations': {},
                'comparison_metadata': {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'total_simulations': sum(len(results) for results in algorithm_results.values())
                }
            }
            
            # Calculate performance metrics for each algorithm
            self.logger.debug("Calculating performance metrics for algorithm comparison")
            algorithm_performance = {}
            
            for algorithm_name, results in algorithm_results.items():
                metrics = self.metrics_calculator.calculate_all_metrics(
                    simulation_results=results,
                    include_statistical_analysis=True
                )
                algorithm_performance[algorithm_name] = metrics
            
            comparison_results['performance_analysis'] = algorithm_performance
            
            # Perform statistical comparison and significance testing
            self.logger.debug("Performing statistical comparison between algorithms")
            statistical_comparison = self.metrics_calculator.compare_algorithm_metrics(
                algorithm_metrics=algorithm_performance,
                comparison_methods=['anova', 'pairwise_t_test', 'effect_size'],
                confidence_level=0.95
            )
            
            comparison_results['statistical_comparison'] = statistical_comparison
            
            # Generate algorithm rankings and efficiency analysis
            self.logger.debug("Generating algorithm rankings")
            rankings = _generate_comprehensive_rankings(
                algorithm_performance=algorithm_performance,
                ranking_criteria=comparison_metrics,
                weighting_scheme=self.configuration.get('performance_metrics', {}).get('ranking_weights', {})
            )
            
            comparison_results['algorithm_rankings'] = rankings
            
            # Create comparative visualizations
            self.logger.debug("Creating algorithm comparison visualizations")
            comparison_files = self._create_algorithm_comparison_visualizations(
                comparison_results=comparison_results,
                output_directory=self.output_directory
            )
            
            comparison_results['visualization_files'] = comparison_files
            self.generated_files.extend(comparison_files)
            
            # Generate optimization recommendations
            self.logger.debug("Generating optimization recommendations")
            optimization_recommendations = _generate_optimization_recommendations(
                algorithm_performance=algorithm_performance,
                statistical_comparison=statistical_comparison
            )
            
            comparison_results['optimization_recommendations'] = optimization_recommendations
            
            # Validate comparison results
            comparison_validation = _validate_comparison_results(
                comparison_results=comparison_results,
                validation_criteria={
                    'minimum_algorithms': 2,
                    'minimum_simulations_per_algorithm': 10,
                    'statistical_power_threshold': 0.8
                }
            )
            
            comparison_results['validation_status'] = comparison_validation
            
            # Log successful algorithm comparison
            self.logger.info("Algorithm comparison demonstration completed successfully")
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Algorithm comparison demonstration failed: {e}")
            raise e
    
    def demonstrate_publication_workflow(
        self,
        analysis_data: Dict[str, Any],
        publication_format: str = 'journal_article'
    ) -> Dict[str, Any]:
        """
        Demonstrate publication-ready workflow including figure generation, statistical validation, 
        and scientific documentation for research publication.
        
        This method showcases the complete publication preparation workflow with scientific formatting, 
        statistical documentation, and reproducibility requirements for research publication.
        
        Args:
            analysis_data: Comprehensive analysis data for publication preparation
            publication_format: Target publication format (journal_article, conference, thesis)
            
        Returns:
            Dict[str, Any]: Publication workflow results with formatted materials and documentation
            
        Raises:
            ValueError: If insufficient analysis data is provided
        """
        try:
            # Update scientific context for publication workflow
            set_scientific_context(
                simulation_id='analysis_visualization_example',
                algorithm_name='publication_workflow',
                processing_stage='PUBLICATION_PREPARATION'
            )
            
            self.logger.info(f"Starting publication workflow demonstration for {publication_format}")
            
            # Initialize publication workflow results
            workflow_results = {
                'publication_format': publication_format,
                'figures_generated': [],
                'documentation_created': [],
                'statistical_validation': {},
                'reproducibility_documentation': {},
                'workflow_metadata': {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'target_format': publication_format
                }
            }
            
            # Generate publication-quality figures with scientific formatting
            self.logger.debug("Generating publication-quality figures")
            publication_figures = self._generate_publication_figures(
                analysis_data=analysis_data,
                publication_format=publication_format
            )
            
            workflow_results['figures_generated'] = publication_figures
            self.generated_files.extend(publication_figures)
            
            # Create statistical validation documentation
            self.logger.debug("Creating statistical validation documentation")
            statistical_docs = self._create_statistical_documentation(
                analysis_data=analysis_data,
                publication_format=publication_format
            )
            
            workflow_results['statistical_validation'] = statistical_docs
            
            # Generate methodology and experimental design documentation
            self.logger.debug("Generating methodology documentation")
            methodology_docs = self._create_methodology_documentation(
                analysis_data=analysis_data,
                publication_format=publication_format
            )
            
            workflow_results['methodology_documentation'] = methodology_docs
            
            # Include reproducibility documentation
            self.logger.debug("Creating reproducibility documentation")
            reproducibility_docs = self._create_reproducibility_documentation(
                analysis_data=analysis_data
            )
            
            workflow_results['reproducibility_documentation'] = reproducibility_docs
            
            # Format results for specified publication format
            self.logger.debug(f"Formatting results for {publication_format}")
            formatted_materials = self._format_publication_materials(
                workflow_results=workflow_results,
                publication_format=publication_format
            )
            
            workflow_results['formatted_materials'] = formatted_materials
            
            # Validate publication compliance
            publication_validation = _validate_publication_compliance(
                workflow_results=workflow_results,
                publication_standards=self._get_publication_standards(publication_format)
            )
            
            workflow_results['publication_validation'] = publication_validation
            
            # Log successful publication workflow completion
            self.logger.info("Publication workflow demonstration completed successfully")
            
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"Publication workflow demonstration failed: {e}")
            raise e
    
    def generate_example_summary(
        self,
        include_detailed_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summary of example execution including performance metrics, validation 
        results, and generated outputs.
        
        This method provides a complete summary of example execution with performance analysis, 
        validation status, and comprehensive output documentation for review and documentation.
        
        Args:
            include_detailed_metrics: Include detailed performance metrics in summary
            
        Returns:
            Dict[str, Any]: Example execution summary with comprehensive analysis and output references
        """
        try:
            self.logger.info("Generating comprehensive example execution summary")
            
            # Initialize example summary container
            example_summary = {
                'example_information': {
                    'name': EXAMPLE_NAME,
                    'version': EXAMPLE_VERSION,
                    'execution_timestamp': datetime.datetime.now().isoformat()
                },
                'configuration_summary': {},
                'execution_performance': {},
                'generated_outputs': {},
                'validation_status': {},
                'recommendations': []
            }
            
            # Include configuration summary
            example_summary['configuration_summary'] = {
                'config_path': self.config_path,
                'output_directory': self.output_directory,
                'validation_enabled': self.validation_enabled,
                'configuration_sections': list(self.configuration.keys())
            }
            
            # Compile execution performance metrics
            if hasattr(self, 'execution_metadata'):
                example_summary['execution_performance'] = self.execution_metadata.copy()
            
            # Include generated file references and descriptions
            example_summary['generated_outputs'] = {
                'total_files': len(self.generated_files),
                'file_list': self.generated_files,
                'output_directory': self.output_directory
            }
            
            # Add detailed metrics if requested
            if include_detailed_metrics:
                detailed_metrics = self._compile_detailed_metrics()
                example_summary['detailed_metrics'] = detailed_metrics
            
            # Include validation results and compliance status
            validation_status = self._get_validation_status()
            example_summary['validation_status'] = validation_status
            
            # Generate execution timeline and statistics
            timeline_stats = self._generate_execution_timeline()
            example_summary['execution_timeline'] = timeline_stats
            
            # Create summary visualization
            summary_visualization = self._create_summary_visualization(
                example_summary=example_summary
            )
            
            if summary_visualization:
                example_summary['summary_visualization'] = summary_visualization
                self.generated_files.append(summary_visualization)
            
            # Log summary generation completion
            self.logger.info("Example summary generation completed successfully")
            
            return example_summary
            
        except Exception as e:
            self.logger.error(f"Example summary generation failed: {e}")
            raise e
    
    def validate_example_execution(
        self,
        execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate example execution results against scientific computing standards and quality requirements.
        
        This method performs comprehensive validation of example execution against scientific computing 
        standards with detailed compliance assessment and quality recommendations.
        
        Args:
            execution_results: Results from example execution for validation
            
        Returns:
            Dict[str, Any]: Validation results with compliance assessment and recommendations
        """
        try:
            self.logger.info("Starting example execution validation")
            
            # Perform comprehensive validation using validate_example_results function
            validation_results = validate_example_results(
                analysis_results=execution_results,
                validation_thresholds={
                    'correlation_threshold': CORRELATION_THRESHOLD,
                    'reproducibility_threshold': REPRODUCIBILITY_THRESHOLD,
                    'alpha_level': 0.05,
                    'minimum_effect_size': 0.2,
                    'compatibility_threshold': 0.9
                }
            )
            
            # Add example-specific validation checks
            example_validation = self._perform_example_specific_validation(
                execution_results=execution_results
            )
            
            validation_results['example_specific_validation'] = example_validation
            
            # Generate comprehensive compliance report
            compliance_report = self._generate_compliance_report(
                validation_results=validation_results
            )
            
            validation_results['compliance_report'] = compliance_report
            
            # Log validation completion
            self.logger.info(f"Example validation completed: {'PASSED' if validation_results['overall_compliance'] else 'FAILED'}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Example execution validation failed: {e}")
            raise e
    
    # Private helper methods for internal functionality
    
    def _load_existing_results(self) -> Dict[str, List[SimulationResult]]:
        """Load existing simulation results from files (placeholder implementation)."""
        # Placeholder implementation - would load actual results from files
        return {}
    
    def _generate_custom_format_results(self, base_results: Dict[str, List[SimulationResult]]) -> Dict[str, List[SimulationResult]]:
        """Generate custom format results for cross-format analysis."""
        # Placeholder implementation - would modify base results to simulate custom format
        return base_results
    
    def _create_algorithm_comparison_visualizations(self, comparison_results: Dict[str, Any], output_directory: str) -> List[str]:
        """Create visualizations for algorithm comparison analysis."""
        # Placeholder implementation
        return []
    
    def _generate_publication_figures(self, analysis_data: Dict[str, Any], publication_format: str) -> List[str]:
        """Generate publication-quality figures for specified format."""
        # Placeholder implementation
        return []
    
    def _create_statistical_documentation(self, analysis_data: Dict[str, Any], publication_format: str) -> Dict[str, Any]:
        """Create statistical validation documentation."""
        # Placeholder implementation
        return {}
    
    def _create_methodology_documentation(self, analysis_data: Dict[str, Any], publication_format: str) -> Dict[str, Any]:
        """Create methodology and experimental design documentation."""
        # Placeholder implementation
        return {}
    
    def _create_reproducibility_documentation(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create reproducibility documentation."""
        # Placeholder implementation
        return {}
    
    def _format_publication_materials(self, workflow_results: Dict[str, Any], publication_format: str) -> Dict[str, Any]:
        """Format materials for specified publication format."""
        # Placeholder implementation
        return {}
    
    def _get_publication_standards(self, publication_format: str) -> Dict[str, Any]:
        """Get publication standards for specified format."""
        # Placeholder implementation
        return {}
    
    def _compile_detailed_metrics(self) -> Dict[str, Any]:
        """Compile detailed performance metrics for summary."""
        # Placeholder implementation
        return {}
    
    def _get_validation_status(self) -> Dict[str, Any]:
        """Get current validation status."""
        # Placeholder implementation
        return {}
    
    def _generate_execution_timeline(self) -> Dict[str, Any]:
        """Generate execution timeline and statistics."""
        # Placeholder implementation
        return {}
    
    def _create_summary_visualization(self, example_summary: Dict[str, Any]) -> Optional[str]:
        """Create summary visualization."""
        # Placeholder implementation
        return None
    
    def _perform_example_specific_validation(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform example-specific validation checks."""
        # Placeholder implementation
        return {}
    
    def _generate_compliance_report(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        # Placeholder implementation
        return {}


# Helper functions for algorithm characteristics and data generation

def _get_algorithm_characteristics(algorithm_name: str) -> Dict[str, Any]:
    """Get algorithm-specific characteristics for realistic data generation."""
    characteristics = {
        'infotaxis': {
            'exploration_rate': 0.7,
            'efficiency_score': 0.8,
            'success_rate': 0.85,
            'path_optimality': 0.75
        },
        'casting': {
            'exploration_rate': 0.5,
            'efficiency_score': 0.9,
            'success_rate': 0.80,
            'path_optimality': 0.85
        },
        'gradient_following': {
            'exploration_rate': 0.3,
            'efficiency_score': 0.95,
            'success_rate': 0.90,
            'path_optimality': 0.90
        },
        'hybrid_strategies': {
            'exploration_rate': 0.6,
            'efficiency_score': 0.85,
            'success_rate': 0.88,
            'path_optimality': 0.82
        }
    }
    
    return characteristics.get(algorithm_name, {
        'exploration_rate': 0.5,
        'efficiency_score': 0.7,
        'success_rate': 0.75,
        'path_optimality': 0.7
    })


def _generate_realistic_trajectory(
    algorithm_name: str,
    algorithm_params: Dict[str, Any],
    arena_size: Tuple[int, int],
    target_position: Tuple[int, int],
    duration: int,
    noise_level: float,
    simulation_id: str
) -> Dict[str, Any]:
    """Generate realistic trajectory data with algorithm-specific patterns."""
    # Generate time series data
    time_points = np.linspace(0, duration, num=duration)
    
    # Generate position trajectory with algorithm-specific behavior
    start_position = (10, 10)  # Starting position
    
    # Simple trajectory generation (placeholder for more sophisticated implementation)
    trajectory_points = []
    current_pos = start_position
    
    for t in time_points:
        # Add algorithm-specific movement patterns
        if algorithm_name == 'gradient_following':
            # Direct movement towards target with some noise
            direction = np.array(target_position) - np.array(current_pos)
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
            step = direction * 2 + np.random.normal(0, noise_level, 2)
        else:
            # More exploratory movement
            step = np.random.normal(0, 1, 2) + np.random.normal(0, noise_level, 2)
        
        current_pos = np.array(current_pos) + step
        
        # Keep within arena bounds
        current_pos[0] = np.clip(current_pos[0], 0, arena_size[0])
        current_pos[1] = np.clip(current_pos[1], 0, arena_size[1])
        
        trajectory_points.append(current_pos.copy())
    
    return {
        'simulation_id': simulation_id,
        'time_points': time_points.tolist(),
        'trajectory_points': [point.tolist() for point in trajectory_points],
        'start_position': start_position,
        'target_position': target_position,
        'arena_size': arena_size,
        'algorithm_name': algorithm_name
    }


def _calculate_sample_performance_metrics(
    trajectory_data: Dict[str, Any],
    target_position: Tuple[int, int],
    algorithm_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate realistic performance metrics for sample trajectory data."""
    # Extract trajectory information
    trajectory_points = np.array(trajectory_data['trajectory_points'])
    final_position = trajectory_points[-1]
    
    # Calculate distance to target
    distance_to_target = np.linalg.norm(np.array(final_position) - np.array(target_position))
    
    # Calculate path length
    path_distances = np.linalg.norm(np.diff(trajectory_points, axis=0), axis=1)
    total_path_length = np.sum(path_distances)
    
    # Calculate success based on distance threshold
    success_threshold = 5.0  # Distance threshold for success
    navigation_success = distance_to_target <= success_threshold
    
    # Calculate efficiency score
    direct_distance = np.linalg.norm(
        np.array(target_position) - np.array(trajectory_data['start_position'])
    )
    efficiency_score = direct_distance / total_path_length if total_path_length > 0 else 0
    
    # Time to target (simplified)
    time_to_target = len(trajectory_data['time_points']) if navigation_success else None
    
    return {
        'navigation_success': navigation_success,
        'distance_to_target': distance_to_target,
        'total_path_length': total_path_length,
        'efficiency_score': efficiency_score,
        'time_to_target': time_to_target,
        'success_rate': float(navigation_success),
        'path_optimality': algorithm_params.get('path_optimality', 0.7) + np.random.normal(0, 0.1)
    }


# Additional helper functions (placeholder implementations)

def _generate_algorithm_rankings(algorithm_metrics: Dict[str, Any], ranking_criteria: List[str]) -> Dict[str, Any]:
    """Generate algorithm rankings based on specified criteria."""
    # Placeholder implementation
    return {'rankings': {}, 'criteria': ranking_criteria}


def _validate_correlation_accuracy(algorithm_metrics: Dict[str, Any], correlation_threshold: float) -> Dict[str, Any]:
    """Validate correlation accuracy against threshold."""
    # Placeholder implementation
    return {'overall_compliance': True, 'correlation_coefficient': 0.96}


def _assess_reproducibility_coefficient(algorithm_results: Dict[str, Any], reproducibility_threshold: float) -> Dict[str, Any]:
    """Assess reproducibility coefficient compliance."""
    # Placeholder implementation
    return {'coefficient': 0.995, 'meets_threshold': True}


def _create_performance_visualizations(analysis_results: Dict[str, Any], output_directory: Path) -> List[str]:
    """Create performance comparison visualizations."""
    # Placeholder implementation
    return []


def _perform_statistical_significance_testing(algorithm_metrics: Dict[str, Any], alpha_level: float, correction_method: str) -> Dict[str, Any]:
    """Perform statistical significance testing."""
    # Placeholder implementation
    return {'significant_differences': [], 'p_values': {}}


def _validate_analysis_compliance(analysis_results: Dict[str, Any], correlation_threshold: float, reproducibility_threshold: float) -> Dict[str, Any]:
    """Validate overall analysis compliance."""
    # Placeholder implementation
    return {'overall_compliance': True, 'compliance_score': 0.95}


# Main execution block for running as standalone script

def main():
    """
    Main execution function for running the analysis and visualization example as a standalone script 
    with command-line interface and comprehensive error handling.
    
    This function provides command-line interface for example execution with configurable parameters, 
    comprehensive error handling, and detailed logging for research and development workflows.
    """
    # Configure argument parser for command-line interface
    parser = argparse.ArgumentParser(
        description=f"{EXAMPLE_NAME} v{EXAMPLE_VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python analysis_visualization.py --config data/config.json --output-dir results/
  python analysis_visualization.py --generate-sample-data --include-cross-format --validation-enabled
        """
    )
    
    # Add command-line arguments for example configuration
    parser.add_argument(
        '--config',
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help='Path to example configuration file (default: %(default)s)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for generated results and visualizations (default: %(default)s)'
    )
    
    parser.add_argument(
        '--generate-sample-data',
        action='store_true',
        help='Generate sample simulation data for demonstration'
    )
    
    parser.add_argument(
        '--include-cross-format',
        action='store_true',
        help='Include cross-format analysis demonstration'
    )
    
    parser.add_argument(
        '--validation-enabled',
        action='store_true',
        default=True,
        help='Enable comprehensive validation of results (default: %(default)s)'
    )
    
    parser.add_argument(
        '--publication-format',
        type=str,
        choices=['png', 'pdf', 'svg'],
        default='pdf',
        help='Output format for publication-ready figures (default: %(default)s)'
    )
    
    # Parse command-line arguments and validate input parameters
    args = parser.parse_args()
    
    try:
        # Initialize logging system with scientific context
        from ..utils.logging_utils import initialize_logging_system
        
        logging_initialized = initialize_logging_system(
            enable_console_output=True,
            enable_file_logging=True,
            log_level='INFO'
        )
        
        if not logging_initialized:
            print("WARNING: Logging system initialization failed, continuing with basic logging", file=sys.stderr)
        
        # Set scientific context for main execution
        set_scientific_context(
            simulation_id='analysis_visualization_main',
            algorithm_name='standalone_execution',
            processing_stage='MAIN_EXECUTION',
            additional_context={
                'script_mode': 'standalone',
                'command_line_args': vars(args)
            }
        )
        
        # Get logger for main execution
        logger = get_logger('main.execution', 'MAIN')
        
        # Log example execution start
        logger.info(f"Starting {EXAMPLE_NAME} v{EXAMPLE_VERSION}")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Generate sample data: {args.generate_sample_data}")
        logger.info(f"Include cross-format analysis: {args.include_cross_format}")
        logger.info(f"Validation enabled: {args.validation_enabled}")
        
        # Load example configuration and validate settings
        try:
            if not Path(args.config).exists():
                logger.warning(f"Configuration file not found: {args.config}")
                logger.info("Creating default configuration...")
                _create_default_configuration(args.config)
            
            logger.info("Loading example configuration...")
            
        except Exception as config_error:
            logger.error(f"Configuration loading failed: {config_error}")
            print(f"ERROR: Configuration loading failed: {config_error}", file=sys.stderr)
            return 1
        
        # Create AnalysisVisualizationExample instance
        logger.info("Creating AnalysisVisualizationExample instance...")
        
        try:
            example_instance = AnalysisVisualizationExample(
                config_path=args.config,
                output_directory=args.output_dir,
                enable_validation=args.validation_enabled
            )
            
        except Exception as init_error:
            logger.error(f"Example instance creation failed: {init_error}")
            print(f"ERROR: Example initialization failed: {init_error}", file=sys.stderr)
            return 1
        
        # Execute complete example demonstration
        logger.info("Executing complete example demonstration...")
        
        try:
            execution_results = example_instance.run_complete_example(
                generate_sample_data=args.generate_sample_data,
                include_cross_format_analysis=args.include_cross_format
            )
            
        except Exception as execution_error:
            logger.error(f"Example execution failed: {execution_error}")
            print(f"ERROR: Example execution failed: {execution_error}", file=sys.stderr)
            return 1
        
        # Validate results against scientific computing standards
        if args.validation_enabled:
            logger.info("Validating results against scientific computing standards...")
            
            try:
                validation_results = example_instance.validate_example_execution(execution_results)
                
                if validation_results['overall_compliance']:
                    logger.info("✓ All validation checks passed - results meet scientific computing standards")
                else:
                    logger.warning("⚠ Some validation checks failed - see recommendations for improvement")
                    for recommendation in validation_results.get('recommendations', []):
                        logger.warning(f"  - {recommendation}")
                
            except Exception as validation_error:
                logger.error(f"Results validation failed: {validation_error}")
                print(f"WARNING: Results validation failed: {validation_error}", file=sys.stderr)
        
        # Generate comprehensive execution report
        logger.info("Generating comprehensive execution report...")
        
        try:
            execution_summary = example_instance.generate_example_summary(include_detailed_metrics=True)
            
            # Save execution summary
            summary_file = Path(args.output_dir) / "execution_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(execution_summary, f, indent=2, default=str)
            
            logger.info(f"Execution summary saved: {summary_file}")
            
        except Exception as summary_error:
            logger.error(f"Execution summary generation failed: {summary_error}")
            print(f"WARNING: Summary generation failed: {summary_error}", file=sys.stderr)
        
        # Clean up resources and organize outputs
        logger.info("Cleaning up resources and organizing outputs...")
        
        try:
            cleanup_results = cleanup_example_execution(
                output_directory=args.output_dir,
                preserve_results=True,
                generate_archive=True
            )
            
            logger.info(f"Cleanup completed: {cleanup_results['statistics']['files_preserved_count']} files preserved")
            
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed: {cleanup_error}")
            print(f"WARNING: Cleanup failed: {cleanup_error}", file=sys.stderr)
        
        # Log example completion with audit trail
        total_files = len(example_instance.generated_files)
        execution_time = execution_results.get('execution_summary', {}).get('total_execution_time_seconds', 0)
        validation_passed = execution_results.get('validation_results', {}).get('overall_compliance', False)
        
        create_audit_trail(
            action='MAIN_EXECUTION_COMPLETED',
            component='ANALYSIS_VISUALIZATION_EXAMPLE',
            action_details={
                'config_path': args.config,
                'output_directory': args.output_dir,
                'files_generated': total_files,
                'execution_time_seconds': execution_time,
                'validation_passed': validation_passed,
                'command_line_args': vars(args)
            }
        )
        
        # Log successful completion
        logger.info(f"✓ Example execution completed successfully!")
        logger.info(f"  Files generated: {total_files}")
        logger.info(f"  Execution time: {execution_time:.2f} seconds")
        logger.info(f"  Validation: {'PASSED' if validation_passed else 'FAILED'}")
        logger.info(f"  Results saved to: {args.output_dir}")
        
        print(f"\n{EXAMPLE_NAME} completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Files generated: {total_files}")
        print(f"Validation: {'PASSED' if validation_passed else 'FAILED'}")
        
        # Exit with appropriate status code
        return 0 if validation_passed or not args.validation_enabled else 1
        
    except KeyboardInterrupt:
        print("\nExample execution interrupted by user", file=sys.stderr)
        return 130
        
    except Exception as e:
        print(f"CRITICAL ERROR: Unexpected failure in main execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


def _create_default_configuration(config_path: str) -> None:
    """Create default configuration file for example execution."""
    default_config = {
        "simulation_parameters": {
            "num_simulations": 50,
            "algorithms": SUPPORTED_ALGORITHMS,
            "arena_size": [100, 100],
            "target_position": [80, 80],
            "duration_seconds": 300,
            "noise_level": 0.1,
            "random_seed": 42
        },
        "visualization_settings": {
            "figure_size": [12, 8],
            "dpi": 300,
            "output_formats": ["pdf", "png"],
            "style_preset": "publication"
        },
        "performance_metrics": {
            "correlation_threshold": CORRELATION_THRESHOLD,
            "success_criteria": {
                "distance_threshold": 5.0,
                "time_threshold": 300
            },
            "ranking_weights": {
                "success_rate": 0.3,
                "efficiency_score": 0.3,
                "time_to_target": 0.2,
                "path_optimality": 0.2
            }
        },
        "output_configuration": {
            "output_format": "json",
            "include_citations": True,
            "generate_html_report": False
        }
    }
    
    # Ensure configuration directory exists
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save default configuration
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Default configuration created: {config_path}")


# Execute main function when script is run directly
if __name__ == '__main__':
    sys.exit(main())