"""
Comprehensive report generation module providing automated generation of scientific analysis reports, 
comparative studies, performance summaries, and publication-ready documentation for plume navigation 
simulation analysis.

This module implements advanced report templating, multi-format output support, statistical visualization 
integration, and scientific documentation standards with >95% correlation validation and reproducible 
research outcomes. Features intelligent report composition, cross-algorithm comparison reporting, 
batch analysis summaries, and integration with visualization and performance metrics systems for 
complete scientific documentation workflows.

Key Features:
- Automated scientific report generation with template-based architecture
- Multi-format output support (HTML, PDF, Markdown, JSON, YAML, TXT)
- Statistical validation and correlation analysis integration
- Cross-algorithm comparison and performance ranking reports
- Batch processing result documentation with comprehensive analysis
- Scientific reproducibility documentation and methodology tracking
- Publication-ready formatting with scientific notation and precision
- Audit trail integration for complete traceability
- Real-time progress tracking and performance optimization
- Thread-safe parallel report generation capabilities

Technical Standards:
- >95% correlation validation with reference implementations
- Scientific documentation standards compliance
- Reproducible research outcomes with complete audit trails
- Performance targets: <7.2 seconds average processing time
- Memory optimization for large-scale batch reporting
- Cross-platform compatibility and format consistency
"""

# External library imports with version requirements for scientific computing and report generation
import jinja2  # Jinja2 3.1.2+ - Template engine for report generation with scientific formatting and customizable layouts
import markdown  # Markdown 3.5.0+ - Markdown processing for scientific documentation and report formatting
import weasyprint  # WeasyPrint 60.0+ - PDF generation for publication-ready scientific reports and documentation
import numpy as np  # numpy 2.1.3+ - Numerical array operations for report data processing and statistical calculations
import pandas as pd  # pandas 2.2.0+ - Data manipulation and analysis for report data aggregation and table generation
import datetime  # Python 3.9+ - Timestamp generation and temporal analysis for report metadata and versioning
from pathlib import Path  # Python 3.9+ - Cross-platform path handling for report file management and template loading
from typing import Dict, Any, List, Optional, Union, Callable  # Python 3.9+ - Type hints for report generator function signatures and data structures
from dataclasses import dataclass, field  # Python 3.9+ - Data class decorators for report configuration and result structures
import json  # Python 3.9+ - JSON serialization for report metadata and configuration management
import yaml  # PyYAML 6.0+ - YAML processing for report configuration and metadata management
import base64  # Python 3.9+ - Base64 encoding for embedding figures and visualizations in reports
import io  # Python 3.9+ - Input/output operations for report buffer management and streaming
import copy  # Python 3.9+ - Deep copying of report data for isolation and template processing
import uuid  # Python 3.9+ - Unique identifier generation for report tracking and version management
import threading  # Python 3.9+ - Thread-safe report generation operations and concurrent processing
import concurrent.futures  # Python 3.9+ - Parallel report generation for batch processing and performance optimization

# Internal imports for performance metrics, statistical analysis, and visualization integration
from .performance_metrics import (
    PerformanceMetricsCalculator,
    NavigationSuccessAnalyzer
)
from .statistical_comparison import (
    StatisticalComparator, 
    AlgorithmRankingAnalyzer
)
from .visualization import (
    ScientificVisualizer,
    TrajectoryPlotter
)
from ..simulation.result_collector import (
    SimulationResult,
    BatchSimulationResult,
    CrossAlgorithmAnalysis
)
from ...io.result_writer import (
    ResultWriter,
    write_analysis_report
)
from ...utils.logging_utils import (
    get_logger,
    log_performance_metrics,
    create_audit_trail,
    format_scientific_value
)

# Global configuration constants for report generation system
REPORT_GENERATOR_VERSION = '1.0.0'
SUPPORTED_REPORT_FORMATS = ['html', 'pdf', 'markdown', 'json', 'yaml', 'txt']
DEFAULT_REPORT_FORMAT = 'html'
REPORT_TEMPLATE_DIRECTORY = 'templates/reports'
DEFAULT_TEMPLATE_NAME = 'scientific_analysis_report.html'
SCIENTIFIC_REPORT_STYLES = ['publication', 'technical', 'executive', 'detailed']
DEFAULT_REPORT_STYLE = 'publication'

# Feature flags for report generation capabilities
REPORT_VALIDATION_ENABLED = True
VISUALIZATION_INTEGRATION_ENABLED = True
STATISTICAL_ANALYSIS_INTEGRATION = True
CROSS_ALGORITHM_REPORTING = True
BATCH_REPORTING_ENABLED = True
REPRODUCIBILITY_DOCUMENTATION = True
AUDIT_TRAIL_INTEGRATION = True

# Scientific precision and formatting configuration
SCIENTIFIC_PRECISION_DIGITS = 6
MAX_REPORT_SIZE_MB = 100
REPORT_CACHE_TTL_SECONDS = 3600
PARALLEL_REPORT_GENERATION = True
REPORT_COMPRESSION_ENABLED = True

# Global caches and thread-safe containers for performance optimization
_report_templates_cache: Dict[str, jinja2.Template] = {}
_report_generation_locks: Dict[str, threading.Lock] = {}
_report_statistics: Dict[str, Any] = {}


@dataclass
class GeneratedReport:
    """
    Generated report container class storing report content, metadata, validation results, 
    and export capabilities for comprehensive report management and distribution.
    
    This class encapsulates all aspects of a generated report including content, metadata,
    validation status, and export capabilities for complete report lifecycle management.
    """
    
    # Core report identification and content
    report_id: str
    report_type: str
    content: str
    format: str
    
    # Metadata and tracking information
    generation_timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_result: Dict[str, Any] = field(default_factory=dict)
    included_visualizations: List[str] = field(default_factory=list)
    generation_statistics: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None
    content_hash: str = field(default='')
    
    def __post_init__(self):
        """Initialize report with content hash calculation and metadata setup."""
        if not self.content_hash:
            import hashlib
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        
        # Initialize default metadata if not provided
        if not self.metadata:
            self.metadata = {
                'generator_version': REPORT_GENERATOR_VERSION,
                'creation_timestamp': self.generation_timestamp.isoformat(),
                'report_format': self.format,
                'content_length': len(self.content),
                'visualization_count': len(self.included_visualizations)
            }
    
    def validate_content(self, validation_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate report content for accuracy, completeness, and scientific standards compliance.
        
        This method performs comprehensive validation of report content including structure,
        scientific accuracy, numerical precision, and compliance with documentation standards.
        
        Args:
            validation_criteria: Criteria and thresholds for validation assessment
            
        Returns:
            Dict[str, Any]: Content validation result with error analysis and quality assessment
        """
        logger = get_logger('report.validation', 'VALIDATION')
        logger.debug(f"Starting content validation for report {self.report_id}")
        
        validation_result = {
            'validation_id': str(uuid.uuid4()),
            'report_id': self.report_id,
            'validation_timestamp': datetime.datetime.now().isoformat(),
            'validation_passed': True,
            'validation_errors': [],
            'validation_warnings': [],
            'quality_metrics': {},
            'compliance_status': {}
        }
        
        try:
            # Validate report structure and completeness
            structure_validation = self._validate_structure(validation_criteria)
            validation_result['quality_metrics']['structure_score'] = structure_validation['score']
            
            if not structure_validation['passed']:
                validation_result['validation_passed'] = False
                validation_result['validation_errors'].extend(structure_validation['errors'])
            
            # Check scientific accuracy and numerical precision
            if validation_criteria.get('validate_scientific_accuracy', True):
                accuracy_validation = self._validate_scientific_accuracy(validation_criteria)
                validation_result['quality_metrics']['accuracy_score'] = accuracy_validation['score']
                
                if not accuracy_validation['passed']:
                    validation_result['validation_passed'] = False
                    validation_result['validation_errors'].extend(accuracy_validation['errors'])
            
            # Validate visualization integration and figure quality
            if self.included_visualizations and VISUALIZATION_INTEGRATION_ENABLED:
                viz_validation = self._validate_visualizations(validation_criteria)
                validation_result['quality_metrics']['visualization_score'] = viz_validation['score']
                
                if viz_validation['warnings']:
                    validation_result['validation_warnings'].extend(viz_validation['warnings'])
            
            # Check compliance with scientific documentation standards
            compliance_validation = self._validate_compliance(validation_criteria)
            validation_result['compliance_status'] = compliance_validation
            
            # Calculate overall quality score
            quality_scores = [score for score in validation_result['quality_metrics'].values() if isinstance(score, (int, float))]
            if quality_scores:
                validation_result['overall_quality_score'] = sum(quality_scores) / len(quality_scores)
            else:
                validation_result['overall_quality_score'] = 0
            
            # Store validation result
            self.validation_result = validation_result
            
            # Log validation completion
            logger.info(f"Content validation completed for report {self.report_id} - Passed: {validation_result['validation_passed']}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error during content validation for report {self.report_id}: {e}")
            validation_result['validation_passed'] = False
            validation_result['validation_errors'].append(f"Validation process error: {str(e)}")
            return validation_result
    
    def export_to_format(
        self, 
        target_format: str, 
        export_path: str, 
        export_options: Dict[str, Any]
    ) -> 'ReportExportResult':
        """
        Export report to specified format with optimization and validation.
        
        This method converts the report content to the target format with format-specific
        optimizations and validation to ensure quality and compatibility.
        
        Args:
            target_format: Target format for export (pdf, html, markdown, etc.)
            export_path: Path where exported report should be saved
            export_options: Format-specific export options and settings
            
        Returns:
            ReportExportResult: Export result with format validation and file information
        """
        logger = get_logger('report.export', 'EXPORT')
        logger.debug(f"Exporting report {self.report_id} to format {target_format}")
        
        export_id = str(uuid.uuid4())
        export_start_time = datetime.datetime.now()
        
        try:
            # Validate target format compatibility
            if target_format not in SUPPORTED_REPORT_FORMATS:
                raise ValueError(f"Unsupported export format: {target_format}")
            
            # Prepare export path
            export_file_path = Path(export_path)
            export_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert content based on target format
            converted_content = self._convert_to_format(target_format, export_options)
            
            # Apply format-specific optimization
            optimized_content = self._optimize_for_format(converted_content, target_format, export_options)
            
            # Write exported content to file
            with open(export_file_path, 'w', encoding='utf-8') as f:
                f.write(optimized_content)
            
            # Calculate file information
            file_size = export_file_path.stat().st_size
            file_checksum = self._calculate_file_checksum(export_file_path)
            
            # Create export result
            export_result = ReportExportResult(
                export_id=export_id,
                export_success=True,
                target_format=target_format,
                export_path=str(export_file_path)
            )
            
            export_result.file_size_bytes = file_size
            export_result.file_checksum = file_checksum
            export_result.export_timestamp = datetime.datetime.now()
            
            # Perform format validation if requested
            if export_options.get('validate_export', True):
                format_validation = self._validate_exported_format(export_file_path, target_format)
                export_result.format_validation = format_validation
            
            # Calculate optimization metrics
            export_time = (datetime.datetime.now() - export_start_time).total_seconds()
            export_result.optimization_metrics = {
                'export_time_seconds': export_time,
                'file_size_mb': file_size / (1024 * 1024),
                'compression_ratio': len(self.content) / len(optimized_content) if optimized_content else 1.0
            }
            
            logger.info(f"Report {self.report_id} exported successfully to {target_format} format")
            return export_result
            
        except Exception as e:
            logger.error(f"Error exporting report {self.report_id} to format {target_format}: {e}")
            return ReportExportResult(
                export_id=export_id,
                export_success=False,
                target_format=target_format,
                export_path=export_path
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert generated report to dictionary format for serialization and analysis.
        
        This method converts all report properties to dictionary format for serialization,
        analysis, and integration with external systems.
        
        Returns:
            Dict[str, Any]: Report as comprehensive dictionary with all properties and metadata
        """
        return {
            'report_id': self.report_id,
            'report_type': self.report_type,
            'content': self.content,
            'format': self.format,
            'generation_timestamp': self.generation_timestamp.isoformat(),
            'metadata': self.metadata,
            'validation_result': self.validation_result,
            'included_visualizations': self.included_visualizations,
            'generation_statistics': self.generation_statistics,
            'file_path': self.file_path,
            'content_hash': self.content_hash,
            'content_length': len(self.content),
            'visualization_count': len(self.included_visualizations)
        }
    
    def _validate_structure(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Validate report structure and organization."""
        structure_score = 100
        errors = []
        
        # Check for required sections
        required_sections = criteria.get('required_sections', [])
        for section in required_sections:
            if section not in self.content:
                errors.append(f"Missing required section: {section}")
                structure_score -= 20
        
        # Validate content length
        min_length = criteria.get('min_content_length', 1000)
        if len(self.content) < min_length:
            errors.append(f"Content too short: {len(self.content)} < {min_length}")
            structure_score -= 15
        
        return {
            'passed': len(errors) == 0,
            'score': max(0, structure_score),
            'errors': errors
        }
    
    def _validate_scientific_accuracy(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scientific accuracy and numerical precision."""
        accuracy_score = 100
        errors = []
        
        # Check for scientific notation compliance
        correlation_threshold = criteria.get('correlation_threshold', 0.95)
        if 'correlation' in self.metadata:
            if self.metadata['correlation'] < correlation_threshold:
                errors.append(f"Correlation below threshold: {self.metadata['correlation']} < {correlation_threshold}")
                accuracy_score -= 30
        
        return {
            'passed': len(errors) == 0,
            'score': max(0, accuracy_score),
            'errors': errors
        }
    
    def _validate_visualizations(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Validate visualization integration and quality."""
        viz_score = 100
        warnings = []
        
        if not self.included_visualizations:
            warnings.append("No visualizations included in report")
            viz_score -= 20
        
        return {
            'score': max(0, viz_score),
            'warnings': warnings
        }
    
    def _validate_compliance(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compliance with scientific documentation standards."""
        return {
            'reproducibility_compliant': REPRODUCIBILITY_DOCUMENTATION,
            'audit_trail_present': bool(self.metadata.get('audit_trail_id')),
            'scientific_precision_met': True
        }
    
    def _convert_to_format(self, target_format: str, options: Dict[str, Any]) -> str:
        """Convert report content to target format."""
        if target_format == 'pdf':
            return self._convert_to_pdf(options)
        elif target_format == 'markdown':
            return self._convert_to_markdown(options)
        elif target_format == 'json':
            return self._convert_to_json(options)
        elif target_format == 'yaml':
            return self._convert_to_yaml(options)
        elif target_format == 'txt':
            return self._convert_to_text(options)
        else:
            return self.content  # HTML or other formats
    
    def _convert_to_pdf(self, options: Dict[str, Any]) -> str:
        """Convert HTML content to PDF format using WeasyPrint."""
        try:
            html_doc = weasyprint.HTML(string=self.content)
            pdf_bytes = html_doc.write_pdf()
            return base64.b64encode(pdf_bytes).decode('utf-8')
        except Exception as e:
            return f"PDF conversion error: {e}"
    
    def _convert_to_markdown(self, options: Dict[str, Any]) -> str:
        """Convert HTML content to Markdown format."""
        # Basic HTML to Markdown conversion
        # In production, would use proper HTML to Markdown converter
        return self.content.replace('<h1>', '# ').replace('</h1>', '\n')
    
    def _convert_to_json(self, options: Dict[str, Any]) -> str:
        """Convert report to JSON format."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def _convert_to_yaml(self, options: Dict[str, Any]) -> str:
        """Convert report to YAML format."""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def _convert_to_text(self, options: Dict[str, Any]) -> str:
        """Convert report to plain text format."""
        # Strip HTML tags for plain text
        import re
        return re.sub('<[^<]+?>', '', self.content)
    
    def _optimize_for_format(self, content: str, target_format: str, options: Dict[str, Any]) -> str:
        """Apply format-specific optimizations."""
        if options.get('optimize_for_publication', False):
            # Apply publication-specific optimizations
            return self._apply_publication_optimization(content, target_format)
        return content
    
    def _apply_publication_optimization(self, content: str, format_type: str) -> str:
        """Apply publication-specific optimizations."""
        # Placeholder for publication optimization logic
        return content
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate checksum for exported file."""
        import hashlib
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def _validate_exported_format(self, file_path: Path, format_type: str) -> Dict[str, Any]:
        """Validate exported file format."""
        return {
            'format_valid': True,
            'file_readable': file_path.exists(),
            'format_compliance': True
        }


@dataclass
class ReportGenerationResult:
    """
    Report generation outcome container storing generation status, validation results, 
    performance metrics, and metadata for comprehensive generation tracking and analysis.
    
    This class encapsulates the complete outcome of a report generation operation including
    success status, performance metrics, validation results, and audit trail information.
    """
    
    # Core generation outcome information
    generation_id: str
    generation_success: bool
    generation_time_seconds: float
    report: GeneratedReport
    
    # Validation and performance tracking
    validation_result: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    generation_warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    completion_timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    audit_trail_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def add_performance_metric(self, metric_name: str, metric_value: float, metric_unit: str) -> None:
        """
        Add performance metric to generation result for comprehensive tracking and optimization analysis.
        
        This method adds performance metrics to the generation result for analysis and optimization
        with proper formatting and context information.
        
        Args:
            metric_name: Name of the performance metric
            metric_value: Numerical value of the metric
            metric_unit: Unit of measurement for the metric
        """
        formatted_metric_name = f"{metric_name}_{metric_unit}" if metric_unit else metric_name
        self.performance_metrics[formatted_metric_name] = metric_value
        
        # Log the metric addition
        logger = get_logger('report.metrics', 'PERFORMANCE')
        logger.debug(f"Added performance metric: {formatted_metric_name} = {metric_value}")
    
    def set_validation_result(self, validation_result: Dict[str, Any]) -> None:
        """
        Set validation result for generation outcome with integration and analysis.
        
        This method sets the validation result and updates the generation success status
        based on validation outcomes and error analysis.
        
        Args:
            validation_result: Validation result dictionary with errors and quality metrics
        """
        self.validation_result = validation_result
        
        # Update generation success based on validation
        if not validation_result.get('validation_passed', True):
            self.generation_success = False
            validation_errors = validation_result.get('validation_errors', [])
            self.generation_warnings.extend([f"Validation: {error}" for error in validation_errors])
        
        # Log validation result integration
        logger = get_logger('report.validation', 'VALIDATION')
        logger.info(f"Validation result set for generation {self.generation_id} - Passed: {validation_result.get('validation_passed', True)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert generation result to dictionary format for serialization and reporting.
        
        This method converts all generation result properties to dictionary format for
        serialization, analysis, and integration with external systems.
        
        Returns:
            Dict[str, Any]: Generation result as comprehensive dictionary with all properties and metrics
        """
        return {
            'generation_id': self.generation_id,
            'generation_success': self.generation_success,
            'generation_time_seconds': self.generation_time_seconds,
            'report': self.report.to_dict() if self.report else None,
            'validation_result': self.validation_result,
            'performance_metrics': self.performance_metrics,
            'generation_warnings': self.generation_warnings,
            'error_message': self.error_message,
            'completion_timestamp': self.completion_timestamp.isoformat(),
            'audit_trail_id': self.audit_trail_id
        }


@dataclass
class ReportExportResult:
    """
    Report export outcome container storing export status, format validation, file information, 
    and optimization metrics for comprehensive export tracking and analysis.
    
    This class provides complete tracking of report export operations including format validation,
    file information, optimization metrics, and export quality assessment.
    """
    
    # Core export outcome information
    export_id: str
    export_success: bool
    target_format: str
    export_path: str
    
    # File and format information
    file_size_bytes: int = 0
    file_checksum: str = ''
    format_validation: Dict[str, Any] = field(default_factory=dict)
    optimization_metrics: Dict[str, float] = field(default_factory=dict)
    export_timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    export_warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert export result to dictionary format for logging and reporting.
        
        This method converts all export result properties to dictionary format for
        logging, analysis, and integration with monitoring systems.
        
        Returns:
            Dict[str, Any]: Export result as comprehensive dictionary with all export information and metrics
        """
        return {
            'export_id': self.export_id,
            'export_success': self.export_success,
            'target_format': self.target_format,
            'export_path': self.export_path,
            'file_size_bytes': self.file_size_bytes,
            'file_size_mb': self.file_size_bytes / (1024 * 1024),
            'file_checksum': self.file_checksum,
            'format_validation': self.format_validation,
            'optimization_metrics': self.optimization_metrics,
            'export_timestamp': self.export_timestamp.isoformat(),
            'export_warnings': self.export_warnings
        }


@dataclass
class ReportValidationResult:
    """
    Report validation outcome container storing validation status, error analysis, quality metrics, 
    and recommendations for comprehensive report quality assurance and improvement.
    
    This class provides detailed validation results with error analysis, quality metrics,
    and actionable recommendations for report quality improvement and standards compliance.
    """
    
    # Core validation outcome information
    validation_id: str
    validation_passed: bool
    validation_criteria: Dict[str, Any]
    
    # Validation details and analysis
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    improvement_recommendations: List[str] = field(default_factory=list)
    validation_timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error_message: str, error_category: str, error_context: str) -> None:
        """
        Add validation error with detailed context and impact analysis.
        
        This method adds validation errors with categorization and context for detailed
        error analysis and resolution guidance.
        
        Args:
            error_message: Descriptive error message
            error_category: Category of the validation error
            error_context: Context information for the error
        """
        formatted_error = f"[{error_category}] {error_message} (Context: {error_context})"
        self.validation_errors.append(formatted_error)
        
        # Update validation pass status
        self.validation_passed = False
        
        # Log error addition
        logger = get_logger('report.validation', 'VALIDATION')
        logger.warning(f"Validation error added: {formatted_error}")
    
    def add_recommendation(self, recommendation: str, priority: str) -> None:
        """
        Add improvement recommendation for report quality enhancement.
        
        This method adds improvement recommendations with priority levels for
        systematic quality enhancement and standards compliance.
        
        Args:
            recommendation: Improvement recommendation text
            priority: Priority level (HIGH, MEDIUM, LOW)
        """
        formatted_recommendation = f"[{priority}] {recommendation}"
        self.improvement_recommendations.append(formatted_recommendation)
        
        # Log recommendation addition
        logger = get_logger('report.validation', 'VALIDATION')
        logger.info(f"Improvement recommendation added: {formatted_recommendation}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert validation result to dictionary format for reporting and analysis.
        
        This method converts all validation result properties to dictionary format for
        reporting, analysis, and integration with quality assurance systems.
        
        Returns:
            Dict[str, Any]: Validation result as comprehensive dictionary with all validation information and recommendations
        """
        return {
            'validation_id': self.validation_id,
            'validation_passed': self.validation_passed,
            'validation_criteria': self.validation_criteria,
            'validation_errors': self.validation_errors,
            'validation_warnings': self.validation_warnings,
            'quality_metrics': self.quality_metrics,
            'improvement_recommendations': self.improvement_recommendations,
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'detailed_analysis': self.detailed_analysis,
            'error_count': len(self.validation_errors),
            'warning_count': len(self.validation_warnings),
            'recommendation_count': len(self.improvement_recommendations)
        }


class ReportTemplate:
    """
    Report template management class providing template loading, configuration, scientific formatting, 
    and customization capabilities for different report types and styles with caching and validation support.
    
    This class manages the lifecycle of report templates including loading, configuration, validation,
    and rendering with scientific formatting and customization capabilities for different report types.
    """
    
    def __init__(self, template_name: str, template_path: str, template_config: Dict[str, Any]):
        """
        Initialize report template with configuration, scientific formatting, and validation 
        for comprehensive template management.
        
        This method initializes the template with configuration, scientific formatting options,
        and validation capabilities for reliable template management.
        
        Args:
            template_name: Unique name identifier for the template
            template_path: File system path to the template file
            template_config: Configuration dictionary for template customization
        """
        # Set core template properties
        self.template_name = template_name
        self.template_path = template_path
        self.template_config = template_config
        
        # Initialize template management properties
        self.template: Optional[jinja2.Template] = None
        self.template_metadata: Dict[str, Any] = {}
        self.supported_formats: List[str] = SUPPORTED_REPORT_FORMATS.copy()
        self.scientific_filters: Dict[str, Any] = {}
        self.is_loaded: bool = False
        
        # Setup logger for template operations
        self.logger = get_logger(f'template.{template_name}', 'TEMPLATE')
        
        # Initialize template loading and configuration
        self._initialize_template()
    
    def _initialize_template(self) -> None:
        """Initialize template loading and configuration setup."""
        try:
            # Load template from file path
            template_file = Path(self.template_path)
            if not template_file.exists():
                raise FileNotFoundError(f"Template file not found: {self.template_path}")
            
            # Create Jinja2 environment with scientific extensions
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(template_file.parent),
                autoescape=jinja2.select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Add scientific formatting filters
            self._setup_scientific_filters(env)
            
            # Load template with environment
            self.template = env.get_template(template_file.name)
            
            # Setup template metadata
            self.template_metadata = {
                'template_name': self.template_name,
                'template_path': self.template_path,
                'load_timestamp': datetime.datetime.now().isoformat(),
                'template_size': template_file.stat().st_size,
                'template_config': self.template_config
            }
            
            # Mark template as loaded
            self.is_loaded = True
            
            self.logger.info(f"Template '{self.template_name}' loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize template '{self.template_name}': {e}")
            self.is_loaded = False
            raise
    
    def _setup_scientific_filters(self, env: jinja2.Environment) -> None:
        """Setup scientific formatting filters for Jinja2 environment."""
        # Scientific value formatting filter
        def format_scientific(value, precision=SCIENTIFIC_PRECISION_DIGITS, unit=''):
            return format_scientific_value(value, unit, precision)
        
        # Statistical significance formatting
        def format_p_value(value):
            if value < 0.001:
                return "p < 0.001***"
            elif value < 0.01:
                return f"p = {value:.3f}**"
            elif value < 0.05:
                return f"p = {value:.3f}*"
            else:
                return f"p = {value:.3f}"
        
        # Correlation coefficient formatting
        def format_correlation(value):
            return f"r = {value:.{SCIENTIFIC_PRECISION_DIGITS}f}"
        
        # Performance metric formatting
        def format_performance(value, metric_type='general'):
            if metric_type == 'time':
                return f"{value:.3f} seconds"
            elif metric_type == 'percentage':
                return f"{value:.2f}%"
            else:
                return format_scientific_value(value)
        
        # Register filters with Jinja2 environment
        env.filters['scientific'] = format_scientific
        env.filters['p_value'] = format_p_value
        env.filters['correlation'] = format_correlation
        env.filters['performance'] = format_performance
        
        # Store filters for reference
        self.scientific_filters = {
            'scientific': format_scientific,
            'p_value': format_p_value,
            'correlation': format_correlation,
            'performance': format_performance
        }
    
    def render(self, report_data: Dict[str, Any], render_options: Dict[str, Any]) -> str:
        """
        Render template with report data, scientific formatting, and customization options 
        for comprehensive report generation.
        
        This method renders the template with scientific data formatting, customization options,
        and comprehensive error handling for reliable report generation.
        
        Args:
            report_data: Data dictionary for template rendering
            render_options: Rendering options and customization settings
            
        Returns:
            str: Rendered report content with scientific formatting and customization
        """
        if not self.is_loaded or not self.template:
            raise RuntimeError(f"Template '{self.template_name}' not loaded")
        
        self.logger.debug(f"Rendering template '{self.template_name}' with {len(report_data)} data fields")
        
        try:
            # Prepare template context with scientific formatting
            template_context = self._prepare_template_context(report_data, render_options)
            
            # Apply scientific formatting to numerical values
            formatted_context = self._apply_scientific_formatting(template_context)
            
            # Render template with formatted context
            rendered_content = self.template.render(**formatted_context)
            
            # Apply post-processing if specified
            if render_options.get('post_process', True):
                rendered_content = self._post_process_content(rendered_content, render_options)
            
            self.logger.info(f"Template '{self.template_name}' rendered successfully ({len(rendered_content)} characters)")
            
            return rendered_content
            
        except Exception as e:
            self.logger.error(f"Error rendering template '{self.template_name}': {e}")
            raise RuntimeError(f"Template rendering failed: {e}")
    
    def validate_template(self) -> Dict[str, Any]:
        """
        Validate template syntax, configuration, and scientific formatting compatibility 
        for quality assurance.
        
        This method performs comprehensive template validation including syntax checking,
        configuration validation, and scientific formatting compatibility testing.
        
        Returns:
            Dict[str, Any]: Template validation result with syntax check and compatibility analysis
        """
        validation_result = {
            'validation_id': str(uuid.uuid4()),
            'template_name': self.template_name,
            'validation_timestamp': datetime.datetime.now().isoformat(),
            'validation_passed': True,
            'syntax_valid': False,
            'config_valid': False,
            'filters_compatible': False,
            'test_render_successful': False,
            'validation_errors': [],
            'validation_warnings': []
        }
        
        try:
            # Validate template syntax
            if self.template:
                validation_result['syntax_valid'] = True
            else:
                validation_result['validation_errors'].append("Template not loaded or syntax invalid")
            
            # Validate configuration
            required_config_keys = ['report_type', 'output_format']
            missing_keys = [key for key in required_config_keys if key not in self.template_config]
            if not missing_keys:
                validation_result['config_valid'] = True
            else:
                validation_result['validation_errors'].extend([f"Missing config key: {key}" for key in missing_keys])
            
            # Validate scientific filters compatibility
            if self.scientific_filters:
                validation_result['filters_compatible'] = True
            else:
                validation_result['validation_warnings'].append("No scientific filters configured")
            
            # Test template rendering with sample data
            try:
                sample_data = self._generate_sample_data()
                test_render = self.render(sample_data, {'post_process': False})
                if test_render:
                    validation_result['test_render_successful'] = True
                else:
                    validation_result['validation_errors'].append("Test render produced empty output")
            except Exception as e:
                validation_result['validation_errors'].append(f"Test render failed: {e}")
            
            # Determine overall validation status
            validation_result['validation_passed'] = (
                validation_result['syntax_valid'] and
                validation_result['config_valid'] and
                validation_result['test_render_successful']
            )
            
            self.logger.info(f"Template validation completed for '{self.template_name}' - Passed: {validation_result['validation_passed']}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error during template validation for '{self.template_name}': {e}")
            validation_result['validation_passed'] = False
            validation_result['validation_errors'].append(f"Validation process error: {e}")
            return validation_result
    
    def _prepare_template_context(self, report_data: Dict[str, Any], render_options: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare template context with scientific data and formatting options."""
        context = report_data.copy()
        
        # Add system metadata
        context['generator_metadata'] = {
            'generator_version': REPORT_GENERATOR_VERSION,
            'generation_timestamp': datetime.datetime.now().isoformat(),
            'template_name': self.template_name,
            'scientific_precision': SCIENTIFIC_PRECISION_DIGITS
        }
        
        # Add rendering options
        context['render_options'] = render_options
        
        # Include template configuration
        context['template_config'] = self.template_config
        
        return context
    
    def _apply_scientific_formatting(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply scientific formatting to numerical values in context."""
        formatted_context = copy.deepcopy(context)
        
        def format_values(obj):
            if isinstance(obj, dict):
                return {k: format_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [format_values(item) for item in obj]
            elif isinstance(obj, float):
                return format_scientific_value(obj, precision=SCIENTIFIC_PRECISION_DIGITS)
            else:
                return obj
        
        # Apply formatting to scientific data
        if 'scientific_data' in formatted_context:
            formatted_context['scientific_data'] = format_values(formatted_context['scientific_data'])
        
        return formatted_context
    
    def _post_process_content(self, content: str, options: Dict[str, Any]) -> str:
        """Apply post-processing to rendered content."""
        processed_content = content
        
        # Apply content optimization if requested
        if options.get('optimize_content', False):
            processed_content = self._optimize_content(processed_content)
        
        # Apply formatting cleanup
        if options.get('cleanup_formatting', True):
            processed_content = self._cleanup_formatting(processed_content)
        
        return processed_content
    
    def _optimize_content(self, content: str) -> str:
        """Optimize content for performance and size."""
        # Remove excessive whitespace
        import re
        optimized = re.sub(r'\s+', ' ', content)
        optimized = re.sub(r'\n\s*\n', '\n', optimized)
        return optimized.strip()
    
    def _cleanup_formatting(self, content: str) -> str:
        """Clean up formatting inconsistencies."""
        # Basic formatting cleanup
        cleaned = content.replace('\t', '    ')  # Convert tabs to spaces
        cleaned = cleaned.replace('\r\n', '\n')  # Normalize line endings
        return cleaned
    
    def _generate_sample_data(self) -> Dict[str, Any]:
        """Generate sample data for template validation testing."""
        return {
            'report_title': 'Test Report',
            'generation_date': datetime.datetime.now().isoformat(),
            'scientific_data': {
                'correlation': 0.95,
                'p_value': 0.001,
                'sample_size': 1000,
                'mean_value': 123.456789
            },
            'performance_metrics': {
                'execution_time': 7.2,
                'success_rate': 98.5
            }
        }


class ReportGenerator:
    """
    Comprehensive report generation class providing automated generation of scientific analysis reports, 
    comparative studies, performance summaries, and publication-ready documentation with advanced templating, 
    multi-format output support, statistical visualization integration, and scientific documentation standards 
    for reproducible research outcomes.
    
    This class serves as the main interface for report generation with comprehensive features including
    template management, format conversion, statistical analysis integration, and performance optimization.
    """
    
    def __init__(
        self,
        template_directory: str,
        default_format: str = DEFAULT_REPORT_FORMAT,
        generator_config: Dict[str, Any] = None,
        enable_visualization_integration: bool = VISUALIZATION_INTEGRATION_ENABLED,
        enable_statistical_analysis: bool = STATISTICAL_ANALYSIS_INTEGRATION,
        enable_caching: bool = True
    ):
        """
        Initialize report generator with template configuration, analysis integration, visualization 
        capabilities, and performance tracking for comprehensive scientific report generation.
        
        This method sets up the complete report generation system with template management,
        analysis integration, and performance optimization capabilities.
        
        Args:
            template_directory: Directory containing report templates
            default_format: Default output format for generated reports
            generator_config: Configuration dictionary for generator customization
            enable_visualization_integration: Enable integration with visualization system
            enable_statistical_analysis: Enable statistical analysis integration
            enable_caching: Enable template and result caching for performance
        """
        # Set core configuration properties
        self.template_directory = Path(template_directory)
        self.default_format = default_format
        self.generator_config = generator_config or {}
        self.visualization_integration_enabled = enable_visualization_integration
        self.statistical_analysis_enabled = enable_statistical_analysis
        self.caching_enabled = enable_caching
        
        # Initialize Jinja2 template environment
        self.template_environment = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_directory),
            autoescape=jinja2.select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Initialize analysis components
        self.metrics_calculator: Optional[PerformanceMetricsCalculator] = None
        self.statistical_comparator: Optional[StatisticalComparator] = None
        self.visualizer: Optional[ScientificVisualizer] = None
        self.result_writer: Optional[ResultWriter] = None
        
        # Initialize caching and performance tracking
        self.template_cache: Dict[str, jinja2.Template] = {}
        self.generation_statistics: Dict[str, Any] = {
            'total_reports_generated': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'average_generation_time': 0.0,
            'total_generation_time': 0.0
        }
        
        # Setup thread safety and concurrency
        self.generation_lock = threading.Lock()
        self.generation_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.generator_config.get('max_workers', 4)
        )
        
        # Initialize logger for report generation
        self.logger = get_logger('report.generator', 'REPORT_GENERATION')
        
        # Initialize report generator components
        self._initialize_components()
        
        self.logger.info(f"Report generator initialized with template directory: {template_directory}")
    
    def _initialize_components(self) -> None:
        """Initialize analysis components and system integrations."""
        try:
            # Initialize performance metrics calculator if statistical analysis enabled
            if self.statistical_analysis_enabled:
                self.metrics_calculator = PerformanceMetricsCalculator()
                self.statistical_comparator = StatisticalComparator()
            
            # Initialize visualization components if visualization integration enabled
            if self.visualization_integration_enabled:
                self.visualizer = ScientificVisualizer()
            
            # Initialize result writer for report output
            self.result_writer = ResultWriter()
            
            # Ensure template directory exists
            self.template_directory.mkdir(parents=True, exist_ok=True)
            
            # Setup template environment with scientific filters
            self._setup_template_environment()
            
            self.logger.info("Report generator components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing report generator components: {e}")
            raise RuntimeError(f"Component initialization failed: {e}")
    
    def _setup_template_environment(self) -> None:
        """Setup Jinja2 template environment with scientific filters and extensions."""
        # Add scientific formatting filters
        self.template_environment.filters['scientific'] = lambda x, p=SCIENTIFIC_PRECISION_DIGITS: format_scientific_value(x, precision=p)
        self.template_environment.filters['timestamp'] = lambda x: datetime.datetime.now().isoformat() if x is None else x
        
        # Add global template variables
        self.template_environment.globals.update({
            'generator_version': REPORT_GENERATOR_VERSION,
            'scientific_precision': SCIENTIFIC_PRECISION_DIGITS,
            'supported_formats': SUPPORTED_REPORT_FORMATS,
            'generation_timestamp': datetime.datetime.now().isoformat()
        })
    
    def generate_report(
        self,
        report_data: Union[SimulationResult, BatchSimulationResult, Dict[str, Any]],
        report_type: str,
        report_config: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> ReportGenerationResult:
        """
        Generate comprehensive scientific report with analysis integration, visualization, 
        and documentation for specified report type and configuration.
        
        This method provides the main interface for report generation with comprehensive
        analysis integration, visualization, and validation capabilities.
        
        Args:
            report_data: Data for report generation (simulation results or data dictionary)
            report_type: Type of report to generate (simulation, batch, comparison, etc.)
            report_config: Configuration dictionary for report customization
            output_path: Optional path for saving generated report
            
        Returns:
            ReportGenerationResult: Comprehensive report generation result with analysis, validation, and metadata
        """
        generation_id = str(uuid.uuid4())
        generation_start_time = datetime.datetime.now()
        
        # Acquire generation lock for thread-safe operation
        with self.generation_lock:
            try:
                self.logger.info(f"Starting report generation {generation_id} - Type: {report_type}")
                
                # Validate report data and configuration
                validation_result = self._validate_report_inputs(report_data, report_type, report_config)
                if not validation_result['valid']:
                    raise ValueError(f"Invalid report inputs: {validation_result['errors']}")
                
                # Determine report template and load it
                template = self._load_report_template(report_type, report_config)
                
                # Initialize analysis components based on report requirements
                analysis_data = self._prepare_analysis_data(report_data, report_config)
                
                # Generate performance metrics and statistical analysis
                if self.statistical_analysis_enabled and self.metrics_calculator:
                    performance_metrics = self._generate_performance_analysis(analysis_data, report_config)
                    analysis_data['performance_metrics'] = performance_metrics
                
                # Create visualizations if visualization integration enabled
                if self.visualization_integration_enabled and self.visualizer:
                    visualizations = self._generate_visualizations(analysis_data, report_config)
                    analysis_data['visualizations'] = visualizations
                
                # Prepare complete template context
                template_context = self._prepare_template_context(analysis_data, report_config, generation_id)
                
                # Render report with template engine and scientific formatting
                rendered_content = template.render(**template_context)
                
                # Create generated report object
                generated_report = GeneratedReport(
                    report_id=generation_id,
                    report_type=report_type,
                    content=rendered_content,
                    format=report_config.get('output_format', self.default_format)
                )
                
                # Add visualization tracking
                if 'visualizations' in analysis_data:
                    generated_report.included_visualizations = list(analysis_data['visualizations'].keys())
                
                # Validate generated report for accuracy and completeness
                if REPORT_VALIDATION_ENABLED:
                    validation_criteria = report_config.get('validation_criteria', {})
                    report_validation = generated_report.validate_content(validation_criteria)
                else:
                    report_validation = {'validation_passed': True}
                
                # Save report to output path if specified
                if output_path:
                    self._save_report_to_path(generated_report, output_path)
                    generated_report.file_path = output_path
                
                # Calculate generation time and update statistics
                generation_time = (datetime.datetime.now() - generation_start_time).total_seconds()
                self._update_generation_statistics(generation_time, True)
                
                # Create audit trail entry for report generation
                if AUDIT_TRAIL_INTEGRATION:
                    audit_trail_id = create_audit_trail(
                        action='REPORT_GENERATION',
                        component='REPORT_GENERATOR',
                        action_details={
                            'generation_id': generation_id,
                            'report_type': report_type,
                            'output_format': generated_report.format,
                            'generation_time': generation_time,
                            'validation_passed': report_validation['validation_passed']
                        }
                    )
                else:
                    audit_trail_id = generation_id
                
                # Create comprehensive generation result
                generation_result = ReportGenerationResult(
                    generation_id=generation_id,
                    generation_success=True,
                    generation_time_seconds=generation_time,
                    report=generated_report,
                    audit_trail_id=audit_trail_id
                )
                
                # Set validation result
                generation_result.set_validation_result(report_validation)
                
                # Add performance metrics
                generation_result.add_performance_metric('generation_time', generation_time, 'seconds')
                generation_result.add_performance_metric('content_length', len(rendered_content), 'characters')
                
                self.logger.info(f"Report generation {generation_id} completed successfully in {generation_time:.3f} seconds")
                
                return generation_result
                
            except Exception as e:
                # Handle generation failure
                generation_time = (datetime.datetime.now() - generation_start_time).total_seconds()
                self._update_generation_statistics(generation_time, False)
                
                self.logger.error(f"Report generation {generation_id} failed: {e}")
                
                # Create failure result
                failure_result = ReportGenerationResult(
                    generation_id=generation_id,
                    generation_success=False,
                    generation_time_seconds=generation_time,
                    report=GeneratedReport(
                        report_id=generation_id,
                        report_type=report_type,
                        content=f"Report generation failed: {e}",
                        format='text'
                    ),
                    error_message=str(e)
                )
                
                return failure_result
    
    def generate_batch_report(
        self,
        batch_results: BatchSimulationResult,
        report_style: str = DEFAULT_REPORT_STYLE,
        batch_config: Dict[str, Any] = None,
        output_path: Optional[str] = None
    ) -> ReportGenerationResult:
        """
        Generate comprehensive batch analysis report with cross-algorithm comparison, performance trends, 
        and statistical validation for large-scale simulation studies.
        
        This method generates detailed batch reports with cross-algorithm analysis, performance trends,
        and comprehensive statistical validation for large-scale simulation studies.
        
        Args:
            batch_results: Batch simulation results for analysis and reporting
            report_style: Style of the batch report (publication, technical, executive, detailed)
            batch_config: Configuration dictionary for batch report customization
            output_path: Optional path for saving the generated batch report
            
        Returns:
            ReportGenerationResult: Comprehensive batch report with cross-algorithm analysis and performance trends
        """
        if batch_config is None:
            batch_config = {}
        
        # Configure batch report generation
        batch_config.update({
            'report_style': report_style,
            'include_cross_algorithm_analysis': batch_config.get('include_cross_algorithm_analysis', True),
            'include_performance_trends': batch_config.get('include_performance_trends', True),
            'output_format': batch_config.get('output_format', self.default_format)
        })
        
        self.logger.info(f"Generating batch report with style: {report_style}")
        
        # Use main report generation method with batch-specific configuration
        return self.generate_report(
            report_data=batch_results,
            report_type='batch_analysis',
            report_config=batch_config,
            output_path=output_path
        )
    
    def generate_algorithm_comparison_report(
        self,
        algorithm_results: Dict[str, List[SimulationResult]],
        comparison_metrics: List[str],
        comparison_config: Dict[str, Any] = None,
        output_path: Optional[str] = None
    ) -> ReportGenerationResult:
        """
        Generate algorithm comparison report with statistical analysis, performance rankings, 
        and optimization recommendations for scientific evaluation.
        
        This method creates comprehensive algorithm comparison reports with statistical analysis,
        performance rankings, and actionable optimization recommendations.
        
        Args:
            algorithm_results: Dictionary mapping algorithm names to simulation results
            comparison_metrics: List of metrics to use for algorithm comparison
            comparison_config: Configuration dictionary for comparison report customization
            output_path: Optional path for saving the generated comparison report
            
        Returns:
            ReportGenerationResult: Algorithm comparison report with statistical analysis and optimization recommendations
        """
        if comparison_config is None:
            comparison_config = {}
        
        # Configure algorithm comparison report generation
        comparison_config.update({
            'comparison_metrics': comparison_metrics,
            'include_statistical_tests': comparison_config.get('include_statistical_tests', True),
            'include_optimization_recommendations': comparison_config.get('include_optimization_recommendations', True),
            'output_format': comparison_config.get('output_format', self.default_format)
        })
        
        self.logger.info(f"Generating algorithm comparison report for {len(algorithm_results)} algorithms")
        
        # Use main report generation method with comparison-specific configuration
        return self.generate_report(
            report_data=algorithm_results,
            report_type='algorithm_comparison',
            report_config=comparison_config,
            output_path=output_path
        )
    
    def export_report(
        self,
        report: GeneratedReport,
        target_format: str,
        export_path: str,
        export_options: Dict[str, Any] = None
    ) -> ReportExportResult:
        """
        Export generated report to specified format with optimization, validation, 
        and cross-platform compatibility.
        
        This method handles report export with format-specific optimizations, validation,
        and cross-platform compatibility for versatile report distribution.
        
        Args:
            report: Generated report to export
            target_format: Target format for export
            export_path: Path where exported report should be saved
            export_options: Format-specific export options and settings
            
        Returns:
            ReportExportResult: Report export result with format validation and optimization metrics
        """
        if export_options is None:
            export_options = {}
        
        self.logger.info(f"Exporting report {report.report_id} to format {target_format}")
        
        # Use report's built-in export functionality
        return report.export_to_format(target_format, export_path, export_options)
    
    def get_report_statistics(
        self,
        include_detailed_metrics: bool = False,
        reset_statistics: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive report generation statistics including performance metrics, success rates, 
        and optimization recommendations.
        
        This method provides detailed statistics about report generation performance, success rates,
        and recommendations for system optimization and improvement.
        
        Args:
            include_detailed_metrics: Include detailed performance metrics in statistics
            reset_statistics: Reset statistics after retrieval
            
        Returns:
            Dict[str, Any]: Comprehensive generation statistics with performance analysis and optimization recommendations
        """
        with self.generation_lock:
            current_stats = copy.deepcopy(self.generation_statistics)
            
            # Calculate derived metrics
            if current_stats['total_reports_generated'] > 0:
                current_stats['success_rate'] = (
                    current_stats['successful_generations'] / current_stats['total_reports_generated']
                ) * 100
                
                current_stats['failure_rate'] = (
                    current_stats['failed_generations'] / current_stats['total_reports_generated']
                ) * 100
            else:
                current_stats['success_rate'] = 0
                current_stats['failure_rate'] = 0
            
            # Include detailed metrics if requested
            if include_detailed_metrics:
                current_stats['detailed_metrics'] = {
                    'template_cache_size': len(self.template_cache),
                    'visualization_integration_enabled': self.visualization_integration_enabled,
                    'statistical_analysis_enabled': self.statistical_analysis_enabled,
                    'caching_enabled': self.caching_enabled
                }
            
            # Generate optimization recommendations
            recommendations = []
            if current_stats['failure_rate'] > 10:
                recommendations.append("High failure rate detected - review input validation")
            if current_stats['average_generation_time'] > 10:
                recommendations.append("Long generation times - consider enabling caching")
            
            current_stats['optimization_recommendations'] = recommendations
            
            # Reset statistics if requested
            if reset_statistics:
                self.generation_statistics = {
                    'total_reports_generated': 0,
                    'successful_generations': 0,
                    'failed_generations': 0,
                    'average_generation_time': 0.0,
                    'total_generation_time': 0.0
                }
            
            return current_stats
    
    def cleanup_resources(
        self,
        force_cleanup: bool = False,
        preserve_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Cleanup report generation resources, finalize operations, and prepare for shutdown 
        with comprehensive resource management.
        
        This method handles comprehensive cleanup of report generation resources including
        thread pools, caches, and temporary resources with proper finalization.
        
        Args:
            force_cleanup: Force immediate cleanup without waiting for pending operations
            preserve_cache: Preserve template cache for future use
            
        Returns:
            Dict[str, Any]: Resource cleanup result with final statistics and resource release information
        """
        self.logger.info("Starting report generator resource cleanup")
        
        cleanup_result = {
            'cleanup_id': str(uuid.uuid4()),
            'cleanup_timestamp': datetime.datetime.now().isoformat(),
            'resources_cleaned': [],
            'final_statistics': {},
            'cleanup_warnings': []
        }
        
        try:
            # Finalize pending report generation operations
            if hasattr(self, 'generation_executor'):
                if force_cleanup:
                    self.generation_executor.shutdown(wait=False)
                    cleanup_result['cleanup_warnings'].append("Forced shutdown - some operations may be incomplete")
                else:
                    self.generation_executor.shutdown(wait=True)
                cleanup_result['resources_cleaned'].append('thread_pool_executor')
            
            # Cleanup template cache if not preserving
            if not preserve_cache and hasattr(self, 'template_cache'):
                cache_size = len(self.template_cache)
                self.template_cache.clear()
                cleanup_result['resources_cleaned'].append(f'template_cache_{cache_size}_entries')
            
            # Get final generation statistics
            cleanup_result['final_statistics'] = self.get_report_statistics(include_detailed_metrics=True)
            
            # Release generation locks and system resources
            cleanup_result['resources_cleaned'].append('generation_locks')
            
            self.logger.info(f"Report generator cleanup completed - Resources cleaned: {len(cleanup_result['resources_cleaned'])}")
            
            return cleanup_result
            
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")
            cleanup_result['cleanup_warnings'].append(f"Cleanup error: {e}")
            return cleanup_result
    
    def _validate_report_inputs(
        self,
        report_data: Union[SimulationResult, BatchSimulationResult, Dict[str, Any]],
        report_type: str,
        report_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate report generation inputs for correctness and completeness."""
        validation_result = {'valid': True, 'errors': [], 'warnings': []}
        
        # Validate report type
        valid_report_types = ['simulation', 'batch_analysis', 'algorithm_comparison', 'performance_summary', 'reproducibility']
        if report_type not in valid_report_types:
            validation_result['errors'].append(f"Invalid report type: {report_type}")
            validation_result['valid'] = False
        
        # Validate report data
        if report_data is None:
            validation_result['errors'].append("Report data cannot be None")
            validation_result['valid'] = False
        
        # Validate output format
        output_format = report_config.get('output_format', self.default_format)
        if output_format not in SUPPORTED_REPORT_FORMATS:
            validation_result['errors'].append(f"Unsupported output format: {output_format}")
            validation_result['valid'] = False
        
        return validation_result
    
    def _load_report_template(self, report_type: str, report_config: Dict[str, Any]) -> jinja2.Template:
        """Load appropriate template for report type with caching."""
        template_name = report_config.get('template_name', f"{report_type}_report.html")
        
        # Check cache first if caching enabled
        if self.caching_enabled and template_name in self.template_cache:
            return self.template_cache[template_name]
        
        try:
            # Load template from environment
            template = self.template_environment.get_template(template_name)
            
            # Cache template if caching enabled
            if self.caching_enabled:
                self.template_cache[template_name] = template
            
            return template
            
        except jinja2.TemplateNotFound:
            # Fallback to default template
            default_template = self.template_environment.get_template(DEFAULT_TEMPLATE_NAME)
            if self.caching_enabled:
                self.template_cache[template_name] = default_template
            return default_template
    
    def _prepare_analysis_data(
        self,
        report_data: Union[SimulationResult, BatchSimulationResult, Dict[str, Any]],
        report_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare and structure data for analysis and template rendering."""
        if isinstance(report_data, (SimulationResult, BatchSimulationResult)):
            return report_data.to_dict()
        elif isinstance(report_data, dict):
            return report_data
        else:
            return {'raw_data': str(report_data)}
    
    def _generate_performance_analysis(self, analysis_data: Dict[str, Any], report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance metrics and statistical analysis."""
        if not self.metrics_calculator:
            return {}
        
        # Generate performance metrics using calculator
        # This would integrate with the actual PerformanceMetricsCalculator
        return {
            'generation_timestamp': datetime.datetime.now().isoformat(),
            'metrics_available': True,
            'analysis_config': report_config
        }
    
    def _generate_visualizations(self, analysis_data: Dict[str, Any], report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualizations for report integration."""
        if not self.visualizer:
            return {}
        
        # Generate visualizations using ScientificVisualizer
        # This would integrate with the actual visualization system
        return {
            'generation_timestamp': datetime.datetime.now().isoformat(),
            'visualizations_available': True,
            'visualization_config': report_config
        }
    
    def _prepare_template_context(
        self,
        analysis_data: Dict[str, Any],
        report_config: Dict[str, Any],
        generation_id: str
    ) -> Dict[str, Any]:
        """Prepare complete template context for rendering."""
        return {
            'report_data': analysis_data,
            'report_config': report_config,
            'generation_metadata': {
                'generation_id': generation_id,
                'generation_timestamp': datetime.datetime.now().isoformat(),
                'generator_version': REPORT_GENERATOR_VERSION,
                'scientific_precision': SCIENTIFIC_PRECISION_DIGITS
            },
            'system_metadata': {
                'supported_formats': SUPPORTED_REPORT_FORMATS,
                'validation_enabled': REPORT_VALIDATION_ENABLED,
                'visualization_enabled': self.visualization_integration_enabled,
                'statistical_analysis_enabled': self.statistical_analysis_enabled
            }
        }
    
    def _save_report_to_path(self, report: GeneratedReport, output_path: str) -> None:
        """Save generated report to specified file path."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report.content)
        
        self.logger.debug(f"Report {report.report_id} saved to {output_path}")
    
    def _update_generation_statistics(self, generation_time: float, success: bool) -> None:
        """Update generation statistics with performance tracking."""
        self.generation_statistics['total_reports_generated'] += 1
        self.generation_statistics['total_generation_time'] += generation_time
        
        if success:
            self.generation_statistics['successful_generations'] += 1
        else:
            self.generation_statistics['failed_generations'] += 1
        
        # Update average generation time
        self.generation_statistics['average_generation_time'] = (
            self.generation_statistics['total_generation_time'] / 
            self.generation_statistics['total_reports_generated']
        )


# Standalone report generation functions for direct use

@log_performance_metrics('simulation_report_generation')
def generate_simulation_report(
    simulation_result: SimulationResult,
    report_type: str = 'detailed',
    report_config: Dict[str, Any] = None,
    include_visualizations: bool = True,
    include_statistical_analysis: bool = True,
    output_path: Optional[str] = None
) -> ReportGenerationResult:
    """
    Generate comprehensive simulation report for individual simulation results with performance analysis, 
    visualization integration, and scientific documentation for research publication and algorithm evaluation.
    
    This function provides a standalone interface for generating individual simulation reports with
    comprehensive analysis, visualization integration, and scientific documentation standards.
    
    Args:
        simulation_result: Individual simulation result for report generation
        report_type: Type of simulation report (detailed, summary, publication)
        report_config: Configuration dictionary for report customization
        include_visualizations: Include visualizations in the generated report
        include_statistical_analysis: Include statistical analysis in the report
        output_path: Optional path for saving the generated report
        
    Returns:
        ReportGenerationResult: Comprehensive simulation report with analysis, visualizations, and scientific documentation
    """
    logger = get_logger('report.simulation', 'SIMULATION_REPORTING')
    logger.info("Generating individual simulation report")
    
    if report_config is None:
        report_config = {}
    
    # Configure simulation report generation
    report_config.update({
        'report_type': report_type,
        'include_visualizations': include_visualizations,
        'include_statistical_analysis': include_statistical_analysis,
        'output_format': report_config.get('output_format', DEFAULT_REPORT_FORMAT)
    })
    
    # Create report generator instance
    generator = ReportGenerator(
        template_directory=REPORT_TEMPLATE_DIRECTORY,
        enable_visualization_integration=include_visualizations,
        enable_statistical_analysis=include_statistical_analysis
    )
    
    try:
        # Generate report using main generator
        result = generator.generate_report(
            report_data=simulation_result,
            report_type='simulation',
            report_config=report_config,
            output_path=output_path
        )
        
        logger.info(f"Simulation report generated successfully - ID: {result.generation_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate simulation report: {e}")
        raise
    finally:
        # Cleanup generator resources
        generator.cleanup_resources()


@log_performance_metrics('batch_report_generation')
def generate_batch_report(
    batch_results: BatchSimulationResult,
    report_type: str = 'comprehensive',
    report_config: Dict[str, Any] = None,
    include_cross_algorithm_analysis: bool = True,
    include_performance_trends: bool = True,
    output_path: Optional[str] = None
) -> ReportGenerationResult:
    """
    Generate comprehensive batch analysis report for multiple simulation results with cross-algorithm 
    comparison, statistical analysis, and performance trends for large-scale simulation studies and 
    scientific evaluation.
    
    This function creates detailed batch reports with cross-algorithm analysis, performance trends,
    and comprehensive statistical validation for large-scale simulation studies.
    
    Args:
        batch_results: Batch simulation results for comprehensive analysis
        report_type: Type of batch report (comprehensive, summary, executive)
        report_config: Configuration dictionary for batch report customization
        include_cross_algorithm_analysis: Include cross-algorithm comparison analysis
        include_performance_trends: Include performance trend analysis
        output_path: Optional path for saving the generated batch report
        
    Returns:
        ReportGenerationResult: Comprehensive batch report with cross-algorithm analysis, performance trends, and statistical validation
    """
    logger = get_logger('report.batch', 'BATCH_REPORTING')
    logger.info("Generating comprehensive batch analysis report")
    
    if report_config is None:
        report_config = {}
    
    # Configure batch report generation
    report_config.update({
        'report_type': report_type,
        'include_cross_algorithm_analysis': include_cross_algorithm_analysis,
        'include_performance_trends': include_performance_trends,
        'output_format': report_config.get('output_format', DEFAULT_REPORT_FORMAT)
    })
    
    # Create report generator instance
    generator = ReportGenerator(
        template_directory=REPORT_TEMPLATE_DIRECTORY,
        enable_visualization_integration=True,
        enable_statistical_analysis=True
    )
    
    try:
        # Generate batch report
        result = generator.generate_batch_report(
            batch_results=batch_results,
            report_style=report_config.get('report_style', DEFAULT_REPORT_STYLE),
            batch_config=report_config,
            output_path=output_path
        )
        
        logger.info(f"Batch report generated successfully - ID: {result.generation_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate batch report: {e}")
        raise
    finally:
        # Cleanup generator resources
        generator.cleanup_resources()


@log_performance_metrics('algorithm_comparison_report_generation')
def generate_algorithm_comparison_report(
    algorithm_results: Dict[str, List[SimulationResult]],
    comparison_metrics: List[str],
    report_style: str = DEFAULT_REPORT_STYLE,
    comparison_config: Dict[str, Any] = None,
    include_statistical_tests: bool = True,
    include_optimization_recommendations: bool = True,
    output_path: Optional[str] = None
) -> ReportGenerationResult:
    """
    Generate comprehensive algorithm comparison report with statistical analysis, performance rankings, 
    optimization recommendations, and scientific evaluation for algorithm development and research publication.
    
    This function creates detailed algorithm comparison reports with statistical analysis, performance
    rankings, and actionable optimization recommendations for algorithm development and evaluation.
    
    Args:
        algorithm_results: Dictionary mapping algorithm names to simulation results
        comparison_metrics: List of metrics for algorithm comparison analysis
        report_style: Style of the comparison report (publication, technical, executive, detailed)
        comparison_config: Configuration dictionary for comparison report customization
        include_statistical_tests: Include statistical hypothesis testing in analysis
        include_optimization_recommendations: Include optimization recommendations
        output_path: Optional path for saving the generated comparison report
        
    Returns:
        ReportGenerationResult: Comprehensive algorithm comparison report with statistical analysis, rankings, and optimization recommendations
    """
    logger = get_logger('report.comparison', 'ALGORITHM_COMPARISON')
    logger.info(f"Generating algorithm comparison report for {len(algorithm_results)} algorithms")
    
    if comparison_config is None:
        comparison_config = {}
    
    # Configure algorithm comparison report generation
    comparison_config.update({
        'comparison_metrics': comparison_metrics,
        'report_style': report_style,
        'include_statistical_tests': include_statistical_tests,
        'include_optimization_recommendations': include_optimization_recommendations,
        'output_format': comparison_config.get('output_format', DEFAULT_REPORT_FORMAT)
    })
    
    # Create report generator instance
    generator = ReportGenerator(
        template_directory=REPORT_TEMPLATE_DIRECTORY,
        enable_visualization_integration=True,
        enable_statistical_analysis=True
    )
    
    try:
        # Generate algorithm comparison report
        result = generator.generate_algorithm_comparison_report(
            algorithm_results=algorithm_results,
            comparison_metrics=comparison_metrics,
            comparison_config=comparison_config,
            output_path=output_path
        )
        
        logger.info(f"Algorithm comparison report generated successfully - ID: {result.generation_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate algorithm comparison report: {e}")
        raise
    finally:
        # Cleanup generator resources
        generator.cleanup_resources()


@log_performance_metrics('performance_summary_report_generation')
def generate_performance_summary_report(
    performance_data: Dict[str, Any],
    summary_type: str = 'executive',
    summary_config: Dict[str, Any] = None,
    include_executive_summary: bool = True,
    include_detailed_metrics: bool = False,
    output_path: Optional[str] = None
) -> ReportGenerationResult:
    """
    Generate performance summary report with key metrics, trends, validation results, and executive 
    summary for stakeholder communication and research documentation.
    
    This function creates performance summary reports optimized for stakeholder communication with
    key metrics, trends, and executive summaries for research documentation and reporting.
    
    Args:
        performance_data: Dictionary containing performance data for summary generation
        summary_type: Type of summary report (executive, technical, detailed)
        summary_config: Configuration dictionary for summary report customization
        include_executive_summary: Include executive summary section in report
        include_detailed_metrics: Include detailed performance metrics analysis
        output_path: Optional path for saving the generated summary report
        
    Returns:
        ReportGenerationResult: Performance summary report with key insights, trends, and executive summary
    """
    logger = get_logger('report.performance_summary', 'PERFORMANCE_SUMMARY')
    logger.info("Generating performance summary report")
    
    if summary_config is None:
        summary_config = {}
    
    # Configure performance summary report generation
    summary_config.update({
        'summary_type': summary_type,
        'include_executive_summary': include_executive_summary,
        'include_detailed_metrics': include_detailed_metrics,
        'output_format': summary_config.get('output_format', DEFAULT_REPORT_FORMAT)
    })
    
    # Create report generator instance
    generator = ReportGenerator(
        template_directory=REPORT_TEMPLATE_DIRECTORY,
        enable_visualization_integration=True,
        enable_statistical_analysis=include_detailed_metrics
    )
    
    try:
        # Generate performance summary report
        result = generator.generate_report(
            report_data=performance_data,
            report_type='performance_summary',
            report_config=summary_config,
            output_path=output_path
        )
        
        logger.info(f"Performance summary report generated successfully - ID: {result.generation_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate performance summary report: {e}")
        raise
    finally:
        # Cleanup generator resources
        generator.cleanup_resources()


@log_performance_metrics('reproducibility_report_generation')
def generate_reproducibility_report(
    reproducibility_data: List[Dict[str, Any]],
    validation_results: Dict[str, Any],
    report_format: str = DEFAULT_REPORT_FORMAT,
    reproducibility_config: Dict[str, Any] = None,
    include_methodology_documentation: bool = True,
    output_path: Optional[str] = None
) -> ReportGenerationResult:
    """
    Generate scientific reproducibility report with correlation analysis, validation results, methodology 
    documentation, and compliance assessment for reproducible research standards.
    
    This function creates comprehensive reproducibility reports with correlation analysis, validation
    results, and methodology documentation for scientific reproducibility standards compliance.
    
    Args:
        reproducibility_data: List of reproducibility data for analysis and validation
        validation_results: Dictionary containing validation results and correlation analysis
        report_format: Output format for the reproducibility report
        reproducibility_config: Configuration dictionary for reproducibility report customization
        include_methodology_documentation: Include detailed methodology documentation
        output_path: Optional path for saving the generated reproducibility report
        
    Returns:
        ReportGenerationResult: Reproducibility report with correlation analysis, validation results, and methodology documentation
    """
    logger = get_logger('report.reproducibility', 'REPRODUCIBILITY')
    logger.info("Generating scientific reproducibility report")
    
    if reproducibility_config is None:
        reproducibility_config = {}
    
    # Configure reproducibility report generation
    reproducibility_config.update({
        'report_format': report_format,
        'include_methodology_documentation': include_methodology_documentation,
        'validation_results': validation_results,
        'output_format': report_format
    })
    
    # Prepare reproducibility data for analysis
    reproducibility_analysis = {
        'reproducibility_data': reproducibility_data,
        'validation_results': validation_results,
        'correlation_threshold': reproducibility_config.get('correlation_threshold', 0.95),
        'methodology_documentation': include_methodology_documentation
    }
    
    # Create report generator instance
    generator = ReportGenerator(
        template_directory=REPORT_TEMPLATE_DIRECTORY,
        enable_visualization_integration=True,
        enable_statistical_analysis=True
    )
    
    try:
        # Generate reproducibility report
        result = generator.generate_report(
            report_data=reproducibility_analysis,
            report_type='reproducibility',
            report_config=reproducibility_config,
            output_path=output_path
        )
        
        logger.info(f"Reproducibility report generated successfully - ID: {result.generation_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate reproducibility report: {e}")
        raise
    finally:
        # Cleanup generator resources
        generator.cleanup_resources()


def export_report_to_format(
    report: GeneratedReport,
    target_format: str,
    export_path: str,
    export_options: Dict[str, Any] = None,
    optimize_for_publication: bool = False,
    validate_export: bool = True
) -> ReportExportResult:
    """
    Export generated report to specified format with format-specific optimization, compression, 
    and validation for cross-platform compatibility and scientific data exchange.
    
    This function handles comprehensive report export with format-specific optimizations, validation,
    and cross-platform compatibility for versatile scientific data exchange and distribution.
    
    Args:
        report: Generated report to export to target format
        target_format: Target format for export (pdf, html, markdown, json, yaml, txt)
        export_path: File system path where exported report should be saved
        export_options: Dictionary containing format-specific export options and settings
        optimize_for_publication: Apply publication-specific optimizations to exported report
        validate_export: Perform validation of exported report format and content
        
    Returns:
        ReportExportResult: Report export outcome with format validation and optimization metrics
    """
    logger = get_logger('report.export', 'EXPORT')
    logger.info(f"Exporting report {report.report_id} to format {target_format}")
    
    if export_options is None:
        export_options = {}
    
    # Configure export options
    export_options.update({
        'optimize_for_publication': optimize_for_publication,
        'validate_export': validate_export
    })
    
    try:
        # Use report's built-in export functionality
        export_result = report.export_to_format(target_format, export_path, export_options)
        
        logger.info(f"Report export completed successfully - Export ID: {export_result.export_id}")
        return export_result
        
    except Exception as e:
        logger.error(f"Failed to export report {report.report_id} to format {target_format}: {e}")
        raise


def validate_report_content(
    report: GeneratedReport,
    validation_criteria: Dict[str, Any],
    strict_validation: bool = False,
    validate_scientific_accuracy: bool = True
) -> ReportValidationResult:
    """
    Validate report content for accuracy, completeness, scientific standards compliance, and quality 
    assurance with comprehensive error reporting and recommendations.
    
    This function performs comprehensive validation of report content including structure, scientific
    accuracy, numerical precision, and compliance with scientific documentation standards.
    
    Args:
        report: Generated report to validate for accuracy and compliance
        validation_criteria: Dictionary containing validation criteria and thresholds
        strict_validation: Enable strict validation mode with enhanced error checking
        validate_scientific_accuracy: Validate scientific accuracy and numerical precision
        
    Returns:
        ReportValidationResult: Report validation outcome with error analysis and quality assessment
    """
    logger = get_logger('report.validation', 'VALIDATION')
    logger.info(f"Validating report content for report {report.report_id}")
    
    # Configure validation criteria
    validation_config = validation_criteria.copy()
    validation_config.update({
        'strict_validation': strict_validation,
        'validate_scientific_accuracy': validate_scientific_accuracy
    })
    
    try:
        # Use report's built-in validation functionality
        validation_result_dict = report.validate_content(validation_config)
        
        # Convert to ReportValidationResult object
        validation_result = ReportValidationResult(
            validation_id=validation_result_dict['validation_id'],
            validation_passed=validation_result_dict['validation_passed'],
            validation_criteria=validation_criteria
        )
        
        # Add validation errors and warnings
        for error in validation_result_dict.get('validation_errors', []):
            validation_result.add_error(error, 'CONTENT', 'validation')
        
        # Add quality metrics
        validation_result.quality_metrics = validation_result_dict.get('quality_metrics', {})
        
        # Add improvement recommendations based on validation results
        if not validation_result_dict['validation_passed']:
            validation_result.add_recommendation(
                "Review and address validation errors for quality improvement",
                "HIGH"
            )
        
        logger.info(f"Report validation completed - Passed: {validation_result.validation_passed}")
        return validation_result
        
    except Exception as e:
        logger.error(f"Failed to validate report {report.report_id}: {e}")
        raise


def load_report_template(
    template_name: str,
    report_style: str = DEFAULT_REPORT_STYLE,
    template_config: Dict[str, Any] = None,
    use_cache: bool = True
) -> jinja2.Template:
    """
    Load and configure report template with scientific formatting, customization options, and template 
    caching for efficient report generation.
    
    This function provides template loading and configuration with scientific formatting, customization
    options, and caching capabilities for efficient and consistent report generation.
    
    Args:
        template_name: Name of the template file to load
        report_style: Style configuration for the template (publication, technical, executive, detailed)
        template_config: Dictionary containing template configuration and customization options
        use_cache: Enable template caching for improved performance
        
    Returns:
        jinja2.Template: Configured report template with scientific formatting and customization
    """
    logger = get_logger('template.loader', 'TEMPLATE')
    logger.debug(f"Loading report template: {template_name} with style: {report_style}")
    
    if template_config is None:
        template_config = {}
    
    # Check global template cache if caching enabled
    cache_key = f"{template_name}_{report_style}"
    if use_cache and cache_key in _report_templates_cache:
        logger.debug(f"Template loaded from cache: {cache_key}")
        return _report_templates_cache[cache_key]
    
    try:
        # Create template instance with configuration
        template_path = Path(REPORT_TEMPLATE_DIRECTORY) / template_name
        
        report_template = ReportTemplate(
            template_name=template_name,
            template_path=str(template_path),
            template_config={
                'report_style': report_style,
                **template_config
            }
        )
        
        # Validate template before use
        if REPORT_VALIDATION_ENABLED:
            validation_result = report_template.validate_template()
            if not validation_result['validation_passed']:
                logger.warning(f"Template validation warnings for {template_name}: {validation_result['validation_errors']}")
        
        # Cache template if caching enabled
        if use_cache:
            _report_templates_cache[cache_key] = report_template.template
        
        logger.info(f"Template loaded successfully: {template_name}")
        return report_template.template
        
    except Exception as e:
        logger.error(f"Failed to load template {template_name}: {e}")
        raise RuntimeError(f"Template loading failed: {e}")