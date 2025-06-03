---
name: Pull Request
description: Contribution to the plume navigation simulation system with scientific computing validation
title: "[TYPE] Brief description of changes"
labels: ["contribution", "needs-review", "scientific-validation"]
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        ## Pull Request for Plume Navigation Simulation System
        
        Thank you for contributing to the **Plume Navigation Simulation Framework** - a scientific computing platform for evaluating navigation algorithms on plume data with cross-format compatibility and batch processing capabilities!
        
        This template ensures your contribution meets our scientific computing standards and quality requirements for reproducible research outcomes.
        
        **Please complete all relevant sections to facilitate efficient review and validation.**
        
        ### 🔬 Scientific Computing Standards
        - **Performance Target**: <7.2 seconds per simulation
        - **Accuracy Requirement**: >95% correlation with reference implementations
        - **Test Coverage**: >95% code coverage
        - **Reproducibility**: >99% consistency across platforms
        - **Error Rate**: <1% processing errors
        - **Batch Processing**: 4000+ simulations within 8 hours
        
        ### 📋 Key Project Features
        - Cross-format compatibility (Crimaldi dataset, custom AVI recordings)
        - Automated normalization and calibration across physical scales
        - Navigation algorithm testing (Infotaxis, Casting, Gradient Following, Hybrid Strategies)
        - Real-time analysis and statistical comparison framework
        - Parallel batch execution with configurable parameters
        
        **Before submitting, ensure you have read the coding standards and testing strategy documentation.**

  - type: textarea
    id: pull_request_summary
    attributes:
      label: Pull Request Summary
      description: Provide a clear and comprehensive summary of your changes
      placeholder: Briefly describe what this pull request accomplishes, the problem it solves, and the approach taken...
      value: ""
    validations:
      required: true

  - type: dropdown
    id: change_type
    attributes:
      label: Change Type
      description: What type of change does this pull request introduce?
      options:
        - Bug Fix - Corrects existing functionality without breaking changes
        - New Feature - Adds new functionality or capabilities
        - Performance Improvement - Optimizes speed, memory, or efficiency
        - Algorithm Implementation - New navigation algorithm or strategy
        - Documentation Update - Improves or adds documentation
        - Test Improvement - Enhances testing coverage or quality
        - Security Enhancement - Addresses security vulnerabilities or improvements
        - Refactoring - Code restructuring without functional changes
        - Dependency Update - Updates external dependencies or requirements
    validations:
      required: true

  - type: checkboxes
    id: affected_components
    attributes:
      label: Affected System Components
      description: Which components of the system are affected by this pull request?
      options:
        - label: Data Normalization Engine - Video processing and calibration
          required: false
        - label: Video Processing Pipeline - Format handling and conversion
          required: false
        - label: Scale and Intensity Calibration - Physical parameter normalization
          required: false
        - label: Temporal Normalization - Time-based data processing
          required: false
        - label: Simulation Engine - Core simulation execution framework
          required: false
        - label: Algorithm Implementation Framework - Navigation algorithm support
          required: false
        - label: Batch Processing System - Parallel execution and job management
          required: false
        - label: Parallel Processing Framework - Multi-core and distributed computing
          required: false
        - label: Analysis Pipeline - Performance metrics and statistical analysis
          required: false
        - label: Performance Metrics Calculation - Accuracy and efficiency measurement
          required: false
        - label: Statistical Analysis Framework - Scientific validation and comparison
          required: false
        - label: Visualization and Reporting - Result presentation and analysis
          required: false
        - label: Configuration Management - System settings and parameter handling
          required: false
        - label: Error Handling and Quality Assurance - Validation and reliability
          required: false
        - label: File I/O and Format Handling - Data input/output operations
          required: false
        - label: Command-Line Interface - User interaction and workflow management
          required: false
        - label: Monitoring and Logging - System observability and debugging
          required: false
        - label: Testing Framework - Quality assurance and validation testing
          required: false
        - label: Documentation - User guides, API docs, and developer resources
          required: false
        - label: Infrastructure - Deployment, CI/CD, and environment management
          required: false

  - type: textarea
    id: scientific_context
    attributes:
      label: Scientific Context and Research Impact
      description: Describe the scientific context and research impact of your changes
      placeholder: |
        **Research Motivation:**
        - What scientific problem does this address?
        - How does this advance plume navigation research?
        
        **Algorithm Context:**
        - What navigation strategies are affected?
        - Are there literature references or mathematical formulations?
        
        **Performance Implications:**
        - How does this impact simulation accuracy or speed?
        - What are the expected improvements or trade-offs?
        
        **Community Benefit:**
        - How will this benefit other researchers?
        - What new research capabilities does this enable?
      value: ""
    validations:
      required: false

  - type: checkboxes
    id: algorithm_impact
    attributes:
      label: Navigation Algorithm Impact
      description: Which navigation algorithms are affected by this pull request?
      options:
        - label: Infotaxis - Entropy-based information seeking navigation
          required: false
        - label: Casting - Bio-inspired crosswind casting behavior
          required: false
        - label: Gradient Following - Direct concentration gradient navigation
          required: false
        - label: Plume Tracking - Memory-based concentration tracking
          required: false
        - label: Hybrid Strategies - Combined algorithm approaches
          required: false
        - label: Reference Implementation - Benchmark algorithms
          required: false
        - label: Custom Algorithm - New algorithm implementation
          required: false
        - label: All Algorithms - Algorithm-agnostic changes
          required: false
        - label: No Algorithm Impact - Non-algorithm related changes
          required: false

  - type: checkboxes
    id: data_format_impact
    attributes:
      label: Data Format Impact
      description: Which data formats are affected by this pull request?
      options:
        - label: Crimaldi Dataset Format - Standard research dataset
          required: false
        - label: Custom AVI Recordings - User-generated video data
          required: false
        - label: Standard Video Formats (MP4, MOV, MKV) - Common video formats
          required: false
        - label: HDF5 Data Files - Scientific data storage
          required: false
        - label: NumPy Arrays - Numerical data arrays
          required: false
        - label: JSON Configuration Files - System configuration
          required: false
        - label: CSV Data Files - Tabular data export
          required: false
        - label: All Supported Formats - Format-agnostic changes
          required: false
        - label: No Format Impact - Non-data related changes
          required: false

  - type: textarea
    id: detailed_changes
    attributes:
      label: Detailed Description of Changes
      description: Provide a comprehensive description of the changes made
      placeholder: |
        **Implementation Details:**
        - What specific changes were made to the codebase?
        - How do the changes work technically?
        - What design patterns or approaches were used?
        
        **Code Structure:**
        - What new files were added or existing files modified?
        - How do the changes integrate with existing components?
        - Are there any architectural considerations?
        
        **Configuration Changes:**
        - Were any configuration files or parameters modified?
        - Are there new configuration options or requirements?
        - How do these changes affect system setup?
      value: ""
    validations:
      required: true

  - type: textarea
    id: testing_validation
    attributes:
      label: Testing and Validation
      description: Describe the testing performed to validate your changes
      placeholder: |
        **Test Coverage:**
        - What new tests were added?
        - What is the current test coverage percentage?
        - Are all critical paths tested?
        
        **Scientific Validation:**
        - How was scientific accuracy validated?
        - What correlation with reference implementations was achieved?
        - Were cross-format compatibility tests performed?
        
        **Performance Testing:**
        - What performance benchmarks were run?
        - How do simulation times compare to the <7.2s target?
        - Were batch processing tests performed?
        
        **Cross-Platform Testing:**
        - On which platforms were the changes tested?
        - Are there any platform-specific considerations?
      value: ""
    validations:
      required: true

  - type: checkboxes
    id: testing_categories
    attributes:
      label: Testing Categories Completed
      description: Which types of testing have been performed?
      options:
        - label: Unit Tests - Individual component testing with >95% coverage
          required: false
        - label: Integration Tests - Component interaction and workflow testing
          required: false
        - label: Performance Tests - Speed and memory usage validation
          required: false
        - label: Cross-Format Compatibility Tests - Multi-format data processing
          required: false
        - label: Scientific Accuracy Tests - >95% correlation validation
          required: false
        - label: Cross-Platform Tests - Linux, macOS, Windows compatibility
          required: false
        - label: Security Tests - Vulnerability and safety validation
          required: false
        - label: Regression Tests - Existing functionality preservation
          required: false

  - type: textarea
    id: performance_impact
    attributes:
      label: Performance Impact Analysis
      description: Analyze the performance impact of your changes
      placeholder: |
        **Performance Metrics:**
        
        | Metric | Before | After | Change | Target |
        |--------|--------|-------|--------|---------|
        | Simulation Time (avg) | | | | <7.2s |
        | Memory Usage (peak) | | | | <8GB |
        | Batch Throughput | | | | 4000 sims/8h |
        | Accuracy Correlation | | | | >95% |
        | Error Rate | | | | <1% |
        
        **Performance Analysis:**
        - Are there any performance improvements or regressions?
        - How does this affect batch processing efficiency?
        - What is the impact on memory usage and resource utilization?
        - Are there any scalability considerations?
      value: ""
    validations:
      required: false

  - type: checkboxes
    id: performance_validation
    attributes:
      label: Performance Validation Requirements
      description: Which performance requirements have been validated?
      options:
        - label: Simulation Speed - Average time <7.2 seconds per simulation
          required: false
        - label: Batch Processing - 4000+ simulations within 8 hours
          required: false
        - label: Scientific Accuracy - >95% correlation with reference implementations
          required: false
        - label: Memory Efficiency - Peak usage <8GB for standard workflows
          required: false
        - label: Error Rate - Processing error rate <1%
          required: false
        - label: Reproducibility - >99% consistency across environments
          required: false
        - label: Cross-Platform Performance - Consistent performance across platforms
          required: false
        - label: No Performance Impact - Changes do not affect performance
          required: false

  - type: dropdown
    id: breaking_changes
    attributes:
      label: Breaking Changes Assessment
      description: Does this pull request introduce any breaking changes?
      options:
        - No Breaking Changes - Fully backward compatible
        - Minor API Changes - Small modifications with deprecation warnings
        - Configuration Format Changes - Updates to configuration file structure
        - Data Format Changes - Modifications to input/output data formats
        - Algorithm Interface Changes - Changes to algorithm implementation interfaces
        - Performance Requirement Changes - Modified performance targets or constraints
        - Dependency Version Changes - Updated external library requirements
        - Major Breaking Changes - Significant API or workflow modifications
    validations:
      required: true

  - type: textarea
    id: breaking_changes_details
    attributes:
      label: Breaking Changes Details
      description: If breaking changes are introduced, provide detailed migration guidance
      placeholder: |
        **Breaking Changes Summary:**
        - What specific changes break backward compatibility?
        - Which APIs, configurations, or workflows are affected?
        
        **Migration Guide:**
        - What steps are required to migrate existing code?
        - Are there automated migration tools or scripts?
        - What is the timeline for deprecation?
        
        **Impact Assessment:**
        - How many users or use cases are affected?
        - What is the severity of the breaking changes?
        - Are there alternative approaches to minimize impact?
      value: ""
    validations:
      required: false

  - type: textarea
    id: documentation_updates
    attributes:
      label: Documentation Updates
      description: Describe any documentation changes or updates required
      placeholder: |
        **Documentation Changes:**
        - What documentation files were updated?
        - Are there new user guides or API documentation?
        - Were code examples and tutorials updated?
        
        **Required Documentation:**
        - What additional documentation is needed?
        - Are there new features that require user guides?
        - Do API changes need documentation updates?
        
        **Documentation Validation:**
        - Have all documentation changes been reviewed?
        - Are code examples tested and working?
        - Is the documentation clear and comprehensive?
      value: ""
    validations:
      required: false

  - type: checkboxes
    id: documentation_categories
    attributes:
      label: Documentation Categories Updated
      description: Which types of documentation have been updated?
      options:
        - label: API Documentation - Function and class documentation
          required: false
        - label: User Guides - Getting started and tutorial documentation
          required: false
        - label: Developer Guides - Contribution and development documentation
          required: false
        - label: Code Examples - Working code samples and demonstrations
          required: false
        - label: Configuration Documentation - Setup and configuration guides
          required: false
        - label: Algorithm Documentation - Navigation algorithm descriptions
          required: false
        - label: Performance Documentation - Benchmarking and optimization guides
          required: false
        - label: Troubleshooting Documentation - Error resolution and debugging
          required: false
        - label: No Documentation Changes - No documentation updates required
          required: false

  - type: textarea
    id: dependencies_changes
    attributes:
      label: Dependencies and Environment Changes
      description: Describe any changes to dependencies or environment requirements
      placeholder: |
        **Dependency Changes:**
        - Were any new dependencies added?
        - Were existing dependencies updated or removed?
        - Are there any version constraint changes?
        
        **Environment Requirements:**
        - Are there new system requirements?
        - Do Python version requirements change?
        - Are there new platform-specific dependencies?
        
        **Compatibility Impact:**
        - How do dependency changes affect existing installations?
        - Are there any compatibility issues with other packages?
        - Is a requirements.txt or setup.py update needed?
      value: ""
    validations:
      required: false

  - type: textarea
    id: security_considerations
    attributes:
      label: Security Considerations
      description: Describe any security implications or considerations
      placeholder: |
        **Security Analysis:**
        - Are there any security implications of the changes?
        - Were security scanning tools run (bandit, safety)?
        - Are there any new attack vectors or vulnerabilities?
        
        **Data Security:**
        - How is sensitive data handled?
        - Are there any data privacy considerations?
        - Are file permissions and access controls appropriate?
        
        **Dependency Security:**
        - Are all dependencies from trusted sources?
        - Were dependency vulnerabilities checked?
        - Are there any known security issues with new dependencies?
      value: ""
    validations:
      required: false

  - type: checkboxes
    id: quality_assurance_checklist
    attributes:
      label: Quality Assurance Checklist
      description: Confirm that quality assurance requirements have been met
      options:
        - label: Code follows PEP 8 style guidelines and passes Black formatting
          required: true
        - label: All functions have comprehensive type hints and pass MyPy validation
          required: true
        - label: Docstrings follow Google style with scientific context and examples
          required: true
        - label: Code passes all linting checks (flake8, isort, bandit)
          required: true
        - label: Test coverage is >95% and all tests pass
          required: true
        - label: Performance requirements are met (<7.2s simulation, >95% correlation)
          required: true
        - label: Cross-platform compatibility is preserved
          required: true
        - label: Scientific accuracy is validated against reference implementations
          required: true
        - label: Error handling is comprehensive and follows established patterns
          required: false
        - label: Security scanning passes without high-severity issues
          required: false
        - label: Documentation is complete and accurate
          required: false
        - label: Changes are backward compatible or migration path is provided
          required: false

  - type: textarea
    id: reviewer_guidance
    attributes:
      label: Reviewer Guidance
      description: Provide guidance for reviewers on how to test and validate your changes
      placeholder: |
        **Review Focus Areas:**
        - What aspects of the changes should reviewers focus on?
        - Are there specific algorithms or components that need careful review?
        - What are the most critical parts of the implementation?
        
        **Testing Instructions:**
        - How can reviewers test the changes locally?
        - Are there specific test commands or datasets to use?
        - What should reviewers look for during testing?
        
        **Validation Steps:**
        - What validation steps should reviewers perform?
        - Are there specific performance benchmarks to run?
        - How can scientific accuracy be verified?
        
        **Known Issues:**
        - Are there any known limitations or issues?
        - What edge cases should reviewers be aware of?
        - Are there any temporary workarounds in place?
      value: ""
    validations:
      required: false

  - type: checkboxes
    id: review_requirements
    attributes:
      label: Review Requirements
      description: What type of review is needed for this pull request?
      options:
        - label: Technical Review - Code architecture and implementation review
          required: false
        - label: Scientific Review - Algorithm correctness and accuracy validation
          required: false
        - label: Performance Review - Speed and efficiency impact assessment
          required: false
        - label: Security Review - Security implications and vulnerability assessment
          required: false
        - label: Documentation Review - Documentation quality and completeness
          required: false
        - label: Cross-Platform Review - Compatibility across different platforms
          required: false
        - label: Integration Review - Component interaction and workflow validation
          required: false
        - label: User Experience Review - CLI and workflow usability assessment
          required: false

  - type: textarea
    id: additional_context
    attributes:
      label: Additional Context and Notes
      description: Provide any additional context, notes, or information
      placeholder: |
        **Related Issues:**
        - GitHub issue numbers: #123, #456
        - Related pull requests or discussions
        
        **Future Work:**
        - What follow-up work is planned?
        - Are there any dependencies on other changes?
        - What improvements could be made in future iterations?
        
        **Acknowledgments:**
        - Credit to contributors, reviewers, or collaborators
        - References to research papers or external resources
        - Thanks to community members who provided feedback
        
        **Additional Notes:**
        - Any other relevant information for reviewers
        - Special considerations or constraints
        - Timeline or deadline information
      value: ""
    validations:
      required: false

  - type: checkboxes
    id: final_checklist
    attributes:
      label: Final Submission Checklist
      description: Confirm final requirements before submission
      options:
        - label: I have read and followed the coding standards and testing strategy documentation
          required: true
        - label: I have tested my changes thoroughly and all tests pass
          required: true
        - label: I have validated scientific accuracy and performance requirements
          required: true
        - label: I have updated documentation as needed
          required: true
        - label: I have considered the impact on other users and the research community
          required: true
        - label: I am willing to address review feedback and make necessary changes
          required: true
        - label: I understand this contribution will be reviewed for scientific accuracy
          required: true
        - label: I confirm this work is my own or properly attributed
          required: true
---