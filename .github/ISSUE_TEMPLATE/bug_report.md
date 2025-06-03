---
name: Bug Report
description: Report a bug or issue in the plume navigation simulation system
title: "[BUG] Brief description of the issue"
labels: ["bug", "needs-triage", "investigation-required"]
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        ## Bug Report for Plume Navigation Simulation System
        
        Thank you for reporting a bug! Please fill out the sections below to help us understand and reproduce the issue. This information is crucial for maintaining the scientific reliability and reproducibility of the plume simulation system.
        
        **Please check existing issues before submitting to avoid duplicates.**

  - type: textarea
    id: bug_summary
    attributes:
      label: Bug Summary
      description: Provide a clear and concise summary of the bug
      placeholder: Briefly describe what went wrong...
      value: ""
    validations:
      required: true

  - type: dropdown
    id: bug_category
    attributes:
      label: Bug Category
      description: Which component or area of the system is affected?
      options:
        - Data Normalization Engine
        - Video Processing
        - Scale Calibration
        - Temporal Normalization
        - Intensity Calibration
        - Simulation Engine
        - Algorithm Implementation
        - Batch Processing
        - Parallel Processing
        - Analysis Pipeline
        - Performance Metrics
        - Statistical Analysis
        - Visualization
        - Configuration Management
        - Error Handling
        - File I/O Operations
        - Cross-Format Compatibility
        - Resource Management
        - Progress Tracking
        - Logging System
        - Installation/Setup
        - Documentation
        - Testing Framework
        - Other (specify in description)
    validations:
      required: true

  - type: dropdown
    id: bug_severity
    attributes:
      label: Bug Severity
      description: How severe is this bug's impact on your research workflow?
      options:
        - Critical - System unusable, blocking all research
        - High - Major functionality broken, significant research impact
        - Medium - Moderate impact, workaround available
        - Low - Minor issue, minimal research impact
    validations:
      required: true

  - type: textarea
    id: reproduction_steps
    attributes:
      label: Steps to Reproduce
      description: Provide detailed steps to reproduce the bug
      placeholder: |
        1. Go to...
        2. Click on...
        3. Enter data...
        4. See error...
      value: ""
    validations:
      required: true

  - type: textarea
    id: expected_behavior
    attributes:
      label: Expected Behavior
      description: Describe what you expected to happen
      placeholder: The system should have...
      value: ""
    validations:
      required: true

  - type: textarea
    id: actual_behavior
    attributes:
      label: Actual Behavior
      description: Describe what actually happened
      placeholder: Instead, the system...
      value: ""
    validations:
      required: true

  - type: textarea
    id: error_details
    attributes:
      label: Error Details
      description: Include any error messages, stack traces, or log output
      placeholder: |
        ```
        Paste error messages, stack traces, or relevant log output here
        ```
      value: ""
    validations:
      required: false

  - type: checkboxes
    id: affected_algorithms
    attributes:
      label: Affected Navigation Algorithms
      description: Which navigation algorithms are affected by this bug?
      options:
        - label: Infotaxis
          required: false
        - label: Casting
          required: false
        - label: Gradient Following
          required: false
        - label: Plume Tracking
          required: false
        - label: Hybrid Strategies
          required: false
        - label: Reference Implementation
          required: false
        - label: All Algorithms
          required: false
        - label: Algorithm-Independent Issue
          required: false

  - type: checkboxes
    id: affected_data_formats
    attributes:
      label: Affected Data Formats
      description: Which data formats are affected by this bug?
      options:
        - label: Crimaldi Dataset Format
          required: false
        - label: Custom AVI Recordings
          required: false
        - label: Standard Video Formats (MP4, MOV, MKV)
          required: false
        - label: HDF5 Data Files
          required: false
        - label: NumPy Arrays
          required: false
        - label: JSON Configuration Files
          required: false
        - label: CSV Data Files
          required: false
        - label: All Supported Formats
          required: false
        - label: Format-Independent Issue
          required: false

  - type: textarea
    id: system_environment
    attributes:
      label: System Environment
      description: Provide details about your system environment
      placeholder: |
        - Operating System: (e.g., Ubuntu 20.04, Windows 10, macOS 12.0)
        - Python Version: (e.g., 3.9.7)
        - Package Versions: (e.g., numpy 1.21.3, opencv 4.11.0)
        - Hardware: (e.g., CPU cores, RAM, GPU if applicable)
        - System Configuration: (e.g., virtual environment, conda, docker)
      value: ""
    validations:
      required: true

  - type: textarea
    id: scientific_context
    attributes:
      label: Scientific Context
      description: Provide context about how this bug affects your research
      placeholder: |
        - Research objective: (e.g., algorithm comparison, performance analysis)
        - Experimental setup: (e.g., batch size, simulation parameters)
        - Data characteristics: (e.g., plume complexity, arena size, temporal resolution)
        - Expected scientific outcomes: (e.g., statistical analysis, publication requirements)
        - Research timeline impact: (e.g., deadline constraints, experiment dependencies)
      value: ""
    validations:
      required: false

  - type: checkboxes
    id: performance_impact
    attributes:
      label: Performance Impact
      description: How does this bug affect system performance?
      options:
        - label: Simulation time exceeds 7.2 seconds average
          required: false
        - label: Batch processing cannot complete 4000 simulations in 8 hours
          required: false
        - label: Correlation with reference implementations <95%
          required: false
        - label: Memory usage exceeds 8GB limit
          required: false
        - label: Cross-format compatibility error rate >1%
          required: false
        - label: Reproducibility coefficient <0.99
          required: false
        - label: System becomes unresponsive
          required: false
        - label: No significant performance impact
          required: false

  - type: textarea
    id: data_sample
    attributes:
      label: Sample Data Information
      description: Provide information about the data that triggers this bug
      placeholder: |
        - Data source: (e.g., Crimaldi dataset, custom recording)
        - File size: (e.g., 100MB, 2GB)
        - Video properties: (e.g., resolution, frame rate, duration)
        - Arena characteristics: (e.g., size, shape, boundary conditions)
        - Plume properties: (e.g., intensity range, temporal dynamics)
        - Data quality: (e.g., noise level, missing frames, artifacts)
      value: ""
    validations:
      required: false

  - type: textarea
    id: configuration_details
    attributes:
      label: Configuration Details
      description: Provide relevant configuration information
      placeholder: |
        - Algorithm parameters: (e.g., convergence criteria, step size)
        - Normalization settings: (e.g., scale factors, intensity calibration)
        - Batch processing configuration: (e.g., parallel workers, chunk size)
        - Analysis parameters: (e.g., statistical methods, visualization options)
        - Custom configurations: (e.g., modified default values, experimental settings)
      value: ""
    validations:
      required: false

  - type: checkboxes
    id: reproducibility_status
    attributes:
      label: Reproducibility Status
      description: Can you consistently reproduce this bug?
      options:
        - label: Always reproducible with same data and configuration
          required: false
        - label: Intermittently reproducible with same setup
          required: false
        - label: Reproducible only with specific data or conditions
          required: false
        - label: Occurred once, cannot reproduce
          required: false
        - label: Reproducible across different systems/environments
          required: false
        - label: Platform-specific reproducibility
          required: false

  - type: textarea
    id: workaround
    attributes:
      label: Workaround
      description: Describe any workaround you have found
      placeholder: If you have found a way to work around this issue, please describe it here...
      value: ""
    validations:
      required: false

  - type: textarea
    id: impact_assessment
    attributes:
      label: Impact Assessment
      description: Assess the impact of this bug on your research and the broader scientific community
      placeholder: |
        - Research workflow impact: (e.g., delays, data loss, invalid results)
        - Scientific validity concerns: (e.g., accuracy, reproducibility, statistical significance)
        - Community impact: (e.g., affects multiple researchers, standard workflows)
        - Urgency level: (e.g., immediate fix needed, can wait for next release)
        - Alternative approaches: (e.g., other tools, manual processing)
      value: ""
    validations:
      required: false

  - type: checkboxes
    id: error_handling_behavior
    attributes:
      label: Error Handling Behavior
      description: How does the system handle this error?
      options:
        - label: System crashes with unhandled exception
          required: false
        - label: Graceful error message with recovery suggestions
          required: false
        - label: Silent failure with incorrect results
          required: false
        - label: Partial processing with degraded results
          required: false
        - label: Automatic retry attempts
          required: false
        - label: System hangs or becomes unresponsive
          required: false
        - label: Inconsistent error handling behavior
          required: false

  - type: textarea
    id: log_files
    attributes:
      label: Log Files and Diagnostic Information
      description: Attach or paste relevant log files, diagnostic output, or debugging information
      placeholder: |
        ```
        Paste relevant log entries, diagnostic output, or debugging information here.
        Include timestamps, error codes, and any system diagnostic data.
        ```
      value: ""
    validations:
      required: false

  - type: textarea
    id: additional_context
    attributes:
      label: Additional Context
      description: Add any other context, screenshots, or supporting materials
      placeholder: |
        - Screenshots or visualizations: (describe or attach)
        - Related issues: (GitHub issue numbers, similar problems)
        - Literature references: (relevant papers, algorithms, methodologies)
        - Community discussions: (forum posts, mailing list threads)
        - External dependencies: (specific library versions, system requirements)
        - Timing information: (when did this start happening, recent changes)
      value: ""
    validations:
      required: false

  - type: checkboxes
    id: testing_assistance
    attributes:
      label: Testing Assistance
      description: Are you willing to help test potential fixes?
      options:
        - label: I can test proposed fixes on my system
          required: false
        - label: I can provide additional data samples for testing
          required: false
        - label: I can help with debugging and diagnostics
          required: false
        - label: I can provide scientific domain expertise
          required: false
        - label: I can help validate fixes against scientific requirements
          required: false
        - label: I prefer to only report the issue
          required: false

  - type: checkboxes
    id: checklist
    attributes:
      label: Checklist
      description: Please confirm the following
      options:
        - label: I have searched existing issues to avoid duplicates
          required: true
        - label: I have provided sufficient detail to reproduce the issue
          required: true
        - label: I have included relevant error messages and logs
          required: true
        - label: I have described the scientific context and impact
          required: true
        - label: I am willing to provide additional information if needed
          required: false
---