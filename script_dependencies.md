# Root Script Dependency Graph

This diagram lists the discovered dependencies among all scripts located in the repository root. Arrows indicate that a script directly references or calls another script. A dependency on `Code` means the script adds the `Code` directory to the MATLAB path or otherwise directly invokes functions stored there.

```mermaid
flowchart TD

    nav_job_paths_fixed_slurm --> Code
    complete_test_slurm --> test_both_plumes_complete_m
    complete_test_slurm --> generate_clean_configs_m
    run_simulation_slurm --> Code
    hpc_batch_submit_sh --> nav_job_smoke_slurm
    hpc_batch_submit_sh --> nav_job_crimaldi_slurm
    fix_wrapper_section_sh --> setup_smoke_plume_config_sh
    fix_wrapper_section_sh --> Code
    test_simple_m --> Code
    fix_matlab_cmd_sh --> setup_smoke_plume_config_sh
    nav_job_smoke_slurm --> Code
    fix_heredoc_sh --> setup_smoke_plume_config_sh
    get_plume_file_m --> setup_env_paths_sh
    run_plume_sim_sh --> hpc_batch_submit_sh
    run_plume_sim_sh --> nav_job_smoke_slurm
    run_plume_sim_sh --> validate_and_submit_plume_sh
    run_plume_sim_sh --> hpc_monitor_results_sh
    submit_range_sh --> Code
    inspect_smoke_hdf5_m --> setup_smoke_plume_config_sh
    debug_and_fix_sh --> setup_smoke_plume_config_sh
    debug_and_fix_sh --> create_new_wrapper_sh
    setup_smoke_plume_config_sh --> Code
    deploy_hpc_tools_sh --> nav_job_smoke_slurm
    deploy_hpc_tools_sh --> run_plume_sim_sh
    deploy_hpc_tools_sh --> nav_job_crimaldi_slurm
    deploy_hpc_tools_sh --> validate_and_submit_plume_sh
    deploy_hpc_tools_sh --> hpc_monitor_results_sh
    deploy_hpc_tools_sh --> cleanup_results_sh
    deploy_hpc_tools_sh --> Code
    generated_job_slurm --> Code
    diagnose_model_state_m --> get_plume_file_m
    diagnose_model_state_m --> Code
    run_direct_sh --> test_simple_m
    run_direct_sh --> Code
    plot_odor_distance_diagnostic_m --> test_both_plumes_complete_m
    nav_job_final_slurm --> Code
    add_startup_confirmation_sh --> setup_smoke_plume_config_sh
    test_wrapper_matlab_sh --> run_test_sh
    simple_fix_sh --> setup_smoke_plume_config_sh
    create_new_wrapper_sh --> setup_smoke_plume_config_sh
    nav_job_slurm_sh --> nav_job_smoke_slurm
    nav_job_slurm_sh --> Code
    add_matlab_logging_sh --> setup_smoke_plume_config_sh
    add_matlab_logging_sh --> Code
    slurm_submit_sh --> slurm_job_template_slurm
    unbuffer_output_sh --> setup_smoke_plume_config_sh
    fix_analysis_logic_sh --> setup_smoke_plume_config_sh
    run_matlab_safe_sh --> Code
    setup_complete_workflow_sh --> quick_validate_sh
    setup_complete_workflow_sh --> submit_range_sh
    setup_complete_workflow_sh --> view_results_py
    setup_complete_workflow_sh --> create_results_report_sh
    setup_complete_workflow_sh --> validate_and_submit_sh
    setup_complete_workflow_sh --> setup_validation_workflow_sh
    setup_complete_workflow_sh --> manage_sessions_sh
    setup_complete_workflow_sh --> Code
    run_nav_model_sh --> nav_job_final_slurm
    nav_job_crimaldi_slurm --> Code
    fix_timeout_issue_sh --> setup_smoke_plume_config_sh
    fix_timeout_issue_sh --> Code
    setup_hpc_scripts_sh --> hpc_batch_submit_sh
    setup_hpc_scripts_sh --> nav_job_smoke_slurm
    setup_hpc_scripts_sh --> run_plume_sim_sh
    setup_hpc_scripts_sh --> setup_smoke_plume_config_sh
    setup_hpc_scripts_sh --> nav_job_crimaldi_slurm
    setup_hpc_scripts_sh --> validate_and_submit_plume_sh
    setup_hpc_scripts_sh --> hpc_monitor_results_sh
    fix_quick_mode_sh --> setup_smoke_plume_config_sh
    setup_env_paths_sh --> get_plume_file_m
    setup_env_paths_sh --> startup_m
    setup_env_paths_sh --> Code
    smoke_simple_slurm --> Code
    validate_and_submit_sh --> Code
    cleanup_junk_files_sh --> plot_both_plumes_m
    cleanup_junk_files_sh --> generate_clean_configs_m
    cleanup_junk_files_sh --> Code
    fix_wrapper_syntax_sh --> setup_smoke_plume_config_sh
    validate_and_submit_plume_sh --> nav_job_smoke_slurm
    validate_and_submit_plume_sh --> nav_job_crimaldi_slurm
    validate_and_submit_plume_sh --> Code
    startup_m --> Code
    setup_validation_workflow_sh --> quick_validate_sh
    setup_validation_workflow_sh --> submit_range_sh
    setup_validation_workflow_sh --> validate_and_submit_sh
    setup_validation_workflow_sh --> manage_sessions_sh
    run_test_sh --> Code
    run_my_pipeline_sh --> generate_clean_configs_m
    run_my_pipeline_sh --> nav_job_crimaldi_slurm
    run_my_pipeline_sh --> nav_job_smoke_slurm
    run_my_pipeline_sh --> create_results_report_sh
    run_my_pipeline_sh --> run_plot_results_sh
    run_my_pipeline_sh --> Code

    analyze_plume_full_m
    analyze_results_m
    checkarena_m
    clean_quick_test_m
    get_hdf5_dimensions_sh
    matlab_results_check_sh
    plot_init_with_plumes_m
    plot_results_m
    plume_config_py
    quick_dimension_check_m
    quick_test_both_plumes_m
    run_plot_results_sh
    run_simulation_m
    setup_env_sh
    setup_plume_info_py
    test_config_loading_m
    test_matlab_basic_sh
    test_matlab_fixed_sh
    update_configs_shifted_init_m
    validate_output_py
    validate_plume_setup_m
```


## Script Categories

### One-Off Patch Scripts
- add_matlab_logging_sh
- add_startup_confirmation_sh
- debug_and_fix_sh
- fix_analysis_logic_sh
- fix_heredoc_sh
- fix_matlab_cmd_sh
- fix_quick_mode_sh
- fix_timeout_issue_sh
- fix_wrapper_section_sh
- fix_wrapper_syntax_sh
- simple_fix_sh

### Connected to run_plume_sim_sh
- cleanup_junk_files_sh
- cleanup_results_sh
- complete_test_slurm
- create_new_wrapper_sh
- create_results_report_sh
- deploy_hpc_tools_sh
- diagnose_model_state_m
- generate_clean_configs_m
- generated_job_slurm
- get_plume_file_m
- hpc_batch_submit_sh
- hpc_monitor_results_sh
- inspect_smoke_hdf5_m
- manage_sessions_sh
- nav_job_crimaldi_slurm
- nav_job_final_slurm
- nav_job_paths_fixed_slurm
- nav_job_slurm_sh
- nav_job_smoke_slurm
- plot_both_plumes_m
- plot_odor_distance_diagnostic_m
- quick_validate_sh
- run_direct_sh
- run_matlab_safe_sh
- run_nav_model_sh
- run_plume_sim_sh
- run_simulation_slurm
- run_test_sh
- setup_complete_workflow_sh
- setup_env_paths_sh
- setup_hpc_scripts_sh
- setup_smoke_plume_config_sh
- setup_validation_workflow_sh
- smoke_simple_slurm
- startup_m
- submit_range_sh
- test_both_plumes_complete_m
- test_simple_m
- test_wrapper_matlab_sh
- unbuffer_output_sh
- validate_and_submit_plume_sh
- validate_and_submit_sh
- view_results_py

### Independent Non-Patch Scripts
- analyze_plume_full_m
- analyze_results_m
- checkarena_m
- clean_quick_test_m
- get_hdf5_dimensions_sh
- matlab_results_check_sh
- plot_init_with_plumes_m
- plot_results_m
- plume_config_py
- quick_dimension_check_m
- quick_test_both_plumes_m
- run_plot_results_sh
- run_simulation_m
- setup_env_sh
- setup_plume_info_py
- slurm_job_template_slurm
- slurm_submit_sh
- test_config_loading_m
- test_matlab_basic_sh
- test_matlab_fixed_sh
- update_configs_shifted_init_m
- validate_output_py
- validate_plume_setup_m

