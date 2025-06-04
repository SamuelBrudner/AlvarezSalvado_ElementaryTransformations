#!/bin/bash

# Find where the analysis decision is made and fix it
perl -i -pe '
# Fix the logic after user input
if (/^if \[\[.*RUN_ANALYSIS.*== "n" \]\]; then$/) {
    # This is where it checks for "n" to skip
    # Keep this section but make sure variables are set
    next;
}

# Make sure QUICK_ANALYSIS is properly set
if (/^elif \[\[.*RUN_ANALYSIS.*== "quick" \]\]; then$/) {
    $_ = "elif [[ \"\$RUN_ANALYSIS\" == \"quick\" ]]; then\n";
    $_ .= "    QUICK_ANALYSIS=1\n";
    $_ .= "    echo \"Will run quick analysis...\"\n";
}

# Fix the else clause
if (/^else$/ && $prev_line =~ /QUICK_ANALYSIS=1/) {
    $_ = "else\n";
    $_ .= "    QUICK_ANALYSIS=0\n";
    $_ .= "    echo \"Will run full analysis...\"\n";
}

$prev_line = $_;
' setup_smoke_plume_config.sh

# Now fix Step 4 to handle all three cases properly
perl -i -pe '
if (/^# Step 4: Run analysis based on user choice$/) {
    $_ = "# Step 4: Run analysis based on user choice\n";
    $_ .= "if [[ \"\$RUN_ANALYSIS\" == \"n\" ]]; then\n";
    $_ .= "    echo \"\"\n";
    $_ .= "    echo \"Step 4: Skipping analysis (using defaults)\"\n";
    $_ .= "    # Set default values\n";
    $_ .= "    width=1024\n";
    $_ .= "    height=1024\n";
    $_ .= "    frames=36000\n";
    $_ .= "    dataset=\"/dataset2\"\n";
    $_ .= "    data_min=0.0\n";
    $_ .= "    data_max=1.0\n";
    $_ .= "    data_mean=0.1\n";
    $_ .= "    data_std=0.1\n";
    $_ .= "    source_x_cm=0.0\n";
    $_ .= "    source_y_cm=0.0\n";
    $_ .= "    arena_width_cm=\$(awk \"BEGIN {printf \\\"%.1f\\\", 1024 * 0.15299877600979192 / 10}\")\n";
    $_ .= "    arena_height_cm=\$(awk \"BEGIN {printf \\\"%.1f\\\", 1024 * 0.15299877600979192 / 10}\")\n";
    $_ .= "    temporal_scale=4.0\n";
    $_ .= "    spatial_scale=0.207\n";
    $_ .= "    beta_suggestion=0.01\n";
    $_ .= "    normalized=1\n";
    $_ .= "else\n";
    # Skip to the original content
    while (<>) {
        last if /^    echo ""/;
    }
    $_ = "    echo \"\"\n";
}
' setup_smoke_plume_config.sh

echo "Analysis logic fixed!"
