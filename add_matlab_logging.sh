#!/bin/bash

# Find the MATLAB script section and add detailed logging
perl -i -pe '
# Add timestamp function at the beginning of MATLAB script
if (/^    % Change to project root to ensure all relative paths work$/) {
    print "    % Helper function for timestamped logging\n";
    print "    log_msg = @(msg) fprintf('"'"'[%s] %s\\n'"'"', datestr(now, '"'"'HH:MM:SS'"'"'), msg);\n";
    print "    \n";
    print "    log_msg('"'"'=== MATLAB Analysis Started ==='"'"');\n";
}

# Add logging after each major step
s/fprintf\('"'"'Changed to project root: %s\\n'"'"', pwd\);/log_msg(sprintf('"'"'Changed to project root: %s'"'"', pwd));/;

s/fprintf\('"'"'Added Code directory to path\\n'"'"'\);/log_msg('"'"'Added Code directory to path'"'"');/;

s/fprintf\('"'"'\\nReading smoke config from JSON\.\.\.\\n'"'"'\);/log_msg('"'"'Reading smoke config from JSON...'"'"');/;

s/fprintf\('"'"'Configuration:\\n'"'"'\);/log_msg('"'"'Configuration loaded successfully'"'"');/;

s/fprintf\('"'"'\\nChecking HDF5 file\.\.\.\\n'"'"'\);/log_msg('"'"'Checking HDF5 file accessibility...'"'"');/;

# Add timing for h5info
if (/info = h5info\(plume_file\);/) {
    print "        log_msg('"'"'Calling h5info - this may take time for large files...'"'"');\n";
    print "        h5_start = tic;\n";
    $_ = "        info = h5info(plume_file);\n";
    $_ .= "        log_msg(sprintf('"'"'h5info completed in %.1f seconds'"'"', toc(h5_start)));\n";
}

# Add timing for dataset info
if (/ds_info = h5info\(plume_file, dataset_name\);/) {
    print "        log_msg(sprintf('"'"'Reading dataset info: %s'"'"', dataset_name));\n";
    print "        ds_start = tic;\n";
    $_ = "        ds_info = h5info(plume_file, dataset_name);\n";
    $_ .= "        log_msg(sprintf('"'"'Dataset info retrieved in %.1f seconds'"'"', toc(ds_start)));\n";
}

# Add logging for frame sampling
s/fprintf\('"'"'\\nSampling frames\.\.\.\\n'"'"'\);/log_msg(sprintf('"'"'Starting frame sampling (%d frames from %d total)...'"'"', n_samples, n_frames));/;

# Add logging in the frame reading loop
if (/frame = h5read\(plume_file, dataset_name,/) {
    print "        if i == 1\n";
    print "            log_msg('"'"'Reading first frame - timing this operation...'"'"');\n";
    print "            frame_start = tic;\n";
    print "        end\n";
    print "        \n";
    $_ = "        frame = h5read(plume_file, dataset_name, ...\n";
    $_ .= "                       [1 1 sample_indices(i)], [inf inf 1]);\n";
    $_ .= "        \n";
    $_ .= "        if i == 1\n";
    $_ .= "            log_msg(sprintf('"'"'First frame read in %.1f seconds'"'"', toc(frame_start)));\n";
    $_ .= "        end\n";
}

# Add final success logging
if (/fprintf\('"'"'\\n✓ Analysis complete\\n'"'"'\);/) {
    print "    log_msg('"'"'Analysis completed successfully!'"'"');\n";
}

# Add logging to error handler
if (/fprintf\('"'"'\\n\\nERROR in MATLAB analysis:\\n'"'"'\);/) {
    print "    log_msg('"'"'ERROR occurred - see details below'"'"');\n";
}
' setup_smoke_plume_config.sh

echo "Logging added to MATLAB script!"
