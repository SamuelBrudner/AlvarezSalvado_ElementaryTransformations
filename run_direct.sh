#!/bin/bash
# run_direct.sh - Run MATLAB with explicit path setup

cd /home/snb6/Documents/AlvarezSalvado_ElementaryTransformations

matlab -nodisplay -nosplash -batch "try, addpath(genpath('Code')), run('test_simple.m'), catch ME, fprintf('Error: %s\n', ME.message), exit(1), end, exit(0);"
