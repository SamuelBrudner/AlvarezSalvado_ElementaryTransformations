#!/bin/bash
# Template SLURM script for navigation model

matlab -batch "navigation_model_vec($TRIAL_LENGTH, '$ENVIRONMENT', 0, $AGENTS_PER_JOB)"
