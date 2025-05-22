function tf = isSlurmCluster()
%ISSLURMCLUSTER  Detect SLURM cluster environment.
%   TF = ISSLURMCLUSTER() returns true if the SLURM_JOB_ID environment
%   variable is set, indicating execution under a SLURM scheduler.

tf = ~isempty(getenv('SLURM_JOB_ID'));
end
