function totalJobs = calculateTotalJobs(numConditions, agentsPerCondition, agentsPerJob)
%CALCULATETOTALJOBS Compute total number of SLURM array jobs required.
%   TOTALJOBS = CALCULATETOTALJOBS(NUMCONDITIONS, AGENTSPERCONDITION, AGENTSPERJOB)
%   returns the total number of array tasks needed to simulate all agents across
%   all conditions when each task handles AGENTSPERJOB agents.
%
%   Example:
%       total = calculateTotalJobs(2, 5, 2);  % returns 6
%
arguments
    numConditions (1,1) {mustBePositive, mustBeInteger}
    agentsPerCondition (1,1) {mustBePositive, mustBeInteger}
    agentsPerJob (1,1) {mustBePositive, mustBeInteger}
end

jobsPerCondition = ceil(agentsPerCondition / agentsPerJob);
totalJobs = numConditions * jobsPerCondition;
end
