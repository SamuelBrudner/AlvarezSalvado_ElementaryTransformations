function params = calculateJobParams(taskId, numConditions, agentsPerCondition, agentsPerJob)
%CALCULATEJOBPARAMS Calculate job parameters for a SLURM array task.
%   PARAMS = CALCULATEJOBPARAMS(TASKID, NUMCONDITIONS,
%   AGENTSPERCONDITION, AGENTSPERJOB) returns a struct with the fields
%   `conditionIndex`, `startAgent`, and `endAgent` describing which
%   condition and agents correspond to the given TASKID.
%
%   Example:
%       p = calculateJobParams(3, 2, 5, 2);
%       %% p.conditionIndex -> 1
%       %% p.startAgent    -> 3
%       %% p.endAgent      -> 4
%
%   See also RUN_BATCH_JOB.M

arguments
    taskId (1,1) {mustBeInteger, mustBeNonnegative}
    numConditions (1,1) {mustBePositive, mustBeInteger}
    agentsPerCondition (1,1) {mustBePositive, mustBeInteger}
    agentsPerJob (1,1) {mustBePositive, mustBeInteger}
end

conditionIndex = mod(taskId - 1, numConditions) + 1;
jobIndex = floor((taskId - 1) / numConditions);
startAgent = jobIndex * agentsPerJob + 1;
endAgent = min((jobIndex + 1) * agentsPerJob, agentsPerCondition);

params = struct('conditionIndex', conditionIndex, ...
                'startAgent', startAgent, ...
                'endAgent', endAgent);
end
