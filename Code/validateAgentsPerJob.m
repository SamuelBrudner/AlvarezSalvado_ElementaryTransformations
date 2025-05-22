function validateAgentsPerJob(value)
%VALIDATEAGENTSPERJOB Ensure agents_per_job is a positive integer.
%   VALIDATEAGENTSPERJOB(VALUE) throws an error if VALUE is not a positive integer.

arguments
    value
end

if ~(isnumeric(value) && isscalar(value) && isfinite(value) && ...
        value > 0 && mod(value,1) == 0)
    error('common:InvalidAgentsPerJob', ...
        'agents_per_job must be a positive integer');
end
end
