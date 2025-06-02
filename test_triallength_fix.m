% Test triallength fix
addpath('Code');

% Show the problem
bad_triallength = '3600  # Match Crimaldi duration';
fprintf('Bad triallength: "%s" (type: %s)\n', bad_triallength, class(bad_triallength));

% Try to use it
try
    x = zeros(bad_triallength, 1);
catch ME
    fprintf('Error with string triallength: %s\n', ME.message);
end

% Fix it
if ischar(bad_triallength) || isstring(bad_triallength)
    % Extract just the number part
    fixed_triallength = str2double(regexp(bad_triallength, '^\d+', 'match', 'once'));
    fprintf('Fixed triallength: %d (type: %s)\n', fixed_triallength, class(fixed_triallength));
end

% Test with fixed value
try
    x = zeros(fixed_triallength, 1);
    fprintf('Success! Can create array with fixed triallength\n');
catch ME
    fprintf('Still failed: %s\n', ME.message);
end

exit;
