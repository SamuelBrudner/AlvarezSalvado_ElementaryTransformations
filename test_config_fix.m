addpath('Code');

% Create a test config struct with problematic values
test_config = struct();
test_config.triallength = '3600  # Match Crimaldi duration';
test_config.ntrials = '10  # Number of trials';
test_config.px_per_mm = '6.536  # pixels per millimeter';
test_config.frame_rate = '60';
test_config.environment = 'video';  % Should remain string

fprintf('Before conversion:\n');
disp(test_config);

% Apply the fix from convert_numeric_fields
fields = fieldnames(test_config);
for i = 1:length(fields)
    if ischar(test_config.(fields{i})) || isstring(test_config.(fields{i}))
        str_val = char(test_config.(fields{i}));
        
        % Extract just the numeric part before any comment
        numeric_part = regexp(str_val, '^\s*(\d+\.?\d*)\s*', 'tokens', 'once');
        if ~isempty(numeric_part)
            num = str2double(numeric_part{1});
        else
            num = str2double(str_val);
        end
        
        if ~isnan(num)
            test_config.(fields{i}) = num;
        end
    end
end

fprintf('\nAfter conversion:\n');
disp(test_config);

% Verify the values work
try
    x = zeros(test_config.triallength, test_config.ntrials);
    fprintf('\nSuccess! Can create %dx%d array\n', size(x,1), size(x,2));
catch ME
    fprintf('\nError: %s\n', ME.message);
end

exit;
