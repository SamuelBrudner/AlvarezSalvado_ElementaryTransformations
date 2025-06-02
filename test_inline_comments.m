addpath('Code');

fprintf('Testing load_config with inline comments...\n');
cfg = load_config('configs/test_inline_comments.yaml');

fprintf('\nParsed values:\n');
fprintf('px_per_mm: %g (type: %s)\n', cfg.px_per_mm, class(cfg.px_per_mm));
fprintf('frame_rate: %g (type: %s)\n', cfg.frame_rate, class(cfg.frame_rate));
fprintf('ntrials: %g (type: %s)\n', cfg.ntrials, class(cfg.ntrials));
fprintf('triallength: %g (type: %s)\n', cfg.triallength, class(cfg.triallength));
fprintf('ws: %g (type: %s)\n', cfg.ws, class(cfg.ws));
fprintf('environment: %s (type: %s)\n', cfg.environment, class(cfg.environment));

% Verify numeric fields work
try
    x = zeros(cfg.triallength, cfg.ntrials);
    fprintf('\nSuccess! Created %dx%d array\n', size(x,1), size(x,2));
catch ME
    fprintf('\nError: %s\n', ME.message);
end

exit;
