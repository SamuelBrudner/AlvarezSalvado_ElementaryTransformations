% Test Crimaldi simulation with dimension fix
addpath('Code');

fprintf('Loading Crimaldi configuration...\n');
cfg = load_config('configs/batch_crimaldi.yaml');
cfg.bilateral = false;
cfg.randomSeed = 1;
cfg.ntrials = 1;
cfg.plotting = 0;

fprintf('Running navigation simulation...\n');
try
    out = run_navigation_cfg(cfg);
    fprintf('SUCCESS: Crimaldi simulation completed!\n');
    fprintf('Success rate: %.2f\n', out.successrate);
catch ME
    fprintf('FAILED: %s\n', ME.message);
    fprintf('Error ID: %s\n', ME.identifier);
    if ~isempty(ME.stack)
        fprintf('Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
end
exit
