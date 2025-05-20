function out = run_navigation_cfg(cfg)
%RUN_NAVIGATION_CFG Run navigation model using configuration struct.
%   OUT = RUN_NAVIGATION_CFG(CFG) executes either the unilateral or
%   bilateral navigation model depending on the 'bilateral' field in CFG.

if isfield(cfg, 'bilateral') && cfg.bilateral
    out = Elifenavmodel_bilateral(cfg.triallength, cfg.environment, ...
        cfg.plotting, cfg.ntrials);
else
    out = navigation_model_vec(cfg.triallength, cfg.environment, ...
        cfg.plotting, cfg.ntrials);
end
end
