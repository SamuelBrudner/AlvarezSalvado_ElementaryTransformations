function out = run_navigation_cfg(cfg)
%RUN_NAVIGATION_CFG Wrapper around navigation_model_vec with config struct.
%   OUT = RUN_NAVIGATION_CFG(CFG) runs the navigation model according to the
%   fields of CFG. If CFG contains a field `plume_video` or `plume_metadata`,
%   the corresponding video file is loaded using `load_plume_video` or
%   `load_custom_plume` (for YAML metadata) and the model is invoked with the
%   'video' environment.
%   Otherwise the environment specified in CFG.environment is used directly.
%
%   Required fields:
%       environment - plume type name or 'video'
%       plotting    - 0 or 1
%       ntrials     - number of trials
%
%   Additional fields for video plumes:
%       plume_video - path to AVI file
%       px_per_mm   - pixels per millimeter for the video
%       frame_rate  - frame rate (Hz)
%
%   Optional fields:
%       bilateral   - true to run the bilateral model
%       outputDir   - directory where config_used.yaml is written
%
%   When using video plumes, triallength is determined from the video unless
%   CFG specifies a triallength to override.

model_fn = @navigation_model_vec;
if isfield(cfg, 'environment') && strcmp(cfg.environment, 'video') && ...
        ~isfield(cfg, 'triallength') && ~isfield(cfg, 'loop')
    warning('Trial truncated to movie length; set cfg.loop=true to repeat.');
end
if isfield(cfg, 'bilateral') && cfg.bilateral
    model_fn = @Elifenavmodel_bilateral;
end

if isfield(cfg, 'outputDir')
    if ~exist(cfg.outputDir, 'dir')
        mkdir(cfg.outputDir);
    end
    cfgPath = fullfile(cfg.outputDir, 'config_used.yaml');
    if exist('yamlwrite', 'file') == 2
        yamlwrite(cfgPath, cfg);
    else
        fid = fopen(cfgPath, 'w');
        fwrite(fid, jsonencode(cfg));
        fclose(fid);
    end

end

if isfield(cfg, 'randomSeed')
    rng(cfg.randomSeed, 'twister');
end

if isfield(cfg, 'plume_metadata')
    plume = load_custom_plume(cfg.plume_metadata);
    if isfield(cfg, 'triallength')
        tl = cfg.triallength;
    else
        tl = size(plume.data, 3);
    end
    out = model_fn(tl, 'video', cfg.plotting, cfg.ntrials, plume, cfg);

elseif isfield(cfg, 'plume_video')
    if ~isfield(cfg, 'px_per_mm') || ~isfield(cfg, 'frame_rate')
        error('px_per_mm and frame_rate must be specified for video plumes');
    end
    plume = load_plume_video(cfg.plume_video, cfg.px_per_mm, cfg.frame_rate);
    if isfield(cfg, 'triallength')
        tl = cfg.triallength;
    else
        tl = size(plume.data, 3);
    end
    out = model_fn(tl, 'video', cfg.plotting, cfg.ntrials, plume, cfg);

else
    out = model_fn(cfg.triallength, cfg.environment, ...
        cfg.plotting, cfg.ntrials, struct(), cfg);
end
end
