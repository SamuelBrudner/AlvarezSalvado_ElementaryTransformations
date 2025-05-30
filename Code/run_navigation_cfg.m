function out = run_navigation_cfg(cfg)
%RUN_NAVIGATION_CFG  High-level wrapper around the navigation model.
%
%   OUT = RUN_NAVIGATION_CFG(CFG) decides which low-level function to call
%   based on the fields of CFG.
%
%   ── Recognised CFG fields ─────────────────────────────────────────────
%   • environment     – plume type name or the literal 'video'
%   • plotting        – 0/1 flag
%   • ntrials         – number of virtual flies
%   • bilateral       – (optional) true → use bilateral model
%   • outputDir       – (optional) folder where config_used.yaml is saved
%   • randomSeed      – (optional) RNG seed for reproducibility
%
%   When environment == 'video' you must ALSO provide either:
%
%       (a) plume_video   – path to video file (AVI / MOV / MP4 …)
%           px_per_mm     – spatial scale
%           frame_rate    – Hz
%
%           + *optional* use_streaming – true → process frames on-the-fly
%
%       (b) plume_metadata – YAML file understood by load_custom_plume
%
%   Extra helpers:
%   • triallength overrides the natural movie length (movie loops automatically
%     when extended).
%
%   Note  The streaming backend is only stubbed in this commit; set
%         cfg.use_streaming = false (or omit the key) until you replace the
%         placeholder at the bottom of this file.

% -------------------------------------------------------------------------
% 0. choose which model variant we will call
% -------------------------------------------------------------------------
model_fn = @navigation_model_vec;
if isfield(cfg,'bilateral') && cfg.bilateral
    model_fn = @Elifenavmodel_bilateral;
end

% guard-rail warning for truncated trials
if isfield(cfg,'environment') && strcmpi(cfg.environment,'video') && ...
        ~isfield(cfg,'triallength')
    warning('Trial will end exactly at movie length. Set cfg.triallength to change this.');
end

% -------------------------------------------------------------------------
% 1. persist a copy of the configuration used in this run
% -------------------------------------------------------------------------
if isfield(cfg,'outputDir') && ~isempty(cfg.outputDir)
    if ~exist(cfg.outputDir,'dir'), mkdir(cfg.outputDir); end
    cfgPath = fullfile(cfg.outputDir,'config_used.yaml');
    try
        if exist('yamlwrite','file') == 2
            yamlwrite(cfgPath,cfg);
        else
            fid = fopen(cfgPath,'w'); fwrite(fid,jsonencode(cfg)); fclose(fid);
        end
    catch ME
        warning('Could not save cfg copy: %s',ME.message);
    end
end

% -------------------------------------------------------------------------
% 2. deterministic randomness if requested
% -------------------------------------------------------------------------
if isfield(cfg,'randomSeed') && ~isempty(cfg.randomSeed)
    rng(cfg.randomSeed,'twister');
end

% -------------------------------------------------------------------------
% 3. branch on the type of plume requested
% -------------------------------------------------------------------------
if isfield(cfg,'plume_metadata')
    % ------------ YAML metadata (pre-processed plume) --------------------
    plume = load_custom_plume(cfg.plume_metadata);
    tl    = chooseTrialLength(cfg,size(plume.data,3));
    out   = model_fn(tl,'video',cfg.plotting,cfg.ntrials,plume,cfg);

elseif isfield(cfg,'plume_video')
    % ------------ raw video plume ----------------------------------------
    assert(all(isfield(cfg,{'px_per_mm','frame_rate'})), ...
        'px_per_mm and frame_rate are required for video plumes');

    % Auto-enable streaming on SLURM clusters when not specified
    if ~isfield(cfg,'use_streaming') && isSlurmCluster()
        if isfield(cfg,'bilateral') && cfg.bilateral
            cfg.use_streaming = false;
        else
            cfg.use_streaming = true;
        end
    end

    if isfield(cfg,'use_streaming') && cfg.use_streaming
        if isfield(cfg,'bilateral') && cfg.bilateral
            error('run_navigation_cfg:BilateralStreamingUnsupported', ...
                  'Bilateral model cannot be run in streaming mode');
        end
        % ~~~ streaming mode (placeholder implementation) ~~~
        vr = VideoReader(cfg.plume_video);
        tl = chooseTrialLength(cfg, floor(vr.Duration * vr.FrameRate));
        out = navigation_model_vec_stream( ...
                    tl,'video',cfg.plotting,cfg.ntrials,vr,cfg);
    else
        % ~~~ load full movie into RAM as before ~~~
        plume = load_plume_video(cfg.plume_video, ...
                                 cfg.px_per_mm,cfg.frame_rate);
        tl    = chooseTrialLength(cfg, size(plume.data,3));
        out   = model_fn(tl,'video',cfg.plotting,cfg.ntrials,plume,cfg);
    end

else
    % ------------ analytical plume environments --------------------------
    assert(isfield(cfg,'triallength'), ...
        'cfg.triallength must be specified for non-video environments');
    out = model_fn(cfg.triallength,cfg.environment, ...
                   cfg.plotting,cfg.ntrials,struct(),cfg);
end
end
% ===== end main function =================================================


% ──────────────────────────────────────────────────────────────────────────
function tl = chooseTrialLength(cfg, defaultTL)
% Return cfg.triallength if present, otherwise fall back to defaultTL.
    if isfield(cfg,'triallength') && ~isempty(cfg.triallength)
        tl = cfg.triallength;
    else
        tl = defaultTL;
    end
end

