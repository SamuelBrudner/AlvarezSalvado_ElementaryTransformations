function tl = chooseTrialLength(cfg, defaultTL)
%CHOOSETRIALLENGTH Return cfg.triallength if present, otherwise default.
%
%   TL = CHOOSETRIALLENGTH(CFG, DEFAULTTL) returns CFG.triallength when
%   it exists and is non-empty; otherwise returns DEFAULTTL.

if isfield(cfg,'triallength') && ~isempty(cfg.triallength)
    tl = cfg.triallength;
else
    tl = defaultTL;
end
end
