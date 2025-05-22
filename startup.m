function startup
% STARTUP  Configure MATLAB path for the Elementary-Transformations project.
%
% Called automatically when MATLAB starts in this folder.

rootDir = fileparts(mfilename('fullpath'));

% ── Code directory ─────────────────────────────────────────────
codeDir = fullfile(rootDir, 'Code');
if isfolder(codeDir)
    addpath(genpath(codeDir));
    path(path); % remove potential duplicates
end

% ── YAML-Matlab toolbox ───────────────────────────────────────
yamlDir = fullfile(rootDir,'external','yamlmatlab');
if isfolder(yamlDir)
    addpath(genpath(yamlDir));
end
end   % ← nothing may follow this line