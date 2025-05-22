function startup
% STARTUP  Configure MATLAB path for the Elementary-Transformations project.
%
% Called automatically when MATLAB starts in this folder.

rootDir = fileparts(mfilename('fullpath'));

% ── Code directory ─────────────────────────────────────────────
codeDir = fullfile(rootDir,'Code');
if isfolder(codeDir)
    addpath(codeDir);
end

% Include data import utilities
importDir = fullfile(codeDir,'import functions feb2017');
if isfolder(importDir)
    addpath(importDir);
end

% ── YAML-Matlab toolbox ───────────────────────────────────────
yamlDir = fullfile(rootDir,'external','yamlmatlab');
if isfolder(yamlDir)
    addpath(genpath(yamlDir));
end
end   % ← nothing may follow this line