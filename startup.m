function startup
% STARTUP Configure MATLAB path for the Elementary Transformations project.
%
% This file adds the Code directory to the MATLAB path automatically when
% MATLAB starts in this project. It uses the location of this startup file
% so that it works regardless of the current working directory.

rootDir = fileparts(mfilename('fullpath'));
codeDir = fullfile(rootDir, 'Code');
if isfolder(codeDir)
    entries = strsplit(path, pathsep);
    if ~any(strcmp(entries, codeDir))
        addpath(codeDir);
    end
end
end

