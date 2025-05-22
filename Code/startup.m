function startup()
% STARTUP Initialize MATLAB environment

    % Add YAML toolbox
    yaml_dir = fullfile('..', 'external', 'yamlmatlab');
    if ~exist(yaml_dir, 'dir')
        error('YAML toolbox not found. Run: git submodule update --init');
    end
    addpath(genpath(yaml_dir));
    
    % Add Code directory to path
    addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'Code')));
    
    fprintf('MATLAB environment initialized.\n');
end
