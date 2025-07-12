function trajs = load_multi_trajectories(envKey, maxTraj, primaryFile)
%LOAD_MULTI_TRAJECTORIES  Collect up to N agent trajectories across result files.
%   TRAJS = load_multi_trajectories(ENVKEY, MAXTRAJ) searches the results/
%   directory for files named <envKey>_nav_results_*.mat and returns a cell
%   array of up to MAXTRAJ trajectories. Each trajectory is an [steps x 2]
%   matrix of (x_cm, y_cm) positions.
%
%   TRAJS = load_multi_trajectories(ENVKEY, MAXTRAJ, PRIMARYFILE) excludes
%   PRIMARYFILE from the search (useful when you already plotted it in a
%   separate panel).
%
%   This utility loads multiple agents per file if necessary, so it works
%   even when few result files are present.  It re-uses the trajectory
%   extraction logic from viz_single_trial (duplicated here to keep the
%   function self-contained).
%
%   Example
%   -------
%   trajs = load_multi_trajectories('smoke', 25);
%   for k = 1:numel(trajs)
%       plot(trajs{k}(:,1), trajs{k}(:,2)); hold on;
%   end

arguments
    envKey         (1,:) char
    maxTraj        (1,1) double {mustBePositive}
    primaryFile    (1,:) char = ''
end

trajs = {};

files = dir(fullfile('results', sprintf('%s*_nav_results_*.mat', envKey)));
if ~isempty(primaryFile)
    files = files(~strcmp({files.name}, primaryFile));
end

if isempty(files)
    warning('No result files found for environment "%s".', envKey);
    return;
end

for fIdx = 1:numel(files)
    if numel(trajs) >= maxTraj, break; end
    S = load(fullfile(files(fIdx).folder, files(fIdx).name), 'out');
    out = S.out;

    nAgents = infer_n_agents(out);

    for a = 1:nAgents
        pos = extract_pos_local(out, a);
        if ~isempty(pos)
            trajs{end+1} = pos; %#ok<AGROW>
        end
        if numel(trajs) >= maxTraj, break; end
    end
end

end  % main function

% ------------------------------------------------------------------------
function n = infer_n_agents(out)
%INFER_N_AGENTS  Best-effort guess of number of agents in result struct.
if isfield(out,'pos') && ndims(out.pos) == 3
    n = size(out.pos,3);
elseif isfield(out,'x') && ~isvector(out.x)
    n = size(out.x,2);
else
    n = 1;
end
end

% ------------------------------------------------------------------------
function pos = extract_pos_local(out, agentIdx)
% Local copy of viz_single_trial/extract_pos (avoids dependency).
if nargin < 2, agentIdx = 1; end

if isfield(out, 'pos') && ~isempty(out.pos)
    p = out.pos;
    switch ndims(p)
        case 3  % [steps x 2 x agents] OR [2 x steps x agents]
            if size(p,2) == 2             % [steps x 2 x agents]
                p = squeeze(p(:,:,agentIdx));         % [steps x 2]
            elseif size(p,1) == 2          % [2 x steps x agents]
                p = squeeze(p(:,:,agentIdx)).';       % [steps x 2]
            else
                p = squeeze(p(:,:,agentIdx));
            end
        case 2
            if size(p,2) == 2
                p = p;                    % [steps x 2]
            elseif size(p,1) == 2
                p = p.';                  % [steps x 2]
            end
        otherwise
            p = [];
    end
    pos = p;
elseif isfield(out,'x') && isfield(out,'y')
    x = out.x; y = out.y;
    if isvector(x)
        pos = [x(:) y(:)];
    else
        pos = [x(:,agentIdx) y(:,agentIdx)];
    end
else
    pos = [];
end
end
