function viz_single_trial(matFile, cfgFile)
%VIZ_SINGLE_TRIAL  Quick visual sanity-check for one navigation trial.
%   viz_single_trial(MATFILE, CFGFILE) overlays the agent trajectory stored
%   in MATFILE (a nav_results_XXXX.mat file) on one odor plume frame taken
%   from the plume referenced in CFGFILE (a JSON plume config).  Basic
%   metrics such as the first odor concentration, minimum success distance
%   and the frame of first success are printed to the console.
%
%   This helper is pure diagnostic – it does not alter the result file.
%
%   Example
%   -------
%   viz_single_trial('results/smoke_nav_results_0000.mat', ...
%                    'configs/plumes/smoke_1a_backgroundsubtracted.json')
%
%   Notes
%   -----
%   • Uses MATLAB built-ins only (jsondecode, h5read, imagesc).
%   • The plume frame chosen is frame #1000 by default; edit below if you
%     prefer another snapshot.
%   • Axis units are pixels; this is for qualitative inspection, not
%     publication.
% -------------------------------------------------------------------------

% Optional arguments handling
if nargin < 2
    cfgFile = '';
end
if nargin < 1 || isempty(matFile)
    matFile = 'smoke'; % default environment keyword
end

% If the first argument is NOT a file but an environment keyword (e.g. "smoke" or "crimaldi"),
% automatically select the first matching result & config files.
if exist(matFile,'file') ~= 2
    envKey  = lower(matFile);
    % Pick first result file for that env
    candRes = dir(fullfile('results', sprintf('%s*_nav_results_*.mat', envKey)));
    assert(~isempty(candRes), 'No %s result files found in results/.', envKey);
    matFile = fullfile(candRes(1).folder, candRes(1).name);
    fprintf('[INFO] Auto-selected %s result file: %s\n', envKey, matFile);

    if nargin < 2 || isempty(cfgFile)
        candCfg = dir(fullfile('configs','plumes', sprintf('%s*.json', envKey)));
        assert(~isempty(candCfg), 'No %s plume config JSON found in configs/plumes/.', envKey);
        cfgFile = fullfile(candCfg(1).folder, candCfg(1).name);
        fprintf('[INFO] Auto-selected %s config file: %s\n', envKey, cfgFile);
    end
end

% After automatic detection above, ensure we have a config path; otherwise abort.
if isempty(cfgFile)
    error('Could not determine matching plume config JSON – please provide it explicitly.');
end

% Validate files exist
if exist(matFile,'file')~=2
    error('Result file not found: %s', matFile);
end
if exist(cfgFile,'file')~=2
    error('Config file not found: %s', cfgFile);
end

% ----------------------------------------------------------------------
% Load data
% ----------------------------------------------------------------------
S   = load(matFile, 'out');
out = S.out;

cfg = jsondecode(fileread(cfgFile));
if isfield(cfg,'plume_file')
    plumePath = cfg.plume_file;
elseif isfield(cfg,'data_path') && isfield(cfg.data_path,'path')
    plumePath = cfg.data_path.path;
else
    error('Could not determine plume file: config has no plume_file or data_path.path field.');
end

% ----------------------------------------------------------------------
% Read a single plume frame (frame 1000)
% ----------------------------------------------------------------------
frameIdx = 1000;  % arbitrary snapshot
% Determine HDF5 dataset name – must be present in config
assert(isfield(cfg,'data_path') && isfield(cfg.data_path,'dataset_name') && ~isempty(cfg.data_path.dataset_name), ...
    'Config is missing data_path.dataset_name; cannot locate plume dataset.');

dset = cfg.data_path.dataset_name;

% Attempt to read the specified dataset – fail loudly if it is absent
try
    img = h5read(plumePath, dset, [1 1 frameIdx], [Inf Inf 1]);
catch ME
    error('Failed to read dataset "%s" from %s: %s', dset, plumePath, ME.message);
end

% Transpose so (x,y) match MATLAB imagesc orientation
img = img';

% ----------------------------------------------------------------------
% Build physical coordinate axes (cm)
% ----------------------------------------------------------------------
[nRows,nCols] = size(img);
if isfield(cfg,'arena')
    xmin = cfg.arena.x_min; xmax = cfg.arena.x_max;
    ymin = cfg.arena.y_min; ymax = cfg.arena.y_max;
elseif isfield(cfg,'spatial') && isfield(cfg.spatial,'arena_bounds')
    ab = cfg.spatial.arena_bounds;
    xmin = ab.x_min; xmax = ab.x_max;
    ymin = ab.y_min; ymax = ab.y_max;
else
    error('Config lacks arena or spatial.arena_bounds needed for plotting.');
end
xs = linspace(xmin, xmax, nCols);
ys = linspace(ymax, ymin, nRows);  % flip so source (y=0) at top

% Compute base file name once (used later)
[~, baseName] = fileparts(matFile);

% ----------------------------------------------------------------------
% Create 1×2 tiled layout: (1) single trial, (2) 25-trial overlay
% ----------------------------------------------------------------------
fig = figure('Name', sprintf('Trial overlay – %s', matFile), 'Color', 'w');
tl  = tiledlayout(fig,1,2,'Padding','compact','TileSpacing','compact');

%% --- Panel 1: single trial ------------------------------------------------
nexttile;
imagesc(xs, ys, img);
axis equal tight xy;
set(gca,'YDir','normal');  % ensure y increases upward (cm)
colormap hot;
hold on;
% Plot source position (magenta star)
if isfield(cfg,'simulation') && isfield(cfg.simulation,'source_position')
    sx = cfg.simulation.source_position.x_cm;
    sy = cfg.simulation.source_position.y_cm;
    hSource = plot(sx, sy, 'm*', 'MarkerSize', 10, 'LineWidth', 1.5);
end

% Trajectory (white line) & start (green circle)
pos = extract_pos(out, 1);
% Ensure pos is [nSteps x 2]
if size(pos,2) ~= 2
    pos = pos.';
end

hTraj  = plot(pos(:,1), pos(:,2), 'w-', 'LineWidth', 1.0);
hStart = plot(pos(1,1), pos(1,2), 'go', 'MarkerSize', 8, 'LineWidth', 1.5);
hEnd   = plot(pos(end,1), pos(end,2), 'bx', 'MarkerSize', 8, 'LineWidth', 1.5);

% Add a simple legend for clarity
legend([hStart, hEnd, hTraj, hSource], {'Start', 'End', 'Trajectory', 'Source'}, ...
       'TextColor', 'b', 'Location', 'southoutside', 'Box', 'off');

title({'Smoke trial overlay', sprintf('File: %s', matFile)}, 'Interpreter', 'none');
colorbar;

%% --- Panel 2: multi-trial overlay (up to 25) -------------------------------
nexttile;
imagesc(xs, ys, img);
axis equal tight xy;
set(gca,'YDir','normal');
colormap hot;
hold on;
overlayAx = gca;  % store handle for positioning histograms

% Overlay initialization zone rectangle if available in config
if isfield(cfg,'simulation') && isfield(cfg.simulation,'agent_initialization')
    init = cfg.simulation.agent_initialization;
    x0 = init.x_range_cm(1);
    y0 = init.y_range_cm(1);
    w  = diff(init.x_range_cm);
    h  = diff(init.y_range_cm);
    rectangle('Position',[x0, y0, w, h], 'EdgeColor',[0 0.7 0], 'LineStyle','--', 'LineWidth',1.2);
end
% Plot source position (magenta star)
if isfield(cfg,'simulation') && isfield(cfg.simulation,'source_position')
    sx = cfg.simulation.source_position.x_cm;
    sy = cfg.simulation.source_position.y_cm;
    hSource = plot(sx, sy, 'm*', 'MarkerSize', 10, 'LineWidth', 1.5);
end

% Determine environment prefix (e.g., 'smoke' or 'crimaldi') from the file name
envPrefix = regexprep(baseName, '_nav_results.*','');

% Use utility to grab up to 25 trajectories across files (excluding the primary file)
trajs = load_multi_trajectories(envPrefix, 25, [baseName '.mat']);
numCases = numel(trajs);
if numCases == 0
    warning('No additional %s trajectories found – multi-trial overlay skipped.', envPrefix);
else
    colors = lines(numCases);
    for k = 1:numCases
        posK = trajs{k};
        plot(posK(:,1), posK(:,2), '-', 'Color', colors(k,:), 'LineWidth', 0.8);
    end
end

% ------------------------------------------------------------------
% Marginal histograms of start (green) and end (blue) locations
% ------------------------------------------------------------------
% Gather start/end positions from primary + overlay trajectories
startX = pos(1,1);  startY = pos(1,2);

% Bounds check counter
outOfBounds = 0;
endX   = pos(end,1); endY   = pos(end,2);
for kk = 1:numCases
    pk = trajs{kk};
    startX(end+1) = pk(1,1); %#ok<AGROW>
    startY(end+1) = pk(1,2);
    % bounds check
    if exist('init','var')
        if pk(1,1) < init.x_range_cm(1) || pk(1,1) > init.x_range_cm(2) || ...
           pk(1,2) < init.y_range_cm(1) || pk(1,2) > init.y_range_cm(2)
            outOfBounds = outOfBounds + 1;
        end
    end
    endX  (end+1) = pk(end,1); %#ok<AGROW>
    endY  (end+1) = pk(end,2);
end

if outOfBounds > 0
    fprintf('[WARN] %d of %d start positions fall outside the configured init zone.\n', ...
        outOfBounds, numCases+1);
end

% Define histogram axes sizes relative to overlay axis
ovPos = get(overlayAx,'Position');
margin = 0.02;
heightFrac = 0.18; % relative size
widthFrac  = 0.18;

% Top histogram (X positions)
axTop = axes('Position', [ovPos(1), ovPos(2)+ovPos(4)+margin, ovPos(3), heightFrac*ovPos(4)], ...
             'Box','off');
edgesX = linspace(xmin, xmax, 20);
histogram(axTop, startX, edgesX, 'FaceColor','g', 'EdgeColor','none', 'FaceAlpha',0.5);
hold(axTop,'on');
histogram(axTop, endX,   edgesX, 'FaceColor','b', 'EdgeColor','none', 'FaceAlpha',0.5);
set(axTop,'XTick',[]); yl = ylabel(axTop,'Count'); yl.FontSize = 8;
axis(axTop,'tight');

% Right histogram (Y positions)
axRight = axes('Position', [ovPos(1)+ovPos(3)+margin, ovPos(2), widthFrac*ovPos(3), ovPos(4)], ...
               'Box','off');
edgesY = linspace(ymin, ymax, 20);
histogram(axRight, startY, edgesY, 'Orientation','horizontal', ...
          'FaceColor','g', 'EdgeColor','none', 'FaceAlpha',0.5);
hold(axRight,'on');
histogram(axRight, endY,   edgesY, 'Orientation','horizontal', ...
          'FaceColor','b', 'EdgeColor','none', 'FaceAlpha',0.5);
set(axRight,'YTick',[]); xl = xlabel(axRight,'Count'); xl.FontSize = 8;
axis(axRight,'tight');

% Ensure overlay axis is topmost for interaction
uistack(overlayAx,'top');

title(sprintf('%d-trial overlay', numCases));

% ----------------------------------------------------------------------
% Save figure to file for headless runs (after both panels)
% ----------------------------------------------------------------------
[~, baseName] = fileparts(matFile);
outFigPath = fullfile('results', [baseName '_overlay_multi.png']);
try
    saveas(fig, outFigPath);
    fprintf('[INFO] Saved multi-trial figure to %s\n', outFigPath);
catch ME
    warning('Could not save figure: %s', ME.message);
end

% ----------------------------------------------------------------------
% Console diagnostics
% ----------------------------------------------------------------------
if isfield(out, 'odor')
    fprintf('First odor at nose    = %.4f\n', out.odor(1,1));
end

if isfield(out, 'success_dist')
    fprintf('Min success distance  = %.3f cm\n', min(out.success_dist));
end

if isfield(out, 'success')
    sIdx = find(out.success == 1, 1);
    if ~isempty(sIdx)
        fprintf('Success time frame    = %d\n', sIdx);
    else
        fprintf('No success recorded.\n');
    end
end

end

function mustBeFile(f)
if exist(f, 'file') ~= 2
    error('File %s does not exist.', f);
end
end

function pos = extract_pos(out, agentIdx)
%EXTRACT_POS Return [nSteps x 2] trajectory matrix for a given agent (default = 1).
if nargin < 2 || isempty(agentIdx), agentIdx = 1; end

if isfield(out, 'pos') && ~isempty(out.pos)
    p = out.pos;
    switch ndims(p)
        case 3  % likely [steps x 2 x agents] OR [2 x steps x agents]
            if size(p,2) == 2             % [steps x 2 x agents]
                p = squeeze(p(:,:,agentIdx));         % [steps x 2]
            elseif size(p,1) == 2          % [2 x steps x agents]
                p = squeeze(p(:,:,agentIdx)).';       % [steps x 2]
            else
                p = squeeze(p(:,:,agentIdx));  % fallback
            end
        case 2
            if size(p,2) == 2              % [steps x 2]
                if agentIdx > 1 && size(p,3) >= agentIdx
                    p = squeeze(p(:,:,agentIdx));
                end
            elseif size(p,1) == 2          % [2 x steps]
                p = p.';
            end
        otherwise
            p = [];
    end
    pos = p;
elseif isfield(out,'x') && isfield(out,'y')
    x = out.x; y = out.y;
    if ~isempty(x)
        if isvector(x)
            pos = [x(:) y(:)];
        else
            pos = [x(:,agentIdx) y(:,agentIdx)];
        end
    else
        pos = [];
    end
else
    pos = [];
end
if isempty(pos)
    warning('Could not extract trajectory for agent %d', agentIdx);
    pos = nan(0,2);
end
end
