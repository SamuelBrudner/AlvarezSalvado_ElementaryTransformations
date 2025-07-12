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
if nargin < 1 || isempty(matFile)
    % auto-detect first Smoke result file
    cand = dir(fullfile('results','*smoke*_nav_results_*.mat'));
    assert(~isempty(cand), 'No smoke result files found in results/.');
    matFile = fullfile(cand(1).folder, cand(1).name);
    fprintf('[INFO] Auto-selected result file: %s\n', matFile);
end
if nargin < 2 || isempty(cfgFile)
    candCfg = dir(fullfile('configs','plumes','*smoke*.json'));
    assert(~isempty(candCfg), 'No smoke plume config JSON found in configs/plumes/.');
    cfgFile = fullfile(candCfg(1).folder, candCfg(1).name);
    fprintf('[INFO] Auto-selected config file: %s\n', cfgFile);
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
ys = linspace(ymin, ymax, nRows);

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

% Trajectory (white line) & start (green circle)
if isfield(out, 'pos')
    pos = out.pos;  % may be [steps x 2 x trials] or [2 x steps] or [steps x 2]
    if ndims(pos) == 3              % [steps x 2 x trials]
        pos = squeeze(pos(:,:,1));  % take first trial
    elseif size(pos,1) == 2 && size(pos,2) > 2  % [2 x steps]
        pos = pos.';               % transpose to [steps x 2]
    end
elseif isfield(out, 'x') && isfield(out, 'y')
    % navigation_model_vec stores trajectories separately as x (cm) and y (cm)
    x = out.x;  y = out.y;
    if isvector(x)
        pos = [x(:) y(:)];           % single trial vector
    else
        pos = [x(:,1) y(:,1)];       % first trial of many
    end
else
    error('Result struct lacks both "pos" and "x"/"y" fields – cannot plot trajectory.');
end
% Ensure pos is [nSteps x 2]
if size(pos,2) ~= 2
    pos = pos.';
end

hTraj  = plot(pos(:,1), pos(:,2), 'w-', 'LineWidth', 1.0);
hStart = plot(pos(1,1), pos(1,2), 'go', 'MarkerSize', 8, 'LineWidth', 1.5);
hEnd   = plot(pos(end,1), pos(end,2), 'bx', 'MarkerSize', 8, 'LineWidth', 1.5);

% Add a simple legend for clarity
legend([hStart, hEnd, hTraj], {'Start', 'End', 'Trajectory'}, ...
       'TextColor', 'w', 'Location', 'southoutside', 'Box', 'off');

title({'Smoke trial overlay', sprintf('File: %s', matFile)}, 'Interpreter', 'none');
colorbar;

%% --- Panel 2: multi-trial overlay (up to 25) -------------------------------
nexttile;
imagesc(xs, ys, img);
axis equal tight xy;
set(gca,'YDir','normal');
colormap hot;
hold on;

caseFiles = dir(fullfile('results','*smoke*nav_results_*.mat'));
numCases  = min(25, numel(caseFiles));
colors = lines(numCases);
for k = 1:numCases
    S = load(fullfile(caseFiles(k).folder, caseFiles(k).name), 'out');
    posK = extract_pos(S.out);
    plot(posK(:,1), posK(:,2), '-', 'Color', colors(k,:), 'LineWidth', 0.8);
end

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

function pos = extract_pos(out)
%EXTRACT_POS Return [nSteps x 2] trajectory matrix from result struct.
if isfield(out, 'pos')
    p = out.pos;
    if ndims(p) == 3, p = squeeze(p(:,:,1)); end
    if size(p,1)==2 && size(p,2) > 2, p = p.'; end
    pos = p;
elseif isfield(out,'x') && isfield(out,'y')
    if isvector(out.x)
        pos = [out.x(:) out.y(:)];
    else
        pos = [out.x(:,1) out.y(:,1)];
    end
else
    pos = nan(0,2);
end
end
