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

arguments
    matFile (1,1) string {mustBeFile}
    cfgFile (1,1) string {mustBeFile}
end

% ----------------------------------------------------------------------
% Load data
% ----------------------------------------------------------------------
S   = load(matFile, 'out');
out = S.out;

cfg = jsondecode(fileread(cfgFile));
assert(isfield(cfg, 'plume_file'), 'Config lacks plume_file field.');
plumePath = cfg.plume_file;

% ----------------------------------------------------------------------
% Read a single plume frame (frame 1000)
% ----------------------------------------------------------------------
frameIdx = 1000;  % arbitrary snapshot
try
    img = h5read(plumePath, '/odor', [1 1 frameIdx], [Inf Inf 1]);
catch ME
    error('Failed to read plume frame from %s: %s', plumePath, ME.message);
end

% Transpose so (x,y) match MATLAB imagesc orientation
img = img';

% ----------------------------------------------------------------------
% Build physical coordinate axes (cm)
% ----------------------------------------------------------------------
[nRows,nCols] = size(img);
xs = linspace(cfg.arena.x_min, cfg.arena.x_max, nCols);
ys = linspace(cfg.arena.y_min, cfg.arena.y_max, nRows);

% ----------------------------------------------------------------------
% Plot plume frame in physical units
% ----------------------------------------------------------------------
figure('Name', sprintf('Trial overlay – %s', matFile), 'Color', 'w');
imagesc(xs, ys, img);
axis equal tight xy;
set(gca,'YDir','normal');  % ensure y increases upward (cm)
colormap hot;
hold on;

% Trajectory (white line) & start (green circle)
pos = squeeze(out.pos);      % [steps x 2 x trials] or [2 x steps] depending on version
if ndims(pos) == 2  % [2 x steps]
    pos = permute(pos, [2 1]);
    pos = reshape(pos, [], 2);
else
    pos = pos(:,:,1);  % first trial
end

plot(pos(:,1), pos(:,2), 'w-', 'LineWidth', 1.0);
plot(pos(1,1), pos(1,2), 'go', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(pos(end,1), pos(end,2), 'bx', 'MarkerSize', 8, 'LineWidth', 1.5);

title({'Smoke trial overlay', sprintf('File: %s', matFile)}, 'Interpreter', 'none');
colorbar;

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
