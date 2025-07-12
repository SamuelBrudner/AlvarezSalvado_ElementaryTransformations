%% test_smoke_odor_sampling.m
% Quick unit-test to ensure odor sampling returns positive values
% for points that lie well inside the Smoke plume.  Run from the
% repository root or add Code/ to the MATLAB path.
%---------------------------------------------------------------------------

cfgFile = fullfile('configs','plumes','smoke_1a_backgroundsubtracted.json');
assert(isfile(cfgFile), 'Smoke config not found: %s', cfgFile);

cfg = jsondecode(fileread(cfgFile));
assert(isfield(cfg,'plume_file'), 'Config lacks plume_file field');
plumePath = cfg.plume_file;

fprintf('Reading plume frame from %s ...\n', plumePath);
plume = h5read(plumePath,'/odor', [1 1 2000],[Inf Inf 1]);  % arbitrary frame 2000

%% Build coordinate grids in cm using arena bounds
[nRows,nCols] = size(plume(:,:,1));
xs = linspace(cfg.arena.x_min, cfg.arena.x_max, nCols);
ys = linspace(cfg.arena.y_min, cfg.arena.y_max, nRows);
[X,Y] = meshgrid(xs, ys);

%% Select top 5 % highest-odor pixels as in-plume candidates
mask = plume(:,:,1) > prctile(plume(:,:,1), 95);
[rowIdx,colIdx] = find(mask);

assert(numel(rowIdx) >= 50, 'Not enough high-odor pixels found – plume data suspicious');
sel = randperm(numel(rowIdx), 20);
points = [ X(sub2ind(size(X), rowIdx(sel), colIdx(sel)))', ...
           Y(sub2ind(size(Y), rowIdx(sel), colIdx(sel)))' ];   % [N x 2]

%% Function handle for the odor sampling routine used at runtime
if exist('sample_odor_xyz','file') == 2
    sampleFn = @sample_odor_xyz;
else
    error(['sample_odor_xyz.m not found on path. ' ...
           'Make sure it is in Code/ or update this test.']);
end

odVals = arrayfun(@(i) sampleFn(points(i,:), plume(:,:,1), cfg), 1:size(points,1));

if all(odVals > 0)
    fprintf('PASS ✓  All %d sampled points returned positive odor values.\n', numel(odVals));
else
    error('FAIL ✗  Some in-plume points returned zero odor – sampling bug likely.');
end
