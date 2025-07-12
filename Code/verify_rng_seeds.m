function verify_rng_seeds()
%VERIFY_RNG_SEEDS  Quick QC to confirm each results file has a unique RNG seed.
%
%   Scans the project "results/" directory for *_nav_results_*.mat files
%   and reports any duplicate or missing seeds. Intended to run after batch
%   simulations to ensure that each SLURM array task used an independent
%   random seed.
%
%   Usage (from repo root):
%       matlab -nodisplay -nosplash -batch "addpath(genpath('Code')); verify_rng_seeds"
%
%   All messages are printed to stdout; the function exits with a non-zero
%   status via error() if any issues are detected so that CI pipelines can
%   fail fast.

proj_dir = fileparts(mfilename('fullpath'));
proj_dir = fileparts(proj_dir);          % repo root
results_dir = fullfile(proj_dir, 'results');

files = dir(fullfile(results_dir, '*_nav_results_*.mat'));
if isempty(files)
    error('No result files found in %s', results_dir);
end

seeds = containers.Map('KeyType','double','ValueType','int32');
missing = 0;

for f = files'
    data = load(fullfile(f.folder, f.name), 'out');
    if ~isfield(data, 'out') || ~isfield(data.out, 'rng_seed')
        fprintf('[WARN] %s lacks out.rng_seed\n', f.name);
        missing = missing + 1;
        continue
    end
    seed = double(data.out.rng_seed);
    if isKey(seeds, seed)
        seeds(seed) = seeds(seed) + 1;
    else
        seeds(seed) = 1;
    end
end

dup_seeds = cell2mat(keys(seeds));
dup_seeds = dup_seeds(cell2mat(values(seeds)) > 1);

fprintf('\n=== RNG Seed QC ===\n');
fprintf('Total result files : %d\n', numel(files));
fprintf('Files with seed    : %d\n', numel(files) - missing);
fprintf('Unique seeds       : %d\n', seeds.Count);
fprintf('Duplicate seeds    : %d\n', numel(dup_seeds));

if missing > 0
    error('QC failed: %d files missing rng_seed', missing);
elseif ~isempty(dup_seeds)
    fprintf('Duplicate seed values: %s\n', mat2str(dup_seeds));
    error('QC failed: duplicate seeds detected');
else
    fprintf('\n✓ QC passed – all seeds present and unique.\n');
end
end
