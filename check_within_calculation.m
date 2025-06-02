% Check the within/out_of_plume calculation
pxscale = 0.74;
plume_xlims = [1 216];
plume_ylims = [1 406];
ntrials = 10;

% Simulate random initial positions
x = (rand(1,ntrials).*15)-7.5;
y = rand(1,ntrials)*4.5-29.5;

fprintf('Testing out_of_plume calculation with %d trials:\n', ntrials);

% Calculate indices
xind = round(10*x/pxscale)+108;
yind = -round(10*y/pxscale)+1;

fprintf('\nInitial positions and indices:\n');
for i = 1:ntrials
    fprintf('Trial %d: (%.2f, %.2f) cm -> [%d, %d]\n', i, x(i), y(i), xind(i), yind(i));
end

% Check out_of_plume calculation
out_of_plume = union(union(find(xind<plume_xlims(1)),find(xind>plume_xlims(2))), ...
                     union(find(yind<plume_ylims(1)),find(yind>plume_ylims(2))));
within = setdiff(1:ntrials, out_of_plume);

fprintf('\nOut of plume: %d trials\n', length(out_of_plume));
fprintf('Within plume: %d trials\n', length(within));

if ~isempty(out_of_plume)
    fprintf('\nOut of bounds trials:\n');
    for it = out_of_plume
        fprintf('  Trial %d: xind=%d, yind=%d\n', it, xind(it), yind(it));
    end
end
