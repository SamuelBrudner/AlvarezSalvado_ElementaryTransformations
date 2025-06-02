% Debug Crimaldi indices
addpath('Code');

% Test parameters
pxscale = 0.74;
triallength = 100; % Just test first 100 steps

% Initial position (same as in the model)
x = zeros(triallength, 1);
y = zeros(triallength, 1);
x(1) = (rand()*16)-8;  % Between -8 and 8
y(1) = rand()*5-30;     % Between -30 and -25

fprintf('Initial position: x=%.2f, y=%.2f\n', x(1), y(1));

% Calculate indices like the model does
for i = 1:10
    xind = round(10*x(i)/pxscale)+108;
    yind = -round(10*y(i)/pxscale)+1;
    tind = mod(i-1,3600)+1;
    
    fprintf('Step %d: xind=%d, yind=%d, tind=%d\n', i, xind, yind, tind);
    
    % Check bounds
    if xind < 1 || xind > 216
        fprintf('  WARNING: xind out of bounds [1,216]\n');
    end
    if yind < 1 || yind > 406
        fprintf('  WARNING: yind out of bounds [1,406]\n');
    end
end

% Check the data dimensions
info = h5info('data/10302017_10cms_bounded.hdf5', '/dataset2');
fprintf('\nDataset shape: %s\n', mat2str(info.Dataspace.Size));
fprintf('Expected indices: [1-%d, 1-%d, 1-%d]\n', info.Dataspace.Size(1), info.Dataspace.Size(2), info.Dataspace.Size(3));
