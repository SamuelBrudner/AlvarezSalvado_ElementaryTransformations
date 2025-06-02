% Test reading odor values at specific positions
hdf5_file = 'data/10302017_10cms_bounded.hdf5';

% Test positions from your debug output
test_cases = [
    1, 345, 176;  % Step 1: correct position
    1, 1, 108;    % Steps 2-10: stuck position
    100, 200, 108; % Middle of plume
];

fprintf('Testing odor readings:\n');
fprintf('Format: [time, y, x] -> odor value\n\n');

for i = 1:size(test_cases, 1)
    tind = test_cases(i, 1);
    yind = test_cases(i, 2);
    xind = test_cases(i, 3);
    
    try
        odor_value = h5read(hdf5_file, '/dataset2', [tind yind xind], [1 1 1]);
        fprintf('[%d, %d, %d] -> %.6f\n', tind, yind, xind, odor_value);
    catch ME
        fprintf('[%d, %d, %d] -> ERROR: %s\n', tind, yind, xind, ME.message);
    end
end
