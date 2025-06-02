% Check how data is actually stored in the HDF5 file
h5file = 'data/10302017_10cms_bounded.hdf5';

% Read a small sample
sample = h5read(h5file, '/dataset2', [1 1 1], [10 10 1]);
fprintf('Sample data shape: %dx%d\n', size(sample));

% Check a specific point that should be accessible
fprintf('\nTrying to read point [1,1,1]:\n');
val1 = h5read(h5file, '/dataset2', [1 1 1], [1 1 1]);
fprintf('Success! Value = %f\n', val1);

fprintf('\nTrying to read point [100,100,1]:\n');
val2 = h5read(h5file, '/dataset2', [100 100 1], [1 1 1]);
fprintf('Success! Value = %f\n', val2);

fprintf('\nTrying to read point [200,200,1] (should work if dims are 216x406):\n');
try
    val3 = h5read(h5file, '/dataset2', [200 200 1], [1 1 1]);
    fprintf('Success! Value = %f\n', val3);
catch ME
    fprintf('Failed: %s\n', ME.message);
end

fprintf('\nTrying to read point [400,200,1] (should fail if first dim is 216):\n');
try
    val4 = h5read(h5file, '/dataset2', [400 200 1], [1 1 1]);
    fprintf('Success! Value = %f\n', val4);
catch ME
    fprintf('Failed: %s\n', ME.message);
end

fprintf('\nDataset dimensions are: [216 406 3600]\n');
fprintf('This means valid indices are:\n');
fprintf('  First dimension:  1-216\n');
fprintf('  Second dimension: 1-406\n');
fprintf('  Third dimension:  1-3600\n');
exit
