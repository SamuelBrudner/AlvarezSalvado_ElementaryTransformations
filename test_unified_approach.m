addpath('Code');

% Test loading plume info without loading data
fprintf('Testing HDF5 info loading...\n');
plume_info = load_plume_hdf5_info('data/10302017_10cms_bounded.hdf5', 1/0.74, 15);
fprintf('Plume dims: [%d %d %d]\n', plume_info.dims);

% Test direct h5read access
fprintf('\nTesting direct h5read access...\n');
test_val = h5read(plume_info.filename, plume_info.dataset, [100 200 1], [1 1 1]);
fprintf('Successfully read value: %.6f\n', test_val);

exit
