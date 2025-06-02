h5file = 'data/10302017_10cms_bounded.hdf5';
try
    info = h5info(h5file);
    fprintf('HDF5 file: %s\n', h5file);
    fprintf('Datasets found:\n');
    for i = 1:length(info.Datasets)
        fprintf('  %s (size: ', info.Datasets(i).Name);
        fprintf('%d ', info.Datasets(i).Dataspace.Size);
        fprintf(')\n');
    end
catch ME
    fprintf('Error reading HDF5 file: %s\n', ME.message);
end
exit
