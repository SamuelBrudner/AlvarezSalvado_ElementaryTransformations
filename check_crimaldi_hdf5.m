% Check Crimaldi HDF5 structure
hdf5_file = 'data/10302017_10cms_bounded.hdf5';
if exist(hdf5_file, 'file')
    info = h5info(hdf5_file);
    fprintf('HDF5 file structure:\n');
    for i = 1:length(info.Datasets)
        fprintf('  Dataset: %s, Size: ', info.Datasets(i).Name);
        fprintf('%d ', info.Datasets(i).Dataspace.Size);
        fprintf('\n');
    end
else
    fprintf('Crimaldi HDF5 file not found!\n');
end
exit;
