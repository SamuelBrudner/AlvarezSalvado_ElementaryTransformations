% Check the structure of the Crimaldi HDF5 file
hdf5_file = 'data/10302017_10cms_bounded.hdf5';

% Display file info
info = h5info(hdf5_file);
disp('File structure:');
disp(info);

% Check dataset info
dataset_info = h5info(hdf5_file, '/dataset2');
disp('Dataset2 info:');
disp(dataset_info);
disp(['Dataset dimensions: ' mat2str(dataset_info.Dataspace.Size)]);

% Test reading a small sample
try
    % Try reading with different index orders
    sample1 = h5read(hdf5_file, '/dataset2', [1 1 1], [1 1 1]);
    disp(['Sample [1,1,1]: ' num2str(sample1)]);
    
    sample2 = h5read(hdf5_file, '/dataset2', [100 200 1], [1 1 1]);
    disp(['Sample [100,200,1]: ' num2str(sample2)]);
catch ME
    disp(['Error reading: ' ME.message]);
end
