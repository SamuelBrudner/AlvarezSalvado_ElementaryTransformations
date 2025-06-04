% inspect_smoke_hdf5.m - Quick inspection of the smoke plume HDF5 file

hdf5_file = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d.h5';

fprintf('\n=== Smoke Plume HDF5 Inspection ===\n\n');

% Check if file exists
if ~exist(hdf5_file, 'file')
    error('HDF5 file not found: %s', hdf5_file);
end

% Get file size
file_info = dir(hdf5_file);
fprintf('File: %s\n', hdf5_file);
fprintf('Size: %.2f GB\n\n', file_info.bytes / 1024^3);

% Get HDF5 info
fprintf('Reading HDF5 structure...\n');
try
    info = h5info(hdf5_file);
    
    % List all datasets
    fprintf('\nDatasets found:\n');
    for i = 1:length(info.Datasets)
        ds = info.Datasets(i);
        fprintf('  %s: ', ds.Name);
        fprintf('[%s]', sprintf('%d ', ds.Dataspace.Size));
        fprintf(' Type: %s\n', ds.Datatype.Class);
    end
    
    % Check for the expected dataset
    dataset_name = '/dataset2';
    if exist('h5info')
        try
            ds_info = h5info(hdf5_file, dataset_name);
            fprintf('\nDataset %s details:\n', dataset_name);
            fprintf('  Dimensions: %d x %d x %d\n', ds_info.Dataspace.Size);
            fprintf('  Width: %d pixels\n', ds_info.Dataspace.Size(1));
            fprintf('  Height: %d pixels\n', ds_info.Dataspace.Size(2));
            fprintf('  Frames: %d\n', ds_info.Dataspace.Size(3));
            
            % Calculate duration at 60 Hz
            n_frames = ds_info.Dataspace.Size(3);
            duration_sec = n_frames / 60.0;
            fprintf('  Duration at 60 Hz: %.1f seconds (%.1f minutes)\n', ...
                    duration_sec, duration_sec/60);
            
            % Sample a frame to check data range
            fprintf('\nSampling middle frame to check data range...\n');
            middle_frame = round(n_frames / 2);
            sample_data = h5read(hdf5_file, dataset_name, [1 1 middle_frame], [inf inf 1]);
            
            fprintf('  Data type: %s\n', class(sample_data));
            fprintf('  Data range: [%.6f, %.6f]\n', min(sample_data(:)), max(sample_data(:)));
            fprintf('  Mean value: %.6f\n', mean(sample_data(:)));
            fprintf('  Appears normalized: %s\n', ...
                    iif(max(sample_data(:)) <= 1.0 && min(sample_data(:)) >= 0, 'Yes', 'No'));
            
        catch ME
            fprintf('  Error reading dataset %s: %s\n', dataset_name, ME.message);
        end
    end
    
    fprintf('\n✓ File inspection complete!\n');
    
catch ME
    fprintf('Error: %s\n', ME.message);
end

fprintf('\nTo use this file, run: ./setup_smoke_plume_config.sh\n');

function r = iif(c,t,f)
    if c, r=t; else, r=f; end
end