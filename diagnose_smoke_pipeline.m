% Diagnose where the smoke data transformation goes wrong
fprintf('Diagnosing smoke data pipeline...\n\n');

% Add paths
addpath('Code');

% Step 1: Load and display a frame from the original AVI
fprintf('=== STEP 1: Original AVI ===\n');
avi_path = 'data/smoke_1a_orig_backgroundsubtracted.avi';
v = VideoReader(avi_path);
fprintf('AVI properties:\n');
fprintf('  Size: %d x %d\n', v.Height, v.Width);
fprintf('  Frame rate: %.1f\n', v.FrameRate);
fprintf('  Duration: %.1f seconds\n', v.Duration);
fprintf('  Number of frames: %d\n', v.NumFrames);

% Read middle frame from AVI
middle_frame_num = round(v.NumFrames / 2);
v.CurrentTime = (middle_frame_num - 1) / v.FrameRate;
avi_frame = readFrame(v);
if size(avi_frame, 3) == 3
    avi_frame = rgb2gray(avi_frame);
end
avi_frame = im2double(avi_frame);

% Step 2: Check each HDF5 file in the pipeline
fprintf('\n=== STEP 2: Checking HDF5 files ===\n');
base_path = '/home/snb6/palmer_scratch/plume/';
files_to_check = {
    'smoke_1a_orig_backgroundsubtracted_raw.h5',
    'smoke_1a_orig_backgroundsubtracted_scaled.h5',
    'smoke_1a_orig_backgroundsubtracted_rotated.h5'
};

figure('Position', [50, 50, 1600, 1200]);

% Plot original AVI frame
subplot(3, 3, 1);
imagesc(avi_frame);
colorbar;
title(sprintf('Original AVI - Frame %d', middle_frame_num));
xlabel(sprintf('Width: %d', size(avi_frame, 2)));
ylabel(sprintf('Height: %d', size(avi_frame, 1)));
axis image;
colormap(gca, 'gray');

% Check each HDF5 file
for i = 1:length(files_to_check)
    file_path = fullfile(base_path, files_to_check{i});
    fprintf('\nChecking: %s\n', files_to_check{i});
    
    try
        % Read info
        info = h5info(file_path, '/dataset1');
        
        % Look for dimensions in attributes
        height = [];
        width = [];
        frames = [];
        for j = 1:length(info.Attributes)
            switch info.Attributes(j).Name
                case 'height'
                    height = double(info.Attributes(j).Value);
                case 'width'
                    width = double(info.Attributes(j).Value);
                case 'frames'
                    frames = double(info.Attributes(j).Value);
            end
        end
        
        fprintf('  Dimensions from attributes: %d x %d x %d\n', height, width, frames);
        fprintf('  Dataset size: %s\n', mat2str(info.Dataspace.Size));
        
        % Read data
        data = h5read(file_path, '/dataset1');
        
        % Try to extract middle frame
        middle_idx = round(frames / 2);
        
        if length(info.Dataspace.Size) == 1
            % 1D array - need to reshape
            fprintf('  Data is 1D, reshaping...\n');
            
            % Try different reshape orders
            subplot(3, 3, 3*i-1);
            try
                % Order 1: [frames, height, width] (C-style)
                data_3d = reshape(data, [frames, height, width]);
                frame = squeeze(data_3d(middle_idx, :, :));
                imagesc(frame);
                colorbar;
                title(sprintf('%s\n[F,H,W] order', strrep(files_to_check{i}, '_', '\_')));
                axis image;
            catch
                title('Reshape failed');
            end
            
            subplot(3, 3, 3*i);
            try
                % Order 2: [width, height, frames] (MATLAB-style)
                data_3d = reshape(data, [width, height, frames]);
                frame = data_3d(:, :, middle_idx)';
                imagesc(frame);
                colorbar;
                title(sprintf('%s\n[W,H,F] order', strrep(files_to_check{i}, '_', '\_')));
                axis image;
            catch
                title('Reshape failed');
            end
            
            subplot(3, 3, 3*i+1);
            try
                % Order 3: [height, width, frames]
                data_3d = reshape(data, [height, width, frames]);
                frame = data_3d(:, :, middle_idx);
                imagesc(frame);
                colorbar;
                title(sprintf('%s\n[H,W,F] order', strrep(files_to_check{i}, '_', '\_')));
                axis image;
            catch
                title('Reshape failed');
            end
        else
            % Already 3D
            fprintf('  Data is already 3D\n');
            subplot(3, 3, 3*i);
            frame = data(:, :, middle_idx);
            imagesc(frame);
            colorbar;
            title(sprintf('%s\n3D data', strrep(files_to_check{i}, '_', '\_')));
            axis image;
        end
        
    catch ME
        fprintf('  Error: %s\n', ME.message);
    end
end

sgtitle('Smoke Pipeline Diagnostic - Middle Frames', 'FontSize', 14);
print('smoke_pipeline_diagnostic.png', '-dpng', '-r150');
fprintf('\nSaved diagnostic plot to smoke_pipeline_diagnostic.png\n');

% Also check if the rotation is the issue
fprintf('\n=== STEP 3: Checking rotation ===\n');
figure('Position', [100, 100, 1200, 400]);

% Load scaled data and apply rotation manually
scaled_path = fullfile(base_path, 'smoke_1a_orig_backgroundsubtracted_scaled.h5');
data_1d = h5read(scaled_path, '/dataset1');
info = h5info(scaled_path, '/dataset1');

% Get dimensions
height = 1088; width = 1728; frames = 3600; % From previous output
middle_idx = round(frames / 2);

% Try the most likely reshape order
data_3d = reshape(data_1d, [frames, height, width]);
frame_before = squeeze(data_3d(middle_idx, :, :));

subplot(1, 3, 1);
imagesc(frame_before);
colorbar;
title('Before rotation');
axis image;

subplot(1, 3, 2);
frame_rot90 = rot90(frame_before, -1); % Rotate 90 clockwise
imagesc(frame_rot90);
colorbar;
title('After rot90(frame, -1)');
axis image;

subplot(1, 3, 3);
frame_rot90_other = rot90(frame_before, 1); % Rotate 90 counter-clockwise
imagesc(frame_rot90_other);
colorbar;
title('After rot90(frame, 1)');
axis image;

print('smoke_rotation_test.png', '-dpng', '-r150');
fprintf('Saved rotation test to smoke_rotation_test.png\n');
