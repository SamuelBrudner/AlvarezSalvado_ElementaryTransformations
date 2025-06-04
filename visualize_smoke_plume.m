% visualize_smoke_plume.m - Visualize the configured smoke plume

% Store current directory
original_dir = pwd;
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
cd(script_dir);

fprintf('Loading smoke plume configuration...\n');

% Add Code directory to path
if exist('Code', 'dir')
    addpath(genpath('Code'));
end

try
    % Load config
    [plume_file, plume_config] = get_plume_file();
    
    fprintf('\nLoaded: %s\n', plume_config.plume_id);
    fprintf('File: %s\n', plume_file);
    
    % Read sample frames
    total_frames = plume_config.temporal.total_frames;
    frames_to_show = [1, round(total_frames/2), total_frames];
    
    figure('Position', [100 100 1200 400]);
    for i = 1:3
        subplot(1, 3, i);
        frame = h5read(plume_file, plume_config.data_path.dataset_name, ...
                       [1 1 frames_to_show(i)], [inf inf 1]);
        imagesc(frame');
        colormap(hot);
        colorbar;
        title(sprintf('Frame %d (t=%.1fs)', frames_to_show(i), ...
                      (frames_to_show(i)-1)/plume_config.temporal.frame_rate));
        axis equal tight;
    end
    
    sgtitle(sprintf('%s: %.1f×%.1f cm, %d Hz', ...
            plume_config.plume_id, ...
            plume_config.spatial.arena_bounds.x_max * 2, ...
            plume_config.spatial.arena_bounds.y_max * 2, ...
            plume_config.temporal.frame_rate));
    
catch ME
    fprintf('Error: %s\n', ME.message);
end

cd(original_dir);
