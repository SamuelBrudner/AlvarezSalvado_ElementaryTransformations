% Unified plume diagnostic using same access method as simulations
addpath('Code');

fprintf('=== Plume Data Diagnostics (Unified Access Method) ===\n');
fprintf('Note: Both plumes are in (width, height, frames) format\n\n');

% Load plume info for both datasets (same as simulation)
crimaldi = load_plume_hdf5_info('data/10302017_10cms_bounded.hdf5', 1/0.74, 15);
smoke = load_plume_hdf5_info('data/smoke_1a_rotated_3d.h5', 6.536, 60);

% Create diagnostic plots
figure('Visible', 'off', 'Position', [100 100 1200 500]);

% Process each plume
plumes = {crimaldi, smoke};
names = {'Crimaldi', 'Smoke'};

for p = 1:2
    plume = plumes{p};
    name = names{p};
    
    fprintf('Checking %s Plume (%s)...\n', name, plume.filename);
    fprintf('  Dataset: %s\n', plume.dataset);
    fprintf('  Shape: [%d %d %d] = [width height frames]\n', plume.dims);
    
    % Get middle frame index (same as Python diagnostic)
    middle_idx = round(plume.dims(3) / 2);
    
    % Read frame data using EXACT same method as simulation
    % We'll sample every 10th pixel for speed in diagnostic
    step = 10;
    width_samples = 1:step:plume.dims(1);
    height_samples = 1:step:plume.dims(2);
    
    fprintf('  Reading frame %d using h5read (sampling every %d pixels)...\n', middle_idx, step);
    
    % Pre-allocate
    slice_data = zeros(length(height_samples), length(width_samples));
    
    % Read using same h5read pattern as simulation
    for xi = 1:length(width_samples)
        for yi = 1:length(height_samples)
            x_idx = width_samples(xi);
            y_idx = height_samples(yi);
            % EXACT same call as in navigation_model_vec.m
            val = h5read(plume.filename, plume.dataset, [x_idx y_idx middle_idx], [1 1 1]);
            slice_data(yi, xi) = val;
        end
    end
    
    % Interpolate for full resolution display
    [X, Y] = meshgrid(width_samples, height_samples);
    [Xi, Yi] = meshgrid(1:plume.dims(1), 1:plume.dims(2));
    slice_data_full = interp2(X, Y, slice_data, Xi, Yi, 'linear');
    
    % Stats
    fprintf('  Range: [%.6f, %.6f]\n', min(slice_data(:)), max(slice_data(:)));
    fprintf('  Mean: %.6f\n', mean(slice_data(:)));
    
    % Plot - note transpose to match visual orientation
    subplot(1, 2, p);
    imagesc(slice_data_full');  % Transpose for display
    colormap(hot);
    colorbar;
    set(gca, 'YDir', 'normal');
    
    % Add reference point at (0,-25) cm - using simulation's coordinate system
    hold on;
    if strcmp(name, 'Crimaldi')
        % From navigation_model_vec.m Crimaldi case:
        % xind = round(10*x(i,:)/pxscale)+108;
        % yind = -round(10*y(i,:)/pxscale)+1;
        pxscale = 0.74;  % mm/pixel
        x_cm = 0; y_cm = -25;
        arrow_x_px = round(10*x_cm/pxscale) + 108;
        arrow_y_px = -round(10*y_cm/pxscale) + 1;
    else  % Smoke - from video case
        % xind = round(10*x(i,:)*plume.px_per_mm) + round(plume.dims(1)/2);
        % yind = round(-10*y(i,:)*plume.px_per_mm)+1;
        x_cm = 0; y_cm = -25;
        arrow_x_px = round(10*x_cm*plume.px_per_mm) + round(plume.dims(1)/2);
        arrow_y_px = round(-10*y_cm*plume.px_per_mm) + 1;
    end
    
    % Plot on transposed axes
    plot(arrow_y_px, arrow_x_px, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    text(arrow_y_px+30, arrow_x_px-30, '(0,-25) cm', ...
        'Color', 'red', 'FontSize', 12, 'FontWeight', 'bold', ...
        'BackgroundColor', 'white', 'EdgeColor', 'red');
    
    % Also mark the source at (0,0)
    if strcmp(name, 'Crimaldi')
        source_x_px = round(10*0/pxscale) + 108;
        source_y_px = -round(10*0/pxscale) + 1;
    else
        source_x_px = round(10*0*plume.px_per_mm) + round(plume.dims(1)/2);
        source_y_px = round(-10*0*plume.px_per_mm) + 1;
    end
    plot(source_y_px, source_x_px, 'g^', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'g');
    text(source_y_px+30, source_x_px+30, 'Source (0,0)', ...
        'Color', 'green', 'FontSize', 10, 'FontWeight', 'bold', ...
        'BackgroundColor', 'white', 'EdgeColor', 'green');
    
    title(sprintf('%s - Frame %d (h5read method)', name, middle_idx));
    xlabel(sprintf('Height: %d pixels', plume.dims(2)));
    ylabel(sprintf('Width: %d pixels', plume.dims(1)));
    axis equal tight;
    grid on;
end

% Save the diagnostic plot
print('plume_diagnostic_unified', '-dpng', '-r150');
fprintf('\n✓ Saved diagnostic plot to plume_diagnostic_unified.png\n');

% Verify a few test accesses match simulation indexing
fprintf('\n=== Testing simulation access pattern ===\n');
x_cm = 5; y_cm = -20; t = 100;

for p = 1:2
    plume = plumes{p};
    name = names{p};
    
    if strcmp(name, 'Crimaldi')
        pxscale = 0.74;
        xind = round(10*x_cm/pxscale) + 108;
        yind = -round(10*y_cm/pxscale) + 1;
    else
        xind = round(10*x_cm*plume.px_per_mm) + round(plume.dims(1)/2);
        yind = round(-10*y_cm*plume.px_per_mm) + 1;
    end
    
    % Check bounds
    if xind >= 1 && xind <= plume.dims(1) && yind >= 1 && yind <= plume.dims(2)
        val = h5read(plume.filename, plume.dataset, [xind yind t], [1 1 1]);
        fprintf('%s: Position (%.1f,%.1f)cm -> pixel [%d,%d] = %.6f\n', ...
            name, x_cm, y_cm, xind, yind, val);
    else
        fprintf('%s: Position (%.1f,%.1f)cm -> pixel [%d,%d] OUT OF BOUNDS\n', ...
            name, x_cm, y_cm, xind, yind);
    end
end

exit;
