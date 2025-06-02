% Test that video plume centering works correctly
% Simulate the coordinate transformation

% Smoke plume dimensions
width = 1088;  % pixels
height = 1728; % pixels
px_per_mm = 6.536;

% Test positions
test_x = [0, -7.5, 7.5, -8.3, 8.3];  % cm

fprintf('Video plume centering test:\n');
fprintf('Plume width: %d pixels (%.1f cm)\n', width, width/px_per_mm/10);
fprintf('Center pixel: %d\n\n', round(width/2));

for i = 1:length(test_x)
    x = test_x(i);
    xind = round(10*x*px_per_mm) + round(width/2);
    
    fprintf('x = %5.1f cm -> xind = %4d', x, xind);
    
    if xind < 1 || xind > width
        fprintf(' [OUT OF BOUNDS!]');
    else
        fprintf(' [OK]');
    end
    fprintf('\n');
end

% Calculate the valid x range
x_min = (1 - round(width/2)) / (10*px_per_mm);
x_max = (width - round(width/2)) / (10*px_per_mm);
fprintf('\nValid x range: [%.2f, %.2f] cm\n', x_min, x_max);
