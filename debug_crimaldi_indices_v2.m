% Debug script to trace the index calculations in detail
fprintf('=== Debugging Crimaldi Indices ===\n');

% Simulation parameters
pxscale = 0.74;
plume_xlims = [1 216];
plume_ylims = [1 406];

% Test our initialization range
test_x = [-7.5, 0, 7.5];  % cm
test_y = [-29.5, -27.25, -25];  % cm

fprintf('\nTesting coordinate transformations:\n');
for i = 1:length(test_x)
    for j = 1:length(test_y)
        x = test_x(i);
        y = test_y(j);
        
        xind = round(10*x/pxscale)+108;
        yind = -round(10*y/pxscale)+1;
        
        fprintf('Position (%.1f, %.1f) cm -> indices [%d, %d]', x, y, xind, yind);
        
        if xind < plume_xlims(1) || xind > plume_xlims(2) || ...
           yind < plume_ylims(1) || yind > plume_ylims(2)
            fprintf(' [OUT OF BOUNDS!]');
        else
            fprintf(' [OK]');
        end
        fprintf('\n');
    end
end

% Now let's check what happens during a simulation
fprintf('\n=== Simulating navigation model calculations ===\n');

% Initialize like the model does
x = (rand(1,1).*15)-7.5;
y = rand(1,1)*4.5-29.5;
fprintf('Initial position: x=%.2f, y=%.2f cm\n', x, y);

% Simulate first few time steps
vbase = 6;  % mm/s
dt = 1/15;  % for Crimaldi at 15 Hz
heading = 360*rand(1,1);

for i = 1:5
    % Calculate indices
    xind = round(10*x/pxscale)+108;
    yind = -round(10*y/pxscale)+1;
    
    fprintf('Step %d: x=%.2f, y=%.2f -> xind=%d, yind=%d', i, x, y, xind, yind);
    
    if xind < plume_xlims(1) || xind > plume_xlims(2) || ...
       yind < plume_ylims(1) || yind > plume_ylims(2)
        fprintf(' [OUT OF BOUNDS!]\n');
        break;
    else
        fprintf(' [OK]\n');
    end
    
    % Simple movement update (no odor influence)
    [dx, dy] = pol2cart((heading-90)/360*2*pi, vbase * dt / 10);
    x = x + dx;
    y = y + dy;
    
    % Random heading change
    heading = heading + randn*20;
end
