% Test coordinate transformations
pxscale = 0.74; % from the navigation model

% Test a few positions
test_positions = [
    0, 0;      % Origin
    5.04, -25.47;  % Your initial position
    -8, -30;   % Lower left
    8, -25;    % Lower right
];

fprintf('Testing coordinate transformations:\n');
fprintf('pxscale = %.2f mm/pixel\n\n', pxscale);

for i = 1:size(test_positions, 1)
    x = test_positions(i, 1);
    y = test_positions(i, 2);
    
    xind = round(10*x/pxscale) + 108;
    yind = -round(10*y/pxscale) + 1;
    
    fprintf('Position (%.2f, %.2f) mm -> indices [%d, %d]\n', x, y, xind, yind);
    
    % Check if indices are in bounds
    if xind < 1 || xind > 216 || yind < 1 || yind > 406
        fprintf('  WARNING: Out of bounds! Valid range: x[1,216], y[1,406]\n');
    end
end

% Also check what position gives xind=108, yind=1
fprintf('\nReverse check: xind=108, yind=1 corresponds to:\n');
x_from_108 = (108 - 108) * pxscale / 10;
y_from_1 = -(1 - 1) * pxscale / 10;
fprintf('  x = %.2f mm, y = %.2f mm\n', x_from_108, y_from_1);
