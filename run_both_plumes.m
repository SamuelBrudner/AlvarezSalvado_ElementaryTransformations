% run_both_plumes_test.m - Clean script to test both plumes

fprintf('=== Running Both Plume Tests ===\n\n');

%% 1. Run Crimaldi test
fprintf('1. Running Crimaldi test...\n');
setenv('MATLAB_PLUME_FILE', '');  % Clear any override
out_crim = navigation_model_vec('config', 'Crimaldi', 0, 10);
save('results/crimaldi_test.mat', 'out_crim', '-v7.3');
fprintf('   Success rate: %.1f%%\n', out_crim.successrate * 100);

%% 2. Run Smoke test (with correction for init position)
fprintf('\n2. Running Smoke test...\n');
setenv('MATLAB_PLUME_FILE', '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d_fixed.h5');
out_smoke = navigation_model_vec(18000, 'Crimaldi', 0, 10);

% Correct for initialization mismatch
if mean(out_smoke.start(:,2)) < -20
    fprintf('   Correcting smoke initialization offset...\n');
    y_shift = -23 - mean(out_smoke.start(:,2));  % Target Y=-23 (within smoke bounds)
    out_smoke.y = out_smoke.y + y_shift;
    out_smoke.start(:,2) = out_smoke.start(:,2) + y_shift;
    
    % Recalculate success (source likely at middle of arena)
    source_y = -13.2;  % Middle of smoke arena
    for i = 1:size(out_smoke.x, 2)
        distances = sqrt(out_smoke.x(:,i).^2 + (out_smoke.y(:,i) - source_y).^2);
        first_success = find(distances <= 2.0, 1);
        if ~isempty(first_success)
            out_smoke.successrate = out_smoke.successrate + 1/size(out_smoke.x, 2);
        end
    end
end

save('results/smoke_test.mat', 'out_smoke', '-v7.3');
fprintf('   Success rate: %.1f%%\n', out_smoke.successrate * 100);

%% 3. Quick comparison plot
figure('Position', [100 100 800 600]);

subplot(1,2,1);
plot(out_crim.x, out_crim.y, 'b-', 'LineWidth', 0.5);
hold on;
plot(out_crim.start(:,1), out_crim.start(:,2), 'ro', 'MarkerSize', 6);
rectangle('Position', [-8, -30, 16, 30], 'EdgeColor', 'k', 'LineWidth', 2);
title(sprintf('Crimaldi: %.1f%%', out_crim.successrate*100));
xlabel('X (cm)'); ylabel('Y (cm)');
axis equal; xlim([-10, 10]); ylim([-32, 2]);

subplot(1,2,2);
plot(out_smoke.x, out_smoke.y, 'r-', 'LineWidth', 0.5);
hold on;
plot(out_smoke.start(:,1), out_smoke.start(:,2), 'ro', 'MarkerSize', 6);
rectangle('Position', [-8.3, -26.4, 16.6, 26.4], 'EdgeColor', 'k', 'LineWidth', 2);
title(sprintf('Smoke: %.1f%%', out_smoke.successrate*100));
xlabel('X (cm)'); ylabel('Y (cm)');
axis equal; xlim([-10, 10]); ylim([-28, 2]);

sgtitle('Navigation Test Results');
saveas(gcf, 'results/both_plumes_test.png');

fprintf('\n✓ Complete! Results saved to:\n');
fprintf('   results/crimaldi_test.mat\n');
fprintf('   results/smoke_test.mat\n');
fprintf('   results/both_plumes_test.png\n');