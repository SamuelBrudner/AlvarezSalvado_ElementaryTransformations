% quick_test_both_plumes.m - Quick validation test for both plumes

fprintf('=== Quick Plume Test (10 agents, 30 seconds each) ===\n\n');

%% Quick test parameters
n_agents = 10;
test_duration_seconds = 30;

%% Test Crimaldi
fprintf('1. Crimaldi test...\n');
setenv('MATLAB_PLUME_FILE', '');
tic;
out_crim = Elifenavmodel_bilateral(test_duration_seconds * 15, 'Crimaldi', 0, n_agents);
fprintf('   Success: %.0f%% in %.1f seconds\n', out_crim.successrate * 100, toc);

%% Test Smoke  
fprintf('\n2. Smoke test...\n');
setenv('MATLAB_PLUME_FILE', '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d_fixed.h5');
tic;
out_smoke = Elifenavmodel_bilateral(test_duration_seconds * 60, 'Crimaldi', 0, n_agents);
fprintf('   Success: %.0f%% in %.1f seconds\n', out_smoke.successrate * 100, toc);

%% Quick plot
figure('Position', [100 100 800 400]);

subplot(1,2,1);
plot(out_crim.x, out_crim.y, '-', 'LineWidth', 1);
hold on;
viscircles([0, 0], 2, 'Color', 'g', 'LineWidth', 2);
title(sprintf('Crimaldi: %.0f%%', out_crim.successrate*100));
axis equal; xlim([-10, 10]); ylim([-32, 2]);

subplot(1,2,2);
plot(out_smoke.x, out_smoke.y, '-', 'LineWidth', 1);
hold on;
viscircles([0, 0], 2, 'Color', 'g', 'LineWidth', 2);
title(sprintf('Smoke: %.0f%%', out_smoke.successrate*100));
axis equal; xlim([-10, 10]); ylim([-28, 2]);

sgtitle('Quick Test Results');

% Clear env
setenv('MATLAB_PLUME_FILE', '');
fprintf('\n✓ Quick test complete!\n');