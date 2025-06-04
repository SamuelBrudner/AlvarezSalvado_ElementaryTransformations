% clean_quick_test.m - Clean test using configs directly

fprintf('=== Clean Quick Test ===\n\n');

%% Test parameters
n_agents = 10;
test_duration_seconds = 30;

%% Test 1: Crimaldi
fprintf('1. Testing Crimaldi plume...\n');

% Clear environment to use default Crimaldi
setenv('MATLAB_PLUME_FILE', '');

% Run test
tic;
out_crim = Elifenavmodel_bilateral(test_duration_seconds * 15, 'Crimaldi', 0, n_agents);
crim_time = toc;

fprintf('   ✓ Success: %.0f%% in %.1f seconds\n', out_crim.successrate * 100, crim_time);

%% Test 2: Smoke
fprintf('\n2. Testing Smoke plume...\n');

% Set environment to use smoke plume
smoke_cfg = jsondecode(fileread('configs/plumes/smoke_1a_backgroundsubtracted.json'));
setenv('MATLAB_PLUME_FILE', smoke_cfg.data_path.path);

% Run test (duration adjusted for 60Hz)
tic;
out_smoke = Elifenavmodel_bilateral(test_duration_seconds * 60, 'Crimaldi', 0, n_agents);
smoke_time = toc;

fprintf('   ✓ Success: %.0f%% in %.1f seconds\n', out_smoke.successrate * 100, smoke_time);

%% Plot results
figure('Position', [100 100 800 400]);

subplot(1,2,1);
plot(out_crim.x, out_crim.y, '-', 'LineWidth', 1);
hold on;
viscircles([0, 0], 2, 'Color', 'g', 'LineWidth', 2);
plot(0, 0, 'g*', 'MarkerSize', 12);
title(sprintf('Crimaldi (15Hz): %.0f%%', out_crim.successrate*100));
xlabel('X (cm)'); ylabel('Y (cm)');
axis equal; xlim([-10, 10]); ylim([-32, 2]);
grid on;

subplot(1,2,2);
plot(out_smoke.x, out_smoke.y, '-', 'LineWidth', 1);
hold on;
viscircles([0, 0], 2, 'Color', 'g', 'LineWidth', 2);
plot(0, 0, 'g*', 'MarkerSize', 12);
title(sprintf('Smoke (60Hz): %.0f%%', out_smoke.successrate*100));
xlabel('X (cm)'); ylabel('Y (cm)');
axis equal; xlim([-10, 10]); ylim([-28, 2]);
grid on;

sgtitle('Navigation Test Results');

% Clean up
setenv('MATLAB_PLUME_FILE', '');

fprintf('\n✓ Test complete!\n');