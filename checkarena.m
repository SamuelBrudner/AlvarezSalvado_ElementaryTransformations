% check_arena_calc.m - Calculate true arena sizes

smoke_cfg = jsondecode(fileread('configs/plumes/smoke_1a_backgroundsubtracted.json'));
crim_cfg = jsondecode(fileread('configs/plumes/crimaldi_10cms_bounded.json'));

fprintf('=== Arena Size Calculations ===\n\n');

% Crimaldi
fprintf('CRIMALDI:\n');
crim_width_px = crim_cfg.spatial.resolution.width;
crim_height_px = crim_cfg.spatial.resolution.height;
crim_scale = crim_cfg.spatial.mm_per_pixel;

crim_width_cm = crim_width_px * crim_scale / 10;
crim_height_cm = crim_height_px * crim_scale / 10;

fprintf('  Pixels: %d x %d\n', crim_width_px, crim_height_px);
fprintf('  Scale: %.3f mm/px\n', crim_scale);
fprintf('  Physical size: %.1f x %.1f cm\n', crim_width_cm, crim_height_cm);
fprintf('  Config arena bounds: %.1f x %.1f cm\n', ...
        crim_cfg.spatial.arena_bounds.x_max - crim_cfg.spatial.arena_bounds.x_min, ...
        crim_cfg.spatial.arena_bounds.y_max - crim_cfg.spatial.arena_bounds.y_min);

% Smoke
fprintf('\nSMOKE:\n');
smoke_width_px = smoke_cfg.spatial.resolution.width;
smoke_height_px = smoke_cfg.spatial.resolution.height;
smoke_scale = smoke_cfg.spatial.mm_per_pixel;

smoke_width_cm = smoke_width_px * smoke_scale / 10;
smoke_height_cm = smoke_height_px * smoke_scale / 10;

fprintf('  Pixels: %d x %d\n', smoke_width_px, smoke_height_px);
fprintf('  Scale: %.3f mm/px\n', smoke_scale);
fprintf('  Physical size: %.1f x %.1f cm\n', smoke_width_cm, smoke_height_cm);
fprintf('  Config arena bounds: %.1f x %.1f cm\n', ...
        smoke_cfg.spatial.arena_bounds.x_max - smoke_cfg.spatial.arena_bounds.x_min, ...
        smoke_cfg.spatial.arena_bounds.y_max - smoke_cfg.spatial.arena_bounds.y_min);