% generate_clean_configs.m - Generate clean configs from ground truth only

fprintf('=== Generating Clean Configs (Overwriting Old) ===\n\n');

%% CRIMALDI CONFIG
fprintf('1. Generating Crimaldi config...\n');

% Known experimental parameters
CRIM_FRAME_RATE = 15;      % Hz
CRIM_MM_PER_PIXEL = 0.74;  % mm/px

% Get actual HDF5 dimensions
crim_file = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/10302017_10cms_bounded.hdf5';
crim_info = h5info(crim_file, '/dataset2');
crim_dims = crim_info.Dataspace.Size;

fprintf('   HDF5: %d x %d x %d\n', crim_dims);

% Build config with only verifiable data
crim_cfg = struct();

% Essential identifiers
crim_cfg.plume_id = 'crimaldi_10cms_bounded';

% Data source
crim_cfg.data_path = struct();
crim_cfg.data_path.path = crim_file;
crim_cfg.data_path.dataset_name = '/dataset2';

% Spatial parameters
crim_cfg.spatial = struct();
crim_cfg.spatial.resolution.width = crim_dims(1);
crim_cfg.spatial.resolution.height = crim_dims(2);
crim_cfg.spatial.mm_per_pixel = CRIM_MM_PER_PIXEL;

% Calculate arena bounds (using model's convention: Y=0 at top)
crim_width_cm = crim_dims(1) * CRIM_MM_PER_PIXEL / 10;
crim_height_cm = crim_dims(2) * CRIM_MM_PER_PIXEL / 10;
crim_cfg.spatial.arena_bounds.x_min = -crim_width_cm/2;
crim_cfg.spatial.arena_bounds.x_max = crim_width_cm/2;
crim_cfg.spatial.arena_bounds.y_min = -crim_height_cm;
crim_cfg.spatial.arena_bounds.y_max = 0;

% Temporal parameters
crim_cfg.temporal = struct();
crim_cfg.temporal.frame_rate = CRIM_FRAME_RATE;
crim_cfg.temporal.total_frames = crim_dims(3);

% Save, overwriting old file
fid = fopen('configs/plumes/crimaldi_10cms_bounded.json', 'w');
fprintf(fid, '%s', jsonencode(crim_cfg));
fclose(fid);

fprintf('   ✓ Saved: Arena %.1f x %.1f cm\n', crim_width_cm, crim_height_cm);

%% SMOKE CONFIG
fprintf('\n2. Generating Smoke config...\n');

% Known experimental parameters
SMOKE_FRAME_RATE = 60;       % Hz
SMOKE_MM_PER_PIXEL = 0.153;  % mm/px

% Get actual HDF5 dimensions
smoke_file = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d_fixed.h5';
smoke_info = h5info(smoke_file, '/dataset2');
smoke_dims = smoke_info.Dataspace.Size;

fprintf('   HDF5: %d x %d x %d\n', smoke_dims);

% Build config with only verifiable data
smoke_cfg = struct();

% Essential identifiers
smoke_cfg.plume_id = 'smoke_1a_backgroundsubtracted';

% Data source
smoke_cfg.data_path = struct();
smoke_cfg.data_path.path = smoke_file;
smoke_cfg.data_path.dataset_name = '/dataset2';

% Spatial parameters
smoke_cfg.spatial = struct();
smoke_cfg.spatial.resolution.width = smoke_dims(1);
smoke_cfg.spatial.resolution.height = smoke_dims(2);
smoke_cfg.spatial.mm_per_pixel = SMOKE_MM_PER_PIXEL;

% Calculate arena bounds (same convention as Crimaldi)
smoke_width_cm = smoke_dims(1) * SMOKE_MM_PER_PIXEL / 10;
smoke_height_cm = smoke_dims(2) * SMOKE_MM_PER_PIXEL / 10;
smoke_cfg.spatial.arena_bounds.x_min = -smoke_width_cm/2;
smoke_cfg.spatial.arena_bounds.x_max = smoke_width_cm/2;
smoke_cfg.spatial.arena_bounds.y_min = -smoke_height_cm;
smoke_cfg.spatial.arena_bounds.y_max = 0;

% Temporal parameters
smoke_cfg.temporal = struct();
smoke_cfg.temporal.frame_rate = SMOKE_FRAME_RATE;
smoke_cfg.temporal.total_frames = smoke_dims(3);

% Save, overwriting old file
fid = fopen('configs/plumes/smoke_1a_backgroundsubtracted.json', 'w');
fprintf(fid, '%s', jsonencode(smoke_cfg));
fclose(fid);

fprintf('   ✓ Saved: Arena %.1f x %.1f cm\n', smoke_width_cm, smoke_height_cm);

%% SUMMARY
fprintf('\n=== Summary ===\n');
fprintf('Crimaldi: %d×%d px = %.1f×%.1f cm, Y∈[%.1f,%.1f]\n', ...
        crim_dims(1), crim_dims(2), crim_width_cm, crim_height_cm, ...
        crim_cfg.spatial.arena_bounds.y_min, crim_cfg.spatial.arena_bounds.y_max);
fprintf('Smoke:    %d×%d px = %.1f×%.1f cm, Y∈[%.1f,%.1f]\n', ...
        smoke_dims(1), smoke_dims(2), smoke_width_cm, smoke_height_cm, ...
        smoke_cfg.spatial.arena_bounds.y_min, smoke_cfg.spatial.arena_bounds.y_max);

fprintf('\n✓ Both configs now use consistent Y=0 at top convention\n');
fprintf('✓ Old configs overwritten with clean, minimal versions\n');