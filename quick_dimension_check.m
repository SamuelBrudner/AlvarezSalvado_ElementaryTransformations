% quick_dimension_check.m - Quick check of plume dimensions

fprintf('\n=== Plume Dimension Comparison ===\n\n');

% Crimaldi (reference)
crimaldi = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/10302017_10cms_bounded.hdf5';
if exist(crimaldi, 'file')
    info = h5info(crimaldi, '/dataset2');
    fprintf('Crimaldi plume:\n');
    fprintf('  Dimensions: [%d, %d, %d]\n', info.Dataspace.Size);
    fprintf('  → %d width × %d height × %d frames\n', ...
            info.Dataspace.Size(1), info.Dataspace.Size(2), info.Dataspace.Size(3));
    fprintf('  → %.1f seconds at 15 Hz\n\n', info.Dataspace.Size(3)/15);
end

% Original smoke (wrong order)
smoke_orig = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d.h5';
if exist(smoke_orig, 'file')
    info = h5info(smoke_orig, '/dataset2');
    fprintf('Smoke plume (original - WRONG order):\n');
    fprintf('  Dimensions: [%d, %d, %d]\n', info.Dataspace.Size);
    fprintf('  → Incorrectly interpreted as %d width × %d height × %d frames\n', ...
            info.Dataspace.Size(1), info.Dataspace.Size(2), info.Dataspace.Size(3));
    fprintf('  → Would be %.1f seconds at 60 Hz (wrong!)\n\n', info.Dataspace.Size(3)/60);
end

% Fixed smoke
smoke_fixed = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d_fixed.h5';
if exist(smoke_fixed, 'file')
    info = h5info(smoke_fixed, '/dataset2');
    fprintf('Smoke plume (fixed - CORRECT order):\n');
    fprintf('  Dimensions: [%d, %d, %d]\n', info.Dataspace.Size);
    fprintf('  → %d width × %d height × %d frames ✓\n', ...
            info.Dataspace.Size(1), info.Dataspace.Size(2), info.Dataspace.Size(3));
    fprintf('  → %.1f seconds at 60 Hz ✓\n', info.Dataspace.Size(3)/60);
else
    fprintf('Smoke plume (fixed) not found yet.\n');
    fprintf('Run fix_smoke_dimensions.m to create it.\n');
end

fprintf('\n');