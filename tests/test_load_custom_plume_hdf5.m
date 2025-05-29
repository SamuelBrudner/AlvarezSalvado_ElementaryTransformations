function tests = test_load_custom_plume_hdf5
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd,'Code'));
    tmp = tempname;
    mkdir(tmp);
    frames(:,:,1) = uint8([0 255;128 64]);
    frames(:,:,2) = uint8([255 128;64 0]);
    h5 = fullfile(tmp,'plume.h5');
    h5create(h5,'/dataset1', size(frames),'Datatype','uint8');
    h5write(h5,'/dataset1', frames);
    fid = fopen(fullfile(tmp,'meta.yaml'),'w');
    fprintf(fid,'output_directory: %s\n',tmp);
    fprintf(fid,'output_filename: plume.h5\n');
    fprintf(fid,'vid_mm_per_px: 1\n');
    fprintf(fid,'fps: 1\n');
    fclose(fid);
    testCase.TestData.tmp = tmp;
    testCase.TestData.meta = fullfile(tmp,'meta.yaml');
    testCase.TestData.h5 = h5;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmp,'s');
end

function testLoadViaFilename(testCase)
    stats = plume_intensity_stats();
    plume = load_custom_plume(testCase.TestData.meta);
    verifySize(testCase, plume.data, [2 2 2]);
    verifyEqual(testCase, plume.px_per_mm, 1);
    verifyEqual(testCase, plume.frame_rate, 1);
    verifyEqual(testCase, min(plume.data(:)), stats.CRIM.min, 'AbsTol',1e-12);
    verifyEqual(testCase, max(plume.data(:)), stats.CRIM.max, 'AbsTol',1e-12);
end

function testLoadViaOutputH5(testCase)
    meta = fullfile(testCase.TestData.tmp,'meta2.yaml');
    fid = fopen(meta,'w');
    fprintf(fid,'output_directory: %s\n',testCase.TestData.tmp);
    fprintf(fid,'output_filename: dummy.avi\n');
    fprintf(fid,'output_h5: plume.h5\n');
    fprintf(fid,'vid_mm_per_px: 1\n');
    fprintf(fid,'fps: 1\n');
    fclose(fid);
    stats = plume_intensity_stats();
    plume = load_custom_plume(meta);
    verifySize(testCase, plume.data, [2 2 2]);
    verifyEqual(testCase, min(plume.data(:)), stats.CRIM.min, 'AbsTol',1e-12);
    verifyEqual(testCase, max(plume.data(:)), stats.CRIM.max, 'AbsTol',1e-12);
    delete(meta);
end
