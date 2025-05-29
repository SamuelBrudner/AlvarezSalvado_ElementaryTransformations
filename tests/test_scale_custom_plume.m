function tests = test_scale_custom_plume
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd,'Code'));
    tmpDir = tempname;
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir,'plume.avi'));
    open(vw);
    writeVideo(vw,uint8([0 255;128 64]));
    writeVideo(vw,uint8([255 128;64 0]));
    close(vw);
    meta = fullfile(tmpDir,'meta.yaml');
    fid = fopen(meta,'w');
    fprintf(fid,'output_directory: %s\n',tmpDir);
    fprintf(fid,'output_filename: plume.avi\n');
    fprintf(fid,'vid_mm_per_px: 1\n');
    fprintf(fid,'fps: 1\n');
    fclose(fid);
    testCase.TestData.tmpDir = tmpDir;
    testCase.TestData.meta = meta;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir,'s');
end

function testScaledOutput(testCase)
    scaledMeta = scale_custom_plume(testCase.TestData.meta);
    plume = load_custom_plume(scaledMeta);
    stats = plume_intensity_stats();
    verifyEqual(testCase,min(plume.data(:)),stats.CRIM.min,'AbsTol',1e-12);
    verifyEqual(testCase,max(plume.data(:)),stats.CRIM.max,'AbsTol',1e-12);
    metaData = load_yaml(scaledMeta);
    verifyTrue(testCase,isfield(metaData,'scaled_to_crim'));%
    verifyTrue(testCase,metaData.scaled_to_crim);
end
