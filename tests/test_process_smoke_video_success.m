function tests = test_process_smoke_video_success
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    addpath(fullfile(pwd, 'scripts'));

    tmpRoot = tempname;
    mkdir(tmpRoot);
    mkdir(fullfile(tmpRoot, 'configs'));
    mkdir(fullfile(tmpRoot, 'scripts'));

    % Create tiny AVI
    vw = VideoWriter(fullfile(tmpRoot, 'tiny.avi'));
    open(vw);
    writeVideo(vw, uint8(zeros(2,2,1)));
    close(vw);

    % Minimal plume config
    plumeFile = fullfile(tmpRoot, 'configs', 'plume.yaml');
    fid = fopen(plumeFile, 'w');
    fprintf(fid, 'px_per_mm: 1\nframe_rate: 1\n');
    fclose(fid);

    % Project paths YAML
    pathsFile = fullfile(tmpRoot, 'configs', 'project_paths.yaml');
    fid = fopen(pathsFile, 'w');
    fprintf(fid, 'project_root: %s\n', tmpRoot);
    fprintf(fid, 'scripts:\n  matlab: %s\n', tmpRoot);
    fprintf(fid, 'data:\n  video: %s\n', fullfile(tmpRoot, 'tiny.avi'));
    fprintf(fid, 'configs:\n  plume: %s\n', plumeFile);
    fprintf(fid, 'output:\n  matlab_temp: %s\n', tmpRoot);
    fclose(fid);

    % Simple YAML loader for this test
    fid = fopen(fullfile(tmpRoot, 'scripts', 'load_paths_config.m'), 'w');
    fprintf(fid, [ ...
        'function cfg = load_paths_config()', '\n', ...
        'yamlFile = fullfile(fileparts(mfilename(''fullpath'')), ''..'', ''configs'', ''project_paths.yaml'');', '\n', ...
        'fid = fopen(yamlFile, ''r'');', '\n', ...
        'lines = textscan(fid, ''%s'', ''Delimiter'', ''\n'', ''Whitespace'', '''');', '\n', ...
        'fclose(fid);', '\n', ...
        'cfg = struct();', '\n', ...
        'section = '''';', '\n', ...
        'for i = 1:numel(lines{1})', '\n', ...
        '    line = strtrim(lines{1}{i});', '\n', ...
        '    if isempty(line) || line(1) == ''#'', continue; end', '\n', ...
        '    if ~isempty(regexp(line, ''^\w+:$'', ''once''))', '\n', ...
        '        section = line(1:end-1); cfg.(section) = struct();', '\n', ...
        '        continue; end', '\n', ...
        '    tok = regexp(line, ''([^:]+):\s*(.*)'', ''tokens'', ''once'');', '\n', ...
        '    if isempty(tok), continue; end', '\n', ...
        '    if isempty(section)', '\n', ...
        '        cfg.(tok{1}) = tok{2};', '\n', ...
        '    else', '\n', ...
        '        cfg.(section).(strtrim(tok{1})) = tok{2};', '\n', ...
        '    end', '\n', ...
        'end']);
    fclose(fid);

    copyfile('process_smoke_video.m', tmpRoot);

    testCase.TestData.tmpRoot = tmpRoot;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpRoot, 's');
end

function testProcessing(testCase)
    orig_script_dir = testCase.TestData.tmpRoot; %#ok<NASGU>
    scriptPath = fullfile(testCase.TestData.tmpRoot, 'process_smoke_video.m');
    out = evalc(sprintf('run(''%s'')', scriptPath));
    token = regexp(out, 'TEMP_MAT_FILE_SUCCESS:(\S+)', 'tokens', 'once');
    verifyNotEmpty(testCase, token, 'Success marker missing');
    matFile = token{1};
    verifyTrue(testCase, isfile(matFile), 'MAT file not created');
    verifyTrue(testCase, startsWith(matFile, testCase.TestData.tmpRoot), ...
        'MAT file outside temp directory');
    data = load(matFile, 'all_intensities');
    verifyGreaterThan(testCase, numel(data.all_intensities), 0);
end

