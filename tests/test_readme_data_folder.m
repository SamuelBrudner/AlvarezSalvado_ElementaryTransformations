function tests = test_readme_data_folder
    tests = functiontests(localfunctions);
end

function testDataDirectoryMention(testCase)
    txt = fileread('README.md');
    hasRaw = contains(lower(txt), 'data/raw/');
    hasProcessed = contains(lower(txt), 'data/processed/');
    testCase.verifyTrue(hasRaw && hasProcessed, ...
        'README should mention raw and processed data directories');
end
