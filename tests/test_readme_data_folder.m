function tests = test_readme_data_folder
    tests = functiontests(localfunctions);
end

function testDataDirectoryMention(testCase)
    txt = fileread('README.md');
    testCase.verifyTrue(contains(lower(txt), 'data/'), ...
        'README should mention data directory');
end
