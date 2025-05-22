function tests = test_data_dirs_exist
    tests = functiontests(localfunctions);
end

function testDirectoriesExist(~)
    assert(isfolder(fullfile('data','raw')), 'data/raw directory missing');
    assert(isfolder(fullfile('data','processed')), 'data/processed directory missing');
end
