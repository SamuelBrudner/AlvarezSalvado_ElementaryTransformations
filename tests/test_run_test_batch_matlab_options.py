with open("run_test_batch.sh") as f:
    content = f.read()

def test_matlab_options_unquoted():
    assert 'MATLAB_OPTIONS="-nodisplay -nosplash"' not in content
    assert 'MATLAB_OPTIONS=-nodisplay\\ -nosplash' in content
