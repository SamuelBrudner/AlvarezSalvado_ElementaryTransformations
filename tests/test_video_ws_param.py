import os
import re

def test_video_ws_parameter():
    with open(os.path.join('Code', 'navigation_model_vec.m')) as f:
        text = f.read()
    matches = list(re.finditer(r"case {'video'}", text))
    assert len(matches) >= 3, 'expected at least three video case statements'
    start = matches[2].start()
    snippet = text[start:start+1000]
    assert re.search(r"if exist\('params','var'\).*?ws = params.ws;", snippet, re.S), \
        'video case should use ws from params when provided'

