import os


def test_citation_file_exists():
    assert os.path.isfile('CITATION.cff'), 'CITATION.cff is missing'


def test_codemeta_file_exists():
    assert os.path.isfile('codemeta.json'), 'codemeta.json is missing'
