import os
import unittest

class TestNavigationModelVecStreamFile(unittest.TestCase):
    def test_file_exists_and_function_name(self):
        path = os.path.join('Code', 'navigation_model_vec_stream.m')
        self.assertTrue(os.path.isfile(path), 'navigation_model_vec_stream.m should exist')
        with open(path) as f:
            first_line = f.readline().strip()
        self.assertTrue(first_line.startswith('function out = navigation_model_vec_stream'),
                        'First function must be navigation_model_vec_stream')

if __name__ == '__main__':
    unittest.main()
