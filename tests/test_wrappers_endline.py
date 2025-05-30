import os
import unittest

class TestWrapperEndsWithNewline(unittest.TestCase):
    def test_wrapper_ends_with_newline(self):
        for fname in ["run_full_batch.sh", "run_test_batch.sh"]:
            with open(fname, "rb") as f:
                content = f.read()
            self.assertTrue(
                content.endswith(b"\n"), f"{fname} should end with a newline"
            )

if __name__ == "__main__":
    unittest.main()
