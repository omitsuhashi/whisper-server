import importlib.util
import unittest


class TestCliRemoved(unittest.TestCase):
    def test_cli_module_is_gone(self) -> None:
        self.assertIsNone(importlib.util.find_spec("src.cmd.cli"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
