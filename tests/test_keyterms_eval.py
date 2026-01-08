import unittest

from src.lib.eval.keyterms import key_term_accuracy


class TestKeyTermEval(unittest.TestCase):
    def test_kta_hits_all(self) -> None:
        self.assertEqual(key_term_accuracy("git checkout dev", ["git", "checkout", "dev"]), 1.0)

    def test_kta_partial(self) -> None:
        self.assertAlmostEqual(
            key_term_accuracy("git checkout", ["git", "checkout", "dev"]), 2 / 3
        )
