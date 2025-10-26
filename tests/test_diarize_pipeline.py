import unittest
from types import SimpleNamespace

from src.lib.diarize.pipeline import annotation_to_turns


class FakeAnnotation:
    def __init__(self):
        self._iter_data = [
            (SimpleNamespace(start=0.0, end=1.0), None, "S0"),
            (SimpleNamespace(start=1.0, end=2.0), None, "S1"),
        ]

    def itertracks(self, *, yield_label: bool = False):
        if not yield_label:
            raise AssertionError("yield_label=True を期待しています")
        for item in self._iter_data:
            yield item


class FakeDiarizeOutput:
    def __init__(self):
        self.speaker_diarization = FakeAnnotation()


class AnnotationToTurnsTests(unittest.TestCase):
    def test_accepts_plain_annotation(self) -> None:
        annotation = FakeAnnotation()
        turns = annotation_to_turns(annotation)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].speaker, "S0")
        self.assertAlmostEqual(turns[1].end, 2.0)

    def test_accepts_diarize_output(self) -> None:
        output = FakeDiarizeOutput()
        turns = annotation_to_turns(output)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].speaker, "S0")

    def test_rejects_unknown_object(self) -> None:
        with self.assertRaises(TypeError):
            annotation_to_turns(object())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
