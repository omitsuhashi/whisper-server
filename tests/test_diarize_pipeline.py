import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from src.lib.diarize.pipeline import annotation_to_turns, load_pipeline
from src.lib.diarize.options import DiarizeOptions


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


class LoadPipelineTests(unittest.TestCase):
    @mock.patch("src.lib.diarize.pipeline._resolve_token", return_value="dummy")
    @mock.patch("src.lib.diarize.pipeline._resolve_device", return_value="mps")
    @mock.patch("src.lib.diarize.pipeline._PyannotePipeline")
    def test_load_pipeline_uses_torch_device(
        self,
        mock_pipeline_cls: mock.Mock,
        _mock_resolve_device: mock.Mock,
        _mock_resolve_token: mock.Mock,
    ) -> None:
        pipeline_instance = mock.Mock()
        mock_pipeline_cls.from_pretrained.return_value = pipeline_instance

        result = load_pipeline(DiarizeOptions())

        self.assertIs(result, pipeline_instance)
        mock_pipeline_cls.from_pretrained.assert_called_once()
        pipeline_instance.to.assert_called_once()
        (device_arg,) = pipeline_instance.to.call_args.args
        self.assertIsInstance(device_arg, torch.device)
        self.assertEqual(device_arg.type, "mps")

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
