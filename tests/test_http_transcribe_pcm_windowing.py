import unittest
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from src.cmd.http import create_app
from src.lib.asr.models import TranscriptionResult, TranscriptionSegment


class TestPCMWindowing(unittest.TestCase):
    @mock.patch("src.cmd.http.analyze_transcription_quality", return_value=None)
    @mock.patch("src.cmd.http.apply_filler_removal", side_effect=lambda res, enabled: res)
    @mock.patch("src.cmd.http.transcribe_waveform_chunked")
    def test_window_slices_waveform_length(self, mock_chunked, *_mocks) -> None:
        def fake_chunked(waveform, *, options, name, chunk_seconds, overlap_seconds):
            self.assertEqual(int(waveform.shape[-1]), 16000 * 2)
            return TranscriptionResult(
                filename=name,
                text="ok",
                language="ja",
                segments=[TranscriptionSegment(start=0.0, end=0.1, text="ok")],
            )

        mock_chunked.side_effect = fake_chunked

        app = create_app()
        client = TestClient(app)

        pcm = np.zeros(16000 * 10, dtype=np.int16).tobytes()
        resp = client.post(
            "/transcribe_pcm",
            files={"file": ("test.pcm", pcm, "application/octet-stream")},
            data={
                "sample_rate": "16000",
                "chunk_seconds": "5",
                "window_start_seconds": "2.0",
                "window_end_seconds": "4.0",
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body[0]["window_start_seconds"], 2.0)
        self.assertEqual(body[0]["window_end_seconds"], 4.0)

    @mock.patch("src.cmd.http.analyze_transcription_quality", return_value=None)
    @mock.patch("src.cmd.http.apply_filler_removal", side_effect=lambda res, enabled: res)
    @mock.patch("src.cmd.http.transcribe_waveform_chunked")
    def test_window_clamps_and_reflects_effective(self, mock_chunked, *_mocks) -> None:
        def fake_chunked(waveform, *, options, name, chunk_seconds, overlap_seconds):
            self.assertEqual(int(waveform.shape[-1]), 16000 * 10)
            return TranscriptionResult(
                filename=name,
                text="ok",
                language="ja",
                segments=[TranscriptionSegment(start=0.0, end=0.1, text="ok")],
            )

        mock_chunked.side_effect = fake_chunked

        app = create_app()
        client = TestClient(app)

        pcm = np.zeros(16000 * 10, dtype=np.int16).tobytes()
        resp = client.post(
            "/transcribe_pcm",
            files={"file": ("test.pcm", pcm, "application/octet-stream")},
            data={
                "sample_rate": "16000",
                "chunk_seconds": "5",
                "window_start_seconds": "-1",
                "window_end_seconds": "999",
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body[0]["window_start_seconds"], 0.0)
        self.assertEqual(body[0]["window_end_seconds"], 10.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
