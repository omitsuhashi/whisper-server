import asyncio
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

# mlx_whisper stub
original_mlx = sys.modules.get("mlx_whisper")
original_mlx_audio = sys.modules.get("mlx_whisper.audio")
mlx_stub = types.ModuleType("mlx_whisper")
audio_stub = types.ModuleType("mlx_whisper.audio")
audio_stub.SAMPLE_RATE = 16000
mlx_stub.transcribe = lambda *args, **kwargs: None
mlx_stub.audio = audio_stub
sys.modules["mlx_whisper.audio"] = audio_stub
sys.modules["mlx_whisper"] = mlx_stub


def _restore() -> None:
    if original_mlx is not None:
        sys.modules["mlx_whisper"] = original_mlx
    else:
        sys.modules.pop("mlx_whisper", None)
    if original_mlx_audio is not None:
        sys.modules["mlx_whisper.audio"] = original_mlx_audio
    else:
        sys.modules.pop("mlx_whisper.audio", None)


class TestTranscribePCM(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        _restore()

    def _pcm_bytes(self, samples: int = 1600) -> bytes:
        payload = (np.sin(np.linspace(0, np.pi, samples)) * 1000).astype(np.int16)
        return payload.tobytes()

    @mock.patch("src.lib.asr.pipeline.transcribe")
    def test_transcribe_pcm_ok(self, mock_transcribe: mock.Mock) -> None:
        mock_transcribe.return_value = {
            "text": "ok",
            "segments": [],
            "language": "ja",
            "duration": 0.1,
        }
        from src.cmd.http import create_app

        client = TestClient(create_app())
        res = client.post(
            "/transcribe_pcm",
            files={"file": ("audio.pcm", self._pcm_bytes(), "application/octet-stream")},
            data={"sample_rate": "16000", "model": "fake-model", "language": "ja"},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertIsInstance(payload, list)
        self.assertEqual(payload[0]["text"], "ok")

    def test_transcribe_pcm_invalid_length(self) -> None:
        from src.cmd.http import create_app

        client = TestClient(create_app())
        res = client.post(
            "/transcribe_pcm",
            files={"file": ("audio.pcm", b"\x00", "application/octet-stream")},
            data={"sample_rate": "16000"},
        )
        self.assertEqual(res.status_code, 400)


class TestTranscribePCMConcurrency(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        _restore()

    def _pcm_bytes(self, samples: int = 1600) -> bytes:
        payload = (np.sin(np.linspace(0, np.pi, samples)) * 1000).astype(np.int16)
        return payload.tobytes()

    async def test_transcribe_pcm_runs_asr_serially(self) -> None:
        from httpx import ASGITransport, AsyncClient

        from src.cmd.http import create_app
        from src.lib.asr.models import TranscriptionResult

        app = create_app()
        counter = {"active": 0, "max_active": 0}
        counter_lock = threading.Lock()

        def _fake_chunked(
            waveform: np.ndarray,
            *,
            options,
            name: str,
            chunk_seconds: float = 25.0,
            overlap_seconds: float = 1.0,
            vad_config=None,
        ) -> TranscriptionResult:
            _ = (waveform, options, chunk_seconds, overlap_seconds, vad_config)
            with counter_lock:
                counter["active"] += 1
                counter["max_active"] = max(counter["max_active"], counter["active"])
            time.sleep(0.15)
            with counter_lock:
                counter["active"] -= 1
            return TranscriptionResult(
                filename=name,
                text="ok",
                language="ja",
                duration=0.15,
                segments=[],
            )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            with mock.patch("src.cmd.http.transcribe_waveform_chunked", side_effect=_fake_chunked):
                pcm = self._pcm_bytes(samples=16000 * 2)

                async def _post_once():
                    return await client.post(
                        "/transcribe_pcm",
                        files={"file": ("audio.pcm", pcm, "application/octet-stream")},
                        data={"sample_rate": "16000", "chunk_seconds": "1"},
                    )

                responses = await asyncio.gather(_post_once(), _post_once())

        self.assertTrue(all(res.status_code == 200 for res in responses))
        self.assertEqual(counter["max_active"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
