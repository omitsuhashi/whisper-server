from __future__ import annotations

import atexit
import os
import sys
import types
import unittest

os.environ.setdefault("MLX_USE_CPU", "1")

original_mlx = sys.modules.get("mlx_whisper")
original_mlx_audio = sys.modules.get("mlx_whisper.audio")

mlx_stub = types.ModuleType("mlx_whisper")
audio_stub = types.ModuleType("mlx_whisper.audio")
audio_stub.SAMPLE_RATE = 16000
mlx_stub.transcribe = lambda *args, **kwargs: None
mlx_stub.audio = audio_stub
sys.modules["mlx_whisper.audio"] = audio_stub
sys.modules["mlx_whisper"] = mlx_stub


def _restore_modules() -> None:
    if original_mlx is not None:
        sys.modules["mlx_whisper"] = original_mlx
    else:
        sys.modules.pop("mlx_whisper", None)

    if original_mlx_audio is not None:
        sys.modules["mlx_whisper.audio"] = original_mlx_audio
    else:
        sys.modules.pop("mlx_whisper.audio", None)


atexit.register(_restore_modules)

from fastapi.testclient import TestClient  # noqa: E402

from src.cmd.http import create_app  # noqa: E402


class TestCorrectorHTTPEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def test_correct_endpoint_inserts_sentence_terminator(self) -> None:
        response = self.client.post(
            "/correct",
            json={
                "text": "これはテスト",
                "options": {"aggressive_kuten": True},
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["text"], "これはテスト。")
        self.assertEqual(data["source_text"], "これはテスト")
        self.assertEqual(data["patch_count"], 1)
        patch = data["patches"][0]
        self.assertEqual(patch["replacement"], "。")
        self.assertEqual(patch["span"]["start"], len("これはテスト"))
        self.assertTrue(data["options"]["aggressive_kuten"])

    def test_correct_endpoint_returns_empty_patches_when_not_needed(self) -> None:
        response = self.client.post(
            "/correct",
            json={
                "text": "了解。",
                "options": {"aggressive_kuten": True},
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["text"], "了解。")
        self.assertEqual(data["patch_count"], 0)
        self.assertEqual(data["patches"], [])

    def test_correct_endpoint_rejects_empty_text(self) -> None:
        response = self.client.post(
            "/correct",
            json={"text": ""},
        )
        self.assertEqual(response.status_code, 422)
