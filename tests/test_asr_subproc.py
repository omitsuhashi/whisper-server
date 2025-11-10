from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from src.lib.asr.models import TranscriptionResult


class ASRSubprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_env = {}
        for key in ("ASR_SUBPROC_TEST_MODE", "ASR_SUBPROC_IDLE_SECONDS"):
            self._original_env[key] = os.environ.get(key)
            os.environ[key] = "0"
        os.environ["ASR_SUBPROC_TEST_MODE"] = "1"
        os.environ["ASR_SUBPROC_IDLE_SECONDS"] = "0.2"
        from src.lib.asr import subproc

        self.module = subproc
        self._original_attrs = {
            "_TEST_MODE": getattr(subproc, "_TEST_MODE", False),
            "_IDLE_TIMEOUT": getattr(subproc, "_IDLE_TIMEOUT", 0.0),
        }
        subproc._TEST_MODE = True  # type: ignore[attr-defined]
        subproc._IDLE_TIMEOUT = 0.2  # type: ignore[attr-defined]
        subproc._shutdown_worker_locked()  # type: ignore[attr-defined]

    def tearDown(self) -> None:
        try:
            self.module._shutdown_worker_locked()  # type: ignore[attr-defined]
        except Exception:
            pass
        if "_TEST_MODE" in self._original_attrs:
            self.module._TEST_MODE = self._original_attrs["_TEST_MODE"]  # type: ignore[attr-defined]
        if "_IDLE_TIMEOUT" in self._original_attrs:
            self.module._IDLE_TIMEOUT = self._original_attrs["_IDLE_TIMEOUT"]  # type: ignore[attr-defined]
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_transcribe_returns_stubbed_result(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            result = self.module.transcribe_paths_via_worker(
                [tmp.name],
                model_name="dummy",
                language="ja",
                task=None,
            )

        self.assertEqual(len(result), 1)
        self.assertIn(Path(tmp.name).name, result[0].text)

    def test_idle_timeout_respawns_worker(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            first = self.module.transcribe_paths_via_worker(
                [tmp.name],
                model_name="dummy",
                language=None,
                task=None,
            )
            pid1 = int(first[0].text.split(":")[1])
            time.sleep(0.5)  # wait longer than idle timeout
            second = self.module.transcribe_paths_via_worker(
                [tmp.name],
                model_name="dummy",
                language=None,
                task=None,
            )
            pid2 = int(second[0].text.split(":")[1])

        self.assertNotEqual(pid1, pid2)

    def test_decode_options_forwarded(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            result = self.module.transcribe_paths_via_worker(
                [tmp.name],
                model_name="dummy",
                language="ja",
                task=None,
                initial_prompt="foo",
            )

        self.assertEqual(len(result), 1)

    @mock.patch("src.lib.asr.subproc._resolve_chunk_transcriber")
    def test_handle_transcribe_uses_chunking_when_configured(self, mock_resolve: mock.Mock) -> None:
        previous_mode = self.module._TEST_MODE
        self.module._TEST_MODE = False  # type: ignore[attr-defined]
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                tmp_path = Path(tmp.name)
                fake_result = TranscriptionResult(filename=tmp_path.name, text="done", language="ja", segments=[])
                mock_chunked = mock.Mock(return_value=[fake_result])
                mock_resolve.return_value = mock_chunked
                rows = self.module._handle_transcribe(  # type: ignore[attr-defined]
                    {
                        "paths": [tmp.name],
                        "model_name": "demo",
                        "language": "ja",
                        "task": None,
                        "decode_options": {"initial_prompt": "foo"},
                        "chunk_seconds": 10.0,
                        "overlap_seconds": 2.0,
                    }
                )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["filename"], tmp_path.name)
            mock_chunked.assert_called_once()
            args, kwargs = mock_chunked.call_args
            self.assertEqual([tmp_path], list(args[0]))
            self.assertEqual(kwargs["chunk_seconds"], 10.0)
            self.assertEqual(kwargs["overlap_seconds"], 2.0)
            self.assertIn("initial_prompt", kwargs)
        finally:
            self.module._TEST_MODE = previous_mode  # type: ignore[attr-defined]

    @mock.patch("src.lib.asr.subproc._transcribe_full_paths")
    @mock.patch("src.lib.asr.subproc._resolve_chunk_transcriber")
    def test_handle_transcribe_skips_chunking_when_zero(
        self,
        mock_resolve: mock.Mock,
        mock_full: mock.Mock,
    ) -> None:
        previous_mode = self.module._TEST_MODE
        self.module._TEST_MODE = False  # type: ignore[attr-defined]
        try:
            result = TranscriptionResult(filename="full.wav", text="ok", language="ja", segments=[])
            mock_full.return_value = [result]
            rows = self.module._handle_transcribe(  # type: ignore[attr-defined]
                {
                    "paths": ["dummy.wav"],
                    "model_name": "demo",
                    "language": "ja",
                    "task": None,
                    "decode_options": {},
                    "chunk_seconds": 0.0,
                    "overlap_seconds": 1.0,
                }
            )
        finally:
            self.module._TEST_MODE = previous_mode  # type: ignore[attr-defined]

        self.assertEqual(len(rows), 1)
        mock_full.assert_called_once()
        mock_resolve.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
