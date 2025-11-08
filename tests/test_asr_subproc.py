from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
