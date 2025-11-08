from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.lib.audio.inspection import is_silent_audio


class IsSilentAudioCacheTests(unittest.TestCase):
    def test_uses_cache_after_first_decode(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"dummy")
            tmp_path = Path(tmp.name)

        decode_calls = {"count": 0}

        def fake_decode(_path: Path) -> np.ndarray:
            decode_calls["count"] += 1
            return np.zeros(16, dtype=np.float32)

        try:
            with patch("src.lib.audio.inspection._decode_audio_bytes", side_effect=fake_decode):
                self.assertTrue(is_silent_audio(tmp_path))
                # 2回目はキャッシュが効くため decode は呼ばれない
                self.assertTrue(is_silent_audio(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertEqual(decode_calls["count"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
