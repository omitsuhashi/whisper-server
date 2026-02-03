import importlib
import sys
import types
import unittest

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


class TestAudioInspectionRemoved(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        _restore()

    def test_inspection_module_is_gone(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("src.lib.audio.inspection")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
