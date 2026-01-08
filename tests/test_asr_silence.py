import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from src.lib.asr.service import transcribe_prepared_audios
from src.lib.audio import prepare_audio


def _write_silent_wav(path: Path, *, duration_sec: float = 0.1, sample_rate: int = 16000) -> None:
    total_samples = int(sample_rate * duration_sec)
    payload = b"\x00\x00" * total_samples
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(payload)


class TestAsrSilenceGuard(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not found")

    def test_silence_produces_empty_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "silence.wav"
            _write_silent_wav(path)
            with mock.patch.dict("os.environ", {"ASR_DUMP_AUDIO": "0"}, clear=False):
                prepared = prepare_audio(path, path.name)

            self.assertTrue(prepared.silent)
            transcribe_all = mock.Mock(return_value=[])

            results = transcribe_prepared_audios(
                [prepared],
                model_name="fake-model",
                language="ja",
                transcribe_all_fn=transcribe_all,
            )

            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.filename, "silence.wav")
            self.assertEqual(result.text, "")
            self.assertEqual(result.segments, [])
            self.assertEqual(result.duration, 0.0)
            transcribe_all.assert_not_called()
