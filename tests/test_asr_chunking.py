import tempfile
import unittest

from src.lib.asr.chunking import transcribe_paths_chunked
from src.lib.asr.models import TranscriptionResult
from tests.test_asr_stream import _generate_wav_bytes


class ChunkingDecodeOptionsTests(unittest.TestCase):
    def test_decode_options_forwarded_to_transcribe_all_bytes(self) -> None:
        captured: dict | None = None

        def fake_transcribe(blobs, *, model_name, language, task, names, **decode_options):
            nonlocal captured
            captured = decode_options
            return [
                TranscriptionResult(
                    filename=names[0],
                    text="ok",
                    language=language,
                    segments=[],
                )
            ]

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(_generate_wav_bytes())
            tmp.flush()
            results = transcribe_paths_chunked(
                [tmp.name],
                model_name="demo",
                language="ja",
                task=None,
                chunk_seconds=0,
                overlap_seconds=0,
                transcribe_all_bytes_fn=fake_transcribe,
                initial_prompt="foo",
            )

        self.assertIsNotNone(captured)
        assert captured is not None
        self.assertEqual(captured.get("initial_prompt"), "foo")
        self.assertEqual(results[0].text, "ok")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
