import unittest

import numpy as np

from src.lib.audio import AudioDecodeError, coerce_to_bytes, decode_audio_bytes


class AudioUtilsTests(unittest.TestCase):
    def test_coerce_to_bytes_handles_memoryview(self) -> None:
        data = bytearray(b"abc")
        self.assertEqual(coerce_to_bytes(memoryview(data)), b"abc")

    def test_decode_audio_bytes_empty_raises(self) -> None:
        with self.assertRaises(AudioDecodeError) as ctx:
            decode_audio_bytes(b"", sample_rate=16000)
        self.assertEqual(ctx.exception.kind, "empty-input")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
