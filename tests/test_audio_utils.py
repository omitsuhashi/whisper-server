import unittest

import numpy as np

from src.lib.audio import AudioDecodeError, coerce_to_bytes, decode_audio_bytes, decode_pcm_s16le_bytes


class AudioUtilsTests(unittest.TestCase):
    def test_coerce_to_bytes_handles_memoryview(self) -> None:
        data = bytearray(b"abc")
        self.assertEqual(coerce_to_bytes(memoryview(data)), b"abc")

    def test_decode_audio_bytes_empty_raises(self) -> None:
        with self.assertRaises(AudioDecodeError) as ctx:
            decode_audio_bytes(b"", sample_rate=16000)
        self.assertEqual(ctx.exception.kind, "empty-input")

    def test_decode_pcm_s16le_bytes_empty_raises(self) -> None:
        with self.assertRaises(AudioDecodeError) as ctx:
            decode_pcm_s16le_bytes(b"", sample_rate=16000)
        self.assertEqual(ctx.exception.kind, "empty-input")

    def test_decode_pcm_s16le_bytes_decodes_pcm(self) -> None:
        pcm = np.array([-32768, 0, 32767], dtype=np.int16).tobytes()
        waveform = decode_pcm_s16le_bytes(pcm, sample_rate=16000)
        expected = np.array([-1.0, 0.0, 32767.0 / 32768.0], dtype=np.float32)
        np.testing.assert_allclose(waveform, expected, rtol=1e-5, atol=1e-6)

    def test_decode_pcm_s16le_bytes_resamples_when_requested(self) -> None:
        pcm = np.array([0, 1000, 2000, 3000], dtype=np.int16).tobytes()
        waveform = decode_pcm_s16le_bytes(
            pcm,
            sample_rate=4,
            target_sample_rate=8,
        )
        self.assertEqual(waveform.shape[0], 8)
        self.assertAlmostEqual(waveform[0], 0.0, places=6)
        self.assertAlmostEqual(waveform[-1], 3000.0 / 32768.0, places=6)

    def test_decode_pcm_s16le_bytes_rejects_odd_length(self) -> None:
        with self.assertRaises(AudioDecodeError) as ctx:
            decode_pcm_s16le_bytes(b"\x00", sample_rate=16000)
        self.assertEqual(ctx.exception.kind, "invalid-length")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
