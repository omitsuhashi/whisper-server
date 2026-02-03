import unittest

import numpy as np

from src.lib.asr.windowing import slice_waveform_by_seconds


class TestWindowing(unittest.TestCase):
    def test_slice_defaults_to_full_waveform(self) -> None:
        wave = np.arange(16000 * 3, dtype=np.float32)
        result = slice_waveform_by_seconds(
            wave,
            sample_rate=16000,
            start_seconds=None,
            end_seconds=None,
        )
        self.assertEqual(result.start_seconds, 0.0)
        self.assertAlmostEqual(result.end_seconds, 3.0, places=6)
        self.assertAlmostEqual(result.total_seconds, 3.0, places=6)
        self.assertEqual(len(result.waveform), len(wave))
        np.testing.assert_array_equal(result.waveform, wave)

    def test_slice_clamps_out_of_range(self) -> None:
        wave = np.arange(30, dtype=np.float32)
        result = slice_waveform_by_seconds(
            wave,
            sample_rate=10,
            start_seconds=-1.0,
            end_seconds=10.0,
        )
        self.assertEqual(result.start_seconds, 0.0)
        self.assertEqual(result.end_seconds, 3.0)
        self.assertEqual(len(result.waveform), len(wave))

    def test_slice_empty_when_end_before_start(self) -> None:
        wave = np.arange(30, dtype=np.float32)
        result = slice_waveform_by_seconds(
            wave,
            sample_rate=10,
            start_seconds=2.5,
            end_seconds=1.0,
        )
        self.assertEqual(result.start_seconds, 2.5)
        self.assertEqual(result.end_seconds, 2.5)
        self.assertEqual(len(result.waveform), 0)

    def test_invalid_sample_rate_raises(self) -> None:
        wave = np.arange(10, dtype=np.float32)
        with self.assertRaises(ValueError):
            slice_waveform_by_seconds(
                wave,
                sample_rate=0,
                start_seconds=None,
                end_seconds=None,
            )
