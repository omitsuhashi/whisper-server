import unittest

import numpy as np

from src.lib.vad import SpeechSegment, VadConfig, detect_voice_segments, segment_waveform


def _tone(duration: float, sample_rate: int = 16000, *, amplitude: float = 0.4) -> np.ndarray:
    total = int(sample_rate * duration)
    t = np.linspace(0, duration, total, endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(duration: float, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(sample_rate * duration), dtype=np.float32)


class VadDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_rate = 16000

    def test_detect_voice_segments_returns_two_regions(self) -> None:
        waveform = np.concatenate([
            _silence(0.2, self.sample_rate),
            _tone(0.6, self.sample_rate),
            _silence(0.3, self.sample_rate),
            _tone(0.5, self.sample_rate),
        ])

        cfg = VadConfig(padding_duration=0.0, min_speech_duration=0.2)
        segments = detect_voice_segments(waveform, self.sample_rate, config=cfg)

        self.assertEqual(len(segments), 2)
        self.assertIsInstance(segments[0], SpeechSegment)

        first, second = segments
        self.assertAlmostEqual(first.start, 0.2, delta=0.05)
        self.assertAlmostEqual(first.end, 0.8, delta=0.05)
        self.assertAlmostEqual(second.start, 1.1, delta=0.05)
        self.assertAlmostEqual(second.end, 1.6, delta=0.05)

    def test_detect_voice_segments_merges_short_gaps(self) -> None:
        waveform = np.concatenate([
            _tone(0.4, self.sample_rate),
            _silence(0.05, self.sample_rate),
            _tone(0.4, self.sample_rate),
        ])

        cfg = VadConfig(padding_duration=0.0, min_silence_duration=0.1)
        segments = detect_voice_segments(waveform, self.sample_rate, config=cfg)

        self.assertEqual(len(segments), 1)
        self.assertGreater(segments[0].duration, 0.75)

    def test_segment_waveform_splits_long_range(self) -> None:
        waveform = np.concatenate([
            _silence(0.1, self.sample_rate),
            _tone(2.0, self.sample_rate),
        ])

        cfg = VadConfig(padding_duration=0.0, min_speech_duration=0.1)
        segments = segment_waveform(
            waveform,
            self.sample_rate,
            vad_config=cfg,
            max_segment_duration=0.5,
        )

        # 0.1s 無音後の 2.0s 音声を 0.5s ごとに区切る → 4 セグメント
        self.assertEqual(len(segments), 4)
        for seg in segments:
            self.assertLessEqual(seg.duration, 0.5002)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
