import unittest
from unittest import mock

import numpy as np

from src.lib.vad import VadConfig, detect_voice_segments


class TestVadFusion(unittest.TestCase):
    @mock.patch("src.lib.vad.transformer.estimate_speech_probabilities")
    def test_fusion_detects_only_when_transformer_confident(self, mock_tf: mock.Mock) -> None:
        def side_effect(waveform, *, sample_rate, frame_bounds, hop_duration, model_name):
            n = len(frame_bounds)
            probs = [0.0] * n
            for i in range(n // 2, n):
                probs[i] = 1.0
            return probs

        mock_tf.side_effect = side_effect

        wave = np.full(16000, 0.01, dtype=np.float32)  # 1 sec
        cfg = VadConfig(
            fusion_enabled=True,
            transformer_model="dummy",
            fusion_energy_weight=0.4,
            fusion_transformer_weight=0.6,
            fusion_on_threshold=0.6,
            fusion_off_threshold=0.4,
            fusion_gate_threshold=0.15,
            min_speech_duration=0.1,
            padding_duration=0.0,
        )
        segments = detect_voice_segments(wave, 16000, config=cfg)
        self.assertEqual(len(segments), 1)
        self.assertGreater(segments[0].start, 0.35)

    @mock.patch("src.lib.vad.transformer.estimate_speech_probabilities", side_effect=RuntimeError("boom"))
    def test_fusion_falls_back_to_energy_when_transformer_fails(self, mock_tf: mock.Mock) -> None:
        wave = np.full(16000, 0.01, dtype=np.float32)
        cfg = VadConfig(
            fusion_enabled=True,
            transformer_model="dummy",
            min_speech_duration=0.1,
            padding_duration=0.0,
        )
        segments = detect_voice_segments(wave, 16000, config=cfg)
        self.assertTrue(segments)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
