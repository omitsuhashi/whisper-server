import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.lib.video import SampledFrame, sample_key_frames


class FrameSamplerTests(unittest.TestCase):
    def test_sample_key_frames_detects_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "changes.mp4"
            self._write_video(
                video_path,
                [
                    np.full((120, 160, 3), (0, 0, 255), dtype=np.uint8),
                    np.full((120, 160, 3), (0, 0, 255), dtype=np.uint8),
                    np.full((120, 160, 3), (0, 255, 0), dtype=np.uint8),
                    np.full((120, 160, 3), (0, 255, 0), dtype=np.uint8),
                    np.full((120, 160, 3), (255, 0, 0), dtype=np.uint8),
                    np.full((120, 160, 3), (255, 0, 0), dtype=np.uint8),
                ],
                fps=2.0,
            )

            frames = sample_key_frames(video_path, min_scene_span=0.5, diff_threshold=0.2)

            self.assertGreaterEqual(len(frames), 3)
            self._assert_sample(frames[0], 0, 0.0)
            timestamps = [round(frame.timestamp, 2) for frame in frames]
            self.assertIn(1.0, timestamps)
            self.assertIn(2.0, timestamps)

    def test_sample_key_frames_respects_max_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "limited.mp4"
            frames = [
                np.full((120, 160, 3), value, dtype=np.uint8) for value in [(0, 0, 0), (50, 50, 50), (200, 200, 200)]
            ]
            self._write_video(video_path, frames, fps=1.0)

            sampled = sample_key_frames(video_path, min_scene_span=0.1, diff_threshold=0.1, max_frames=2)
            self.assertEqual(len(sampled), 2)

    def test_sample_key_frames_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            sample_key_frames("missing-file.mp4")

    def _write_video(self, path: Path, frames: list[np.ndarray], fps: float) -> None:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("VideoWriter を初期化できませんでした。")

        for frame in frames:
            writer.write(frame)
        writer.release()

    def _assert_sample(self, sample: SampledFrame, index: int, timestamp: float) -> None:
        self.assertEqual(sample.index, index)
        self.assertAlmostEqual(sample.timestamp, timestamp, places=2)
        self.assertTrue(isinstance(sample.image, np.ndarray))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
