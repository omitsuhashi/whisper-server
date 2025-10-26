import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.cmd import cli
from src.lib.video import SampledFrame


class CLIFrameExtractionTests(unittest.TestCase):
    @mock.patch("src.cmd.cli.cv2.imwrite", return_value=True)
    @mock.patch("src.cmd.cli.sample_key_frames")
    def test_run_cli_frames_outputs_metadata(
        self,
        mock_sample_key_frames: mock.Mock,
        mock_imwrite: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "meeting.mp4"
            video_path.write_bytes(b"dummy")
            output_dir = Path(tmpdir) / "frames"
            output_dir.mkdir()
            (output_dir / "old.png").write_bytes(b"older")

            sample_frame = SampledFrame(
                index=15,
                timestamp=2.5,
                image=np.zeros((12, 12, 3), dtype=np.uint8),
            )
            mock_sample_key_frames.return_value = [sample_frame]

            args = cli.parse_args(
                [
                    "frames",
                    str(video_path),
                    "--output-dir",
                    str(output_dir),
                    "--image-format",
                    "jpg",
                    "--overwrite",
                ]
            )

            results = cli.run_cli(args)

            self.assertEqual(len(results), 1)
            extraction = results[0]
            self.assertEqual(extraction.video, video_path)
            self.assertEqual(len(extraction.frames), 1)
            frame_info = extraction.frames[0]
            self.assertAlmostEqual(frame_info.timestamp, 2.5, places=2)
            self.assertEqual(frame_info.frame_index, 15)
            expected_dir = output_dir / "meeting"
            self.assertEqual(frame_info.path.parent, expected_dir)
            expected_name = "meeting_00002500_000015_0000.jpg"
            self.assertEqual(frame_info.path.name, expected_name)

            mock_sample_key_frames.assert_called_once_with(
                video_path,
                min_scene_span=1.0,
                diff_threshold=0.3,
                max_frames=None,
            )
            mock_imwrite.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
