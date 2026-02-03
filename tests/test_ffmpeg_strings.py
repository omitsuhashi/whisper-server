import subprocess
import unittest


class TestFfmpegStrings(unittest.TestCase):
    def test_ffmpeg_grep_is_empty(self) -> None:
        result = subprocess.run(
            ["git", "grep", "-nE", "ffmpeg|ffprobe", "--", "src/lib"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
