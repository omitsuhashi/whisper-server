import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from src.cmd import cli
from src.lib.asr import TranscriptionResult
from src.lib.diarize import (
    DiarizationResult,
    SpeakerAnnotatedTranscript,
    SpeakerSegment,
    SpeakerTurn,
)


class CLIDiarizeTests(unittest.TestCase):
    @mock.patch("src.cmd.cli.attach_speaker_labels")
    @mock.patch("src.cmd.cli.diarize_all")
    @mock.patch("src.cmd.cli.transcribe_all")
    def test_run_cli_with_diarize_option(
        self,
        mock_transcribe_all: mock.Mock,
        mock_diarize_all: mock.Mock,
        mock_attach: mock.Mock,
    ) -> None:
        args = cli.parse_args(
            [
                "files",
                "sample.wav",
                "--diarize",
                "--diarize-num-speakers",
                "2",
            ]
        )

        transcription = TranscriptionResult(filename="sample.wav", text="hello")
        mock_transcribe_all.return_value = [transcription]

        diarization = DiarizationResult(
            filename="sample.wav",
            turns=[SpeakerTurn(start=0.0, end=1.0, speaker="S0")],
        )
        mock_diarize_all.return_value = [diarization]

        annotated = SpeakerAnnotatedTranscript(
            filename="sample.wav",
            segments=[
                SpeakerSegment(start=0.0, end=1.0, text="hello", speaker="S0"),
            ],
        )
        mock_attach.return_value = annotated

        results = cli.run_cli(args)

        self.assertEqual(results, [transcription])
        mock_transcribe_all.assert_called_once()
        mock_diarize_all.assert_called_once()
        diarize_call = mock_diarize_all.call_args
        self.assertEqual(diarize_call.args[0], ["sample.wav"])
        options = diarize_call.kwargs["options"]
        self.assertEqual(options.num_speakers, 2)
        self.assertFalse(options.require_mps)
        self.assertIn("sample.wav", args._speaker_transcripts)  # type: ignore[attr-defined]
        self.assertIs(args._speaker_transcripts["sample.wav"], annotated)  # type: ignore[index]
        self.assertIn("sample.wav", args._diarization_results)  # type: ignore[attr-defined]

    @mock.patch("src.cmd.cli.attach_speaker_labels")
    @mock.patch("src.cmd.cli.diarize_all")
    @mock.patch("src.cmd.cli.transcribe_all")
    def test_main_prints_speaker_segments(
        self,
        mock_transcribe_all: mock.Mock,
        mock_diarize_all: mock.Mock,
        mock_attach: mock.Mock,
    ) -> None:
        transcription = TranscriptionResult(filename="sample.wav", text="こんにちは")
        mock_transcribe_all.return_value = [transcription]

        diarization = DiarizationResult(
            filename="sample.wav",
            turns=[SpeakerTurn(start=0.0, end=1.0, speaker="S1")],
        )
        mock_diarize_all.return_value = [diarization]

        annotated = SpeakerAnnotatedTranscript(
            filename="sample.wav",
            segments=[
                SpeakerSegment(start=0.0, end=1.0, text="こんにちは", speaker="S1"),
            ],
        )
        mock_attach.return_value = annotated

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["files", "sample.wav", "--diarize", "--show-segments"])

        output = buf.getvalue()
        self.assertIn("話者:", output)
        self.assertIn("S1:", output)
        self.assertIn("こんにちは", output)
        mock_transcribe_all.assert_called_once()
        mock_diarize_all.assert_called_once()
        mock_attach.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
