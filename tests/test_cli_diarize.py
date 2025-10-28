import io
import os
import tempfile
import unittest
import wave
from dataclasses import dataclass, field
from contextlib import redirect_stdout
from pathlib import Path
from typing import List
from unittest import mock

import numpy as np

from src.cmd import cli


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str


@dataclass
class FakeTranscriptionResult:
    filename: str
    text: str
    language: str | None = None
    segments: List[FakeSegment] = field(default_factory=list)


@dataclass
class FakeSpeakerSegment(FakeSegment):
    speaker: str


@dataclass
class FakeSpeakerAnnotatedTranscript:
    filename: str
    segments: List[FakeSpeakerSegment]

    @property
    def speakers(self) -> List[str]:
        return sorted({segment.speaker for segment in self.segments})


@dataclass
class FakeSpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass
class FakeDiarizationResult:
    filename: str
    turns: List[FakeSpeakerTurn]

    @property
    def speakers(self) -> List[str]:
        return sorted({turn.speaker for turn in self.turns})


class FakeDiarizeOptions:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class CLIDiarizeTests(unittest.TestCase):
    @mock.patch("src.cmd.cli._load_diarize_components")
    @mock.patch("src.cmd.cli._load_asr_components")
    def test_run_cli_with_diarize_option(
        self,
        mock_load_asr: mock.Mock,
        mock_load_diarize: mock.Mock,
    ) -> None:
        temp_audio = _create_temp_wav()
        self.addCleanup(temp_audio.unlink)
        args = cli.parse_args(
            [
                "files",
                str(temp_audio),
                "--diarize",
                "--diarize-num-speakers",
                "2",
            ]
        )

        transcription = FakeTranscriptionResult(filename=temp_audio.name, text="hello")
        mock_transcribe_all = mock.Mock(return_value=[transcription])
        mock_transcribe_all_bytes = mock.Mock()
        mock_load_asr.return_value = (
            FakeTranscriptionResult,
            mock_transcribe_all,
            mock_transcribe_all_bytes,
        )

        diarization = FakeDiarizationResult(
            filename=temp_audio.name,
            turns=[FakeSpeakerTurn(start=0.0, end=1.0, speaker="S0")],
        )
        mock_diarize_all = mock.Mock(return_value=[diarization])
        annotated = FakeSpeakerAnnotatedTranscript(
            filename=temp_audio.name,
            segments=[
                FakeSpeakerSegment(start=0.0, end=1.0, text="hello", speaker="S0"),
            ],
        )
        mock_attach = mock.Mock(return_value=annotated)
        mock_load_diarize.return_value = (
            FakeDiarizeOptions,
            mock_attach,
            mock_diarize_all,
        )

        results = cli.run_cli(args)

        self.assertEqual(results, [transcription])
        mock_transcribe_all.assert_called_once()
        mock_diarize_all.assert_called_once()
        diarize_call = mock_diarize_all.call_args
        self.assertEqual(diarize_call.args[0], [str(temp_audio)])
        options = diarize_call.kwargs["options"]
        self.assertEqual(options.num_speakers, 2)
        self.assertTrue(options.require_mps)
        self.assertIn(temp_audio.name, args._speaker_transcripts)  # type: ignore[attr-defined]
        self.assertIs(args._speaker_transcripts[temp_audio.name], annotated)  # type: ignore[index]
        self.assertIn(temp_audio.name, args._diarization_results)  # type: ignore[attr-defined]

    @mock.patch("src.cmd.cli._load_diarize_components")
    @mock.patch("src.cmd.cli._load_asr_components")
    def test_main_prints_speaker_segments(
        self,
        mock_load_asr: mock.Mock,
        mock_load_diarize: mock.Mock,
    ) -> None:
        temp_audio = _create_temp_wav()
        self.addCleanup(temp_audio.unlink)

        transcription = FakeTranscriptionResult(
            filename=temp_audio.name,
            text="こんにちは",
            segments=[FakeSegment(start=0.0, end=1.0, text="こんにちは")],
        )
        mock_transcribe_all = mock.Mock(return_value=[transcription])
        mock_transcribe_all_bytes = mock.Mock()
        mock_load_asr.return_value = (
            FakeTranscriptionResult,
            mock_transcribe_all,
            mock_transcribe_all_bytes,
        )

        diarization = FakeDiarizationResult(
            filename=temp_audio.name,
            turns=[FakeSpeakerTurn(start=0.0, end=1.0, speaker="S1")],
        )
        mock_diarize_all = mock.Mock(return_value=[diarization])
        annotated = FakeSpeakerAnnotatedTranscript(
            filename=temp_audio.name,
            segments=[
                FakeSpeakerSegment(start=0.0, end=1.0, text="こんにちは", speaker="S1"),
            ],
        )
        mock_attach = mock.Mock(return_value=annotated)
        mock_load_diarize.return_value = (
            FakeDiarizeOptions,
            mock_attach,
            mock_diarize_all,
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["files", str(temp_audio), "--diarize", "--show-segments"])

        output = buf.getvalue()
        self.assertIn("話者:", output)
        self.assertIn("S1:", output)
        self.assertIn("こんにちは", output)
        mock_transcribe_all.assert_called_once()
        mock_diarize_all.assert_called_once()
        mock_attach.assert_called_once()

    @mock.patch("src.cmd.cli._load_diarize_components")
    @mock.patch("src.cmd.cli._load_asr_components")
    def test_run_cli_disallows_non_mps_device(
        self,
        mock_load_asr: mock.Mock,
        mock_load_diarize: mock.Mock,
    ) -> None:
        temp_audio = _create_temp_wav()
        self.addCleanup(temp_audio.unlink)
        args = cli.parse_args(
            [
                "files",
                str(temp_audio),
                "--diarize",
                "--diarize-device",
                "cpu",
            ]
        )

        transcription = FakeTranscriptionResult(filename=temp_audio.name, text="hello")
        mock_transcribe_all = mock.Mock(return_value=[transcription])
        mock_transcribe_all_bytes = mock.Mock()
        mock_load_asr.return_value = (
            FakeTranscriptionResult,
            mock_transcribe_all,
            mock_transcribe_all_bytes,
        )

        mock_load_diarize.return_value = (FakeDiarizeOptions, mock.Mock(), mock.Mock())

        with self.assertRaises(RuntimeError) as ctx:
            cli.run_cli(args)

        self.assertIn("MPS 専用", str(ctx.exception))

    @mock.patch("src.cmd.cli._load_diarize_components")
    def test_diarize_subcommand_outputs_turns(
        self,
        mock_load_diarize: mock.Mock,
    ) -> None:
        captured_kwargs: dict[str, object] = {}

        class CapturingOptions(FakeDiarizeOptions):
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)
                super().__init__(**kwargs)

        fake_result = FakeDiarizationResult(
            filename="sample.wav",
            turns=[FakeSpeakerTurn(start=0.0, end=1.5, speaker="S0")],
        )

        mock_diarize_all = mock.Mock(return_value=[fake_result])
        mock_load_diarize.return_value = (CapturingOptions, mock.Mock(), mock_diarize_all)

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["diarize", "sample.wav", "--show-turns"])

        output = buf.getvalue()
        self.assertIn("sample.wav", output)
        self.assertIn("S0", output)
        self.assertIn("0.00s - 1.50s", output)
        self.assertTrue(captured_kwargs["require_mps"])
        self.assertIsNone(captured_kwargs["device"])
        self.assertEqual(captured_kwargs["sample_rate"], 16000)
        mock_diarize_all.assert_called_once_with(["sample.wav"], options=mock.ANY)

    @mock.patch("src.cmd.cli._load_diarize_components")
    def test_diarize_subcommand_respects_sample_rate(
        self,
        mock_load_diarize: mock.Mock,
    ) -> None:
        captured_kwargs: dict[str, object] = {}

        class CapturingOptions(FakeDiarizeOptions):
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)
                super().__init__(**kwargs)

        mock_load_diarize.return_value = (
            CapturingOptions,
            mock.Mock(),
            mock.Mock(return_value=[]),
        )

        args = cli.parse_args(["diarize", "sample.wav", "--sample-rate", "8000"])
        cli.run_cli(args)
        self.assertEqual(captured_kwargs["sample_rate"], 8000)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
def _create_temp_wav() -> Path:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        payload = (np.sin(np.linspace(0, np.pi, 160)).astype(np.float32) * 1000).astype(np.int16)
        wav.writeframes(payload.tobytes())
    return Path(path)
