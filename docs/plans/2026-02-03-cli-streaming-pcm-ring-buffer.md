# CLI Streaming PCM Ring Buffer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** ストリーミング入力を PCM に正規化し、固定長 ring buffer を導入して stream-interval の入力サイズを上限固定にする。

**Architecture:** CLI の stream-interval では、stdin から PCM を読む reader と ring buffer を使って waveform を保持する。`auto` は ffmpeg を 1 回起動して s16le/mono/16k に変換し、`pcm` は raw s16le/mono を直接読む。ASR 呼び出しは現行の `transcribe_all_bytes_fn` を維持し、ring buffer を WAV へエンコードして渡す（ISSUE-2 で waveform 直渡しに切替予定）。

**Tech Stack:** Python 3.13, numpy, unittest, ffmpeg (auto のみ)

### Task 1: PCM reader / ring buffer のテスト追加

**Files:**
- Create: `tests/test_asr_streaming_input.py`

**Step 1: Write the failing test**

```python
class TestPcmStreamReader(unittest.TestCase):
    def test_reader_handles_odd_bytes_across_reads(self):
        samples = np.array([0, 1000], dtype=np.int16)
        stream = io.BytesIO(samples.tobytes())
        reader = PcmStreamReader(stream, sample_rate=16000, target_sample_rate=16000)
        first, eof1 = reader.read_waveform(3)
        second, eof2 = reader.read_waveform(3)
        self.assertFalse(eof1)
        self.assertFalse(eof2)
        self.assertEqual(first.size + second.size, 2)

class TestPcmRingBuffer(unittest.TestCase):
    def test_ring_buffer_keeps_last_samples(self):
        ring = PcmRingBuffer(max_samples=3)
        ring.append(np.array([1.0, 2.0], dtype=np.float32))
        ring.append(np.array([3.0, 4.0], dtype=np.float32))
        self.assertEqual(ring.total_samples, 4)
        self.assertEqual(ring.samples.tolist(), [2.0, 3.0, 4.0])
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_asr_streaming_input.py -q`
Expected: FAIL (module/class not found)

**Step 3: Write minimal implementation**

```python
@dataclass
class PcmRingBuffer:
    max_samples: int
    samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    total_samples: int = 0

    def append(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return
        self.total_samples += int(chunk.size)
        self.samples = np.concatenate([self.samples, chunk]) if self.samples.size else chunk
        if self.max_samples > 0 and self.samples.size > self.max_samples:
            self.samples = self.samples[-self.max_samples:]

class PcmStreamReader:
    def __init__(self, stream: BinaryIO, *, sample_rate: int, target_sample_rate: int | None):
        self._stream = stream
        self._leftover = b""
        self._sample_rate = sample_rate
        self._target_sample_rate = target_sample_rate

    def read_waveform(self, chunk_size: int) -> tuple[np.ndarray, bool]:
        chunk = self._stream.read(chunk_size)
        if not chunk:
            return np.zeros(0, dtype=np.float32), True
        if self._leftover:
            chunk = self._leftover + chunk
            self._leftover = b""
        if len(chunk) % 2 != 0:
            self._leftover = chunk[-1:]
            chunk = chunk[:-1]
        if not chunk:
            return np.zeros(0, dtype=np.float32), False
        waveform = decode_pcm_s16le_bytes(
            chunk,
            sample_rate=self._sample_rate,
            target_sample_rate=self._target_sample_rate,
        )
        return waveform, False
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_asr_streaming_input.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_asr_streaming_input.py src/lib/asr/streaming_input.py
git commit -m "✨ stream: add pcm reader and ring buffer"
```

### Task 2: CLI stream-interval に PCM reader / ring buffer を組み込み

**Files:**
- Modify: `src/cmd/cli.py`

**Step 1: Write the failing test**

```python
class CliStreamingWindowTests(unittest.TestCase):
    def test_streaming_transcription_uses_fixed_window_for_pcm(self):
        sr = 16000
        window_seconds = 0.01
        window_samples = int(sr * window_seconds)
        samples = np.arange(window_samples * 3, dtype=np.int16)
        fake_stdin = FakeStdin(samples.tobytes())
        captured = {}

        def fake_transcribe_all_bytes(payloads, **kwargs):
            captured["payload"] = payloads[0]
            return [TranscriptionResult(filename="stdin", text="ok")]

        with mock.patch.object(sys, "stdin", fake_stdin):
            cli._streaming_transcription(
                model_name="fake",
                language="ja",
                task=None,
                name="stdin",
                chunk_size=window_samples * 2,
                interval=0.0,
                transcribe_all_bytes_fn=fake_transcribe_all_bytes,
                decode_options={},
                emit_stdout=False,
                window_seconds=window_seconds,
                stream_input="pcm",
                stream_sample_rate=sr,
            )

        with closing(wave.open(io.BytesIO(captured["payload"]), "rb")) as wav:
            self.assertLessEqual(wav.getnframes(), window_samples)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_cli_streaming_window.py -q`
Expected: FAIL (args/behavior not implemented)

**Step 3: Write minimal implementation**

```python
# _configure_stream_parser に引数追加
parser.add_argument("--stream-window-seconds", ...)
parser.add_argument("--stream-lookback-seconds", ...)
parser.add_argument("--stream-input", choices=["auto", "pcm"], ...)
parser.add_argument("--stream-sample-rate", type=int, default=16000, ...)

# _streaming_transcription に window/input の引数追加
# interval>0 の場合のみ PCM reader + ring buffer を使用
# ring buffer -> WAV bytes を作り transcribe_all_bytes_fn に渡す
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_cli_streaming_window.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cmd/cli.py tests/test_cli_streaming_window.py
git commit -m "✨ stream: use pcm ring buffer for interval"
```
