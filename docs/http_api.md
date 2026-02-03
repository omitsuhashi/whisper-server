# HTTP API

## /transcribe_pcm

### 仕様
- PCM は `s16le mono` を推奨
- `sample_rate` は入力 PCM の Hz。サーバ側で 16kHz にリサンプルされる
- `window_start_seconds/window_end_seconds` はサーバ側で clamp され、レスポンスに effective 値が返る

### 運用の目安（rolling window）
- 長尺を毎回送らず、`(t-25, t)` のような直近窓を送る
- クライアント側で `lookback` を保持して時刻を管理（サーバ側は受け取った範囲のみ処理）

### 例
```bash
# 例: 10 秒の PCM を送るが、サーバは 2-4 秒だけ処理
curl -sS -X POST http://localhost:8000/transcribe_pcm \
  -F file=@audio.pcm \
  -F sample_rate=16000 \
  -F window_start_seconds=2.0 \
  -F window_end_seconds=4.0
```
