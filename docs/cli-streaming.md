# CLI streaming 利用例

## 推奨: auto（ffmpeg decode）

```bash
cat sample.wav | python -m src.cmd.cli stream --plain-text --stream-interval 1 --stream-input auto
```

## raw PCM（s16le / mono）

```bash
ffmpeg -i sample.wav -f s16le -ac 1 -ar 16000 - \
  | python -m src.cmd.cli stream --plain-text --stream-interval 1 --stream-input pcm --stream-sample-rate 16000
```
