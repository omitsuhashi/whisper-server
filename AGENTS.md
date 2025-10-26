# Repository Guidelines

## Language

Japanese

## Project Structure & Module Organization
アプリケーションコードは `src/` に集約され、CLI エントリーポイントは `src/cmd/cli.py`、FastAPI アプリケーションファクトリは `src/cmd/http.py` にあります。音声認識や話者分離などのドメインロジックは `src/lib/asr`・`audio`・`diarize`・`polish` に整理され、共通設定は `src/config` を確認してください。依存関係は `pyproject.toml` と `uv.lock` で管理されるため、ライブラリ更新時は両ファイルを同期させましょう。ユニットテストは `tests/` の `test_*.py` 形式で配置され、サンプル音声は `media/` に保管されています。上位の `Makefile` には開発作業で頻用するターゲットがまとまっています。

## Build, Test, and Development Commands
- `uv sync` : `uv.lock` に記録された依存関係を再現します。`uv` が使えない場合は仮想環境を作成し `pip install -e .` を利用してください。
- `source .venv/bin/activate` : プロジェクト用 virtualenv を有効化し、`python -m src.cmd.cli --help` で CLI オプションを確認します。
- `make cli-files AUDIO="audio.wav" MODEL=medium` : 指定ファイルを CLI パイプラインで書き起こします。
- `make cli-stream MODEL=small LANGUAGE=ja` : 標準入力からの音声ストリームを逐次処理します。
- `make audio-streaming DEVICE=:1` : macOS の入力デバイスを `ffmpeg` で録音しながら CLI ストリームへパイプします。
- `uvicorn src.cmd.http:create_app --reload --port 8000` : HTTP API をホットリロード付きで起動します。
- `make test` または `python -m unittest discover -s tests` : 回帰テストを実行します。

## Coding Style & Naming Conventions
ターゲットは Python 3.13 で、PEP 8 に沿った 4 スペースインデントと snake_case の命名、型ヒントの併用を推奨します。FastAPI のルートは最小限に保ち、処理は `src/lib` のヘルパーへ委譲してください。既存モジュールに合わせて日本語ドキュメントやログ出力を維持し、ユーザー向け文言のローカライズを統一します。

## Testing Guidelines
テスト実行には標準ライブラリの `unittest` を利用します。新規テストファイルは `test_<feature>.py` とし、クラス名は `Test<Subject>` の形式に統一してください。既存例として `tests/test_asr_stream.py` を参照し、共通フィクスチャやヘルパーを再利用します。長尺音声の検証には `media/` のダミー素材を使い、外部サービスに依存しない再現性を確保してください。ストリーミングや話者分離機能を変更する際は、期待する JSON ペイロードや例外ハンドリングを確認する回帰テストを最低一件追加します。

## Commit & Pull Request Guidelines
コミットメッセージは履歴にならい、`✨` や `🔧` のプレフィックスに続けて行動を端的に記述します（日本語歓迎）。件名は 72 文字以内に収め、挙動変更時は本文に背景や影響範囲を追記し、関連 Issue やフォローアップをリンクしてください。Pull Request ではユーザー影響、新規 CLI フラグやエンドポイント、確認に使ったログやスクリーンショットを共有し、`make test` の完了を明記します。

## Security & Configuration Tips
`HF_TOKEN` などの機密値は `.env`（`direnv` 経由で読込）やシェル環境に設定し、リポジトリへコミットしないでください。
