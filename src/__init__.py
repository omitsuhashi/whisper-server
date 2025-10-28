"""root package for project modules."""

from __future__ import annotations

import os


# Hugging Face tokenizers はフォーク後に並列モードが有効なままだと警告を出すため
# デフォルトで無効化しておき、サーバー/CLI 双方で安定稼働させる。
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
