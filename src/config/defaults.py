from __future__ import annotations

import os
from urllib.parse import quote_plus

"""アプリ全体で共有する既定値。"""

DEFAULT_LANGUAGE = "ja"
"""書き起こし時に利用するデフォルト言語。"""

DEFAULT_MODEL_NAME = "mlx-community/whisper-large-v3-mlx"
"""デフォルトで使用するWhisper Largeモデル名。"""


def _build_db_url() -> str:
    explicit = os.getenv("KB_DB_URL")
    if explicit:
        return explicit

    user = os.getenv("KB_DB_USER", "kb")
    password = os.getenv("KB_DB_PASSWORD", "kbpass")
    host = os.getenv("KB_DB_HOST", "localhost")
    port = os.getenv("KB_DB_PORT", "5433")
    name = os.getenv("KB_DB_NAME", "kb")
    return (
        f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{quote_plus(name)}"
    )


KB_DB_CONFIG = {
    "host": os.getenv("KB_DB_HOST", "localhost"),
    "port": int(os.getenv("KB_DB_PORT", "5433")),
    "user": os.getenv("KB_DB_USER", "kb"),
    "password": os.getenv("KB_DB_PASSWORD", "kbpass"),
    "name": os.getenv("KB_DB_NAME", "kb"),
    "url": _build_db_url(),
}
"""知識ベース Postgres 用の接続情報。"""
