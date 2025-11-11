from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.defaults import KB_DB_CONFIG

engine = create_engine(
    KB_DB_CONFIG["url"],
    future=True,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_engine():
    """外部から Engine を参照したい場合のアクセサ。"""

    return engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """トランザクションを伴うセッションを提供する。"""

    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database() -> None:
    """必要なテーブルを作成する。"""

    from . import models  # noqa: WPS433  遅延インポートで循環を避ける

    models.Base.metadata.create_all(bind=engine)
