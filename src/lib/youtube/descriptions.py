from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence
import time

import requests

_API_ROOT = "https://www.googleapis.com/youtube/v3"
_MAX_BATCH = 50


@dataclass(slots=True)
class VideoDescription:
    """YouTube 動画のスニペット情報を保持するデータクラス。"""

    video_id: str
    title: str | None
    description: str | None
    published_at: str | None


class YouTubeDescriptionFetcher:
    """YouTube Data API v3 からアップロード動画の説明文を取得するヘルパー。"""

    def __init__(
        self,
        api_key: str,
        *,
        pause_seconds: float = 0.1,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._pause_seconds = max(0.0, pause_seconds)
        self._session = session or requests.Session()

    def fetch(self, *, handle: str | None = None, channel_id: str | None = None) -> list[VideoDescription]:
        """指定チャンネルの全動画をまとめて取得する。"""

        return list(self.iter(handle=handle, channel_id=channel_id))

    def iter(self, *, handle: str | None = None, channel_id: str | None = None) -> Iterator[VideoDescription]:
        """アップロード動画を逐次 yield するジェネレーター。"""

        channel_id = channel_id or self._channel_id_from_handle(handle)
        uploads_pid = self._uploads_playlist(channel_id)
        ids_iter = self._iter_playlist_video_ids(uploads_pid)
        for chunk in _chunked(ids_iter, _MAX_BATCH):
            yield from self._fetch_video_snippets(chunk)
    def fetch_and_save(
        self,
        output_path: str | Path,
        *,
        handle: str | None = None,
        channel_id: str | None = None,
        ensure_ascii: bool = False,
    ) -> Path:
        """取得したデータを JSON として保存し、書き込んだパスを返す。"""

        descriptions = [asdict(item) for item in self.fetch(handle=handle, channel_id=channel_id)]
        path = Path(output_path)
        path.write_text(
            json.dumps(descriptions, ensure_ascii=ensure_ascii, indent=2),
            encoding="utf-8",
        )
        return path

    # --- 内部処理 ---------------------------------------------------------
    def _channel_id_from_handle(self, handle: str | None) -> str:
        if not handle:
            raise ValueError("handle または channel_id のいずれかを指定してください")
        normalized = handle.lstrip("@")
        payload = self._request(
            "search",
            part="snippet",
            q=normalized,
            type="channel",
            maxResults=5,
        )
        items = payload.get("items", [])
        if not items:
            raise LookupError(f"チャンネルが見つかりません: {handle}")
        return items[0]["snippet"]["channelId"]
    def _uploads_playlist(self, channel_id: str) -> str:
        payload = self._request("channels", part="contentDetails", id=channel_id)
        try:
            return payload["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        except (IndexError, KeyError) as exc:  # pragma: no cover - 想定外のレスポンス
            raise LookupError(f"uploads プレイリストを取得できませんでした: {channel_id}") from exc

    def _iter_playlist_video_ids(self, playlist_id: str) -> Iterator[str]:
        page_token: str | None = None
        while True:
            params: dict[str, str | int] = {
                "part": "contentDetails",
                "playlistId": playlist_id,
                "maxResults": _MAX_BATCH,
            }
            if page_token:
                params["pageToken"] = page_token
            payload = self._request("playlistItems", **params)
            for item in payload.get("items", []):
                yield item["contentDetails"]["videoId"]
            page_token = payload.get("nextPageToken")
            if not page_token:
                break
            self._sleep()
    def _fetch_video_snippets(self, video_ids: Sequence[str]) -> Iterator[VideoDescription]:
        payload = self._request("videos", part="snippet", id=",".join(video_ids))
        for item in payload.get("items", []):
            snippet = item.get("snippet", {})
            yield VideoDescription(
                video_id=item["id"],
                title=snippet.get("title"),
                description=snippet.get("description"),
                published_at=snippet.get("publishedAt"),
            )
        self._sleep()

    def _request(self, endpoint: str, **params: str | int) -> dict:
        params_with_key = {"key": self._api_key, **params}
        resp = self._session.get(
            f"{_API_ROOT}/{endpoint}",
            params=params_with_key,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _sleep(self) -> None:
        if self._pause_seconds:
            time.sleep(self._pause_seconds)


def _chunked(iterable: Iterable[str], size: int) -> Iterator[list[str]]:
    chunk: list[str] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


__all__ = ["VideoDescription", "YouTubeDescriptionFetcher"]
