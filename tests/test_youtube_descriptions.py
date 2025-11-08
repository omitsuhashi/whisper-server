from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.lib.youtube import YouTubeDescriptionFetcher


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - 成功ケースのみ
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def get(self, url: str, *, params: dict[str, Any] | None = None, timeout: int | None = None):
        self.calls.append((url, params))
        endpoint = url.rsplit("/", 1)[-1]
        if endpoint == "search":
            return _FakeResponse(
                {
                    "items": [
                        {"snippet": {"channelId": "CHAN123"}},
                    ]
                }
            )
        if endpoint == "channels":
            return _FakeResponse(
                {
                    "items": [
                        {
                            "contentDetails": {
                                "relatedPlaylists": {"uploads": "UPLOADS123"},
                            }
                        }
                    ]
                }
            )
        if endpoint == "playlistItems":
            return _FakeResponse(
                {
                    "items": [
                        {"contentDetails": {"videoId": "video-a"}},
                        {"contentDetails": {"videoId": "video-b"}},
                    ]
                }
            )
        if endpoint == "videos":
            return _FakeResponse(
                {
                    "items": [
                        {
                            "id": "video-a",
                            "snippet": {
                                "title": "A",
                                "description": "desc-a",
                                "publishedAt": "2024-01-01T00:00:00Z",
                            },
                        },
                        {
                            "id": "video-b",
                            "snippet": {
                                "title": "B",
                                "description": "desc-b",
                                "publishedAt": "2024-01-02T00:00:00Z",
                            },
                        },
                    ]
                }
            )
        raise AssertionError(f"unexpected endpoint: {endpoint}")


class YouTubeDescriptionFetcherTests(unittest.TestCase):
    def test_fetch_and_save_from_handle(self) -> None:
        session = _FakeSession()
        fetcher = YouTubeDescriptionFetcher(
            api_key="test-key",
            pause_seconds=0.0,
            session=session,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.json"
            path = fetcher.fetch_and_save(
                output_path=output,
                handle="@example",
            )
            self.assertEqual(path, output)
            payload = json.loads(output.read_text())
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["title"], "A")
        self.assertEqual(payload[1]["description"], "desc-b")
        self.assertEqual(session.calls[0][0].rsplit("/", 1)[-1], "search")
        self.assertEqual(session.calls[0][1]["key"], "test-key")
    def test_fetch_requires_handle_or_channel_id(self) -> None:
        fetcher = YouTubeDescriptionFetcher(api_key="k", pause_seconds=0.0, session=_FakeSession())
        with self.assertRaises(ValueError):
            list(fetcher.iter())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
