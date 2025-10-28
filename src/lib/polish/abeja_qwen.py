from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Sequence

import httpx

from .models import PolishedSentence

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api-inference.huggingface.co/models/abeja/ABEJA-Qwen2.5-7b-Japanese-v0.1"
"""既定の Hugging Face Inference API エンドポイント。"""

TOKEN_ENV_VARS: Sequence[str] = (
    "ABEJA_QWEN_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)
"""利用可能なトークン環境変数の優先順位。"""


class AbejaQwenError(RuntimeError):
    """ABEJA-Qwen 推論呼び出しの失敗を表す例外。"""


class AbejaQwenPolisher:
    """ABEJA-Qwen2.5-7B-Japanese を利用した校正クライアント。"""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        token: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_interval: float = 0.5,
        disable_cache: bool = True,
    ) -> None:
        self.endpoint = endpoint or os.getenv("ABEJA_QWEN_ENDPOINT") or DEFAULT_ENDPOINT
        self.token = token or self._resolve_token()
        if not self.token:
            raise ValueError(
                "ABEJA-Qwen API トークンが指定されていません。環境変数 HF_TOKEN などを設定してください。"
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.disable_cache = disable_cache

    def polish(
        self,
        sentences: Sequence[PolishedSentence],
        *,
        style: str = "ですます",
        extra_instructions: str | None = None,
        parameters: Dict[str, object] | None = None,
    ) -> List[PolishedSentence]:
        """文列を LLM で校正して返す。"""

        if not sentences:
            return []

        prompt = self._build_prompt(sentences, style=style, extra_instructions=extra_instructions)
        payload_parameters: Dict[str, object] = {
            "max_new_tokens": 800,
            "temperature": 0.3,
            "top_p": 0.95,
            "return_full_text": False,
        }
        if parameters:
            payload_parameters.update(parameters)

        payload = {"inputs": prompt, "parameters": payload_parameters}
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        if self.disable_cache:
            headers["x-use-cache"] = "0"

        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status < 500 or attempt == self.max_retries:
                    raise AbejaQwenError(f"ABEJA-Qwen API が HTTP {status} を返しました。") from exc
                logger.warning("ABEJA-Qwen API が一時的に失敗 (%s)。リトライします。", status)
            except httpx.RequestError as exc:  # ネットワークエラー
                if attempt == self.max_retries:
                    raise AbejaQwenError("ABEJA-Qwen API への接続に失敗しました。") from exc
                logger.warning("ABEJA-Qwen API への接続が失敗しました。リトライします。")
            else:
                data = response.json()
                generated = self._extract_generated_text(data)
                if generated is None:
                    raise AbejaQwenError("ABEJA-Qwen API の応答を解釈できませんでした。")
                return self._merge_with_original(sentences, generated)

            time.sleep(self.retry_interval * (attempt + 1))

        raise AbejaQwenError("ABEJA-Qwen API の呼び出しに連続して失敗しました。")

    def _build_prompt(
        self,
        sentences: Sequence[PolishedSentence],
        *,
        style: str,
        extra_instructions: str | None,
    ) -> str:
        """LLM に渡すプロンプトを組み立てる。"""

        formatted_sentences = "\n".join(
            f"{index + 1}. {sentence.text.strip()}" for index, sentence in enumerate(sentences)
        )
        style_note = "敬体（ですます調）" if style == "ですます" else style

        instruction_lines = [
            "あなたは日本語の専門校正者です。以下の文を意味を変えずに自然で統一感のある文章へ校正してください。",
            f"文体は必ず「{style_note}」に合わせてください。",
            "固有名詞・数値・時間表記は変えないでください。",
            "入力された文の順序は保ち、文の数を変更しないでください。",
            "出力は JSON オブジェクトのみとし、前後に説明文を付けないでください。",
            "出力形式: {\"sentences\": [{\"index\": 1, \"text\": \"...\"}, ...]}",
        ]
        if extra_instructions:
            instruction_lines.append(f"追加指示: {extra_instructions.strip()}")

        instruction_block = "\n".join(instruction_lines)
        prompt = f"{instruction_block}\n\n校正対象:\n{formatted_sentences}\n\nJSON:"
        return prompt

    def _extract_generated_text(self, data: object) -> str | None:
        """Inference API の応答から生成テキストを抽出する。"""

        if isinstance(data, dict):
            error = data.get("error")
            if error:
                raise AbejaQwenError(f"ABEJA-Qwen API エラー: {error}")
            generated_text = data.get("generated_text")
            if isinstance(generated_text, str):
                return generated_text
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                generated_text = item.get("generated_text")
                if isinstance(generated_text, str):
                    return generated_text
                payload = item.get("output_text") or item.get("text")
                if isinstance(payload, str):
                    return payload
        return None

    def _merge_with_original(
        self,
        sentences: Sequence[PolishedSentence],
        generated_text: str,
    ) -> List[PolishedSentence]:
        """LLM の結果を既存の時刻情報へマージする。"""

        parsed = self._parse_generated_json(generated_text)
        if parsed is None:
            logger.warning("ABEJA-Qwen の応答を JSON として解析できませんでした。元の文を返します。")
            return list(sentences)

        mapping: Dict[int, str] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            text = item.get("text")
            if isinstance(index, int) and isinstance(text, str):
                mapping[index] = text.strip()
            elif isinstance(index, str) and isinstance(text, str):
                if index.isdigit():
                    mapping[int(index)] = text.strip()

        polished: List[PolishedSentence] = []
        for offset, original in enumerate(sentences, start=1):
            new_text = mapping.get(offset, original.text)
            polished.append(
                PolishedSentence.model_validate(
                    {"start": original.start, "end": original.end, "text": new_text.strip() or original.text}
                )
            )
        return polished

    def _parse_generated_json(self, generated: str) -> List[dict] | None:
        """出力テキストから JSON を抽出する。"""

        snippet = generated.strip()
        start = snippet.find("{")
        end = snippet.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            payload = json.loads(snippet[start : end + 1])
        except json.JSONDecodeError:
            return None

        sentences = payload.get("sentences")
        if isinstance(sentences, list):
            return sentences
        return None

    @staticmethod
    def _resolve_token() -> str | None:
        """利用可能な環境変数からトークンを取得する。"""

        for key in TOKEN_ENV_VARS:
            value = os.getenv(key)
            if value:
                return value
        return None


__all__ = ["AbejaQwenPolisher", "AbejaQwenError", "DEFAULT_ENDPOINT"]
