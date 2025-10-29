from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Sequence, Tuple
import gc

from importlib import import_module

from .exceptions import LLMPolishError
from .models import PolishedSentence

logger = logging.getLogger(__name__)

mlx_load = None  # type: ignore[assignment]
mlx_generate = None  # type: ignore[assignment]
mlx_make_sampler = None  # type: ignore[assignment]

_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}
_MODEL_LOCK = threading.Lock()
_USE_CACHE = os.getenv("LLM_POLISH_CACHE", "1").lower() not in {"0", "false", "off", "no"}


def _ensure_mlx_loaded() -> None:
    global mlx_load, mlx_generate, mlx_make_sampler
    if mlx_load is not None and mlx_generate is not None and mlx_make_sampler is not None:
        return
    try:
        module = import_module("mlx_lm")
        mlx_load = getattr(module, "load")
        mlx_generate = getattr(module, "generate")
        sampler_module = import_module("mlx_lm.sample_utils")
        mlx_make_sampler = getattr(sampler_module, "make_sampler")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "mlx-lm がロードできません。'pip install -U mlx mlx-lm' を実行してから再度試してください。"
        ) from exc

class LLMPolisher:
    """mlx-lm を利用したローカル LLM 校正クライアント。"""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 800,
        load_fn: Any | None = None,
        generate_fn: Any | None = None,
        sampler_factory: Any | None = None,
    ) -> None:
        self.model_id = model_id or os.getenv("LLM_POLISH_MODEL")
        if not self.model_id:
            raise ValueError("LLM_POLISH_MODEL 環境変数でモデル ID を指定してください。")

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        if load_fn is None or generate_fn is None or sampler_factory is None:
            _ensure_mlx_loaded()
        self._load_fn = load_fn or mlx_load
        self._generate_fn = generate_fn or mlx_generate
        self._sampler_factory = sampler_factory or mlx_make_sampler
        if self._sampler_factory is None:
            raise ValueError(
                "mlx-lm の make_sampler が利用できません。'pip install -U mlx mlx-lm' を実行してから再度試してください。"
            )

        # キャッシュ利用可否に応じてロード戦略を切替
        if _USE_CACHE:
            with _MODEL_LOCK:
                if self.model_id not in _MODEL_CACHE:
                    _MODEL_CACHE[self.model_id] = self._load_fn(self.model_id)
                self._model, self._tokenizer = _MODEL_CACHE[self.model_id]
        else:
            # 非キャッシュ: インスタンスローカルに読み込み、スコープ終了時に解放可能
            self._model, self._tokenizer = self._load_fn(self.model_id)

    def polish(
        self,
        sentences: Sequence[PolishedSentence],
        *,
        style: str = "ですます",
        extra_instructions: str | None = None,
        parameters: Dict[str, Any] | None = None,
    ) -> List[PolishedSentence]:
        if not sentences:
            return []

        prompt = self._build_prompt(sentences, style=style, extra_instructions=extra_instructions)

        extra_parameters: Dict[str, Any] = dict(parameters or {})

        max_tokens_value = extra_parameters.pop("max_tokens", self.max_tokens)
        if max_tokens_value is None:
            max_tokens_value = self.max_tokens

        sampler_override = extra_parameters.pop("sampler", None)

        effective_temp = extra_parameters.pop("temperature", self.temperature)
        effective_temp = extra_parameters.pop("temp", effective_temp)
        effective_top_p = extra_parameters.pop("top_p", self.top_p)

        sampler_kwargs: Dict[str, Any] = {
            "temp": effective_temp,
            "top_p": effective_top_p,
        }
        for key in ("min_p", "min_tokens_to_keep", "top_k", "xtc_probability", "xtc_threshold", "xtc_special_tokens"):
            if key in extra_parameters:
                sampler_kwargs[key] = extra_parameters.pop(key)

        if sampler_override is not None:
            sampler = sampler_override
        else:
            sampler = self._sampler_factory(**sampler_kwargs)

        gen_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": int(max_tokens_value),
            "sampler": sampler,
        }
        if extra_parameters:
            gen_kwargs.update(extra_parameters)

        try:
            output = self._generate_fn(self._model, self._tokenizer, **gen_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LLMPolishError("LLM 生成に失敗しました。") from exc

        if isinstance(output, list):
            generated_text = "".join(str(item) for item in output)
        else:
            generated_text = str(output)

        parsed = self._parse_generated_json(generated_text)
        if parsed is None:
            logger.warning("LLM 応答を JSON として解析できませんでした。元の文を返します。")
            return list(sentences)

        mapping: Dict[int, str] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            text = item.get("text")
            if isinstance(index, int) and isinstance(text, str):
                mapping[index] = text.strip()
            elif isinstance(index, str) and isinstance(text, str) and index.isdigit():
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

    def _build_prompt(
        self,
        sentences: Sequence[PolishedSentence],
        *,
        style: str,
        extra_instructions: str | None,
    ) -> str:
        style_note = "敬体（ですます調）" if style == "ですます" else style
        formatted_sentences = "\n".join(f"{idx + 1}. {sentence.text.strip()}" for idx, sentence in enumerate(sentences))

        system_prompt = "日本語の校正・構成エディタ。利用者の文章を読みやすく整えます。"
        instructions = [
            "以下の文を意味を変えずに、構成と論理を自然に整えてください。",
            f"文体は必ず「{style_note}」に揃えてください。",
            "固有名詞・数字・時間表現は変更しないでください。",
            "文の順序と文数は維持してください。",
            '出力は JSON 形式: {"sentences": [{"index": 1, "text": "..."}]} のみを返してください。',
        ]
        if extra_instructions:
            instructions.append(f"追加指示: {extra_instructions.strip()}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(instructions) + f"\n\n対象:\n{formatted_sentences}\n\nJSON:"},
        ]
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def _parse_generated_json(generated: str) -> List[dict] | None:
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

    # 明示解放（ベストエフォート）
    def close(self) -> None:
        try:
            # インスタンス保持参照を破棄
            self._model = None  # type: ignore[attr-defined]
            self._tokenizer = None  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - 失敗しても致命的ではない
            pass
        gc.collect()


def unload_llm_models(model_id: str | None = None) -> int:
    """LLM モデルキャッシュを解放する。

    戻り値は解放できたエントリ数。
    """
    removed = 0
    with _MODEL_LOCK:
        global _MODEL_CACHE
        if model_id is None:
            removed = len(_MODEL_CACHE)
            _MODEL_CACHE = {}
        else:
            if model_id in _MODEL_CACHE:
                _MODEL_CACHE.pop(model_id, None)
                removed = 1
    gc.collect()
    return removed


__all__ = ["LLMPolisher", "LLMPolishError", "unload_llm_models"]
