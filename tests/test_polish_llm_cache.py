import os
import importlib
import types
import unittest


class TestLLMPolishCache(unittest.TestCase):
    def setUp(self) -> None:
        # 保険で環境をクリア
        os.environ.pop("LLM_POLISH_MODEL", None)

    def _fake_loader(self, counter: dict):
        def _load(model_id: str):
            counter[model_id] = counter.get(model_id, 0) + 1
            # ダミー model/tokenizer
            return object(), types.SimpleNamespace(apply_chat_template=lambda *a, **k: "<prompt>")

        def _generate(model, tokenizer, **kwargs):  # noqa: ARG001 - テスト用ダミー
            # JSON 風の想定出力
            return '{"sentences": [{"index": 1, "text": "OK"}]}'

        def _sampler_factory(**kwargs):  # noqa: ARG001 - テスト用ダミー
            return object()

        return _load, _generate, _sampler_factory

    def test_cache_enabled_then_unload(self):
        os.environ["LLM_POLISH_CACHE"] = "1"
        module = importlib.import_module("src.lib.polish.llm_client")
        importlib.reload(module)
        counter = {}
        load, gen, make_sampler = self._fake_loader(counter)

        p1 = module.LLMPolisher(model_id="foo", load_fn=load, generate_fn=gen, sampler_factory=make_sampler)
        p2 = module.LLMPolisher(model_id="foo", load_fn=load, generate_fn=gen, sampler_factory=make_sampler)
        self.assertEqual(counter.get("foo", 0), 1, "キャッシュ有効時は1回のみロード")

        # 明示アンロード
        removed = module.unload_llm_models("foo")
        self.assertEqual(removed, 1)

        # 再度生成すると再ロードされる
        p3 = module.LLMPolisher(model_id="foo", load_fn=load, generate_fn=gen, sampler_factory=make_sampler)
        self.assertEqual(counter.get("foo", 0), 2)

        # 明示 close は例外にならないことのみ確認
        p1.close()
        p2.close()
        p3.close()

    def test_cache_disabled_instance_scoped(self):
        os.environ["LLM_POLISH_CACHE"] = "0"
        module = importlib.import_module("src.lib.polish.llm_client")
        importlib.reload(module)
        counter = {}
        load, gen, make_sampler = self._fake_loader(counter)

        p1 = module.LLMPolisher(model_id="bar", load_fn=load, generate_fn=gen, sampler_factory=make_sampler)
        p2 = module.LLMPolisher(model_id="bar", load_fn=load, generate_fn=gen, sampler_factory=make_sampler)
        self.assertEqual(counter.get("bar", 0), 2, "非キャッシュ時は毎回ロード")

        # キャッシュがないので unload は 0
        removed = module.unload_llm_models("bar")
        self.assertEqual(removed, 0)

        # 明示 close は例外にならないことのみ確認
        p1.close()
        p2.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

