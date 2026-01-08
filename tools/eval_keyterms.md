# 評価ループ（KTA）

## 目的
- 改善の有無を AVG_KTA で再現可能に比較する。

## データセット形式（JSONL）
- 1行1サンプル。
- audio_path は JSONL のあるディレクトリからの相対パスでも可。

```jsonl
{"id":"sample-01","audio_path":"../media/sample.mp4","key_terms":["termA","termB"]}
```

## A/B 実験（deep context）
- deep context off: prompt_agenda / prompt_participants / prompt_products / prompt_style を付けない。
- deep context on: 上記を付与する（必要なら prompt_terms / prompt_dictionary も併用）。

### A（off）
```bash
make eval-kta JSONL=tools/eval_keyterms.sample.jsonl MODE=A
```

### B（on）
```bash
make eval-kta JSONL=tools/eval_keyterms.sample.jsonl MODE=B \
  PROMPT_AGENDA="議題A,議題B" \
  PROMPT_PARTICIPANTS="参加者A,参加者B" \
  PROMPT_PRODUCTS="製品A,製品B" \
  PROMPT_STYLE="句読点は、。" \
  PROMPT_TERMS="重要語1,重要語2" \
  PROMPT_DICTIONARY="表記固定1,表記固定2"
```

## パラメータ比較（chunk/overlap）
```bash
make eval-kta JSONL=tools/eval_keyterms.sample.jsonl MODE=A \
  CHUNK_SECONDS=25 OVERLAP_SECONDS=1
```

## 検証フロー
1. 20件で回す → AVG_KTA を記録。
2. 50件へ拡張して再実行。

## 補足
- `tools/eval_keyterms.sample.jsonl` は形式確認用のダミーです。実評価では `key_terms` と音声パスを差し替えてください。
