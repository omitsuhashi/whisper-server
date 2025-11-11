-- pgvector を有効化してベクトル演算を利用可能にする
CREATE EXTENSION IF NOT EXISTS vector;

-- Postgres 内で BM25 を実行したい場合、ParadeDB の pg_search 拡張を追加インストールする。
-- 参考: https://docs.neon.tech/extensions/paradedb/
-- 1. parade_db schema を導入する SQL をここに追加するか、別スクリプトを配置する。
-- 2. `CREATE EXTENSION IF NOT EXISTS pg_search;`
