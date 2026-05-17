# Phase 29: ETL Expansion - Context

**Gathered:** 2026-05-17
**Status:** Ready for planning

<domain>
## Phase Boundary

三連複(n_odds_sanren)/馬連(n_odds_umaren)/三連単(n_odds_sanrentan)オッズをEveryDB2からParquetファイルとして抽出し、Phase 32(市場クロス整合性特徴量)で利用可能なデータ基盤を整える。

**In scope:**
- ETL config (etl_tables.yaml) PK定義の修正 (umaban1/2/3 → kumi)
- run_etl.py --mode full --tables による6テーブル抽出 (本体3 + head3)
- DataRepository クラスの新規作成 (新オッズ3メソッドのみ)
- カバレッジ検証の run_etl.py 組み込み (行数・年度カバレッジ・欠損率)
- テスト作成 (DataRepository + カバレッジ検証)

**Out of scope:**
- 既存 readers.py の変更・DataRepositoryへの移行
- 特徴量計算 (Phase 31/32)
- delta ETL対応 (s_テーブル)
- DataRepositoryによる既存関数のラッピング

</domain>

<decisions>
## Implementation Decisions

### データアクセス層
- **D-01:** 新規 `src/db/repository.py` に `DataRepository` クラスを作成。3メソッドのみ実装: `load_trio_odds()`, `load_exacta_odds()`, `load_trifecta_odds()`
- **D-02:** 既存 `readers.py` は変更しない。将来的な移行は別フェーズで対応
- **D-03:** DataRepository は ParquetStore を内部で使用し、`_date_filters` + `_coerce_types` パターンを踏襲

### カバレッジ検証方式
- **D-04:** カバレッジ検証を `run_etl.py` に組み込む。ETL実行後、抽出テーブルの行数・年度カバレッジ・欠損率を自動出力
- **D-05:** 検証基準: 2015-2025カバー、欠損率30%以下 (ETL-04)

### _headテーブルの扱い
- **D-06:** 本体 + head 両方抽出。ETL configに既に定義済みの6テーブル全てを対象とする
- **D-07:** headテーブルは datakubun (5=確定/9=最終) と sanrenflag/sanrentanflag/umarenflag を含む

### ETL実行スコープ
- **D-08:** 特定テーブルのみ抽出: `run_etl.py --mode full --start 20150101 --end 20251231 --tables odds_sanren odds_sanren_head odds_umaren odds_umaren_head odds_sanrentan odds_sanrentan_head`

### etl_tables.yaml PK定義修正
- **D-09:** 以下のテーブルPKを修正 (umaban1/2/3 → kumi):
  - n_odds_sanren, s_odds_sanren: pk [year, monthday, jyocd, kaiji, nichiji, racenum, kumi]
  - n_odds_sanrentan, s_odds_sanrentan: pk [year, monthday, jyocd, kaiji, nichiji, racenum, kumi]
  - n_odds_umaren, s_odds_umaren: pk [year, monthday, jyocd, kaiji, nichiji, racenum, kumi]
- **D-10:** 既存 n_odds_wide のPK定義も確認 (現在 umaban1/umaban2 → 実際は kumi)

### DB構造確認結果
- **D-11:** 3テーブル共通カラム構造: makedate, year, monthday, jyocd, kaiji, nichiji, racenum, kumi, odds, ninki
- **D-12:** kumiフォーマット: 三連複="010203"(6桁), 馬連="0102"(4桁), 三連単="010203"(6桁・順列)
- **D-13:** n_odds_wide は oddslow/oddshigh の2列だが、sanren/umaren/sanrentanは単一 odds 列
- **D-14:** データは継続蓄積中 (2025年5月時点で sanren=454K, sanrentan=2.6M, umaren=108K 行)

### Claude's Discretion
- DataRepository の内部実装詳細 (型アノテーション、キャッシュ等)
- カバレッジレポートの出力フォーマット
- テストケースの具体的な設計

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### ETL / Data Layer
- `config/etl_tables.yaml` — 103テーブル定義。PK定義修正対象のテーブルを含む
- `src/db/etl.py` — ETLエンジン本体。`_read_db_table`, `_compute_race_date`, `_compute_race_id`, `run_full_load` の処理フロー
- `src/db/readers.py` — 既存データアクセスパターン。`load_wide_odds()` 等のreader関数群。DataRepositoryはこのパターンを踏襲
- `src/db/parquet_store.py` — 低レベルParquet I/O。ParquetStoreのcategory/nameアドレッシング
- `scripts/run_etl.py` — ETLエントリポイント。--tables指定時の動作確認が必要

### Requirements
- `.planning/REQUIREMENTS.md` §ETL Expansion — ETL-01〜ETL-04

### DB Schema (実測値 — 本CONTEXT.mdで確認済み)
- n_odds_sanren: (makedate, year, monthday, jyocd, kaiji, nichiji, racenum, kumi, odds, ninki) — 全てvarchar
- n_odds_sanrentan: 同上
- n_odds_umaren: 同上
- n_odds_wide: (makedate, year, monthday, jyocd, kaiji, nichiji, racenum, kumi, oddslow, oddshigh, ninki)
- n_odds_sanren_head: (recordspec, datakubun, makedate, year, monthday, jyocd, kaiji, nichiji, racenum, happyotime, torokutosu, syussotosu, sanrenflag, totalhyosusanren)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/db/readers.py::_date_filters(start, end)`: 日付フィルタ生成。DataRepositoryでそのまま利用可能
- `src/db/readers.py::_coerce_types(df)`: 型変換・フォールバック。DataRepositoryでそのまま利用可能
- `src/db/etl.py::_compute_race_date(df)`, `_compute_race_id(df)`: ETLで自動付与される派生列。Parquetファイルにはrace_date/race_idが含まれる
- `src/db/parquet_store.py::ParquetStore`: category/nameアドレッシング。"odds"カテゴリに配置

### Established Patterns
- reader関数パターン: `store.read(category, name, filters=...)` → `_coerce_types(df)` → return
- ETL config駆動: etl_tables.yamlのエントリがあれば`run_etl.py`が自動処理
- 全テストmock使用: DB不要テストパターン

### Integration Points
- `src/db/readers.py`: 新しいreader関数を追加する場合の接続点 (今回はDataRepositoryに分離)
- `config/etl_tables.yaml`: PK修正の対象 (n_odds_sanren/sanrentan/umaren + s_系)
- `scripts/run_etl.py`: --tables指定で6テーブルのみ抽出
- Phase 32 (Market Cross-Consistency): DataRepository経由でtrio/exactaオッズを利用

</code_context>

<specifics>
## Specific Ideas

- DataRepository クラスはPhase 29で新規作成だが、将来的にreaders.py関数群を段階的に移行する拡張点として設計
- 三連単(n_odds_sanrentan)は260万行とデータ量が最大。Parquet圧縮後のサイズに注意
- n_odds_wide のPK定義も umaban1/umaban2 となっているが実際は kumi。ついでに確認・修正すべき

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope
</deferred>

---

*Phase: 29-ETL Expansion*
*Context gathered: 2026-05-17*
