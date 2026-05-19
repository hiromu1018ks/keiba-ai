# Phase 35: ETL Data Foundation - Context

**Gathered:** 2026-05-19
**Status:** Ready for planning

<domain>
## Phase Boundary

HaronTimeL3/L4, LapTime1~25, Jyuni1c~4c をEveryDB2からfloat64としてParquetに抽出し、センチネル値をNaN化し、POST_RACE安全性を確保するETL基盤構築。

**In scope:**
- ETL-01: HaronTimeL3/L4 (SE table) をfloat64変換してentries Parquetに格納、センチネル値(000/999)をNaN化
- ETL-02: LapTime1~25 (RA table) をfloat64変換してraces Parquetに格納、センチネル値(000)をNaN化
- ETL-03: Jyuni1c~4c (SE table) コーナー通過順位を数値化してentries Parquetに格納
- ETL-04: 全新POST_RACE列を domain/types.py の POST_RACE_COLS に登録し、v1.6の3層CI漏洩検出が機能することを確認
- ETL-05: HaronTimeL3/L4の相互排他性を検証し、結果を文書化
- _TABLE_TYPE_RULESに宣言的sentinel ruleを追加
- readers.pyの_INT_COLS/_FLOAT_COLS更新（旧ETL互換性）
- POST_RACE_COLSの重複解消（3箇所→types.pyを唯一の正としimport集約）
- ETL後品質確認（Claude手動検証）

**Out of scope:**
- 特徴量計算 (Phase 36)
- harontime_last3f統合ロジック (Phase 36)
- バックテスト実行 (Phase 38)
- モデル変更・学習
- DataRepositoryへの新規メソッド追加（LapTimeはraces Parquet内、既存load_racesで利用可能）

</domain>

<decisions>
## Implementation Decisions

### LapTime POST_RACE管理
- **D-01:** LapTime1~25（25列全て）をPOST_RACE_COLSに追加。POST_RACE_COLSは16列→41列に拡張
- **D-02:** 既存の3層CI漏洩テストは変更不要。LapTimeはrace-levelのためbuild_all()出力に自然に含まれない。Layer 1テスト（build_all出力検証）とLayer 2テスト（FEATURE_COLS検証）はそのまま通過する
- **D-03:** ETL実行後にClaudeが品質確認（float64型・NaN化の手動検証）
- **D-04:** CI用の自動テストは追加しない（全テストmock使用のため実際のParquet検証は不可）

### HaronTimeL3/L4統合
- **D-05:** HaronTimeL3/L4を別々にfloat64化（coalesceしない）。統合ロジック(harontime_last3f)はPhase 36で検証結果に基づいて決定
- **D-06:** 相互排他性はETL実行後にClaudeが品質確認で検証（L3のみ/L4のみ/両方/なしの4分類分布確認）

### センチネル値NaN化方式
- **D-07:** _TABLE_TYPE_RULESに宣言的sentinel rule型（sentinel_float / sentinel_int）を追加。構造: `{"sentinel_float": {"columns": [...], "sentinels": ("000", "999")}}`。_apply_type_conversionsでsentinel置換→astypeの2段階処理
- **D-08:** センチネル値定義: HaronTimeL3/L4は000と999、LapTime1~25は000、Jyuni2c/3cは000。全てsentinel_floatとして処理（Jyuniもfloat→Int64変換）
- **D-09:** readers.pyの_INT_COLS/_FLOAT_COLSも更新（旧ETL Parquetとの互換性維持）。追加: _FLOAT_COLSにharontimel4 + laptime1~25、_INT_COLSにjyuni2c/jyuni3c

### POST_RACE_COLS重複解消
- **D-10:** POST_RACE_COLS定義をtypes.pyに集約。test_paper_trading_guards.pyとrun_paper_trading.pyの重複定義をimportに変更。DRY原則で将来の変更が1箇所で済む

### Claude's Discretion
- _TABLE_TYPE_RULESのsentinel_float/sentinel_intルールの具体的な実装（sentinel置換のロジック）
- _apply_type_conversions内での処理順序（sentinel置換→型変換のパイプライン）
- readers.py更新の具体的な列リスト
- ETL品質確認の具体的な検証手順
- テストケースの設計（sentinel rule妥当性、型変換、POST_RACE_COLS更新）

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### ETL Engine
- `src/db/etl.py` — ETLエンジン本体。_TABLE_TYPE_RULES (lines 83-131)、_apply_type_conversions (lines 134-176)、SELECT * 抽出パターン。sentinel rule追加の主要変更対象
- `config/etl_tables.yaml` — テーブル定義。n_race (races) と n_uma_race (entries) のPK定義。列フィルタリングはなし（SELECT *）

### 型変換・データアクセス
- `src/db/readers.py` — _INT_COLS (lines 26-40)、_FLOAT_COLS (lines 41-47)、_STRING_COLUMNS (lines 49-82)、coerce_types (lines 140-186)。旧ETL互換の型強制
- `src/db/parquet_store.py` — 低レベルParquet I/O。ParquetStoreのcategory/nameアドレッシング

### POST_RACE安全性
- `src/domain/types.py` (lines 38-55) — POST_RACE_COLS定義（現在16列→41列に拡張）。唯一の正とする
- `tests/test_post_race_leakage.py` — 3層CI漏洩検出テスト。Layer 1: build_all出力、Layer 2: FEATURE_COLS、Layer 3: EV odds
- `tests/test_paper_trading_guards.py` (lines 5-22) — POST_RACE_COLS重複定義。types.py importに変更
- `scripts/run_paper_trading.py` (lines 57-74) — POST_RACE_COLS重複定義。types.py importに変更

### EveryDB2スキーマドキュメント
- `docs/everydb2/04-UMA_RACE.md` — SEテーブル: HaronTimeL3 (field 59), HaronTimeL4 (field 58), Jyuni1c-4c (fields 48-51)。全てvarchar
- `docs/everydb2/03-RACE.md` — RAテーブル: LapTime1-25 (fields 68-92)。全てvarchar(3)

### 要件定義
- `.planning/REQUIREMENTS.md` §ETL — ETL-01~05

### Prior Phase Context
- `.planning/phases/29-etl-expansion/29-CONTEXT.md` — ETL拡張パターン（SELECT *、_TABLE_TYPE_RULES、DataRepository DI）
- `.planning/phases/34-validation-and-manifest-update/34-CONTEXT.md` — 3層CI漏洩テストパターン、Manifest凍結

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/db/etl.py::_TABLE_TYPE_RULES`: 既存の型変換ルール。entries=float/int/odds10、races=int。ここにsentinel_float/sentinel_intを追加
- `src/db/etl.py::_apply_type_conversions()`: 型変換適用関数。sentinel rule処理ステップを追加する拡張点
- `src/db/readers.py::_FLOAT_COLS`: harontimel3既存含む。harontimel4 + laptime1~25を追加
- `src/db/readers.py::_INT_COLS`: jyuni1c/jyuni4c既存含む。jyuni2c/jyuni3cを追加
- `src/domain/types.py::POST_RACE_COLS`: harontimel3/harontimel4/jyuni1c-4c既存含む。laptime1~25を追加

### Established Patterns
- ETL config駆動: etl_tables.yamlのエントリがあればrun_etl.pyが自動処理。列フィルタリングなし（SELECT *）
- 型変換パターン: _TABLE_TYPE_RULESで型ルール定義 → _apply_type_conversionsで一括適用
- POST_RACE whitelist: FEATURE_COLSベースで漏洩防止。POST_RACE_COLS追加で自動的に保護対象
- 全テストmock使用: DB不要テストパターン

### Integration Points
- `src/db/etl.py`: sentinel rule追加の主要変更対象
- `src/db/readers.py`: _INT_COLS/_FLOAT_COLS拡張
- `src/domain/types.py`: POST_RACE_COLS拡張（唯一の正）
- `tests/test_paper_trading_guards.py`: POST_RACE_COLS import化
- `scripts/run_paper_trading.py`: POST_RACE_COLS import化
- Phase 36: 特徴量計算が本フェーズのParquet列を消費

### 既存のETL変換状況（重要）
- **HaronTimeL3**: float64変換済み（_TABLE_TYPE_RULES entries.float に登録済み）
- **HaronTimeL4**: 未変換（varcharのまま）→ 追加必要
- **Jyuni1c/4c**: int変換済み → OK
- **Jyuni2c/3c**: 未変換（varcharのまま）→ 追加必要
- **LapTime1~25**: 完全未対応 → racesにsentinel_floatルール追加必要

</code_context>

<specifics>
## Specific Ideas

- ETLはSELECT *で全列抽出するため、etl_tables.yamlの変更は不要。_TABLE_TYPE_RULESの変更だけで列がfloat64としてParquetに格納される
- sentinel_floatルールの処理順序: (1) 対象列のセンチネル値をNaNに置換 (2) astype(float64)で型変換。既存のfloatルールとは独立して処理
- HaronTimeL3はすでにfloat64変換済みだが、センチネル値(000/999)のNaN化は未処理の可能性がある。ETL品質確認で要検証
- LapTime列名はEveryDB2では小文字(laptime1~25)だが、Parquet列名としてどう格納されるか確認が必要（etl.pyの列名正規化に依存）
- Jyuni2c/3cは00や99等のセンチネル値を持つ可能性。sentinel_floatで処理し、float→Int64に変換することでNaNを許容

</specifics>

<deferred>
## Deferred Ideas

- harontime_last3f統合ロジック (coalesce/距離別選択) — Phase 36でETL後データに基づき決定
- LapTime特徴量化（前半/中盤/後半ペース比等） — Phase 36 HLF-03
- Jyuniコーナー通過順位からの展開特徴量 — Phase v2 HLF-06
- ETL品質確認の自動テスト化 — PostgreSQL環境依存のため手動対応

</deferred>

---

*Phase: 35-ETL Data Foundation*
*Context gathered: 2026-05-19*
