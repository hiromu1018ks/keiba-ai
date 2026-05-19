# Phase 35: ETL Data Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-19
**Phase:** 35-ETL Data Foundation
**Areas discussed:** LapTime POST_RACE管理, harontime_last3f統合ロジック, センチネル値NaN化方式

---

## LapTime POST_RACE管理

| Option | Description | Selected |
|--------|-------------|----------|
| POST_RACE_COLSに追加 | LapTime1~25を既存POST_RACE_COLSに追加。シンプルで安全だがLayer 1テストが通るか確認必要 | ✓ |
| race-level専用リストを新規作成 | POST_RACE_RACE_COLSを別定義。漏洩検出も別クラス。完全カバレッジ | |
| LapTimeはPOST_RACEとして登録のみ | 漏洩テストはPhase 36で対応。Phase 35は登録のみ | |

**User's choice:** POST_RACE_COLSに追加
**Notes:** シンプルで安全なアプローチを選択。LapTimeはrace-levelのためbuild_all()出力に自然に含まれない

| Option | Description | Selected |
|--------|-------------|----------|
| 25列全て登録 | LapTime1~25の25列を個別登録。POST_RACE_COLSが41列に増加 | ✓ |
| 一括グループとして別管理 | LapTime1~25をPOST_RACE_RACE_LEVELとして別管理 | |

**User's choice:** 25列全て登録
**Notes:** POST_RACE_COLSが16列→41列に拡張。全列を均一に管理

| Option | Description | Selected |
|--------|-------------|----------|
| 既存テストそのままでOK | LapTimeはrace-levelなのでbuild_all()出力に含まれない。既存テスト通過 | ✓ |
| ETL品質テスト追加 | races ParquetのLapTimeの型・NaN検証を追加 | |

**User's choice:** 既存テストそのままでOK
**Notes:** 全テストmock使用のため実際のParquet検証は不可

| Option | Description | Selected |
|--------|-------------|----------|
| ETL後の手動確認のみ | run_etl.py実行後にClaudeが品質確認 | ✓ |
| CIテスト追加 | 自動テストでParquet品質検証 | |

**User's choice:** ETL後の手動確認のみ
**Notes:** PostgreSQL環境依存のためCIテスト不可

---

## harontime_last3f統合ロジック

| Option | Description | Selected |
|--------|-------------|----------|
| L3優先 coalesce(L3, L4) | 上がり3FがJRA標準指標 | |
| L4優先 coalesce(L4, L3) | L4が多くのレースで計測 | |
| 統合せず別々に保持 | L3/L4を別カラムで保持 | |
| データ確認後に決定 | ETL後に分布確認してから決定 | ✓ |

**User's choice:** 別々保持 + Phase 36で統合（推奨）
**Notes:** ベストプラクティスを追求。L3(600m)とL4(800m)はスケールが異なるため、coalesceは分析的に不適切。Phase 35では別々にfloat64化し、Phase 36で検証結果に基づいて統合ロジックを決定

| Option | Description | Selected |
|--------|-------------|----------|
| ETL後Claude検証 | L3のみ/L4のみ/両方/なしの4分類分布確認 | ✓ |
| CIテスト追加 | 自動テストで相互排他性を検証 | |

**User's choice:** ETL後Claude検証
**Notes:** データ分布を確認後にPhase 36で統合ロジック決定

---

## センチネル値NaN化方式

| Option | Description | Selected |
|--------|-------------|----------|
| sentinel専用type rule追加 | _TABLE_TYPE_RULESにsentinel_float/sentinel_int型を追加 | |
| 3ケース専用のハードコード | HaronTime/LapTime/Jyuni専用のNaN化処理 | |
| 汎用センチネル処理層の追加 | 宣言的ルールで将来拡張可能なセンチネル処理 | ✓ |

**User's choice:** 宣言的sentinel rule
**Notes:** ベストプラクティスを追求。_TABLE_TYPE_RULESに構造化されたsentinel_float/sentinel_intルールを追加。拡張可能で自己文書化

| Option | Description | Selected |
|--------|-------------|----------|
| Harmonized sentinels | HaronTime=000/999, LapTime=000, Jyuni=000。全てsentinel_floatで処理 | ✓ |
| データ確認後に精密化 | ETL後に実際のセンチナル値を確認 | |

**User's choice:** Harmonized sentinels
**Notes:** 全センチネル値を宣言的に定義。Jyuniもfloat→Int64変換でNaN許容

| Option | Description | Selected |
|--------|-------------|----------|
| readers.pyは変更しない | Phase 29決定維持。新ETL Parquetはtypes already correct | |
| readers.pyも更新 | _INT_COLS/_FLOAT_COLSも更新。旧ETL互換性維持 | ✓ |

**User's choice:** readers.pyも更新
**Notes:** ベストプラクティスを追求。新旧両方のETL Parquetに対応

| Option | Description | Selected |
|--------|-------------|----------|
| 3箇所手動更新 | types.py + test_paper_trading_guards.py + run_paper_trading.py を手動更新 | |
| import元を1箇所に集約 | types.pyを唯一の正とし、他2箇所はimport。DRY原則 | ✓ |

**User's choice:** import元を1箇所に集約
**Notes:** ベストプラクティスを追求。将来の変更が1箇所で済む

---

## Claude's Discretion

なし — 全ての決定事項についてユーザーが明確な選択を行った

## Deferred Ideas

- harontime_last3f統合ロジック（coalesce/距離別選択）— Phase 36でETL後データに基づき決定
- LapTime特徴量化（前半/中盤/後半ペース比等）— Phase 36 HLF-03
