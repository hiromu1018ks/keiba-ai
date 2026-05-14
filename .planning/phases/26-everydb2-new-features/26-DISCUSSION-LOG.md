# Phase 26: EveryDB2 New Features - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-14
**Phase:** 26-EveryDB2 New Features
**Areas discussed:** ETL・データ取得戦略, n_mining PIT監査, 血統特徴量の設計, 相対比較特徴量の範囲

---

## ETL・データ取得戦略

| Option | Description | Selected |
|--------|-------------|----------|
| 個別テーブル抽出 | n_hansyoku/n_record/n_miningの3テーブル限定抽出。~1-2分 | ✓ |
| フルETL再実行 | --mode full --start 20140101 --end 20251231。~10分 | |
| 手動実行前提 | Phase外でユーザーが手動実行。Parquet存在チェックのみ | |

**User's choice:** 個別テーブル抽出
**Notes:** フルETLは不要。3テーブルのみで十分。

| Option | Description | Selected |
|--------|-------------|----------|
| Phase内でETL含む | Plan内にETLステップを含める。PostgreSQL環境依存のためCI不可 | ✓ |
| 前提条件とする | ETL実行はPhase外。Parquet不存在時はskip/エラー | |

**User's choice:** Phase内でETL含む
**Notes:** Plan 1-01等にETL実行ステップを含める。

| Option | Description | Selected |
|--------|-------------|----------|
| run_etl.py --tables | 既存CLIの--tables引数を使用。個別テーブル抽出既にサポート済み | ✓ |
| 新規スクリプト作成 | Parquet存在チェック + 抽出 + スキーマ検証を一括 | |

**User's choice:** run_etl.py --tables
**Notes:** 既存機能で十分。

---

## n_mining PIT監査

| Option | Description | Selected |
|--------|-------------|----------|
| 自動監査スクリプト | 82列を分析。列名パターン + データ統計的分析でPRE/POST自動推定 | ✓ |
| ドキュメント手動分類 | JRA-VANデータ定義書を参照して手動分類 | |
| n_mining除外 | 高リスクとして除外。他テーブルに集中 | |

**User's choice:** 自動監査スクリプト
**Notes:** FEATURES.md D-08評価に基づく。

| Option | Description | Selected |
|--------|-------------|----------|
| ドキュメント照合優先 | docs/everyDB2/44-MINING.mdの列説明を主軸に分類 | ✓ |
| データパターン分析優先 | レース前後の値変化検出を第一手段 | |
| 交差検証 | 両方を独立実行し不一致を手動確認 | |

**User's choice:** ドキュメント照合優先
**Notes:** ユーザーから「@docs/everyDB2/に格納されているdocsと照合するのはどうか」という提案あり。docs/everyDB2/配下に34-HANSYOKU.md等のテーブル定義ドキュメントが豊富に存在することを確認。

| Option | Description | Selected |
|--------|-------------|----------|
| PRE列のみ使用 | POST列は除外、PRE列から特徴量抽出。分類結果は文書化 | ✓ |
| 高POST比率なら全体除外 | 安全側に倒す | |

**User's choice:** PRE列のみ使用
**Notes:** テーブル全体の除外は行わない。

---

## 血統特徴量の設計

| Option | Description | Selected |
|--------|-------------|----------|
| 包括的血統 | n_hansyoku + n_sanku活用。繁殖牝馬産駒成績、BMS拡張等（D-01完全実装） | ✓ |
| n_hansyoku限定 | 既存sire_features.pyのBMS拡張と組み合わせ、新モジュールは最小限 | |
| 既存拡張のみ | sire_features.pyにBMS distance_wr/surface_wr追加のみ。新テーブル不使用 | |

**User's choice:** 包括的血統
**Notes:** DATA-01要件「種牡馬系統、母系BMS等」を完全にカバー。

| Option | Description | Selected |
|--------|-------------|----------|
| 新規モジュール | features/dam_pedigree_features.py等。既存モジュールはそのまま | ✓ |
| 既存モジュール拡張 | bloodline_features.pyにdam pedigree機能を追加 | |
| 3モジュール分離 | bloodline + sire + dam_pedigreeの3モジュール体制 | |

**User's choice:** 新規モジュール
**Notes:** 既存モジュールは変更しない。

| Option | Description | Selected |
|--------|-------------|----------|
| Stage1 AbilityModel | 既存blood_*、sire_*もStage1に配置されている | |
| Stage2のみ | 他馬との比較コンテキスト | |
| 両方 | モデル別に最適化 | |

**User's choice:** ベストな方法を選定してくれ（Claude discretion）

---

## 相対比較特徴量の範囲

| Option | Description | Selected |
|--------|-------------|----------|
| 5-10特徴量 | TS-05推奨。norm_finish_logit_vs_mean等。過学習リスク低 | ✓ |
| 包括的（20-30特徴量） | 全Stage1特徴量の相対ランク・偏差値 | |
| 最小限 | 2-3特徴量の追加に留める | |

**User's choice:** ベストプラクティスを追求（5-10特徴量推奨を採用）

| Option | Description | Selected |
|--------|-------------|----------|
| intra_race拡張 | 既存モジュールに追加 | |
| 新規モジュール | features/relative_features.py（FEATURES.md TS-05推奨） | |

**User's choice:** ベストプラクティスを追求（新規モジュール採用）
**Notes:** FEATURES.md推奨の新規relative_features.pyを作成。

---

## Claude's Discretion

- 血統特徴量の配置先モデル（Stage1 / Stage2 / 両方）
- BMS拡張の実装場所（sire_features.py拡張 vs 新規モジュール）
- 相対比較特徴量の具体的な特徴量名・計算方法
- n_record特徴量の具体的な設計
- n_mining PRE列から抽出する特徴量の選定
- 各FEATURE_COLSへの挿入位置

## Deferred Ideas

None — discussion stayed within phase scope
