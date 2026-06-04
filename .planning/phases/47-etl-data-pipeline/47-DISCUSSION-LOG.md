# Phase 47: ETL Data Pipeline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-04
**Phase:** 47-ETL Data Pipeline
**Areas discussed:** スクリプト構成, ID集約検証, 欠損値・異常値処理, DataRepository統合方法

---

## スクリプト構成

### Q1: スクリプト数

| Option | Description | Selected |
|--------|-------------|----------|
| 1スクリプト (推奨) | precompute_track_condition.py 1ファイルで両方処理。共通ロジック共有 | ✓ |
| 2スクリプト分離 | precompute_dirt_moisture.py + precompute_turf_cushion.py。既存1:1パターン準拠 | |
| Claudeに任せる | 既存パターンと要件から最適構成を決定 | |

**User's choice:** 1スクリプト (推奨)

### Q2: ロジック配置

| Option | Description | Selected |
|--------|-------------|----------|
| スクリプト内完結 (推奨) | 変換ロジックをscript内に直接記述。sire_stats方式 | |
| src/モジュールに分離 | src/features/track_condition_data.py にロジック、スクリプトはthin orchestrator | ✓ |
| Claudeに任せる | Claudeが最適な配置を決定 | |

**User's choice:** src/モジュールに分離
**Notes:** Phase 48以降のFeatureEngine統合と単体テスト容易性を優先。ETL寄りだがfeatures/に配置する判断。

### Q3: 出力先

| Option | Description | Selected |
|--------|-------------|----------|
| data/raw/ (推奨) | 既存のhorse_career_stats.parquet等と同じ場所 | |
| data/raw/track_condition/ | サブディレクトリ。将来的にデータ追加時に整理しやすい | |
| Claudeに任せる | Claudeが決定 | |

**User's choice:** data/raw/ に配置。ただし2ファイルではなく `track_conditions.parquet` の単一race-level表に統合。列: race_id, race_date, dirt_moisture, turf_cushion。FeatureEngineから1回のmergeで利用可能。

---

## ID集約検証

### Q4: IDマッピング検証方法

| Option | Description | Selected |
|--------|-------------|----------|
| 先頭14桁切り出し + 一致検証 (推奨) | 同一race_id内値一致検証、不一致は警告+最初の値採用 | |
| races.parquetとの交差検証 | 既存racesと照合、未マッチはスキップ/エラー | |
| Claudeに任せる | Claudeが決定 | |

**User's choice:** entry_idは18桁として検証。race_idは既存races.parquetと同じ先頭**16桁**で切り出す(※REQUIREMENTS.mdの14桁は誤記)。同一race_id内で非NaN値が複数種類ある場合はエラー。NaN/非NaN混在は非NaN採用。可能ならraces.parquetのrace_idとも交差検証(ログのみ、強依存しない)。

### Q5: 検証方針確認

| Option | Description | Selected |
|--------|-------------|----------|
| これで確定 | 先頭16桁+値一致検証+NaN/非NaN処理+交差検証(ログのみ) | ✓ |
| さらに議論 | 追加の検証ルールやエッジケースを議論 | |

**User's choice:** 確定。race_id=先頭16桁、CSVの18桁IDは `race_id(16桁)+umaban(2桁)` 構造。REQUIREMENTS.mdの14桁は誤記としてPhase 47計画時に修正。

---

## 欠損値・異常値処理

### Q6: NaN処理方針

| Option | Description | Selected |
|--------|-------------|----------|
| NaNのまま保持 (推奨) | LightGBMネイティブNaN対応。Phase 49で異常値検出 | ✓ |
| 統計的補完 | コース別/季節平均で補完 | |
| 欠損行除外 | NaN行を除外。データ損失大 | |

**User's choice:** NaNのまま保持 (推奨)

### Q7: 物理的異常値処理

| Option | Description | Selected |
|--------|-------------|----------|
| 物理的異常値のみNaN化 (推奨) | 含水率0%/100%、クッション値0等をNaN化。±3σはPhase 49 | ✓ |
| 無処理(生値保持) | 異常値検出はPhase 49の特徴量で対応 | |
| Claudeに任せる | Claudeが決定 | |

**User's choice:** 物理的にあり得ない値のみNaN化。含水率 `0 < value < 100`、クッション値 `value > 0` が有効範囲。範囲外はNaN化+件数ログ出力。±3σ等の統計的外れ値処理はPhase 49。

---

## DataRepository統合方法

### Q8: ローダー追加先

| Option | Description | Selected |
|--------|-------------|----------|
| readers.py standalone関数 (推奨) | 既存load_career_stats/load_sire_statsパターン | |
| DataRepositoryメソッド | ETL-03要件通り。現在のDataRepositoryはodds専用 | ✓ |
| Claudeに任せる | Claudeが決定 | |

**User's choice:** DataRepositoryメソッドとして追加。ETL-03要件を優先。`DataRepository.load_track_conditions(start, end)` を実装。内部は store.exists → date_filters → store.read → coerce_types。FeatureEngineからはDataRepository経由で呼び出す。

---

## Claude's Discretion

- テスト構成・テストケースの詳細設計
- `_compute_race_id()` との整合性検証の実装詳細
- ログフォーマット・進捗表示の設計

## Deferred Ideas

None — discussion stayed within phase scope
