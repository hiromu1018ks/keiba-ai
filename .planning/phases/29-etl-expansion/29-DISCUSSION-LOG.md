# Phase 29: ETL Expansion - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-17
**Phase:** 29-ETL Expansion
**Areas discussed:** データアクセス層, カバレッジ検証方式, _headテーブルの扱い, スキップ判定, DB構造確認

---

## データアクセス層

| Option | Description | Selected |
|--------|-------------|----------|
| readers.py に追加 | load_trio_odds等を追加。既存パターン踏襲。 | |
| ParquetStore 直 | readers.pyを介さず直接呼び出し | |
| DataRepository クラス作成 | CLAUDE.md/ROADMAP想定のクラスを作成 | |

**User's choice:** "実装難易度は問わないのでベストプラクティスを追求"

→ 追加質問で範囲を絞り込み:

| Option | Description | Selected |
|--------|-------------|----------|
| 新規のみ DataRepository | 3メソッドのみ。既存readers.py変更なし | ✓ |
| readers.py 追加に留める | 最小スコープ | |
| フル DataRepository 移行 | 大規模リファクタリング | |

**User's choice:** 新規のみ DataRepository (Recommended)
**Notes:** ベストプラクティスを追求するが、Phase 29のスコープはParquet抽出と検証に集中

---

## カバレッジ検証方式

| Option | Description | Selected |
|--------|-------------|----------|
| ETLスクリプトに組み込み | run_etl.py実行後に自動でカバレッジレポート出力 | ✓ |
| 別バリデーションスクリプト | scripts/validate_odds_coverage.py等 | |
| テストスイートに組み込み | CI自動実行 (mockでは不可) | |

**User's choice:** ETLスクリプトに組み込み (Recommended)

---

## _head テーブルの扱い

| Option | Description | Selected |
|--------|-------------|----------|
| 本体のみ | Phase 32は組合せオッズのみで計算可能 | |
| 本体 + head 両方 | 将来利用可能性を考慮 | |

**User's choice:** "ベストプラクティスを追求"

→ Extract once, use later の原則で両方抽出に決定

---

## スキップ判定

| Option | Description | Selected |
|--------|-------------|----------|
| スキップ不可 | DataRepository作成+カバレッジ検証+テストが必要 | ✓ |
| 最小実装に縮小 | ETL実行は手動、DataRepositoryのみ実装 | |

**User's choice:** "実際にETLを実行し、データの品質を確認するところまでやる"
**Notes:** .envにパスワードあり。PostgreSQL接続可能

---

## ETL実行スコープ

| Option | Description | Selected |
|--------|-------------|----------|
| 特定テーブルのみ | 6テーブル約1分 | ✓ |
| 全テーブル再抽出 | 全103テーブル約10分 | |

**User's choice:** 特定テーブルのみ (Recommended)

---

## DB構造確認

**User's question:** "実際にDBのカラムやデータの形式などは確認しなくていいのか"

→ 実際にPostgreSQLにクエリを実行して確認:
- 3テーブル共通: (makedate, year, monthday, jyocd, kaiji, nichiji, racenum, kumi, odds, ninki)
- kumi形式: sanren="010203"(6桁), umaren="0102"(4桁), sanrentan="010203"(6桁)
- **etl_tables.yamlのPK定義が間違っていることを発見** (umaban1/2/3 → 実際は kumi)

→ 追加決定: Phase 29でPK定義を修正

---

## Claude's Discretion

- DataRepository の内部実装詳細
- カバレッジレポートの出力フォーマット
- テストケースの具体的な設計

## Deferred Ideas

None — discussion stayed within phase scope
