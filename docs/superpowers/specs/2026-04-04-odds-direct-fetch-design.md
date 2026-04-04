# predict でレース・出走馬・オッズを EveryDB2 から直接取得

**日付**: 2026-04-04
**ステータス**: Draft

## 背景

predict / dry-run / setup 実行前に ETL delta を手動で走らせる必要があり、レース直前 (特にオッズ最終確定後) に予測が間に合わない。EveryDB2 は1分ごとに `s_` テーブルを更新しているため、predict 実行時に直接読めば常に最新データを使える。

## 要件

- `_run_predict()`, `_run_dry_run()`, `_run_setup()` で当日データ (races, entries, odds snapshots, odds time series) を EveryDB2 から直接取得する
- ETL を predict/dry-run/setup の前工程から完全に除去する
- 歴史データ (5年分 entries/races) は Parquet のまま (定期 ETL で更新)
- 静的テーブル (horses, kisyu, chokyo 等) は Parquet のまま
- `s_` テーブル (差分/速報) を優先し、空なら `n_` テーブル (蓄積) にフォールバック
- DB接続は既存 `EveryDB2Queries` (psycopg2) を拡張
- 戻り値の型は Parquet 版と完全に互換 (FeatureEngine 以降は変更不要)

## 前提事項

- EveryDB2 公式ドキュメントでは PascalCase (`Year`, `MonthDay`, `TanOdds` 等) で記載されているが、PostgreSQL の識別子 folding により DB 上の列名はすべて **小文字** (`year`, `monthday`, `tanodds` 等) に正規化される。psycopg2 / pd.read_sql_query 経由で取得した DataFrame の列名も小文字となる
- EveryDB2 は全列 `character varying` で保存 — 型変換は呼び出し側で行う
- `s_` テーブルと `n_` テーブルは同じ列構造 (`s_` には `datakubun` が追加される場合がある)
- `datakubun` フィルタは本フェーズでは未対応 (後続フェーズで対応)
- `config.everydb2_connection_string` は `PaperTradingConfig` で既に設定済み
- `etl.py` に既存の `_apply_type_conversions(df, table_key)` 関数があるため、これを再利用する (重複実装しない)
- `_TABLE_TYPE_RULES` の一部列名が実際の EveryDB2 テーブル列と不一致の可能性がある (既知の課題、後述)。型変換関数は存在しない列を安全にスキップするため動作に影響しない

### 既知の _TABLE_TYPE_RULES 列名不一致 (既存 ETL で共通の課題)

以下の不一致は `_TABLE_TYPE_RULES` 自体の既存課題であり、ETL パイプラインと DB ダイレクトの双方に等しく適用される。本フェーズでは修正せず、後続フェーズで対応する:

- `entries` の `timediff` → EveryDB2 ドキュメント上は `TimeDIFN` (小文字化: `timedifn`)
- `entries` の `bataijyu`, `zogensa` → `s_bataijyu` テーブルの列であり `n_uma_race` には存在しない可能性
- `entries` の `harontimel3` → `n_race` テーブルの列であり `n_uma_race` には存在しない可能性
- `races` の `honsyokin` → 実際は `Honsyokin1` 〜 `Honsyokin7` の複数列

型変換関数は `if col in df.columns` で安全にスキップするため、不一致があっても実行時エラーにはならない。ただし該当列が欠落するため、依存する特徴量が NaN になる可能性がある。

## 設計

### 1. EveryDB2Queries へのメソッド追加 (4メソッド)

**ファイル**: `src/db/everydb2_queries.py`

全メソッド共通のパターン:
1. `s_` テーブルを `SELECT *` で日付フィルタ (`WHERE year || monthday = %s`)
2. 結果が空なら `n_` テーブルにフォールバック
3. 戻り値は EveryDB2 生データ (全列 varchar、型変換なし)
4. 例外時は `logger.exception()` + 空 DataFrame を返す

| メソッド | s_ テーブル | n_ テーブル |
|---------|------------|------------|
| `get_races(date_str)` | `s_race` | `n_race` |
| `get_entries(date_str)` | `s_uma_race` | `n_uma_race` |
| `get_odds_snapshots(date_str)` | `s_odds_tanpuku` | `n_odds_tanpuku` |
| `get_odds_time_series(date_str)` | `s_jodds_tanpuku` | `n_jodds_tanpuku` |

既存の `get_latest_odds()` 等のプレースホルダーメソッドは **削除しない** — `watcher.py` から使用されているため。新しいメソッドは別名で追加する。

> **注意**: 既存のプレースホルダーメソッドは PascalCase 列名を使用しているが、これは EveryDB2 の PostgreSQL 上の実際の列名 (小文字) と不一致。新メソッドでは小文字列名を使用する。

### 2. readers.py に DB 版ローダーを追加 (etl.py の型変換を再利用)

**ファイル**: `src/db/readers.py`

#### 型変換: `etl._apply_type_conversions()` をインポートして再利用

新規の型変換関数は作成せず、`etl.py` に既存の `_apply_type_conversions(df, table_key)` をインポートして使用する。この関数は `_TABLE_TYPE_RULES` に基づいて int/float/odds10/odds100 の変換を行い、存在しない列は安全にスキップする。

```python
from db.etl import _apply_type_conversions, _compute_race_date, _compute_race_id
```

対応テーブルと型変換ルール (etl.py の `_TABLE_TYPE_RULES` から引用):

| table_key | int 列 | float 列 | odds10 列 |
|-----------|--------|---------|-----------|
| `races` | trackcd, kyori, tenkocd, syussotosu, honsyokin | | |
| `entries` | umaban, kakuteijyuni, ninki, kyakusitukubun, jyuni1c, jyuni4c, zogenfugo | time, bataijyu, zogensa, harontimel3, timediff | odds |
| `odds_tanpuku` | umaban | | tanodds, fukuoddslow |
| `jodds_tanpuku` | umaban, tanninki | | tanodds, fukuoddslow |

#### DB 版ローダー (4関数)

| 関数 | 対応 Parquet 版 | steeple 除外 | happyotime 保護 |
|------|---------------|-------------|----------------|
| `load_races_from_db(db, ymd)` | `load_races()` | あり | なし |
| `load_entries_from_db(db, ymd)` | `load_entries()` | あり | なし |
| `load_odds_snapshots_from_db(db, ymd)` | `load_odds_snapshots()` | なし | なし |
| `load_odds_time_series_from_db(db, ymd)` | `load_odds_time_series_range()` | なし | あり |

実行順序 (全ローダー共通、**この順序を厳守**):

1. `_apply_type_conversions(df, table_key)` — 型変換 (etl.py からインポート)
2. `_compute_race_date(df)` — `race_date` 派生列 (year, monthday が文字列である必要がある)
3. `_compute_race_id(df)` — `race_id` 派生列 (jyocd 等が文字列である必要がある)
4. `_coerce_types(df)` — 残りの文字列列を数値に変換。`surface` / `track_condition_code` の欠損時はフォールバック計算を自動実行 (readers.py の既存ロジック)
5. `_exclude_steeple(df)` — races, entries のみ (trackcd 51-59 除外)

#### `happyotime` 列の保護 (jodds_tanpuku のみ)

- `happyotime` は `_STRING_COLUMNS` に含まれていないため、`_coerce_types()` で数値変換されてしまう
- `load_odds_time_series_from_db()` では `_coerce_types()` 呼び出し前に `happyotime` を `_STRING_COLUMNS` に一時追加し、呼び出し後に除去する
- `odds_dynamics_features.py` が `happyotime` でソートするため、文字列のまま維持が必須

### 3. _run_setup(), _run_predict(), _run_dry_run() の変更

**ファイル**: `scripts/run_paper_trading.py`

#### _run_setup() の変更

```python
# 変更前
from db.readers import load_entries, load_races
race_df = load_races(store, ymd, ymd)
entry_df = load_entries(store, ymd, ymd)

# 変更後
from db.everydb2_queries import EveryDB2Queries
from db.readers import load_entries_from_db, load_races_from_db

db = EveryDB2Queries(config.everydb2_connection_string)
race_df = load_races_from_db(db, ymd)
entry_df = load_entries_from_db(db, ymd)
```

#### _run_predict() の変更

```python
# 変更前
race_df = load_races(store, ymd, ymd)
entry_df = load_entries(store, ymd, ymd)
odds_df = load_odds_snapshots(store, ymd, ymd)
odds_ts_df = load_odds_time_series_range(store, ymd, ymd)

# 変更後
from db.everydb2_queries import EveryDB2Queries
from db.readers import (
    load_entries_from_db,
    load_odds_snapshots_from_db,
    load_odds_time_series_from_db,
    load_races_from_db,
)

db = EveryDB2Queries(config.everydb2_connection_string)
race_df = load_races_from_db(db, ymd)
entry_df = load_entries_from_db(db, ymd)
odds_df = load_odds_snapshots_from_db(db, ymd)
odds_ts_df = load_odds_time_series_from_db(db, ymd)

if race_df.empty or entry_df.empty or odds_df.empty or odds_ts_df.empty:
    logger.error("EveryDB2 からデータ取得失敗: %s", ymd)
    return
```

履歴データのローダーは変更なし:
- `load_history_entries()`, `load_history_races()` — Parquet のまま
- `load_horses()`, `load_jockey_stats()`, `load_trainer_stats()` — Parquet のまま

#### _run_dry_run() の変更

日付範囲分を日付ごとに DB クエリして連結:

```python
db = EveryDB2Queries(config.everydb2_connection_string)
race_frames, entry_frames, odds_frames, odds_ts_frames = [], [], [], []
for d in dates:
    ymd = d.strftime("%Y%m%d")
    race_frames.append(load_races_from_db(db, ymd))
    entry_frames.append(load_entries_from_db(db, ymd))
    odds_frames.append(load_odds_snapshots_from_db(db, ymd))
    odds_ts_frames.append(load_odds_time_series_from_db(db, ymd))

race_df = pd.concat(race_frames, ignore_index=True) if race_frames else pd.DataFrame()
entry_df = pd.concat(entry_frames, ignore_index=True) if entry_frames else pd.DataFrame()
odds_df = pd.concat(odds_frames, ignore_index=True) if odds_frames else pd.DataFrame()
odds_ts_df = pd.concat(odds_ts_frames, ignore_index=True) if odds_ts_frames else pd.DataFrame()

if race_df.empty or entry_df.empty or odds_df.empty or odds_ts_df.empty:
    logger.error("EveryDB2 からデータ取得失敗: %s ~ %s", all_start, all_end)
    return
```

> **パフォーマンス注意**: 日付範囲が長い場合 (31日 等)、4テーブル × N日 = 4N 回の DB クエリが発生する。`EveryDB2Queries` はクエリごとに接続を開閉する設計のため接続数は問題ないが、I/O 待ちが累積する。長期間の dry-run は `WHERE (year || monthday)::int BETWEEN %s AND %s` のバッチクエリに最適化する余地があるが、本フェーズでは日付ごとのクエリを採用する (実装の単純さを優先)。

`feat_engine.build_all()` の呼び出しは現状と同じ一括パスのまま変更不要。

**変更しないもの:**
- `_run_reconcile()` — 精算処理 (Parquet のまま)
- 特徴量生成パイプライン (FeatureEngine, SubModelManager 等) — 一切変更なし
- 既存の Parquet 版ローダー関数 — 削除せず残す

### 4. エラーハンドリング

- DB接続失敗: `EveryDB2Queries.__init__()` で例外 → 呼び出し側で catch して `logger.error()` + return
- クエリ結果が空: 各 `get_*()` メソッドが空 DataFrame を返す → 呼び出し側で `df.empty` チェック
- 型変換失敗: `_apply_type_conversions()` は空文字/None を None に変換 → 欠損値として扱われる

### 5. テスト戦略

- **EveryDB2Queries の新メソッド** (4メソッド): `unittest.mock` で `psycopg2.connect` をモック。s_ 空 → n_ フォールバック、両方空、例外発生の各パスをテスト
- **DB 版ローダー** (4関数): `EveryDB2Queries` をモックして生データ DataFrame を返させ、`_apply_type_conversions` 適用後の型、race_date/race_id 派生列、`_coerce_types` 適用後の型を検証
- **`happyotime` 保護**: `load_odds_time_series_from_db()` 経由で `happyotime` が文字列のまま維持されることを確認
- **steeple 除外**: `load_races_from_db()`, `load_entries_from_db()` で障害レースが除外されることを確認
- **実行順序**: `_apply_type_conversions` → `_compute_race_date` → `_compute_race_id` → `_coerce_types` の順で正しく動作することを確認
- 既存テスト (Parquet 版、`get_latest_oddds()` 関連) は影響なし
- DB不要 (全て mock) の原則を維持

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/db/everydb2_queries.py` | `get_races()`, `get_entries()`, `get_odds_snapshots()`, `get_odds_time_series()` 追加 |
| `src/db/readers.py` | `load_races_from_db()`, `load_entries_from_db()`, `load_odds_snapshots_from_db()`, `load_odds_time_series_from_db()` 追加。`etl._apply_type_conversions` をインポート |
| `scripts/run_paper_trading.py` | `_run_setup()`, `_run_predict()`, `_run_dry_run()` のデータ取得を DB に変更 |
| `tests/test_everydb2_queries.py` | 新メソッドのテスト追加 (既存テストは維持) |
| `tests/test_readers_db.py` (新規) | DB 版ローダーのテスト |

## 既存プランとの差分

| 項目 | 既存プラン (オッズのみ) | 新設計 (フル DB ダイレクト) |
|------|------------------------|--------------------------|
| 対象テーブル | odds_tanpuku, jodds_tanpuku | + race, uma_race |
| 対象関数 | predict, dry-run | + setup |
| EveryDB2Queries 追加数 | 2メソッド | 4メソッド |
| readers.py 追加数 | 3関数 (型変換+ローダー2) | 4関数 (ローダー4、型変換は etl.py を再利用) |
| 列名リネーム | PascalCase → snake_case | 不要 (小文字のまま) |
| 型変換 | オッズ専用関数 (新規実装) | `etl._apply_type_conversions` を再利用 |
| ETL 依存度 | 低減 (races/entries用に必要) | 除去 (predict/dry-run/setup から) |
