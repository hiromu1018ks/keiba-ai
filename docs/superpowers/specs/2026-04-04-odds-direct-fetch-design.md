# predict でオッズを EveryDB2 から直接取得

**日付**: 2026-04-04
**ステータス**: Approved

## 背景

predict / dry-run 実行前に ETL delta を手動で走らせる必要があり、レース直前の最新オッズを取得しきれない。EveryDB2 は1分ごとに `s_` テーブルを更新しているため、predict 実行時に直接読めば常に最新データを使える。

## 要件

- `_run_predict()` と `_run_dry_run()` でオッズ (snapshots + 時系列) を EveryDB2 から直接取得する
- レース・出走馬・血統などは Parquet のまま (変更頻度が低いため)
- `s_` テーブル (差分) を優先し、空なら `n_` テーブル (ベース) にフォールバック
- DB接続は既存 `EveryDB2Queries` (psycopg2) を拡張
- 戻り値の型は Parquet 版と完全に互換 (FeatureEngine 以降は変更不要)

## 設計

### 1. EveryDB2Queries へのメソッド追加

**ファイル**: `src/db/everydb2_queries.py`

**`get_odds_snapshots(date_str: str) -> pd.DataFrame`**
- `s_odds_tanpuku` から `WHERE (Year || MonthDay) = :date` で当日分を SELECT
- 空なら `n_odds_tanpuku` にフォールバック
- 戻り値: EveryDB2 生データ (全列 `character varying`、型変換なし)

**`get_odds_time_series(date_str: str) -> pd.DataFrame`**
- `s_jodds_tanpuku` から `WHERE (Year || MonthDay) = :date` で当日分を SELECT
- 空なら `n_jodds_tanpuku` にフォールバック
- 戻り値: EveryDB2 生データ

既存の `get_latest_odds()` は **削除しない** — `RaceWatcher` (`src/paper_trading/watcher.py`) から使用されているため。新しいメソッドは別名で追加し、`watcher.py` の移行は別タスクとする。

### 2. readers.py に DB 版ローダーと型変換を追加

**ファイル**: `src/db/readers.py`

**`load_odds_snapshots_from_db(db: EveryDB2Queries, ymd: str) -> pd.DataFrame`**
- `db.get_odds_snapshots(ymd)` を呼び、ETL と同じ型変換を適用
- `_compute_race_date`, `_compute_race_id` は `db.etl` から import
- `_apply_odds_type_conversions(df, "odds_tanpuku")` で変換
- `race_date`, `race_id` の派生列を計算
- `_coerce_types()` を通して Parquet 版と同じ型の DataFrame を返す

**`load_odds_time_series_from_db(db: EveryDB2Queries, ymd: str) -> pd.DataFrame`**
- `db.get_odds_time_series(ymd)` を呼び、`_apply_odds_type_conversions(df, "jodds_tanpuku")` で変換
- 派生列 + `_coerce_types()` → Parquet 版と同じ型

**実行順序 (重要)**:
型変換と派生列の計算は以下の順序で実行すること。順序を間違えると `_coerce_types()` が文字列列を数値に変換してしまい、`_compute_race_id()` のゼロパディングが壊れる:

1. `_apply_odds_type_conversions(df, table_key)` — オッズ固有の型変換
2. `_compute_race_date(df)` — `race_date` 派生列 (year, monthday が文字列である必要がある)
3. `_compute_race_id(df)` — `race_id` 派生列 (jyocd 等が文字列である必要がある)
4. `_coerce_types(df)` — 残りの文字列列を数値に変換

**`_apply_odds_type_conversions(df: pd.DataFrame, table_key: str) -> pd.DataFrame`** (private helper)
- `odds_tanpuku` の場合: `umaban` → `Int64`, `tanodds` → `_to_odds(v, 10)`, `fukuoddslow` → `_to_odds(v, 10)`
- `jodds_tanpuku` の場合: `umaban` → `Int64`, `tanninki` → `Int64`, `tanodds` → `_to_odds(v, 10)`, `fukuoddslow` → `_to_odds(v, 10)`
- `_to_odds(v, divisor)` = `float(v) / divisor` (EveryDB2 はオッズを整数倍で保存)

**`happyotime` 列の扱い**:
- EveryDB2 の `n_jodds_tanpuku` / `s_jodds_tanpuku` では列名は `happyotime` (文字列、例: `"03251500"`)
- `happyotime` は `_STRING_COLUMNS` に含まれていないため、`_coerce_types()` で数値変換されてしまう
- `_apply_odds_type_conversions()` の前に `happyotime` を `_STRING_COLUMNS` に一時追加するか、DB 版ローダー内で `_coerce_types()` 呼び出し時に `happyotime` を保護する
- `odds_dynamics_features.py` が `happyotime` でソートするため、文字列のまま維持が必須

### 3. _run_predict() と _run_dry_run() の変更

**ファイル**: `scripts/run_paper_trading.py`

#### _run_predict() の変更

```python
# 変更前
odds_df = load_odds_snapshots(store, ymd, ymd)
odds_ts_df = load_odds_time_series_range(store, ymd, ymd)

# 変更後
from db.everydb2_queries import EveryDB2Queries
from db.readers import load_odds_snapshots_from_db, load_odds_time_series_from_db

db = EveryDB2Queries(connection_string)  # config/settings.yaml から構築
odds_df = load_odds_snapshots_from_db(db, ymd)
odds_ts_df = load_odds_time_series_from_db(db, ymd)
```

#### _run_dry_run() の変更

`_run_dry_run()` は現在、全データを一括ロードしてから `feat_engine.build_all()` を実行するバッチアーキテクチャ。日付ごとに DB クエリを実行する場合、**特徴量計算の前に全日付分のオッズを一括取得して連結**する:

```python
# 変更後
db = EveryDB2Queries(connection_string)

# 日付範囲分のオッズを一括取得
odds_frames = []
odds_ts_frames = []
for d in dates:
    ymd = d.strftime("%Y%m%d")
    odds_frames.append(load_odds_snapshots_from_db(db, ymd))
    odds_ts_frames.append(load_odds_time_series_from_db(db, ymd))

odds_df = pd.concat(odds_frames, ignore_index=True)
odds_ts_df = pd.concat(odds_ts_frames, ignore_index=True)
```

これにより、`feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)` の呼び出しは現状と同じ一括パスのまま変更不要。

- `EveryDB2Queries` はクエリごとに接続を開閉する設計のため、複数日付のループでも問題なし
- レース・出走馬・血統データは Parquet のまま変更なし

**エラーハンドリング**:
- DB接続失敗: ログ出力 + `sys.exit(1)` (オッズは predict に必須)
- クエリ結果が空: ログ出力 + `sys.exit(1)` (オッズなしでは予測不可能)

### 4. テスト戦略

- **EveryDB2Queries の新メソッド**: `unittest.mock` で `psycopg2.connect` をモック。s_ テーブル空 → n_ フォールバックのパスもテスト
- **readers.py の DB 版ローダー**: `EveryDB2Queries` をモックして生データ DataFrame を返させ、型変換後の値を検証 (`_to_odds(150, 10) == 15.0` など)
- **`happyotime` 保護テスト**: DB 版ローダー経由で `happyotime` が文字列のまま維持されることを確認
- **実行順序テスト**: `_apply_odds_type_conversions` → `_compute_race_date` → `_compute_race_id` → `_coerce_types` の順で正しく動作することを確認
- **`_run_predict()` 統合テスト**: DB接続をモックし、Parquet 版と DB 版で同じ特徴量が生成されることを確認
- 既存テスト (Parquet 版、`get_latest_odds()` 関連) は影響なし
- DB不要 (全て mock) の原則を維持

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/db/everydb2_queries.py` | `get_odds_snapshots()`, `get_odds_time_series()` 追加。`get_latest_odds()` は **削除しない** |
| `src/db/readers.py` | `load_odds_snapshots_from_db()`, `load_odds_time_series_from_db()`, `_apply_odds_type_conversions()` 追加。`happyotime` 保護対応 |
| `scripts/run_paper_trading.py` | `_run_predict()` のオッズ取得を DB に変更。`_run_dry_run()` は日付ループで一括取得後に連結 |
| `tests/test_everydb2_queries.py` (既存) | 新メソッドのテスト追加。既存 `get_latest_odds()` テストは維持 |
| `tests/test_readers.py` (新規または既存) | DB 版ローダーのテスト追加 |
