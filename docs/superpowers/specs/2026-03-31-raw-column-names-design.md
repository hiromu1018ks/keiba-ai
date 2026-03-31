# EveryDB2生カラム名一貫化 — DataRepository排除

**日付**: 2026-03-31
**ステータス**: Approved
**関連**: Generic ETL v2 (2026-03-31), Pipeline Performance (2026-03-30)

## 背景

最近のコミットでETLをEveryDB2生カラム名ベースに移行した。現在 `DataRepository` が読み込み時にリネーム+型変換を行っているが、これは毎回の実行で不要なオーバーヘッドとなっている。また `FeatureEngine._map_basic_features()` で2段階目のリネームが発生し、`distance_bin` 計算が3箇所に重複するなど保守性の問題がある。

## 目標

1. DataRepositoryを削除し、ParquetStore直接アクセスに移行
2. 全リネーム層を排除し、EveryDB2生カラム名をParquetからMLパイプラインまで一貫して使用
3. 型変換はETLで一度だけ実行し、Parquetに正しい型で保存
4. FeatureEngineは派生ML特徴量の計算のみに責任を限定

## アーキテクチャ

### 変更前

```
EveryDB2 (全文字列)
  → etl.py (race_id/race_date計算のみ)
  → Parquet (文字列, EveryDB2生カラム名)
  → DataRepository (リネーム+型変換+障害除外)
  → FeatureEngine (2段階目リネーム+派生列計算)
  → MLモデル
```

### 変更後

```
EveryDB2 (全文字列)
  → etl.py (race_id/race_date/surface計算 + 型変換)
  → Parquet (int/float/datetime, EveryDB2生カラム名)
  → readers.py (薄いヘルパー: 日付フィルタのみ)
  → FeatureEngine (派生列計算のみ: distance_bin, track_condition_code, grade_code)
  → MLモデル
```

## セクション1: カラム名戦略

### カラムの3分類

| カテゴリ | 例 | 戦略 |
|---------|-----|------|
| **生カラム** (Parquetから直接) | `trackcd`, `kyori`, `kettonum`, `kisyucode`, `kakuteijyuni`, `odds`, `bataijyu`, `harontimel3`, `timediff`, `jyuni1c`, `jyuni4c`, `kyakusitukubun`, `syussotosu`, `ninki`, `umaban`, `honsyokin`, `tenkocd`, `syubetucd`, `jyokencd1`, `gradecd`, `zogenfugo`, `zogensa` | リネームなし、型のみint/floatに変換 |
| **ETL計算列** | `race_id`, `race_date`, `surface` | ETLで計算してParquetに保存 |
| **FeatureEngine計算列** | `distance_bin`, `track_condition_code`, `grade_code`, `weight_diff_from_mean`, `difficulty_score`, `odds_rank`, `norm_finish_logit_avg`, `haron_time_l3_avg`, `blood_surface_wr`, `jockey_wr_overall` 等 | FeatureEngine/sub-feature modulesで計算（変更なし） |

### 型変換マップ (ETLで実行)

| EveryDB2列 | 変換後の型 | 備考 |
|-----------|-----------|------|
| `trackcd`, `kyori`, `tenkocd`, `syussotosu`, `umaban`, `kakuteijyuni`, `ninki`, `honsyokin`, `kyakusitukubun`, `jyuni1c`, `jyuni4c`, `zogenfugo` | int | 空文字→None |
| `time`, `bataijyu`, `zogensa`, `harontimel3`, `timediff` | float | 空文字→None |
| `odds` | float (÷10) | `"0054"` → 5.4 |
| `tanodds`, `fukuoddslow` | float (÷10) | オッズ系 |
| `oddslow`, `oddshigh` | float (÷100) | ワイドオッズ系 |

### カラム名対応表 (旧→新)

| 旧名 (Repository出力) | 新名 (EveryDB2生) |
|----------------------|------------------|
| `track_cd` | `trackcd` |
| `distance` | `kyori` |
| `ketto_num` | `kettonum` |
| `kisyu_code` | `kisyucode` |
| `chokyosi_code` | `chokyosicode` |
| `finish_pos` | `kakuteijyuni` |
| `finish_time` | `time` |
| `win_odds` | `odds` |
| `ba_taijyu` | `bataijyu` |
| `haron_time_l3` | `harontimel3` |
| `time_diff` | `timediff` |
| `corner_1c` | `jyuni1c` |
| `corner_4c` | `jyuni4c` |
| `kyakusitu` | `kyakusitukubun` |
| `field_size` | `syussotosu` |
| `tenko_cd` | `tenkocd` |
| `syubetu_cd` | `syubetucd` |
| `jyoken_cd` | `jyokencd1` |
| `grade_cd` | `gradecd` |
| `zogen_fugo` | `zogenfugo` |
| `zogen_sa` | `zogensa` |
| `month_day` | `monthday` |
| `jyo_cd` | `jyocd` |
| `race_num` | `racenum` |

**削除される別名 (コピーのみ)**:

| 削除される名前 | 代わりに使う名前 |
|-------------|---------------|
| `popularity_rank` | `ninki` |
| `win_odds_actual` | `odds` |
| `place_odds_actual` | `fukuoddslow` |
| `running_style` | `kyakusitukubun` |
| `surface_key` | `surface` |

**変更なしの名前**: `race_id`, `race_date`, `umaban`, `ninki`, `surface`

## セクション2: ETL変更

### `src/db/etl.py`

#### 型変換テーブル定義

```python
_TABLE_TYPE_RULES: dict[str, dict[str, list[str]]] = {
    "races": {
        "int": ["trackcd", "kyori", "tenkocd", "syussotosu", "honsyokin"],
    },
    "entries": {
        "int": ["umaban", "kakuteijyuni", "ninki", "kyakusitukubun",
                "jyuni1c", "jyuni4c", "zogenfugo"],
        "float": ["time", "bataijyu", "zogensa", "harontimel3", "timediff"],
        "odds10": ["odds"],
    },
    "odds_tanpuku": {
        "int": ["umaban"],
        "odds10": ["tanodds", "fukuoddslow"],
    },
    "odds_wide": {
        "odds100": ["oddslow", "oddshigh"],
    },
    "jodds_tanpuku": {
        "int": ["umaban", "tanninki"],
        "odds10": ["tanodds", "fukuoddslow"],
    },
    "payouts": {
        "int": ["paytansyoumaban1"] + [f"payfukusyoumaban{i}" for i in range(1, 6)],
        "float": ["paytansyopay1"] + [f"payfukusyopay{i}" for i in range(1, 6)],
    },
}
```

#### 新規関数

- `_apply_type_conversions(df, table_key)` — テーブルキーに基づいて型変換を適用
- `_compute_surface(df)` — `trackcd` → `surface` (turf/dirt/other)

#### 呼び出し箇所

- `run_full_load()`: Parquet書き込み前に `_apply_type_conversions()` + `_compute_surface()` を呼ぶ
- `_merge_delta()`: マージ後の書き込み前に同様に呼ぶ

### 変更しないこと

- `config/etl_tables.yaml` — 変更不要
- `_read_db_table()` — 変更不要
- `_compute_race_id()` / `_compute_race_date()` — 変更不要

## セクション3: DataRepository排除

### 削除

- `src/db/repository.py` — DataRepositoryクラス全体を削除
- `src/db/schema.py` — ローカルPostgreSQLスキーマ（現在未使用）

### 新規: `src/db/readers.py`

モジュールレベルのヘルパー関数。型変換・リネームは一切しない。

```python
def load_races(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
def load_entries(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
def load_odds_snapshots(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
def load_odds_time_series_range(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
def load_odds_time_series(store: ParquetStore, race_id: str) -> pd.DataFrame:
def load_wide_odds(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
def load_payouts(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
def load_history_entries(store: ParquetStore, lookback_years: int = 5) -> pd.DataFrame:
def load_history_races(store: ParquetStore, lookback_years: int = 5) -> pd.DataFrame:
def load_horses(store: ParquetStore) -> pd.DataFrame:
def load_jockey_stats(store: ParquetStore) -> pd.DataFrame:
def load_trainer_stats(store: ParquetStore) -> pd.DataFrame:
def load_features(store: ParquetStore, start: str, end: str) -> pd.DataFrame | None:
def save_features(store: ParquetStore, df: pd.DataFrame) -> None:
def save_predictions(store: ParquetStore, df: pd.DataFrame) -> None:
def save_bets(store: ParquetStore, df: pd.DataFrame) -> None:
```

### `db/__init__.py`

DataRepositoryの再エクスポートを削除。

### 障害飛越除外

`_exclude_steeple()` をRepositoryからTrainingPipeline/BacktestEngineの `build_all()` に移動。FeatureEngineの `exclude_steeple` パラメータに委譲（`trackcd` 51-59）。

## セクション4: FeatureEngine変更

### `_map_basic_features()` の簡素化

**削除**: 全リネーム処理（`grade_cd`→`grade_code`、`ninki`→`popularity_rank`、`win_odds`→`win_odds_actual` 等）

**残す処理** (派生列計算のみ):

1. `distance_bin`: `kyori` + `surface` から計算（入力を `distance` → `kyori` に変更）
2. `track_condition_code`: `sibababacd`/`dirtbabacd` + `trackcd` から計算（入力を `track_cd` → `trackcd` に変更）
3. `grade_code`: `gradecd` をそのままコピー（数値マッピングは不要、ETLでint変換済み）

### `build_all()` の変更

- merge keys (`race_id`, `umaban`) は変更なし
- 障害除外: `track_cd` → `trackcd`
- 引数 `repo` → `store` (ParquetStore)

### `build_features()` (推論パス) の変更

ドメインモデル→DataFrame変換で生カラム名を使用:

```python
race_data = {
    "race_id": race.race_id, "trackcd": race.track_cd, "kyori": race.distance,
    "gradecd": race.grade_cd, "syussotosu": race.field_size, "tenkocd": race.tenko_cd,
    "syubetucd": race.syubetu_cd, "jyokencd1": race.jyoken_cd,
}
entry_data = {
    "race_id": race.race_id, "umaban": e.umaban, "kettonum": e.ketto_num,
    "kakuteijyuni": e.finish_pos, "odds": e.win_odds_actual, "ninki": e.popularity_rank,
    "bataijyu": e.ba_taijyu, "kisyucode": e.kisyu_code, "chokyosicode": e.chokyosi_code,
}
```

### `distance_bin` 重複排除

- `FeatureEngine._map_basic_features()` が唯一の計算箇所
- `HorseHistoryFeatures.compute()` 内の `distance_bin` 計算ブロックを削除
- 代わりに `past_df` に `_map_basic_features()` を適用
- `domain.models._distance_band()` は未使用なら削除

## セクション5: サブ特徴量モジュール変更

### BloodlineFeatures

```python
# merge: ketto_num/kettonum 不一致を解消
merged = entry_df[["race_id", "umaban", "kettonum"]].merge(
    horses_df, on="kettonum", how="left"  # was left_on="ketto_num", right_on="kettonum"
)
```

### JockeyContextFeatures

```python
# merge: kisyu_code/kisyucode 不一致を解消
merged = entry_df[["race_id", "umaban", "kisyucode", "race_year"]].merge(
    stats_df, on="kisyucode", how="left"  # was left_on="kisyu_code", right_on="kisyucode"
)
```

### TrainerContextFeatures

```python
# merge: chokyosi_code/chokyosicode 不一致を解消
merged = entry_df[["race_id", "umaban", "chokyosicode", "race_year"]].merge(
    stats_df, on="chokyosicode", how="left"
)
```

### HorseHistoryFeatures

全カラム参照を生名に変更:

| 旧名 | 新名 |
|------|------|
| `ketto_num` | `kettonum` |
| `kisyu_code` | `kisyucode` |
| `finish_pos` | `kakuteijyuni` |
| `win_odds` | `odds` |
| `field_size` | `syussotosu` |
| `track_cd` | `trackcd` |
| `distance` | `kyori` |
| `haron_time_l3` | `harontimel3` |
| `time_diff` | `timediff` |
| `corner_1c` | `jyuni1c` |
| `corner_4c` | `jyuni4c` |
| `kyakusitu` | `kyakusitukubun` |
| `ba_taijyu` | `bataijyu` |

### intra_race_features.py

```python
# 変更後
df["weight_diff_from_mean"] = df["bataijyu"] - weight_mean
df["odds_rank"] = df.groupby("race_id")["odds"].rank(...)
```

### その他featureモジュール

| モジュール | 変更内容 |
|-----------|---------|
| `odds_dynamics_features.py` | `tan_odds`→`tanodds`, `fuku_odds`→`fukuoddslow` |
| `market_bias_features.py` | `win_odds`→`odds`, `popularity_rank`→`ninki` |
| `race_difficulty_model.py` | `grade_cd`/`grade_code` fallback → `gradecd`/`grade_code` |
| `interaction_features.py` | `kyakusitu`→`kyakusitukubun`, `ba_taijyu`→`bataijyu` |
| `info_asymmetry_features.py` | `win_odds_actual`→`odds`, `place_odds_actual`→`fukuoddslow` |
| `submodel_manager.py` | `surface_key`→`surface` |

## セクション6: MLモデル・パイプライン変更

### モデル FEATURE_COLS — 変更なし

全モデルの `FEATURE_COLS` は計算済み特徴量名を参照。変更不要。

### label・キャリブレーション列の変更

| 旧名 | 新名 | 影響モデル |
|------|------|-----------|
| `finish_pos` | `kakuteijyuni` | AbilityModel, PlaceAbilityModel, WinTwoStageModel, EVCorrectionModel |
| `win_odds_actual` | `odds` | WinTwoStageModel, EVCorrectionModel, RobustConfidenceEstimator |
| `place_odds_actual` | `fukuoddslow` | PlaceTwoStageModel, RobustConfidenceEstimator |
| `popularity_rank` | `ninki` | WinTwoStageModel, EVCorrectionModel, RaceQualityScreener |

### TrainingPipelineV5

- コンストラクタ: `repo: DataRepository` → `store: ParquetStore`
- データロード: `self.repo.load_xxx()` → `load_xxx(self.store, ...)`
- wide_odds pivot: `odds_low` → `oddslow`
- キャリブレーション: `win_odds_actual` → `odds`, `place_odds_actual` → `fukuoddslow`, `finish_pos` → `kakuteijyuni`
- SubModelManager: `surface_key` → `surface`
- HorseHistoryFeatures/JockeyContextFeatures/TrainerContextFeatures: `repo=self.repo` → `store=self.store`

### BacktestEngine

TrainingPipelineと同パターン。`repo` → `store`、カラム名を生名に変更。

## セクション7: Ingestion・Paper Trading・テスト変更

### JVLinkFetcher

`_row_to_race()` / `_row_to_entry()` のカラム参照を生名に変更。ドメインモデルのコンストラクタ引数名は変更しない（`track_cd`, `distance` 等のまま）。

### OddsCollector

`repo.save_predictions(df)` → `store.write("predictions", "predictions", df)`

### Paper Trading

- `predictor.py`: `repo` → `store`
- `reconciler.py`: `repo` → `store`

### テスト

**更新**: `test_backtest_engine`, `test_horse_history_features`, `test_jvlink_fetcher`, `test_odds_collector`, `test_history_features_v2`

**削除**: `test_repository.py`

**新規**: `test_readers.py`, `test_etl_type_conversion.py`

## セクション8: 移行戦略

### 3フェーズ移行

**Phase 1: ETL型変換 + readers.py**
- `etl.py` に型変換追加
- `db/readers.py` 作成
- `repository.py` は並存
- ETL再実行 → `pytest` 確認

**Phase 2: FeatureEngine + Sub-feature 生名化**
- `_map_basic_features()` 簡素化
- 全featureモジュールのカラム名変更
- MLモデルlabel列名変更
- `pytest` 確認

**Phase 3: Repository削除 + 全コンシューマ移行**
- `DataRepository` 削除
- 全コンシューマを `readers.py` + `ParquetStore` に移行
- テスト更新
- バックテストでROI回帰確認

### リスク

| リスク | 対策 |
|--------|------|
| カラム名変更の漏れ | Phaseごとに `pytest` + `grep` で旧名の残存を確認 |
| ROI回帰 | リネームのみでロジック変更なし。バックテストで確認 |
| ETL再実行時間 | `--tables` で必要テーブルのみ選択可能 |
| MLflow保存モデルの互換性 | 入力DataFrameの列名が変わるため再学習が必要 |
