# Spike 002: バックテスト実行時間のボトルネック特定と半減手法

**Status:** VALIDATED (code analysis)
**Date:** 2026-05-08
**Context:** `--train-start 20200101 --train-end 20231231 --test-start 20240101 --test-end 20241231 --ensemble --report --profile`

---

## Executive Summary

バックテストパイプラインは3段階の実行フローを持つ:

```
Phase A: Training (TrainingPipelineV5.run)
Phase B: Calibration BT (_collect_training_bet_history)  ← 学習期間のフルBT
Phase C: Test BT (BacktestEngine.run)
Phase D: Report (軽微)
```

**前回プロファイル実測値 (8239s = 2.29h):**

| Phase | Time | % | Description |
|-------|------|---|-------------|
| Phase B: Calibration BT | ~4901s | 60.2% | 学習期間(4年)のフルバックテスト |
| Phase A: Training | ~1952s | 23.7% | モデル学習 (ThreadPoolExecutor) |
| Phase C: Test BT | ~1381s | 16.8% | テスト期間(1年)のバックテスト |
| Phase D: Report | ~5s | <0.1% | HTML/JSON/Parquet生成 |

**結論:** Phase B が全体の60%を占める。これを最適化または排除できれば、単独で 1.5〜2時間の短縮になる。

---

## Bottleneck #1: OddsBandFilter キャリブレーションバックテスト (60.2%)

### 現状

`scripts/run_backtest.py:211-261` の `_collect_training_bet_history()` が、
学習期間(4年分)の**フルバックテスト**を実行している。これは OddsBandFilter の
キャリブレーション用 bet_history を収集するため。

```python
# run_backtest.py:247-255
train_engine = BacktestEngine(models=models, store=store, ...)
train_result = train_engine.run(train_start, train_end)
```

### 時間内訳 (Phase B 内部)

| 操作 | 時間 | 備考 |
|------|------|------|
| HorseHistoryFeatures.compute() | ~389s | itertuples Python loop |
| compute_batch (各種特徴量) | ~130s | object列比較が主因 |
| load_odds_time_series_range | ~121s | 452MBのParquet読み込み+型変換 |
| その他 (推論ループ, 診断ログ) | ~4260s | 12,000+レースのper-race処理 |

### 改善案

**案A: キャリブレーションBTの軽量化 (推定効果: -3000〜3500s)**
- 学習期間のフルBTではなく、直近Nヶ月のみでキャリブレーション
- `train_start` を `train_end - 6months` に短縮
- OddsBandFilter は直近のオッズ分布に依存するため、4年不要

**案B: BacktestEngine への data caching (推定効果: -200〜400s)**
- Phase A でロード済みのデータ (races, entries, odds) を Phase B に渡す
- ParquetStore に LRU cache を追加
- `BacktestEngine.__init__()` に `preloaded_data: dict` パラメータを追加

**案C: キャリブレーション不要なアプローチ (推定効果: -4900s)**
- OddsBandFilter を静的パラメータ化 (strategy_manifest に含める)
- キャリブレーション自体を Optuna 最適化フェーズに移行

---

## Bottleneck #2: Object列のgroupby/merge/比較 (累計 ~800s)

### 現状

pandasの `comp_method_OBJECT_ARRAY` が多数の compute_batch/compute で
実行されており、プロファイルでも顕著な時間を占めている。

主な原因: `race_id`, `kettonum`, `kisyucode` 等が object (string) dtype で
groupby/merge/isin されている。

### 影響が大きい箇所

| ファイル | 行 | 操作 | 影響度 |
|----------|-----|------|--------|
| `horse_history_features.py` | 424 | `groupby("kettonum")` on object | CRITICAL |
| `horse_history_features.py` | 432,439 | `groupby("kisyucode")` on object (x2) | CRITICAL |
| `horse_history_features.py` | 592 | `itertuples()` 全馬ループ | CRITICAL |
| `horse_history_features.py` | 553 | `groupby(group_cols)` on string columns | MEDIUM |
| `horse_history_features.py` | 391,354 | `merge(on="race_id")` on object | MEDIUM |
| `sire_features.py` | 138-157 | `stats[stats["sire_id"] == sid]` loop内 | HIGH |
| `jockey_context_features.py` | 63-65 | `merge(on="kisyucode")` 多対多 | HIGH |
| `jockey_context_features.py` | 70-75 | `groupby(["race_id","umaban","kisyucode"])` 3列string | HIGH |
| `trainer_context_features.py` | 63-65 | `merge(on="chokyosicode")` 同上 | HIGH |
| `trainer_context_features.py` | 70-75 | `groupby(["race_id","umaban","chokyosicode"])` 同上 | HIGH |
| `pace_aptitude_features.py` | 100-105 | `isin` + `merge` on string race_id | MEDIUM |
| `course_features.py` | 87-92 | `isin` + `merge` on string race_id | MEDIUM |
| `course_features.py` | 184,188 | `jc_slice == tjc` Python文字列比較 | MEDIUM |

### 改善案

**案: String列の Categorical 化 (推定効果: -300〜500s)**

ParquetStore または DataRepository レベルで、高カーディナリティの文字列列を
`pd.Categorical` に変換する。

```python
# 対象列: race_id, kettonum, kisyucode, chokyosicode, sire_id, bms_id
CATEGORICAL_COLUMNS = {
    "race_id", "kettonum", "kisyucode", "chokyosicode",
    "sire_id", "bms_id", "umaban"
}

def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLUMNS & set(df.columns):
        if df[col].dtype == object:
            df[col] = df[col].astype("category")
    return df
```

**効果:**
- `groupby()` が整数ベースのハッシュテーブルを使用 (10-50x高速)
- `merge()` がカテゴリコードで結合 (5-20x高速)
- `.isin()` がセットルックアップではなくコード比較に
- メモリ使用量も大幅削減 (string重複排除)

**リスク:**
- カテゴリの union 処理が必要 (merge 時のカテゴリ不一致)
- 既存テストの mock データで型アサーションが失敗する可能性

---

## Bottleneck #3: load_odds_time_series_range 3回ロード (~360s)

### 現状

452MB (jodds_tanpuku) のオッズ時系列データが3回ロードされる:

| 呼び出し元 | ファイル | 行 | 期間 |
|-----------|---------|-----|------|
| Training | `training_pipeline.py` | 124 | train_start〜train_end |
| Calibration BT | `engine.py` | 553 | train_start〜train_end (**重複**) |
| Test BT | `engine.py` | 553 | test_start〜test_end |

`_coerce_types()` も3回実行され、全列の `pd.to_numeric()` が走る。

### 改善案

**案A: ParquetStore に LRU cache 追加 (推定効果: -120s)**
- 同一 (table, subpath, filters) の read 結果をキャッシュ
- TTL なし (1実行中にデータ変更なし)
- メモリ消費に注意 (452MB x 2期間 = ~900MB)

**案B: 呼び出し側でのデータ受け渡し (推定効果: -120s)**
- `BacktestEngine.__init__()` に `preloaded_odds_ts: DataFrame` を追加
- Training でロード済みのデータを Calibration BT に直接渡す

**案C: 新ETL形式で _coerce_types スキップ (推定効果: -80s)**
- `jodds_tanpuku.parquet` は既に正しい型で保存されている
- `_coerce_types` に `if all(df[col].dtype != object for col in df.columns): return df`
  の早期returnを追加

---

## Bottleneck #4: MLflow pip推論オーバーヘッド (~97s)

### 現状

`training_pipeline.py:1213-1357` の `_log_to_mlflow()` で、
**26回の `mlflow.lightgbm.log_model()`** が呼ばれる。
毎回 `infer_pip_requirements()` が実行され、子プロセスで `pip freeze` を起動。

### 改善案

**案: 明示的 pip_requirements 指定 (推定効果: -80s)**

```python
# 変更前
mlflow.lightgbm.log_model(model, "model_name")

# 変更後
_PIP_REQS = ["lightgbm", "scikit-learn", "pandas", "numpy", "joblib"]
mlflow.lightgbm.log_model(model, "model_name", pip_requirements=_PIPS_REQS)
```

1回あたり ~3.7s の pip推論が 26回で ~97s → ほぼゼロに。

---

## Bottleneck #5: HorseHistoryFeatures itertuples ループ (~566s)

### 現状

`horse_history_features.py:592` で `itertuples()` により全馬を Python レベルで
ループ処理。1馬あたり数十の numpy 操作を実行。

3回呼ばれる:
1. `_train_submodel()` (芝) — line 336
2. `_train_submodel()` (ダート) — line 336
3. `BacktestEngine.run()` (テスト) — line 663

### 改善案

**案A: Vector化 (推定効果: -200〜300s)**
- `itertuples` ループを groupby + rolling に置換
- 過去成績の集計を expanding window でベクトル化
- 困難: フォームサイクル等の複雑なロジックは直感的に vector化しにくい

**案B: Numba/Cython (推定効果: -300〜400s)**
- ホットループを Numba JIT に置換
- numpy 配列のみを渡す (DataFrame accessor なし)
- 高効果だが実装コストが高い

**案C: インスタンスキャッシュの強化 (推定効果: -100s)**
- 現在のインスタンスキャッシュ `_entries_cache` / `_races_cache` を
  クラス変数に昇格 (同一プロセス内での再利用)
- HorseHistoryFeatures インスタンス間で履歴データを共有

---

## Bottleneck #6: BacktestEngine per-race オーバーヘッド

### 現状

`engine.py:765-1166` のレースループで、各レースごとに:

| 操作 | コスト |
|------|--------|
| `race_df_single.copy()` | DataFrameコピー |
| `add_race_transforms()` | 既に計算済みの特徴量を再計算 |
| `RegimeDetector.detect()` | DataFrame再構築 |
| 診断ログ `log_horse()` | 20属性/horse の getattr ループ |

### 改善案

**案: add_race_transforms の重複排除 (推定効果: -50〜100s)**
- `engine.py` で pre-computed features に race_rank 等を含める
- `RacePredictor.predict()` 内の `add_race_transforms()` をスキップ

---

## 改善優先順位と期待効果

| 優先度 | 改善案 | 推定削減時間 | 実装難易度 | リスク |
|--------|--------|-------------|-----------|--------|
| **P0** | #1-B: キャリブレーションBT期間短縮 | **-3000〜3500s** | 低 | 低 |
| **P0** | #1-C: キャリブレーション排除 (静的パラメータ) | **-4900s** | 中 | 中 |
| **P1** | #3-B: odds_ts データ受け渡し | **-120s** | 低 | 低 |
| **P1** | #4: MLflow pip指定 | **-80s** | 極低 | 無 |
| **P1** | #3-C: _coerce_types 早期return | **-80s** | 極低 | 無 |
| **P2** | #2: Categorical化 | **-300〜500s** | 中 | 中 |
| **P2** | #5-C: インスタンスキャッシュ強化 | **-100s** | 低 | 低 |
| **P3** | #5-A/B: HorseHistoryFeatures vector化 | **-200〜400s** | 高 | 高 |
| **P3** | #6: per-race重複排除 | **-50〜100s** | 中 | 低 |

### 推定累積効果 (P0+P1)

**現在: ~8239s (2.29h) → 最適化後: ~3859〜4659s (1.07〜1.30h)**

P0+P1のみで **43〜53%の短縮** (目標の「半減」をほぼ達成)。

---

## Verification Plan

各改善案の効果検証方法:

1. **実行時間計測スクリプト** — 各Phase A/B/C/Dのwall timeを自動計測
2. **結果一致テスト** — 最適化前後で bet_history, ROI, total_bets が一致することを確認
3. **段階的適用** — P0→P1→P2の順で適用し、各段階の効果を測定

---

## Files Changed

| ファイル | 変更内容 |
|---------|---------|
| `scripts/run_backtest.py` | キャリブレーション期間の短縮/制御 |
| `src/db/parquet_store.py` | LRU cache / Categorical 変換 |
| `src/db/readers.py` | _coerce_types 早期return |
| `src/pipelines/training_pipeline.py` | MLflow pip指定、データ受け渡し |
| `src/backtest/engine.py` | preloaded_data 対応 |
| `src/features/horse_history_features.py` | キャッシュ強化 |
