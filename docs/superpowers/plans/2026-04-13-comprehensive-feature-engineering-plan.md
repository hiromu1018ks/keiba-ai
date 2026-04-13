# 包括的特徴量エンジニアリング実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 4フェーズの段階的特徴量改善によりROI 98.8%→黒字化を達成する

**Architecture:** 各PhaseでTDD実装→バックテスト検証→コミット。各Phaseは独立してテスト可能。PIT安全性を全Phaseで厳守。

**Tech Stack:** Python 3.11, LightGBM, pandas, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-04-13-comprehensive-feature-engineering-design.md`

**Baseline:** ROI 98.8%, 499 bets, -¥610 (2021-2024学習/2025テスト/ensemble/flat/JRAのみ)

---

## File Structure

### Phase 1: 既存特徴量の穴埋めと活用
| File | Action | Responsibility |
|------|--------|----------------|
| `src/features/horse_career_stats.py` | Modify | baba_cd別累積統計追加 (line 72: race_infoにtrack_condition_code追加) |
| `scripts/precompute_career_stats.py` | Modify | 新しい出力列の追加 |
| `src/features/bloodline_features.py` | Modify | blood_condition_wr (line 112), blood_keito_cd (line 115) の実装 |
| `src/db/readers.py` | Modify | load_keito() 追加 (line 270のパターン) |
| `src/features/feature_engine.py` | Modify | build_all() に compute_flb_slope() 呼び出し追加 (line 112の後) |
| `src/pipelines/training_pipeline.py` | Modify | compute_roi_ema() 呼び出し追加 (_build_race_level_features内) |
| `src/models/two_stage_return_model.py` | Modify | FEATURE_COLS に odds_skewness 追加 |
| `src/models/ev_correction_model.py` | Modify | FEATURE_COLS に implied_prob_hhi 追加 |
| `src/models/race_quality_screener.py` | Modify | FEATURE_COLS に overround_ema, entropy_ema 追加 |
| `tests/test_bloodline_features.py` | Modify | condition_wr, keito_cd テスト追加 |
| `tests/test_readers.py` | Modify | load_keito テスト追加 |

### Phase 2: 真の種牡馬産駒特徴量
| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/precompute_sire_stats.py` | Create | 種牡馬産駒累積統計の事前計算スクリプト |
| `src/features/sire_features.py` | Create | 種牡馬産駒特徴量計算モジュール |
| `src/db/readers.py` | Modify | load_sire_stats() 追加 |
| `src/models/stage1_ability_model.py` | Modify | FEATURE_COLS に sire_* 5列追加 (37→42) |
| `src/models/place_ability_model.py` | Modify | FEATURE_COLS に sire_* 5列追加 (38→43) |
| `src/pipelines/training_pipeline.py` | Modify | sire_features 呼び出し追加 |
| `tests/test_sire_features.py` | Create | 種牡馬特徴量のユニットテスト |

### Phase 3: Market Model OOF対応
| File | Action | Responsibility |
|------|--------|----------------|
| `src/models/market_model.py` | Modify | predict_oof() メソッド追加 |
| `src/pipelines/training_pipeline.py` | Modify | OOF予測値の適用 (line 306の後) |
| `tests/test_market_model_oof.py` | Create | OOF予測のテスト |

### Phase 4: 過去走拡張 + ペース適性 + コース適性
| File | Action | Responsibility |
|------|--------|----------------|
| `src/features/horse_history_features.py` | Modify | harontime 3→5走拡張 + late_trend追加 |
| `src/features/pace_aptitude_features.py` | Create | ペース適性特徴量 |
| `src/features/course_features.py` | Create | コース別適性特徴量 |
| `src/models/stage1_ability_model.py` | Modify | FEATURE_COLS に新特徴量追加 |
| `src/features/feature_engine.py` | Modify | 新特徴量のwiring |
| `src/pipelines/training_pipeline.py` | Modify | 新特徴量のパイプライン統合 |
| `tests/test_pace_aptitude_features.py` | Create | ペース適性テスト |
| `tests/test_course_features.py` | Create | コース適性テスト |

---

## Phase 1: 既存特徴量の穴埋めと活用

### Task 1.1: `blood_condition_wr` — horse_career_stats に baba_cd 列追加

**Files:**
- Modify: `src/features/horse_career_stats.py:72-82`
- Modify: `scripts/precompute_career_stats.py`

- [ ] **Step 1: テストを書く**

`tests/test_horse_career_stats.py` に追加:

```python
def test_compute_condition_columns():
    """baba_cd別の累積成績が正しく計算される"""
    import pandas as pd
    from features.horse_career_stats import compute_career_stats

    entries = pd.DataFrame({
        "kettonum": ["001", "001", "001"],
        "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "race_id": ["R1", "R2", "R3"],
        "kakuteijyuni": [1, 3, 2],
        "trackcd": [11, 11, 11],
        "kyori": [1600, 1800, 1600],
        "track_condition_code": [1, 3, 2],
        "syussotosu": [10, 12, 8],
    })
    races = entries[["race_id", "trackcd", "kyori", "track_condition_code"]].copy()

    result = compute_career_stats(entries, races)

    # track_condition_code がマージされている
    assert "track_condition_code" in result.columns
    # 条件別累積列が存在する
    for col in ["turf_good_starts", "turf_good_wins", "turf_heavy_starts", "turf_heavy_wins"]:
        assert col in result.columns
    # PIT: shift(1) により当日の結果は含まれない
    row0 = result.iloc[0]
    assert row0["turf_good_starts"] == 0  # 最初のレース前は0
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_horse_career_stats.py::test_compute_condition_columns -v`
Expected: FAIL (track_condition_code 列がまだ存在しない)

- [ ] **Step 3: 実装 — horse_career_stats.py に baba_cd 処理を追加**

`src/features/horse_career_stats.py` line 72 を変更:

```python
# Before:
race_info = races_df[["race_id", "trackcd", "kyori"]].copy()

# After:
_race_cols = [c for c in ["race_id", "trackcd", "kyori", "track_condition_code"]
              if c in races_df.columns]
race_info = races_df[_race_cols].copy()
```

line 82 の後に追加 (surface/is_short の後):

```python
    # 馬場状態 (good: 1,2 / heavy: 3,4)
    if "track_condition_code" in ent.columns:
        ent["is_good"] = ent["track_condition_code"].isin([1, 2]).astype(int)
        ent["is_heavy"] = ent["track_condition_code"].isin([3, 4]).astype(int)
    else:
        ent["is_good"] = 0
        ent["is_heavy"] = 0
```

`_compute_cumulative_before` の累積列リストに追加:

```python
    # 条件別
    _add_cum("turf_good", is_turf & is_good)
    _add_cum("turf_heavy", is_turf & is_heavy)
    _add_cum("dirt_good", ~is_turf & is_good)
    _add_cum("dirt_heavy", ~is_turf & is_heavy)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_horse_career_stats.py::test_compute_condition_columns -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/features/horse_career_stats.py tests/test_horse_career_stats.py
git commit -m "feat: horse_career_stats に馬場状態別累積統計を追加"
```

---

### Task 1.2: `blood_condition_wr` — bloodline_features で計算実装

**Files:**
- Modify: `src/features/bloodline_features.py:112`
- Modify: `tests/test_bloodline_features.py`

- [ ] **Step 1: テストを書く**

```python
def test_blood_condition_wr_uses_track_condition():
    """blood_condition_wr が馬場状態別勝率を返す"""
    # good track (baba_cd=1) の場合、turf_good_wins/turf_good_starts を使用
    career_df = pd.DataFrame({
        "kettonum": ["001"],
        "race_date": [pd.Timestamp("2024-06-01")],
        "turf_good_starts": [10], "turf_good_wins": [3],
        "turf_heavy_starts": [5], "turf_heavy_wins": [1],
        "dirt_good_starts": [0], "dirt_good_wins": [0],
        "dirt_heavy_starts": [0], "dirt_heavy_wins": [0],
    })
    entry_row = pd.Series({
        "kettonum": "001", "race_date": pd.Timestamp("2024-06-01"),
    })
    # surface=turf, baba_cd=1 (good) → turf_good 勝率を使用
    result = compute_condition_wr(career_df, entry_row, surface="turf", baba_cd=1)
    expected = (1 + 3) / (1 + 10 + 10 - 3)  # Beta(1+3, 1+10+10-3)
    assert abs(result - expected) < 0.001
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_bloodline_features.py::test_blood_condition_wr_uses_track_condition -v`
Expected: FAIL

- [ ] **Step 3: 実装 — bloodline_features.py line 112 を np.nan から実計算に変更**

```python
# Before (line 112):
result["blood_condition_wr"] = np.nan

# After:
# --- 馬場状態別勝率 ---
surface = _get_surface(track_cd)
baba_cd = row.get("track_condition_code", np.nan)
if pd.notna(baba_cd):
    baba_cd = int(baba_cd)
    if surface == "turf":
        if baba_cd in (1, 2):
            wr = _beta_smooth(
                stats_row.get("turf_good_wins", 0), stats_row.get("turf_good_starts", 0)
            )
        else:
            wr = _beta_smooth(
                stats_row.get("turf_heavy_wins", 0), stats_row.get("turf_heavy_starts", 0)
            )
    else:
        if baba_cd in (1, 2):
            wr = _beta_smooth(
                stats_row.get("dirt_good_wins", 0), stats_row.get("dirt_good_starts", 0)
            )
        else:
            wr = _beta_smooth(
                stats_row.get("dirt_heavy_wins", 0), stats_row.get("dirt_heavy_starts", 0)
            )
    result["blood_condition_wr"] = wr
else:
    result["blood_condition_wr"] = np.nan
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_bloodline_features.py::test_blood_condition_wr_uses_track_condition -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/features/bloodline_features.py tests/test_bloodline_features.py
git commit -m "feat: blood_condition_wr を馬場状態別勝率で実装"
```

---

### Task 1.3: `blood_keito_cd` — readers + bloodline で実装

**Files:**
- Modify: `src/db/readers.py` (load_keito 追加)
- Modify: `src/features/bloodline_features.py:115`
- Modify: `tests/test_bloodline_features.py`, `tests/test_readers.py`

- [ ] **Step 1: テストを書く (readers)**

```python
def test_load_keito():
    """keito.parquet から系統コードマスタを読み込む"""
    store = ParquetStore(BASE_DIR)
    df = load_keito(store)
    # ファイルが存在しない場合は空DataFrame
    if df.empty:
        assert True
    else:
        assert "keitoucode" in df.columns
```

- [ ] **Step 2: load_keito() を実装**

`src/db/readers.py` に追加 (line 275 の後):

```python
def load_keito(store: ParquetStore) -> pd.DataFrame:
    """系統コードマスタを読み込む。"""
    if not store.exists("raw", "keito"):
        return pd.DataFrame()
    df = store.read("raw", "keito")
    return _coerce_types(df)
```

- [ ] **Step 3: テストを書く (bloodline)**

```python
def test_blood_keito_cd_from_sire():
    """blood_keito_cd が種牡馬の系統コードを返す"""
    # モックで horses と keito を用意
    # entries.kettonum -> horses.kettonum -> horses.ketto3infohansyokunum1 -> keito.keitoucode
    # "SS系" 等の系統コードが返ることを確認
    ...
```

- [ ] **Step 4: blood_keito_cd を実装**

`src/features/bloodline_features.py` line 115 を変更:

```python
# Before:
result["blood_keito_cd"] = np.nan

# After:
# --- 系統コード ---
# sire_keito_map は __init__ または compute() の冒頭でロード済み
sire_id = row.get("sire_id")  # horses.ketto3infohansyokunum1
if sire_id and sire_id in self._keito_map:
    result["blood_keito_cd"] = self._keito_map[sire_id]
else:
    result["blood_keito_cd"] = "unknown"
```

- [ ] **Step 5: テストが通ることを確認**

Run: `python -m pytest tests/test_bloodline_features.py tests/test_readers.py -v`
Expected: ALL PASS

- [ ] **Step 6: コミット**

```bash
git add src/db/readers.py src/features/bloodline_features.py tests/
git commit -m "feat: blood_keito_cd を種牡馬系統コードで実装"
```

---

### Task 1.4: `compute_flb_slope()` の wiring (odds_skewness, implied_prob_hhi)

**Files:**
- Modify: `src/features/feature_engine.py:112` (build_all 内)
- Modify: `src/models/two_stage_return_model.py` (FEATURE_COLS)
- Modify: `src/models/ev_correction_model.py` (FEATURE_COLS)

- [ ] **Step 1: feature_engine.py に呼び出し追加**

`src/features/feature_engine.py` line 112 の後に追加:

```python
from features.market_bias_features import compute_flb_slope

with TimingContext("build_all/flb_slope"):
    flb_result = compute_flb_slope(df)
    df["odds_skewness"] = flb_result["odds_skewness"]
    df["implied_prob_hhi"] = flb_result["implied_prob_hhi"]
```

- [ ] **Step 2: two_stage_return_model.py の FEATURE_COLS に追加**

`src/models/two_stage_return_model.py` の `FEATURE_COLS` に `"odds_skewness"` を追加。

- [ ] **Step 3: ev_correction_model.py の FEATURE_COLS に追加**

`src/models/ev_correction_model.py` の `FEATURE_COLS` に `"implied_prob_hhi"` を追加。

- [ ] **Step 4: テスト実行**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: コミット**

```bash
git add src/features/feature_engine.py src/models/two_stage_return_model.py src/models/ev_correction_model.py
git commit -m "feat: compute_flb_slope をパイプラインに統合 + odds_skewness, implied_prob_hhi をモデルに追加"
```

---

### Task 1.5: `compute_roi_ema()` の wiring (overround_ema, entropy_ema)

**Files:**
- Modify: `src/pipelines/training_pipeline.py` (_build_race_level_features 内)
- Modify: `src/models/race_quality_screener.py` (FEATURE_COLS)

- [ ] **Step 1: training_pipeline.py に呼び出し追加**

`_build_race_level_features()` 内 (line 524 の後) に追加:

```python
from features.odds_dynamics_features import compute_roi_ema

if "tanodds" in race_feat.columns:
    race_feat = compute_roi_ema(race_feat)
```

- [ ] **Step 2: race_quality_screener.py の FEATURE_COLS に追加**

`src/models/race_quality_screener.py` の `FEATURE_COLS` に `"overround_ema"`, `"entropy_ema"` を追加。

- [ ] **Step 3: テスト実行**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: コミット**

```bash
git add src/pipelines/training_pipeline.py src/models/race_quality_screener.py
git commit -m "feat: compute_roi_ema をパイプラインに統合 + overround_ema, entropy_ema をRaceQualityScreenerに追加"
```

---

### Task 1.6: Phase 1 全体テスト + バックテスト検証

- [ ] **Step 1: 全テスト実行**

Run: `python -m pytest tests/ -v --cov=src --cov-report=term-missing`
Expected: ALL PASS

- [ ] **Step 2: precompute_career_stats 再実行 (baba_cd列追加のため)**

Run: `python scripts/precompute_career_stats.py`

- [ ] **Step 3: バックテスト実行**

Run: `python scripts/run_backtest.py --train-start 20210101 --train-end 20241231 --test-start 20250101 --test-end 20251231 --ensemble`

- [ ] **Step 4: 結果を記録**

`docs/backlog/2026-04-13-phase1-result.md` にバックテスト結果を記録。
比較対象: ROI 98.8% (499 bets, -¥610)。

---

## Phase 2: 真の種牡馬産駒特徴量

<!-- Phase 2 の詳細は Phase 1 完了後に記述 -->

### Task 2.1: 種牡馬産駒統計の事前計算スクリプト
### Task 2.2: SireFeatures モジュール実装
### Task 2.3: モデル統合 (AbilityModel, PlaceAbilityModel)
### Task 2.4: Phase 2 バックテスト検証

---

## Phase 3: Market Model OOF対応

<!-- Phase 3 の詳細は Phase 2 完了後に記述 -->

### Task 3.1: MarketModel.predict_oof() 実装
### Task 3.2: TrainingPipeline への OOF 統合
### Task 3.3: Phase 3 バックテスト検証

---

## Phase 4: 過去走拡張 + ペース適性 + コース適性

<!-- Phase 4 の詳細は Phase 3 完了後に記述 -->

### Task 4.1: 過去走 3→5 拡張
### Task 4.2: ペース適性特徴量
### Task 4.3: コース別適性特徴量
### Task 4.4: Phase 4 バックテスト検証
