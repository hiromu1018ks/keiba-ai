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

### Task 2.1: 種牡馬産駒統計の事前計算スクリプト

**Files:**
- Create: `scripts/precompute_sire_stats.py`
- Create: `tests/test_precompute_sire_stats.py`

- [ ] **Step 1: テストを書く**

```python
def test_precompute_sire_stats_creates_parquet(tmp_path):
    """種牡馬産駒累積統計が正しく計算される"""
    import pandas as pd
    from pathlib import Path

    # モックデータ: 2頭の馬、同じ種牡馬
    entries = pd.DataFrame({
        "kettonum": ["001", "002", "001", "002"],
        "race_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]),
        "race_id": ["R1", "R1", "R2", "R2"],
        "kakuteijyuni": [1, 2, 3, 1],
        "kyori": [1600, 1600, 1800, 1800],
        "trackcd": [11, 11, 11, 11],
        "track_condition_code": [1, 1, 2, 2],
    })
    horses = pd.DataFrame({
        "kettonum": ["001", "002"],
        "ketto3infohansyokunum1": ["SIRE_A", "SIRE_A"],
    })

    result = compute_sire_stats(entries, horses)

    # 出力列の確認
    assert "sire_id" in result.columns
    assert "sire_starts" in result.columns
    assert "sire_wins" in result.columns
    assert "sire_turf_starts" in result.columns
    assert "sire_short_starts" in result.columns

    # PIT: cumsum().shift(1) — 当日の結果を含まない
    # 2/1 の時点で sire_starts は 1/1 の分だけ (2頭 x 1レース = 2)
    row_feb = result[result["race_date"] == "2024-02-01"]
    assert row_feb["sire_starts"].iloc[0] == 2  # 1/1の2件
    assert row_feb["sire_wins"].iloc[0] == 1    # 1/1のkettonum=001が1着
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_precompute_sire_stats.py -v`
Expected: FAIL (モジュールが存在しない)

- [ ] **Step 3: precompute_sire_stats.py を実装**

```python
#!/usr/bin/env python3
"""種牡馬産駒累積統計の事前計算スクリプト。

PIT保証: cumsum().shift(1) により当日の結果を含まない。
出力: data/raw/sire_career_stats.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def compute_sire_stats(
    entries_df: pd.DataFrame,
    horses_df: pd.DataFrame,
) -> pd.DataFrame:
    """種牡馬ごとの日次産駒成績を累積計算する。

    Args:
        entries_df: entries.parquet (kettonum, race_date, race_id, kakuteijyuni, ...)
        horses_df: horses.parquet (kettonum, ketto3infohansyokunum1, ...)

    Returns:
        sire_career_stats: (sire_id, race_date) ごとの累積統計
    """
    # entries → horses → sire_id を結合
    sire_map = horses_df.set_index("kettonum")["ketto3infohansyokunum1"]
    entries = entries_df.copy()
    entries["sire_id"] = entries["kettonum"].map(sire_map)

    # サーフェス判定 (trackcd: 10-22=turf, 23-29=dirt)
    entries["is_turf"] = entries["trackcd"].between(10, 22)
    entries["is_short"] = entries["kyori"] <= 1600
    entries["is_win"] = entries["kakuteijyuni"] == 1
    entries["is_place"] = entries["kakuteijyuni"].between(1, 3)

    # (sire_id, race_date) で日次集計
    daily = entries.groupby(["sire_id", "race_date"]).agg(
        daily_starts=("kakuteijyuni", "count"),
        daily_wins=("is_win", "sum"),
        daily_places=("is_place", "sum"),
        daily_turf_starts=("is_turf", "sum"),
        daily_turf_wins=("is_win", lambda x: x[entries.loc[x.index, "is_turf"]].sum()),
        daily_dirt_starts=("is_turf", lambda x: (~entries.loc[x.index, "is_turf"]).sum()),
        daily_dirt_wins=("is_win", lambda x: x[~entries.loc[x.index, "is_turf"]].sum()),
        daily_short_starts=("is_short", "sum"),
        daily_short_wins=("is_win", lambda x: x[entries.loc[x.index, "is_short"]].sum()),
        daily_long_starts=("is_short", lambda x: (~entries.loc[x.index, "is_short"]).sum()),
        daily_long_wins=("is_win", lambda x: x[~entries.loc[x.index, "is_short"]].sum()),
    ).reset_index()

    # cumsum + shift(1) で PIT保証
    cum_cols = [c for c in daily.columns if c.startswith("daily_")]
    for col in cum_cols:
        prefix = col.replace("daily_", "sire_")
        daily[prefix] = daily.groupby("sire_id")[col].cumsum().shift(1).fillna(0).astype(int)

    # 不要な daily_* 列を削除
    result = daily.drop(columns=cum_cols)

    # NaN の sire_id を除外
    result = result.dropna(subset=["sire_id"])

    return result


def main() -> None:
    data_dir = _PROJECT_ROOT / "data"
    entries = pd.read_parquet(data_dir / "raw" / "entries.parquet")
    horses = pd.read_parquet(data_dir / "raw" / "horses.parquet")
    races = pd.read_parquet(data_dir / "raw" / "races.parquet")

    # entries に race 条件をマージ
    race_info = races[["race_id", "trackcd", "kyori", "track_condition_code"]].copy()
    entries = entries.merge(race_info, on="race_id", how="left")

    stats = compute_sire_stats(entries, horses)
    out_path = data_dir / "raw" / "sire_career_stats.parquet"
    stats.to_parquet(out_path, index=False)
    print(f"Saved {len(stats)} rows to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_precompute_sire_stats.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add scripts/precompute_sire_stats.py tests/test_precompute_sire_stats.py
git commit -m "feat: 種牡馬産駒累積統計の事前計算スクリプトを追加"
```

---

### Task 2.2: SireFeatures モジュール + readers.load_sire_stats()

**Files:**
- Create: `src/features/sire_features.py`
- Create: `tests/test_sire_features.py`
- Modify: `src/db/readers.py` (load_sire_stats 追加)

- [ ] **Step 1: readers.py に load_sire_stats() を追加**

`src/db/readers.py` line 275 の後に追加:

```python
def load_sire_stats(store: ParquetStore) -> pd.DataFrame:
    """種牡馬産駎累積統計を読み込む。"""
    if not store.exists("raw", "sire_career_stats"):
        return pd.DataFrame()
    df = store.read("raw", "sire_career_stats")
    return _coerce_types(df)
```

- [ ] **Step 2: テストを書く (sire_features)**

```python
def test_sire_wr_beta_smoothed():
    """sire_wr が Beta 平滑化勝率を返す"""
    import pandas as pd
    from features.sire_features import SireFeatures

    sire_stats = pd.DataFrame({
        "sire_id": ["SIRE_A", "SIRE_A"],
        "race_date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
        "sire_starts": [0, 50],
        "sire_wins": [0, 6],
        "sire_places": [0, 15],
        "sire_turf_starts": [0, 30], "sire_turf_wins": [0, 4],
        "sire_dirt_starts": [0, 20], "sire_dirt_wins": [0, 2],
        "sire_short_starts": [0, 25], "sire_short_wins": [0, 4],
        "sire_long_starts": [0, 25], "sire_long_wins": [0, 2],
        "sire_prize_total": [0.0, 500000.0],
    })

    feat = SireFeatures(sire_stats)
    # 2024-06-01 時点: sire_starts=50, sire_wins=6
    row = feat.compute(sire_id="SIRE_A", race_date="2024-06-01",
                       surface="turf", kyori=1600)
    # Beta(1+6, 1+10+50-6) = 7/61 ≈ 0.115
    assert abs(row["sire_wr"] - 7/61) < 0.001
    # surface=turf → sire_turf_wr = (1+4)/(1+10+30-4) = 5/37
    assert abs(row["sire_surface_wr"] - 5/37) < 0.001
    # kyori=1600 (short) → sire_short_wr = (1+4)/(1+10+25-4) = 5/32
    assert abs(row["sire_distance_wr"] - 5/32) < 0.001

def test_sire_features_missing_sire_returns_nan():
    """未知の種牡馬はNaNを返す"""
    feat = SireFeatures(pd.DataFrame(columns=["sire_id", "race_date"]))
    row = feat.compute(sire_id="UNKNOWN", race_date="2024-01-01",
                       surface="turf", kyori=1600)
    assert pd.isna(row["sire_wr"])
```

- [ ] **Step 3: sire_features.py を実装**

```python
"""種牡馬産駒特徴量 — PIT安全な累積統計ベース"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _beta_smooth(wins: int, starts: int, alpha: int = 1, beta: int = 10) -> float:
    """Beta 平滑化勝率: (alpha + wins) / (alpha + beta + starts)"""
    return (alpha + wins) / (alpha + beta + starts)


class SireFeatures:
    """種牡馬産駒特徴量の計算 (PIT安全)"""

    def __init__(self, sire_stats_df: pd.DataFrame) -> None:
        self._stats = sire_stats_df
        if not self._stats.empty:
            self._stats = self._stats.sort_values(["sire_id", "race_date"])
            self._stats["_sire_date_key"] = (
                self._stats["sire_id"].astype(str) + "_" +
                self._stats["race_date"].astype(str)
            )

    def compute(
        self,
        sire_id: str | None,
        race_date: str | pd.Timestamp,
        surface: str,
        kyori: int,
    ) -> dict[str, float]:
        """1頭分の種牡馬特徴量を計算"""
        result: dict[str, float] = {}

        if sire_id is None or pd.isna(sire_id) or self._stats.empty:
            for col in ["sire_wr", "sire_place_rate", "sire_surface_wr",
                        "sire_distance_wr", "sire_prize_avg"]:
                result[col] = np.nan
            return result

        # searchsorted で該当日以前の最新行を取得
        mask = self._stats["sire_id"] == sire_id
        subset = self._stats[mask]
        if subset.empty:
            for col in ["sire_wr", "sire_place_rate", "sire_surface_wr",
                        "sire_distance_wr", "sire_prize_avg"]:
                result[col] = np.nan
            return result

        ts = pd.Timestamp(race_date)
        idx = subset["race_date"].searchsorted(ts, side="right") - 1
        if idx < 0:
            # 当日以前のデータなし → Beta(1,10) 事前分布
            prior = _beta_smooth(0, 0)
            result["sire_wr"] = prior
            result["sire_place_rate"] = prior
            result["sire_surface_wr"] = prior
            result["sire_distance_wr"] = prior
            result["sire_prize_avg"] = 0.0
            return result

        row = subset.iloc[idx]

        # 全体勝率
        result["sire_wr"] = _beta_smooth(int(row.get("sire_wins", 0)),
                                          int(row.get("sire_starts", 0)))
        # 複勝率
        result["sire_place_rate"] = _beta_smooth(int(row.get("sire_places", 0)),
                                                   int(row.get("sire_starts", 0)))
        # サーフェス別勝率
        if surface == "turf":
            result["sire_surface_wr"] = _beta_smooth(
                int(row.get("sire_turf_wins", 0)),
                int(row.get("sire_turf_starts", 0)))
        else:
            result["sire_surface_wr"] = _beta_smooth(
                int(row.get("sire_dirt_wins", 0)),
                int(row.get("sire_dirt_starts", 0)))

        # 距離別勝率
        if kyori <= 1600:
            result["sire_distance_wr"] = _beta_smooth(
                int(row.get("sire_short_wins", 0)),
                int(row.get("sire_short_starts", 0)))
        else:
            result["sire_distance_wr"] = _beta_smooth(
                int(row.get("sire_long_wins", 0)),
                int(row.get("sire_long_starts", 0)))

        # 平均賞金
        starts = max(1, int(row.get("sire_starts", 0)))
        result["sire_prize_avg"] = float(np.log1p(row.get("sire_prize_total", 0) / starts))

        return result
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_sire_features.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/features/sire_features.py src/db/readers.py tests/test_sire_features.py
git commit -m "feat: SireFeatures モジュール + load_sire_stats() を追加"
```

---

### Task 2.3: モデル統合 (AbilityModel, PlaceAbilityModel) + パイプライン wiring

**Files:**
- Modify: `src/models/stage1_ability_model.py` (FEATURE_COLS: 37→42)
- Modify: `src/models/place_ability_model.py` (FEATURE_COLS: 38→43)
- Modify: `src/pipelines/training_pipeline.py` (sire_features 呼び出し)

- [ ] **Step 1: AbilityModel.FEATURE_COLS に5列追加**

`src/models/stage1_ability_model.py` の FEATURE_COLS に追加:

```python
# 種牡馬産駎 (5)
"sire_wr",
"sire_surface_wr",
"sire_distance_wr",
"sire_prize_avg",
"bms_wr",
```

合計: 37 + 5 = 42

- [ ] **Step 2: PlaceAbilityModel.FEATURE_COLS に5列追加**

`src/models/place_ability_model.py` の FEATURE_COLS に同じ5列を追加。
合計: 38 + 5 = 43

- [ ] **Step 3: training_pipeline.py に sire_features 呼び出しを追加**

`_train_submodel` 内で、`FeatureEngine.build_all()` の後に sire 特徴量の計算を追加:

```python
from features.sire_features import SireFeatures

# 種牡馬産駎特徴量の追加
sire_stats = load_sire_stats(store)
if not sire_stats.empty:
    horses_df = store.read("raw", "horses")
    sire_feat = SireFeatures(sire_stats)
    # entry_df に sire_id 列を追加 (kettonum → horses → sire_id)
    sire_map = horses_df.set_index("kettonum")["ketto3infohansyokunum1"]
    df["sire_id"] = df["kettonum"].map(sire_map)
    # bms_id (母父) も同様
    bms_map = horses_df.set_index("kettonum")["ketto3infohansyokunum3"]
    df["bms_id"] = df["kettonum"].map(bms_map)
    # 各行に特徴量を計算
    for col in ["sire_wr", "sire_surface_wr", "sire_distance_wr", "sire_prize_avg", "bms_wr"]:
        df[col] = np.nan
    for idx, row in df.iterrows():
        surface = row.get("surface", "turf")
        kyori = int(row.get("kyori", 1600)) if pd.notna(row.get("kyori")) else 1600
        feats = sire_feat.compute(row.get("sire_id"), row.get("race_date"), surface, kyori)
        for k, v in feats.items():
            df.at[idx, k] = v
        # bms_wr (母父)
        bms_feats = sire_feat.compute(row.get("bms_id"), row.get("race_date"), surface, kyori)
        df.at[idx, "bms_wr"] = bms_feats.get("sire_wr", np.nan)
```

**注:** 本番実装では `iterrows` をベクトル化するが、初期実装では可読性優先。パフォーマンス問題があれば後で最適化。

- [ ] **Step 4: テスト実行**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: コミット**

```bash
git add src/models/stage1_ability_model.py src/models/place_ability_model.py src/pipelines/training_pipeline.py
git commit -m "feat: sire_features を Stage1 モデルに統合 (37→42, 38→43)"
```

---

### Task 2.4: Phase 2 バックテスト検証

- [ ] **Step 1: precompute_sire_stats 実行**

Run: `python scripts/precompute_sire_stats.py`

- [ ] **Step 2: 全テスト実行**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: バックテスト実行**

Run: `python scripts/run_backtest.py --train-start 20210101 --train-end 20241231 --test-start 20250101 --test-end 20251231 --ensemble`

- [ ] **Step 4: 結果を記録**

`docs/backlog/2026-04-13-phase2-result.md` にバックテスト結果を記録。
比較対象: Phase 1 完了後のROI。
失敗基準: ROI低下 > 5pt → Feature Importance を分析。

---

## Phase 3: Market Model OOF対応

### Task 3.1: MarketModel.predict_oof() 実装

**Files:**
- Modify: `src/models/market_model.py` (predict_oof メソッド追加)
- Create: `tests/test_market_model_oof.py`

- [ ] **Step 1: テストを書く**

```python
def test_predict_oof_returns_oof_predictions():
    """predict_oof が全データのOOF予測値を返す"""
    import pandas as pd
    import numpy as np
    from models.market_model import MarketModel

    n = 1000
    df = pd.DataFrame({
        "p_market_win_adj": np.random.rand(n),
        "surface": np.random.choice(["turf", "dirt"], n),
        "distance_bin": np.random.choice(["sprint", "mile", "intermediate"], n),
        "track_condition_code": np.random.randint(1, 5, n).astype(float),
        "grade_code": np.random.choice(["A", "B", "C", "X"], n),
        "field_size": np.random.randint(8, 18, n).astype(float),
        "weight_diff_from_mean": np.random.randn(n),
        "difficulty_score": np.random.rand(n),
    })

    model = MarketModel()
    model.train(df, num_threads=1)
    oof = model.predict_oof(df, n_splits=5)

    # 全行の予測が返る
    assert len(oof) == n
    # NaNなし (全foldカバー)
    assert oof.notna().all()
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_market_model_oof.py -v`
Expected: FAIL (predict_oof が存在しない)

- [ ] **Step 3: predict_oof() を実装**

`src/models/market_model.py` の `predict_and_calc_error()` の後に追加:

```python
def predict_oof(self, df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """OOF (out-of-fold) 予測を生成し、DataFrame の該当列を上書きする。

    学習データ内で KFold CV を行い、各foldのvalid予測を結合。
    最後に全データで再学習して推論用モデルを更新。
    """
    from sklearn.model_selection import KFold

    features = df[self.FEATURE_COLS].copy()
    for col in features.columns:
        if pd.api.types.is_integer_dtype(features[col]):
            features[col] = features[col].astype(float)
    for col in ["surface", "distance_bin", "grade_code"]:
        if col in features.columns:
            features[col] = features[col].astype("category")

    target = df["p_market_win_adj"]
    oof_pred = pd.Series(np.nan, index=df.index, name="_p_market_pred_win_oof")

    kf = KFold(n_splits=n_splits, shuffle=False)
    for train_idx, valid_idx in kf.split(features):
        train_data = lgb.Dataset(features.iloc[train_idx], label=target.iloc[train_idx])
        valid_data = lgb.Dataset(
            features.iloc[valid_idx], label=target.iloc[valid_idx], reference=train_data
        )
        fold_model = lgb.train(
            {
                "objective": "regression_l1",
                "metric": "mae",
                "learning_rate": 0.03,
                "num_leaves": 31,
                "feature_fraction": 0.7,
                "verbose": -1,
            },
            train_data,
            num_boost_round=300,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        oof_pred.iloc[valid_idx] = fold_model.predict(
            features.iloc[valid_idx], num_iteration=fold_model.best_iteration
        )

    # 全データで再学習 (推論用)
    self.train(df)

    # OOF予測で log_error を再計算
    df = df.copy()
    df["_p_market_pred_win"] = oof_pred
    p_pred = oof_pred.clip(0.01, 0.99)
    p_actual = df["p_market_win_adj"].clip(0.01, 0.99)
    df["signed_log_error_win"] = np.log(p_pred / p_actual)
    df["abs_log_error_win"] = np.abs(df["signed_log_error_win"])

    return df
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_market_model_oof.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/models/market_model.py tests/test_market_model_oof.py
git commit -m "feat: MarketModel.predict_oof() を追加 (5-Fold CV)"
```

---

### Task 3.2: TrainingPipeline への OOF 統合

**Files:**
- Modify: `src/pipelines/training_pipeline.py` (line 305-306 の後)

- [ ] **Step 1: training_pipeline.py の _train_submodel を修正**

`line 305-306` の後:

```python
# Before:
market = MarketModel()
market.train(df, num_threads=num_threads)
df = market.predict_and_calc_error(df)

# After:
market = MarketModel()
market.train(df, num_threads=num_threads)
df = market.predict_and_calc_error(df)
# OOF予測で学習データのリークを除去 (テストデータは予測値そのまま)
train_mask = df["race_date"] < test_start_date
if train_mask.any():
    df_train = df[train_mask].copy()
    df_train = market.predict_oof(df_train, n_splits=5)
    df.loc[train_mask, "signed_log_error_win"] = df_train["signed_log_error_win"]
    df.loc[train_mask, "abs_log_error_win"] = df_train["abs_log_error_win"]
```

**注:** `test_start_date` は `_train_submodel` のパラメータとして渡す必要あり。
既存の呼び出し側で test_start を渡すように修正。

- [ ] **Step 2: テスト実行**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: コミット**

```bash
git add src/pipelines/training_pipeline.py
git commit -m "feat: Market Model OOF を TrainingPipeline に統合"
```

---

### Task 3.3: Phase 3 バックテスト検証

- [ ] **Step 1: 全テスト実行**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: バックテスト実行**

Run: `python scripts/run_backtest.py --train-start 20210101 --train-end 20241231 --test-start 20250101 --test-end 20251231 --ensemble`

- [ ] **Step 3: 結果を記録**

`docs/backlog/2026-04-13-phase3-result.md` にバックテスト結果を記録。
成功指標: BT/PT乖離の縮小 (現状 +24.3pt → 15pt以下が目標)。
注意: ROIは短期的に低下する可能性 (リーク排除の正常な動作)。

---

## Phase 4: 過去走拡張 + ペース適性 + コース適性

### Task 4.1: 過去走 3→5 拡張 + harontime_late_trend

**Files:**
- Modify: `src/features/horse_history_features.py`
- Modify: `src/models/stage1_ability_model.py` (FEATURE_COLS 更新)

- [ ] **Step 1: テストを書く**

```python
def test_harontimel5_avg_uses_5_races():
    """harontimel5_avg が5走分のハロンタイム平均を返す"""
    # 5走分のハロンタイムがあり、NaNをスキップして平均を計算することを確認
    ...
    assert result["harontimel5_avg"] == expected_avg

def test_harontime_late_trend():
    """harontime_late_trend が最後2走 vs 最初3走の差を返す"""
    # 最後2走が速ければ負の値 (改善傾向)
    ...
    assert result["harontime_late_trend"] < 0  # 改善
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_horse_history_features.py -v`
Expected: FAIL (harontimel5_avg 列が存在しない)

- [ ] **Step 3: horse_history_features.py を修正**

- `harontimel3_avg` → `harontimel5_avg`: 5走のタイムから平均を計算 (NaNスキップ)
- `harontimel3_zscore` → `harontimel5_zscore`: 5走に拡張
- 新規 `harontime_late_trend`: 最後2走平均 - 最初3走平均 (負=改善)

**注:** 既存の `harontimel3_avg` 列名を変更すると全モデルのFEATURE_COLSも更新が必要。
列名は `harontimel5_avg` にリネームし、モデル側も追随する。

- [ ] **Step 4: AbilityModel.FEATURE_COLS を更新**

`"harontimel3_avg"` → `"harontimel5_avg"`, `"harontimel3_zscore"` → `"harontimel5_zscore"`
+ `"harontime_late_trend"` を追加。

- [ ] **Step 5: テストが通ることを確認**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: コミット**

```bash
git add src/features/horse_history_features.py src/models/stage1_ability_model.py src/models/place_ability_model.py tests/
git commit -m "feat: 過去走3→5拡張 + harontime_late_trend を追加"
```

---

### Task 4.2: ペース適性特徴量

**Files:**
- Create: `src/features/pace_aptitude_features.py`
- Create: `tests/test_pace_aptitude_features.py`
- Modify: `src/models/stage1_ability_model.py` (FEATURE_COLS)
- Modify: `src/features/feature_engine.py` (wiring)

- [ ] **Step 1: テストを書く**

```python
def test_pace_aptitude_front偏好():
    """逃げ/先行馬が front_pace で好成績の場合、pace_aptitude > 0"""
    import pandas as pd
    from features.pace_aptitude_features import PaceAptitudeFeatures

    history = pd.DataFrame({
        "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "kakuteijyuni": [1, 2, 5],
        "jyuni1c": [1, 2, 3],
        "jyuni4c": [1, 2, 5],
        "kyakusitukubun_cd": [1, 1, 1],  # 逃げ
        "syussotosu": [10, 12, 10],
    })

    feat = PaceAptitudeFeatures()
    result = feat.compute(history, target_date="2024-04-01")
    # 逃げ馬が1C上位で好走 → front_pace_wr が高い
    assert result["front_pace_wr"] > 0
```

- [ ] **Step 2: pace_aptitude_features.py を実装**

```python
"""ペース適性特徴量 — 角通過順位から推定"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _beta_smooth(wins: int, starts: int, alpha: int = 1, beta: int = 10) -> float:
    return (alpha + wins) / (alpha + beta + starts)


class PaceAptitudeFeatures:
    """過去走の jyuni1c/jyuni4c からペース適性を計算"""

    def compute(self, history: pd.DataFrame, target_date: str | pd.Timestamp) -> dict[str, float]:
        """1頭分のペース適性特徴量を計算"""
        result: dict[str, float] = {
            "pace_aptitude": np.nan,
            "front_pace_wr": np.nan,
            "closing_pace_wr": np.nan,
        }
        if history.empty:
            return result

        ts = pd.Timestamp(target_date)
        past = history[history["race_date"] < ts]
        if len(past) < 2:
            return result

        # ペース分類: 1C着順上位が勝った = front pace レース
        # 1C着順 <= field_size/3 かつ着順が良い → front pace
        norm_finish = past["kakuteijyuni"] / past["syussotosu"]
        norm_1c = past["jyuni1c"] / past["syussotosu"]

        front_mask = norm_1c <= 0.33  # 前 Peyton
        closing_mask = norm_1c > 0.66  # 後ろ待ち

        # front pace での勝率
        front_races = past[front_mask]
        if len(front_races) > 0:
            result["front_pace_wr"] = _beta_smooth(
                int((front_races["kakuteijyuni"] == 1).sum()), len(front_races))
        else:
            result["front_pace_wr"] = _beta_smooth(0, 0)

        # closing pace での勝率
        closing_races = past[closing_mask]
        if len(closing_races) > 0:
            result["closing_pace_wr"] = _beta_smooth(
                int((closing_races["kakuteijyuni"] == 1).sum()), len(closing_races))
        else:
            result["closing_pace_wr"] = _beta_smooth(0, 0)

        # ペース適性: front vs closing の着順差
        front_avg = norm_finish[front_mask].mean() if front_mask.any() else np.nan
        closing_avg = norm_finish[closing_mask].mean() if closing_mask.any() else np.nan
        if pd.notna(front_avg) and pd.notna(closing_avg):
            result["pace_aptitude"] = float(closing_avg - front_avg)
        else:
            result["pace_aptitude"] = np.nan

        return result
```

- [ ] **Step 3: モデルの FEATURE_COLS に追加**

`AbilityModel`, `PlaceAbilityModel` の FEATURE_COLS に `"pace_aptitude"`, `"front_pace_wr"`, `"closing_pace_wr"` を追加。

- [ ] **Step 4: feature_engine.py に wiring 追加**

- [ ] **Step 5: テストが通ることを確認**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: コミット**

```bash
git add src/features/pace_aptitude_features.py tests/test_pace_aptitude_features.py src/models/stage1_ability_model.py src/models/place_ability_model.py src/features/feature_engine.py
git commit -m "feat: ペース適性特徴量を追加 (pace_aptitude, front_pace_wr, closing_pace_wr)"
```

---

### Task 4.3: コース別適性特徴量

**Files:**
- Create: `src/features/course_features.py`
- Create: `tests/test_course_features.py`
- Modify: `src/models/stage1_ability_model.py` (FEATURE_COLS)
- Modify: `src/features/feature_engine.py` (wiring)

- [ ] **Step 1: テストを書く**

```python
def test_course_wr_returns_beta_smoothed():
    """course_wr が競馬場別のBeta平滑化勝率を返す"""
    import pandas as pd
    from features.course_features import CourseFeatures

    history = pd.DataFrame({
        "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "jyocd": ["01", "01", "05"],
        "kakuteijyuni": [1, 3, 2],
        "distance_bin": ["sprint", "sprint", "mile"],
        "syussotosu": [10, 12, 8],
    })

    feat = CourseFeatures()
    result = feat.compute(history, jyocd="01", distance_bin="sprint",
                          target_date="2024-04-01")
    # 01競馬場のsprint: 1着1回/2出走 → Beta(2,12)
    assert abs(result["course_wr"] - 2/13) < 0.001
    # 01競馬場のsprint: 1着1回/2出走
    assert abs(result["course_distance_wr"] - 2/13) < 0.001
```

- [ ] **Step 2: course_features.py を実装**

```python
"""コース別適性特徴量 — 競馬場×距離帯の過去勝率"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _beta_smooth(wins: int, starts: int, alpha: int = 1, beta: int = 10) -> float:
    return (alpha + wins) / (alpha + beta + starts)


class CourseFeatures:
    """競馬場別・距離帯別の過去勝率を計算"""

    def compute(
        self,
        history: pd.DataFrame,
        jyocd: str,
        distance_bin: str,
        target_date: str | pd.Timestamp,
    ) -> dict[str, float]:
        """1頭分のコース適性特徴量を計算"""
        result: dict[str, float] = {"course_wr": np.nan, "course_distance_wr": np.nan}

        if history.empty:
            return result

        ts = pd.Timestamp(target_date)
        past = history[history["race_date"] < ts]
        if past.empty:
            return result

        # 競馬場別勝率
        venue_races = past[past["jyocd"] == jyocd]
        if len(venue_races) > 0:
            wins = int((venue_races["kakuteijyuni"] == 1).sum())
            result["course_wr"] = _beta_smooth(wins, len(venue_races))
        else:
            result["course_wr"] = _beta_smooth(0, 0)

        # 競馬場×距離帯別勝率
        vd_races = venue_races[venue_races["distance_bin"] == distance_bin]
        if len(vd_races) > 0:
            wins = int((vd_races["kakuteijyuni"] == 1).sum())
            result["course_distance_wr"] = _beta_smooth(wins, len(vd_races))
        else:
            result["course_distance_wr"] = _beta_smooth(0, 0)

        return result
```

- [ ] **Step 3: モデルの FEATURE_COLS に追加 + wiring**

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: コミット**

```bash
git add src/features/course_features.py tests/test_course_features.py src/models/stage1_ability_model.py src/features/feature_engine.py
git commit -m "feat: コース別適性特徴量を追加 (course_wr, course_distance_wr)"
```

---

### Task 4.4: Phase 4 バックテスト検証

- [ ] **Step 1: 全テスト実行**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: バックテスト実行**

Run: `python scripts/run_backtest.py --train-start 20210101 --train-end 20241231 --test-start 20250101 --test-end 20251231 --ensemble`

- [ ] **Step 3: 結果を記録**

`docs/backlog/2026-04-13-phase4-result.md` にバックテスト結果を記録。
比較対象: Phase 3 完了後のROI。
期待効果: ROI +3-8pt。

- [ ] **Step 4: 最終マルチ年度バックテスト**

Run: `python scripts/run_backtest.py --years 2023 2024 2025 --train-window 4 --ensemble --report`

全体の堅牢性を確認 (3年度すべてで黒字なら成功)。
