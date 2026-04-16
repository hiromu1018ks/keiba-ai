# Odds Movement Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** レース直前のオッズ変動（急落・急騰）が複勝率や回収率に与える影響を統計的に分析するスタンドアロンスクリプトを作成する。

**Architecture:** 単一スクリプト `scripts/analyze_odds_movement.py` が4つのParquetファイル（jodds_tanpuku, entries, races, payouts）を直接pandasで読み込み、オッズ変動特徴量をベクトル化計算し、3次元のクロス集計（基本統計/騎手調教師/レース条件）を実行してコンソール+CSVに出力する。MLパイプラインには依存しない。

**Tech Stack:** Python 3.11, pandas, pyarrow, argparse, logging (標準ライブラリのみ + プロジェクト既存依存)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `scripts/analyze_odds_movement.py` | メインスクリプト：CLI、データ読み込み、特徴量計算、分析、出力 |
| `tests/test_analyze_odds_movement.py` | テスト：特徴量計算、分類、ROI計算のユニットテスト |

**No modifications to existing files.** This is a pure-addition script.

---

### Task 1: スクリプト雛形とデータ読み込み関数

**Files:**
- Create: `scripts/analyze_odds_movement.py`
- Test: `tests/test_analyze_odds_movement.py` (Task 2で作成)

- [ ] **Step 1: スクリプト雛形を作成**

以下の構成で `scripts/analyze_odds_movement.py` を作成:

```python
"""オッズ急落・急騰分析スクリプト

レース直前のオッズ変動が複勝率や回収率に与える影響を統計的に分析する。

Usage:
    python scripts/analyze_odds_movement.py --start 20230101 --end 20251231
    python scripts/analyze_odds_movement.py --start 20240101 --end 20251231 --detail
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── プロジェクトルート設定 ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(ROOT) / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="オッズ変動分析")
    parser.add_argument("--start", type=str, default="20230101",
                        help="開始日 YYYYMMDD (default: 20230101)")
    parser.add_argument("--end", type=str, default="20251231",
                        help="終了日 YYYYMMDD (default: 20251231)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="出力ディレクトリ (default: output/odds_movement_analysis_{date})")
    parser.add_argument("--drop-threshold", type=float, default=0.20,
                        help="分類閾値 (default: 0.20)")
    parser.add_argument("--min-points", type=int, default=5,
                        help="1頭あたり最低データポイント数 (default: 5)")
    parser.add_argument("--detail", action="store_true",
                        help="詳細レコードCSVも出力")
    return parser


def load_time_series(start_year: int, end_year: int) -> pd.DataFrame:
    """jodds_tanpuku.parquet を読み込み、年フィルタ適用"""
    path = DATA_DIR / "odds" / "jodds_tanpuku.parquet"
    if not path.exists():
        raise FileNotFoundError(f"jodds_tanpuku.parquet not found at {path}")

    logger.info("Loading jodds_tanpuku.parquet (year %d-%d)...", start_year, end_year)
    # year列はstring型なのでフィルタ値も文字列にする（pyarrow述語プッシュダウン対応）
    df = pd.read_parquet(
        path,
        filters=[("year", ">=", str(start_year)), ("year", "<=", str(end_year))],
    )
    logger.info("Loaded %d rows", len(df))
    return df


def load_entries(start_date: str, end_date: str) -> pd.DataFrame:
    """entries.parquet を読み込み、日付フィルタ適用"""
    path = DATA_DIR / "raw" / "entries.parquet"
    df = pd.read_parquet(path)
    # race_date は datetime64 なので文字列に変換して比較
    df["_race_date_str"] = pd.to_datetime(df["race_date"]).dt.strftime("%Y%m%d")
    df = df[(df["_race_date_str"] >= start_date) & (df["_race_date_str"] <= end_date)]
    df = df.drop(columns=["_race_date_str"])
    logger.info("Loaded %d entries", len(df))
    return df


def load_races(start_date: str, end_date: str) -> pd.DataFrame:
    """races.parquet を読み込み、日付フィルタ適用"""
    path = DATA_DIR / "raw" / "races.parquet"
    df = pd.read_parquet(path)
    df["_race_date_str"] = pd.to_datetime(df["race_date"]).dt.strftime("%Y%m%d")
    df = df[(df["_race_date_str"] >= start_date) & (df["_race_date_str"] <= end_date)]
    df = df.drop(columns=["_race_date_str"])
    logger.info("Loaded %d races", len(df))
    return df


def load_payouts(start_date: str, end_date: str) -> pd.DataFrame:
    """payouts.parquet を読み込み、確定結果のみ抽出"""
    path = DATA_DIR / "raw" / "payouts.parquet"
    df = pd.read_parquet(path)
    df["_race_date_str"] = pd.to_datetime(df["race_date"]).dt.strftime("%Y%m%d")
    df = df[(df["_race_date_str"] >= start_date) & (df["_race_date_str"] <= end_date)]
    df = df.drop(columns=["_race_date_str"])
    # 確定結果のみ (datakubun='2')
    df = df[df["datakubun"] == "2"]
    logger.info("Loaded %d confirmed payouts", len(df))
    return df


if __name__ == "__main__":
    main()
```

**重要ポイント:**
- `load_time_series` は年フィルタ（int）、他は日付フィルタ（YYYYMMDD文字列）を使用 — jodds_tanpuku は year 列があるが他ファイルは race_date のみ
- `sys.path.insert(0, ROOT)` でプロジェクトルートを設定（他スクリプトと同じパターン）
- `load_payouts` で `datakubun == '2'` フィルタ（'0'ではない）

- [ ] **Step 2: 雛形のシンタックス確認**

Run: `python -c "import ast; ast.parse(open('scripts/analyze_odds_movement.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: コミット**

```bash
git add scripts/analyze_odds_movement.py
git commit -m "feat: オッズ急落分析スクリプトの雛形とデータ読み込み関数"
```

---

### Task 2: コア特徴量計算（compute_movement_features）

**Files:**
- Modify: `scripts/analyze_odds_movement.py`
- Create: `tests/test_analyze_odds_movement.py`

- [ ] **Step 1: compute_movement_features 関数を実装**

```python
def compute_movement_features(ts_df: pd.DataFrame) -> pd.DataFrame:
    """時系列オッズから各馬のオッズ変動特徴量をベクトル化計算

    Args:
        ts_df: jodds_tanpuku データ。必須列: race_id, umaban(str),
               happyotime(str), tanodds(float), tanninki(Int64), race_date(datetime)

    Returns:
        各(race_id, umaban)ごとに1行のDataFrame。
        列: race_id, umaban, early_odds, mid_odds, late_odds, final_odds,
            early_pop, late_pop, n_points,
            odds_drop_60_10, odds_drop_30_10, odds_drop_10_final,
            pop_change_30_10
    """
    # ── 前処理 ──
    df = ts_df.copy()

    # umaban を string → int に正規化（結合用）
    df["umaban_int"] = pd.to_numeric(df["umaban"], errors="coerce").astype("Int64")

    # tanninki の NaN を -1 で埋める
    df["tanninki"] = df["tanninki"].fillna(-1)

    # 有効なオッズのみ残す（ゼロとNaN除外）
    df = df[df["tanodds"].notna() & (df["tanodds"] > 0)]

    # NAR除外 (jyocdはobject型なので数値変換して比較)
    if "jyocd" in df.columns:
        jyocd_num = pd.to_numeric(df["jyocd"], errors="coerce")
        df = df[jyocd_num < 30]

    # ソート: (race_id, umaban) ごとに (race_date, happyotime) で昇順
    df = df.sort_values(["race_id", "umaban", "race_date", "happyotime"])

    # ── groupby agg ──
    def _first(series):
        return series.iloc[0]

    def _mid(series):
        idx = len(series) // 2
        return series.iloc[idx]

    def _late(series):
        idx = int(len(series) * 0.9)
        return series.iloc[idx]

    g = df.groupby(["race_id", "umaban"], sort=False)

    features = g.agg(
        early_odds=("tanodds", _first),
        mid_odds=("tanodds", _mid),
        late_odds=("tanodds", _late),
        final_odds=("tanodds", "last"),
        early_pop=("tanninki", _first),
        late_pop=("tanninki", _late),
        n_points=("tanodds", "count"),
    ).reset_index()

    # ── 変動率計算 ──
    features["odds_drop_60_10"] = (features["early_odds"] - features["late_odds"]) / features["early_odds"]
    features["odds_drop_30_10"] = (features["mid_odds"] - features["late_odds"]) / features["mid_odds"]
    features["odds_drop_10_final"] = (features["late_odds"] - features["final_odds"]) / features["late_odds"]
    features["pop_change_30_10"] = features["mid_pop"] - features["late_pop"]  # 30→10分の人気変化

    return features
```

- [ ] **Step 2: テストを書く**

Create `tests/test_analyze_odds_movement.py`:

```python
"""tests/test_analyze_odds_movement.py — オッズ変動分析のユニットテスト"""

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_odds_movement import compute_movement_features, classify_movement


@pytest.fixture
def sample_time_series():
    """3頭×5時点のサンプル時系列データ"""
    rows = []
    base_time = "01051200"  # MMDDHHmm 形式

    # 馬1: 急落パターン (50→30→20→15→10)
    for i, odds in enumerate([50.0, 40.0, 30.0, 20.0, 10.0]):
        rows.append({
            "race_id": "20240101010101", "umaban": "1",
            "happyotime": base_time, "tanodds": odds,
            "tanninki": 10 - i, "race_date": pd.Timestamp("2024-01-01"),
            "year": 2024, "jyocd": 10,
        })
        base_time = f"0105{1200 + i * 10:04d}"  # 10分刻み

    # 馬2: 安定パターン (5.0→4.8→5.0→4.9→5.1)
    base_time2 = "01051200"
    for i, odds in enumerate([5.0, 4.8, 5.0, 4.9, 5.1]):
        rows.append({
            "race_id": "20240101010101", "umaban": "2",
            "happyotime": base_time2, "tanodds": odds,
            "tanninki": 2, "race_date": pd.Timestamp("2024-01-01"),
            "year": 2024, "jyocd": 10,
        })
        base_time2 = f"0105{1200 + i * 10:04d}"

    # 馬3: 急騰パターン (3.0→4.0→6.0→8.0→12.0)
    base_time3 = "01051200"
    for i, odds in enumerate([3.0, 4.0, 6.0, 8.0, 12.0]):
        rows.append({
            "race_id": "20240101010101", "umaban": "3",
            "happyotime": base_time3, "tanodds": odds,
            "tanninki": 1, "race_date": pd.Timestamp("2024-01-01"),
            "year": 2024, "jyocd": 10,
        })
        base_time3 = f"0105{1200 + i * 10:04d}"

    return pd.DataFrame(rows)


class TestComputeMovementFeatures:
    def test_returns_correct_shape(self, sample_time_series):
        result = compute_movement_features(sample_time_series)
        assert result.shape[0] == 3  # 3頭
        assert "odds_drop_60_10" in result.columns
        assert "odds_drop_30_10" in result.columns
        assert "n_points" in result.columns

    def test_steamer_detection(self, sample_time_series):
        """馬1: 50→10 で80%急落"""
        result = compute_movement_features(sample_time_series)
        horse1 = result[result["umaban"] == "1"].iloc[0]
        assert horse1["odds_drop_60_10"] > 0.5  # 50%以上低下
        assert horse1["final_odds"] == 10.0

    def test_stable_horse(self, sample_time_series):
        """馬2: ほぼ変動なし"""
        result = compute_movement_features(sample_time_series)
        horse2 = result[result["umaban"] == "2"].iloc[0]
        assert abs(horse2["odds_drop_60_10"]) < 0.1  # 10%未満

    def test_drifter_detection(self, sample_time_series):
        """馬3: 3→12 で急騰（オッズ上昇 = dropが負）"""
        result = compute_movement_features(sample_time_series)
        horse3 = result[result["umaban"] == "3"].iloc[0]
        assert horse3["odds_drop_60_10"] < -0.5  # 50%以上上昇（負のdrop）

    def test_n_points_count(self, sample_time_series):
        result = compute_movement_features(sample_time_series)
        assert (result["n_points"] == 5).all()

    def test_excludes_nan_odds(self):
        """NaNオッズの行は除外される"""
        df = pd.DataFrame([
            {"race_id": "r1", "umaban": "1", "happyotime": "01051200",
             "tanodds": float("nan"), "tanninki": 1,
             "race_date": pd.Timestamp("2024-01-01"), "year": 2024},
            {"race_id": "r1", "umaban": "1", "happyotime": "01051300",
             "tanodds": 5.0, "tanninki": 1,
             "race_date": pd.Timestamp("2024-01-01"), "year": 2024},
        ])
        result = compute_movement_features(df)
        assert len(result) == 1
        assert result.iloc[0]["n_points"] == 1

    def test_excludes_zero_odds(self):
        """ゼロオッズの行は除外される"""
        df = pd.DataFrame([
            {"race_id": "r1", "umaban": "1", "happyotime": "01051200",
             "tanodds": 0.0, "tanninki": 1,
             "race_date": pd.Timestamp("2024-01-01"), "year": 2024},
            {"race_id": "r1", "umaban": "1", "happyotime": "01051300",
             "tanodds": 5.0, "tanninki": 1,
             "race_date": pd.Timestamp("2024-01-01"), "year": 2024},
        ])
        result = compute_movement_features(df)
        assert result.iloc[0]["n_points"] == 1

    def test_excludes_nar_races(self):
        """NARレース(jyocd>=30)は除外"""
        df = pd.DataFrame([
            {"race_id": "r1", "umaban": "1", "happyotime": "01051200",
             "tanodds": 5.0, "tanninki": 1,
             "race_date": pd.Timestamp("2024-01-01"), "year": 2024, "jyocd": 35},
            {"race_id": "r1", "umaban": "1", "happyotime": "01051300",
             "tanodds": 4.0, "tanninki": 1,
             "race_date": pd.Timestamp("2024-01-01"), "year": 2024, "jyocd": 35},
        ])
        result = compute_movement_features(df)
        assert len(result) == 0
```

- [ ] **Step 3: テスト実行（失敗を確認）**

Run: `pytest tests/test_analyze_odds_movement.py::TestComputeMovementFeatures -v`
Expected: FAIL (module not found or import error — 関数がまだないため)

- [ ] **Step 4: 実装を追加**

`compute_movement_features` 関数を `scripts/analyze_odds_movement.py` に追加（Step 1のコードを貼り付け）。また `main()` 関数の空実装を追加:

```python
def main() -> None:
    args = build_parser().parse_args()
    start_year = int(args.start[:4])
    end_year = int(args.end[:4])

    logger.info("=" * 60)
    logger.info("オッズ変動分析: %s ~ %s", args.start, args.end)
    logger.info("=" * 60)

    # 1. データ読み込み
    ts_df = load_time_series(start_year, end_year)
    entries_df = load_entries(args.start, args.end)
    races_df = load_races(args.start, args.end)
    payouts_df = load_payouts(args.start, args.end)

    # 2. 特徴量計算
    movement_df = compute_movement_features(ts_df)
    logger.info("Computed movement features for %d horses", len(movement_df))

    # TODO: 残りのステップで実装
    logger.info("Analysis complete (placeholder).")
```

- [ ] **Step 5: テスト実行（成功を確認）**

Run: `pytest tests/test_analyze_odds_movement.py::TestComputeMovementFeatures -v`
Expected: 全てPASS

- [ ] **Step 6: コミット**

```bash
git add scripts/analyze_odds_movement.py tests/test_analyze_odds_movement.py
git commit -m "feat: compute_movement_features 実装とテスト"
```

---

### Task 3: 分類関数と結果結合

**Files:**
- Modify: `scripts/analyze_odds_movement.py`
- Modify: `tests/test_analyze_odds_movement.py`

- [ ] **Step 1: classify_movement 関数を実装**

```python
def classify_movement(
    df: pd.DataFrame,
    threshold: float = 0.20,
) -> pd.DataFrame:
    """オッズ変動量に基づいて Steamer/Stable/Drifter 分類

    Args:
        df: compute_movement_features の出力
        threshold: 分類閾値（デフォルト20%）

    Returns:
        分類列 ('movement_class', 'movement_bucket') が追加されたDataFrame
    """
    df = df.copy()
    drop = df["odds_drop_30_10"]  # 主要指標: 30→10分の変動

    def _bucket(x: float) -> str:
        if x >= 0.40:
            return "strong_drop"
        elif x >= 0.25:
            return "moderate_drop"
        elif x >= threshold:
            return "mild_drop"
        elif x > -threshold:
            return "stable"
        elif x >= -0.25:
            return "mild_rise"
        elif x >= -0.40:
            return "moderate_rise"
        else:
            return "strong_rise"

    def _category(x: float) -> str:
        if x >= threshold:
            return "steamer"
        elif x > -threshold:
            return "stable"
        else:
            return "drifter"

    df["movement_bucket"] = drop.apply(_bucket)
    df["movement_class"] = drop.apply(_category)

    return df
```

- [ ] **Step 2: join_results 関数を実装**

```python
def join_results(
    movement_df: pd.DataFrame,
    entries: pd.DataFrame,
    races: pd.DataFrame,
    payouts: pd.DataFrame,
    min_points: int = 5,
) -> pd.DataFrame:
    """オッズ変動特徴量に着順・払戻金・レース条件を結合

    Args:
        movement_df: classify_movement 後のDataFrame
        entries: entries.parquet 読み込み
        races: races.parquet 読み込み
        payouts: payouts.parquet 読み込み
        min_points: 最低データポイント数

    Returns:
        分析用完全結合DataFrame
    """
    df = movement_df.copy()

    # 最低ポイント数フィルタ
    df = df[df["n_points"] >= min_points].copy()
    logger.info("After min_points filter: %d horses", len(df))

    # umaban 型合わせ: movement側はstr → int
    df["umaban_int"] = pd.to_numeric(df["umaban"], errors="coerce").astype("Int64")

    # ── entries 結合 ──
    entry_cols = ["race_id", "umaban", "kakuteijyuni", "ninki",
                   "kisyucode", "chokyosicode"]
    entries_sub = entries[entry_cols].copy()
    # 両側ともstringで結合（movement側のumabanはgroupbyからstring、entries側もobject）
    entries_sub["umaban"] = entries_sub["umaban"].astype(str)
    df["umaban"] = df["umaban"].astype(str)
    df = df.merge(entries_sub, on=["race_id", "umaban"], how="left")

    # ── races 絶合（レース条件） ──
    race_cols = ["race_id", "kyori", "syussotosu", "trackcd"]
    # sibababacd / dirtbabacd があれば含める
    available_race_cols = [c for c in race_cols + ["sibababacd", "dirtbabacd"]
                           if c in races.columns]
    races_sub = races[available_race_cols].drop_duplicates("race_id")
    df = df.merge(races_sub, on="race_id", how="left")

    # surface マッピング（trackcd: 10-22=芝, 23-29=ダート）
    if "trackcd" in df.columns:
        def _map_surface(tc):
            if pd.isna(tc):
                return "other"
            tc_int = int(tc)
            if 10 <= tc_int <= 22:
                return "turf"
            elif 23 <= tc_int <= 29:
                return "dirt"
            return "other"
        df["surface"] = df["trackcd"].apply(_map_surface)

    # ── payouts 絶合（複勝払戻金） ──
    pay_cols = ["race_id"] + [f"payfukusyoumaban{i}" for i in range(1, 6)] \
              + [f"payfukusyopay{i}" for i in range(1, 6)]
    pay_available = [c for c in pay_cols if c in payouts.columns]
    payouts_sub = payouts[pay_available].drop_duplicates("race_id")
    df = df.merge(payouts_sub, on="race_id", how="left")

    # ── 複勝判定 & 払戻金取得 ──
    def _get_place_payout(row: pd.Series) -> float:
        if pd.isna(row.get("kakuteijyuni")) or row["kakuteijyuni"] == 0:
            return 0.0
        if row["kakuteijyuni"] > 3:
            return 0.0
        umaban_val = row.get("umaban_int", row.get("umaban"))
        if pd.isna(umaban_val):
            return 0.0
        try:
            umaban_int = int(umaban_val)
        except (ValueError, TypeError):
            return 0.0
        for i in range(1, 6):
            maban_col = f"payfukusyoumaban{i}"
            pay_col = f"payfukusyopay{i}"
            if maban_col not in row.index:
                continue
            maban = row[maban_col]
            if pd.notna(maban) and umaban_int == int(maban):
                payout = row[pay_col]
                return float(payout) if pd.notna(payout) else 0.0
        return 0.0

    df["place_payout"] = df.apply(_get_place_payout, axis=1)
    df["is_place"] = (df["place_payout"] > 0).astype(int)
    df["is_win"] = (df["kakuteijyuni"] == 1).astype(int)

    logger.info("Joined results: %d records (%d place hits)",
                len(df), df["is_place"].sum())
    return df
```

- [ ] **Step 3: 分類のテストを追加**

`tests/test_analyze_odds_movement.py` に追加:

```python
class TestClassifyMovement:
    def test_steamer_classification(self):
        df = pd.DataFrame({
            "race_id": ["r1"], "umaban": ["1"],
            "odds_drop_60_10": [0.7], "odds_drop_30_10": [0.5],
            "odds_drop_10_final": [0.2], "pop_change_30_10": [3],
            "n_points": [10], "early_odds": [50.0], "final_odds": [15.0],
        })
        result = classify_movement(df, threshold=0.20)
        assert result.iloc[0]["movement_class"] == "steamer"
        assert result.iloc[0]["movement_bucket"] == "strong_drop"

    def test_drifter_classification(self):
        df = pd.DataFrame({
            "race_id": ["r1"], "umaban": ["1"],
            "odds_drop_60_10": [-0.7], "odds_drop_30_10": [-0.5],
            "odds_drop_10_final": [-0.2], "pop_change_30_10": [-3],
            "n_points": [10], "early_odds": [5.0], "final_odds": [15.0],
        })
        result = classify_movement(df, threshold=0.20)
        assert result.iloc[0]["movement_class"] == "drifter"
        assert result.iloc[0]["movement_bucket"] == "strong_rise"

    def test_stable_classification(self):
        df = pd.DataFrame({
            "race_id": ["r1"], "umaban": ["1"],
            "odds_drop_60_10": [0.05], "odds_drop_30_10": [0.03],
            "odds_drop_10_final": [0.02], "pop_change_30_10": [0],
            "n_points": [10], "early_odds": [5.0], "final_odds": [4.9],
        })
        result = classify_movement(df, threshold=0.20)
        assert result.iloc[0]["movement_class"] == "stable"
        assert result.iloc[0]["movement_bucket"] == "stable"

    def test_custom_threshold(self):
        df = pd.DataFrame({
            "race_id": ["r1"], "umaban": ["1"],
            "odds_drop_60_10": [0.18], "odds_drop_30_10": [0.18],
            "odds_drop_10_final": [0.05], "pop_change_30_10": [1],
            "n_points": [10], "early_odds": [10.0], "final_odds": [8.2],
        })
        # threshold=0.20 → stable
        result_loose = classify_movement(df, threshold=0.20)
        assert result_loose.iloc[0]["movement_class"] == "stable"
        # threshold=0.15 → steamer
        result_tight = classify_movement(df, threshold=0.15)
        assert result_tight.iloc[0]["movement_class"] == "steamer"


class TestJoinResults:
    @pytest.fixture
    def sample_joined_data(self):
        """join_results 用のモックデータ"""
        movement = pd.DataFrame({
            "race_id": ["r1", "r1", "r2"],
            "umaban": ["1", "2", "1"],
            "odds_drop_30_10": [0.3, -0.1, 0.5],
            "n_points": [10, 10, 10],
            "final_odds": [15.0, 5.0, 8.0],
            "movement_class": ["steamer", "stable", "steamer"],
            "movement_bucket": ["moderate_drop", "stable", "strong_drop"],
        })

        entries = pd.DataFrame({
            "race_id": ["r1", "r1", "r2"],
            "umaban": [1, 2, 1],
            "kakuteijyuni": [1, 4, 3],
            "ninki": [1, 5, 3],
            "kisyucode": ["00001", "00002", "00001"],
            "chokyosicode": ["A001", "A002", "A001"],
        })

        races = pd.DataFrame({
            "race_id": ["r1", "r2"],
            "kyori": [1800, 1200],
            "syussotosu": [16, 10],
            "trackcd": [10, 23],  # 10=芝(turf), 23=ダート(dirt)
        })

        payouts = pd.DataFrame({
            "race_id": ["r1", "r2"],
            "payfukusyoumaban1": [1, 1],
            "payfukusyoumaban2": [3, 2],
            "payfukusyoumaban3": [pd.NA, pd.NA],
            "payfukusyopay1": [120.0, 80.0],
            "payfukusyopay2": [40.0, 30.0],
            "payfukusyopay3": [pd.NA, pd.NA],
        })

        return movement, entries, races, payouts

    def test_place_detection_win(self, sample_joined_data):
        mov, ent, rac, pay = sample_joined_data
        result = join_results(mov, ent, rac, pay)
        # r1-umaban1: 1着 → 複勝的中
        r1_h1 = result[(result["race_id"] == "r1") & (result["umaban"] == "1")].iloc[0]
        assert r1_h1["is_place"] == 1
        assert r1_h1["place_payout"] == 120.0

    def test_place_detection_third(self, sample_joined_data):
        mov, ent, rac, pay = sample_joined_data
        result = join_results(mov, ent, rac, pay)
        # r2-umaban1: 3着 → 複勝的中
        r2_h1 = result[(result["race_id"] == "r2") & (result["umaban"] == "1")].iloc[0]
        assert r2_h1["is_place"] == 1
        assert r2_h1["place_payout"] == 80.0

    def test_no_place_fourth(self, sample_joined_data):
        mov, ent, rac, pay = sample_joined_data
        result = join_results(mov, ent, rac, pay)
        # r1-umaban2: 4着 → 複勝外
        r1_h2 = result[(result["race_id"] == "r1") & (result["umaban"] == "2")].iloc[0]
        assert r1_h2["is_place"] == 0
        assert r1_h2["place_payout"] == 0.0

    def test_surface_mapping(self, sample_joined_data):
        mov, ent, rac, pay = sample_joined_data
        result = join_results(mov, ent, rac, pay)
        r1 = result[result["race_id"] == "r1"].iloc[0]  # trackcd=2 → turf
        r2 = result[result["race_id"] == "r2"].iloc[0]  # trackcd=1 → dirt
        assert r1["surface"] == "turf"
        assert r2["surface"] == "dirt"

    def test_min_points_filter(self, sample_joined_data):
        mov, ent, rac, pay = sample_joined_data
        # 1頭だけ n_points=3 にしてフィルタされるか確認
        mov.loc[mov.index[0], "n_points"] = 3
        result = join_results(mov, ent, rac, min_points=5)
        assert len(result) == 2  # 3 pointsの馬が除外
```

- [ ] **Step 4: テスト実行**

Run: `pytest tests/test_analyze_odds_movement.py -v`
Expected: 全てPASS

- [ ] **Step 5: コミット**

```bash
git add scripts/analyze_odds_movement.py tests/test_analyze_odds_movement.py
git commit -m "feat: classify_movement, join_results 実装とテスト"
```

---

### Task 4: 分析関数群（基本統計・騎手調教師・レース条件）

**Files:**
- Modify: `scripts/analyze_odds_movement.py`

- [ ] **Step 1: analyze_basic_stats を実装**

```python
def analyze_basic_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """基本統計: テーブルA/B/C

    Returns:
        dict with keys: table_a, table_b, table_c
    """
    stake = 100  # 100円固定

    # ── テーブルA: バケット別成績 ──
    bucket_stats = df.groupby("movement_bucket").agg(
        count=("is_place", "count"),
        place_rate=("is_place", "mean"),
        win_rate=("is_win", "mean"),
        avg_final_odds=("final_odds", "mean"),
        total_payout=("place_payout", "sum"),
        total_bets=("is_place", "count"),
    ).reset_index()
    bucket_stats["place_roi"] = (
        bucket_stats["total_payout"] / (bucket_stats["total_bets"] * stake) * 100
    )
    bucket_stats["place_rate"] = (bucket_stats["place_rate"] * 100).round(1)
    bucket_stats["win_rate"] = (bucket_stats["win_rate"] * 100).round(1)
    bucket_stats["place_roi"] = bucket_stats["place_roi"].round(1)

    # ── テーブルB: 人気セグメント × クラス クロス ──
    def _pop_segment(ninki):
        if pd.isna(ninki):
            return "unknown"
        if ninki <= 3:
            return "1-3番人気"
        elif ninki <= 7:
            return "4-7番人気"
        else:
            return "8番人気以降"

    df["pop_segment"] = df["ninki"].apply(_pop_segment)

    cross = df.groupby(["pop_segment", "movement_class"]).agg(
        count=("is_place", "count"),
        place_rate=("is_place", "mean"),
    ).reset_index()
    cross["place_rate"] = (cross["place_rate"] * 100).round(1)

    # ── テーブルC: 時間枠別予測力比較 ──
    windows = {
        "60->10min": "odds_drop_60_10",
        "30->10min": "odds_drop_30_10",
        "10->final": "odds_drop_10_final",
    }
    window_rows = []
    for label, col in windows.items():
        for thresh in [0.15, 0.20, 0.25]:
            mask = df[col] >= thresh
            sub = df[mask]
            if len(sub) > 0:
                window_rows.append({
                    "window": label,
                    "threshold": f"{thresh*100:.0f}%",
                    "count": len(sub),
                    "place_rate": round(sub["is_place"].mean() * 100, 1),
                    "roi": round(sub["place_payout"].sum() / (len(sub) * stake) * 100, 1),
                })
    window_comparison = pd.DataFrame(window_rows)

    return {
        "table_a": bucket_stats,
        "table_b": cross,
        "table_c": window_comparison,
    }
```

- [ ] **Step 2: analyze_jockey_trainer を実装**

```python
def analyze_jockey_trainer(df: pd.DataFrame, top_n: int = 20) -> dict[str, pd.DataFrame]:
    """騎手・調教師別の急落傾向分析

    Returns:
        dict with keys: by_jockey, by_trainer
    """
    steamer_mask = df["movement_class"] == "steamer"
    stable_mask = df["movement_class"] == "stable"

    def _analyze_group(group_col: str) -> pd.DataFrame:
        # ベクトル化: groupby + agg で一括計算
        is_steamer = (df["movement_class"] == "steamer").astype(int)
        is_stable = (df["movement_class"] == "stable").astype(int)

        grouped = df.groupby(group_col, dropna=False).agg(
            rides=("is_place", "count"),
            steam_count=("movement_class", lambda x: (x == "steamer").sum()),
            steam_place_rate=("is_place", lambda x: x[df.loc[x.index, "movement_class"] == "steamer"].mean()
                                          if (df.loc[x.index, "movement_class"] == "steamer").any() else float("nan")),
            stable_place_rate=("is_place", lambda x: x[df.loc[x.index, "movement_class"] == "stable"].mean()
                                           if (df.loc[x.index, "movement_class"] == "stable").any() else float("nan")),
        ).reset_index()

        grouped["steam_rate"] = (grouped["steam_count"] / grouped["rides"] * 100).round(1)
        grouped["diff"] = (grouped["steam_place_rate"] - grouped["stable_place_rate"]).round(1)

        # 最小サンプル数フィルタ
        grouped = grouped[grouped["rides"] >= 10]
        # 急落率でソート（高い順）
        grouped = grouped.sort_values("steam_rate", ascending=False)
        return grouped.head(top_n)

    return {
        "by_jockey": _analyze_group("kisyucode"),
        "by_trainer": _analyze_group("chokyosicode"),
    }
```

- [ ] **Step 3: analyze_race_conditions を実装**

```python
def analyze_race_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """レース条件別のマトリックス分析 (Table E)"""
    def _distance_band(kyori):
        if pd.isna(kyori):
            return "unknown"
        if kyori <= 1400:
            return "短距離(<=1400m)"
        elif kyori <= 2000:
            return "中距離(1400-2000m)"
        else:
            return "長距離(>2000m)"

    def _field_size(syussotosu):
        if pd.isna(syussotosu):
            return "unknown"
        if syussotosu <= 8:
            return "8頭以下"
        elif syussotosu <= 12:
            return "9-12頭"
        else:
            return "13頭以上"

    df_analysis = df.copy()
    df_analysis["distance_band"] = df_analysis["kyori"].apply(_distance_band)
    df_analysis["field_size_cat"] = df_analysis["syussotosu"].apply(_field_size)

    dimensions = [
        ("surface", "surface"),
        ("distance_band", "distance_band"),
        ("field_size_cat", "field_size_cat"),
    ]

    rows = []
    for label, col in dimensions:
        for cls in ["steamer", "stable"]:
            sub = df_analysis[df_analysis["movement_class"] == cls]
            grouped = sub.groupby(col, dropna=False)
            for cat, grp in grouped:
                if len(grp) < 5:
                    continue
                rows.append({
                    "dimension": label,
                    "category": cat,
                    "movement_class": cls,
                    "count": len(grp),
                    "place_rate": round(grp["is_place"].mean() * 100, 1),
                    "roi": round(grp["place_payout"].sum() / (len(grp) * 100) * 100, 1),
                })

    return pd.DataFrame(rows)
```

- [ ] **Step 4: 出力関数を実装**

```python
def print_summary(results: dict, title: str = "") -> None:
    """コンソールに分析結果を表形式で出力"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    for name, df in results.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            print(f"\n--- {name} ---")
            print(df.to_string(index=False))


def save_csv(results: dict, output_dir: str, detail_df: pd.DataFrame | None = None) -> None:
    """CSVファイル群を出力"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, df in results.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            path = out / f"{name}.csv"
            df.to_csv(path, index=False)
            logger.info("Saved: %s (%d rows)", path, len(df))

    if detail_df is not None and len(detail_df) > 0:
        detail_path = out / "detail_records.csv"
        # 主要列のみ出力
        keep_cols = ["race_id", "umaban", "movement_class", "movement_bucket",
                     "odds_drop_30_10", "final_odds", "ninki",
                     "kakuteijyuni", "is_place", "place_payout",
                     "kisyucode", "chokyosicode", "surface"]
        available = [c for c in keep_cols if c in detail_df.columns]
        detail_df[available].to_csv(detail_path, index=False)
        logger.info("Saved: %s", detail_path)
```

- [ ] **Step 5: main() を完成させる**

`main()` 関数を更新して全パイプラインをつなぐ:

```python
def main() -> None:
    args = build_parser().parse_args()
    start_year = int(args.start[:4])
    end_year = int(args.end[:4])
    output_dir = args.output_dir or f"output/odds_movement_analysis_{datetime.now().strftime('%Y%m%d')}"

    logger.info("=" * 60)
    logger.info("オッズ変動分析: %s ~ %s", args.start, args.end)
    logger.info("=" * 60)

    # 1. データ読み込み
    ts_df = load_time_series(start_year, end_year)
    entries_df = load_entries(args.start, args.end)
    races_df = load_races(args.start, args.end)
    payouts_df = load_payouts(args.start, args.end)

    # 2. 特徴量計算
    movement_df = compute_movement_features(ts_df)
    logger.info("Computed movement features for %d horses", len(movement_df))

    # 3. 分類
    classified_df = classify_movement(movement_df, threshold=args.drop_threshold)
    logger.info("Classified movements")

    # 4. 結果結合
    joined_df = join_results(classified_df, entries_df, races_df, payouts_df,
                              min_points=args.min_points)
    logger.info("Final dataset: %d records", len(joined_df))

    # 5. 分析
    basic = analyze_basic_stats(joined_df)
    jt = analyze_jockey_trainer(joined_df)
    rc = analyze_race_conditions(joined_df)

    # 6. 出力
    all_results = {**basic, **jt, "by_race_condition": rc}
    print_summary(all_results, title=f"オッズ変動分析結果 ({args.start} ~ {args.end})")
    save_csv(all_results, output_dir,
             detail_df=joined_df if args.detail else None)

    logger.info("Done. Output saved to: %s", output_dir)
```

- [ ] **Step 6: 全テスト実行**

Run: `pytest tests/test_analyze_odds_movement.py -v`
Expected: 全てPASS

- [ ] **Step 7: コミット**

```bash
git add scripts/analyze_odds_movement.py tests/test_analyze_odds_movement.py
git commit -m "feat: 分析関数群と main パイプラインの実装"
```

---

### Task 5: 総合テストと実行検証

**Files:**
- Modify: `scripts/analyze_odds_movement.py` (バグ修正のみ)

- [ ] **Step 1: 実際のデータで実行**

Run: `python scripts/analyze_odds_movement.py --start 20240101 --end 20241231 --output-dir output/test_odds_analysis`
Expected: エラーなく終了し、コンソールに表が出力される。output/test_odds_analysis/ にCSVが生成される。

- [ ] **Step 2: 出力内容を確認**

- `summary_main.csv` に7バケットの行があること
- `by_jockey.csv` に騎手コード別の統計があること
- `by_race_condition.csv` に surface/distance/field_size のカテゴリ別データがあること
- 各CSVの place_rate が 0-100 の範囲内であること

- [ ] **Step 3: 2023-2025 全期間で実行**

Run: `python scripts/analyze_odds_movement.py --start 20230101 --end 20251231`
Expected: 正常終了。処理対象件数が 2024-only より多いこと。

- [ ] **Step 4: 全テスト再実行**

Run: `pytest tests/test_analyze_odds_movement.py -v`
Expected: 全てPASS

- [ ] **Step 5: 最終コミット**

```bash
git add -A
git commit -m "feat: オッズ急落分析スクリプト完成"
```

---

## Implementation Order Summary

| Task | Description | Files | Depends On |
|------|-------------|-------|------------|
| 1 | スクリプト雛形 + データ読み込み | `scripts/analyze_odds_movement.py` | — |
| 2 | 特徴量計算 + テスト | `.py`, `tests/test_*.py` | Task 1 |
| 3 | 分類 + 結合 + テスト | `.py`, `tests/test_*.py` | Task 2 |
| 4 | 分析関数 + 出力 + main | `.py` | Task 3 |
| 5 | 総合テスト + 実行検証 | バグ修正 | Task 4 |
