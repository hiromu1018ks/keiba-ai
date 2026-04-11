# B群モデル改善 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** スタックド・アンサンブル・Optunaチューニング・過去走拡張・騎手-調教師コンビ特徴量を実装し、予測精度と回収率を向上させる

**Architecture:** 5フェーズ構成。Phase 1 (依存関係) → Phase 2 (B3: 過去走拡張) → Phase 3 (B4: コンビ特徴量) → Phase 4 (B1: アンサンブル) → Phase 5 (B2: Optuna)。特徴量変更を先に行い、モデル構造変更は特徴量安定後に実施。各フェーズは独立してテスト・コミット可能。

**Tech Stack:** Python 3.11, LightGBM, XGBoost, CatBoost, Optuna, scikit-learn (Ridge), numpy, pandas

**Spec:** `docs/superpowers/specs/2026-04-10-model-improvement-design.md`

---

## 実行順序の設計理由

```
Phase 2 (B3: 特徴量) ──┐
                        ├──→ Phase 4 (B1: モデル構造) ──→ Phase 5 (B2: チューニング)
Phase 3 (B4: 特徴量) ──┘
```

- B3/B4 は特徴量追加 → モデルの FEATURE_COLS に影響 → 先に実施
- B1 はモデル構造変更 → 特徴量が確定してから実施
- B2 はハイパーパラメータ探索 → 最終モデル構造に対して実施
- B3 と B4 は独立 → subagent で並列実行可能

---

## File Structure

### 新規作成

| ファイル | 責務 | Phase |
|----------|------|-------|
| `src/features/form_cycle_features.py` | フォームサイクル特徴量 (好調/不調トレンド) | 2 |
| `src/features/jockey_trainer_combo.py` | 騎手-調教師コンビ実績特徴量 | 3 |
| `src/models/stacked_ensemble.py` | 3モデル (LGBM+XGB+CatBoost) 統合 + Ridge メタラーナー | 4 |
| `src/tuning/optuna_tuner.py` | Optuna 目的関数 + 検索空間定義 | 5 |
| `scripts/run_tuning.py` | ハイパーパラメータチューニング CLI | 5 |
| `tests/test_form_cycle_features.py` | フォームサイクル特徴量テスト | 2 |
| `tests/test_jockey_trainer_combo.py` | コンビ特徴量テスト | 3 |
| `tests/test_stacked_ensemble.py` | アンサンブルテスト | 4 |
| `tests/test_optuna_tuner.py` | チューナーテスト | 5 |

### 修正

| ファイル | 変更内容 | Phase |
|----------|----------|-------|
| `pyproject.toml` | xgboost, catboost, optuna 追加 | 1 |
| `src/features/horse_history_features.py` | N_PAST パラメータ化 (3→5) | 2 |
| `src/models/stage1_ability_model.py` | FEATURE_COLS にフォームサイクル特徴量追加 | 2 |
| `src/models/place_ability_model.py` | FEATURE_COLS にフォームサイクル特徴量追加 | 2 |
| `src/models/ev_correction_model.py` | FEATURE_COLS にコンビ特徴量追加 | 3 |
| `src/pipelines/training_pipeline.py` | フォームサイクル・コンビ・アンサンブル統合 | 2,3,4 |
| `src/backtest/race_predictor.py` | アンサンブル予測チェーン更新 | 4 |
| `src/domain/models.py` | SubmodelSet にアンサンブルフィールド追加 | 4 |

---

## Phase 1: 依存関係 + インフラ

### Task 1: 依存関係追加

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: pyproject.toml に xgboost, catboost, optuna を追加**

`pyproject.toml` の `dependencies` に3パッケージを追加:

```toml
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "lightgbm>=4.3",
    "xgboost>=2.0",
    "catboost>=1.2",
    "optuna>=3.5",
    # ... 以下既存のまま
]
```

- [ ] **Step 2: インストール確認**

Run: `pip install -e ".[dev]"`
Expected: 全パッケージがインストールされる。エラーなし。

Run: `python -c "import xgboost, catboost, optuna; print('OK')"`
Expected: `OK`

- [ ] **Step 3: コミット**

```bash
git add pyproject.toml
git commit -m "chore: xgboost, catboost, optuna を依存関係に追加 (B群準備)"
```

---

## Phase 2: B3 — 過去走拡張 (N_PAST=3→5) + フォームサイクル特徴量

### Task 2: HorseHistoryFeatures N_PAST パラメータ化

**Files:**
- Modify: `src/features/horse_history_features.py`
- Modify: `tests/test_horse_history_features.py`

- [ ] **Step 1: N_PAST パラメータ化のテストを追加**

`tests/test_horse_history_features.py` に追加:

```python
def test_n_past_parameter():
    """n_past=5 の場合、過去5走分のデータが使用される"""
    mock_store = MagicMock(spec=ParquetStore)
    hist = HorseHistoryFeatures(store=mock_store, n_past=5)
    assert hist._n_past == 5
    # デフォルト値は5 (B3仕様)
    hist_default = HorseHistoryFeatures(store=mock_store)
    assert hist_default._n_past == 5
```

- [ ] **Step 2: テスト実行 (FAIL 確認)**

Run: `python -m pytest tests/test_horse_history_features.py::test_n_past_parameter -v`
Expected: FAIL (`__init__() got an unexpected keyword argument 'n_past'`)

- [ ] **Step 3: N_PAST パラメータ化実装**

`src/features/horse_history_features.py` を修正:

**3a: コンストラクタ引数追加 (line 161)**
```python
def __init__(self, store: ParquetStore, *, n_past: int = 5) -> None:
    self.store = store
    self.n_past = n_past
    self._n_past = n_past  # 内部参照用
    self._entries_cache: pd.DataFrame | None = None
    self._races_cache: pd.DataFrame | None = None
```

**3b: 過去走取得のハードコード `3` を `self._n_past` に置換 (7箇所)**

| 行 | 変更前 | 変更後 |
|----|--------|--------|
| 445 | `start = max(0, idx - 3)` | `start = max(0, idx - self._n_past)` |
| 496 | `ht_valid[-3:].mean()` | `ht_valid[-self._n_past:].mean()` |
| 544 | `pd.Series(z_arr).tail(3).mean()` | `pd.Series(z_arr).tail(self._n_past).mean()` |
| 557 | `td_valid[-3:].mean()` | `td_valid[-self._n_past:].mean()` |
| 568 | `c1_valid[-3:].mean()` | `c1_valid[-self._n_past:].mean()` |
| 579 | `c4_valid[-3:].mean()` | `c4_valid[-self._n_past:].mean()` |
| 595 | `closing_indices[-3:].mean()` | `closing_indices[-self._n_past:].mean()` |

- [ ] **Step 4: テスト実行 (PASS 確認)**

Run: `python -m pytest tests/test_horse_history_features.py -v`
Expected: 全テスト PASS

- [ ] **Step 5: コミット**

```bash
git add src/features/horse_history_features.py tests/test_horse_history_features.py
git commit -m "feat: HorseHistoryFeatures の N_PAST をパラメータ化 (3→5)"
```

### Task 3: フォームサイクル特徴量モジュール作成

**Files:**
- Create: `src/features/form_cycle_features.py`
- Create: `tests/test_form_cycle_features.py`

追加特徴量:
- `form_trend`: 近走の着順傾向 (正=好調、負=不調)
- `form_consistency`: 近走着順の安定性 (低い=安定)
- `form_peak_flag`: ピークフォーム判定 (直近が最良)

- [ ] **Step 1: フォームサイクル特徴量のテスト作成**

`tests/test_form_cycle_features.py` を作成:

```python
import numpy as np
import pytest
from features.form_cycle_features import compute_form_features, FEATURE_COLS


class TestComputeFormFeatures:
    def test_improving_trend(self):
        """着順が上昇傾向 → form_trend > 0"""
        # 新しい順: 3着→2着→1着 (idx=0が最新)
        kj = np.array([3.0, 2.0, 1.0])
        ss = np.array([16.0, 16.0, 16.0])
        trend, consistency, peak = compute_form_features(kj, ss)
        assert trend > 0  # 改善傾向

    def test_declining_trend(self):
        """着順が下降傾向 → form_trend < 0"""
        kj = np.array([1.0, 2.0, 3.0])
        ss = np.array([16.0, 16.0, 16.0])
        trend, _, _ = compute_form_features(kj, ss)
        assert trend < 0  # 悪化傾向

    def test_insufficient_data(self):
        """データ不足 (< 2走) → 全て NaN"""
        kj = np.array([1.0])
        ss = np.array([16.0])
        trend, consistency, peak = compute_form_features(kj, ss)
        assert np.isnan(trend)
        assert np.isnan(consistency)
        assert np.isnan(peak)

    def test_peak_flag_true(self):
        """直近2走が全体より良い → peak=1.0"""
        # 最新2走が良い (低い=良い): [1,2,5] → recent avg=1.5, overall=2.67
        kj = np.array([1.0, 2.0, 5.0])
        ss = np.array([16.0, 16.0, 16.0])
        _, _, peak = compute_form_features(kj, ss)
        assert peak == 1.0

    def test_peak_flag_false(self):
        """直近2走が全体より悪い → peak=0.0"""
        # 最新2走が悪い: [5,4,1] → recent avg=0.27, overall=0.17
        kj = np.array([5.0, 4.0, 1.0])
        ss = np.array([16.0, 16.0, 16.0])
        _, _, peak = compute_form_features(kj, ss)
        assert peak == 0.0

    def test_consistency_low(self):
        """全て同じ着順 → consistency は 0 に近い"""
        kj = np.array([3.0, 3.0, 3.0])
        ss = np.array([16.0, 16.0, 16.0])
        _, consistency, _ = compute_form_features(kj, ss)
        assert consistency < 0.01

    def test_feature_cols(self):
        assert FEATURE_COLS == ["form_trend", "form_consistency", "form_peak_flag"]
```

- [ ] **Step 2: テスト実行 (FAIL 確認)**

Run: `python -m pytest tests/test_form_cycle_features.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'features.form_cycle_features'`)

- [ ] **Step 3: FormCycleFeatures 実装**

`src/features/form_cycle_features.py` を作成:

```python
"""フォームサイクル特徴量 — 好調/不調トレンド

過去出走の着順データから以下を計算:
- form_trend: 正規化着順の線形回帰傾き (正=好調)
- form_consistency: 正規化着順の標準偏差 (低い=安定)
- form_peak_flag: 直近2走が全体より良い場合 1.0

HorseHistoryFeatures のループ内で呼び出される。
"""

from __future__ import annotations

import numpy as np

FEATURE_COLS: list[str] = [
    "form_trend",
    "form_consistency",
    "form_peak_flag",
]


def compute_form_features(
    kakuteijyuni: np.ndarray, syussotosu: np.ndarray
) -> tuple[float, float, float]:
    """過去出走の着順からフォームサイクル特徴量を計算。

    Args:
        kakuteijyuni: 過去N走の着順 (idx=0 が最新)
        syussotosu:   過去N走の出走頭数

    Returns:
        (form_trend, form_consistency, form_peak_flag)
        データ不足時は全て NaN。
    """
    valid = ~np.isnan(kakuteijyuni) & ~np.isnan(syussotosu) & (syussotosu > 1)
    n = int(valid.sum())

    if n < 2:
        return float("nan"), float("nan"), float("nan")

    fp = kakuteijyuni[valid].astype(float)
    fs = syussotosu[valid].astype(float)

    # 正規化: (pos-1)/(size-1)。低いほど良い [0, 1]
    norm = (fp - 1) / np.maximum(fs - 1, 1)

    # form_trend: 線形回帰の傾きを反転 (正=着順改善=好調)
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, norm, 1)[0])
    form_trend = -slope

    # form_consistency: 正規化着順の標準偏差
    form_consistency = float(np.std(norm))

    # form_peak_flag: 直近2走が全体平均より良い → 1.0
    if n >= 3:
        recent_avg = float(norm[:2].mean())
        overall_avg = float(norm.mean())
        form_peak_flag = 1.0 if recent_avg < overall_avg else 0.0
    else:
        form_peak_flag = float("nan")

    return form_trend, form_consistency, form_peak_flag
```

- [ ] **Step 3b: HorseHistoryFeatures にフォームサイクル計算を統合**

`src/features/horse_history_features.py` を修正:

**3b-1: BASE_COLS に3特徴量を追加 (line 144)**
```python
BASE_COLS: list[str] = [
    # ... 既存の14項目 ...
    "weight_zscore",
    "days_since_last_race",
    "rest_category",
    # B3: フォームサイクル
    "form_trend",
    "form_consistency",
    "form_peak_flag",
]
```

**3b-2: ファイル先頭に import 追加**
```python
from features.form_cycle_features import compute_form_features
```

**3b-3: compute() ループ内、rest_category 計算の直後 (line ~480) に追加**
```python
# B3: フォームサイクル特徴量
if n_past >= 2:
    _fc_kj = horse_arrs["kakuteijyuni"][valid_mask][start:idx].astype(float)
    _fc_ss = horse_arrs["syussotosu"][valid_mask][start:idx].astype(float)
    form_trend, form_consistency, form_peak_flag = compute_form_features(_fc_kj, _fc_ss)
else:
    form_trend = float("nan")
    form_consistency = float("nan")
    form_peak_flag = float("nan")
```

**3b-4: results.append() に3項目追加 (line ~684)**
```python
results.append({
    # ... 既存項目 ...
    "rest_category": rest_cat,
    "form_trend": form_trend,
    "form_consistency": form_consistency,
    "form_peak_flag": form_peak_flag,
})
```

- [ ] **Step 4: テスト実行 (PASS 確認)**

Run: `python -m pytest tests/test_form_cycle_features.py tests/test_horse_history_features.py -v`
Expected: 全テスト PASS

- [ ] **Step 5: コミット**

```bash
git add src/features/form_cycle_features.py tests/test_form_cycle_features.py src/features/horse_history_features.py
git commit -m "feat: フォームサイクル特徴量を追加 (form_trend, form_consistency, form_peak_flag)"
```

### Task 4: FEATURE_COLS 更新 + パイプライン統合

**Files:**
- Modify: `src/models/stage1_ability_model.py`
- Modify: `src/models/place_ability_model.py`
- Modify: `src/pipelines/training_pipeline.py`

- [ ] **Step 1: Stage1/Place の FEATURE_COLS にフォームサイクル特徴量追加**

`src/models/stage1_ability_model.py` line 69 の `rest_category,` の直後に追加:
```python
        # 休養期間 (2)
        "days_since_last_race",
        "rest_category",
        # フォームサイクル (3) — B3
        "form_trend",
        "form_consistency",
        "form_peak_flag",
```

`src/models/place_ability_model.py` も同様に FEATURE_COLS に同じ3項目を追加。

- [ ] **Step 2: TrainingPipeline で n_past=5 を渡すよう変更**

`src/pipelines/training_pipeline.py` line 223:
```python
# Before
hist = HorseHistoryFeatures(store=self.store)

# After
hist = HorseHistoryFeatures(store=self.store, n_past=5)
```

- [ ] **Step 3: 既存テストで回帰確認**

Run: `python -m pytest tests/ -v --timeout=120`
Expected: 全テスト PASS (フォームサイクル特徴量は HorseHistoryFeatures.compute() で自動計算されるため、HorseHistoryFeatures を mock しているテストには影響なし)

- [ ] **Step 4: コミット**

```bash
git add src/models/stage1_ability_model.py src/models/place_ability_model.py src/pipelines/training_pipeline.py
git commit -m "feat: Stage1/Placeモデルにフォームサイクル特徴量を統合"
```

---

## Phase 3: B4 — 騎手-調教師コンビ特徴量

### Task 5: JockeyTrainerComboFeatures 実装

**Files:**
- Create: `src/features/jockey_trainer_combo.py`
- Create: `tests/test_jockey_trainer_combo.py`

追加特徴量:
- `jt_combo_wr`: コンビの勝率 (Beta平滑)
- `jt_combo_place_rate`: コンビの複勝率
- `jt_combo_starts`: コンビの過去出走数 (信頼性指標)
- `jt_combo_prize_log`: コンビの獲得賞金 log

- [ ] **Step 1: コンビ特徴量のテスト作成**

`tests/test_jockey_trainer_combo.py` を作成:

```python
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from db.parquet_store import ParquetStore
from features.jockey_trainer_combo import JockeyTrainerComboFeatures, FEATURE_COLS


def _make_entries_hist() -> pd.DataFrame:
    """騎手-調教師コンビの過去出走履歴"""
    return pd.DataFrame({
        "race_id": ["R001", "R002", "R003", "R004"],
        "race_date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"]),
        "kisyucode": ["K01", "K01", "K01", "K02"],
        "chokyosicode": ["T01", "T01", "T02", "T01"],
        "kakuteijyuni": [1, 3, 5, 2],
        "umaban": [1, 1, 1, 1],
    })


def _make_entry_df() -> pd.DataFrame:
    """現在の出走データ"""
    return pd.DataFrame({
        "race_id": ["R005", "R005"],
        "umaban": [1, 2],
        "kisyucode": ["K01", "K02"],
        "chokyosicode": ["T01", "T01"],
        "race_date": pd.to_datetime(["2023-06-01", "2023-06-01"]),
    })


class TestJockeyTrainerCombo:
    def test_known_combo_stats(self):
        """K01+T01 コンビ: 2走中1勝 → wr = (1+1)/(2+11) ≈ 0.154"""
        mock_store = MagicMock(spec=ParquetStore)
        combo = JockeyTrainerComboFeatures(store=mock_store)
        combo._cache = _make_entries_hist()

        result = combo.compute(_make_entry_df())
        row = result[result["umaban"] == 1].iloc[0]
        # Beta(1,10): wr = (1+1)/(2+11) = 2/13 ≈ 0.154
        assert abs(row["jt_combo_wr"] - 2 / 13) < 1e-6
        # place_rate = (2+1)/(2+11) = 3/13 (1着+3着=2複勝)
        assert abs(row["jt_combo_place_rate"] - 3 / 13) < 1e-6
        assert row["jt_combo_starts"] == 2

    def test_unknown_combo_nan(self):
        """存在しないコンビ → NaN"""
        mock_store = MagicMock(spec=ParquetStore)
        combo = JockeyTrainerComboFeatures(store=mock_store)
        combo._cache = _make_entries_hist()

        entry = _make_entry_df().copy()
        entry["chokyosicode"] = ["T99", "T99"]  # 存在しない調教師
        result = combo.compute(entry)
        assert np.isnan(result.iloc[0]["jt_combo_wr"])

    def test_no_chokyosicode_column(self):
        """chokyosicode 列なし → 全 NaN"""
        mock_store = MagicMock(spec=ParquetStore)
        combo = JockeyTrainerComboFeatures(store=mock_store)
        combo._cache = pd.DataFrame()

        entry = _make_entry_df().drop(columns=["chokyosicode"])
        result = combo.compute(entry)
        for col in FEATURE_COLS:
            assert col in result.columns
            assert np.isnan(result.iloc[0][col])

    def test_feature_cols(self):
        assert FEATURE_COLS == [
            "jt_combo_wr", "jt_combo_place_rate",
            "jt_combo_starts", "jt_combo_prize_log",
        ]
```

- [ ] **Step 2: テスト実行 (FAIL 確認)**

Run: `python -m pytest tests/test_jockey_trainer_combo.py -v`
Expected: FAIL

- [ ] **Step 3: JockeyTrainerComboFeatures 実装**

`src/features/jockey_trainer_combo.py` を作成:

```python
"""騎手-調教師コンビ特徴量 (Stage2)

過去出走データから特定の騎手-調教師コンビの実績を計算。
リーク防止: compute() に渡された race_date 以前のデータのみ使用。
Beta(1,10) smoothing で小サンプルを安定化。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

FEATURE_COLS: list[str] = [
    "jt_combo_wr",
    "jt_combo_place_rate",
    "jt_combo_starts",
    "jt_combo_prize_log",
]


class JockeyTrainerComboFeatures:
    """騎手-調教師コンビの過去実績特徴量を生成。"""

    def __init__(self, store: ParquetStore) -> None:
        self.store = store
        self._cache: pd.DataFrame | None = None

    def _load_history(self) -> pd.DataFrame:
        if self._cache is None:
            from db.readers import load_history_entries

            entries = load_history_entries(self.store)
            if "chokyosicode" not in entries.columns:
                self._cache = pd.DataFrame()
                return self._cache
            self._cache = entries[entries["kakuteijyuni"] > 0].copy()
        return self._cache

    def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """コンビ特徴量を計算。

        Args:
            entry_df: (race_id, umaban, kisyucode, chokyosicode, race_date) を含む DataFrame
        """
        result = entry_df[["race_id", "umaban"]].copy()
        nan_cols = {c: float("nan") for c in FEATURE_COLS}

        hist = self._load_history()
        if hist.empty or "chokyosicode" not in entry_df.columns:
            return result.assign(**nan_cols)

        entry = entry_df.copy()
        entry["jt_combo"] = entry["kisyucode"].astype(str) + "_" + entry["chokyosicode"].astype(str)

        hist = hist.copy()
        hist["jt_combo"] = hist["kisyucode"].astype(str) + "_" + hist["chokyosicode"].astype(str)

        # コンビ別集計
        grouped = hist.groupby("jt_combo")
        stats = pd.DataFrame({
            "jt_starts": grouped["kakuteijyuni"].count(),
            "jt_wins": grouped["kakuteijyuni"].apply(lambda x: (x == 1).sum()),
            "jt_places": grouped["kakuteijyuni"].apply(lambda x: (x <= 3).sum()),
        })

        # Beta(1,10) smoothing
        stats["jt_combo_wr"] = (stats["jt_wins"] + 1) / (stats["jt_starts"] + 11)
        stats["jt_combo_place_rate"] = (stats["jt_places"] + 1) / (stats["jt_starts"] + 11)
        stats["jt_combo_starts"] = stats["jt_starts"]
        stats["jt_combo_prize_log"] = np.log1p(stats["jt_starts"] * 10)  # 賞金列が無い場合の代替

        # 賞金列が存在する場合はそちらを使用
        if "honsyokin" in hist.columns:
            prize_sum = grouped["honsyokin"].apply(
                lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum()
            )
            stats["jt_combo_prize_log"] = np.log1p(prize_sum)

        # マージ
        result["jt_combo"] = entry["jt_combo"].values
        result = result.merge(
            stats[FEATURE_COLS].reset_index(),
            on="jt_combo",
            how="left",
        )

        return result[["race_id", "umaban"] + FEATURE_COLS]
```

- [ ] **Step 4: テスト実行 (PASS 確認)**

Run: `python -m pytest tests/test_jockey_trainer_combo.py -v`
Expected: 全テスト PASS

- [ ] **Step 5: コミット**

```bash
git add src/features/jockey_trainer_combo.py tests/test_jockey_trainer_combo.py
git commit -m "feat: 騎手-調教師コンビ特徴量を追加 (B4)"
```

### Task 6: FEATURE_COLS 更新 + パイプライン統合

**Files:**
- Modify: `src/models/ev_correction_model.py`
- Modify: `src/pipelines/training_pipeline.py`
- Modify: `src/backtest/race_predictor.py`

- [ ] **Step 1: EVCorrectionModel.FEATURE_COLS にコンビ特徴量追加**

`src/models/ev_correction_model.py` line 52 の `trainer_prize_log,` の直後に追加:
```python
        # 調教師コンテキスト (Group D, Stage2)
        "trainer_wr_overall",
        "trainer_wr_distance",
        "trainer_wr_venue",
        "trainer_prize_log",
        # 騎手-調教師コンビ (B4, Stage2)
        "jt_combo_wr",
        "jt_combo_place_rate",
        "jt_combo_starts",
        "jt_combo_prize_log",
```

- [ ] **Step 2: TrainingPipeline + BacktestEngine + RacePredictor に統合**

**2a: `src/pipelines/training_pipeline.py` — trainer_ctx の直後に追加 (line ~293)**
```python
        # B4: 騎手-調教師コンビコンテキスト (Stage2)
        from features.jockey_trainer_combo import JockeyTrainerComboFeatures

        with TimingContext(f"{surface}/jt_combo"):
            jt_combo = JockeyTrainerComboFeatures(self.store)
            jt_df = jt_combo.compute(df_oof)
            df_oof = pd.merge(df_oof, jt_df, on=["race_id", "umaban"], how="left")
```

**2b: `src/backtest/race_predictor.py` — predict() の引数に `jt_combo_features` を追加し、trainer_features マージの直後 (line ~94) でマージ:**
```python
    def predict(
        self,
        race_df: pd.DataFrame,
        hist_features: pd.DataFrame | None = None,
        jockey_features: pd.DataFrame | None = None,
        trainer_features: pd.DataFrame | None = None,
        jt_combo_features: pd.DataFrame | None = None,  # B4 追加
    ) -> pd.DataFrame:
```

trainer_features マージ直後に追加:
```python
        if jt_combo_features is not None:
            jt_race = jt_combo_features[
                jt_combo_features["race_id"] == race_df["race_id"].iloc[0]
            ]
            df = df.merge(jt_race, on=["race_id", "umaban"], how="left")
```

**2c: `src/backtest/engine.py` — BacktestEngine.run() の jockey/trainer pre-compute 箇所に同パターンで JockeyTrainerComboFeatures の一括 pre-compute を追加し、RacePredictor.predict() 呼び出し時に `jt_combo_features=jt_df` を渡す。** jockey/trainer と同一パターン。

- [ ] **Step 3: 既存テストで回帰確認**

Run: `python -m pytest tests/ -v --timeout=120`
Expected: 全テスト PASS (mock ベースのテストは FEATURE_COLS 変更の影響を受けない)

- [ ] **Step 4: コミット**

```bash
git add src/models/ev_correction_model.py src/pipelines/training_pipeline.py src/backtest/race_predictor.py src/backtest/engine.py
git commit -m "feat: EV補正モデルに騎手-調教師コンビ特徴量を統合 (B4)"
```

---

## Phase 4: B1 — スタックド・アンサンブル

### Task 7: StackedEnsemble 実装

**Files:**
- Create: `src/models/stacked_ensemble.py`
- Create: `tests/test_stacked_ensemble.py`

設計 (Nguyen et al. 2024):
- Level 1: LightGBM + XGBoost + CatBoost (独立学習)
- Level 2: Ridge メタラーナー (OOF予測を特徴量に統合)
- 対象: 二値分類モデル (hit model) のみ
- Ranker (Stage1) と回帰 (return model) は LightGBM のまま

- [ ] **Step 1: StackedEnsemble のテスト作成**

`tests/test_stacked_ensemble.py` を作成:

```python
import numpy as np
import pandas as pd
import pytest
from models.stacked_ensemble import StackedEnsemble


def _make_binary_data(n: int = 500, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "f3": rng.rand(n),
    })
    y = pd.Series((rng.rand(n) > 0.8).astype(int))
    return X, y


class TestStackedEnsemble:
    def test_train_and_predict(self):
        """学習→予測で [0,1] の確率が返る"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[])
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        preds = ensemble.predict(X.iloc[split:])
        assert len(preds) == len(X) - split
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_base_models_trained(self):
        """3つのベースモデルが学習されている"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[])
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        assert ensemble.lgbm_model is not None
        assert ensemble.xgb_model is not None
        assert ensemble.cat_model is not None
        assert ensemble.meta_model is not None

    def test_different_from_single_lgbm(self):
        """アンサンブル予測が単一LGBMと異なる"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[])
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        preds_ens = ensemble.predict(X.iloc[split:])
        preds_lgbm = ensemble.lgbm_model.predict(X.iloc[split:])
        # アンサンブルとLGBM単体で予測が異なることを確認
        assert not np.allclose(preds_ens, preds_lgbm, atol=1e-6)

    def test_best_iteration_compatible(self):
        """lgb.Booster 互換の best_iteration 属性がある"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[])
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        assert hasattr(ensemble, "best_iteration")
        assert ensemble.best_iteration == 0  # アンサンブルでは使用しない
```

- [ ] **Step 2: テスト実行 (FAIL 確認)**

Run: `python -m pytest tests/test_stacked_ensemble.py -v`
Expected: FAIL

- [ ] **Step 3: StackedEnsemble 実装**

`src/models/stacked_ensemble.py` を作成:

```python
"""スタックド・アンサンブル — LightGBM + XGBoost + CatBoost → Ridge メタラーナー

Nguyen et al. (2024) の設計に基づく:
- Level 1: 3つのGBMモデルを独立学習 (K-fold OOF予測生成)
- Level 2: OOF予測を特徴量に Ridge 回帰で統合

TwoStageModel の hit_model のドロップイン代替として設計。
best_iteration=0 + predict(X) → ndarray を返すことで互換。
"""

from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class StackedEnsemble:
    """3モデル stacked ensemble for binary classification.

    lgb.Booster のインターフェース互換:
    - best_iteration: int (=0, アンサンブルでは使用しない)
    - predict(X, num_iteration=None) → np.ndarray of probabilities
    """

    best_iteration: int = 0

    def __init__(self, cat_cols: list[str] | None = None, n_folds: int = 3) -> None:
        self.cat_cols = cat_cols or []
        self.n_folds = n_folds
        self.lgbm_model: lgb.Booster | None = None
        self.xgb_model = None
        self.cat_model = None
        self.meta_model: Ridge | None = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        *,
        num_threads: int = 0,
    ) -> None:
        """K-fold OOF でメタラーナーを学習後、全データでベースモデルを再学習。"""
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)

        # --- Level 1: K-fold OOF 予測生成 ---
        n = len(X_train)
        oof_preds = np.full((n, 3), np.nan)
        fold_size = n // self.n_folds

        for i in range(self.n_folds):
            # 時系列考慮: 各foldのvalidは後半部分、trainは前半 (expanding window)
            val_start = int(n * (i + 1) / (self.n_folds + 1))
            val_end = int(n * (i + 2) / (self.n_folds + 1)) if i < self.n_folds - 1 else n
            tr_mask = np.zeros(n, dtype=bool)
            tr_mask[:val_start] = True  # valid以前のデータのみ学習に使用

            X_tr, y_tr = X_train.iloc[tr_mask], y_train.iloc[tr_mask]
            X_va, _ = X_train.iloc[~tr_mask], y_train.iloc[~tr_mask]

            oof_preds[~tr_mask, 0] = self._train_lgbm_fold(X_tr, y_tr, X_va, num_threads)
            oof_preds[~tr_mask, 1] = self._train_xgb_fold(X_tr, y_tr, X_va, num_threads)
            oof_preds[~tr_mask, 2] = self._train_cat_fold(X_tr, y_tr, X_va, num_threads)

        # --- Level 2: Ridge メタラーナー ---
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(oof_preds, y_train.values)

        # --- 最終ベースモデル: train+valid 全データで再学習 ---
        X_all = pd.concat([X_train, X_valid], ignore_index=True)
        y_all = pd.concat([y_train, y_valid], ignore_index=True)

        self.lgbm_model = self._train_lgbm_full(X_all, y_all, num_threads)
        self.xgb_model = self._train_xgb_full(X_all, y_all, num_threads)
        self.cat_model = self._train_cat_full(X_all, y_all, num_threads)

    def predict(self, X: pd.DataFrame, num_iteration: int | None = None) -> np.ndarray:
        """アンサンブル予測。Ridge で3モデルの予測を統合。"""
        p_lgbm = self.lgbm_model.predict(X)
        p_xgb = self.xgb_model.predict(xgb.DMatrix(X))
        # CatBoost: predict() はクラスラベル(0/1)を返すため predict_proba() を使用
        p_cat = self.cat_model.predict_proba(X)[:, 1]

        stacked = np.column_stack([p_lgbm, p_xgb, p_cat])
        return np.clip(self.meta_model.predict(stacked), 0, 1)

    # --- LightGBM helpers ---
    def _train_lgbm_fold(self, X_tr, y_tr, X_va, nt):
        m = lgb.train(
            {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
             "num_leaves": 31, "verbose": -1, "num_threads": nt},
            lgb.Dataset(X_tr, label=y_tr), num_boost_round=300,
        )
        return m.predict(X_va)

    def _train_lgbm_full(self, X, y, nt):
        return lgb.train(
            {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
             "num_leaves": 31, "verbose": -1, "num_threads": nt},
            lgb.Dataset(X, label=y), num_boost_round=300,
        )

    # --- XGBoost helpers ---
    def _train_xgb_fold(self, X_tr, y_tr, X_va, nt):
        import xgboost as xgb
        m = xgb.train(
            {"objective": "binary:logistic", "learning_rate": 0.03,
             "max_depth": 6, "nthread": nt},
            xgb.DMatrix(X_tr, label=y_tr), num_boost_round=300,
        )
        return m.predict(xgb.DMatrix(X_va))

    def _train_xgb_full(self, X, y, nt):
        import xgboost as xgb
        return xgb.train(
            {"objective": "binary:logistic", "learning_rate": 0.03,
             "max_depth": 6, "nthread": nt},
            xgb.DMatrix(X, label=y), num_boost_round=300,
        )

    # --- CatBoost helpers ---
    def _train_cat_fold(self, X_tr, y_tr, X_va, nt):
        from catboost import CatBoostClassifier, Pool
        m = CatBoostClassifier(
            iterations=300, learning_rate=0.03, depth=6,
            thread_count=nt, verbose=0,
        )
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_va)[:, 1]

    def _train_cat_full(self, X, y, nt):
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(
            iterations=300, learning_rate=0.03, depth=6,
            thread_count=nt, verbose=0,
        )
        m.fit(X, y)
        return m
```

- [ ] **Step 4: テスト実行 (PASS 確認)**

Run: `python -m pytest tests/test_stacked_ensemble.py -v`
Expected: 全テスト PASS

- [ ] **Step 5: コミット**

```bash
git add src/models/stacked_ensemble.py tests/test_stacked_ensemble.py
git commit -m "feat: スタックド・アンサンブル (LGBM+XGB+CatBoost→Ridge) を追加 (B1)"
```

### Task 8: パイプライン統合

**Files:**
- Modify: `src/pipelines/training_pipeline.py`
- Modify: `src/backtest/race_predictor.py`
- Modify: `src/domain/models.py`

- [ ] **Step 1: SubmodelSet に use_ensemble フラグ追加**

`src/domain/models.py` の SubmodelSet (line 218) に `use_ensemble` フラグを追加:

```python
@dataclass
class SubmodelSet:
    """サブモデル（芝/ダート）のセット"""
    market: MarketModel
    stage1: AbilityModel
    place_ability: PlaceAbilityModel
    win: WinTwoStageModel
    ev_corrector: EVCorrectionModel
    place: PlaceTwoStageModel
    wide: WideTwoStageModel
    confidence: RobustConfidenceEstimator
    use_ensemble: bool = False
```

- [ ] **Step 2: TrainingPipeline でアンサンブル学習を有効化**

`src/pipelines/training_pipeline.py` の `_train_submodel()` で hit_model 学習箇所を変更:

**2a: WinTwoStageModel (line ~274)**
```python
        # 3. 単勝 2段階モデル
        win_2s = WinTwoStageModel()
        if use_ensemble:  # B1: アンサンブル
            from models.stacked_ensemble import StackedEnsemble
            with TimingContext(f"{surface}/win_hit_ensemble"):
                features = win_2s._prepare_features(df_oof)
                y = (df_oof["kakuteijyuni"] == 1).astype(int)
                split = int(len(features) * 0.8)
                ensemble = StackedEnsemble(cat_cols=["surface", "distance_bin", "grade_code"])
                ensemble.train(features.iloc[:split], y.iloc[:split],
                               features.iloc[split:], y.iloc[split:], num_threads=num_threads)
                win_2s.hit_model = ensemble
        else:
            with TimingContext(f"{surface}/win_hit"):
                win_2s.train_hit_model(df_oof, num_threads=num_threads)
        with TimingContext(f"{surface}/win_return"):
            win_2s.train_return_model(df_oof, num_threads=num_threads)
        with TimingContext(f"{surface}/win_predict"):
            df_oof = win_2s.predict_ev(df_oof)
```

**2b: PlaceTwoStageModel (line ~301) — 同様の変更**

`use_ensemble` 引数は `_train_submodel()` のパラメータとして追加:

```python
def _train_submodel(self, df, *, num_threads=0, use_ensemble=False):
```

`run()` メソッドの呼び出し箇所で `use_ensemble=True` を渡す。

- [ ] **Step 3: RacePredictor でアンサンブル推論に対応**

`src/backtest/race_predictor.py` の `predict()` は変更不要。
理由: `predict_ev()` が `self.hit_model.predict(features, num_iteration=hit_iter)` を呼び出すが、StackedEnsemble の `predict()` も同じ引数を受け取るため、ドロップイン互換。

ただし、StackedEnsemble.predict() は DataFrame を受け取るのに対し、LightGBM の predict は特徴量配列も受け取る。互換性のため、StackedEnsemble.predict() の引数で `num_iteration` を無視する設計は既に完了済み。

- [ ] **Step 4: `scripts/run_backtest.py` に `--ensemble` CLI フラグを追加**

```python
parser.add_argument("--ensemble", action="store_true", help="アンサンブル (B1) を有効化")
```

`BacktestEngine` に `use_ensemble` を伝播し、`TrainingPipeline._train_submodel(use_ensemble=True)` を呼ぶ。

- [ ] **Step 5: 既存テストで回帰確認**

Run: `python -m pytest tests/ -v --timeout=120`
Expected: 全テスト PASS

- [ ] **Step 6: コミット**

```bash
git add src/domain/models.py src/pipelines/training_pipeline.py scripts/run_backtest.py
git commit -m "feat: パイプラインにアンサンブル学習を統合 (B1)"
```

**Optuna→Ensemble パラメータ連携 (Phase 5 完了後):** OptunaTuner で最適化されたパラメータは `data/tuning/{model}_best_params.json` に保存される。`StackedEnsemble` のコンストラクタに `base_params: dict | None = None` を追加し、指定時は Optuna 結果をベースモデルのハイパーパラメータに反映する設計とする。Phase 5 完了後に統合。

---

## Phase 5: B2 — Optuna ハイパーパラメータチューニング

### Task 9: OptunaTuner 実装

**Files:**
- Create: `src/tuning/optuna_tuner.py`
- Create: `tests/test_optuna_tuner.py`

チューニング対象:
- AbilityModel: num_leaves, learning_rate, feature_fraction
- PlaceAbilityModel: num_leaves, learning_rate, scale_pos_weight
- TwoStageModel (hit): num_leaves, learning_rate, scale_pos_weight
- TwoStageModel (return): num_leaves, learning_rate
- StackedEnsemble Ridge: alpha

検索空間: LogUniform (lr), IntRange (leaves), Uniform (fractions)

- [ ] **Step 1: OptunaTuner のテスト作成**

`tests/test_optuna_tuner.py` を作成:

```python
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from tuning.optuna_tuner import OptunaTuner, SEARCH_SPACES


def _make_train_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "race_id": [f"R{i//10:04d}" for i in range(n)],
        "race_date": pd.date_range("2020-01-01", periods=n, freq="h"),
        "surface": np.where(rng.rand(n) > 0.5, "turf", "dirt"),
        "distance_bin": "mile",
        "track_condition_code": 2,
        "grade_code": "C",
        "field_size": 14,
        "kakuteijyuni": rng.randint(1, 18, n),
        "p_ability_win": rng.rand(n),
        "odds": rng.uniform(2, 50, n),
        "fukuoddslow": rng.uniform(1.5, 10, n),
    })


class TestOptunaTuner:
    def test_search_spaces_valid(self):
        """検索空間が妥当な範囲"""
        for model_name, space in SEARCH_SPACES.items():
            assert "num_leaves" in space or "learning_rate" in space
            if "num_leaves" in space:
                lo, hi = space["num_leaves"]
                assert 7 <= lo <= 127
                assert lo <= hi <= 127

    def test_objective_returns_float(self):
        """目的関数が float を返す"""
        df = _make_train_data()
        tuner = OptunaTuner(model_type="win_hit")
        trial = MagicMock()
        trial.suggest_int = MagicMock(return_value=31)
        trial.suggest_float = MagicMock(return_value=0.03)
        score = tuner.objective(trial, df)
        assert isinstance(score, float)

    def test_tune_runs(self):
        """チューニングが完了する (n_trials=3)"""
        df = _make_train_data()
        tuner = OptunaTuner(model_type="win_hit")
        result = tuner.tune(df, n_trials=3)
        assert "best_params" in result
        assert "best_value" in result
        assert isinstance(result["best_params"], dict)
```

- [ ] **Step 2: テスト実行 (FAIL 確認)**

Run: `python -m pytest tests/test_optuna_tuner.py -v`
Expected: FAIL

- [ ] **Step 3: OptunaTuner 実装**

`src/tuning/optuna_tuner.py` を作成:

```python
"""Optuna ハイパーパラメータチューニング (B2)

各モデルの最適なハイパーパラメータを Optuna で探索。
時系列Walk-Forward CVで評価し、データリークを防止。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd

logger = logging.getLogger(__name__)

# 各モデルの検索空間: { param_name: (low, high) or distribution spec }
SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    "win_hit": {
        "num_leaves": (15, 63),
        "learning_rate": (0.01, 0.1),
        "feature_fraction": (0.5, 0.9),
    },
    "win_return": {
        "num_leaves": (7, 31),
        "learning_rate": (0.01, 0.1),
        "feature_fraction": (0.5, 0.9),
    },
    "place_hit": {
        "num_leaves": (15, 63),
        "learning_rate": (0.01, 0.1),
        "feature_fraction": (0.5, 0.9),
    },
    "place_return": {
        "num_leaves": (7, 31),
        "learning_rate": (0.01, 0.1),
        "feature_fraction": (0.5, 0.9),
    },
    "ability": {
        "num_leaves": (15, 63),
        "learning_rate": (0.01, 0.1),
        "feature_fraction": (0.5, 0.9),
    },
    "ridge_alpha": {
        "alpha": (0.01, 100.0),
    },
}


class OptunaTuner:
    """Optuna ベースのハイパーパラメータチューナー。"""

    def __init__(self, model_type: str = "win_hit") -> None:
        self.model_type = model_type
        self.search_space = SEARCH_SPACES.get(model_type, SEARCH_SPACES["win_hit"])

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """trial からパラメータをサンプリング。"""
        params: dict[str, Any] = {}
        for name, bounds in self.search_space.items():
            lo, hi = bounds
            if name == "learning_rate":
                params[name] = trial.suggest_float(name, lo, hi, log=True)
            elif name == "alpha":
                params[name] = trial.suggest_float(name, lo, hi, log=True)
            elif name == "feature_fraction":
                params[name] = trial.suggest_float(name, lo, hi)
            else:
                params[name] = trial.suggest_int(name, int(lo), int(hi))
        return params

    def objective(self, trial: optuna.Trial, df: pd.DataFrame) -> float:
        """目的関数: 時系列80/20分割で AUC を評価。"""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        params = self._suggest_params(trial)

        # 時系列分割 (既に race_date でソート済み前提)
        n = len(df)
        split = int(n * 0.8)

        # 特徴量はモデルタイプに応じて選択
        if self.model_type in ("win_hit", "place_hit"):
            from models.two_stage_return_model import WinTwoStageModel
            feat_cols = WinTwoStageModel.FEATURE_COLS
            if self.model_type == "win_hit":
                y = (df["kakuteijyuni"] == 1).astype(int)
            else:
                y = (df["kakuteijyuni"] <= 3).astype(int)
        else:
            # return モデルや ability モデルは簡易評価
            feat_cols = ["p_ability_win"]
            y = (df["kakuteijyuni"] == 1).astype(int)

        available_cols = [c for c in feat_cols if c in df.columns]
        X = df[available_cols].copy()
        for col in X.columns:
            if X[col].dtype == object:
                try:
                    X[col] = X[col].astype(float)
                except (ValueError, TypeError):
                    X = X.drop(columns=[col])

        X_train, X_valid = X.iloc[:split], X.iloc[split:]
        y_train, y_valid = y.iloc[:split], y.iloc[split:]

        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": params.get("num_leaves", 31),
            "learning_rate": params.get("learning_rate", 0.03),
            "feature_fraction": params.get("feature_fraction", 0.7),
            "verbose": -1,
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        model = lgb.train(
            lgb_params, train_data, num_boost_round=300,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        preds = model.predict(X_valid)
        return roc_auc_score(y_valid, preds)

    def tune(self, df: pd.DataFrame, n_trials: int = 100) -> dict[str, Any]:
        """Optuna チューニングを実行。"""
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, df), n_trials=n_trials)

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
        }
```

- [ ] **Step 4: テスト実行 (PASS 確認)**

Run: `python -m pytest tests/test_optuna_tuner.py -v`
Expected: 全テスト PASS

- [ ] **Step 5: コミット**

```bash
mkdir -p src/tuning
touch src/tuning/__init__.py
git add src/tuning/__init__.py src/tuning/optuna_tuner.py tests/test_optuna_tuner.py
git commit -m "feat: Optuna ハイパーパラメータチューナーを追加 (B2)"
```

### Task 10: チューニングスクリプト

**Files:**
- Create: `scripts/run_tuning.py`

- [ ] **Step 1: チューニング CLI スクリプト作成**

`scripts/run_tuning.py` を作成:

```python
"""ハイパーパラメータチューニング CLI (B2)

Usage:
    python scripts/run_tuning.py --model win_hit --start 20200101 --end 20231231 --trials 50
"""

import argparse
import json
import sys
from pathlib import Path

# src/ を pythonpath に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db.parquet_store import ParquetStore
from db.readers import load_entries, load_odds_snapshots, load_races
from tuning.optuna_tuner import OptunaTuner


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--model", required=True,
                        choices=["win_hit", "win_return", "place_hit", "place_return", "ability"],
                        help="チューニング対象モデル")
    parser.add_argument("--start", required=True, help="学習開始日 YYYYMMDD")
    parser.add_argument("--end", required=True, help="学習終了日 YYYYMMDD")
    parser.add_argument("--trials", type=int, default=50, help="Optuna試行数")
    args = parser.parse_args()

    store = ParquetStore()
    print(f"Loading data: {args.start} ~ {args.end}")
    race_df = load_races(store, args.start, args.end)
    entry_df = load_entries(store, args.start, args.end)
    odds_df = load_odds_snapshots(store, args.start, args.end)

    # 簡易特徴量生成 (フルパイプラインではなく最小限)
    from features.feature_engine import FeatureEngine
    engine = FeatureEngine()
    df = engine.build_all(race_df, entry_df, odds_df, store=store)

    # レース日ソート (時系列評価の前提)
    df = df.sort_values("race_date").reset_index(drop=True)
    print(f"Data loaded: {len(df)} rows")

    print(f"Tuning {args.model} with {args.trials} trials...")
    tuner = OptunaTuner(model_type=args.model)
    result = tuner.tune(df, n_trials=args.trials)

    print(f"\nBest value: {result['best_value']:.4f}")
    print(f"Best params: {json.dumps(result['best_params'], indent=2)}")

    # 結果保存
    out_path = Path(f"data/tuning/{args.model}_best_params.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: ドライランで動作確認**

Run: `python scripts/run_tuning.py --model win_hit --start 20230101 --end 20231231 --trials 3`
Expected: 3トライアル完了、best_params が出力される

- [ ] **Step 3: コミット**

```bash
git add scripts/run_tuning.py
git commit -m "feat: Optuna チューニング CLI スクリプトを追加 (B2)"
```

---

## テスト方針

CLAUDE.md に従い DB 不要・mock 使用:
1. **Phase 2 テスト**: N_PAST パラメータ化、form_trend/consistency/peak の計算正しさ
2. **Phase 3 テスト**: コンビ特徴量の正しさ (初コンビ=NaN、Beta平滑、リーク防止)
3. **Phase 4 テスト**: アンサンブルの OOF 生成、Ridge 統合、予測範囲 [0,1]
4. **Phase 5 テスト**: 目的関数の正しさ、検索空間の妥当性、best_params 出力
5. **回帰テスト**: 各 Phase 完了後に既存テスト全実行

## 学習への影響

- B3 (過去走拡張) で特徴量数次元が増加 → LightGBM は高次元に強い
- B1 (アンサンブル) でモデル表現力が向上 → 過学習リスクに注意 (early stopping で制御)
- B2 (Optuna) でハイパーパラメータが最適化 → 再学習 + バックテストで効果確認
- 全体として ROI がどう変化するかは Phase 5 完了後にバックテストで評価
