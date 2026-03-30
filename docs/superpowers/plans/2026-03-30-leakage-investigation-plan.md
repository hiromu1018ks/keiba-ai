# In-Sample Cascade Leakage Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate in-sample cascade leakage by implementing K-fold expanding window OOF predictions for Stage1 AbilityModel.

**Architecture:** Add `train_oof()` method to AbilityModel that generates out-of-fold `p_ability_win` using expanding time-window folds. Modify `_train_submodel()` to use OOF predictions for all downstream model training. Ablation study first to quantify leakage contribution.

**Tech Stack:** Python 3.11, LightGBM, pandas, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-03-30-leakage-investigation-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `tests/test_oof_leakage.py` | Create | OOF予測の正確性とリーク防止のテスト |
| `src/models/stage1_ability_model.py:136+` | Modify | `train_oof()` メソッド追加 |
| `src/pipelines/training_pipeline.py:208-265` | Modify | `_train_submodel()` で train_oof + df_oof 使用 |

---

## Phase 1: Ablation Study

### Task 1: Ablation — p_ability_win リーク寄与の定量化

**Files:**
- Modify: `src/models/place_ability_model.py:60`
- Modify: `src/models/two_stage_return_model.py:26`

- [ ] **Step 1: p_ability_win を PlaceAbilityModel.FEATURE_COLS から一時削除**

`src/models/place_ability_model.py` line 60 をコメントアウト:

```python
# Ablation: "p_ability_win",  ← 一時的にコメントアウト
```

- [ ] **Step 2: p_ability_win を WinTwoStageModel.FEATURE_COLS から一時削除**

`src/models/two_stage_return_model.py` line 26 をコメントアウト:

```python
# Ablation: "p_ability_win",  ← 一時的にコメントアウト
```

- [ ] **Step 3: バックテスト実行**

```bash
python scripts/run_backtest.py --train-start 20200101 --train-end 20231231 --test-start 20240101 --test-end 20241231
```

Expected: ~24分。ROI は 143.3% から大幅に変化するはず。

- [ ] **Step 4: 結果を記録**

`docs/superpowers/ablation-results.md` を作成:

```markdown
# Ablation Results (2026-03-30)

| Pattern | Stage1 | p_ability_win | ROI | Bets | Stake | Return |
|---------|--------|--------------|-----|------|-------|--------|
| C (current) | 30 cols | in-sample | 143.3% | 6,134 | 613,400 | 879,110 |
| B (ablation) | 30 cols | removed | ???% | ??? | ??? | ??? |
```

- [ ] **Step 5: 変更をリバート**

```bash
git checkout -- src/models/place_ability_model.py src/models/two_stage_return_model.py
```

- [ ] **Step 6: 結果をコミット**

```bash
git add docs/superpowers/ablation-results.md
git commit -m "docs: アブレーション検証結果を記録 (p_ability_win削除)"
```

---

## Phase 2: OOF Implementation

### Task 2: テスト作成 — train_oof() の TDD Red Phase

**Files:**
- Create: `tests/test_oof_leakage.py`

- [ ] **Step 1: テストファイルを作成**

```python
"""train_oof() のテスト — OOF予測の正確性とリーク防止を検証"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from models.stage1_ability_model import AbilityModel


def _make_oof_df(n_races: int = 40, horses_per_race: int = 8) -> pd.DataFrame:
    """OOFテスト用の合成DataFrameを生成（~2.2年分、40レース）。"""
    rng = np.random.default_rng(42)
    n_rows = n_races * horses_per_race
    dates = pd.date_range("2020-01-01", periods=n_races, freq="20D")
    data = {
        "race_id": [f"R{i:04d}" for i in range(n_races) for _ in range(horses_per_race)],
        "race_date": np.repeat(dates, horses_per_race),
        "umaban": list(range(1, horses_per_race + 1)) * n_races,
        "surface": ["turf"] * n_rows,
        "distance_bin": ["mile"] * n_rows,
        "track_condition_code": [1] * n_rows,
        "grade_code": [0] * n_rows,
        "field_size": [horses_per_race] * n_rows,
        "weight_diff_from_mean": rng.standard_normal(n_rows),
        "difficulty_score": rng.standard_normal(n_rows),
        "norm_finish_logit_avg": rng.standard_normal(n_rows),
        "haron_time_l3_avg": rng.standard_normal(n_rows),
        "haron_time_l3_zscore": rng.standard_normal(n_rows),
        "time_diff_avg": rng.standard_normal(n_rows),
        "corner_1c_avg": rng.standard_normal(n_rows),
        "corner_4c_avg": rng.standard_normal(n_rows),
        "closing_index_avg": rng.standard_normal(n_rows),
        "kyakusitu_cd": [1] * n_rows,
        "blood_surface_wr": rng.uniform(0, 0.5, n_rows),
        "blood_distance_wr": rng.uniform(0, 0.5, n_rows),
        "blood_condition_wr": [np.nan] * n_rows,
        "blood_total_wr": rng.uniform(0, 0.5, n_rows),
        "blood_prize_log": rng.standard_normal(n_rows),
        "blood_keito_cd": [np.nan] * n_rows,
        "kyakusitu_x_distance": [1.0] * n_rows,
        "kyakusitu_x_surface": [1.0] * n_rows,
        "weight_x_distance": rng.standard_normal(n_rows),
        "norm_finish_logit_avg_race_rank": rng.uniform(0, 1, n_rows),
        "haron_time_l3_avg_race_rank": rng.uniform(0, 1, n_rows),
        "time_diff_avg_race_rank": rng.uniform(0, 1, n_rows),
        "corner_1c_avg_race_rank": rng.uniform(0, 1, n_rows),
        "closing_index_avg_race_rank": rng.uniform(0, 1, n_rows),
        "weight_absolute": rng.uniform(400, 500, n_rows),
        "finish_pos": rng.integers(1, horses_per_race + 1, n_rows),
    }
    return pd.DataFrame(data)


class TestTrainOof:
    """AbilityModel.train_oof() のテスト。"""

    def test_train_oof_returns_df_with_oof_predictions(self):
        """train_oof は p_ability_win 列を含む df を返すこと。"""
        df = _make_oof_df()
        model = AbilityModel()

        with patch.object(AbilityModel, "train"), \
             patch.object(AbilityModel, "add_ability_probs") as mock_add:
            def side_effect(d):
                d = d.copy()
                d["p_ability_win"] = np.random.default_rng(0).uniform(
                    0.01, 0.5, len(d)
                )
                return d
            mock_add.side_effect = side_effect

            result = model.train_oof(df, n_folds=3)

            assert "p_ability_win" in result.columns
            assert result["p_ability_win"].notna().any()

    def test_train_oof_expanding_window_no_date_overlap(self):
        """各foldの学習データが予測データより前のみであること。"""
        df = _make_oof_df()
        model = AbilityModel()

        train_dates_list = []
        predict_dates_list = []

        def track_train(self_inner, train_df):
            train_dates_list.append(set(train_df["race_date"]))

        def track_add(self_inner, pred_df):
            predict_dates_list.append(set(pred_df["race_date"]))
            pred_df = pred_df.copy()
            pred_df["p_ability_win"] = 0.1
            return pred_df

        with patch.object(AbilityModel, "train", track_train), \
             patch.object(AbilityModel, "add_ability_probs", track_add):
            model.train_oof(df, n_folds=3)

        for i, (td, pd_) in enumerate(zip(train_dates_list, predict_dates_list)):
            overlap = td & pd_
            assert len(overlap) == 0, (
                f"Fold {i}: {len(overlap)} overlapping dates between train and predict"
            )
            # Expanding: all training dates are earlier
            assert max(td) < min(pd_), (
                f"Fold {i}: train max {max(td)} >= predict min {min(pd_)}"
            )

    def test_train_oof_trains_final_model_on_all_data(self):
        """最終モデルは全データで学習されること（推論用）。"""
        df = _make_oof_df()
        model = AbilityModel()

        train_calls = []

        def capture_train(self_inner, train_df):
            train_calls.append(len(train_df))

        with patch.object(AbilityModel, "train", capture_train), \
             patch.object(AbilityModel, "add_ability_probs") as mock_add:
            mock_add.side_effect = lambda d: d.assign(p_ability_win=0.1)
            model.train_oof(df, n_folds=3)

        # 最後の train() 呼び出しは全データ（最終モデル）
        assert train_calls[-1] == len(df)

    def test_train_oof_first_fold_has_nan_predictions(self):
        """最初のfoldの学習期間の p_ability_win は NaN であること。"""
        df = _make_oof_df()
        model = AbilityModel()

        with patch.object(AbilityModel, "train"), \
             patch.object(AbilityModel, "add_ability_probs") as mock_add:
            mock_add.side_effect = lambda d: d.assign(p_ability_win=0.1)
            result = model.train_oof(df, n_folds=3)

        # 最初のfoldの学習期間（最早の~25%の日付）はNaN
        dates = sorted(result["race_date"].unique())
        first_quarter_end = dates[len(dates) // 4]
        first_quarter = result[result["race_date"] < first_quarter_end]
        assert first_quarter["p_ability_win"].isna().all()

    def test_train_oof_fallback_when_insufficient_data(self):
        """データ不足時は通常の train+predict にフォールバックすること。"""
        # 1レースのみ（fold分割不可）
        df = _make_oof_df(n_races=1, horses_per_race=4)
        model = AbilityModel()

        with patch.object(AbilityModel, "train") as mock_train, \
             patch.object(AbilityModel, "add_ability_probs") as mock_add:
            mock_add.return_value = df.assign(p_ability_win=0.25)
            result = model.train_oof(df, n_folds=3)

            # フォールバック: train + add_ability_probs 各1回
            assert mock_train.call_count == 1
            assert mock_add.call_count == 1
            assert "p_ability_win" in result.columns
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
python -m pytest tests/test_oof_leakage.py -v
```

Expected: FAIL — `AttributeError: 'AbilityModel' object has no attribute 'train_oof'`

- [ ] **Step 3: テストをコミット**

```bash
git add tests/test_oof_leakage.py
git commit -m "test: train_oof() のテストを追加 (TDD Red phase)"
```

---

### Task 3: 実装 — train_oof() in AbilityModel

**Files:**
- Modify: `src/models/stage1_ability_model.py` (add method after line 135)

- [ ] **Step 1: train_oof() メソッドを追加（line 135 の後）**

```python
    def train_oof(self, df: pd.DataFrame, n_folds: int = 3) -> pd.DataFrame:
        """K-fold expanding window で OOF p_ability_win を生成。

        各 fold で過去データのみで学習し、未来データを予測。
        最終的に全データで学習したモデルを self.models に格納（推論用）。

        Args:
            df: 学習データ（race_date, FEATURE_COLS, finish_pos を含む）
            n_folds: fold数

        Returns:
            df に OOF p_ability_win を追加した DataFrame。
            最初の fold の学習期間は NaN。
        """
        df = df.copy()
        df = df.sort_values("race_date").reset_index(drop=True)
        oof_preds = pd.Series(np.nan, index=df.index, dtype=np.float64)

        dates = sorted(df["race_date"].unique())
        n_dates = len(dates)

        # データ不足時はフォールバック
        if n_dates < n_folds + 1:
            self.train(df)
            return self.add_ability_probs(df)

        # fold 境界: n_folds+1 個の等分割点
        boundaries = [
            dates[n_dates * (i + 1) // (n_folds + 1)]
            for i in range(n_folds)
        ]

        for i in range(n_folds):
            train_end = boundaries[i]
            test_end = (
                boundaries[i + 1]
                if i + 1 < n_folds
                else dates[-1] + pd.Timedelta(days=1)
            )

            train_mask = df["race_date"] < train_end
            test_mask = (df["race_date"] >= train_end) & (df["race_date"] < test_end)

            train_df = df.loc[train_mask].copy()
            test_df = df.loc[test_mask].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                continue

            fold_model = AbilityModel()
            fold_model.train(train_df)
            test_df = fold_model.add_ability_probs(test_df)

            oof_preds.loc[test_mask] = test_df["p_ability_win"].values

        # 最終モデルを全データで学習（推論用）
        self.train(df)

        # OOF 予測を設定
        df["p_ability_win"] = oof_preds
        return df
```

- [ ] **Step 2: テストが通ることを確認**

```bash
python -m pytest tests/test_oof_leakage.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 3: 全テストスイートを実行**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass (no regressions).

- [ ] **Step 4: コミット**

```bash
git add src/models/stage1_ability_model.py
git commit -m "feat: AbilityModel.train_oof() を追加 (K-fold expanding window OOF)"
```

---

### Task 4: 修正 — _train_submodel() で OOF 使用

**Files:**
- Modify: `src/pipelines/training_pipeline.py:208-265`

> **注意:** `_train_submodel()` の冒頭部分（lines 174-206: HorseHistoryFeatures,
> compute_interaction_features, MarketModel）は `df` をそのまま使用し、
> **変更しない**。OOFマスクは Stage1 以降の下流モデルのみに適用する。

- [ ] **Step 1: Stage1 を train_oof に変更し、OOFマスクを作成**

`src/pipelines/training_pipeline.py` lines 208-210 を置換:

```python
        # --- Stage1: OOF predictions ---
        df = submodel.stage1.train_oof(df, n_folds=3)
        oof_mask = df["p_ability_win"].notna()
        df_oof = df[oof_mask].copy()
```

- [ ] **Step 2: PlaceAbilityModel を df_oof に変更**

Lines 213-217:

```python
        submodel.place_ability.train(df_oof)
        df_oof = submodel.place_ability.predict(df_oof)
```

- [ ] **Step 3: WinTwoStageModel を df_oof に変更**

Lines 220-223:

```python
        submodel.win.train_hit_model(df_oof)
        submodel.win.train_return_model(df_oof)
        df_oof = submodel.win.predict_ev(df_oof)
```

- [ ] **Step 4: Jockey/Trainer context を df_oof に変更**

Lines 226-235。`compute()` は `entry_df` のみを受け取る（`target_race_ids` 引数なし）:

```python
        jockey_ctx = JockeyContextFeatures(self.repo)
        jockey_df = jockey_ctx.compute(df_oof)
        df_oof = pd.merge(df_oof, jockey_df, on=["race_id", "umaban"], how="left")

        trainer_ctx = TrainerContextFeatures(self.repo)
        trainer_df = trainer_ctx.compute(df_oof)
        df_oof = pd.merge(df_oof, trainer_df, on=["race_id", "umaban"], how="left")
```

- [ ] **Step 5: EVCorrectionModel を df_oof に変更**

Lines 238-240:

```python
        submodel.ev_corrector.train(df_oof)
        df_oof = submodel.ev_corrector.correct_ev(df_oof)
```

- [ ] **Step 6: PlaceTwoStageModel を df_oof に変更**

Lines 243-246:

```python
        submodel.place.train_hit_model(df_oof)
        submodel.place.train_return_model(df_oof)
        df_oof = submodel.place.predict_ev(df_oof)
```

- [ ] **Step 7: WideTwoStageModel と Confidence を df_oof に変更**

Lines 249-265。`df` → `df_oof` の具体的な置換:

```python
        # WideTwoStageModel
        pair_df = WideJointPairBuilder().build(df_oof)
        wide_2s = WideTwoStageModel()
        if len(pair_df) > 0:
            wide_2s.train_hit_model(pair_df)
            wide_2s.train_return_model(pair_df)

        # RobustConfidenceEstimator
        conf = RobustConfidenceEstimator()
        win_calib_df = df_oof.copy()
        win_calib_df["actual_ev_win"] = (
            df_oof["win_odds_actual"] * (df_oof["finish_pos"] == 1).astype(int)
        )
        place_calib_df = df_oof.copy()
        place_calib_df["actual_ev_place"] = (
            df_oof["place_odds_actual"] * (df_oof["finish_pos"] <= 3).astype(int)
        )
        place_calib_df["ev_place_corrected"] = df_oof["ev_place"]
        conf.calibrate(win_calib_df, place_calib_df)
```

- [ ] **Step 8: テストを実行**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 9: コミット**

```bash
git add src/pipelines/training_pipeline.py
git commit -m "fix: _train_submodel() を train_oof() に変更しOOF予測で下流モデルを学習"
```

---

### Task 5: OOF バックテスト実行と結果分析

**Files:**
- Modify: `docs/superpowers/ablation-results.md`（結果追記）

- [ ] **Step 1: OOF バックテストを実行**

```bash
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231
```

Expected: 学習 ~25-50分。ROI は in-sample (143.3%) より低いはず。

- [ ] **Step 2: 結果を比較・記録**

`docs/superpowers/ablation-results.md` を更新:

```markdown
| Pattern | Stage1 | p_ability_win | ROI | Bets | Stake | Return |
|---------|--------|--------------|-----|------|-------|--------|
| C (current) | 30 cols | in-sample | 143.3% | 6,134 | 613,400 | 879,110 |
| B (ablation) | 30 cols | removed | ???% | ??? | ??? | ??? |
| D (OOF fix) | 30 cols | OOF | ???% | ??? | ??? | ??? |
```

- [ ] **Step 3: 結果をコミット**

```bash
git add docs/superpowers/ablation-results.md backtest_result.json
git commit -m "docs: OOF修正後のバックテスト結果を記録"
```

- [ ] **Step 4: 合否判定**

| 条件 | 判定 |
|------|------|
| OOF ROI > 110% | 特徴量設計有効。実運用検討 |
| OOF ROI 100-110% | 妥当。微調整で改善の余地あり |
| OOF ROI < 100% | 特徴量の再設計が必要 |
